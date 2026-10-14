"""RAG 核心：加载知识库 → 切分 → 向量化 → 检索 → 生成带来源的答案"""
import logging
import re
import shutil

from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from app.core.config import (
    BASE_URL,
    LLM_MODEL,
    EMBEDDING_MODEL,
    ALIBABA_API_KEY,
    KNOWLEDGE_DIR,
    CHROMA_DIR,
    TOP_K,
)
from app.core.errors import ServiceUnavailableError
from app.utils.document_loader import load_knowledge_dir

logger = logging.getLogger("RAG")


class RAGService:
    """负责知识库索引与带来源的问答"""

    def __init__(self):
        # 向量化模型：把文本变成向量
        self.embeddings = OpenAIEmbeddings(
            model=EMBEDDING_MODEL,
            api_key=ALIBABA_API_KEY,
            base_url=BASE_URL,
            check_embedding_ctx_length=False,  # 关键：关闭长度/tokenize 预处理，DashScope 兼容接口要原始字符串
            chunk_size=25,  # DashScope text-embedding-v2 单次请求最多 25 条，超出需分批
            request_timeout=30,
            max_retries=0,
        )
        # 大语言模型：生成回答
        self.llm = ChatOpenAI(
            model=LLM_MODEL,
            api_key=ALIBABA_API_KEY,
            base_url=BASE_URL,
            temperature=0,
            request_timeout=30,
            max_retries=0,
        )
        self.vectorstore = None

    def build_index(self, knowledge_dir=None):
        """加载知识库文档并（重建）Chroma 向量索引"""
        knowledge_dir = knowledge_dir or KNOWLEDGE_DIR
        logger.info("[RAG] 正在加载知识库: %s", knowledge_dir)
        docs = load_knowledge_dir(knowledge_dir)
        
        # 确保原始文档内容都是字符串，过滤空内容
        for doc in docs:
            if doc.page_content is None:
                doc.page_content = ""
            elif not isinstance(doc.page_content, str):
                doc.page_content = str(doc.page_content)
        docs = [doc for doc in docs if doc.page_content.strip()]
        
        if not docs:
            logger.warning("[RAG] 知识库为空，跳过索引构建")
            self.vectorstore = None
            return

        # 重建索引：先删除旧的持久化目录
        if CHROMA_DIR.exists():
            shutil.rmtree(CHROMA_DIR)
        
        # CSV 每行已由 CSVLoader 作为一条原始文档返回，只切分 PDF 页面。
        csv_docs = []
        pdf_docs = []
        for doc in docs:
            src = str(doc.metadata.get("source", "")).lower()
            if ".csv" in src:
                csv_docs.append(doc)
            else:
                pdf_docs.append(doc)

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )
        # PDF 正常切分，CSV 作为完整文档直接进入索引
        texts = text_splitter.split_documents(pdf_docs) + csv_docs
        logger.info("[RAG] 切分完成：PDF %d 块，CSV 保留 %d 条完整文档", len(texts) - len(csv_docs), len(csv_docs))
        
        # 修复：创建全新的 Document 对象，确保 page_content 100%是字符串
        clean_texts = []
        for t in texts:
            content = t.page_content
            if content is None:
                continue
            # 强制转字符串
            content = str(content)
            # 移除不可见控制字符
            content = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', content)
            content = content.strip()
            if content:
                # 必须新建 Document，旧对象的 page_content 可能是只读属性
                # 同时清理 metadata：所有值统一转为 str，避免数字/None 混入 embedding 输入
                metadata = self._sanitize_metadata(t.metadata)
                clean_texts.append(Document(page_content=content, metadata=metadata))
        
        texts = clean_texts
        logger.info("[RAG] 清洗后剩余 %d 个有效文本块", len(texts))
        
        if not texts:
            logger.warning("[RAG] 切分后无有效内容")
            self.vectorstore = None
            return
        
        # 万无一失：断言所有文本块都是非空字符串
        assert all(isinstance(t.page_content, str) and len(t.page_content.strip()) > 0 for t in texts), "存在非字符串或空内容"

        self.vectorstore = self._build_chroma_index(texts)
        logger.info("[RAG] 向量索引构建完成，共 %d 条", len(texts))

    @staticmethod
    def _sanitize_metadata(metadata):
        """把 metadata 中的所有值统一转为 str，避免数字/None 等混入 embedding 输入导致 400"""
        if not metadata:
            return {}
        clean = {}
        for k, v in metadata.items():
            clean[str(k)] = "" if v is None else str(v)
        return clean

    def _build_chroma_index(self, texts):
        """构建 Chroma 索引；优先 from_documents，失败则降级为手动 embed + from_texts"""
        try:
            return Chroma.from_documents(
                documents=texts,
                embedding=self.embeddings,
                persist_directory=str(CHROMA_DIR),
            )
        except Exception as exc:
            logger.warning("[RAG] from_documents 失败（%s），改用 from_texts 构建索引", type(exc).__name__)
            contents = [t.page_content for t in texts]
            metadatas = [t.metadata for t in texts]
            return Chroma.from_texts(
                texts=contents,
                metadatas=metadatas,
                embedding=self.embeddings,
                persist_directory=str(CHROMA_DIR),
            )

    def load_index(self):
        """从磁盘加载已有的向量索引"""
        if CHROMA_DIR.exists():
            self.vectorstore = Chroma(
                embedding_function=self.embeddings,
                persist_directory=str(CHROMA_DIR),
            )
            logger.info("[RAG] 已加载本地向量索引: %s", CHROMA_DIR)
        else:
            logger.warning("[RAG] 未找到本地索引，启动时将会构建")

    def load_or_build(self):
        """有索引就加载，没有就构建"""
        if CHROMA_DIR.exists():
            self.load_index()
        else:
            self.build_index()

    def query(self, question, history=None):
        """检索 Top-K 相关片段并生成带来源标注的回答"""
         # 确保 question 是纯字符串
        question = str(question) if question else ""
        if self.vectorstore is None:
            raise ServiceUnavailableError("知识库尚未就绪，请检查知识库文件和服务端启动日志。")

        # 构造检索 query：把最近一轮用户历史问题拼进来，让指代（如"这门课"）能命中完整语义
        search_query = question
        if history and len(history) >= 2:
            last_user_q = ""
            for msg in reversed(history):
                if msg.get("role") == "user":
                    last_user_q = str(msg.get("content", "")).strip()
                    break
            if last_user_q and last_user_q != question:
                search_query = f"{last_user_q} {question}"

        # 1. 检索 Top-K（至少取 5 条提升召回率）
        try:
            docs = self.vectorstore.similarity_search(search_query, k=max(TOP_K, 5))
        except Exception as exc:
            logger.warning("[RAG] 知识库检索失败: %s", type(exc).__name__)
            raise ServiceUnavailableError("知识库检索暂时不可用，请稍后重试。") from exc
        if not docs:
            return {"answer": "没有在知识库中找到相关内容。", "sources": []}

        # 2. 组织带来源编号的上下文 + 收集来源信息
        context_parts = []
        sources = []
        for i, doc in enumerate(docs, start=1):
            src = doc.metadata.get("source", "未知来源")
            page = doc.metadata.get("page")
            page_text = f" 第{page}页" if page else ""
            context_parts.append(f"[{i}] 来源: {src}{page_text}\n{doc.page_content}")
            sources.append({"source": src, "page": page})

        context = "\n\n".join(context_parts)
        history_text = self._format_history(history)

        # 3. 生成答案
        prompt = self._build_prompt(question, context, history_text)
        try:
            answer = str(self.llm.invoke(prompt).content).strip()
        except Exception as exc:
            logger.warning("[RAG] 模型生成失败: %s", type(exc).__name__)
            raise ServiceUnavailableError("模型服务暂时不可用，请检查 API Key、额度和网络后重试。") from exc

        logger.info("[RAG] 已生成回答，引用 %d 个来源", len(sources))
        return {"answer": answer, "sources": sources}

    def chat_directly(self, question, history=None):
        """无需检索的普通对话（闲聊等），同样带历史上下文"""
        history_text = self._format_history(history)
        prompt = f"""你是一名友好的高校学业咨询助手，可以回答一般性问题，请使用中文。

【历史对话】
{history_text}

【用户问题】
{question}

请回答："""
        try:
            return str(self.llm.invoke(prompt).content).strip()
        except Exception as exc:
            logger.warning("[RAG] 普通对话生成失败: %s", type(exc).__name__)
            raise ServiceUnavailableError("模型服务暂时不可用，请检查 API Key、额度和网络后重试。") from exc

    def _format_history(self, history):
        """把历史消息列表转成纯文本"""
        if not history:
            return "（无历史对话）"
        lines = []
        for msg in history:
            role = "用户" if msg["role"] == "user" else "助手"
            lines.append(f"{role}: {msg['content']}")
        return "\n".join(lines)

    def _build_prompt(self, question, context, history_text):
        """构造带来源要求的提示词"""
        return f"""你是一名高校学业咨询助手，请基于下方“资料”回答用户问题。

要求：
1. 只能依据资料回答，资料中没有的内容请说明“资料中未提及”。
2. 引用资料时用方括号标注来源编号，例如 [1]、[2]。
3. 回答要简洁、准确，使用中文。

【历史对话】
{history_text}

【资料】
{context}

【用户问题】
{question}

请回答："""


# 全局单例
rag_service = RAGService()
