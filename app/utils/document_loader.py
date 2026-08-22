"""多格式文档加载：PDF 按段落切分，CSV 按行加载"""
import logging
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader, CSVLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger("Loader")

# 中文友好的分隔符：优先按空行/换行，再按中文标点
CHINESE_SEPARATORS = ["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""]


def load_pdf(path):
    """加载单个 PDF，每页作为一个文档，并补充页码（从 1 开始）"""
    loader = PyPDFLoader(str(path))
    docs = loader.load()
    result = []
    for doc in docs:
        page = doc.metadata.get("page", 0) + 1  # PyPDFLoader 页码从 0 开始
        result.append(
            Document(
                page_content=doc.page_content,
                metadata={"source": Path(path).name, "page": page},
            )
        )
    return result


def load_csv(path):
    """加载单个 CSV，每行作为一个文档，来源记为文件名"""
    loader = CSVLoader(str(path), encoding="utf-8")
    docs = loader.load()
    result = []
    for doc in docs:
        result.append(
            Document(
                page_content=doc.page_content,
                metadata={"source": Path(path).name},  # 无页码概念
            )
        )
    return result


def split_documents(docs, chunk_size=500, chunk_overlap=50):
    """把长文档按段落/标点切分为固定大小的文本块"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=CHINESE_SEPARATORS,
    )
    return splitter.split_documents(docs)


def load_knowledge_dir(knowledge_dir):
    """加载知识库目录下所有 PDF 与 CSV，返回切分后的文档列表"""
    all_docs = []
    for pdf in sorted(Path(knowledge_dir).glob("*.pdf")):
        logger.info("[Loader] 加载 PDF: %s", pdf.name)
        all_docs.extend(load_pdf(pdf))
    for csv in sorted(Path(knowledge_dir).glob("*.csv")):
        logger.info("[Loader] 加载 CSV: %s", csv.name)
        all_docs.extend(load_csv(csv))

    chunks = split_documents(all_docs)
    logger.info("[Loader] 共切分为 %d 个文本块", len(chunks))
    return chunks
