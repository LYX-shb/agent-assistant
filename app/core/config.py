"""全局配置：阿里云 DashScope（通义千问）API 配置"""
import os
import logging
from pathlib import Path

# 阿里云百炼 / DashScope 兼容 OpenAI 的接口地址
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

# 大语言模型（用于问答、意图识别）
LLM_MODEL = "qwen-turbo"

# 向量化模型（用于把文本变成向量，存入 ChromaDB）
EMBEDDING_MODEL = "text-embedding-v2"

# API Key：优先从环境变量读取，其次使用默认值
# 安全提示：这个 key 已出现在聊天记录中，建议去阿里云控制台轮换，
#          并改用环境变量 export ALIBABA_API_KEY=xxx 的方式注入。
ALIBABA_API_KEY = os.getenv("ALIBABA_API_KEY", "")

# 项目根目录（app/core/config.py 向上三级）
BASE_DIR = Path(__file__).resolve().parent.parent.parent

# 知识库目录（存放 PDF / CSV）
KNOWLEDGE_DIR = BASE_DIR / "knowledge"

# 向量库持久化目录
CHROMA_DIR = BASE_DIR / "chroma_db"

# SQLite 记忆库文件路径
MEMORY_DB_PATH = BASE_DIR / "chat_memory.db"

# 检索返回的 Top-K 数量
TOP_K = 3

# 多轮对话记忆保留的轮数
MEMORY_ROUNDS = 3


def configure_logging():
    """统一日志格式：[模块名] 消息内容"""
    logging.basicConfig(
        level=logging.INFO,
        format="[%(name)s] %(message)s",
    )
