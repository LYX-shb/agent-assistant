"""FastAPI 入口：创建应用、注册路由、启动时构建向量索引"""
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.chat import router as chat_router
from app.core.config import configure_logging
from app.services.rag_service import rag_service

configure_logging()
logger = logging.getLogger("App")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用启动/关闭时的钩子：构建向量索引"""
    logger.info("[App] 正在启动，准备知识库索引...")
    try:
        rag_service.load_or_build()
    except Exception as e:
        # 索引构建失败不阻断启动，聊天时会返回友好提示
        logger.error("[App] 知识库索引构建失败: %s", type(e).__name__)
    yield
    logger.info("[App] 正在关闭...")


app = FastAPI(title="基于 RAG 的学业咨询助手", version="1.0.0", lifespan=lifespan)

# 注册聊天路由
app.include_router(chat_router)


@app.get("/")
def root():
    return {"message": "智能学业咨询 Agent API 已启动，请访问 /docs 查看接口文档"}
