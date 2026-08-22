"""聊天接口：/api/chat 与 /api/health"""
import logging
from typing import List, Optional

from fastapi import APIRouter
from pydantic import BaseModel, Field

from app.core.memory import memory
from app.services.intent import detect_intent, INTENT_ACADEMIC, INTENT_ANALYSIS
from app.services.rag_service import rag_service

logger = logging.getLogger("Chat")

router = APIRouter()


# ---------- Pydantic 模型 ----------

class Source(BaseModel):
    """引用来源"""
    source: str
    page: Optional[int] = None


class ChatRequest(BaseModel):
    """聊天请求"""
    session_id: str = Field(default="default", description="会话 ID，用于区分不同用户/会话")
    message: str = Field(..., description="用户问题")


class ChatResponse(BaseModel):
    """聊天响应"""
    answer: str
    intent: str
    sources: List[Source] = Field(default_factory=list)


# ---------- 路由 ----------

@router.get("/api/health")
def health_check():
    """健康检查接口"""
    return {"status": "ok"}


@router.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    logger.info("[Chat] 收到提问: %.30s...", req.message)

    # 1. 意图识别
    intent = detect_intent(req.message)

    # 2. 读取最近几轮历史，用于追问
    history = memory.get_history(req.session_id)

    # 3. 根据意图路由：学业咨询 / 数据分析走 RAG，其他走普通对话
    if intent in (INTENT_ACADEMIC, INTENT_ANALYSIS):
        result = rag_service.query(req.message, history)
        answer, sources = result["answer"], result["sources"]
    else:
        answer = rag_service.chat_directly(req.message, history)
        sources = []

    # 4. 记录本轮对话（供下一轮追问使用）
    memory.add_message(req.session_id, "user", req.message)
    memory.add_message(req.session_id, "assistant", answer)

    return ChatResponse(answer=answer, intent=intent, sources=sources)
