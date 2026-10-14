"""聊天接口：/api/chat 与 /api/health"""
import logging
from typing import List, Optional
from uuid import uuid4

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.core.errors import ServiceUnavailableError
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
    session_id: str = Field(
        default_factory=lambda: uuid4().hex,
        min_length=1,
        max_length=128,
        description="会话 ID；调用方应在同一会话的多次请求中复用",
    )
    message: str = Field(..., description="用户问题")


class ChatResponse(BaseModel):
    """聊天响应"""
    session_id: str
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
    logger.info("[Chat] 收到提问（session=%s）", req.session_id[:8])
    try:
        # 意图识别失败时由 API 显式返回 503，不悄悄绕过知识库路由。
        intent = detect_intent(req.message)
        history = memory.get_history(req.session_id)

        if intent in (INTENT_ACADEMIC, INTENT_ANALYSIS):
            result = rag_service.query(req.message, history)
            answer, sources = result["answer"], result["sources"]
        else:
            answer = rag_service.chat_directly(req.message, history)
            sources = []
    except ServiceUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    # 仅在问答成功后写入，失败请求不会留下不完整的用户轮次。
    memory.add_message(req.session_id, "user", req.message)
    memory.add_message(req.session_id, "assistant", answer)

    return ChatResponse(
        session_id=req.session_id,
        answer=answer,
        intent=intent,
        sources=sources,
    )
