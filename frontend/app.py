"""Gradio 前端：唯一职责是调用 FastAPI /api/chat 接口"""
import logging

import gradio as gr
import requests

logging.basicConfig(level=logging.INFO, format="[%(name)s] %(message)s")
logger = logging.getLogger("Frontend")

# 后端地址（先启动 FastAPI，再启动本前端）
BACKEND_URL = "http://127.0.0.1:8000/api/chat"
# 历史对话由后端 memory 管理，前端固定一个 session_id 即可
SESSION_ID = "user_001"


def respond(message):
    """调用后端接口，返回后端返回的 answer 字段"""
    message = (message or "").strip()
    if not message:
        return "（请输入问题）"

    try:
        resp = requests.post(
            BACKEND_URL,
            json={"session_id": SESSION_ID, "message": message},
            timeout=60,
        )
        resp.raise_for_status()
        data = resp.json()
        return data.get("answer", "（后端无回答）")
    except Exception as e:
        logger.error("[Frontend] 请求后端失败: %s", e)
        return f"请求后端失败：{e}"


# 简单输入框 + 输出框，不涉及 ChatInterface / Chatbot 消息格式
demo = gr.Interface(
    fn=respond,
    inputs=gr.Textbox(label="输入你的问题", placeholder="例如：高等数学多少学分？"),
    outputs=gr.Textbox(label="回答", lines=10),
    title="智能学业咨询 Agent",
    description="基于 RAG 的学业咨询助手，回答会标注引用来源。",
)


if __name__ == "__main__":
    demo.launch()
