"""Gradio 前端：唯一职责是调用 FastAPI /api/chat 接口"""
import logging
from uuid import uuid4

import gradio as gr
import requests

logging.basicConfig(level=logging.INFO, format="[%(name)s] %(message)s")
logger = logging.getLogger("Frontend")

# 后端地址（先启动 FastAPI，再启动本前端）
BACKEND_URL = "http://127.0.0.1:8000/api/chat"


def respond(message, session_id):
    """请求后端并分别展示回答和可核查的引用来源。"""
    message = (message or "").strip()
    if not message:
        return "（请输入问题）", ""

    try:
        resp = requests.post(
            BACKEND_URL,
            json={"session_id": session_id, "message": message},
            timeout=100,
        )
        resp.raise_for_status()
        data = resp.json()
        sources = data.get("sources") or []
        source_text = "\n".join(
            f"[{index}] {item.get('source', '未知来源')}"
            + (f"（第 {item['page']} 页）" if item.get("page") is not None else "")
            for index, item in enumerate(sources, start=1)
        )
        return data.get("answer", "（后端无回答）"), source_text or "未返回可核查的来源。"
    except requests.HTTPError as exc:
        response = exc.response
        status_code = response.status_code if response is not None else "未知"
        detail = "后端暂时无法处理请求。"
        if response is not None and (status_code < 500 or status_code == 503):
            try:
                detail = response.json().get("detail", detail)
            except ValueError:
                pass
        logger.warning("[Frontend] 后端返回 HTTP %s", status_code)
        return f"请求失败（HTTP {status_code}）：{detail}", ""
    except requests.RequestException as exc:
        logger.warning("[Frontend] 后端请求失败: %s", type(exc).__name__)
        return "请求失败，请检查 FastAPI 服务或稍后重试。", ""
    except ValueError as exc:
        logger.warning("[Frontend] 后端响应无法解析: %s", type(exc).__name__)
        return "后端返回了无法解析的响应。", ""


# 每个 Gradio 浏览器会话使用独立 session_id；同一会话的连续请求复用该值。
demo = gr.Interface(
    fn=respond,
    inputs=[
        gr.Textbox(label="输入你的问题", placeholder="例如：高等数学多少学分？"),
        gr.State(value=lambda: uuid4().hex),
    ],
    outputs=[
        gr.Textbox(label="回答", lines=10),
        gr.Textbox(label="引用来源", lines=4, interactive=False),
    ],
    title="基于 RAG 的学业咨询助手",
    description="本地 Demo：回答与知识库来源分开展示。",
)


if __name__ == "__main__":
    demo.launch()
