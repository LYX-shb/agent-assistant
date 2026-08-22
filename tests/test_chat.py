"""pytest 测试：健康检查、单轮问答、多轮记忆

说明：测试用 monkeypatch 替换掉真实的 LLM / 向量检索调用，
      这样测试不联网、不花 API 费用，且结果稳定可重复。
"""
from fastapi.testclient import TestClient

from app.main import app
import app.api.chat as chat_module
from app.core.memory import MemoryStore

# 不进入 with 上下文，避免触发 lifespan 去联网构建向量索引
client = TestClient(app)


def test_health_check():
    """健康检查接口应返回 ok"""
    resp = client.get("/api/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_single_round_chat(monkeypatch):
    """单轮问答：意图为学业咨询时走 RAG，返回答案与来源"""
    monkeypatch.setattr(chat_module, "detect_intent", lambda q: "学业咨询")
    monkeypatch.setattr(
        chat_module.rag_service,
        "query",
        lambda q, h: {
            "answer": "根据资料，高等数学是 5 学分 [1]。",
            "sources": [{"source": "sample.csv", "page": None}],
        },
    )

    resp = client.post(
        "/api/chat",
        json={"session_id": "test_single", "message": "高等数学多少学分？"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["answer"]
    assert data["intent"] == "学业咨询"
    assert len(data["sources"]) == 1
    assert data["sources"][0]["source"] == "sample.csv"


def test_multi_round_memory(monkeypatch, tmp_path):
    """多轮记忆：两轮对话后，临时库中应有 4 条消息"""
    # 用临时库，避免污染真实记忆库
    tmp_memory = MemoryStore(tmp_path / "test.db")
    monkeypatch.setattr(chat_module, "memory", tmp_memory)
    monkeypatch.setattr(chat_module, "detect_intent", lambda q: "其他")
    monkeypatch.setattr(chat_module.rag_service, "chat_directly", lambda q, h: f"收到：{q}")

    sid = "mem_test"
    client.post("/api/chat", json={"session_id": sid, "message": "你好"})
    client.post("/api/chat", json={"session_id": sid, "message": "我上句说了什么"})

    history = tmp_memory.get_history(sid)
    assert len(history) == 4  # 2 轮 = 2 条 user + 2 条 assistant
    assert history[0]["role"] == "user"
    assert history[0]["content"] == "你好"
    assert history[-1]["role"] == "assistant"
    assert history[-1]["content"] == "收到：我上句说了什么"
