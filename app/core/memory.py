"""SQLite 多轮对话记忆：保存并读取最近 N 轮对话"""
import logging
import sqlite3
from datetime import datetime

from app.core.config import MEMORY_DB_PATH, MEMORY_ROUNDS

logger = logging.getLogger("Memory")


class MemoryStore:
    """基于 SQLite 的对话记忆，按会话（session_id）隔离不同用户"""

    def __init__(self, db_path=None):
        # 允许测试时传入临时路径
        self.db_path = str(db_path or MEMORY_DB_PATH)
        self._init_db()

    def _init_db(self):
        """建表（不存在才建）"""
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
            conn.commit()
        finally:
            conn.close()
        logger.info("[Memory] 记忆库已就绪: %s", self.db_path)

    def add_message(self, session_id, role, content):
        """新增一条消息，role 为 user / assistant"""
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute(
                "INSERT INTO messages (session_id, role, content, created_at) VALUES (?, ?, ?, ?)",
                (session_id, role, content, datetime.now().isoformat()),
            )
            conn.commit()
        finally:
            conn.close()
        logger.info("[Memory] 已记录 %s 消息: %.30s...", role, content)

    def get_history(self, session_id, n_rounds=None):
        """读取最近 n_rounds 轮对话（每轮 = 1 条 user + 1 条 assistant）"""
        n_rounds = n_rounds or MEMORY_ROUNDS
        limit = n_rounds * 2
        conn = sqlite3.connect(self.db_path)
        try:
            rows = conn.execute(
                """
                SELECT role, content FROM (
                    SELECT id, role, content FROM messages
                    WHERE session_id = ?
                    ORDER BY id DESC LIMIT ?
                ) ORDER BY id ASC
                """,
                (session_id, limit),
            ).fetchall()
        finally:
            conn.close()
        return [{"role": r, "content": c} for r, c in rows]

    def clear(self, session_id=None):
        """清空记忆：不传 session_id 时清空全部"""
        conn = sqlite3.connect(self.db_path)
        try:
            if session_id:
                conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
            else:
                conn.execute("DELETE FROM messages")
            conn.commit()
        finally:
            conn.close()
        logger.info("[Memory] 已清空记忆（session=%s）", session_id or "全部")


# 全局单例，供 API 与测试复用
memory = MemoryStore()
