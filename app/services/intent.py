"""意图识别：用 LLM 判断用户问题属于学业咨询 / 数据分析 / 其他"""
import logging

from langchain_openai import ChatOpenAI

from app.core.config import BASE_URL, LLM_MODEL, ALIBABA_API_KEY
from app.core.errors import ServiceUnavailableError

logger = logging.getLogger("Intent")

# 三种可能的意图
INTENT_ACADEMIC = "学业咨询"
INTENT_ANALYSIS = "数据分析"
INTENT_OTHER = "其他"

# 让 LLM 只输出类别名，便于解析
INTENT_PROMPT = """你是意图识别助手。请判断用户的问题是下面哪一类，只输出类别名称，不要解释。

类别：
1. 学业咨询 —— 询问课程、选课、培养方案、学分、专业、成绩等学业相关问题
2. 数据分析 —— 询问数据统计、成绩对比、平均分等需要分析数据的问题
3. 其他 —— 闲聊或其他无关问题

用户问题：{question}

类别："""


def _build_llm():
    """创建通义千问的 LLM 实例"""
    return ChatOpenAI(
        model=LLM_MODEL,
        api_key=ALIBABA_API_KEY,
        base_url=BASE_URL,
        temperature=0,  # 意图识别要稳定，温度设为 0
        request_timeout=30,
        max_retries=0,
    )


def detect_intent(question):
    """返回三个意图之一；上游服务失败时向 API 层报告错误。"""
    try:
        llm = _build_llm()
        prompt = INTENT_PROMPT.format(question=question)
        raw = str(llm.invoke(prompt).content).strip()
    except Exception as e:
        logger.warning("[Intent] 意图识别失败: %s", type(e).__name__)
        raise ServiceUnavailableError("意图识别服务暂时不可用，请稍后重试。") from e

    # 归一化：从返回文本中匹配关键词，避免多余符号干扰
    if INTENT_ANALYSIS in raw:
        intent = INTENT_ANALYSIS
    elif INTENT_ACADEMIC in raw:
        intent = INTENT_ACADEMIC
    else:
        intent = INTENT_OTHER

    logger.info("[Intent] 问题「%.20s...」 → %s", question, intent)
    return intent
