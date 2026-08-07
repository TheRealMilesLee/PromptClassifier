from __future__ import annotations

import re
from typing import Mapping

from .features import matched_signal_categories, normalize_text


COMPLEXITY_LEVELS = ("low", "medium", "high")
EFFORT_LEVELS = ("XS", "S", "M", "L", "XL")

COMPLEXITY_NAMES = {"low": "低", "medium": "中", "high": "高"}
EFFORT_NAMES = {"XS": "极小", "S": "小", "M": "中等", "L": "大", "XL": "很大"}
EFFORT_RANGES = {
    "XS": "5–15 分钟",
    "S": "15–45 分钟",
    "M": "45 分钟–2 小时",
    "L": "2–8 小时",
    "XL": "1–3 天以上",
}

_MULTI_STEP_RE = re.compile(r"多个|同时|并且|以及|然后|另外|还要|再加上|and|then|also|plus|multi", re.IGNORECASE)
_SIMPLE_RE = re.compile(r"只需|简单|一句话|一个|单个|仅|only|just|one", re.IGNORECASE)


def _bounded(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _level_value(probabilities: Mapping[str, float], levels: tuple[str, ...]) -> float:
    return sum(index * float(probabilities.get(label, 0.0)) for index, label in enumerate(levels))


def heuristic_complexity_score(text: str, work_type: str, intent: str) -> float:
    """Return a 0..2 complexity signal from scope and environment indicators."""

    normalized = normalize_text(text)
    signals = matched_signal_categories(normalized)
    compact_length = len(normalized.replace(" ", ""))
    score = 0.25

    score += {
        "text": 0.0,
        "code": 0.35,
        "analysis": 0.2,
        "operations": 0.65,
        "planning": 0.3,
        "mixed": 0.75,
    }.get(work_type, 0.25)
    score += {
        "create": 0.1,
        "modify": 0.1,
        "debug": 0.25,
        "explain": -0.1,
        "analyze": 0.1,
        "summarize": -0.15,
        "translate": -0.2,
        "plan": 0.2,
        "review": 0.1,
        "extract": 0.0,
        "classify": 0.0,
    }.get(intent, 0.0)

    if "scope_large" in signals:
        score += 0.45
    if "external_system" in signals:
        score += 0.35
    if _MULTI_STEP_RE.search(normalized):
        score += 0.35
    if compact_length >= 100:
        score += 0.15
    if compact_length >= 240:
        score += 0.25
    if "simple_request" in signals or _SIMPLE_RE.search(normalized):
        score -= 0.35

    return _bounded(score, 0.0, 2.0)


def complexity_label_from_score(score: float) -> str:
    if score < 0.72:
        return "low"
    if score < 1.45:
        return "medium"
    return "high"


def heuristic_effort_index(text: str, work_type: str, intent: str, complexity: str) -> float:
    normalized = normalize_text(text)
    signals = matched_signal_categories(normalized)
    compact_length = len(normalized.replace(" ", ""))
    score = {"low": 0.25, "medium": 1.0, "high": 2.1}.get(complexity, 1.0)

    if work_type in {"operations", "mixed"}:
        score += 0.45
    elif work_type == "code":
        score += 0.25
    if "scope_large" in signals:
        score += 0.65
    if "external_system" in signals:
        score += 0.45
    if _MULTI_STEP_RE.search(normalized):
        score += 0.35
    if compact_length >= 160:
        score += 0.25
    if compact_length >= 360:
        score += 0.35
    if "simple_request" in signals or _SIMPLE_RE.search(normalized):
        score -= 0.35
    if intent in {"translate", "summarize", "explain"} and work_type == "text":
        score -= 0.2

    return _bounded(score, 0.0, 5.0)


def effort_label_from_index(index: float) -> str:
    if index <= 0.35:
        return "XS"
    if index <= 0.95:
        return "S"
    if index <= 1.9:
        return "M"
    if index <= 3.2:
        return "L"
    return "XL"


def clarification_reasons(
    text: str,
    work_probabilities: Mapping[str, float],
    intent_probabilities: Mapping[str, float],
) -> list[str]:
    normalized = normalize_text(text)
    compact = normalized.replace(" ", "")
    work_ranked = sorted(work_probabilities.values(), reverse=True)
    intent_ranked = sorted(intent_probabilities.values(), reverse=True)
    reasons: list[str] = []

    if len(compact) < 8:
        reasons.append("输入过短，缺少任务目标或交付物")
    if _SIMPLE_RE.search(normalized) and len(compact) < 20:
        reasons.append("只描述了动作，没有说明对象或验收标准")
    if matched_signal_categories(normalized) & {"vague_request"}:
        reasons.append("目标描述偏模糊")
    if work_ranked and work_ranked[0] < 0.48:
        reasons.append("工作类型置信度偏低")
    if len(work_ranked) > 1 and work_ranked[0] - work_ranked[1] < 0.1:
        reasons.append("可能同时涉及多种工作类型")
    if intent_ranked and intent_ranked[0] < 0.45:
        reasons.append("用户意图置信度偏低")
    return reasons
