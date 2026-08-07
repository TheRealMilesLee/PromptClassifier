from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from .estimator import (
    COMPLEXITY_LEVELS,
    COMPLEXITY_NAMES,
    EFFORT_LEVELS,
    EFFORT_NAMES,
    EFFORT_RANGES,
    clarification_reasons,
    complexity_label_from_score,
    heuristic_complexity_score,
    heuristic_effort_index,
    effort_label_from_index,
)
from .features import SIGNAL_DISPLAY_NAMES, matched_signal_categories, normalize_text
from .nb import MultinomialNB
from .schema import TrainingExample
from .vectorizer import TfidfVectorizer


WORK_TYPE_NAMES = {
    "text": "文字",
    "code": "代码",
    "analysis": "分析",
    "operations": "运维",
    "planning": "规划",
    "mixed": "混合",
    "unknown": "未明确",
}

INTENT_NAMES = {
    "create": "创建/生成",
    "modify": "修改/优化",
    "explain": "解释/讲解",
    "debug": "排障/修复",
    "analyze": "分析/比较",
    "summarize": "总结/提炼",
    "translate": "翻译",
    "plan": "规划/拆解",
    "review": "评审/检查",
    "extract": "提取/结构化",
    "classify": "分类/归类",
    "unknown": "未明确",
}

HEADS = ("work_type", "intent", "complexity", "effort")

WORK_SIGNAL_TO_LABEL = {
    "work_text": "text",
    "work_code": "code",
    "work_operations": "operations",
}

INTENT_SIGNAL_TO_LABEL = {
    "action_translate": "translate",
    "action_debug": "debug",
    "action_summarize": "summarize",
    "action_extract": "extract",
    "action_classify": "classify",
    "action_review": "review",
    "action_plan": "plan",
    "action_modify": "modify",
    "action_explain": "explain",
    "action_analyze": "analyze",
    "action_create": "create",
}


def _as_examples(examples: Iterable[TrainingExample | Mapping[str, object]]) -> list[TrainingExample]:
    result: list[TrainingExample] = []
    for example in examples:
        if isinstance(example, TrainingExample):
            result.append(example)
        else:
            result.append(TrainingExample.from_mapping(example))
    if not result:
        raise ValueError("at least one training example is required")
    return result


def _ranked(probabilities: Mapping[str, float], names: Mapping[str, str], limit: int = 3) -> list[dict[str, object]]:
    return [
        {
            "label": label,
            "name": names.get(label, label),
            "score": round(score, 4),
        }
        for label, score in sorted(probabilities.items(), key=lambda item: (-item[1], item[0]))[:limit]
    ]


def _prediction(
    label: str,
    probability: float,
    names: Mapping[str, str],
    candidates: list[dict[str, object]],
    **extra: object,
) -> dict[str, object]:
    result: dict[str, object] = {
        "label": label,
        "name": names.get(label, label),
        "confidence": round(probability, 4),
        "candidates": candidates,
    }
    result.update(extra)
    return result


class TaskModel:
    """Trainable multi-head classifier for task routing and rough effort sizing."""

    model_version = "0.1.0"

    def __init__(self, vectorizer: TfidfVectorizer, classifiers: Mapping[str, MultinomialNB]) -> None:
        self.vectorizer = vectorizer
        self.classifiers = dict(classifiers)
        missing = [head for head in HEADS if head not in self.classifiers]
        if missing:
            raise ValueError(f"missing classifiers: {', '.join(missing)}")

    @classmethod
    def fit(
        cls,
        examples: Iterable[TrainingExample | Mapping[str, object]],
        *,
        max_features: int = 20_000,
    ) -> "TaskModel":
        labelled = _as_examples(examples)
        vectorizer = TfidfVectorizer(max_features=max_features).fit(example.text for example in labelled)
        vectors = vectorizer.transform_many(example.text for example in labelled)
        classifiers = {
            head: MultinomialNB(alpha=0.1).fit(
                vectors,
                [getattr(example, head) for example in labelled],
            )
            for head in HEADS
        }
        return cls(vectorizer, classifiers)

    @classmethod
    def fit_jsonl(cls, path: str | Path, *, max_features: int = 20_000) -> "TaskModel":
        return cls.fit(load_examples(path), max_features=max_features)

    def predict(self, text: str) -> dict[str, object]:
        if not str(text).strip():
            raise ValueError("text must not be empty")

        text = str(text)
        vector = self.vectorizer.transform(text)
        probabilities = {
            head: self.classifiers[head].predict_proba(vector)
            for head in HEADS
        }

        work_type = _select_work_type(text, probabilities["work_type"])
        intent = _select_intent(text, probabilities["intent"])

        learned_complexity_value = sum(
            index * probabilities["complexity"].get(label, 0.0)
            for index, label in enumerate(COMPLEXITY_LEVELS)
        )
        rule_complexity_value = heuristic_complexity_score(text, work_type, intent)
        combined_complexity_value = 0.6 * learned_complexity_value + 0.4 * rule_complexity_value
        complexity = complexity_label_from_score(combined_complexity_value)
        complexity_confidence = _confidence_for_value(
            combined_complexity_value,
            thresholds=(0.72, 1.45),
            learned_max=max(probabilities["complexity"].values()),
        )

        rule_effort_value = heuristic_effort_index(text, work_type, intent, complexity)
        learned_effort_value = sum(
            index * probabilities["effort"].get(label, 0.0)
            for index, label in enumerate(EFFORT_LEVELS)
        )
        learned_effort_value = learned_effort_value * (5.0 / max(len(EFFORT_LEVELS) - 1, 1))
        combined_effort_value = 0.7 * rule_effort_value + 0.3 * learned_effort_value
        effort = effort_label_from_index(combined_effort_value)
        effort_confidence = _confidence_for_effort(
            combined_effort_value,
            max(probabilities["effort"].values()),
        )

        reasons = clarification_reasons(
            text,
            probabilities["work_type"],
            probabilities["intent"],
        )
        ambiguous = _should_mark_unknown(text, reasons)
        display_work_type = "unknown" if ambiguous else work_type
        display_intent = "unknown" if ambiguous else intent
        if ambiguous:
            complexity_confidence = min(complexity_confidence, 0.35)
            effort_confidence = min(effort_confidence, 0.35)
        signals = self._explain_signals(text, display_work_type, display_intent, complexity, effort)

        return {
            "text": text,
            "work_type": (
                _unknown_prediction(WORK_TYPE_NAMES)
                if ambiguous
                else _prediction(
                    work_type,
                    probabilities["work_type"].get(work_type, 0.0),
                    WORK_TYPE_NAMES,
                    _ranked(probabilities["work_type"], WORK_TYPE_NAMES),
                )
            ),
            "intent": (
                _unknown_prediction(INTENT_NAMES)
                if ambiguous
                else _prediction(
                    intent,
                    probabilities["intent"].get(intent, 0.0),
                    INTENT_NAMES,
                    _ranked(probabilities["intent"], INTENT_NAMES),
                )
            ),
            "complexity": _prediction(
                complexity,
                complexity_confidence,
                COMPLEXITY_NAMES,
                _ranked(probabilities["complexity"], COMPLEXITY_NAMES),
                estimated_score=round(combined_complexity_value, 3),
            ),
            "effort": _prediction(
                effort,
                effort_confidence,
                EFFORT_NAMES,
                _ranked(probabilities["effort"], EFFORT_NAMES),
                time_range=EFFORT_RANGES[effort],
                estimated_score=round(combined_effort_value, 3),
            ),
            "needs_clarification": bool(reasons),
            "clarification_reasons": reasons,
            "signals": signals,
            "model": {
                "name": "trendmodel-baseline",
                "version": self.model_version,
                "method": "TF-IDF + multi-head Naive Bayes + rule-based sizing",
            },
        }

    def _explain_signals(
        self,
        text: str,
        work_type: str,
        intent: str,
        complexity: str,
        effort: str,
    ) -> list[str]:
        categories = matched_signal_categories(text)
        explanations: list[str] = []
        preferred = [
            "work_text",
            "work_code",
            "work_operations",
            "work_analysis",
            "work_planning",
            "work_mixed",
            "action_create",
            "action_modify",
            "action_debug",
            "action_explain",
            "action_analyze",
            "action_summarize",
            "action_translate",
            "action_plan",
            "action_review",
            "action_extract",
            "action_classify",
            "scope_large",
            "external_system",
            "simple_request",
            "vague_request",
        ]
        for category in preferred:
            if category in categories:
                explanations.append(SIGNAL_DISPLAY_NAMES[category])

        if not explanations:
            explanations.append("未检测到明显关键词，结果主要依赖整体文本特征")
        explanations.append(f"综合判断为：{WORK_TYPE_NAMES.get(work_type, work_type)}类，目标是{INTENT_NAMES.get(intent, intent)}")
        explanations.append(f"复杂度为{COMPLEXITY_NAMES[complexity]}，预计 effort 为{effort}（{EFFORT_RANGES[effort]}）")
        return explanations

    def save(self, path: str | Path) -> None:
        payload = {
            "model_version": self.model_version,
            "vectorizer": self.vectorizer.to_dict(),
            "classifiers": {head: self.classifiers[head].to_dict() for head in HEADS},
        }
        Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "TaskModel":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        vectorizer = TfidfVectorizer.from_dict(payload["vectorizer"])
        classifiers = {
            head: MultinomialNB.from_dict(payload["classifiers"][head])
            for head in HEADS
        }
        return cls(vectorizer, classifiers)


def _confidence_for_value(value: float, thresholds: tuple[float, float], learned_max: float) -> float:
    if value < thresholds[0]:
        distance = thresholds[0] - value
    elif value < thresholds[1]:
        distance = min(value - thresholds[0], thresholds[1] - value)
    else:
        distance = value - thresholds[1]
    return round(_bounded_confidence(0.45 + min(distance / 1.1, 0.35) + learned_max * 0.2), 4)


def _confidence_for_effort(value: float, learned_max: float) -> float:
    boundaries = (0.35, 0.95, 1.9, 3.2)
    distance = min((abs(value - boundary) for boundary in boundaries), default=1.0)
    return round(_bounded_confidence(0.48 + min(distance / 2.0, 0.25) + learned_max * 0.2), 4)


def _bounded_confidence(value: float) -> float:
    return max(0.01, min(0.99, value))


def _unknown_prediction(names: Mapping[str, str]) -> dict[str, object]:
    return {
        "label": "unknown",
        "name": names["unknown"],
        "confidence": 0.0,
        "candidates": [{"label": "unknown", "name": names["unknown"], "score": 0.0}],
    }


def _should_mark_unknown(text: str, reasons: Sequence[str]) -> bool:
    if not reasons:
        return False
    categories = matched_signal_categories(text)
    strong_categories = categories & (
        set(WORK_SIGNAL_TO_LABEL)
        | set(INTENT_SIGNAL_TO_LABEL)
        | {"work_mixed"}
    )
    vague_reasons = {"输入过短，缺少任务目标或交付物", "目标描述偏模糊"}
    return bool(vague_reasons & set(reasons)) and not strong_categories


def _select_work_type(text: str, probabilities: Mapping[str, float]) -> str:
    """Use unambiguous domain signals to stabilize tiny-corpus predictions."""

    categories = matched_signal_categories(text)
    if "work_mixed" in categories:
        return "mixed"

    primary = [
        label
        for category, label in WORK_SIGNAL_TO_LABEL.items()
        if category in categories
    ]
    if len(primary) == 1 and primary[0] in probabilities:
        return primary[0]
    if not primary:
        if "work_planning" in categories and "work_analysis" not in categories:
            return "planning"
        if "work_analysis" in categories and "work_planning" not in categories:
            return "analysis"
    return max(probabilities, key=probabilities.get)


def _select_intent(text: str, probabilities: Mapping[str, float]) -> str:
    categories = matched_signal_categories(text)
    matched = [
        label
        for category, label in INTENT_SIGNAL_TO_LABEL.items()
        if category in categories
    ]
    # A single explicit action is safer than a prior learned from a tiny seed
    # set. If several actions coexist, retain the statistical prediction.
    if len(matched) == 1 and matched[0] in probabilities:
        return matched[0]
    return max(probabilities, key=probabilities.get)


def load_examples(path: str | Path) -> list[TrainingExample]:
    examples: list[TrainingExample] = []
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        try:
            examples.append(TrainingExample.from_mapping(json.loads(line)))
        except (json.JSONDecodeError, ValueError) as error:
            raise ValueError(f"invalid training example at {path}:{line_number}: {error}") from error
    if not examples:
        raise ValueError(f"no training examples found in {path}")
    return examples
