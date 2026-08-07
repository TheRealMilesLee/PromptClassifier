"""Inference wrapper for a saved TensorFlow task model."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

from .tf_data import EFFORT_BUCKET_ORDER, bucket_from_minutes, normalize_model_text
from .tf_model import require_tensorflow
from .tagger import TaskTagger


class TensorFlowTaskModel:
    def __init__(self, model: Any, metadata: dict[str, Any]) -> None:
        self.model = model
        self.metadata = metadata
        self.workload_labels = list(metadata["labels"]["workload_type"])
        self.intent_labels = list(metadata["labels"]["intent"])
        self.complexity_labels = list(metadata["labels"]["complexity"])
        self.model_id = str(metadata.get("model_id", "trendmodel-text-v1"))
        # The TensorFlow heads learn the prompt distribution.  This compact,
        # dependency-free teacher contributes only high-confidence structured
        # evidence (language, explicit action, command/file signals, etc.) at
        # inference time.  It is the evidence-fusion layer of the router
        # design, not a replacement for the trained model.
        self.structural_tagger = TaskTagger()

    @classmethod
    def load(cls, artifact_dir: str | Path) -> "TensorFlowTaskModel":
        tf = require_tensorflow()
        directory = Path(artifact_dir)
        model_path = directory / "model.keras"
        metadata_path = directory / "metadata.json"
        if not model_path.exists() or not metadata_path.exists():
            raise FileNotFoundError(
                f"artifact must contain model.keras and metadata.json: {directory}"
            )
        model = tf.keras.models.load_model(model_path, compile=False)
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        return cls(model, metadata)

    def predict(self, prompt: str) -> dict[str, Any]:
        if not str(prompt).strip():
            raise ValueError("prompt must not be empty")
        prompt = str(prompt)
        tf = require_tensorflow()
        raw = self.model.predict(tf.constant([normalize_model_text(prompt)]), verbose=0)

        workload_probabilities = _probabilities(raw, "workload_type")
        intent_probabilities = _probabilities(raw, "intent")
        complexity_probabilities = _probabilities(raw, "complexity")
        structural = self.structural_tagger.predict(prompt).to_dict()
        structural_evidence = structural.get("evidence", [])
        workload_probabilities = _fuse_structural_evidence(
            workload_probabilities,
            self.workload_labels,
            structural.get("workload_type", {}).get("label"),
            _evidence_strength(structural_evidence, {"language", "operation", "domain_hint", "constraint"}),
        )
        intent_probabilities = _fuse_structural_evidence(
            intent_probabilities,
            self.intent_labels,
            structural.get("intent", {}).get("label"),
            _evidence_strength(structural_evidence, {"intent_hint"}),
        )
        complexity_probabilities = _fuse_structural_evidence(
            complexity_probabilities,
            self.complexity_labels,
            structural.get("complexity", {}).get("label"),
            _evidence_strength(structural_evidence, {"language", "operation", "constraint", "intent_hint"}),
        )
        workload_type = self.workload_labels[_argmax(workload_probabilities)]
        intent = self.intent_labels[_argmax(intent_probabilities)]
        complexity = self.complexity_labels[_argmax(complexity_probabilities)]

        clarification_probability = float(_first_value(raw, "needs_clarification"))
        reasons = _clarification_reasons(prompt, clarification_probability)
        needs_clarification = bool(reasons) or clarification_probability >= 0.5
        if needs_clarification and not structural_evidence:
            workload_type = "unknown"
            intent = "unknown"

        workload_confidence = max(workload_probabilities)
        intent_confidence = max(intent_probabilities)
        complexity_confidence = max(complexity_probabilities)
        effort_available = _effort_head_is_supervised(self.metadata)
        if effort_available:
            effort_log_minutes = float(_first_value(raw, "effort_log_minutes"))
            estimated_minutes = max(1, min(round(math.expm1(effort_log_minutes)), 10_080))
            effort_bucket = bucket_from_minutes(estimated_minutes)
            if effort_bucket not in EFFORT_BUCKET_ORDER:
                effort_bucket = "M"
            effort_range = _effort_range(effort_bucket)
            estimated_minutes = max(effort_range[0], min(estimated_minutes, effort_range[1]))
            if _has_observed_effort_supervision(self.metadata):
                effort_confidence = min(
                    0.95,
                    max(0.25, 0.45 + 0.2 * complexity_confidence - 0.2 * bool(reasons)),
                )
                effort_basis = "trained_observed_effort_head"
            else:
                # Bucket-only labels teach an ordinal prior, not calibrated
                # wall-clock minutes.  Keep the estimate useful for ranking,
                # but make the uncertainty visible to the downstream router.
                effort_confidence = min(
                    0.35,
                    max(0.15, 0.20 + 0.10 * complexity_confidence - 0.10 * bool(reasons)),
                )
                effort_basis = "bucket_seed_only"
            effort_payload = {
                "label": effort_bucket,
                "estimated_minutes": estimated_minutes,
                "range_minutes": effort_range,
                "confidence": round(effort_confidence, 4),
                "basis": effort_basis,
            }
        else:
            # A public-only corpus has no effort target.  Returning a number in
            # that situation would turn missing supervision into fake ground
            # truth and could make the router choose an unsafe model.
            effort_payload = {
                "label": "unknown",
                "estimated_minutes": None,
                "range_minutes": None,
                "confidence": 0.0,
                "basis": "no_effort_supervision",
            }

        prediction_payload = {
            "workload_type": _classification_payload(
                workload_type,
                workload_probabilities,
                self.workload_labels,
            ),
            "intent": _classification_payload(
                intent,
                intent_probabilities,
                self.intent_labels,
            ),
            "complexity": _classification_payload(
                complexity,
                complexity_probabilities,
                self.complexity_labels,
            ),
            "effort": effort_payload,
        }
        overall_confidence = round(
            max(
                0.0,
                min(
                    1.0,
                    0.30 * workload_confidence
                    + 0.30 * intent_confidence
                    + 0.25 * complexity_confidence
                    + 0.15 * float(effort_payload["confidence"]),
                ),
            ),
            4,
        )

        return {
            "original_prompt": prompt,
            # Keep these three scalar fields for the original router adapter.
            # The richer, probability-bearing representation lives under
            # ``predictions`` and is the preferred new contract.
            "workload_type": workload_type,
            "intent": intent,
            "complexity": complexity,
            "predictions": prediction_payload,
            "evidence": structural_evidence,
            "fusion": {
                "semantic_model": self.model_id,
                "structural_teacher": getattr(self.structural_tagger, "model_version", "unknown"),
                "policy": "neural probabilities plus high-confidence structured evidence",
            },
            "effort": {
                "bucket": effort_payload["label"],
                "estimated_minutes": effort_payload["estimated_minutes"],
                "range_minutes": effort_payload["range_minutes"],
                "confidence": effort_payload["confidence"],
                "basis": effort_payload["basis"],
            },
            "needs_clarification": needs_clarification,
            "clarification_reasons": reasons,
            "confidence": {
                "workload_type": round(workload_confidence, 4),
                "intent": round(intent_confidence, 4),
                "complexity": round(complexity_confidence, 4),
                "effort": effort_payload["confidence"],
                "overall": overall_confidence,
            },
            "model_version": self.metadata.get("model_version", "0.1.0"),
        }

    def predict_many(self, prompts: list[str]) -> list[dict[str, Any]]:
        """Predict a batch using the same JSON contract as ``predict``."""

        return [self.predict(prompt) for prompt in prompts]


def _probabilities(raw: Any, name: str) -> list[float]:
    value = raw[name] if isinstance(raw, dict) else raw
    if hasattr(value, "tolist"):
        value = value.tolist()
    while isinstance(value, list) and len(value) == 1:
        value = value[0]
    return [float(item) for item in value]


def _first_value(raw: Any, name: str) -> float:
    value = raw[name] if isinstance(raw, dict) else raw
    if hasattr(value, "numpy"):
        value = value.numpy()
    if hasattr(value, "tolist"):
        value = value.tolist()
    while isinstance(value, list):
        value = value[0]
    return float(value)


def _argmax(values: list[float]) -> int:
    return max(range(len(values)), key=values.__getitem__)


def _classification_payload(
    label: str,
    probabilities: list[float],
    labels: list[str],
    *,
    limit: int = 3,
) -> dict[str, Any]:
    ranked = sorted(
        zip(labels, probabilities),
        key=lambda item: (-item[1], item[0]),
    )[:limit]
    return {
        "label": label,
        "probability": round(
            probabilities[labels.index(label)] if label in labels else 0.0,
            4,
        ),
        "candidates": [
            {"label": candidate, "probability": round(float(probability), 4)}
            for candidate, probability in ranked
        ],
    }


def _evidence_strength(evidence: Any, kinds: set[str]) -> float:
    """Return the strongest structured signal for a fusion head."""

    strongest = 0.0
    if not isinstance(evidence, list):
        return strongest
    for signal in evidence:
        if not isinstance(signal, dict) or signal.get("kind") not in kinds:
            continue
        try:
            strongest = max(strongest, float(signal.get("strength", 0.0)))
        except (TypeError, ValueError):
            continue
    return max(0.0, min(1.0, strongest))


def _fuse_structural_evidence(
    probabilities: list[float],
    labels: list[str],
    structural_label: Any,
    strength: float,
) -> list[float]:
    """Apply a bounded one-class likelihood boost from explicit evidence.

    The neural distribution is retained as the base prior.  A high-confidence
    structural observation contributes at most 72% of the posterior, so an
    evidence extractor cannot silently turn every request into a hard rule.
    """

    if structural_label not in labels or strength <= 0.0:
        return probabilities
    alpha = min(0.72, 0.72 * strength)
    fused = [
        (1.0 - alpha) * float(probability)
        + (alpha if label == structural_label else 0.0)
        for label, probability in zip(labels, probabilities)
    ]
    total = sum(fused)
    if total <= 0.0:
        return probabilities
    return [value / total for value in fused]


def _effort_head_is_supervised(metadata: dict[str, Any]) -> bool:
    value = metadata.get("effort_supervision_samples")
    if value is None:
        # Artifacts produced before the explicit supervision counter are
        # treated as legacy models and retain their previous behavior.
        return True
    try:
        return int(value) > 0
    except (TypeError, ValueError):
        return False


def _has_observed_effort_supervision(metadata: dict[str, Any]) -> bool:
    try:
        return int(metadata.get("observed_effort_samples", 0)) > 0
    except (TypeError, ValueError):
        return False


def _clarification_reasons(prompt: str, probability: float) -> list[str]:
    compact = re.sub(r"\s+", "", prompt)
    reasons: list[str] = []
    if len(compact) < 8:
        reasons.append("输入过短，缺少任务目标或交付物")
    if re.search(r"帮我弄一下|处理一下|看看|搞一下|优化一下|fix this|take a look", prompt, re.I):
        reasons.append("目标描述偏模糊")
    if probability >= 0.5:
        reasons.append("模型判断需要补充信息")
    return reasons


def _has_specific_signal(prompt: str) -> bool:
    return bool(
        re.search(
            r"代码|函数|脚本|bug|api|sql|python|部署|kubernetes|k8s|dns|服务器|邮件|翻译|文章|报告|计划|分析|评估",
            prompt,
            re.I,
        )
    )


def _effort_range(bucket: str) -> list[int]:
    return {
        "XS": [5, 15],
        "S": [15, 45],
        "M": [45, 120],
        "L": [120, 480],
        "XL": [1_440, 4_320],
    }.get(bucket, [45, 120])
