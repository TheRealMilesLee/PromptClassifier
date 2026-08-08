"""Inference wrapper for a saved TensorFlow task model."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Mapping

from .tf_data import (
    EFFORT_BUCKET_MIDPOINTS,
    EFFORT_BUCKET_ORDER,
    bucket_from_minutes,
    normalize_model_text,
)
from .tf_features import structured_feature_matrix
from .tf_model import require_tensorflow
from .tagger import TaskTagger


class TensorFlowTaskModel:
    def __init__(self, model: Any, metadata: dict[str, Any]) -> None:
        self.model = model
        self.metadata = metadata
        self.workload_labels = list(metadata["labels"]["workload_type"])
        self.intent_labels = list(metadata["labels"]["intent"])
        self.complexity_labels = list(metadata["labels"]["complexity"])
        self.effort_bucket_labels = list(
            metadata.get("labels", {}).get("effort_bucket", EFFORT_BUCKET_ORDER)
        )
        self.model_id = str(metadata.get("model_id", "trendmodel-text-v1"))
        self.structured_feature_dim = int(metadata.get("structured_feature_dim", 0) or 0)
        output_names = getattr(model, "output_names", ())
        self.has_effort_bucket_head = "effort_bucket" in output_names
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
        raw = self.model.predict(self._model_inputs([prompt]), verbose=0)

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
        effort_payload = _effort_payload_from_raw(
            raw,
            complexity_probabilities,
            self.metadata,
            effort_bucket_labels=self.effort_bucket_labels,
            needs_clarification=bool(reasons),
        )

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

    def predict_raw_many(
        self,
        prompts: list[str],
        *,
        batch_size: int = 128,
    ) -> list[dict[str, Any]]:
        """Predict with only the neural heads, without structural fusion.

        This is intentionally separate from :meth:`predict`, whose output is
        the downstream router contract and includes TaskTagger evidence.  It
        exists for honest evaluation of the saved TensorFlow artifact.
        """

        if any(not str(prompt).strip() for prompt in prompts):
            raise ValueError("prompts must not contain empty text")
        if not prompts:
            return []

        raw = self.model.predict(
            self._model_inputs([str(prompt) for prompt in prompts]),
            batch_size=batch_size,
            verbose=0,
        )
        return [
            _raw_prediction_payload(
                str(prompt),
                _raw_batch_row(raw, index),
                self.workload_labels,
                self.intent_labels,
                self.complexity_labels,
                self.metadata,
                effort_bucket_labels=self.effort_bucket_labels,
            )
            for index, prompt in enumerate(prompts)
        ]

    def _model_inputs(self, prompts: list[str]) -> Any:
        """Build inputs for both new two-branch and legacy text-only artifacts."""

        tf = require_tensorflow()
        normalized = [normalize_model_text(str(prompt)) for prompt in prompts]
        text_input = tf.constant(normalized)
        if self.structured_feature_dim <= 0:
            return text_input
        features = structured_feature_matrix(prompts)
        return {
            "original_prompt": text_input,
            "structured_features": tf.constant(features, dtype=tf.float32),
        }


def _probabilities(raw: Any, name: str) -> list[float]:
    value = raw[name] if isinstance(raw, dict) else raw
    if hasattr(value, "tolist"):
        value = value.tolist()
    while isinstance(value, list) and len(value) == 1:
        value = value[0]
    return [float(item) for item in value]


def _raw_batch_row(raw: Any, index: int) -> dict[str, Any]:
    """Extract one sample from a batched Keras output mapping."""

    row: dict[str, Any] = {}
    names = [
        "workload_type",
        "intent",
        "complexity",
        "effort_log_minutes",
        "needs_clarification",
    ]
    if isinstance(raw, dict) and "effort_bucket" in raw:
        names.insert(3, "effort_bucket")
    for name in names:
        value = raw[name] if isinstance(raw, dict) else raw
        if hasattr(value, "numpy"):
            value = value.numpy()
        if hasattr(value, "tolist"):
            value = value.tolist()
        row[name] = value[index]
    return row


def _raw_prediction_payload(
    prompt: str,
    raw: Mapping[str, Any],
    workload_labels: list[str],
    intent_labels: list[str],
    complexity_labels: list[str],
    metadata: Mapping[str, Any],
    *,
    effort_bucket_labels: list[str] | None = None,
) -> dict[str, Any]:
    """Build the evaluation contract from neural outputs only."""

    workload_probabilities = _probabilities(raw, "workload_type")
    intent_probabilities = _probabilities(raw, "intent")
    complexity_probabilities = _probabilities(raw, "complexity")
    workload_type = workload_labels[_argmax(workload_probabilities)]
    intent = intent_labels[_argmax(intent_probabilities)]
    complexity = complexity_labels[_argmax(complexity_probabilities)]
    clarification_probability = float(_first_value(raw, "needs_clarification"))

    effort_payload = _effort_payload_from_raw(
        raw,
        complexity_probabilities,
        metadata,
        effort_bucket_labels=effort_bucket_labels or list(EFFORT_BUCKET_ORDER),
    )

    return {
        "original_prompt": prompt,
        "workload_type": workload_type,
        "intent": intent,
        "complexity": complexity,
        "predictions": {
            "workload_type": _classification_payload(
                workload_type,
                workload_probabilities,
                workload_labels,
            ),
            "intent": _classification_payload(
                intent,
                intent_probabilities,
                intent_labels,
            ),
            "complexity": _classification_payload(
                complexity,
                complexity_probabilities,
                complexity_labels,
            ),
            "effort": effort_payload,
        },
        "effort": {
            "bucket": effort_payload["label"],
            "estimated_minutes": effort_payload["estimated_minutes"],
            "range_minutes": effort_payload["range_minutes"],
            "confidence": effort_payload["confidence"],
            "basis": effort_payload["basis"],
        },
        "needs_clarification": clarification_probability >= 0.5,
        "confidence": {
            "workload_type": round(max(workload_probabilities), 4),
            "intent": round(max(intent_probabilities), 4),
            "complexity": round(max(complexity_probabilities), 4),
            "effort": effort_payload["confidence"],
            "overall": round(
                0.30 * max(workload_probabilities)
                + 0.30 * max(intent_probabilities)
                + 0.25 * max(complexity_probabilities)
                + 0.15 * float(effort_payload["confidence"]),
                4,
            ),
        },
        "fusion": {
            "semantic_model": metadata.get("model_id", "trendmodel-text-v1"),
            "structural_teacher": None,
            "policy": "raw neural probabilities without structural evidence fusion",
        },
    }


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
    entropy, margin = _distribution_uncertainty(probabilities)
    return {
        "label": label,
        "probability": round(
            probabilities[labels.index(label)] if label in labels else 0.0,
            4,
        ),
        # These are uncertainty diagnostics for the router, not calibrated
        # probabilities.  Calibration requires an independent gold set.
        "entropy": entropy,
        "margin": margin,
        "candidates": [
            {"label": candidate, "probability": round(float(probability), 4)}
            for candidate, probability in ranked
        ],
    }


def _distribution_uncertainty(probabilities: list[float]) -> tuple[float, float]:
    """Return normalized entropy and top-1/top-2 margin for one head."""

    if not probabilities:
        return 1.0, 0.0
    bounded = [max(0.0, min(1.0, float(value))) for value in probabilities]
    total = sum(bounded)
    if total <= 0.0:
        return 1.0, 0.0
    normalized = [value / total for value in bounded]
    entropy = -sum(value * math.log(value) for value in normalized if value > 0.0)
    if len(normalized) > 1:
        entropy /= math.log(len(normalized))
    ranked = sorted(normalized, reverse=True)
    margin = ranked[0] - (ranked[1] if len(ranked) > 1 else 0.0)
    return round(max(0.0, min(1.0, entropy)), 4), round(max(0.0, min(1.0, margin)), 4)


def _effort_payload_from_raw(
    raw: Mapping[str, Any],
    complexity_probabilities: list[float],
    metadata: Mapping[str, Any],
    *,
    effort_bucket_labels: list[str],
    needs_clarification: bool = False,
) -> dict[str, Any]:
    """Convert effort heads into a coarse, uncertainty-aware router value.

    The bucket head is useful with sparse human annotations because it can
    learn an ordinal routing prior even when only some tasks have observed
    minutes.  A numeric estimate is only treated as observed-effort output
    when the artifact has real active-minute supervision; bucket-only data is
    reported as a range midpoint with an explicitly weaker basis.
    """

    has_bucket_head = "effort_bucket" in raw
    effort_available = _effort_head_is_supervised(metadata)
    if not effort_available and not has_bucket_head:
        return {
            "label": "unknown",
            "probability": 0.0,
            "candidates": [],
            "entropy": 1.0,
            "margin": 0.0,
            "estimated_minutes": None,
            "range_minutes": None,
            "confidence": 0.0,
            "basis": "no_effort_supervision",
        }

    bucket_probabilities: list[float]
    if has_bucket_head:
        bucket_probabilities = _probabilities(raw, "effort_bucket")
        effort_bucket = effort_bucket_labels[_argmax(bucket_probabilities)]
        bucket_probability = max(bucket_probabilities)
        bucket_candidates = _classification_payload(
            effort_bucket,
            bucket_probabilities,
            effort_bucket_labels,
        )
    else:
        bucket_probabilities = []
        effort_log_minutes = float(_first_value(raw, "effort_log_minutes"))
        estimated_from_regression = max(
            1,
            min(round(math.expm1(effort_log_minutes)), 10_080),
        )
        effort_bucket = bucket_from_minutes(estimated_from_regression)
        if effort_bucket not in EFFORT_BUCKET_ORDER:
            effort_bucket = "M"
        bucket_probability = 0.0
        bucket_candidates = {
            "label": effort_bucket,
            "probability": 0.0,
            "candidates": [],
            "entropy": 1.0,
            "margin": 0.0,
        }

    effort_range = _effort_range(effort_bucket)
    has_observed_effort = _has_observed_effort_supervision(metadata)
    if has_observed_effort:
        effort_log_minutes = float(_first_value(raw, "effort_log_minutes"))
        estimated_minutes = max(1, min(round(math.expm1(effort_log_minutes)), 10_080))
        estimated_minutes = max(effort_range[0], min(estimated_minutes, effort_range[1]))
        effort_confidence = min(
            0.95,
            max(
                0.25,
                0.45
                + 0.2 * max(complexity_probabilities)
                + 0.15 * bucket_probability
                - 0.2 * bool(needs_clarification),
            ),
        )
        effort_basis = (
            "trained_observed_effort_and_bucket_heads"
            if has_bucket_head
            else "trained_observed_effort_head"
        )
    else:
        # There is no honest minute estimate from public prompts or bucket-only
        # labels.  A midpoint is retained solely to make range-based routing
        # convenient, and the basis/confidence expose its limitation.
        estimated_minutes = round(EFFORT_BUCKET_MIDPOINTS.get(effort_bucket, 90.0))
        effort_confidence = min(
            0.35 if not has_bucket_head else 0.45,
            max(
                0.15,
                (0.20 + 0.10 * max(complexity_probabilities))
                if not has_bucket_head
                else 0.15 + 0.35 * bucket_probability,
            ),
        )
        effort_basis = "trained_effort_bucket_head" if has_bucket_head else "bucket_seed_only"

    return {
        "label": effort_bucket,
        "probability": round(bucket_probability, 4),
        "candidates": bucket_candidates.get("candidates", []),
        "entropy": bucket_candidates.get("entropy", 1.0),
        "margin": bucket_candidates.get("margin", 0.0),
        "estimated_minutes": estimated_minutes,
        "range_minutes": effort_range,
        "confidence": round(effort_confidence, 4),
        "basis": effort_basis,
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
    value = metadata.get("effort_bucket_supervision_samples")
    if value is None:
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
