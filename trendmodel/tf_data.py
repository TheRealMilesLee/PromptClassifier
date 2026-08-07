"""Training data loading and label preparation for the TensorFlow model.

The loader accepts both the original baseline schema and the newer
``original_prompt`` schema so the first model can be trained from the existing
seed data without duplicating the corpus.
"""

from __future__ import annotations

import json
import math
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping


WORKLOAD_LABEL_ORDER = ("text", "code", "analysis", "operations", "planning", "mixed", "other")
INTENT_LABEL_ORDER = (
    "create",
    "modify",
    "debug",
    "explain",
    "execute",
    "analyze",
    "review",
    "summarize",
    "translate",
    "plan",
    "extract",
    "classify",
    "unknown",
)
COMPLEXITY_LABEL_ORDER = ("low", "medium", "high")
EFFORT_BUCKET_ORDER = ("XS", "S", "M", "L", "XL")
EFFORT_BUCKET_MIDPOINTS = {
    "XS": 10.0,
    "S": 30.0,
    "M": 90.0,
    "L": 300.0,
    "XL": 1_440.0,
}

_MODEL_LINE_BREAK_RE = re.compile(r"[\r\n\u2028\u2029]+")


def normalize_model_text(value: str) -> str:
    """Normalize line separators before character-level tokenization.

    The raw prompt remains unchanged in the dataset/artifact.  This only
    prevents newline characters from becoming literal lines in Keras'
    vocabulary asset, which otherwise makes a saved TextVectorization layer
    impossible to reload when the corpus contains multi-line prompts.
    """

    return _MODEL_LINE_BREAK_RE.sub(" ", str(value))


@dataclass(frozen=True)
class TaskRecord:
    original_prompt: str
    workload_type: str | None
    intent: str | None
    complexity: str | None
    effort_minutes: float | None
    effort_bucket: str | None
    needs_clarification: bool | None = None
    source: str = "unknown"
    observed_effort: bool = False
    split_group: str | None = None
    label_source: str = "unknown"
    label_confidence: float | None = None
    label_confidence_by_field: Mapping[str, float] | None = None

    @property
    def effort_supervised(self) -> bool:
        """Whether an effort target exists, including an explicit bucket label."""

        return self.effort_minutes is not None and self.effort_bucket is not None

    def supervision_weight(self, field: str) -> float:
        """Return a conservative sample weight for one label head.

        Gold/manual labels receive full weight.  Weak labels retain their
        confidence as a weight, with a floor so a low-confidence example does
        not accidentally become an all-zero batch.  Effort is handled
        separately because its target is either an observed minute value or an
        explicitly supplied bucket.
        """

        if field == "effort_log_minutes":
            return 1.0 if self.effort_supervised else 0.0
        if field == "needs_clarification":
            if self.needs_clarification is None:
                return 0.0
        elif getattr(self, field, None) is None:
            return 0.0

        source = self.label_source.casefold()
        if any(marker in source for marker in ("manual", "gold", "observed")):
            return 1.0
        confidence = None
        if self.label_confidence_by_field:
            confidence = self.label_confidence_by_field.get(field)
        if confidence is None:
            confidence = self.label_confidence
        if "pseudo" in source or "weak" in source:
            try:
                return max(0.25, min(1.0, float(confidence if confidence is not None else 0.6)))
            except (TypeError, ValueError):
                return 0.6
        if confidence is not None:
            try:
                return max(0.25, min(1.0, float(confidence)))
            except (TypeError, ValueError):
                pass
        return 1.0

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "TaskRecord":
        prompt = str(value.get("original_prompt", value.get("text", ""))).strip()
        if not prompt:
            raise ValueError("record is missing original_prompt/text")

        workload_type = _first_label(
            value.get("primary_workload_type"),
            value.get("primary_task_type"),
            value.get("workload_types"),
            value.get("work_types"),
            value.get("workload_type"),
            value.get("work_type"),
            default=None,
        )
        intent = _first_label(value.get("primary_intent"), value.get("intent"), default=None)
        complexity = _first_label(value.get("complexity"), default=None)
        raw_effort_bucket = _first_label(
            value.get("effort_bucket"),
            value.get("effort"),
            default=None,
        )
        effort_bucket = raw_effort_bucket.upper() if raw_effort_bucket is not None else None

        raw_minutes = value.get("effort_minutes", value.get("active_minutes"))
        observed_effort = raw_minutes is not None and str(raw_minutes).strip() != ""
        effort_minutes: float | None
        if observed_effort:
            try:
                effort_minutes = max(float(raw_minutes), 1.0)
            except (TypeError, ValueError) as error:
                raise ValueError(f"invalid effort_minutes: {raw_minutes!r}") from error
            if effort_bucket is None or effort_bucket not in EFFORT_BUCKET_MIDPOINTS:
                effort_bucket = bucket_from_minutes(effort_minutes)
        elif effort_bucket is not None:
            if effort_bucket not in EFFORT_BUCKET_MIDPOINTS:
                raise ValueError(f"unsupported effort bucket: {effort_bucket}")
            # Keep legacy bucket-only seed records runnable, but expose that
            # this is not observed wall-clock effort in metadata.
            effort_minutes = EFFORT_BUCKET_MIDPOINTS[effort_bucket]
        else:
            effort_minutes = None

        if complexity is not None and complexity not in COMPLEXITY_LABEL_ORDER:
            raise ValueError(f"unsupported complexity label: {complexity}")

        needs_value: bool | None
        if "needs_clarification" in value:
            needs_value = _as_optional_bool(value.get("needs_clarification"))
        elif "text" in value and "original_prompt" not in value:
            # Preserve the original 48-row seed schema as negative examples;
            # new prompt-only public records must remain unlabelled instead.
            needs_value = False
        else:
            needs_value = None

        raw_confidence = value.get("label_confidence")
        try:
            label_confidence = float(raw_confidence) if raw_confidence is not None else None
        except (TypeError, ValueError) as error:
            raise ValueError(f"invalid label_confidence: {raw_confidence!r}") from error

        raw_field_confidence = value.get("label_confidence_by_field")
        field_confidence: dict[str, float] | None = None
        if isinstance(raw_field_confidence, Mapping):
            field_confidence = {}
            for field, confidence in raw_field_confidence.items():
                try:
                    field_confidence[str(field)] = max(0.0, min(1.0, float(confidence)))
                except (TypeError, ValueError) as error:
                    raise ValueError(
                        f"invalid label_confidence_by_field[{field!r}]: {confidence!r}"
                    ) from error

        return cls(
            original_prompt=prompt,
            workload_type=workload_type,
            intent=intent,
            complexity=complexity,
            effort_minutes=effort_minutes,
            effort_bucket=effort_bucket,
            needs_clarification=needs_value,
            source=str(value.get("source", "seed" if "text" in value else "unknown")),
            observed_effort=observed_effort,
            split_group=_optional_string(value.get("split_group")),
            label_source=str(value.get("label_source", "manual" if any(
                value.get(key) is not None for key in ("workload_type", "work_type", "intent", "complexity")
            ) else "unknown")),
            label_confidence=label_confidence,
            label_confidence_by_field=field_confidence,
        )


@dataclass(frozen=True)
class LabelMaps:
    workload_type: tuple[str, ...]
    intent: tuple[str, ...]
    complexity: tuple[str, ...] = COMPLEXITY_LABEL_ORDER

    def to_dict(self) -> dict[str, list[str]]:
        return {
            "workload_type": list(self.workload_type),
            "intent": list(self.intent),
            "complexity": list(self.complexity),
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "LabelMaps":
        return cls(
            workload_type=tuple(str(item) for item in value["workload_type"]),
            intent=tuple(str(item) for item in value["intent"]),
            complexity=tuple(str(item) for item in value.get("complexity", COMPLEXITY_LABEL_ORDER)),
        )


def load_task_records(path: str | Path) -> list[TaskRecord]:
    records: list[TaskRecord] = []
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").split("\n"), start=1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        try:
            records.append(TaskRecord.from_mapping(json.loads(line)))
        except (json.JSONDecodeError, ValueError) as error:
            raise ValueError(f"invalid task record at {path}:{line_number}: {error}") from error
    if not records:
        raise ValueError(f"no task records found in {path}")
    return records


def build_label_maps(records: Iterable[TaskRecord]) -> LabelMaps:
    records = list(records)
    if not records:
        raise ValueError("at least one record is required")
    return LabelMaps(
        workload_type=_ordered_observed_labels(
            (record.workload_type for record in records),
            WORKLOAD_LABEL_ORDER,
        ),
        intent=_ordered_observed_labels(
            (record.intent for record in records),
            INTENT_LABEL_ORDER,
        ),
    )


def split_records(
    records: Iterable[TaskRecord],
    *,
    validation_ratio: float = 0.2,
    seed: int = 42,
) -> tuple[list[TaskRecord], list[TaskRecord]]:
    records = list(records)
    if len(records) < 4:
        raise ValueError("at least four records are required")
    if not 0 < validation_ratio < 1:
        raise ValueError("validation_ratio must be between 0 and 1")

    records = list(records)
    groups: dict[str, list[TaskRecord]] = {}
    for index, record in enumerate(records):
        group = record.split_group or f"row:{index}"
        groups.setdefault(group, []).append(record)

    # Public sources carry conversation/session groups so near-duplicate turns
    # do not cross the holdout boundary.  Tiny legacy data without groups keeps
    # the old record-level behavior.
    if len(groups) < 2:
        shuffled = list(records)
        random.Random(seed).shuffle(shuffled)
        validation_size = max(1, round(len(shuffled) * validation_ratio))
        return shuffled[validation_size:], shuffled[:validation_size]

    group_keys = list(groups)
    random.Random(seed).shuffle(group_keys)
    validation_size = max(1, round(len(records) * validation_ratio))
    validation_keys: list[str] = []
    validation_count = 0
    for key in group_keys:
        if validation_count >= validation_size:
            break
        # Keep at least one group in training even when the requested ratio is
        # very high and a single group is large.
        if len(validation_keys) == len(group_keys) - 1:
            break
        validation_keys.append(key)
        validation_count += len(groups[key])

    validation_set = set(validation_keys)
    validation = [record for key in validation_keys for record in groups[key]]
    training = [record for key in group_keys if key not in validation_set for record in groups[key]]
    if not training or not validation:
        raise ValueError("unable to create non-empty conversation-level train/validation split")
    return training, validation


def make_dataset(
    records: Iterable[TaskRecord],
    label_maps: LabelMaps,
    *,
    batch_size: int = 16,
    shuffle: bool = False,
):
    """Create a tf.data.Dataset without importing TensorFlow at module load time."""

    try:
        import tensorflow as tf
    except ImportError as error:  # pragma: no cover - depends on local environment
        raise RuntimeError(
            "TensorFlow is not installed; install requirements-tensorflow-macos.txt "
            "or requirements-tensorflow.txt first"
        ) from error

    records = list(records)
    workload_index = {label: index for index, label in enumerate(label_maps.workload_type)}
    intent_index = {label: index for index, label in enumerate(label_maps.intent)}
    complexity_index = {label: index for index, label in enumerate(label_maps.complexity)}

    def indexed_labels(records: list[TaskRecord], field: str, index: Mapping[str, int]) -> tuple[list[int], list[float]]:
        values: list[int] = []
        masks: list[float] = []
        for record in records:
            label = getattr(record, field)
            if label is None:
                values.append(0)
                masks.append(0.0)
                continue
            if label not in index:
                raise ValueError(f"label {label!r} for {field} is not present in label maps")
            values.append(index[label])
            masks.append(record.supervision_weight(field))
        return values, masks

    texts = [normalize_model_text(record.original_prompt) for record in records]
    workload_values, workload_masks = indexed_labels(records, "workload_type", workload_index)
    intent_values, intent_masks = indexed_labels(records, "intent", intent_index)
    complexity_values, complexity_masks = indexed_labels(records, "complexity", complexity_index)
    targets = {
        "workload_type": workload_values,
        "intent": intent_values,
        "complexity": complexity_values,
        "effort_log_minutes": [
            math.log1p(record.effort_minutes) if record.effort_minutes is not None else 0.0
            for record in records
        ],
        "needs_clarification": [
            float(record.needs_clarification) if record.needs_clarification is not None else 0.0
            for record in records
        ],
    }
    sample_weights = {
        "workload_type": workload_masks,
        "intent": intent_masks,
        "complexity": complexity_masks,
        "effort_log_minutes": [record.supervision_weight("effort_log_minutes") for record in records],
        "needs_clarification": [
            record.supervision_weight("needs_clarification") for record in records
        ],
    }
    dataset = tf.data.Dataset.from_tensor_slices((texts, targets, sample_weights))
    if shuffle:
        dataset = dataset.shuffle(max(len(records), 1), seed=42, reshuffle_each_iteration=True)
    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def bucket_from_minutes(minutes: float) -> str:
    if minutes <= 15:
        return "XS"
    if minutes <= 45:
        return "S"
    if minutes <= 120:
        return "M"
    if minutes <= 480:
        return "L"
    return "XL"


def _first_label(*values: Any, default: str | None) -> str | None:
    for value in values:
        if isinstance(value, Mapping):
            value = value.get("label", value.get("name"))
        if isinstance(value, list):
            value = value[0] if value else None
        if isinstance(value, (tuple, set)):
            value = next(iter(value), None)
        if value is not None and str(value).strip():
            return str(value).strip().lower()
    return default


def _ordered_observed_labels(values: Iterable[str | None], preferred: Iterable[str]) -> tuple[str, ...]:
    observed = {str(value).strip() for value in values if value is not None and str(value).strip()}
    # Keep a stable output vocabulary even when a public-only file contains
    # only weak code labels.  Unseen classes receive zero supervision until
    # internal labels are added, but the API contract does not change.
    ordered = list(preferred)
    ordered.extend(sorted(observed - set(ordered)))
    return tuple(ordered)


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _as_optional_bool(value: Any) -> bool | None:
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    return _as_bool(value)


def _optional_string(value: Any) -> str | None:
    if value is None or not str(value).strip():
        return None
    return str(value).strip()
