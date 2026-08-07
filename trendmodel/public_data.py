"""Prepare prompt-only records from public Hugging Face conversations.

The public corpora are useful for learning prompt distribution and weak
signals, but they do not contain task effort ground truth.  This module keeps
the ingestion boundary explicit: only the first user prompt is exported;
assistant messages, model names, IP-related fields, and other source metadata
are deliberately discarded.

The Hugging Face ``datasets`` dependency is optional and imported only by the
CLI.  The normalization helpers stay dependency-free so they can be tested
without downloading any corpus.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any


DATASET_IDS = {
    "wildchat": "allenai/WildChat-4.8M",
    "lmsys": "lmsys/lmsys-chat-1m",
    "arena": "lmarena-ai/arena-human-preference-140k",
}

_SECRET_PATTERNS = (
    re.compile(r"-----BEGIN [^-]+ PRIVATE KEY-----.*?-----END [^-]+ PRIVATE KEY-----", re.I | re.S),
    re.compile(r"\bBearer\s+[A-Za-z0-9._~+/=-]{12,}", re.I),
    re.compile(
        r"\b(?:api[_ -]?key|access[_ -]?token|auth[_ -]?token|password|passwd|secret)"
        r"\s*[:=]\s*['\"]?[^\s,'\"]+",
        re.I,
    ),
    re.compile(r"\b(?:sk-[A-Za-z0-9]{16,}|gh[pousr]_[A-Za-z0-9_]{16,}|AKIA[0-9A-Z]{16})\b"),
)
_EMAIL_PATTERN = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I)
_PHONE_PATTERN = re.compile(
    r"(?<!\w)(?:\+?\d[\d .()\-]{7,}\d)(?!\w)"
)


def normalize_public_row(
    source: str,
    row: Mapping[str, Any],
    *,
    language_filter: set[str] | None = None,
    redact: bool = True,
    row_index: int = 0,
) -> dict[str, Any] | None:
    """Convert one source row to the prompt-only intermediate schema.

    ``None`` means the row has no usable first user prompt or is filtered by
    language.  The function intentionally does not infer effort.
    """

    source = source.strip().lower()
    if source not in DATASET_IDS:
        raise ValueError(f"unsupported public source: {source}")

    conversation, conversation_id = _conversation_for_source(source, row)
    extracted = extract_first_user_prompt(conversation)
    if extracted is None:
        return None
    prompt, turn_index = extracted

    language = normalize_language(row.get("language", row.get("lang")))
    if language_filter and language not in language_filter:
        return None

    prompt = redact_sensitive_text(prompt) if redact else prompt.strip()
    if not prompt:
        return None

    prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    raw_id = conversation_id or _first_non_empty(
        row.get("id"), row.get("conversation_hash"), row.get("evaluation_session_id")
    )
    if not raw_id:
        raw_id = f"row-{row_index}-{prompt_hash[:12]}"
    source_id = f"{source}:{raw_id}:{turn_index}"

    auxiliary_labels: dict[str, Any] = {}
    workload_type: str | None = None
    label_source = "unlabeled"
    label_confidence: float | None = None
    if source == "arena":
        is_code = _as_optional_bool(row.get("is_code"))
        category_tag = _normalize_category_tag(row.get("category_tag"))
        winner = _first_non_empty(row.get("winner"))
        if is_code is not None:
            auxiliary_labels["is_code"] = is_code
        if category_tag:
            auxiliary_labels["category_tag"] = category_tag
        if winner in {"model_a", "model_b", "tie", "both_bad"}:
            auxiliary_labels["preference_label"] = winner
        # This is the only direct workload signal in the three public corpora.
        # ``is_code=false`` is not strong evidence for the text class.
        if is_code is True:
            workload_type = "code"
            label_source = "arena_is_code"
            label_confidence = 0.95

    split_group = f"{source}:{conversation_id or raw_id}"
    return {
        "source": source,
        "source_id": source_id,
        "split_group": split_group,
        "original_prompt": prompt,
        "language": language,
        "workload_type": workload_type,
        "intent": None,
        "complexity": None,
        "effort_minutes": None,
        "effort_bucket": None,
        "needs_clarification": None,
        "effort_observed": False,
        "label_source": label_source,
        "label_confidence": label_confidence,
        "auxiliary_labels": auxiliary_labels,
    }


def extract_first_user_prompt(conversation: Any) -> tuple[str, int] | None:
    """Extract the first non-empty user message from common chat formats."""

    conversation = _decode_json(conversation)
    if isinstance(conversation, Mapping):
        for key in ("messages", "conversation", "turns", "chat"):
            if key in conversation:
                return extract_first_user_prompt(conversation[key])
        # A single message-shaped mapping is also accepted.
        if _message_role(conversation) in {"user", "human"}:
            content = _message_content(conversation)
            if content:
                return content, 0
        return None

    if not isinstance(conversation, (list, tuple)):
        return None
    for index, message in enumerate(conversation):
        if not isinstance(message, Mapping):
            continue
        if _message_role(message) not in {"user", "human"}:
            continue
        content = _message_content(message)
        if content:
            return content, index
    return None


def normalize_language(value: Any) -> str | None:
    """Normalize the common language spellings used by the corpora."""

    value = _first_non_empty(value)
    if not value:
        return None
    lowered = value.casefold().replace("_", "-")
    if lowered in {"en", "eng", "english", "英文"} or lowered.startswith("en-"):
        return "en"
    if lowered in {"zh", "cn", "zh-cn", "zh-tw", "chinese", "中文"}:
        return "zh"
    return lowered


def redact_sensitive_text(text: str) -> str:
    """Remove obvious credentials and direct contact identifiers.

    This is a defense-in-depth filter, not a complete anonymizer.  Public
    corpus terms still require a manual privacy review before production use.
    """

    result = str(text).strip()
    for pattern in _SECRET_PATTERNS:
        result = pattern.sub("[REDACTED]", result)
    result = _EMAIL_PATTERN.sub("[REDACTED_EMAIL]", result)
    result = _PHONE_PATTERN.sub("[REDACTED_PHONE]", result)
    return result


def apply_baseline_pseudo_labels(
    records: Iterable[Mapping[str, Any]],
    model: Any,
) -> list[dict[str, Any]]:
    """Add optional baseline-derived labels while never inventing effort.

    The result is explicitly marked as pseudo-labelled.  This is useful for a
    first encoder pretraining pass, but those labels must not be counted as a
    gold evaluation set.
    """

    labelled: list[dict[str, Any]] = []
    for original in records:
        record = dict(original)
        prediction = model.predict(str(record["original_prompt"]))
        weak_labels: list[str] = []
        for field, prediction_key in (
            ("workload_type", "work_type"),
            ("intent", "intent"),
            ("complexity", "complexity"),
        ):
            if record.get(field) is not None:
                continue
            value = prediction.get(prediction_key, {})
            label = value.get("label") if isinstance(value, Mapping) else None
            if label and label != "unknown":
                record[field] = label
                weak_labels.append(field)
        if weak_labels:
            prior_source = str(record.get("label_source") or "unlabeled")
            record["label_source"] = f"{prior_source}+baseline_pseudo"
            confidences = [
                float(prediction[key]["confidence"])
                for key in ("work_type", "intent", "complexity")
                if isinstance(prediction.get(key), Mapping)
                and prediction[key].get("label") not in {None, "unknown"}
            ]
            record["label_confidence"] = round(min(confidences), 4) if confidences else None
        labelled.append(record)
    return labelled


def apply_tagger_pseudo_labels(
    records: Iterable[Mapping[str, Any]],
    tagger: Any | None = None,
    *,
    min_confidence: float = 0.0,
    keep_low_confidence: bool = True,
) -> list[dict[str, Any]]:
    """Apply the explainable TaskTagger as a weak-label teacher.

    This function only fills classification fields.  In particular, the
    teacher's heuristic effort estimate is intentionally not copied into the
    record: public conversations do not contain observed active minutes.

    ``label_confidence`` and the per-field map are provenance metadata for the
    TensorFlow loader.  They are not claims that the pseudo labels are
    calibrated probabilities.
    """

    if not 0.0 <= min_confidence <= 1.0:
        raise ValueError("min_confidence must be between 0 and 1")
    if tagger is None:
        from .tagger import TaskTagger

        tagger = TaskTagger()

    labelled: list[dict[str, Any]] = []
    for original in records:
        record = dict(original)
        prompt = str(record.get("original_prompt", "")).strip()
        if not prompt:
            labelled.append(record)
            continue
        result = tagger.predict(prompt)
        payload = result.to_dict()
        candidates = {
            "workload_type": payload.get("workload_type", {}),
            "intent": payload.get("intent", {}),
            "complexity": payload.get("complexity", {}),
        }
        pseudo_fields: list[str] = []
        field_confidences: dict[str, float] = {}
        for field, prediction in candidates.items():
            if not isinstance(prediction, Mapping):
                continue
            label = prediction.get("label")
            probability = _bounded_float(prediction.get("probability"))
            if record.get(field) is not None or label in {None, "unknown"}:
                if record.get(field) is not None and field not in field_confidences:
                    field_confidences[field] = _bounded_float(record.get("label_confidence", 1.0))
                continue
            if probability < min_confidence and not keep_low_confidence:
                continue
            record[field] = label
            pseudo_fields.append(field)
            field_confidences[field] = probability

        # Clarification is useful as a routing feature, but it remains marked
        # as teacher-derived and can be masked or down-weighted during training.
        if record.get("needs_clarification") is None and "needs_clarification" in payload:
            record["needs_clarification"] = bool(payload["needs_clarification"])
            pseudo_fields.append("needs_clarification")

        classification_pseudo_fields = [
            field for field in pseudo_fields
            if field in {"workload_type", "intent", "complexity"}
        ]
        if classification_pseudo_fields:
            previous_source = str(record.get("label_source") or "unlabeled")
            suffix = "task_tagger_pseudo"
            if suffix not in previous_source:
                record["label_source"] = f"{previous_source}+{suffix}"
            record["label_source_detail"] = "weak_label_teacher"
            record["teacher_model_version"] = str(getattr(tagger, "model_version", "unknown"))
            record["pseudo_label_fields"] = sorted(set(pseudo_fields))
            record["label_confidence_by_field"] = {
                key: round(value, 4) for key, value in sorted(field_confidences.items())
            }
            used_confidences = [
                field_confidences[field]
                for field in classification_pseudo_fields
                if field in field_confidences
            ]
            record["label_confidence"] = round(min(used_confidences), 4) if used_confidences else None
        elif pseudo_fields:
            # A legacy/gold record may already have all classification labels.
            # Teacher-derived clarification is retained, but must not downgrade
            # the provenance or training weight of those existing labels.
            record["teacher_fields"] = sorted(set(pseudo_fields))
        # Do not touch effort_minutes, effort_bucket, or effort_observed here.
        labelled.append(record)
    return labelled


def _conversation_for_source(source: str, row: Mapping[str, Any]) -> tuple[Any, str | None]:
    if source in {"wildchat", "lmsys"}:
        conversation = row.get("conversation", row.get("messages"))
        conversation_id = _first_non_empty(
            row.get("conversation_hash"),
            row.get("conversation_id"),
            row.get("id"),
        )
        return conversation, conversation_id
    conversation = row.get("conversation_a") or row.get("full_conversation")
    conversation_id = _first_non_empty(
        row.get("evaluation_session_id"),
        row.get("conversation_id"),
        row.get("id"),
    )
    return conversation, conversation_id


def _decode_json(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    stripped = value.strip()
    if not stripped or stripped[0] not in "[{":
        return value
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        return value


def _message_role(message: Mapping[str, Any]) -> str:
    value = _first_non_empty(message.get("role"), message.get("from"), message.get("author"))
    return (value or "").casefold()


def _message_content(message: Mapping[str, Any]) -> str:
    value = message.get("content", message.get("value", message.get("text", "")))
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            if isinstance(item, Mapping):
                part = item.get("text", item.get("content", ""))
            else:
                part = item
            if part is not None and str(part).strip():
                parts.append(str(part).strip())
        return "\n".join(parts).strip()
    return str(value).strip() if value is not None else ""


def _first_non_empty(*values: Any) -> str | None:
    for value in values:
        if isinstance(value, (list, tuple)):
            value = value[0] if value else None
        if value is not None and str(value).strip():
            return str(value).strip()
    return None


def _as_optional_bool(value: Any) -> bool | None:
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    if isinstance(value, bool):
        return value
    return str(value).strip().casefold() in {"1", "true", "yes", "y"}


def _normalize_category_tag(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, Mapping):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, (list, tuple, set)):
        values = value
    else:
        text = str(value).strip()
        if text.startswith("{") or text.startswith("["):
            try:
                parsed = ast.literal_eval(text)
            except (SyntaxError, ValueError):
                parsed = None
            if isinstance(parsed, Mapping):
                return [str(item).strip() for item in parsed if str(item).strip()]
            if isinstance(parsed, (list, tuple, set)):
                values = parsed
            else:
                values = re.split(r"[,;/]", text)
        else:
            values = re.split(r"[,;/]", text)
    return [str(item).strip() for item in values if str(item).strip()]


def _bounded_float(value: Any) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return 0.0


def normalize_languages(value: str | None) -> set[str] | None:
    if value is None or not value.strip():
        return None
    result = {normalize_language(item) for item in value.split(",")}
    result.discard(None)
    return result or None


def _load_huggingface_source(
    source: str,
    *,
    split: str,
    seed: int,
    shuffle_buffer_size: int = 10_000,
) -> Iterable[Mapping[str, Any]]:
    try:
        from datasets import load_dataset
    except ImportError as error:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "需要先安装 requirements-data.txt，才能读取 Hugging Face 数据集"
        ) from error

    dataset = load_dataset(DATASET_IDS[source], split=split, streaming=True)
    # Streaming shuffle avoids selecting a potentially ordered prefix while
    # keeping the initial prototype bounded by the per-source limit.
    if shuffle_buffer_size > 1 and hasattr(dataset, "shuffle"):
        dataset = dataset.shuffle(seed=seed, buffer_size=shuffle_buffer_size)
    return dataset


def prepare_public_corpora(
    *,
    sources: Iterable[str] = ("wildchat", "lmsys", "arena"),
    limits: Mapping[str, int] | None = None,
    languages: set[str] | None = None,
    split: str = "train",
    seed: int = 42,
    redact: bool = True,
    pseudo_label_data: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Download bounded prompt-only samples and return normalized records."""

    selected = [str(source).strip().lower() for source in sources if str(source).strip()]
    invalid = sorted(set(selected) - set(DATASET_IDS))
    if invalid:
        raise ValueError(f"unsupported public sources: {', '.join(invalid)}")

    limits = (
        {"wildchat": 50_000, "lmsys": 50_000, "arena": 20_000}
        if limits is None
        else dict(limits)
    )
    seen_prompt_hashes: set[str] = set()
    records: list[dict[str, Any]] = []
    for source_index, source in enumerate(selected):
        limit = int(limits.get(source, 0))
        if limit == 0:
            continue
        rows = _load_huggingface_source(
            source,
            split=split,
            seed=seed + source_index,
            # A small requested sample should not fetch a 10k-row shuffle
            # buffer before yielding its first record.  Large runs retain a
            # wider reservoir for better source-level randomization.
            # For tiny probes, take the first rows directly.  A reservoir
            # shuffle starts background readers across parquet shards and can
            # make a 10-row health check wait minutes on a slow connection.
            shuffle_buffer_size=(
                0
                if 0 < limit <= 1_000
                else 10_000
                if limit < 0
                else min(10_000, limit * 4)
            ),
        )
        accepted = 0
        for row_index, row in enumerate(rows):
            record = normalize_public_row(
                source,
                row,
                language_filter=languages,
                redact=redact,
                row_index=row_index,
            )
            if record is None:
                continue
            prompt_hash = hashlib.sha256(record["original_prompt"].encode("utf-8")).hexdigest()
            if prompt_hash in seen_prompt_hashes:
                continue
            seen_prompt_hashes.add(prompt_hash)
            records.append(record)
            accepted += 1
            if limit > 0 and accepted >= limit:
                break

    if pseudo_label_data:
        from .model import TaskModel

        teacher = TaskModel.fit_jsonl(pseudo_label_data)
        records = apply_baseline_pseudo_labels(records, teacher)
    return records


def write_jsonl(records: Iterable[Mapping[str, Any]], output: str | Path) -> int:
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(dict(record), ensure_ascii=False, separators=(",", ":")) + "\n")
            count += 1
    return count


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m trendmodel.public_data",
        description="Prepare prompt-only samples from WildChat, LMSYS and Arena.",
    )
    parser.add_argument(
        "--sources",
        default="wildchat,lmsys,arena",
        help="comma-separated sources (wildchat,lmsys,arena)",
    )
    parser.add_argument("--wildchat-limit", type=int, default=50_000)
    parser.add_argument("--lmsys-limit", type=int, default=50_000)
    parser.add_argument("--arena-limit", type=int, default=20_000)
    parser.add_argument(
        "--languages",
        default="en,zh",
        help="comma-separated languages; use an empty value to keep all languages",
    )
    parser.add_argument("--split", default="train")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="data/public/prompts.jsonl")
    parser.add_argument(
        "--pseudo-label-data",
        help="optional seed JSONL for baseline pseudo-labels; effort remains unlabelled",
    )
    parser.add_argument(
        "--no-redact",
        action="store_true",
        help="disable the conservative credential/contact redaction filter",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    sources = [item.strip() for item in args.sources.split(",") if item.strip()]
    limits = {
        "wildchat": args.wildchat_limit,
        "lmsys": args.lmsys_limit,
        "arena": args.arena_limit,
    }
    try:
        records = prepare_public_corpora(
            sources=sources,
            limits=limits,
            languages=normalize_languages(args.languages),
            split=args.split,
            seed=args.seed,
            redact=not args.no_redact,
            pseudo_label_data=args.pseudo_label_data,
        )
    except Exception as error:
        raise SystemExit(f"读取公共数据集失败：{error}") from error

    count = write_jsonl(records, args.output)
    by_source: dict[str, int] = {}
    for record in records:
        by_source[record["source"]] = by_source.get(record["source"], 0) + 1
    print(json.dumps({"output": args.output, "records": count, "by_source": by_source}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
