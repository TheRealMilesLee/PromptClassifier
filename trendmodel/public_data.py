"""Prepare prompt-only records from public Hugging Face datasets.

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
    "xlam": "Salesforce/xlam-function-calling-60k",
    "swebench": "SWE-bench/SWE-bench",
    "swe_trajectories": "nebius/SWE-agent-trajectories",
    "massive": "AmazonScience/massive",
    "oasst1": "OpenAssistant/oasst1",
    "code_feedback": "m-a-p/Code-Feedback",
    "coig": "BAAI/COIG",
    "dolly": "databricks/databricks-dolly-15k",
    "mtop": "tasksource/mtop",
    "helpful_instructions": "HuggingFaceH4/helpful-instructions",
}

# MASSIVE has multiple configurations.  The local parquet workflow does not
# need this value, but it makes the optional streaming path deterministic when
# someone explicitly requests the remote source.
DATASET_CONFIGS = {"massive": "all"}

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

    extracted, conversation_id = _extract_prompt_for_source(source, row)
    if extracted is None:
        return None
    prompt, turn_index = extracted

    language = normalize_language(
        row.get("language", row.get("lang", row.get("locale")))
    )
    if language is None and source in {
        "xlam",
        "swebench",
        "swe_trajectories",
        "code_feedback",
        "dolly",
        "helpful_instructions",
    }:
        language = "en"
    elif language is None and source == "coig":
        language = "zh"
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
    intent: str | None = None
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

    elif source == "xlam":
        tools = _as_sequence(row.get("tools"))
        answers = _as_sequence(row.get("answers"))
        auxiliary_labels.update(
            {
                "requires_tools": True,
                "tool_count": len(tools),
                "tool_call_count": len(answers),
                "multi_step": len(answers) > 1,
                "label_scope": "tool_use_only",
            }
        )
        # APIGen records are explicitly verified function calls.  They are
        # strong weak supervision for tool/execute behavior, but not gold
        # effort labels and not proof that every query is an operations task.
        workload_type = "operations"
        intent = "execute"
        label_source = "xlam_tool_use_weak"
        label_confidence = 0.85

    elif source == "swebench":
        patch = str(row.get("patch", "") or "")
        test_patch = str(row.get("test_patch", "") or "")
        auxiliary_labels.update(
            {
                "requires_files": True,
                "has_patch": bool(patch.strip()),
                "has_test_patch": bool(test_patch.strip()),
                "patch_lines": _patch_line_count(patch),
                "test_patch_lines": _patch_line_count(test_patch),
                "repository": _first_non_empty(row.get("repo")),
            }
        )
        # An issue/PR pair is high-confidence evidence for a code change task.
        # Complexity remains unset because patch size is only a proxy, not
        # observed human effort.
        workload_type = "code"
        intent = "modify"
        label_source = "swebench_issue_weak"
        label_confidence = 0.9

    elif source == "swe_trajectories":
        trajectory = _as_sequence(row.get("trajectory"))
        auxiliary_labels.update(
            {
                "requires_tools": True,
                "trajectory_steps": len(trajectory),
                "trajectory_target": _as_optional_bool(row.get("target")),
                "exit_status": _first_non_empty(row.get("exit_status")),
                "has_generated_patch": bool(
                    str(row.get("generated_patch", "") or "").strip()
                ),
            }
        )
        workload_type = "code"
        intent = "modify"
        label_source = "swe_agent_trajectory_weak"
        label_confidence = 0.8

    elif source == "massive":
        scenario = _first_non_empty(row.get("scenario"))
        massive_intent = _first_non_empty(row.get("intent"), row.get("class"))
        if scenario:
            auxiliary_labels["scenario"] = scenario
        if massive_intent:
            auxiliary_labels["massive_intent"] = massive_intent
        auxiliary_labels["locale"] = _first_non_empty(row.get("locale"))
        # MASSIVE consists of assistant-directed, service-oriented utterances.
        # Preserve the original fine-grained intent and use a conservative
        # mapping for the generic router intent.  A weather/calendar query is
        # not the same workload as an explicit "set alarm" command.
        workload_type = "operations"
        intent = _massive_generic_intent(massive_intent) or "execute"
        label_source = "massive_service_intent_weak"
        label_confidence = 0.72 if massive_intent else 0.55

    elif source == "oasst1":
        auxiliary_labels["oasst_role"] = _first_non_empty(row.get("role"))

    elif source == "code_feedback":
        messages = _as_sequence(row.get("messages"))
        auxiliary_labels.update(
            {
                "message_count": len(messages),
                "label_scope": "code_generation_only",
            }
        )
        workload_type = "code"
        intent = "create"
        label_source = "code_feedback_create_weak"
        label_confidence = 0.88

    elif source == "coig":
        # COIG is valuable Chinese instruction coverage, but its general
        # instruction rows do not provide a trustworthy generic workload or
        # intent label.  Keep it as unlabeled representation data; the
        # optional TaskTagger pass may add explicitly weak pseudo-labels.
        auxiliary_labels["label_scope"] = "chinese_instruction_unlabeled"

    elif source == "dolly":
        category = _first_non_empty(row.get("category"))
        if category:
            auxiliary_labels["dolly_category"] = category
        workload_type = "text"
        intent = _dolly_generic_intent(category)
        label_source = "dolly_category_weak"
        label_confidence = 0.82 if intent else 0.55

    elif source == "mtop":
        mtop_intent = _first_non_empty(row.get("intent"))
        domain = _first_non_empty(row.get("domain"))
        if mtop_intent:
            auxiliary_labels["mtop_intent"] = mtop_intent
        if domain:
            auxiliary_labels["mtop_domain"] = domain
        # MTOP is a task-oriented semantic parsing corpus.  Its intent names
        # are useful weak supervision for the router's explain-vs-execute
        # boundary, but the domain is much narrower than general chat.
        workload_type = "operations"
        intent = _mtop_generic_intent(mtop_intent)
        label_source = "mtop_intent_weak"
        label_confidence = 0.76 if intent else 0.55

    elif source == "helpful_instructions":
        # Curated instructions add expression coverage, but do not expose
        # trustworthy workload, intent, or effort labels.
        auxiliary_labels["label_scope"] = "english_instruction_unlabeled"

    split_group = f"{source}:{conversation_id or raw_id}"
    return {
        "source": source,
        "source_id": source_id,
        "split_group": split_group,
        "original_prompt": prompt,
        "language": language,
        "workload_type": workload_type,
        "intent": intent,
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


def _extract_prompt_for_source(
    source: str,
    row: Mapping[str, Any],
) -> tuple[tuple[str, int] | None, str | None]:
    """Extract one task prompt and its source grouping key.

    The public sources intentionally converge to one prompt-only record, but
    their schemas are not interchangeable: some are conversations, some are
    issue reports, and some expose a direct ``query``/``utt`` field.
    """

    if source in {"wildchat", "lmsys", "arena"}:
        conversation, conversation_id = _conversation_for_source(source, row)
        return extract_first_user_prompt(conversation), conversation_id

    conversation_id = _first_non_empty(
        row.get("conversation_id"),
        row.get("conversation_hash"),
        row.get("evaluation_session_id"),
        row.get("instance_id"),
        row.get("id"),
    )
    if source == "xlam":
        query = _first_non_empty(
            row.get("query"), row.get("prompt"), row.get("instruction")
        )
        return ((query, 0) if query else None), conversation_id
    if source == "swebench":
        statement = _first_non_empty(
            row.get("problem_statement"), row.get("issue"), row.get("prompt")
        )
        return ((statement, 0) if statement else None), conversation_id
    if source == "massive":
        utterance = _first_non_empty(
            row.get("utt"), row.get("utterance"), row.get("text")
        )
        return ((utterance, 0) if utterance else None), conversation_id
    if source == "oasst1":
        role = _message_role(row)
        if role not in {"prompter", "user", "human"}:
            return None, conversation_id
        text = _first_non_empty(row.get("text"), row.get("content"))
        return ((text, 0) if text else None), conversation_id
    if source == "swe_trajectories":
        extracted = extract_first_user_prompt(row.get("trajectory"))
        if extracted is not None:
            return extracted, conversation_id
        prompt = _first_non_empty(row.get("problem_statement"), row.get("prompt"))
        return ((prompt, 0) if prompt else None), conversation_id
    if source == "code_feedback":
        extracted = extract_first_user_prompt(row.get("messages", row.get("conversation")))
        if extracted is not None:
            return extracted, conversation_id
        prompt = _first_non_empty(row.get("instruction"), row.get("prompt"), row.get("text"))
        return ((prompt, 0) if prompt else None), conversation_id
    if source == "coig":
        prompt = _first_non_empty(row.get("instruction"), row.get("prompt"), row.get("text"))
        return ((prompt, 0) if prompt else None), conversation_id
    if source == "dolly":
        prompt = _first_non_empty(row.get("instruction"), row.get("prompt"), row.get("text"))
        return ((prompt, 0) if prompt else None), conversation_id
    if source == "mtop":
        prompt = _first_non_empty(row.get("question"), row.get("utterance"), row.get("text"))
        return ((prompt, 0) if prompt else None), conversation_id
    if source == "helpful_instructions":
        prompt = _first_non_empty(
            row.get("instruction"), row.get("prompt"), row.get("text")
        )
        return ((prompt, 0) if prompt else None), conversation_id
    raise ValueError(f"unsupported public source: {source}")


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


def _as_sequence(value: Any) -> list[Any]:
    """Decode a list-like JSON field without trusting arbitrary objects."""

    decoded = _decode_json(value)
    if isinstance(decoded, (list, tuple)):
        return list(decoded)
    return []


def _patch_line_count(patch: str) -> int:
    return sum(
        1
        for line in str(patch).splitlines()
        if line.startswith(("+", "-")) and not line.startswith(("+++", "---"))
    )


def _dolly_generic_intent(category: str | None) -> str | None:
    """Map Dolly's task categories to the router's coarser intent vocabulary."""

    normalized = str(category or "").strip().casefold()
    return {
        "brainstorming": "create",
        "classification": "classify",
        "closed_qa": "explain",
        "creative_writing": "create",
        "general_qa": "explain",
        "information_extraction": "extract",
        "open_qa": "explain",
        "summarization": "summarize",
    }.get(normalized)


def _massive_generic_intent(intent: str | None) -> str | None:
    """Map MASSIVE service intents without treating every utterance as execute."""

    normalized = str(intent or "").strip().casefold().replace("-", "_")
    if not normalized:
        return None
    if "translat" in normalized:
        return "translate"
    if any(
        marker in normalized
        for marker in (
            "cancel",
            "set_",
            "create_",
            "add_",
            "remove_",
            "delete_",
            "activate",
            "deactivate",
            "turn_on",
            "turn_off",
            "play_",
            "stop_",
            "send_",
            "call_",
            "open_",
            "close_",
            "schedule_",
        )
    ):
        return "execute"
    if any(marker in normalized for marker in ("query", "where", "when", "what", "which", "info")):
        return "explain"
    return None


def _mtop_generic_intent(intent: str | None) -> str | None:
    """Map MTOP semantic parses to the router's coarse generic intent."""

    normalized = str(intent or "").strip().casefold()
    if not normalized:
        return None
    normalized = normalized.removeprefix("in:").replace("-", "_")
    if any(
        marker in normalized
        for marker in (
            "create",
            "send",
            "set_",
            "cancel",
            "delete",
            "add_",
            "remove",
            "turn_on",
            "turn_off",
            "play_",
            "stop_",
            "book",
            "call",
            "schedule",
            "make_",
        )
    ):
        return "execute"
    if any(
        marker in normalized
        for marker in (
            "get_",
            "check_",
            "query",
            "search",
            "find",
            "show",
            "list",
            "where",
            "when",
            "what",
        )
    ):
        return "explain"
    return None


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

    config = DATASET_CONFIGS.get(source)
    if config:
        dataset = load_dataset(DATASET_IDS[source], config, split=split, streaming=True)
    else:
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
        {
            "wildchat": 50_000,
            "lmsys": 50_000,
            "arena": 20_000,
            "xlam": 0,
            "swebench": 0,
            "swe_trajectories": 0,
            "massive": 0,
            "oasst1": 0,
            "code_feedback": 0,
            "coig": 0,
            "dolly": 0,
            "mtop": 0,
            "helpful_instructions": 0,
        }
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
        description="Prepare prompt-only samples from local or remote public datasets.",
    )
    parser.add_argument(
        "--sources",
        default="wildchat,lmsys,arena",
        help=(
            "comma-separated sources; supported: wildchat,lmsys,arena,xlam,"
            "swebench,swe_trajectories,massive,oasst1,code_feedback,coig,dolly"
            ",mtop"
        ),
    )
    parser.add_argument("--wildchat-limit", type=int, default=50_000)
    parser.add_argument("--lmsys-limit", type=int, default=50_000)
    parser.add_argument("--arena-limit", type=int, default=20_000)
    parser.add_argument("--xlam-limit", type=int, default=0)
    parser.add_argument("--swebench-limit", type=int, default=0)
    parser.add_argument("--swe-trajectories-limit", type=int, default=0)
    parser.add_argument("--massive-limit", type=int, default=0)
    parser.add_argument("--oasst1-limit", type=int, default=0)
    parser.add_argument("--code-feedback-limit", type=int, default=0)
    parser.add_argument("--coig-limit", type=int, default=0)
    parser.add_argument("--dolly-limit", type=int, default=0)
    parser.add_argument("--mtop-limit", type=int, default=0)
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
        "xlam": args.xlam_limit,
        "swebench": args.swebench_limit,
        "swe_trajectories": args.swe_trajectories_limit,
        "massive": args.massive_limit,
        "oasst1": args.oasst1_limit,
        "code_feedback": args.code_feedback_limit,
        "coig": args.coig_limit,
        "dolly": args.dolly_limit,
        "mtop": args.mtop_limit,
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
