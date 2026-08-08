"""Dependency-free prompt cleaning and dataset quality reporting.

The public Hugging Face corpora and internal task logs have different schemas,
but they need the same safety boundary before they reach a trainer:

* only a normalized ``original_prompt`` is used as model input;
* obvious credentials and direct contact identifiers are redacted;
* empty, pathological, and over-sized prompts are rejected with a reason;
* duplicate prompts are removed after normalization;
* source/session provenance is retained for group-aware splitting and audits.

This module intentionally does not infer task labels or effort.  Labeling is a
separate step so that a pseudo-label can never be mistaken for observed work.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping


@dataclass(frozen=True)
class CleaningConfig:
    """Quality and privacy policy for one cleaning pass."""

    min_non_whitespace_chars: int = 3
    max_chars: int = 16_000
    max_control_char_ratio: float = 0.08
    redact: bool = True
    deduplicate: bool = True
    allowed_languages: frozenset[str] | None = None
    drop_unknown_language: bool = False


@dataclass
class CleaningReport:
    """Counters that make data loss and filtering decisions inspectable."""

    input_rows: int = 0
    accepted_rows: int = 0
    rejected_rows: int = 0
    redacted_rows: int = 0
    duplicate_rows: int = 0
    label_conflict_rows: int = 0
    label_conflict_groups: int = 0
    reject_reasons: Counter[str] = field(default_factory=Counter)
    conflict_fields: Counter[str] = field(default_factory=Counter)
    conflict_examples: list[dict[str, Any]] = field(default_factory=list)
    _conflict_hashes: set[str] = field(default_factory=set, repr=False)
    by_source: dict[str, Counter[str]] = field(default_factory=lambda: defaultdict(Counter))

    def reject(self, source: str, reason: str) -> None:
        self.rejected_rows += 1
        self.reject_reasons[reason] += 1
        self.by_source[source][f"rejected_{reason}"] += 1

    def accept(self, source: str, *, redacted: bool) -> None:
        self.accepted_rows += 1
        self.by_source[source]["accepted"] += 1
        if redacted:
            self.redacted_rows += 1
            self.by_source[source]["redacted"] += 1

    def note_label_conflict(
        self,
        *,
        source: str,
        prompt_hash: str,
        existing_source: str,
        fields: list[str],
    ) -> None:
        """Record a duplicate whose non-null labels disagree with the winner."""

        self.label_conflict_rows += 1
        if prompt_hash not in self._conflict_hashes:
            self._conflict_hashes.add(prompt_hash)
            self.label_conflict_groups += 1
        for field_name in fields:
            self.conflict_fields[field_name] += 1
        self.by_source[source]["label_conflict"] += 1
        if len(self.conflict_examples) < 20:
            self.conflict_examples.append(
                {
                    "prompt_hash": prompt_hash,
                    "existing_source": existing_source,
                    "duplicate_source": source,
                    "fields": sorted(fields),
                }
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "input_rows": self.input_rows,
            "accepted_rows": self.accepted_rows,
            "rejected_rows": self.rejected_rows,
            "redacted_rows": self.redacted_rows,
            "duplicate_rows": self.duplicate_rows,
            "label_conflict_rows": self.label_conflict_rows,
            "label_conflict_groups": self.label_conflict_groups,
            "reject_reasons": dict(sorted(self.reject_reasons.items())),
            "label_conflicts": {
                "fields": dict(sorted(self.conflict_fields.items())),
                "examples": list(self.conflict_examples),
            },
            "by_source": {
                source: dict(sorted(counters.items()))
                for source, counters in sorted(self.by_source.items())
            },
        }


def normalize_prompt_text(value: Any) -> str:
    """Apply conservative Unicode and whitespace normalization.

    Newlines are kept because a prompt may contain a code block or a list.  We
    only collapse horizontal whitespace and repeated blank lines, so the
    classifier still sees a useful approximation of the original structure.
    """

    text = unicodedata.normalize("NFKC", str(value))
    # JSONL readers treat only LF/CRLF as record separators.  Normalize the
    # Unicode line/paragraph separators that appear in scraped chat text so
    # they remain content inside one JSON object rather than looking like a
    # second physical record to ``str.splitlines()`` callers.
    text = text.replace("\u2028", "\n").replace("\u2029", "\n")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    cleaned_chars: list[str] = []
    for char in text:
        category = unicodedata.category(char)
        if char in {"\n", "\t"} or not category.startswith("C"):
            cleaned_chars.append(char)
        else:
            cleaned_chars.append(" ")
    text = "".join(cleaned_chars)
    lines = [re.sub(r"[ \t]+", " ", line).rstrip() for line in text.split("\n")]
    text = re.sub(r"\n{3,}", "\n\n", "\n".join(lines))
    return text.strip()


def clean_records(
    records: Iterable[Mapping[str, Any]],
    *,
    config: CleaningConfig = CleaningConfig(),
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Clean records and return ``(accepted_records, report)``.

    The function is deliberately deterministic.  The first occurrence of a
    normalized prompt wins, which makes repeated preparation runs stable and
    preserves the earliest source provenance in merged inputs.
    """

    report = CleaningReport()
    accepted: list[dict[str, Any]] = []
    seen_records: dict[str, dict[str, Any]] = {}

    for row in records:
        report.input_rows += 1
        if not isinstance(row, Mapping):
            report.reject("unknown", "invalid_record")
            continue
        source = _source_name(row)
        report.by_source[source]["input"] += 1

        raw_prompt = row.get("original_prompt", row.get("text"))
        if raw_prompt is None:
            report.reject(source, "empty")
            continue
        prompt = normalize_prompt_text(raw_prompt)
        if not prompt:
            report.reject(source, "empty")
            continue

        control_ratio = _control_character_ratio(str(raw_prompt))
        if control_ratio > config.max_control_char_ratio:
            report.reject(source, "control_characters")
            continue

        non_whitespace_length = len(re.sub(r"\s", "", prompt))
        if non_whitespace_length < config.min_non_whitespace_chars:
            report.reject(source, "too_short")
            continue
        if len(prompt) > config.max_chars:
            report.reject(source, "too_long")
            continue

        if config.allowed_languages:
            language = _normalize_language(row.get("language"))
            if language is None and config.drop_unknown_language:
                report.reject(source, "unknown_language")
                continue
            if language is not None and language not in config.allowed_languages:
                report.reject(source, "language_filtered")
                continue

        redacted_prompt = _redact(prompt) if config.redact else prompt
        was_redacted = redacted_prompt != prompt
        canonical_prompt = re.sub(r"\s+", " ", redacted_prompt).strip()
        prompt_hash = hashlib.sha256(canonical_prompt.encode("utf-8")).hexdigest()
        record = dict(row)
        record.pop("text", None)
        _canonicalize_legacy_fields(record)
        record.setdefault("source", source)
        if config.deduplicate and prompt_hash in seen_records:
            existing_record = seen_records[prompt_hash]
            conflicting_fields = _label_conflict_fields(existing_record, record)
            if conflicting_fields:
                report.note_label_conflict(
                    source=source,
                    prompt_hash=prompt_hash,
                    existing_source=_source_name(existing_record),
                    fields=conflicting_fields,
                )
            report.duplicate_rows += 1
            report.reject(source, "duplicate")
            continue
        seen_records[prompt_hash] = record

        record["original_prompt"] = redacted_prompt
        record["prompt_hash"] = prompt_hash
        record["cleaned"] = True
        record["cleaning_version"] = "prompt-clean-v1"
        if record.get("language") is not None:
            record["language"] = _normalize_language(record.get("language"))
        record.setdefault("source", source)
        record.setdefault("split_group", f"{source}:row-{report.accepted_rows}")
        report.accept(source, redacted=was_redacted)
        accepted.append(record)

    return accepted, report.to_dict()


def read_jsonl(paths: Iterable[str | Path]) -> list[dict[str, Any]]:
    """Read one or more JSONL files without importing optional dependencies."""

    records: list[dict[str, Any]] = []
    for path_value in paths:
        path = Path(path_value)
        if not path.exists():
            raise FileNotFoundError(path)
        for line_number, line in enumerate(path.read_text(encoding="utf-8").split("\n"), start=1):
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSON at {path}:{line_number}: {error}") from error
            if not isinstance(value, Mapping):
                raise ValueError(f"JSONL row at {path}:{line_number} must be an object")
            records.append(dict(value))
    if not records:
        raise ValueError("no JSONL records found")
    return records


def write_jsonl(records: Iterable[Mapping[str, Any]], path_value: str | Path) -> int:
    """Write JSON-compatible records and return the number of rows."""

    path = Path(path_value)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(dict(record), ensure_ascii=False, separators=(",", ":")) + "\n")
            count += 1
    return count


def _source_name(record: Mapping[str, Any]) -> str:
    value = record.get("source", "unknown")
    text = str(value).strip().lower()
    return text or "unknown"


def _canonicalize_legacy_fields(record: dict[str, Any]) -> None:
    """Normalize the original 48-row seed schema without losing its labels."""

    if record.get("workload_type") is None and record.get("work_type") is not None:
        record["workload_type"] = _label_value(record.get("work_type"))
    if record.get("effort_bucket") is None and record.get("effort") is not None:
        record["effort_bucket"] = _label_value(record.get("effort"))
    if "label_source" not in record:
        has_label = any(
            record.get(key) is not None
            for key in ("workload_type", "work_type", "intent", "complexity", "effort_bucket", "effort")
        )
        record["label_source"] = "seed" if has_label else "unlabeled"


def _label_conflict_fields(
    existing: Mapping[str, Any],
    duplicate: Mapping[str, Any],
) -> list[str]:
    """Find fields with two present, normalized but disagreeing labels."""

    fields = (
        "workload_type",
        "intent",
        "complexity",
        "effort_bucket",
        "needs_clarification",
    )
    conflicts: list[str] = []
    for field_name in fields:
        left = _comparable_label(existing.get(field_name))
        right = _comparable_label(duplicate.get(field_name))
        if left is not None and right is not None and left != right:
            conflicts.append(field_name)
    return conflicts


def _comparable_label(value: Any) -> str | None:
    """Normalize scalar/list/mapping labels for duplicate conflict checks."""

    normalized = _label_value(value)
    if normalized is None or not str(normalized).strip():
        return None
    return str(normalized).strip().casefold()


def _label_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return value.get("label", value.get("name"))
    if isinstance(value, (list, tuple, set)):
        return next(iter(value), None)
    return value


def _control_character_ratio(value: str) -> float:
    if not value:
        return 0.0
    control_count = sum(
        1
        for char in value
        if unicodedata.category(char).startswith("C") and char not in {"\n", "\r", "\t"}
    )
    return control_count / len(value)


def _normalize_language(value: Any) -> str | None:
    if value is None or not str(value).strip():
        return None
    lowered = str(value).strip().casefold().replace("_", "-")
    if lowered in {"en", "eng", "english", "英文"} or lowered.startswith("en-"):
        return "en"
    if lowered in {"zh", "cn", "zh-cn", "zh-tw", "chinese", "中文"}:
        return "zh"
    return lowered


def _redact(value: str) -> str:
    # Keep the source-specific patterns in one place for backwards compatibility
    # with public_data.py, while making the cleaning module usable by itself.
    from .public_data import redact_sensitive_text

    return redact_sensitive_text(value)
