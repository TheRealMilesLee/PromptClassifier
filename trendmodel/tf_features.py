"""Fixed-width structured features for the TensorFlow prompt model.

The text encoder handles wording and generalization.  This branch exposes
high-confidence shape/evidence signals explicitly so the small local model
does not have to rediscover obvious facts such as a fenced code block, a
shell command, or a C++ marker from a short character sequence.

These features are deterministic and are also computed at inference time.
They are not a replacement for the sentence encoder or the downstream
evidence-fusion policy.
"""

from __future__ import annotations

import math
import re
from typing import Iterable

from .evidence import extract_evidence


_LANGUAGES = (
    "cpp",
    "python",
    "typescript",
    "javascript",
    "java",
    "go",
    "rust",
    "swift",
    "sql",
    "shell",
)
_INTENTS = (
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
_DOMAINS = (
    "conversation",
    "creative",
    "programming",
    "mathematics",
    "shell",
    "networking",
    "document",
    "data_analysis",
    "operations",
    "planning",
    "other",
)
_OPERATIONS = (
    "multithreading",
    "cartesian_product",
    "routing",
    "dns",
    "http_api",
    "database",
    "kubernetes",
    "deployment",
    "testing",
)


STRUCTURED_FEATURE_NAMES = (
    "prompt_length_log",
    "line_count_log",
    "cjk_ratio",
    "latin_ratio",
    "digit_ratio",
    "has_code_block",
    "has_file_paths",
    "has_shell_commands",
    "has_math_symbols",
    "has_urls",
    "requires_files",
    "requires_tools",
    "high_risk",
    *(f"language:{label}" for label in _LANGUAGES),
    *(f"intent_hint:{label}" for label in _INTENTS),
    *(f"domain_hint:{label}" for label in _DOMAINS),
    *(f"operation:{label}" for label in _OPERATIONS),
)


def structured_feature_vector(text: str) -> list[float]:
    """Return a stable, bounded feature vector for one prompt."""

    original = str(text)
    compact = re.sub(r"\s+", "", original)
    total = max(len(compact), 1)
    cjk_count = len(re.findall(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]", compact))
    latin_count = len(re.findall(r"[A-Za-z]", compact))
    digit_count = len(re.findall(r"\d", compact))
    evidence = extract_evidence(original)
    constraints = evidence.constraints

    values = [
        _bounded_log(len(compact), 10.0),
        _bounded_log(max(1, original.count("\n") + 1), 5.0),
        cjk_count / total,
        latin_count / total,
        digit_count / total,
        float(constraints.has_code_block),
        float(constraints.has_file_paths),
        float(constraints.has_shell_commands),
        float(constraints.has_math_symbols),
        float(constraints.has_urls),
        float(constraints.requires_files),
        float(constraints.requires_tools),
        float(evidence.high_risk),
    ]
    language = evidence.language
    values.extend(float(language == label) for label in _LANGUAGES)
    values.extend(float(evidence.intent_scores.get(label, 0.0)) for label in _INTENTS)
    values.extend(float(evidence.domain_scores.get(label, 0.0)) for label in _DOMAINS)
    values.extend(
        float(label in evidence.operations)
        for label in _OPERATIONS
    )
    return [max(0.0, min(1.0, float(value))) for value in values]


def structured_feature_matrix(texts: Iterable[str]) -> list[list[float]]:
    """Vectorize a batch using the same deterministic inference path."""

    return [structured_feature_vector(text) for text in texts]


def _bounded_log(value: int, scale: float) -> float:
    return min(1.0, math.log1p(max(0, value)) / scale)
