"""Structured task evidence tree used between text and Task IR.

``TaskAST`` is intentionally not a programming-language syntax tree.  It is
the normalized, router-oriented representation of facts and candidate scores
that can be extracted from a prompt before the model heads make a decision.
The class gives the extractor, TensorFlow feature branch and prototype tagger
one explicit contract instead of passing a collection of loosely related
fields around.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, ClassVar, Mapping

from .task_ir import TaskConstraints, TaskEvidence


@dataclass(frozen=True)
class TaskAST:
    """Normalized AST-like evidence for one task request.

    The score mappings contain bounded candidate evidence, not final labels.
    A request may therefore have multiple non-zero intent/domain/artifact
    candidates.  Final selection remains the responsibility of the semantic
    model and safety/clarification rules.  ``TaskEvidence`` items retain
    provenance so a downstream router can distinguish active, negated and
    quoted signals.
    """

    original_prompt: str
    constraints: TaskConstraints
    language: str | None
    operations: tuple[str, ...]
    # Keep ``signals`` as the stored field for source compatibility with the
    # former StructuredEvidence dataclass.  ``evidence`` below is the clearer
    # AST-oriented name used by new callers.
    signals: tuple[TaskEvidence, ...]
    intent_scores: Mapping[str, float]
    domain_scores: Mapping[str, float]
    artifact_scores: Mapping[str, float]
    high_risk: bool
    risk_reasons: tuple[str, ...]

    schema_version: ClassVar[str] = "task-ast-v1"

    def __post_init__(self) -> None:
        if not isinstance(self.original_prompt, str) or not self.original_prompt.strip():
            raise ValueError("original_prompt must not be empty")
        for field_name, scores in self._score_fields():
            normalized_scores: dict[str, float] = {}
            for label, value in scores.items():
                try:
                    numeric = float(value)
                except (TypeError, ValueError) as error:
                    raise ValueError(
                        f"{field_name}[{label!r}] must be numeric"
                    ) from error
                if not 0.0 <= numeric <= 1.0:
                    raise ValueError(
                        f"{field_name}[{label!r}] must be between 0 and 1"
                    )
                normalized_scores[str(label)] = numeric
            object.__setattr__(self, field_name, MappingProxyType(normalized_scores))
        object.__setattr__(self, "operations", tuple(self.operations))
        object.__setattr__(self, "signals", tuple(self.signals))
        object.__setattr__(self, "risk_reasons", tuple(self.risk_reasons))
        if any(not str(operation).strip() for operation in self.operations):
            raise ValueError("operations must contain non-empty labels")
        if any(not str(reason).strip() for reason in self.risk_reasons):
            raise ValueError("risk_reasons must contain non-empty labels")

    @property
    def evidence(self) -> tuple[TaskEvidence, ...]:
        """AST-oriented name for the observable evidence sequence."""

        return self.signals

    @property
    def action_scores(self) -> Mapping[str, float]:
        """Expose intent candidates under the AST-oriented ``action`` name."""

        return self.intent_scores

    def top_candidates(self, field: str, *, limit: int = 3) -> tuple[tuple[str, float], ...]:
        """Return deterministic top candidates from one score mapping."""

        if limit < 1:
            raise ValueError("limit must be positive")
        fields = {
            "intent": self.intent_scores,
            "action": self.intent_scores,
            "domain": self.domain_scores,
            "artifact": self.artifact_scores,
        }
        try:
            scores = fields[field]
        except KeyError as error:
            raise ValueError(
                "field must be one of: intent, action, domain, artifact"
            ) from error
        return tuple(
            (str(label), float(value))
            for label, value in sorted(
                scores.items(),
                key=lambda item: (-float(item[1]), str(item[0])),
            )[:limit]
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible representation for audit/debug output."""

        return {
            "schema_version": self.schema_version,
            "root": {
                "type": "task_request",
                "original_prompt": self.original_prompt,
            },
            "action_candidates": _sorted_scores(self.intent_scores),
            "domain_candidates": _sorted_scores(self.domain_scores),
            "artifact_candidates": _sorted_scores(self.artifact_scores),
            "language": self.language,
            "operations": list(self.operations),
            "constraints": self.constraints.to_dict(),
            "risk": {
                "high": self.high_risk,
                "reasons": list(self.risk_reasons),
            },
            "evidence": [item.to_dict() for item in self.evidence],
        }

    as_dict = to_dict

    def _score_fields(self) -> tuple[tuple[str, Mapping[str, float]], ...]:
        return (
            ("intent_scores", self.intent_scores),
            ("domain_scores", self.domain_scores),
            ("artifact_scores", self.artifact_scores),
        )


def _sorted_scores(scores: Mapping[str, float]) -> list[dict[str, float | str]]:
    return [
        {"label": str(label), "score": float(value)}
        for label, value in sorted(
            scores.items(),
            key=lambda item: (-float(item[1]), str(item[0])),
        )
    ]
