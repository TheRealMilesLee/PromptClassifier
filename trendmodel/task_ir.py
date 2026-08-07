"""Task IR types emitted by the prompt tagger.

The IR is deliberately independent from any concrete model router.  A router
can consume this object, while the tagger remains responsible only for
understanding the request and exposing uncertainty/evidence.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


def clamp_probability(value: float) -> float:
    """Keep a score JSON-safe and inside the probability interval."""

    return round(max(0.0, min(1.0, float(value))), 4)


@dataclass(frozen=True)
class LabelCandidate:
    label: str
    probability: float
    name: str | None = None

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "label": self.label,
            "probability": clamp_probability(self.probability),
        }
        if self.name is not None:
            result["name"] = self.name
        return result


@dataclass(frozen=True)
class TaskPrediction:
    """One probabilistic head of the hierarchical classifier."""

    label: str
    probability: float
    candidates: tuple[LabelCandidate, ...] = ()
    name: str | None = None

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "label": self.label,
            "probability": clamp_probability(self.probability),
            "candidates": [candidate.to_dict() for candidate in self.candidates],
        }
        if self.name is not None:
            result["name"] = self.name
        return result


@dataclass(frozen=True)
class TaskConstraints:
    requires_files: bool = False
    requires_tools: bool = False
    has_code_block: bool = False
    has_file_paths: bool = False
    has_shell_commands: bool = False
    has_math_symbols: bool = False
    has_urls: bool = False

    def to_dict(self) -> dict[str, bool]:
        return {
            "requires_files": self.requires_files,
            "requires_tools": self.requires_tools,
            "has_code_block": self.has_code_block,
            "has_file_paths": self.has_file_paths,
            "has_shell_commands": self.has_shell_commands,
            "has_math_symbols": self.has_math_symbols,
            "has_urls": self.has_urls,
        }


@dataclass(frozen=True)
class CapabilityDemand:
    reasoning: float
    coding: float
    context: float
    tool_use: float

    def to_dict(self) -> dict[str, float]:
        return {
            "reasoning": clamp_probability(self.reasoning),
            "coding": clamp_probability(self.coding),
            "context": clamp_probability(self.context),
            "tool_use": clamp_probability(self.tool_use),
        }


@dataclass(frozen=True)
class TaskEvidence:
    """An observable signal used during fusion, not an opaque explanation."""

    kind: str
    label: str
    value: str | bool | float
    strength: float
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "label": self.label,
            "value": self.value,
            "strength": clamp_probability(self.strength),
            "detail": self.detail,
        }


@dataclass(frozen=True)
class TaskIR:
    """Stable, router-facing representation of one user request."""

    original_prompt: str
    intent: TaskPrediction
    domains: tuple[TaskPrediction, ...]
    workload_type: TaskPrediction
    artifact: str
    artifact_probability: float
    operations: tuple[str, ...]
    language: str | None
    constraints: TaskConstraints
    capability_demand: CapabilityDemand
    complexity: TaskPrediction
    effort: Mapping[str, Any]
    risk_level: str
    confidence: float
    needs_clarification: bool
    evidence: tuple[TaskEvidence, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-compatible output without leaking model internals."""

        return {
            "original_prompt": self.original_prompt,
            "intent": self.intent.to_dict(),
            "domains": [domain.to_dict() for domain in self.domains],
            "workload_type": self.workload_type.to_dict(),
            "artifact": self.artifact,
            "artifact_probability": clamp_probability(self.artifact_probability),
            "operations": list(self.operations),
            "language": self.language,
            "constraints": self.constraints.to_dict(),
            "capability_demand": self.capability_demand.to_dict(),
            "complexity": _json_value(self.complexity.to_dict()),
            "effort": _json_value(dict(self.effort)),
            "risk_level": self.risk_level,
            "confidence": clamp_probability(self.confidence),
            "needs_clarification": self.needs_clarification,
            "evidence": [item.to_dict() for item in self.evidence],
        }

    # ``as_dict`` is convenient for callers that use the IR as a plain value.
    as_dict = to_dict


def _json_value(value: Any) -> Any:
    """Recursively convert tuples and mappings into ordinary JSON values."""

    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_json_value(item) for item in value]
    if isinstance(value, list):
        return [_json_value(item) for item in value]
    return value
