"""Task IR types emitted by the prompt tagger.

The IR is deliberately independent from any concrete model router.  A router
can consume this object, while the tagger remains responsible only for
understanding the request and exposing uncertainty/evidence.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar, Mapping


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
    model_version: str = "task-ir-v1"
    fusion: Mapping[str, Any] = field(default_factory=dict)
    confidence_by_head: Mapping[str, float] = field(default_factory=dict)

    contract_version: ClassVar[str] = "task-ir-v1"

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-compatible output without leaking model internals."""

        result = {
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
        # Keep the original prototype shape while exposing one canonical
        # contract that the TensorFlow adapter can produce as well.
        result["task_ir"] = self.to_contract()
        return result

    def to_contract(self) -> dict[str, Any]:
        """Return the shared model-independent Task IR contract."""

        fusion = {
            "semantic_model": "task-ir-prototype",
            "task_ast_schema": "task-ast-v1",
            "post_hoc_probability_fusion": True,
            "policy": "prototype semantic, structural and optional context fusion",
            **dict(self.fusion),
        }
        confidence = {
            "overall": clamp_probability(self.confidence),
            **{
                str(label): clamp_probability(value)
                for label, value in self.confidence_by_head.items()
            },
        }
        return {
            "contract_version": self.contract_version,
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
            "complexity": self.complexity.to_dict(),
            "effort": _json_value(dict(self.effort)),
            "risk_level": self.risk_level,
            "confidence": confidence,
            "needs_clarification": self.needs_clarification,
            "evidence": [item.to_dict() for item in self.evidence],
            "fusion": _json_value(fusion),
            "model_version": self.model_version,
        }

    @classmethod
    def from_tensorflow(
        cls,
        *,
        predictions: Mapping[str, Any],
        effort: Mapping[str, Any],
        task_ast: Any,
        confidence: Mapping[str, Any],
        needs_clarification: bool,
        model_version: str,
        fusion: Mapping[str, Any],
    ) -> "TaskIR":
        """Adapt TensorFlow heads and TaskAST into the canonical Task IR.

        TensorFlow currently has no learned domain or artifact heads.  Those
        fields are therefore explicitly marked as TaskAST-derived rather than
        presented as hidden neural predictions.
        """

        intent = _prediction_from_mapping(predictions.get("intent", {}))
        workload = _prediction_from_mapping(predictions.get("workload_type", {}))
        complexity = _prediction_from_mapping(predictions.get("complexity", {}))
        domain_scores = _mapping_scores(getattr(task_ast, "domain_scores", {}))
        domains = _predictions_from_scores(domain_scores)
        artifact_scores = _mapping_scores(getattr(task_ast, "artifact_scores", {}))
        artifact, artifact_probability = _top_score(artifact_scores, "unknown")
        language = getattr(task_ast, "language", None)
        constraints = getattr(task_ast, "constraints")
        operations = tuple(getattr(task_ast, "operations", ()))
        intent_label = intent.label
        workload_label = workload.label
        capability = _capability_from_task_ast(
            task_ast,
            intent_label=intent_label,
            workload_label=workload_label,
            artifact=artifact,
            domains=domains,
        )
        risk_level = _risk_from_task_ast(task_ast, intent_label)
        evidence = tuple(getattr(task_ast, "evidence", getattr(task_ast, "signals", ())))
        confidence_by_head = {
            str(label): float(value)
            for label, value in confidence.items()
            if str(label) != "overall"
        }
        return cls(
            original_prompt=str(getattr(task_ast, "original_prompt", "")),
            intent=intent,
            domains=domains,
            workload_type=workload,
            artifact=artifact,
            artifact_probability=artifact_probability,
            operations=operations,
            language=language,
            constraints=constraints,
            capability_demand=capability,
            complexity=complexity,
            effort=effort,
            risk_level=risk_level,
            confidence=float(confidence.get("overall", 0.0)),
            needs_clarification=bool(needs_clarification),
            evidence=evidence,
            model_version=model_version,
            fusion=fusion,
            confidence_by_head=confidence_by_head,
        )

    # ``as_dict`` is convenient for callers that use the IR as a plain value.
    as_dict = to_dict


def _prediction_from_mapping(value: Any) -> TaskPrediction:
    if not isinstance(value, Mapping):
        value = {}
    candidates = tuple(
        LabelCandidate(
            str(item.get("label", "unknown")),
            float(item.get("probability", 0.0)),
            item.get("name"),
        )
        for item in value.get("candidates", [])
        if isinstance(item, Mapping)
    )
    label = str(value.get("label", "unknown"))
    return TaskPrediction(
        label=label,
        probability=float(value.get("probability", 0.0)),
        candidates=candidates,
        name=value.get("name"),
    )


def _mapping_scores(value: Any) -> dict[str, float]:
    if not isinstance(value, Mapping):
        return {}
    return {
        str(label): max(0.0, min(1.0, float(score)))
        for label, score in value.items()
    }


def _top_score(scores: Mapping[str, float], fallback: str) -> tuple[str, float]:
    if not scores:
        return fallback, 0.0
    label, score = max(scores.items(), key=lambda item: (float(item[1]), item[0]))
    return str(label), clamp_probability(score)


def _predictions_from_scores(scores: Mapping[str, float]) -> tuple[TaskPrediction, ...]:
    if not scores:
        return ()
    total = sum(max(0.0, float(score)) for score in scores.values())
    if total <= 0.0:
        return ()
    probabilities = {
        label: max(0.0, float(score)) / total
        for label, score in scores.items()
    }
    candidates = tuple(
        LabelCandidate(label, probability)
        for label, probability in sorted(
            probabilities.items(),
            key=lambda item: (-item[1], item[0]),
        )[:3]
    )
    return tuple(
        TaskPrediction(
            label=label,
            probability=probability,
            candidates=candidates,
            name=label,
        )
        for label, probability in sorted(
            probabilities.items(),
            key=lambda item: (-item[1], item[0]),
        )[:4]
    )


def _capability_from_task_ast(
    task_ast: Any,
    *,
    intent_label: str,
    workload_label: str,
    artifact: str,
    domains: tuple[TaskPrediction, ...],
) -> CapabilityDemand:
    constraints = task_ast.constraints
    domain_labels = {item.label for item in domains}
    reasoning = 0.75 if intent_label in {"analyze", "debug", "explain", "review", "plan"} else 0.35
    if intent_label in {"create", "modify", "execute"}:
        reasoning = max(reasoning, 0.55)
    coding = 0.85 if workload_label == "code" or artifact == "source_code" else 0.2
    if getattr(task_ast, "language", None) and task_ast.language != "shell":
        coding = max(coding, 0.7)
    context = 0.75 if constraints.requires_files or constraints.has_urls else 0.25
    if len(domains) > 1:
        context = max(context, 0.65)
    tool_use = 0.9 if constraints.requires_tools else 0.1
    if "shell" in domain_labels or "operations" in domain_labels:
        tool_use = max(tool_use, 0.55)
    return CapabilityDemand(reasoning, coding, context, tool_use)


def _risk_from_task_ast(task_ast: Any, intent_label: str) -> str:
    if getattr(task_ast, "high_risk", False) and intent_label in {"execute", "create", "modify", "debug"}:
        return "high"
    if getattr(task_ast, "high_risk", False) or task_ast.constraints.requires_tools or intent_label in {"execute", "modify", "debug"}:
        return "medium"
    return "low"


def _json_value(value: Any) -> Any:
    """Recursively convert tuples and mappings into ordinary JSON values."""

    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_json_value(item) for item in value]
    if isinstance(value, list):
        return [_json_value(item) for item in value]
    return value
