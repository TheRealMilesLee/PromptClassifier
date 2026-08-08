"""Deterministic evidence-only baseline used by TensorFlow ablation reports."""

from __future__ import annotations

from typing import Any, Iterable, Mapping

from .evidence import extract_evidence
from .task_ast import TaskAST


ABLATION_MODES = ("text_only", "text_ast", "evidence_only")


def normalize_ablation_mode(value: str) -> str:
    """Normalize CLI-friendly ablation names to the internal identifiers."""

    aliases = {
        "none": "none",
        "all": "all",
        "text-only": "text_only",
        "text_only": "text_only",
        "text+ast": "text_ast",
        "text-ast": "text_ast",
        "text_ast": "text_ast",
        "evidence-only": "evidence_only",
        "evidence_only": "evidence_only",
    }
    try:
        return aliases[str(value).casefold()]
    except KeyError as error:
        raise ValueError(
            "ablation must be 'none', 'all', 'text-only', 'text+ast', or 'evidence-only'"
        ) from error


def evidence_only_prediction(
    prompt: str,
    *,
    workload_labels: Iterable[str],
    intent_labels: Iterable[str],
    complexity_labels: Iterable[str],
) -> dict[str, Any]:
    """Build a deliberately non-semantic prediction from TaskAST evidence.

    This is a diagnostic baseline, not a second production classifier.  It
    measures how far deterministic structure alone can go and keeps effort
    unknown because the evidence extractor has no observed time supervision.
    """

    task_ast = extract_evidence(prompt)
    workload_labels = list(workload_labels)
    intent_labels = list(intent_labels)
    complexity_labels = list(complexity_labels)

    workload_scores = _workload_scores(task_ast, workload_labels)
    intent_scores = _scores_for_labels(task_ast.intent_scores, intent_labels)
    workload_label, workload_probabilities = _select_scores(
        workload_scores,
        workload_labels,
        fallback="unknown",
    )
    intent_label, intent_probabilities = _select_scores(
        intent_scores,
        intent_labels,
        fallback="unknown",
    )
    complexity_label = _complexity_label(task_ast, complexity_labels)
    complexity_scores = {
        label: 1.0 if label == complexity_label else 0.0
        for label in complexity_labels
    }
    complexity_probability = 1.0 if complexity_label in complexity_labels else 0.0
    evidence = [item.to_dict() for item in task_ast.evidence]
    needs_clarification = not evidence and not task_ast.intent_scores
    reasons = ["no_structured_evidence"] if needs_clarification else []
    effort = {
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

    return {
        "original_prompt": str(prompt),
        "workload_type": workload_label,
        "intent": intent_label,
        "complexity": complexity_label,
        "predictions": {
            "workload_type": _classification_payload(
                workload_label,
                workload_probabilities,
                workload_labels,
            ),
            "intent": _classification_payload(
                intent_label,
                intent_probabilities,
                intent_labels,
            ),
            "complexity": _classification_payload(
                complexity_label,
                complexity_scores,
                complexity_labels,
            ),
            "effort": effort,
        },
        "effort": effort,
        "evidence": evidence,
        "needs_clarification": needs_clarification,
        "clarification_reasons": reasons,
        "confidence": {
            "workload_type": _selected_probability(workload_label, workload_probabilities, workload_labels),
            "intent": _selected_probability(intent_label, intent_probabilities, intent_labels),
            "complexity": complexity_probability,
            "effort": 0.0,
            "overall": round(
                0.35 * _selected_probability(workload_label, workload_probabilities, workload_labels)
                + 0.35 * _selected_probability(intent_label, intent_probabilities, intent_labels)
                + 0.30 * complexity_probability,
                4,
            ),
        },
        "fusion": {
            "semantic_model": "none",
            "task_ast_schema": task_ast.schema_version,
            "post_hoc_probability_fusion": False,
            "policy": "TaskAST deterministic evidence-only diagnostic baseline; not a production model",
        },
        "model_version": "evidence-only-v1",
    }


def _workload_scores(task_ast: TaskAST, labels: list[str]) -> dict[str, float]:
    scores = {label: 0.0 for label in labels}

    def add(label: str, value: float) -> None:
        if label in scores:
            scores[label] = max(scores[label], max(0.0, min(1.0, float(value))))

    domain_to_workload = {
        "programming": "code",
        "shell": "operations",
        "operations": "operations",
        "mathematics": "analysis",
        "data_analysis": "analysis",
        "planning": "planning",
        "creative": "text",
        "conversation": "text",
        "document": "text",
    }
    for domain, value in task_ast.domain_scores.items():
        workload = domain_to_workload.get(domain)
        if workload:
            add(workload, value)
    if task_ast.artifact_scores.get("source_code", 0.0) > 0.0:
        add("code", task_ast.artifact_scores["source_code"])
    if task_ast.constraints.has_shell_commands or task_ast.constraints.requires_tools:
        add("operations", 0.95)
    if task_ast.high_risk:
        add("operations", 1.0)

    nonzero = sorted(
        ((label, value) for label, value in scores.items() if value > 0.0),
        key=lambda item: (-item[1], item[0]),
    )
    if len(nonzero) >= 2 and "mixed" in scores:
        top, second = nonzero[0], nonzero[1]
        if top[1] - second[1] <= 0.15:
            add("mixed", (top[1] + second[1]) / 2.0)
    return scores


def _scores_for_labels(source: Mapping[str, float], labels: list[str]) -> dict[str, float]:
    return {
        label: max(0.0, min(1.0, float(source.get(label, 0.0))))
        for label in labels
    }


def _select_scores(
    scores: Mapping[str, float],
    labels: list[str],
    *,
    fallback: str,
) -> tuple[str, list[float]]:
    if not labels:
        return fallback, []
    total = sum(max(0.0, float(scores.get(label, 0.0))) for label in labels)
    if total <= 0.0:
        label = fallback if fallback in labels else labels[0]
        probabilities = [1.0 if candidate == label else 0.0 for candidate in labels]
        return label, probabilities
    probabilities = [max(0.0, float(scores.get(label, 0.0))) / total for label in labels]
    index = max(range(len(labels)), key=lambda position: (probabilities[position], -position))
    return labels[index], probabilities


def _complexity_label(task_ast: TaskAST, labels: list[str]) -> str:
    if not labels:
        return "unknown"
    if "high" in labels and (
        task_ast.high_risk
        or task_ast.constraints.requires_tools
        or "execute" in task_ast.intent_scores
        or len(task_ast.operations) >= 2
    ):
        return "high"
    if "medium" in labels and (
        task_ast.constraints.has_code_block
        or task_ast.constraints.has_file_paths
        or task_ast.constraints.has_shell_commands
        or len(task_ast.operations) >= 1
        or len(task_ast.evidence) >= 2
    ):
        return "medium"
    return "low" if "low" in labels else labels[0]


def _classification_payload(
    label: str,
    probabilities: list[float] | Mapping[str, float],
    labels: list[str],
) -> dict[str, Any]:
    if isinstance(probabilities, Mapping):
        values = [float(probabilities.get(candidate, 0.0)) for candidate in labels]
    else:
        values = [float(value) for value in probabilities]
    ranked = sorted(
        zip(labels, values),
        key=lambda item: (-item[1], item[0]),
    )[:3]
    positive = [value for value in values if value > 0.0]
    entropy = 1.0
    if positive and len(values) > 1:
        total = sum(values)
        normalized = [value / total for value in values if value > 0.0]
        import math

        entropy = -sum(value * math.log(value) for value in normalized) / math.log(len(values))
    ranked_values = sorted(values, reverse=True)
    margin = ranked_values[0] - (ranked_values[1] if len(ranked_values) > 1 else 0.0)
    return {
        "label": label,
        "probability": round(_selected_probability(label, values, labels), 4),
        "entropy": round(max(0.0, min(1.0, entropy)), 4),
        "margin": round(max(0.0, min(1.0, margin)), 4),
        "candidates": [
            {"label": candidate, "probability": round(float(value), 4)}
            for candidate, value in ranked
        ],
    }


def _selected_probability(label: str, probabilities: list[float], labels: list[str]) -> float:
    try:
        return max(0.0, min(1.0, float(probabilities[labels.index(label)])))
    except (ValueError, IndexError):
        return 0.0
