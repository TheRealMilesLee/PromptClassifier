"""Small dependency-free probability calibration and abstention utilities."""

from __future__ import annotations

import math
from typing import Any, Iterable, Mapping, Sequence


def temperature_scale(probabilities: Sequence[float], temperature: float) -> list[float]:
    """Apply scalar temperature scaling to a probability distribution."""

    if temperature <= 0.0:
        raise ValueError("temperature must be positive")
    if not probabilities:
        return []
    logits = [math.log(max(1e-9, min(1.0, float(value)))) / temperature for value in probabilities]
    maximum = max(logits)
    exponentials = [math.exp(value - maximum) for value in logits]
    total = sum(exponentials)
    if total <= 0.0:
        return [1.0 / len(probabilities)] * len(probabilities)
    return [value / total for value in exponentials]


def negative_log_likelihood(
    probability_rows: Iterable[Sequence[float]],
    targets: Iterable[int],
) -> float:
    rows = list(probability_rows)
    labels = list(targets)
    if not rows:
        return 0.0
    losses = []
    for probabilities, target in zip(rows, labels):
        if not 0 <= int(target) < len(probabilities):
            continue
        losses.append(-math.log(max(1e-9, min(1.0, float(probabilities[int(target)])))))
    return sum(losses) / len(losses) if losses else 0.0


def expected_calibration_error(
    probability_rows: Iterable[Sequence[float]],
    targets: Iterable[int],
    *,
    bins: int = 10,
) -> float:
    """Compute top-label ECE without requiring scikit-learn."""

    if bins < 1:
        raise ValueError("bins must be positive")
    rows = list(probability_rows)
    labels = list(targets)
    buckets: list[list[tuple[float, bool]]] = [[] for _ in range(bins)]
    for probabilities, target in zip(rows, labels):
        if not probabilities:
            continue
        top_index = max(range(len(probabilities)), key=probabilities.__getitem__)
        confidence = max(0.0, min(1.0, float(probabilities[top_index])))
        bucket = min(bins - 1, int(confidence * bins))
        buckets[bucket].append((confidence, top_index == int(target)))
    total = sum(len(bucket) for bucket in buckets)
    if total == 0:
        return 0.0
    error = 0.0
    for bucket in buckets:
        if not bucket:
            continue
        mean_confidence = sum(item[0] for item in bucket) / len(bucket)
        accuracy = sum(item[1] for item in bucket) / len(bucket)
        error += len(bucket) / total * abs(mean_confidence - accuracy)
    return round(error, 6)


def fit_temperature(
    probability_rows: Iterable[Sequence[float]],
    targets: Iterable[int],
    *,
    minimum: float = 0.25,
    maximum: float = 4.0,
    steps: int = 151,
) -> dict[str, float | int]:
    """Fit one temperature with a deterministic logarithmic grid search."""

    rows = [list(row) for row in probability_rows]
    labels = [int(label) for label in targets]
    if not rows:
        return {"temperature": 1.0, "samples": 0, "nll_before": 0.0, "nll_after": 0.0}
    if minimum <= 0.0 or maximum < minimum or steps < 2:
        raise ValueError("invalid temperature search range")
    candidates = [
        math.exp(
            math.log(minimum)
            + (math.log(maximum) - math.log(minimum)) * index / (steps - 1)
        )
        for index in range(steps)
    ]
    before = negative_log_likelihood(rows, labels)
    best_temperature = min(
        candidates,
        key=lambda temperature: negative_log_likelihood(
            (temperature_scale(row, temperature) for row in rows),
            labels,
        ),
    )
    after_rows = [temperature_scale(row, best_temperature) for row in rows]
    return {
        "temperature": round(best_temperature, 6),
        "samples": len(rows),
        "nll_before": round(before, 6),
        "nll_after": round(negative_log_likelihood(after_rows, labels), 6),
        "ece_before": expected_calibration_error(rows, labels),
        "ece_after": expected_calibration_error(after_rows, labels),
    }


def fit_head_calibration(
    predictions: Iterable[Mapping[str, Any]],
    actual_labels: Iterable[str | None],
    *,
    head: str,
    labels: Sequence[str],
) -> dict[str, float | int]:
    """Fit calibration from the full ``distribution`` emitted by inference."""

    rows: list[list[float]] = []
    targets: list[int] = []
    index = {label: position for position, label in enumerate(labels)}
    for prediction, actual in zip(predictions, actual_labels):
        if actual is None or str(actual) not in index:
            continue
        nested = prediction.get("predictions", {})
        head_payload = nested.get(head, {}) if isinstance(nested, Mapping) else {}
        distribution = head_payload.get("distribution", {}) if isinstance(head_payload, Mapping) else {}
        if not isinstance(distribution, Mapping):
            continue
        row = [float(distribution.get(label, 0.0)) for label in labels]
        if sum(row) <= 0.0:
            continue
        rows.append(row)
        targets.append(index[str(actual)])
    return fit_temperature(rows, targets)


def calibration_temperature(metadata: Mapping[str, Any], head: str) -> float:
    calibration = metadata.get("calibration", {})
    heads = calibration.get("heads", {}) if isinstance(calibration, Mapping) else {}
    payload = heads.get(head, {}) if isinstance(heads, Mapping) else {}
    if not isinstance(payload, Mapping):
        return 1.0
    try:
        return max(0.01, float(payload.get("temperature", 1.0)))
    except (TypeError, ValueError):
        return 1.0


def abstention_decision(
    predictions: Mapping[str, Mapping[str, Any]],
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    """Return a conservative, configurable abstention decision."""

    config = metadata.get("abstention", {})
    if not isinstance(config, Mapping) or not bool(config.get("enabled", False)):
        return {"abstain": False, "reasons": [], "enabled": False}
    minimum_probability = float(config.get("min_probability", 0.45))
    minimum_margin = float(config.get("min_margin", 0.05))
    maximum_entropy = float(config.get("max_entropy", 0.98))
    reasons: list[str] = []
    for head in ("workload_type", "intent", "complexity"):
        payload = predictions.get(head, {})
        if not isinstance(payload, Mapping):
            continue
        label = str(payload.get("label", "unknown"))
        probability = float(payload.get("probability", 0.0))
        margin = float(payload.get("margin", 0.0))
        entropy = float(payload.get("entropy", 1.0))
        if label == "unknown" or probability < minimum_probability:
            reasons.append(f"{head}:low_probability")
        elif margin < minimum_margin:
            reasons.append(f"{head}:low_margin")
        elif entropy > maximum_entropy:
            reasons.append(f"{head}:high_entropy")
    return {
        "abstain": bool(reasons),
        "reasons": reasons,
        "enabled": True,
        "thresholds": {
            "min_probability": minimum_probability,
            "min_margin": minimum_margin,
            "max_entropy": maximum_entropy,
        },
    }
