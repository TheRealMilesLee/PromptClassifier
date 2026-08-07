"""Probability and uncertainty utilities for task evidence fusion."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True)
class FusionWeights:
    """Initial weights from the design document.

    These are policy defaults, not calibrated probabilities.  They should be
    fitted against a held-out internal annotation set once one exists.
    """

    semantic: float = 0.70
    structural: float = 0.20
    context_prior: float = 0.10

    def normalized(self) -> tuple[float, float, float]:
        total = self.semantic + self.structural + self.context_prior
        if total <= 0:
            raise ValueError("fusion weights must have a positive sum")
        return (
            self.semantic / total,
            self.structural / total,
            self.context_prior / total,
        )


@dataclass(frozen=True)
class FusionResult:
    semantic: dict[str, float]
    structural: dict[str, float]
    context_prior: dict[str, float]
    posterior: dict[str, float]


def cosine_similarity(left: list[float], right: list[float]) -> float:
    """Compute cosine similarity without requiring NumPy."""

    if len(left) != len(right):
        raise ValueError("embedding dimensions do not match")
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return sum(a * b for a, b in zip(left, right)) / (left_norm * right_norm)


def softmax(scores: Mapping[str, float], *, temperature: float = 1.0) -> dict[str, float]:
    """Turn arbitrary finite scores into a stable probability distribution."""

    if not scores:
        return {}
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    labels = list(scores)
    scaled = [float(scores[label]) / temperature for label in labels]
    maximum = max(scaled)
    exponentials = [math.exp(max(-80.0, min(80.0, value - maximum))) for value in scaled]
    total = sum(exponentials)
    if total == 0.0:
        return uniform_distribution(labels)
    return {label: value / total for label, value in zip(labels, exponentials)}


def uniform_distribution(labels: list[str] | tuple[str, ...]) -> dict[str, float]:
    if not labels:
        return {}
    probability = 1.0 / len(labels)
    return {label: probability for label in labels}


def fuse_probabilities(
    semantic_scores: Mapping[str, float],
    structural_scores: Mapping[str, float] | None = None,
    context_prior: Mapping[str, float] | None = None,
    *,
    weights: FusionWeights = FusionWeights(),
    semantic_temperature: float = 0.18,
    structural_temperature: float = 0.22,
    prior_temperature: float = 0.5,
) -> FusionResult:
    """Fuse semantic, structural and contextual evidence.

    The implementation uses the proposed weighted probability mixture:

    ``0.70 * semantic + 0.20 * structural + 0.10 * prior``.

    A missing evidence source is neutral (uniform), so an absent file or
    command signal cannot silently push a class down.  This is a deliberately
    conservative Bayesian-style layer; it does not claim calibrated posterior
    probabilities until calibration data is available.
    """

    labels = list(dict.fromkeys(
        list(semantic_scores)
        + list(structural_scores or {})
        + list(context_prior or {})
    ))
    if not labels:
        raise ValueError("at least one label is required for fusion")

    semantic = _distribution_for_labels(
        semantic_scores,
        labels,
        temperature=semantic_temperature,
        neutral_if_missing=False,
    )
    structural = _distribution_for_labels(
        structural_scores,
        labels,
        temperature=structural_temperature,
        neutral_if_missing=True,
    )
    prior = _distribution_for_labels(
        context_prior,
        labels,
        temperature=prior_temperature,
        neutral_if_missing=True,
    )
    semantic_weight, structural_weight, prior_weight = weights.normalized()
    posterior = {
        label: (
            semantic_weight * semantic[label]
            + structural_weight * structural[label]
            + prior_weight * prior[label]
        )
        for label in labels
    }
    posterior = _renormalize(posterior)
    return FusionResult(semantic, structural, prior, posterior)


def bayesian_product_fusion(
    semantic_scores: Mapping[str, float],
    structural_scores: Mapping[str, float] | None = None,
    context_prior: Mapping[str, float] | None = None,
    *,
    weights: FusionWeights = FusionWeights(),
) -> FusionResult:
    """Fuse likelihoods multiplicatively in log space.

    This helper is useful for experiments comparing the linear mixture with a
    closer ``P(class | evidence) ∝ likelihood * prior`` formulation.  The
    public tagger defaults to :func:`fuse_probabilities` because sparse rule
    evidence should not collapse a result when a signal is imperfect.
    """

    labels = list(dict.fromkeys(
        list(semantic_scores)
        + list(structural_scores or {})
        + list(context_prior or {})
    ))
    if not labels:
        raise ValueError("at least one label is required for fusion")
    semantic = _distribution_for_labels(semantic_scores, labels, temperature=0.18, neutral_if_missing=False)
    structural = _distribution_for_labels(structural_scores, labels, temperature=0.22, neutral_if_missing=True)
    prior = _distribution_for_labels(context_prior, labels, temperature=0.5, neutral_if_missing=True)
    semantic_weight, structural_weight, prior_weight = weights.normalized()
    product = {
        label: math.exp(
            semantic_weight * math.log(max(semantic[label], 1e-12))
            + structural_weight * math.log(max(structural[label], 1e-12))
            + prior_weight * math.log(max(prior[label], 1e-12))
        )
        for label in labels
    }
    return FusionResult(semantic, structural, prior, _renormalize(product))


def ranked(probabilities: Mapping[str, float], *, limit: int = 3) -> list[tuple[str, float]]:
    return sorted(probabilities.items(), key=lambda item: (-item[1], item[0]))[:limit]


def confidence(probabilities: Mapping[str, float]) -> float:
    """Estimate confidence from dominance and normalized entropy.

    This is an uncertainty score for routing thresholds, not a calibrated
    accuracy claim.  Calibration can replace this function without changing
    the Task IR contract.
    """

    if not probabilities:
        return 0.0
    values = [max(0.0, float(value)) for value in probabilities.values()]
    total = sum(values)
    if total <= 0.0:
        return 0.0
    values = [value / total for value in values]
    peak = max(values)
    if len(values) == 1:
        return peak
    entropy = -sum(value * math.log(value, 2) for value in values if value > 0.0)
    max_entropy = math.log(len(values), 2)
    certainty = 1.0 - entropy / max_entropy if max_entropy else 1.0
    return max(0.0, min(1.0, 0.55 * peak + 0.45 * certainty))


def _distribution_for_labels(
    scores: Mapping[str, float] | None,
    labels: list[str],
    *,
    temperature: float,
    neutral_if_missing: bool,
) -> dict[str, float]:
    if scores is None or not scores:
        return uniform_distribution(labels)
    completed = {label: float(scores.get(label, 0.0)) for label in labels}
    if neutral_if_missing and max(completed.values()) - min(completed.values()) < 1e-12:
        return uniform_distribution(labels)
    return softmax(completed, temperature=temperature)


def _renormalize(values: Mapping[str, float]) -> dict[str, float]:
    total = sum(max(0.0, float(value)) for value in values.values())
    if total <= 0.0:
        return uniform_distribution(list(values))
    return {label: max(0.0, float(value)) / total for label, value in values.items()}
