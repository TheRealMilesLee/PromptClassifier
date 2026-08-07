from __future__ import annotations

import random
from collections import Counter
from typing import Iterable, Mapping

from .model import HEADS, TaskModel
from .schema import TrainingExample


def _accuracy(actual: list[str], predicted: list[str]) -> float:
    if not actual:
        return 0.0
    return sum(left == right for left, right in zip(actual, predicted)) / len(actual)


def _macro_f1(actual: list[str], predicted: list[str]) -> float:
    labels = sorted(set(actual) | set(predicted))
    if not labels:
        return 0.0
    scores: list[float] = []
    for label in labels:
        true_positive = sum(a == label and p == label for a, p in zip(actual, predicted))
        false_positive = sum(a != label and p == label for a, p in zip(actual, predicted))
        false_negative = sum(a == label and p != label for a, p in zip(actual, predicted))
        precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
        recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
        scores.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return sum(scores) / len(scores)


def evaluate_predictions(
    model: TaskModel,
    examples: Iterable[TrainingExample],
) -> dict[str, object]:
    labelled = list(examples)
    metrics: dict[str, object] = {"samples": len(labelled), "heads": {}}
    for head in HEADS:
        actual = [getattr(example, head) for example in labelled]
        predicted = [str(model.predict(example.text)[head]["label"]) for example in labelled]
        metrics["heads"][head] = {
            "accuracy": round(_accuracy(actual, predicted), 4),
            "macro_f1": round(_macro_f1(actual, predicted), 4),
            "actual_distribution": dict(Counter(actual)),
            "predicted_distribution": dict(Counter(predicted)),
        }
    return metrics


def holdout_evaluate(
    examples: Iterable[TrainingExample],
    *,
    test_ratio: float = 0.25,
    seed: int = 42,
) -> dict[str, object]:
    labelled = list(examples)
    if len(labelled) < 4:
        raise ValueError("at least four examples are required for holdout evaluation")
    if not 0 < test_ratio < 1:
        raise ValueError("test_ratio must be between 0 and 1")

    shuffled = list(labelled)
    random.Random(seed).shuffle(shuffled)
    test_size = max(1, round(len(shuffled) * test_ratio))
    test = shuffled[:test_size]
    train = shuffled[test_size:]
    model = TaskModel.fit(train)
    result = evaluate_predictions(model, test)
    result["train_samples"] = len(train)
    result["test_samples"] = len(test)
    result["seed"] = seed
    return result
