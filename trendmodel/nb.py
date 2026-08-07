from __future__ import annotations

import math
from collections import Counter
from typing import Iterable, Mapping


class MultinomialNB:
    """Sparse multinomial Naive Bayes with additive smoothing."""

    def __init__(self, alpha: float = 0.5, temperature: float = 2.0) -> None:
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self.alpha = alpha
        self.temperature = temperature
        self.labels: list[str] = []
        self.class_doc_counts: Counter[str] = Counter()
        self.feature_counts: dict[str, Counter[int]] = {}
        self.class_weight_totals: dict[str, float] = {}
        self.n_features = 0

    def fit(self, vectors: Iterable[Mapping[int, float]], labels: Iterable[str]) -> "MultinomialNB":
        vector_list = list(vectors)
        label_list = [str(label) for label in labels]
        if not vector_list or len(vector_list) != len(label_list):
            raise ValueError("vectors and labels must be non-empty and have equal length")

        self.labels = sorted(set(label_list))
        self.class_doc_counts = Counter(label_list)
        self.feature_counts = {label: Counter() for label in self.labels}
        self.class_weight_totals = {label: 0.0 for label in self.labels}

        for vector, label in zip(vector_list, label_list):
            for index, value in vector.items():
                self.feature_counts[label][int(index)] += float(value)
                self.class_weight_totals[label] += float(value)

        self.n_features = max(
            (max(vector.keys()) for vector in vector_list if vector),
            default=-1,
        ) + 1
        return self

    def predict_proba(self, vector: Mapping[int, float]) -> dict[str, float]:
        if not self.labels:
            raise ValueError("classifier has not been fitted")

        total_docs = sum(self.class_doc_counts.values())
        raw_scores: dict[str, float] = {}
        for label in self.labels:
            prior = self.class_doc_counts[label] / total_docs
            denominator = self.class_weight_totals[label] + self.alpha * max(self.n_features, 1)
            score = math.log(prior)
            for index, value in vector.items():
                numerator = self.feature_counts[label].get(int(index), 0.0) + self.alpha
                score += float(value) * math.log(numerator / denominator)
            raw_scores[label] = score

        maximum = max(raw_scores.values())
        exponentials = {
            label: math.exp((score - maximum) / self.temperature)
            for label, score in raw_scores.items()
        }
        normalizer = sum(exponentials.values())
        probabilities = {label: value / normalizer for label, value in exponentials.items()}
        # A tiny labelled corpus produces brittle, overconfident posteriors.
        # Shrink them slightly toward a uniform prior until calibration data is
        # available; the top label itself is unchanged.
        shrinkage = 0.1
        label_count = len(probabilities)
        return {
            label: (1 - shrinkage) * probability + shrinkage / label_count
            for label, probability in probabilities.items()
        }

    def top(self, vector: Mapping[int, float], limit: int = 3) -> list[dict[str, float | str]]:
        probabilities = self.predict_proba(vector)
        return [
            {"label": label, "score": round(score, 4)}
            for label, score in sorted(probabilities.items(), key=lambda item: (-item[1], item[0]))[:limit]
        ]

    def to_dict(self) -> dict[str, object]:
        return {
            "alpha": self.alpha,
            "temperature": self.temperature,
            "labels": self.labels,
            "class_doc_counts": dict(self.class_doc_counts),
            "feature_counts": {
                label: {str(index): value for index, value in counts.items()}
                for label, counts in self.feature_counts.items()
            },
            "class_weight_totals": self.class_weight_totals,
            "n_features": self.n_features,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "MultinomialNB":
        classifier = cls(
            alpha=float(value.get("alpha", 0.5)),
            temperature=float(value.get("temperature", 2.0)),
        )
        classifier.labels = [str(label) for label in value.get("labels", [])]
        classifier.class_doc_counts = Counter(
            {str(label): int(count) for label, count in dict(value.get("class_doc_counts", {})).items()}
        )
        raw_features = value.get("feature_counts", {})
        classifier.feature_counts = {
            str(label): Counter({int(index): float(count) for index, count in dict(counts).items()})
            for label, counts in dict(raw_features).items()
        }
        classifier.class_weight_totals = {
            str(label): float(total)
            for label, total in dict(value.get("class_weight_totals", {})).items()
        }
        classifier.n_features = int(value.get("n_features", 0))
        return classifier
