from __future__ import annotations

import math
from collections import Counter
from typing import Iterable, Mapping

from .features import extract_feature_counts


class TfidfVectorizer:
    """Small sparse TF-IDF vectorizer backed only by the Python standard library."""

    def __init__(self, max_features: int = 20_000) -> None:
        self.max_features = max_features
        self.vocabulary: dict[str, int] = {}
        self.idf: list[float] = []

    def fit(self, texts: Iterable[str]) -> "TfidfVectorizer":
        documents = [extract_feature_counts(text) for text in texts]
        if not documents:
            raise ValueError("at least one training text is required")

        document_frequency: Counter[str] = Counter()
        for document in documents:
            document_frequency.update(document.keys())

        selected = sorted(
            document_frequency.items(),
            key=lambda item: (-item[1], item[0]),
        )[: self.max_features]
        self.vocabulary = {feature: index for index, (feature, _) in enumerate(selected)}
        document_count = len(documents)
        self.idf = [
            math.log((1 + document_count) / (1 + document_frequency[feature])) + 1
            for feature, _ in selected
        ]
        return self

    def transform(self, text: str) -> dict[int, float]:
        if not self.vocabulary:
            raise ValueError("vectorizer has not been fitted")

        counts = extract_feature_counts(text)
        vector: dict[int, float] = {}
        for feature, count in counts.items():
            index = self.vocabulary.get(feature)
            if index is None:
                continue
            vector[index] = (1 + math.log(count)) * self.idf[index]

        return vector

    def transform_many(self, texts: Iterable[str]) -> list[dict[int, float]]:
        return [self.transform(text) for text in texts]

    def to_dict(self) -> dict[str, object]:
        return {
            "max_features": self.max_features,
            "vocabulary": self.vocabulary,
            "idf": self.idf,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "TfidfVectorizer":
        vectorizer = cls(max_features=int(value.get("max_features", 20_000)))
        raw_vocabulary = value.get("vocabulary", {})
        raw_idf = value.get("idf", [])
        if not isinstance(raw_vocabulary, Mapping) or not isinstance(raw_idf, list):
            raise ValueError("invalid vectorizer payload")
        vectorizer.vocabulary = {str(feature): int(index) for feature, index in raw_vocabulary.items()}
        vectorizer.idf = [float(item) for item in raw_idf]
        return vectorizer
