"""Sentence embedding backends used by the prototype classifier.

The default ``HashEmbedder`` keeps the tagger runnable in a clean Python
installation and is useful for tests.  A real deployment should install the
optional embedding dependency and select a Chinese/multilingual sentence
model.  The classifier itself does not depend on the backend implementation.
"""

from __future__ import annotations

import hashlib
import math
import re
import unicodedata
from collections.abc import Iterable, Sequence
from typing import Any, Protocol


class Embedder(Protocol):
    """Minimal interface required by the prototype and fusion layers."""

    backend_name: str
    model_name: str
    dimension: int

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        ...

    def embed(self, text: str) -> list[float]:
        ...


class HashEmbedder:
    """Deterministic feature-hashing sentence representation.

    This is a dependency-free development fallback, not a replacement for a
    semantic encoder.  Character n-grams make it useful for Chinese and
    mixed Chinese/English prompts; exact words and word pairs receive a
    slightly higher weight.  Stable cryptographic hashing avoids Python's
    process-randomized ``hash()`` and makes test results reproducible.
    """

    backend_name = "hash"

    def __init__(self, dimension: int = 384, *, model_name: str = "hash-ngram-v1") -> None:
        if dimension < 16:
            raise ValueError("dimension must be at least 16")
        self.dimension = int(dimension)
        self.model_name = model_name

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        if isinstance(texts, str):
            texts = [texts]
        return [self.embed(text) for text in texts]

    def embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimension
        features = list(_features(str(text)))
        for feature, weight in features:
            digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=16).digest()
            index = int.from_bytes(digest[:8], "big") % self.dimension
            sign = 1.0 if digest[8] & 1 else -1.0
            vector[index] += sign * weight

            # A second bucket reduces the impact of a single collision while
            # keeping the representation inexpensive on a laptop CPU.
            second_index = int.from_bytes(digest[8:], "big") % self.dimension
            vector[second_index] += sign * weight * 0.35

        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0.0:
            return vector
        return [value / norm for value in vector]


class SentenceTransformerEmbedder:
    """Adapter around ``sentence-transformers`` with lazy, normalized output."""

    backend_name = "sentence-transformers"

    def __init__(self, model_name: str = "BAAI/bge-small-zh-v1.5", **model_kwargs: Any) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as error:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "sentence-transformers is not installed; install requirements-embedding.txt"
            ) from error

        self.model_name = model_name
        self._model = SentenceTransformer(model_name, **model_kwargs)
        dimension = self._model.get_sentence_embedding_dimension()
        if dimension is None:
            raise RuntimeError(f"unable to determine embedding dimension for {model_name}")
        self.dimension = int(dimension)

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        if isinstance(texts, str):
            texts = [texts]
        values = self._model.encode(
            list(texts),
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        # ``tolist`` works for numpy arrays and the fallback below also makes
        # the adapter friendly to simple fake models in downstream tests.
        if hasattr(values, "tolist"):
            values = values.tolist()
        if values and isinstance(values[0], (int, float)):
            values = [values]
        return [[float(item) for item in row] for row in values]

    def embed(self, text: str) -> list[float]:
        return self.encode([text])[0]


def create_embedder(
    backend: str = "hash",
    *,
    model_name: str = "BAAI/bge-small-zh-v1.5",
    dimension: int = 384,
) -> Embedder:
    """Create a configured backend without importing optional packages eagerly."""

    normalized = str(backend).strip().casefold().replace("_", "-")
    if normalized in {"hash", "local", "fallback"}:
        return HashEmbedder(dimension=dimension)
    if normalized in {"sentence-transformers", "sentence-transformer", "st"}:
        return SentenceTransformerEmbedder(model_name=model_name)
    raise ValueError(f"unsupported embedding backend: {backend}")


_CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
_CJK_RUN_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]+")
_LATIN_TOKEN_RE = re.compile(r"[a-z][a-z0-9_+#.-]*|\d+(?:\.\d+)?", re.IGNORECASE)


def _features(text: str) -> Iterable[tuple[str, float]]:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    compact = re.sub(r"\s+", " ", normalized).strip()

    # Chinese character n-grams are intentionally overlapping: tokenization
    # ambiguity should not make "路由协议" and "网络路由协议" unrelated.  Keep
    # punctuation-separated runs independent so unrelated clauses do not
    # accidentally form a semantic token across sentence boundaries.
    for run in _CJK_RUN_RE.findall(compact):
        for size, weight in ((1, 0.35), (2, 0.95), (3, 1.15), (4, 0.95), (5, 0.75), (6, 0.55)):
            for start in range(max(0, len(run) - size + 1)):
                yield f"cjk:{size}:{run[start:start + size]}", weight
        if len(run) <= 16:
            yield f"cjk_run:{run}", 0.5

    words = _LATIN_TOKEN_RE.findall(compact)
    for word in words:
        yield f"word:{word}", 1.35
    for left, right in zip(words, words[1:]):
        yield f"word2:{left}:{right}", 0.75

    # Keep a small amount of punctuation/shape information for code and
    # formulas without making punctuation the dominant signal.
    for marker in ("```", "->", "=>", "::", "{}", "()", "[]", "=", "?"):
        if marker in compact:
            yield f"marker:{marker}", 0.25
