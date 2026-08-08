"""Small Keras multi-task text model."""

from __future__ import annotations

from typing import Iterable

from .tf_data import LabelMaps, normalize_model_text
from .tf_features import STRUCTURED_FEATURE_NAMES


def require_tensorflow():
    try:
        import tensorflow as tf
    except ImportError as error:  # pragma: no cover - depends on local environment
        raise RuntimeError(
            "TensorFlow is not installed. On Apple Silicon run "
            "python3 -m pip install -r requirements-tensorflow-macos.txt; "
            "on Linux or WSL2 with NVIDIA CUDA run "
            "python3 -m pip install -r requirements-tensorflow-cuda.txt"
        ) from error
    return tf


def build_text_model(
    training_texts: Iterable[str],
    label_maps: LabelMaps,
    *,
    max_tokens: int = 12_000,
    ngram_max_tokens: int = 24_000,
    sequence_length: int = 192,
    embedding_dim: int = 96,
    recurrent_units: int = 64,
    learning_rate: float = 1e-3,
):
    """Build a CPU-friendly character/n-gram encoder plus evidence classifier.

    Character splitting is deliberate for the first Chinese/mixed-language
    prototype: it avoids requiring a separate word segmentation service.  A
    character n-gram TF-IDF branch adds phrase-level lexical evidence without
    requiring a tokenizer or external model.  A fixed-width structured branch
    exposes deterministic code/file/command evidence so the small text encoder
    can spend its capacity on wording.
    """

    tf = require_tensorflow()
    layers = tf.keras.layers

    prompt_input = layers.Input(
        shape=(),
        dtype=tf.string,
        name="original_prompt",
    )
    structured_input = layers.Input(
        shape=(len(STRUCTURED_FEATURE_NAMES),),
        dtype=tf.float32,
        name="structured_features",
    )
    vectorizer = layers.TextVectorization(
        max_tokens=max_tokens,
        output_mode="int",
        output_sequence_length=sequence_length,
        split="character",
        # Punctuation is part of the task signal here: ``C++``, file
        # extensions, paths, shell flags and code delimiters distinguish
        # workloads.  Do not use Keras' default punctuation stripping.
        standardize=None,
        name="character_vectorizer",
    )
    normalized_training_texts = [normalize_model_text(text) for text in training_texts]
    text_dataset = tf.data.Dataset.from_tensor_slices(normalized_training_texts).batch(128)
    vectorizer.adapt(text_dataset)
    ngram_vectorizer = layers.TextVectorization(
        max_tokens=ngram_max_tokens,
        output_mode="tf_idf",
        ngrams=(2, 5),
        split="character",
        standardize=None,
        name="character_ngram_vectorizer",
    )
    ngram_vectorizer.adapt(text_dataset)

    tokens = vectorizer(prompt_input)
    embedded = layers.Embedding(
        input_dim=max_tokens,
        output_dim=embedding_dim,
        mask_zero=True,
        name="character_embedding",
    )(tokens)
    encoded = layers.Bidirectional(
        layers.GRU(recurrent_units, dropout=0.15),
        name="bigru_encoder",
    )(embedded)
    recurrent_representation = layers.Dense(
        128,
        activation="relu",
        name="recurrent_representation",
    )(encoded)
    ngram_representation = layers.Dense(
        128,
        activation="relu",
        name="ngram_representation",
    )(ngram_vectorizer(prompt_input))
    shared = layers.Concatenate(name="text_representation")(
        [recurrent_representation, ngram_representation]
    )
    shared = layers.Dense(160, activation="relu", name="shared_representation")(shared)
    structured = layers.Dense(32, activation="relu", name="structured_projection")(
        structured_input
    )
    shared = layers.Concatenate(name="text_and_structure")([shared, structured])
    shared = layers.Dense(128, activation="relu", name="fused_representation")(shared)
    shared = layers.Dropout(0.2)(shared)

    outputs = {
        "workload_type": layers.Dense(
            len(label_maps.workload_type),
            activation="softmax",
            name="workload_type",
        )(shared),
        "intent": layers.Dense(
            len(label_maps.intent),
            activation="softmax",
            name="intent",
        )(shared),
        "complexity": layers.Dense(
            len(label_maps.complexity),
            activation="softmax",
            name="complexity",
        )(shared),
        "effort_bucket": layers.Dense(
            len(label_maps.effort_bucket),
            activation="softmax",
            name="effort_bucket",
        )(shared),
        "effort_ordinal": layers.Dense(
            max(1, len(label_maps.effort_bucket) - 1),
            activation="sigmoid",
            name="effort_ordinal",
        )(shared),
        "effort_log_minutes": layers.Dense(1, name="effort_log_minutes")(shared),
        "needs_clarification": layers.Dense(
            1,
            activation="sigmoid",
            name="needs_clarification",
        )(shared),
    }

    model = tf.keras.Model(
        inputs=[prompt_input, structured_input],
        outputs=outputs,
        name="trendmodel_text_multitask",
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss={
            "workload_type": tf.keras.losses.SparseCategoricalCrossentropy(),
            "intent": tf.keras.losses.SparseCategoricalCrossentropy(),
            "complexity": tf.keras.losses.SparseCategoricalCrossentropy(),
            "effort_bucket": tf.keras.losses.SparseCategoricalCrossentropy(),
            "effort_ordinal": tf.keras.losses.BinaryCrossentropy(),
            "effort_log_minutes": tf.keras.losses.Huber(),
            "needs_clarification": tf.keras.losses.BinaryCrossentropy(),
        },
        loss_weights={
            "workload_type": 1.0,
            "intent": 1.0,
            "complexity": 0.8,
            "effort_bucket": 0.8,
            "effort_ordinal": 0.8,
            "effort_log_minutes": 1.0,
            "needs_clarification": 0.7,
        },
        weighted_metrics={
            "workload_type": ["accuracy"],
            "intent": ["accuracy"],
            "complexity": ["accuracy"],
            "effort_bucket": ["accuracy"],
            "effort_ordinal": [tf.keras.metrics.BinaryAccuracy(name="accuracy")],
            "effort_log_minutes": [tf.keras.metrics.MeanAbsoluteError(name="mae")],
            "needs_clarification": [tf.keras.metrics.BinaryAccuracy(name="accuracy")],
        },
    )
    return model
