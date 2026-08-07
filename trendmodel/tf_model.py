"""Small Keras multi-task text model."""

from __future__ import annotations

from typing import Iterable

from .tf_data import LabelMaps, normalize_model_text


def require_tensorflow():
    try:
        import tensorflow as tf
    except ImportError as error:  # pragma: no cover - depends on local environment
        raise RuntimeError(
            "TensorFlow is not installed. On Apple Silicon run "
            "python3 -m pip install -r requirements-tensorflow-macos.txt"
        ) from error
    return tf


def build_text_model(
    training_texts: Iterable[str],
    label_maps: LabelMaps,
    *,
    max_tokens: int = 12_000,
    sequence_length: int = 192,
    embedding_dim: int = 96,
    recurrent_units: int = 64,
    learning_rate: float = 1e-3,
):
    """Build a CPU-friendly character-level BiGRU classifier.

    Character splitting is deliberate for the first Chinese/mixed-language
    prototype: it avoids requiring a separate word segmentation service.
    """

    tf = require_tensorflow()
    layers = tf.keras.layers

    prompt_input = layers.Input(
        shape=(),
        dtype=tf.string,
        name="original_prompt",
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
    shared = layers.Dense(128, activation="relu", name="shared_representation")(encoded)
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
        "effort_log_minutes": layers.Dense(1, name="effort_log_minutes")(shared),
        "needs_clarification": layers.Dense(
            1,
            activation="sigmoid",
            name="needs_clarification",
        )(shared),
    }

    model = tf.keras.Model(prompt_input, outputs, name="trendmodel_text_multitask")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss={
            "workload_type": tf.keras.losses.SparseCategoricalCrossentropy(),
            "intent": tf.keras.losses.SparseCategoricalCrossentropy(),
            "complexity": tf.keras.losses.SparseCategoricalCrossentropy(),
            "effort_log_minutes": tf.keras.losses.Huber(),
            "needs_clarification": tf.keras.losses.BinaryCrossentropy(),
        },
        loss_weights={
            "workload_type": 1.0,
            "intent": 1.0,
            "complexity": 0.8,
            "effort_log_minutes": 1.0,
            "needs_clarification": 0.7,
        },
        weighted_metrics={
            "workload_type": ["accuracy"],
            "intent": ["accuracy"],
            "complexity": ["accuracy"],
            "effort_log_minutes": [tf.keras.metrics.MeanAbsoluteError(name="mae")],
            "needs_clarification": [tf.keras.metrics.BinaryAccuracy(name="accuracy")],
        },
    )
    return model
