"""CLI for training and evaluating the local TensorFlow model."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

from .tf_data import (
    build_label_maps,
    label_balance_weights,
    load_task_records,
    make_dataset,
    split_records,
)
from .tf_inference import TensorFlowTaskModel
from .tf_model import build_text_model, require_tensorflow
from .tf_features import STRUCTURED_FEATURE_NAMES
from .tf_runtime import inspect_tensorflow_runtime


def train_model(
    data_path: str | Path,
    artifact_dir: str | Path,
    *,
    epochs: int = 20,
    batch_size: int = 16,
    validation_ratio: float = 0.2,
    seed: int = 42,
    require_gpu: bool = False,
    gold_replay_multiplier: float = 8.0,
) -> dict[str, object]:
    tf = require_tensorflow()
    runtime = inspect_tensorflow_runtime(tf)
    gpu_devices = runtime["gpu_devices"]
    if require_gpu and not gpu_devices:
        raise RuntimeError(
            "TensorFlow did not detect a GPU/Metal/CUDA device; on Linux or WSL2 "
            "install requirements-tensorflow-cuda.txt and verify nvidia-smi, or "
            "rerun without --require-gpu to use CPU fallback"
        )
    if gold_replay_multiplier < 1.0:
        raise ValueError("gold_replay_multiplier must be at least 1")
    tf.keras.utils.set_random_seed(seed)

    records = load_task_records(data_path)
    supervision_counts = {
        "workload_type": sum(record.workload_type is not None for record in records),
        "intent": sum(record.intent is not None for record in records),
        "complexity": sum(record.complexity is not None for record in records),
        "effort_bucket": sum(record.effort_bucket_supervised for record in records),
        # ``effort`` is retained as a stable metadata key, but for new
        # artifacts it means real numeric-minute supervision.  Bucket-only
        # targets are counted separately above and never train the regressor.
        "effort": sum(record.effort_minutes_supervised for record in records),
        "effort_minutes": sum(record.effort_minutes_supervised for record in records),
        "observed_effort": sum(record.observed_effort for record in records),
        "needs_clarification": sum(record.needs_clarification is not None for record in records),
    }
    required_heads = ("workload_type", "intent", "complexity")
    missing_heads = [head for head in required_heads if supervision_counts[head] == 0]
    if missing_heads:
        raise ValueError(
            "training data has no labels for: "
            + ", ".join(missing_heads)
            + "; add internal annotations or pass --pseudo-label-data when preparing public data"
        )
    label_maps = build_label_maps(records)
    train_records, validation_records = split_records(
        records,
        validation_ratio=validation_ratio,
        seed=seed,
    )
    balance_weights = label_balance_weights(train_records, label_maps)
    train_dataset = make_dataset(
        train_records,
        label_maps,
        batch_size=batch_size,
        shuffle=True,
        balance_labels=True,
        gold_replay_multiplier=gold_replay_multiplier,
        seed=seed,
    )
    validation_dataset = make_dataset(
        validation_records,
        label_maps,
        batch_size=batch_size,
        seed=seed,
    )
    model = build_text_model(
        [record.original_prompt for record in train_records],
        label_maps,
    )

    output_dir = Path(artifact_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=4,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-5,
        ),
    ]
    history = model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=epochs,
        callbacks=callbacks,
        shuffle=False,
        verbose=2,
    )
    evaluation = model.evaluate(validation_dataset, return_dict=True, verbose=0)
    model.save(output_dir / "model.keras")
    warnings = _training_warnings(supervision_counts)
    metadata = {
        "model_id": "trendmodel-text-v1",
        "model_version": "0.3.0",
        "architecture": "TextVectorization(character) + Embedding + BiGRU + character n-gram TF-IDF branch + structured feature branch + multi-task heads (including effort bucket) + evidence fusion",
        "text_encoder": {
            "character_max_tokens": 12_000,
            "character_sequence_length": 192,
            "ngram_max_tokens": 24_000,
            "ngram_range": [2, 5],
        },
        "structured_feature_dim": len(STRUCTURED_FEATURE_NAMES),
        "structured_feature_names": list(STRUCTURED_FEATURE_NAMES),
        "runtime": runtime,
        "labels": label_maps.to_dict(),
        "training_samples": len(train_records),
        "validation_samples": len(validation_records),
        "supervision_counts": supervision_counts,
        "label_balance_weights": balance_weights,
        "gold_replay_multiplier": gold_replay_multiplier,
        "gold_replay_policy": (
            "trusted manual/gold/seed/observed labels receive the configured "
            "training sample-weight multiplier; validation is not replay-weighted"
        ),
        "observed_effort_samples": supervision_counts["observed_effort"],
        "effort_bucket_supervision_samples": supervision_counts["effort_bucket"],
        "effort_supervision_samples": supervision_counts["effort"],
        "effort_minutes_supervision_samples": supervision_counts["effort_minutes"],
        "label_source_counts": dict(Counter(record.label_source for record in records)),
        "source_counts": dict(Counter(record.source for record in records)),
        "warnings": warnings,
        "data_path": str(data_path),
        "epochs_requested": epochs,
        "epochs_completed": len(history.history.get("loss", [])),
        "validation_metrics": {key: float(value) for key, value in evaluation.items()},
        # Write a complete enough metadata file before loading the artifact
        # again.  TensorFlowTaskModel.load intentionally requires both files.
        "artifact_validation_metrics": None,
    }
    metadata_path = output_dir / "metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    saved_model = TensorFlowTaskModel.load(output_dir)
    artifact_evaluation = evaluate_tensorflow_model(
        saved_model,
        validation_records,
        mode="raw",
        batch_size=batch_size,
    )
    warnings.extend(_roundtrip_warnings(evaluation, artifact_evaluation))
    metadata["warnings"] = warnings
    metadata["artifact_validation_metrics"] = artifact_evaluation
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return metadata


def evaluate_tensorflow_model(
    model: TensorFlowTaskModel,
    records: Iterable[Any],
    *,
    mode: str = "fused",
    batch_size: int = 128,
) -> dict[str, Any]:
    """Evaluate classification heads and effort on available labels.

    Missing labels are excluded per head.  This is important for a merged
    corpus where public prompts can supervise workload/intent through a
    teacher but cannot supervise effort.  Metrics are intentionally kept in
    this dependency-free module instead of requiring scikit-learn.

    ``raw`` evaluates only the saved neural heads.  ``fused`` evaluates the
    downstream contract, including the structural TaskTagger.  They must be
    reported separately because public labels may themselves come from that
    teacher.
    """

    if mode not in {"raw", "fused"}:
        raise ValueError("mode must be 'raw' or 'fused'")
    records = list(records)
    prompts = [record.original_prompt for record in records]
    if mode == "raw":
        predictions = model.predict_raw_many(prompts, batch_size=batch_size)
    else:
        predictions = model.predict_many(prompts)
    metrics: dict[str, Any] = {
        "mode": mode,
        "samples": len(records),
        "heads": {},
        "effort": {},
        "by_source": {},
        "label_quality": {},
    }
    for field in (
        "workload_type",
        "intent",
        "complexity",
        "effort_bucket",
        "needs_clarification",
    ):
        actual: list[str] = []
        predicted: list[str] = []
        weights: list[float] = []
        for record, prediction in zip(records, predictions):
            label = getattr(record, field)
            if label is None:
                continue
            actual.append(str(label))
            predicted.append(str(_prediction_value(prediction, field)))
            weights.append(record.supervision_weight(field))
        metrics["heads"][field] = _classification_metrics(actual, predicted, weights)

    effort_actual: list[float] = []
    effort_predicted: list[float] = []
    bucket_actual: list[str] = []
    bucket_predicted: list[str] = []
    observed_actual: list[float] = []
    observed_predicted: list[float] = []
    for record, prediction in zip(records, predictions):
        if record.effort_bucket_supervised:
            bucket_actual.append(str(record.effort_bucket))
            bucket_predicted.append(str(prediction["effort"].get("bucket")))
        if not record.effort_minutes_supervised or prediction["effort"].get("estimated_minutes") is None:
            continue
        actual_minutes = float(record.effort_minutes)
        predicted_minutes = float(prediction["effort"]["estimated_minutes"])
        effort_actual.append(actual_minutes)
        effort_predicted.append(predicted_minutes)
        observed_actual.append(actual_minutes)
        observed_predicted.append(predicted_minutes)
    metrics["effort"] = {
        "supervised_samples": len(effort_actual),
        "bucket_supervised_samples": len(bucket_actual),
        "bucket_accuracy": _accuracy(bucket_actual, bucket_predicted),
        "mae_minutes": _mean_absolute_error(effort_actual, effort_predicted),
        "observed_samples": len(observed_actual),
        "observed_mae_minutes": _mean_absolute_error(observed_actual, observed_predicted),
    }

    source_groups: dict[str, list[int]] = defaultdict(list)
    quality_groups: dict[str, list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        source_groups[record.source].append(index)
        quality_groups[_label_quality(record.label_source)].append(index)
    for name, indexes in source_groups.items():
        metrics["by_source"][name] = _evaluate_indexes(records, predictions, indexes)
    for name, indexes in quality_groups.items():
        metrics["label_quality"][name] = _evaluate_indexes(records, predictions, indexes)
    return metrics


def _training_warnings(supervision_counts: Mapping[str, int]) -> list[str]:
    warnings: list[str] = []
    effort_bucket_samples = supervision_counts.get(
        "effort_bucket",
        supervision_counts.get("effort", 0),
    )
    if supervision_counts.get("observed_effort", 0) == 0 and effort_bucket_samples > 0:
        warnings.append(
            "effort 只有 bucket/种子标签，没有 observed active_minutes；输出只能视为粗粒度先验"
        )
    if effort_bucket_samples == 0:
        warnings.append("没有 effort supervision；推理时 effort 将输出 unknown")
    return warnings


def _roundtrip_warnings(
    fit_metrics: Mapping[str, Any],
    artifact_metrics: Mapping[str, Any],
) -> list[str]:
    """Warn when saved-model raw metrics drift from in-memory fit metrics."""

    warnings: list[str] = []
    heads = artifact_metrics.get("heads", {})
    for field in (
        "workload_type",
        "intent",
        "complexity",
        "effort_bucket",
        "needs_clarification",
    ):
        fit_key = f"{field}_accuracy"
        artifact_head = heads.get(field, {})
        fit_value = fit_metrics.get(fit_key)
        artifact_value = artifact_head.get("weighted_accuracy")
        if fit_value is None or artifact_value is None:
            continue
        if abs(float(fit_value) - float(artifact_value)) > 0.05:
            warnings.append(
                f"saved artifact raw {field} weighted accuracy differs from in-memory fit "
                f"({float(fit_value):.4f} vs {float(artifact_value):.4f}); inspect the round-trip evaluation"
            )
    return warnings


def _prediction_label(prediction: Mapping[str, Any], field: str) -> str:
    nested = prediction.get("predictions", {})
    if isinstance(nested, Mapping) and isinstance(nested.get(field), Mapping):
        return str(nested[field].get("label", "unknown"))
    return str(prediction.get(field, "unknown"))


def _prediction_value(prediction: Mapping[str, Any], field: str) -> Any:
    if field == "needs_clarification":
        return bool(prediction.get(field, False))
    if field == "effort_bucket":
        effort = prediction.get("effort", {})
        if isinstance(effort, Mapping):
            return str(effort.get("bucket", "unknown"))
        return "unknown"
    return _prediction_label(prediction, field)


def _classification_metrics(
    actual: list[str],
    predicted: list[str],
    weights: list[float] | None = None,
) -> dict[str, Any]:
    metrics = {
        "samples": len(actual),
        "accuracy": round(_accuracy(actual, predicted), 4),
        "macro_f1": round(_macro_f1(actual, predicted), 4),
        "actual_distribution": dict(Counter(actual)),
        "predicted_distribution": dict(Counter(predicted)),
    }
    if weights and len(weights) == len(actual) and sum(weights) > 0:
        metrics["weighted_accuracy"] = round(
            sum(weight * (left == right) for left, right, weight in zip(actual, predicted, weights))
            / sum(weights),
            4,
        )
    return metrics


def _evaluate_indexes(records: list[Any], predictions: list[Mapping[str, Any]], indexes: list[int]) -> dict[str, Any]:
    result: dict[str, Any] = {"samples": len(indexes), "heads": {}}
    for field in (
        "workload_type",
        "intent",
        "complexity",
        "effort_bucket",
        "needs_clarification",
    ):
        actual: list[str] = []
        predicted: list[str] = []
        weights: list[float] = []
        for index in indexes:
            label = getattr(records[index], field)
            if label is None:
                continue
            actual.append(str(label))
            predicted.append(str(_prediction_value(predictions[index], field)))
            weights.append(records[index].supervision_weight(field))
        result["heads"][field] = _classification_metrics(actual, predicted, weights)
    return result


def _label_quality(label_source: str) -> str:
    source = str(label_source).casefold()
    if "pseudo" in source or "weak" in source:
        return "pseudo_or_weak"
    if "manual" in source or "gold" in source or "seed" in source:
        return "gold_or_seed"
    return "unknown"


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


def _mean_absolute_error(actual: list[float], predicted: list[float]) -> float | None:
    if not actual:
        return None
    return round(sum(abs(left - right) for left, right in zip(actual, predicted)) / len(actual), 4)


def evaluate_saved_model(
    data_path: str | Path,
    artifact_dir: str | Path,
    *,
    split: str = "validation",
    mode: str = "both",
    validation_ratio: float = 0.2,
    seed: int = 42,
    batch_size: int = 128,
) -> dict[str, Any]:
    """Evaluate a saved artifact with an explicit data split and mode."""

    if split not in {"all", "train", "validation"}:
        raise ValueError("split must be 'all', 'train', or 'validation'")
    if mode not in {"raw", "fused", "both"}:
        raise ValueError("mode must be 'raw', 'fused', or 'both'")

    records = load_task_records(data_path)
    if split == "all":
        selected_records = records
    else:
        train_records, validation_records = split_records(
            records,
            validation_ratio=validation_ratio,
            seed=seed,
        )
        selected_records = train_records if split == "train" else validation_records

    model = TensorFlowTaskModel.load(artifact_dir)
    modes = ("raw", "fused") if mode == "both" else (mode,)
    evaluations = {
        current_mode: evaluate_tensorflow_model(
            model,
            selected_records,
            mode=current_mode,
            batch_size=batch_size,
        )
        for current_mode in modes
    }
    warnings = [
        "raw evaluates neural heads without structural evidence fusion",
        "fused includes TaskTagger structural evidence and is not an independent teacher-free metric",
    ]
    if split == "all":
        warnings.append("all includes training records; use split=validation for a holdout estimate")
    return {
        "data_path": str(data_path),
        "artifact": str(artifact_dir),
        "split": split,
        "total_samples": len(records),
        "evaluated_samples": len(selected_records),
        "validation_ratio": validation_ratio,
        "seed": seed,
        "evaluations": evaluations,
        "warnings": warnings,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m trendmodel.tf_train")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train = subparsers.add_parser("train", help="train a local TensorFlow model")
    train.add_argument("--data", default="data/examples.jsonl")
    train.add_argument("--artifact", default="artifacts/trendmodel-text-v1")
    train.add_argument("--epochs", type=int, default=20)
    train.add_argument("--batch-size", type=int, default=16)
    train.add_argument("--validation-ratio", type=float, default=0.2)
    train.add_argument("--seed", type=int, default=42)
    train.add_argument(
        "--gold-replay-multiplier",
        type=float,
        default=8.0,
        help="multiply trusted manual/gold/seed labels during training so weak public data cannot drown them",
    )
    train.add_argument(
        "--require-gpu",
        action="store_true",
        help="fail instead of silently falling back to CPU when no GPU/Metal/CUDA device is visible",
    )

    predict = subparsers.add_parser("predict", help="run one local TensorFlow prediction")
    predict.add_argument("--artifact", default="artifacts/trendmodel-text-v1")
    predict.add_argument("text", nargs="+")

    evaluate = subparsers.add_parser("evaluate", help="evaluate a saved local TensorFlow model")
    evaluate.add_argument("--data", default="data/examples.jsonl")
    evaluate.add_argument("--artifact", default="artifacts/trendmodel-text-v1")
    evaluate.add_argument(
        "--split",
        choices=("validation", "train", "all"),
        default="validation",
        help="evaluate the deterministic validation split by default; all includes training records",
    )
    evaluate.add_argument(
        "--mode",
        choices=("raw", "fused", "both"),
        default="both",
        help="raw neural heads, fused downstream contract, or both",
    )
    evaluate.add_argument("--validation-ratio", type=float, default=0.2)
    evaluate.add_argument("--seed", type=int, default=42)
    evaluate.add_argument("--batch-size", type=int, default=128)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "train":
        metadata = train_model(
            args.data,
            args.artifact,
            epochs=args.epochs,
            batch_size=args.batch_size,
            validation_ratio=args.validation_ratio,
            seed=args.seed,
            require_gpu=args.require_gpu,
            gold_replay_multiplier=args.gold_replay_multiplier,
        )
        print(json.dumps(metadata, ensure_ascii=False, indent=2))
        return 0

    if args.command == "predict":
        model = TensorFlowTaskModel.load(args.artifact)
        print(json.dumps(model.predict(" ".join(args.text)), ensure_ascii=False, indent=2))
        return 0

    print(
        json.dumps(
            evaluate_saved_model(
                args.data,
                args.artifact,
                split=args.split,
                mode=args.mode,
                validation_ratio=args.validation_ratio,
                seed=args.seed,
                batch_size=args.batch_size,
            ),
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
