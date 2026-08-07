"""CLI for training and evaluating the local TensorFlow model."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

from .tf_data import build_label_maps, load_task_records, make_dataset, split_records
from .tf_inference import TensorFlowTaskModel
from .tf_model import build_text_model, require_tensorflow


def train_model(
    data_path: str | Path,
    artifact_dir: str | Path,
    *,
    epochs: int = 20,
    batch_size: int = 16,
    validation_ratio: float = 0.2,
    seed: int = 42,
    require_gpu: bool = False,
) -> dict[str, object]:
    tf = require_tensorflow()
    physical_devices = [device.name for device in tf.config.list_physical_devices()]
    gpu_devices = [device.name for device in tf.config.list_physical_devices("GPU")]
    if require_gpu and not gpu_devices:
        raise RuntimeError(
            "TensorFlow did not detect a GPU/Metal device; rerun outside a restricted "
            "environment or omit --require-gpu to use CPU fallback"
        )
    tf.keras.utils.set_random_seed(seed)

    records = load_task_records(data_path)
    supervision_counts = {
        "workload_type": sum(record.workload_type is not None for record in records),
        "intent": sum(record.intent is not None for record in records),
        "complexity": sum(record.complexity is not None for record in records),
        "effort": sum(record.effort_supervised for record in records),
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
    train_dataset = make_dataset(train_records, label_maps, batch_size=batch_size, shuffle=True)
    validation_dataset = make_dataset(validation_records, label_maps, batch_size=batch_size)
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
    metadata = {
        "model_id": "trendmodel-text-v1",
        "model_version": "0.1.1",
        "architecture": "TextVectorization(character) + Embedding + BiGRU + multi-task heads + structured evidence fusion",
        "runtime": {
            "tensorflow_version": str(tf.__version__),
            "physical_devices": physical_devices,
            "gpu_devices": gpu_devices,
            "accelerator": "metal" if gpu_devices else "cpu",
        },
        "labels": label_maps.to_dict(),
        "training_samples": len(train_records),
        "validation_samples": len(validation_records),
        "supervision_counts": supervision_counts,
        "observed_effort_samples": supervision_counts["observed_effort"],
        "effort_supervision_samples": supervision_counts["effort"],
        "label_source_counts": dict(Counter(record.label_source for record in records)),
        "source_counts": dict(Counter(record.source for record in records)),
        "warnings": _training_warnings(supervision_counts),
        "data_path": str(data_path),
        "epochs_requested": epochs,
        "epochs_completed": len(history.history.get("loss", [])),
        "validation_metrics": {key: float(value) for key, value in evaluation.items()},
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return metadata


def evaluate_tensorflow_model(
    model: TensorFlowTaskModel,
    records: Iterable[Any],
) -> dict[str, Any]:
    """Evaluate classification heads and effort on available labels.

    Missing labels are excluded per head.  This is important for a merged
    corpus where public prompts can supervise workload/intent through a
    teacher but cannot supervise effort.  Metrics are intentionally kept in
    this dependency-free module instead of requiring scikit-learn.
    """

    records = list(records)
    predictions = [model.predict(record.original_prompt) for record in records]
    metrics: dict[str, Any] = {
        "samples": len(records),
        "heads": {},
        "effort": {},
        "by_source": {},
        "label_quality": {},
    }
    for field in ("workload_type", "intent", "complexity"):
        actual: list[str] = []
        predicted: list[str] = []
        for record, prediction in zip(records, predictions):
            label = getattr(record, field)
            if label is None:
                continue
            actual.append(str(label))
            predicted.append(_prediction_label(prediction, field))
        metrics["heads"][field] = _classification_metrics(actual, predicted)

    effort_actual: list[float] = []
    effort_predicted: list[float] = []
    bucket_actual: list[str] = []
    bucket_predicted: list[str] = []
    observed_actual: list[float] = []
    observed_predicted: list[float] = []
    for record, prediction in zip(records, predictions):
        if not record.effort_supervised or prediction["effort"].get("estimated_minutes") is None:
            continue
        actual_minutes = float(record.effort_minutes)
        predicted_minutes = float(prediction["effort"]["estimated_minutes"])
        effort_actual.append(actual_minutes)
        effort_predicted.append(predicted_minutes)
        bucket_actual.append(str(record.effort_bucket))
        bucket_predicted.append(str(prediction["effort"].get("bucket")))
        if record.observed_effort:
            observed_actual.append(actual_minutes)
            observed_predicted.append(predicted_minutes)
    metrics["effort"] = {
        "supervised_samples": len(effort_actual),
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
    if supervision_counts.get("observed_effort", 0) == 0 and supervision_counts.get("effort", 0) > 0:
        warnings.append(
            "effort 只有 bucket/种子标签，没有 observed active_minutes；输出只能视为粗粒度先验"
        )
    if supervision_counts.get("effort", 0) == 0:
        warnings.append("没有 effort supervision；推理时 effort 将输出 unknown")
    return warnings


def _prediction_label(prediction: Mapping[str, Any], field: str) -> str:
    nested = prediction.get("predictions", {})
    if isinstance(nested, Mapping) and isinstance(nested.get(field), Mapping):
        return str(nested[field].get("label", "unknown"))
    return str(prediction.get(field, "unknown"))


def _classification_metrics(actual: list[str], predicted: list[str]) -> dict[str, Any]:
    return {
        "samples": len(actual),
        "accuracy": round(_accuracy(actual, predicted), 4),
        "macro_f1": round(_macro_f1(actual, predicted), 4),
        "actual_distribution": dict(Counter(actual)),
        "predicted_distribution": dict(Counter(predicted)),
    }


def _evaluate_indexes(records: list[Any], predictions: list[Mapping[str, Any]], indexes: list[int]) -> dict[str, Any]:
    result: dict[str, Any] = {"samples": len(indexes), "heads": {}}
    for field in ("workload_type", "intent", "complexity"):
        actual: list[str] = []
        predicted: list[str] = []
        for index in indexes:
            label = getattr(records[index], field)
            if label is None:
                continue
            actual.append(str(label))
            predicted.append(_prediction_label(predictions[index], field))
        result["heads"][field] = _classification_metrics(actual, predicted)
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
        "--require-gpu",
        action="store_true",
        help="fail instead of silently falling back to CPU when no GPU/Metal device is visible",
    )

    predict = subparsers.add_parser("predict", help="run one local TensorFlow prediction")
    predict.add_argument("--artifact", default="artifacts/trendmodel-text-v1")
    predict.add_argument("text", nargs="+")

    evaluate = subparsers.add_parser("evaluate", help="evaluate a saved local TensorFlow model")
    evaluate.add_argument("--data", default="data/examples.jsonl")
    evaluate.add_argument("--artifact", default="artifacts/trendmodel-text-v1")

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
        )
        print(json.dumps(metadata, ensure_ascii=False, indent=2))
        return 0

    if args.command == "predict":
        model = TensorFlowTaskModel.load(args.artifact)
        print(json.dumps(model.predict(" ".join(args.text)), ensure_ascii=False, indent=2))
        return 0

    model = TensorFlowTaskModel.load(args.artifact)
    records = load_task_records(args.data)
    print(json.dumps(evaluate_tensorflow_model(model, records), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
