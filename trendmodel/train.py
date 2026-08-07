"""End-to-end data preparation and TensorFlow training workflow.

This is the project-facing entry point.  The legacy Naive Bayes CLI remains
available for backwards compatibility, while this module owns the intended
pipeline for the router tagger:

    prepare -> train -> evaluate

The default preparation path is local and dependency-free.  Hugging Face
streaming ingestion is optional and is only activated when ``--sources`` is
provided.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .data_cleaning import CleaningConfig, clean_records, read_jsonl, write_jsonl
from .local_data import prepare_local_corpora
from .public_data import (
    apply_tagger_pseudo_labels,
    normalize_languages,
    prepare_public_corpora,
)


def prepare_training_data(
    *,
    input_paths: list[str | Path] | None = None,
    public_sources: list[str] | None = None,
    local_raw_dir: str | Path | None = None,
    local_sources: list[str] | None = None,
    public_limits: dict[str, int] | None = None,
    languages: set[str] | None = None,
    split: str = "train",
    seed: int = 42,
    output_path: str | Path,
    report_path: str | Path,
    redact: bool = True,
    pseudo_label: bool = True,
    embedding_backend: str = "hash",
    embedding_model: str = "BAAI/bge-small-zh-v1.5",
    hash_dimension: int = 384,
    min_confidence: float = 0.0,
    keep_low_confidence: bool = True,
    min_non_whitespace_chars: int = 3,
    max_chars: int = 16_000,
) -> dict[str, Any]:
    """Merge, clean, weak-label, and write a training JSONL artifact."""

    raw_records: list[dict[str, Any]] = []
    if input_paths:
        raw_records.extend(read_jsonl(input_paths))
    selected_local_sources = [
        item.strip().lower() for item in (local_sources or []) if item.strip()
    ]
    if local_raw_dir and selected_local_sources:
        raw_records.extend(
            prepare_local_corpora(
                local_raw_dir,
                sources=selected_local_sources,
                limits=public_limits,
                languages=languages,
                redact=redact,
            )
        )
    selected_sources = [item.strip().lower() for item in (public_sources or []) if item.strip()]
    if selected_sources:
        raw_records.extend(
            prepare_public_corpora(
                sources=selected_sources,
                limits=public_limits,
                languages=languages,
                split=split,
                seed=seed,
                redact=redact,
            )
        )
    if not raw_records:
        raise ValueError("prepare requires --input JSONL or --sources")

    cleaned, report = clean_records(
        raw_records,
        config=CleaningConfig(
            min_non_whitespace_chars=min_non_whitespace_chars,
            max_chars=max_chars,
            redact=redact,
            deduplicate=True,
            allowed_languages=frozenset(languages) if languages else None,
        ),
    )

    labeling_report: dict[str, Any] = {
        "enabled": pseudo_label,
        "teacher": None,
        "records_with_pseudo_labels": 0,
        "pseudo_field_counts": {},
        "effort_fields_filled": 0,
        "warning": (
            "pseudo labels are weak supervision; they must not be used as gold evaluation labels"
            if pseudo_label
            else None
        ),
    }
    if pseudo_label and cleaned:
        from .embedding import create_embedder
        from .tagger import TaskTagger

        embedder = create_embedder(
            embedding_backend,
            model_name=embedding_model,
            dimension=hash_dimension,
        )
        tagger = TaskTagger(embedder)
        labelled = apply_tagger_pseudo_labels(
            cleaned,
            tagger,
            min_confidence=min_confidence,
            keep_low_confidence=keep_low_confidence,
        )
        pseudo_counts: dict[str, int] = {}
        records_with_pseudo = 0
        effort_filled = 0
        for before, after in zip(cleaned, labelled):
            fields = after.get("pseudo_label_fields", [])
            if fields:
                records_with_pseudo += 1
            for field in fields:
                pseudo_counts[field] = pseudo_counts.get(field, 0) + 1
            if before.get("effort_minutes") != after.get("effort_minutes"):
                effort_filled += 1
        cleaned = labelled
        labeling_report.update(
            {
                "teacher": getattr(tagger, "model_version", "unknown"),
                "records_with_pseudo_labels": records_with_pseudo,
                "pseudo_field_counts": dict(sorted(pseudo_counts.items())),
                "effort_fields_filled": effort_filled,
                "embedding_backend": embedding_backend,
            }
        )

    # This assertion is intentionally close to the write boundary.  If a
    # future teacher accidentally starts producing effort, preparation fails
    # loudly rather than silently creating false supervision.
    if labeling_report["effort_fields_filled"]:
        raise AssertionError("pseudo-label preparation must never fill effort fields")

    report["labeling"] = labeling_report
    report["output_path"] = str(output_path)
    report["output_rows"] = write_jsonl(cleaned, output_path)
    report_path = Path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m trendmodel.train",
        description="Prepare data and train the TensorFlow router tagger.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare", help="clean and weak-label JSONL/public data")
    prepare.add_argument("--input", nargs="+", default=[], help="one or more local JSONL files")
    prepare.add_argument(
        "--sources",
        default="",
        help="optional comma-separated Hugging Face sources: wildchat,lmsys,arena",
    )
    prepare.add_argument(
        "--local-raw-dir",
        help="read already downloaded parquet files from this local directory",
    )
    prepare.add_argument(
        "--local-sources",
        default="wildchat,arena",
        help="comma-separated local source folders under --local-raw-dir (add lmsys after approval)",
    )
    prepare.add_argument("--wildchat-limit", type=int, default=50_000, help="0 skips; -1 reads all local rows")
    prepare.add_argument("--lmsys-limit", type=int, default=50_000, help="0 skips; -1 reads all local rows")
    prepare.add_argument("--arena-limit", type=int, default=20_000, help="0 skips; -1 reads all local rows")
    prepare.add_argument("--languages", default="en,zh")
    prepare.add_argument("--split", default="train")
    prepare.add_argument("--seed", type=int, default=42)
    prepare.add_argument("--output", default="data/processed/prompts.cleaned.jsonl")
    prepare.add_argument("--report", default="data/processed/cleaning_report.json")
    prepare.add_argument("--no-redact", action="store_true")
    prepare.add_argument("--no-pseudo-label", action="store_true")
    prepare.add_argument(
        "--embedding-backend",
        choices=("hash", "sentence-transformers"),
        default="hash",
    )
    prepare.add_argument("--embedding-model", default="BAAI/bge-small-zh-v1.5")
    prepare.add_argument("--hash-dimension", type=int, default=384)
    prepare.add_argument("--min-confidence", type=float, default=0.0)
    prepare.add_argument("--drop-low-confidence", action="store_true")
    prepare.add_argument("--min-chars", type=int, default=3)
    prepare.add_argument("--max-chars", type=int, default=16_000)

    train = subparsers.add_parser("train", help="train and save a TensorFlow model")
    train.add_argument("--data", default="data/processed/prompts.cleaned.jsonl")
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

    evaluate = subparsers.add_parser("evaluate", help="evaluate a saved TensorFlow model")
    evaluate.add_argument("--data", default="data/processed/prompts.cleaned.jsonl")
    evaluate.add_argument("--artifact", default="artifacts/trendmodel-text-v1")

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "prepare":
        sources = [item.strip() for item in args.sources.split(",") if item.strip()]
        local_sources = [item.strip() for item in args.local_sources.split(",") if item.strip()]
        limits = {
            "wildchat": args.wildchat_limit,
            "lmsys": args.lmsys_limit,
            "arena": args.arena_limit,
        }
        report = prepare_training_data(
            input_paths=args.input,
            public_sources=sources,
            local_raw_dir=args.local_raw_dir,
            local_sources=local_sources,
            public_limits=limits,
            languages=normalize_languages(args.languages),
            split=args.split,
            seed=args.seed,
            output_path=args.output,
            report_path=args.report,
            redact=not args.no_redact,
            pseudo_label=not args.no_pseudo_label,
            embedding_backend=args.embedding_backend,
            embedding_model=args.embedding_model,
            hash_dimension=args.hash_dimension,
            min_confidence=args.min_confidence,
            keep_low_confidence=not args.drop_low_confidence,
            min_non_whitespace_chars=args.min_chars,
            max_chars=args.max_chars,
        )
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0

    from .tf_train import evaluate_tensorflow_model, train_model
    from .tf_data import load_task_records
    from .tf_inference import TensorFlowTaskModel

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

    model = TensorFlowTaskModel.load(args.artifact)
    metrics = evaluate_tensorflow_model(model, load_task_records(args.data))
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
