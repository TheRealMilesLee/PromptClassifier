from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .evaluation import evaluate_predictions, holdout_evaluate
from .model import TaskModel, load_examples


def _default_data_path() -> Path:
    return Path(__file__).resolve().parent.parent / "data" / "examples.jsonl"


def _load_or_train(model_path: str | None, data_path: str) -> TaskModel:
    if model_path:
        return TaskModel.load(model_path)
    return TaskModel.fit_jsonl(data_path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="trendmodel",
        description="Classify task intent, work type, complexity and rough effort.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    predict = subparsers.add_parser("predict", help="classify one text request")
    predict.add_argument("text", nargs="*", help="request text; read stdin when omitted")
    predict.add_argument("--model", help="path to a trained model JSON")
    predict.add_argument("--data", default=str(_default_data_path()), help="JSONL seed/training data")
    predict.add_argument("--compact", action="store_true", help="print compact JSON")

    tag = subparsers.add_parser("tag", help="generate the embedding/evidence Task IR")
    tag.add_argument("text", nargs="*", help="request text; read stdin when omitted")
    tag.add_argument(
        "--embedding-backend",
        choices=("hash", "sentence-transformers"),
        default="hash",
        help="hash is offline fallback; sentence-transformers is recommended for production",
    )
    tag.add_argument("--embedding-model", default="BAAI/bge-small-zh-v1.5")
    tag.add_argument("--hash-dimension", type=int, default=384)
    tag.add_argument("--context-json", help="optional JSON object containing router context/prior")
    tag.add_argument("--compact", action="store_true", help="print compact JSON")

    train = subparsers.add_parser("train", help="train and save a model")
    train.add_argument("--data", default=str(_default_data_path()), help="JSONL training data")
    train.add_argument("--output", default="model.json", help="output model JSON")

    evaluate = subparsers.add_parser("evaluate", help="evaluate a model or a deterministic holdout")
    evaluate.add_argument("--data", default=str(_default_data_path()), help="JSONL training data")
    evaluate.add_argument("--model", help="evaluate an existing model on all examples")
    evaluate.add_argument("--test-ratio", type=float, default=0.25)
    evaluate.add_argument("--seed", type=int, default=42)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.command == "tag":
        from .tag_cli import tag_prompt

        text = " ".join(args.text).strip() if args.text else sys.stdin.read().strip()
        if not text:
            raise SystemExit("tag requires text or non-empty stdin")
        context: dict[str, object] = {}
        if args.context_json:
            try:
                parsed_context = json.loads(args.context_json)
            except json.JSONDecodeError as error:
                raise SystemExit(f"--context-json must be valid JSON: {error}") from error
            if not isinstance(parsed_context, dict):
                raise SystemExit("--context-json must contain a JSON object")
            context = parsed_context
        result = tag_prompt(
            text,
            embedding_backend=args.embedding_backend,
            embedding_model=args.embedding_model,
            hash_dimension=args.hash_dimension,
            context=context,
        )
        print(json.dumps(result, ensure_ascii=False, indent=None if args.compact else 2))
        return 0

    if args.command == "predict":
        text = " ".join(args.text).strip() if args.text else sys.stdin.read().strip()
        if not text:
            raise SystemExit("predict requires text or non-empty stdin")
        model = _load_or_train(args.model, args.data)
        result = model.predict(text)
        print(json.dumps(result, ensure_ascii=False, indent=None if args.compact else 2))
        return 0

    if args.command == "train":
        model = TaskModel.fit_jsonl(args.data)
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        model.save(output)
        print(json.dumps({"model": str(output), "training_samples": len(load_examples(args.data))}, ensure_ascii=False))
        return 0

    if args.model:
        result = evaluate_predictions(TaskModel.load(args.model), load_examples(args.data))
    else:
        result = holdout_evaluate(load_examples(args.data), test_ratio=args.test_ratio, seed=args.seed)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0
