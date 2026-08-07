"""Command-line entry point for the Task IR tagger."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping
from typing import Any

from .embedding import create_embedder
from .tagger import TaskTagger


def tag_prompt(
    prompt: str,
    *,
    embedding_backend: str = "hash",
    embedding_model: str = "BAAI/bge-small-zh-v1.5",
    hash_dimension: int = 384,
    context: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    embedder = create_embedder(
        embedding_backend,
        model_name=embedding_model,
        dimension=hash_dimension,
    )
    return TaskTagger(embedder).predict(prompt, context=context).to_dict()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m trendmodel.tag_cli",
        description="Generate a router-facing Task IR from one prompt.",
    )
    parser.add_argument("text", nargs="*", help="prompt text; read stdin when omitted")
    parser.add_argument(
        "--embedding-backend",
        choices=("hash", "sentence-transformers"),
        default="hash",
        help="hash is offline fallback; sentence-transformers is recommended for production",
    )
    parser.add_argument(
        "--embedding-model",
        default="BAAI/bge-small-zh-v1.5",
        help="SentenceTransformer model name when that backend is selected",
    )
    parser.add_argument("--hash-dimension", type=int, default=384)
    parser.add_argument("--context-json", help="optional JSON object containing router context/prior")
    parser.add_argument("--compact", action="store_true", help="print compact JSON")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    prompt = " ".join(args.text).strip() if args.text else sys.stdin.read().strip()
    if not prompt:
        raise SystemExit("a non-empty prompt is required")
    context = _parse_context(args.context_json)
    result = tag_prompt(
        prompt,
        embedding_backend=args.embedding_backend,
        embedding_model=args.embedding_model,
        hash_dimension=args.hash_dimension,
        context=context,
    )
    print(json.dumps(result, ensure_ascii=False, separators=(",", ":") if args.compact else None, indent=None if args.compact else 2))
    return 0


def _parse_context(value: str | None) -> dict[str, Any]:
    if not value:
        return {}
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as error:
        raise SystemExit(f"--context-json must be valid JSON: {error}") from error
    if not isinstance(parsed, dict):
        raise SystemExit("--context-json must contain a JSON object")
    return parsed


if __name__ == "__main__":
    raise SystemExit(main())
