"""OpenAI-compatible HTTP adapter for the TensorFlow task model."""

from __future__ import annotations

import argparse
import json
import time
import uuid
from typing import Any

from .tf_inference import TensorFlowTaskModel


def create_app(artifact_dir: str):
    try:
        from fastapi import FastAPI, HTTPException
    except ImportError as error:  # pragma: no cover - depends on local environment
        raise RuntimeError(
            "FastAPI is not installed; install requirements-serving.txt first"
        ) from error

    classifier = TensorFlowTaskModel.load(artifact_dir)
    app = FastAPI(title="TrendModel OpenAI-compatible API", version="0.1.0")

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok", "model": classifier.model_id}

    @app.get("/v1/models")
    async def models() -> dict[str, Any]:
        now = int(time.time())
        return {
            "object": "list",
            "data": [
                {
                    "id": classifier.model_id,
                    "object": "model",
                    "created": now,
                    "owned_by": "trendmodel",
                }
            ],
        }

    @app.post("/v1/chat/completions")
    async def chat_completions(payload: dict[str, Any]) -> dict[str, Any]:
        if payload.get("stream"):
            raise HTTPException(status_code=400, detail="streaming is not supported in v1")
        try:
            prompt = extract_user_prompt(payload.get("messages"))
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        result = classifier.predict(prompt)
        content = json.dumps(result, ensure_ascii=False, separators=(",", ":"))
        model_id = str(payload.get("model") or classifier.model_id)
        prompt_tokens = len(prompt)
        completion_tokens = len(content)
        return {
            "id": f"chatcmpl-{uuid.uuid4().hex}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model_id,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": content,
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        }

    return app


def extract_user_prompt(messages: Any) -> str:
    if not isinstance(messages, list) or not messages:
        raise ValueError("messages must be a non-empty list")
    user_contents: list[str] = []
    for message in messages:
        if not isinstance(message, dict) or message.get("role") != "user":
            continue
        user_contents.append(_content_to_text(message.get("content")))
    prompt = "\n".join(content for content in user_contents if content)
    if not prompt.strip():
        raise ValueError("at least one non-empty user message is required")
    return prompt


def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_parts: list[str] = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                text_parts.append(str(part.get("text", "")))
            else:
                raise ValueError("v1 accepts text content only")
        return "".join(text_parts)
    raise ValueError("message content must be a string or text-part list")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m trendmodel.api")
    parser.add_argument("--artifact", default="artifacts/trendmodel-text-v1")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args(argv)
    try:
        import uvicorn
    except ImportError as error:  # pragma: no cover - depends on local environment
        raise SystemExit("uvicorn is not installed; install requirements-serving.txt first") from error
    uvicorn.run(create_app(args.artifact), host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
