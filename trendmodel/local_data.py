"""Read locally downloaded Hugging Face parquet shards.

Hugging Face is intentionally absent from this module's runtime contract.  A
user can download parquet files once with ``hf download`` and then run all
subsequent cleaning, labeling, training, and evaluation from local storage.
``pyarrow`` is imported lazily because the dependency-free unit tests do not
need it.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from .public_data import DATASET_IDS, normalize_public_row


def discover_parquet_files(raw_dir: str | Path, source: str) -> list[Path]:
    """Find parquet shards under ``raw_dir/source`` or a matching repo folder."""

    root = Path(raw_dir)
    source = str(source).strip().lower()
    if source not in DATASET_IDS:
        raise ValueError(f"unsupported local source: {source}")
    candidates = [
        root / source,
        root / DATASET_IDS[source].split("/", 1)[-1],
        root / DATASET_IDS[source].replace("/", "__"),
    ]
    files: set[Path] = set()
    for candidate in candidates:
        if candidate.exists():
            files.update(path for path in candidate.rglob("*.parquet") if path.is_file())
    if not files and root.exists() and any(root.glob("*.parquet")):
        # This fallback is useful when the user downloaded one repo directly
        # into raw_dir rather than creating a source-named subdirectory.
        files.update(path for path in root.glob("*.parquet") if path.is_file())
    return sorted(files)


def iter_parquet_rows(
    files: Iterable[str | Path],
    *,
    batch_size: int = 512,
) -> Iterable[Mapping[str, Any]]:
    """Yield rows from local parquet files in bounded memory batches."""

    try:
        import pyarrow.parquet as parquet
    except ImportError as error:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "读取本地 parquet 需要安装 requirements-data.txt（包含 pyarrow）"
        ) from error
    for file_value in files:
        parquet_file = parquet.ParquetFile(file_value)
        for batch in parquet_file.iter_batches(batch_size=batch_size):
            for row in batch.to_pylist():
                yield row


def prepare_local_corpora(
    raw_dir: str | Path,
    *,
    sources: Iterable[str] = ("wildchat", "lmsys", "arena"),
    limits: Mapping[str, int] | None = None,
    languages: set[str] | None = None,
    redact: bool = True,
    batch_size: int = 512,
) -> list[dict[str, Any]]:
    """Normalize local shards using the same prompt-only boundary as remote data."""

    selected = [str(source).strip().lower() for source in sources if str(source).strip()]
    invalid = sorted(set(selected) - set(DATASET_IDS))
    if invalid:
        raise ValueError(f"unsupported local sources: {', '.join(invalid)}")
    limits = (
        {"wildchat": 50_000, "lmsys": 50_000, "arena": 20_000}
        if limits is None
        else dict(limits)
    )

    records: list[dict[str, Any]] = []
    seen_prompt_hashes: set[str] = set()
    for source in selected:
        limit = int(limits.get(source, 0))
        if limit == 0:
            continue
        files = discover_parquet_files(raw_dir, source)
        if not files:
            raise FileNotFoundError(
                f"没有找到 {source} 的 parquet 文件，请检查本地目录：{Path(raw_dir) / source}"
            )
        accepted = 0
        for row_index, row in enumerate(iter_parquet_rows(files, batch_size=batch_size)):
            record = normalize_public_row(
                source,
                row,
                language_filter=languages,
                redact=redact,
                row_index=row_index,
            )
            if record is None:
                continue
            prompt_hash = hashlib.sha256(record["original_prompt"].encode("utf-8")).hexdigest()
            if prompt_hash in seen_prompt_hashes:
                continue
            seen_prompt_hashes.add(prompt_hash)
            records.append(record)
            accepted += 1
            if limit > 0 and accepted >= limit:
                break
    return records
