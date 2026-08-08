from __future__ import annotations

import gzip
import json
import tempfile
import unittest
from pathlib import Path

from trendmodel.local_data import discover_data_files, iter_local_rows


class LocalDataTest(unittest.TestCase):
    def test_discovers_and_reads_json_variants(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source_dir = Path(temporary) / "dolly"
            source_dir.mkdir()
            rows = [
                {"instruction": "one"},
                {"instruction": "two"},
            ]
            (source_dir / "rows.json").write_text(
                json.dumps(rows), encoding="utf-8"
            )
            (source_dir / "rows.jsonl").write_text(
                "\n".join(json.dumps(row) for row in rows) + "\n",
                encoding="utf-8",
            )
            with gzip.open(source_dir / "rows.json.gz", "wt", encoding="utf-8") as handle:
                json.dump(rows, handle)
            with gzip.open(source_dir / "rows.jsonl.gz", "wt", encoding="utf-8") as handle:
                handle.write("\n".join(json.dumps(row) for row in rows))

            files = discover_data_files(temporary, "dolly")
            self.assertEqual(len(files), 4)
            loaded = list(iter_local_rows(files))
            self.assertEqual(len(loaded), 8)
            self.assertTrue(all(row["instruction"] in {"one", "two"} for row in loaded))


if __name__ == "__main__":
    unittest.main()
