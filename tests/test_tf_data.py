from __future__ import annotations

import unittest
from pathlib import Path

from trendmodel.tf_data import build_label_maps, load_task_records, split_records


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "examples.jsonl"


class TensorFlowDataTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.records = load_task_records(DATA_PATH)

    def test_legacy_seed_schema_is_supported(self) -> None:
        self.assertEqual(len(self.records), 48)
        self.assertEqual(self.records[0].original_prompt, "请写一封礼貌的邮件，向客户说明交付会延期两天。")
        self.assertGreater(self.records[0].effort_minutes, 0)
        self.assertFalse(self.records[0].observed_effort)

    def test_label_maps_and_split_are_deterministic(self) -> None:
        maps = build_label_maps(self.records)
        self.assertIn("code", maps.workload_type)
        self.assertIn("debug", maps.intent)
        self.assertIn("execute", maps.intent)
        first_train, first_validation = split_records(self.records, seed=9)
        second_train, second_validation = split_records(self.records, seed=9)
        self.assertEqual(first_train, second_train)
        self.assertEqual(first_validation, second_validation)


if __name__ == "__main__":
    unittest.main()
