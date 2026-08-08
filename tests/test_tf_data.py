from __future__ import annotations

import unittest
from pathlib import Path

from trendmodel.tf_data import (
    TaskRecord,
    build_label_maps,
    label_balance_weights,
    load_task_records,
    split_records,
    replay_supervision_weight,
)


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
        self.assertEqual(maps.effort_bucket, ("XS", "S", "M", "L", "XL"))
        first_train, first_validation = split_records(self.records, seed=9)
        second_train, second_validation = split_records(self.records, seed=9)
        self.assertEqual(first_train, second_train)
        self.assertEqual(first_validation, second_validation)

    def test_balance_weights_raise_rare_supervised_labels_without_exploding(self) -> None:
        maps = build_label_maps(self.records)
        weights = label_balance_weights(self.records, maps)

        self.assertGreater(weights["intent"]["translate"], weights["intent"]["create"])
        self.assertLessEqual(max(weights["effort_bucket"].values()), 3.0)
        self.assertGreater(weights["effort_bucket"]["XS"], 0.0)

    def test_bucket_only_effort_is_not_observed_minutes(self) -> None:
        record = TaskRecord.from_mapping(
            {
                "original_prompt": "请总结这份文档",
                "effort_bucket": "M",
                "label_source": "manual_bucket",
            }
        )

        self.assertTrue(record.effort_bucket_supervised)
        self.assertTrue(record.effort_supervised)
        self.assertFalse(record.effort_minutes_supervised)
        self.assertFalse(record.observed_effort)
        self.assertEqual(record.supervision_weight("effort_log_minutes"), 0.0)

        self.assertEqual(
            replay_supervision_weight(record, "effort_bucket", multiplier=8.0),
            8.0,
        )

    def test_weak_labels_are_not_gold_replayed(self) -> None:
        record = TaskRecord.from_mapping(
            {
                "original_prompt": "Explain the error",
                "workload_type": "text",
                "intent": "explain",
                "complexity": "low",
                "label_source": "task_tagger_pseudo",
                "label_confidence": 0.6,
            }
        )

        self.assertEqual(
            replay_supervision_weight(record, "intent", multiplier=8.0),
            0.6,
        )

    def test_observed_minutes_supervise_numeric_effort(self) -> None:
        record = TaskRecord.from_mapping(
            {
                "original_prompt": "重构模块并补充测试",
                "active_minutes": 120,
                "effort_bucket": "M",
                "label_source": "manual_gold",
            }
        )

        self.assertTrue(record.effort_bucket_supervised)
        self.assertTrue(record.effort_minutes_supervised)
        self.assertEqual(record.supervision_weight("effort_log_minutes"), 1.0)


if __name__ == "__main__":
    unittest.main()
