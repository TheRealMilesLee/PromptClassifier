from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from trendmodel.data_cleaning import CleaningConfig, clean_records, read_jsonl, write_jsonl
from trendmodel.public_data import apply_tagger_pseudo_labels
from trendmodel.tagger import TaskTagger
from trendmodel.tf_data import TaskRecord


class DataCleaningTest(unittest.TestCase):
    def test_cleaning_redacts_normalizes_and_reports_duplicates(self) -> None:
        records, report = clean_records(
            [
                {
                    "source": "internal",
                    "original_prompt": "  写一个函数\r\n\r\n\r\n api_key = secret-value-123456  ",
                    "split_group": "session-1",
                },
                {
                    "source": "internal",
                    "original_prompt": "写一个函数 api_key=secret-value-123456",
                    "split_group": "session-2",
                },
                {"source": "internal", "original_prompt": "  "},
            ],
            config=CleaningConfig(min_non_whitespace_chars=3),
        )

        self.assertEqual(len(records), 1)
        self.assertNotIn("secret-value-123456", records[0]["original_prompt"])
        self.assertEqual(records[0]["split_group"], "session-1")
        self.assertEqual(report["input_rows"], 3)
        self.assertEqual(report["accepted_rows"], 1)
        self.assertEqual(report["duplicate_rows"], 1)
        self.assertEqual(report["reject_reasons"]["empty"], 1)
        self.assertEqual(report["redacted_rows"], 1)

    def test_pseudo_label_teacher_never_invents_effort(self) -> None:
        source = {
            "source": "wildchat",
            "original_prompt": "帮我实现一个 Python 多线程程序",
            "split_group": "wildchat:1",
            "effort_minutes": None,
            "effort_bucket": None,
        }
        labelled = apply_tagger_pseudo_labels([source], TaskTagger())[0]
        self.assertIn("workload_type", labelled)
        self.assertIn("intent", labelled)
        self.assertIn("complexity", labelled)
        self.assertIsNone(labelled["effort_minutes"])
        self.assertIsNone(labelled["effort_bucket"])
        self.assertIn("task_tagger_pseudo", labelled["label_source"])

        parsed = TaskRecord.from_mapping(labelled)
        self.assertFalse(parsed.effort_supervised)
        self.assertGreater(parsed.supervision_weight("workload_type"), 0.0)

    def test_jsonl_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "records.jsonl"
            rows = [{"original_prompt": "解释 DNS", "source": "test"}]
            self.assertEqual(write_jsonl(rows, path), 1)
            self.assertEqual(read_jsonl([path]), rows)
            self.assertEqual(json.loads(path.read_text(encoding="utf-8"))["source"], "test")

    def test_duplicate_label_conflicts_are_reported_without_leaking_prompt_text(self) -> None:
        records, report = clean_records(
            [
                {
                    "source": "source-a",
                    "original_prompt": "解释这个 API 设计",
                    "workload_type": "text",
                    "intent": "explain",
                },
                {
                    "source": "source-b",
                    "original_prompt": "解释这个 API 设计",
                    "workload_type": "code",
                    "intent": "create",
                },
            ]
        )

        self.assertEqual(len(records), 1)
        self.assertEqual(report["label_conflict_rows"], 1)
        self.assertEqual(report["label_conflict_groups"], 1)
        self.assertEqual(report["label_conflicts"]["fields"], {"intent": 1, "workload_type": 1})
        self.assertNotIn("解释这个 API 设计", json.dumps(report, ensure_ascii=False))
        self.assertEqual(len(report["label_conflicts"]["examples"]), 1)

    def test_duplicate_with_missing_label_is_not_a_conflict(self) -> None:
        _, report = clean_records(
            [
                {"source": "a", "original_prompt": "总结这份报告", "intent": "summarize"},
                {"source": "b", "original_prompt": "总结这份报告"},
            ]
        )

        self.assertEqual(report["duplicate_rows"], 1)
        self.assertEqual(report["label_conflict_rows"], 0)


if __name__ == "__main__":
    unittest.main()
