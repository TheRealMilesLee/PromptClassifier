from __future__ import annotations

import unittest

from trendmodel.public_data import apply_baseline_pseudo_labels, normalize_public_row
from trendmodel.tf_data import TaskRecord


class _FakeTeacher:
    def predict(self, prompt: str) -> dict[str, object]:
        return {
            "work_type": {"label": "code", "confidence": 0.8},
            "intent": {"label": "debug", "confidence": 0.7},
            "complexity": {"label": "high", "confidence": 0.6},
            "effort": {"label": "XL", "confidence": 0.99},
        }


class PublicDataTest(unittest.TestCase):
    def test_first_user_prompt_and_arena_weak_label(self) -> None:
        record = normalize_public_row(
            "arena",
            {
                "id": "arena-1",
                "evaluation_session_id": "session-1",
                "language": "Chinese",
                "conversation_a": [
                    {"role": "user", "content": "写一个 Python 函数"},
                    {"role": "assistant", "content": "不要被导出"},
                ],
                "is_code": True,
                "category_tag": ["coding", "hard_prompts"],
                "winner": "model_a",
            },
        )
        assert record is not None
        self.assertEqual(record["original_prompt"], "写一个 Python 函数")
        self.assertEqual(record["workload_type"], "code")
        self.assertEqual(record["split_group"], "arena:session-1")
        self.assertEqual(record["auxiliary_labels"]["preference_label"], "model_a")
        self.assertIsNone(record["effort_minutes"])

    def test_public_record_does_not_turn_missing_effort_into_a_target(self) -> None:
        record = normalize_public_row(
            "wildchat",
            {
                "conversation_hash": "wild-1",
                "language": "English",
                "conversation": [{"role": "user", "content": "Explain this error."}],
            },
        )
        assert record is not None
        parsed = TaskRecord.from_mapping(record)
        self.assertIsNone(parsed.effort_minutes)
        self.assertFalse(parsed.effort_supervised)
        self.assertFalse(parsed.observed_effort)
        self.assertIsNone(parsed.intent)

    def test_redaction_and_pseudo_labels_never_fill_effort(self) -> None:
        record = normalize_public_row(
            "lmsys",
            {
                "conversation_id": "lmsys-1",
                "language": "en",
                "conversation": [
                    {
                        "role": "user",
                        "content": "Send to a@example.com using api_key=secret-value-123456.",
                    }
                ],
            },
        )
        assert record is not None
        self.assertNotIn("a@example.com", record["original_prompt"])
        labelled = apply_baseline_pseudo_labels([record], _FakeTeacher())[0]
        self.assertEqual(labelled["workload_type"], "code")
        self.assertEqual(labelled["intent"], "debug")
        self.assertEqual(labelled["complexity"], "high")
        self.assertIsNone(labelled["effort_minutes"])


if __name__ == "__main__":
    unittest.main()
