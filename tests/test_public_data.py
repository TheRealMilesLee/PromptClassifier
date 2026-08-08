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

    def test_xlam_row_preserves_tool_use_as_weak_supervision(self) -> None:
        record = normalize_public_row(
            "xlam",
            {
                "id": "xlam-1",
                "query": "查一下北京今天的天气",
                "tools": '[{"name":"get_weather"}]',
                "answers": '[{"name":"get_weather","arguments":{"city":"北京"}}]',
            },
            language_filter={"en"},
        )
        assert record is not None
        self.assertEqual(record["workload_type"], "operations")
        self.assertEqual(record["intent"], "execute")
        self.assertEqual(record["label_source"], "xlam_tool_use_weak")
        self.assertEqual(record["auxiliary_labels"]["tool_count"], 1)
        self.assertTrue(record["auxiliary_labels"]["requires_tools"])

    def test_swebench_row_is_code_modify_without_fake_effort(self) -> None:
        record = normalize_public_row(
            "swebench",
            {
                "instance_id": "demo-1",
                "problem_statement": "Fix the parser when an empty input is supplied.",
                "repo": "demo/project",
                "patch": "@@\n-old\n+new\n",
                "test_patch": "@@\n+assert result\n",
            },
            language_filter={"en"},
        )
        assert record is not None
        self.assertEqual(record["workload_type"], "code")
        self.assertEqual(record["intent"], "modify")
        self.assertEqual(record["auxiliary_labels"]["patch_lines"], 2)
        self.assertIsNone(record["complexity"])
        self.assertIsNone(record["effort_minutes"])

    def test_massive_row_keeps_locale_and_original_intent(self) -> None:
        record = normalize_public_row(
            "massive",
            {
                "id": 7,
                "locale": "zh-CN",
                "scenario": "weather",
                "intent": "weather_query",
                "utt": "北京今天下雨吗？",
            },
            language_filter={"zh"},
        )
        assert record is not None
        self.assertEqual(record["language"], "zh")
        self.assertEqual(record["workload_type"], "operations")
        self.assertEqual(record["intent"], "explain")
        self.assertEqual(record["auxiliary_labels"]["massive_intent"], "weather_query")

    def test_dolly_category_maps_to_generic_intent(self) -> None:
        record = normalize_public_row(
            "dolly",
            {
                "id": "dolly-1",
                "instruction": "Summarize the incident report in three bullets.",
                "category": "summarization",
            },
            language_filter={"en"},
        )
        assert record is not None
        self.assertEqual(record["workload_type"], "text")
        self.assertEqual(record["intent"], "summarize")
        self.assertEqual(record["auxiliary_labels"]["dolly_category"], "summarization")

    def test_code_feedback_is_code_creation_without_fake_effort(self) -> None:
        record = normalize_public_row(
            "code_feedback",
            {
                "id": 12,
                "messages": [
                    {"role": "user", "content": "Write a Rust parser for this format."},
                    {"role": "assistant", "content": "Here is an implementation."},
                ],
            },
            language_filter={"en"},
        )
        assert record is not None
        self.assertEqual(record["original_prompt"], "Write a Rust parser for this format.")
        self.assertEqual(record["workload_type"], "code")
        self.assertEqual(record["intent"], "create")
        self.assertIsNone(record["effort_minutes"])

    def test_mtop_maps_task_intent_and_keeps_domain(self) -> None:
        record = normalize_public_row(
            "mtop",
            {
                "idx": "mtop-1",
                "question": "Please send a message to my family.",
                "intent": "IN:SEND_MESSAGE",
                "domain": "messaging",
                "lang": "en_US",
            },
            language_filter={"en"},
        )

        assert record is not None
        self.assertEqual(record["workload_type"], "operations")
        self.assertEqual(record["intent"], "execute")
        self.assertEqual(record["auxiliary_labels"]["mtop_domain"], "messaging")
        self.assertEqual(record["label_source"], "mtop_intent_weak")

    def test_coig_keeps_chinese_coverage_unlabeled(self) -> None:
        record = normalize_public_row(
            "coig",
            {"id": "coig-1", "instruction": "解释一下 DNS 缓存污染。"},
            language_filter={"zh"},
        )
        assert record is not None
        self.assertEqual(record["language"], "zh")
        self.assertIsNone(record["workload_type"])
        self.assertIsNone(record["intent"])
        self.assertEqual(
            record["auxiliary_labels"]["label_scope"],
            "chinese_instruction_unlabeled",
        )

    def test_helpful_instructions_adds_english_coverage_without_fake_labels(self) -> None:
        record = normalize_public_row(
            "helpful_instructions",
            {"id": "helpful-1", "instruction": "Explain how DNS caching works."},
            language_filter={"en"},
        )
        assert record is not None
        self.assertEqual(record["language"], "en")
        self.assertIsNone(record["workload_type"])
        self.assertIsNone(record["intent"])
        self.assertEqual(
            record["auxiliary_labels"]["label_scope"],
            "english_instruction_unlabeled",
        )

    def test_oasst1_exports_only_prompter_messages(self) -> None:
        record = normalize_public_row(
            "oasst1",
            {"message_id": "m-1", "role": "prompter", "lang": "zh", "text": "解释一下 DNS"},
            language_filter={"zh"},
        )
        self.assertIsNotNone(record)
        self.assertIsNone(
            normalize_public_row(
                "oasst1",
                {"message_id": "m-2", "role": "assistant", "lang": "zh", "text": "回答"},
                language_filter={"zh"},
            )
        )


if __name__ == "__main__":
    unittest.main()
