from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from trendmodel.evaluation import holdout_evaluate
from trendmodel.model import TaskModel, load_examples


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "examples.jsonl"


class TaskModelTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.examples = load_examples(DATA_PATH)
        cls.model = TaskModel.fit(cls.examples)

    def test_translation_is_text_and_low_effort(self) -> None:
        result = self.model.predict("请把下面的英文翻译成中文。")
        self.assertEqual(result["work_type"]["label"], "text")
        self.assertEqual(result["intent"]["label"], "translate")
        self.assertIn(result["effort"]["label"], {"XS", "S"})
        self.assertFalse(result["needs_clarification"])

    def test_production_dns_request_is_operations_and_expensive(self) -> None:
        result = self.model.predict("排查生产环境 Kubernetes 的 DNS 超时，并给出恢复和验证步骤。")
        self.assertEqual(result["work_type"]["label"], "operations")
        self.assertEqual(result["intent"]["label"], "debug")
        self.assertIn(result["complexity"]["label"], {"medium", "high"})
        self.assertIn(result["effort"]["label"], {"L", "XL"})
        self.assertTrue(result["signals"])

    def test_model_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.json"
            self.model.save(path)
            restored = TaskModel.load(path)
            text = "请评审这个 Terraform 变更的风险，并给出修改建议。"
            original = self.model.predict(text)
            loaded = restored.predict(text)
            for head in ("work_type", "intent", "complexity", "effort"):
                self.assertEqual(original[head]["label"], loaded[head]["label"])

            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertIn("vectorizer", payload)
            self.assertIn("classifiers", payload)

    def test_empty_text_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            self.model.predict("  ")

    def test_vague_request_is_not_overclassified(self) -> None:
        result = self.model.predict("帮我弄一下")
        self.assertTrue(result["needs_clarification"])
        self.assertEqual(result["work_type"]["label"], "unknown")
        self.assertEqual(result["intent"]["label"], "unknown")

    def test_holdout_evaluation_is_deterministic(self) -> None:
        first = holdout_evaluate(self.examples, test_ratio=0.25, seed=7)
        second = holdout_evaluate(self.examples, test_ratio=0.25, seed=7)
        self.assertEqual(first, second)
        self.assertEqual(first["train_samples"] + first["test_samples"], len(self.examples))


if __name__ == "__main__":
    unittest.main()
