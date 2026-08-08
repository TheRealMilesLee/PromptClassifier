from __future__ import annotations

import unittest

from trendmodel.tf_data import TaskRecord
from trendmodel.tf_train import evaluate_tensorflow_model


def _prediction(
    *,
    workload: str,
    intent: str,
    complexity: str,
    needs_clarification: bool = False,
) -> dict[str, object]:
    return {
        "predictions": {
            "workload_type": {"label": workload},
            "intent": {"label": intent},
            "complexity": {"label": complexity},
        },
        "effort": {
            "bucket": "M",
            "estimated_minutes": 90,
        },
        "needs_clarification": needs_clarification,
    }


class _FakeTensorFlowModel:
    def __init__(self) -> None:
        self.fused_calls = 0
        self.raw_calls = 0

    def predict_many(self, prompts: list[str]) -> list[dict[str, object]]:
        self.fused_calls += 1
        return [
            _prediction(
                workload="code" if "写" in prompt else "text",
                intent="create" if "写" in prompt else "explain",
                complexity="medium" if "写" in prompt else "low",
            )
            for prompt in prompts
        ]

    def predict_raw_many(
        self,
        prompts: list[str],
        *,
        batch_size: int = 128,
    ) -> list[dict[str, object]]:
        self.raw_calls += 1
        return [
            _prediction(workload="text", intent="explain", complexity="low")
            for _ in prompts
        ]


class TensorFlowEvaluationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.records = [
            TaskRecord(
                original_prompt="写代码",
                workload_type="code",
                intent="create",
                complexity="medium",
                effort_minutes=90,
                effort_bucket="M",
                needs_clarification=False,
                label_source="manual_gold",
            ),
            TaskRecord(
                original_prompt="解释概念",
                workload_type="text",
                intent="explain",
                complexity="low",
                effort_minutes=None,
                effort_bucket=None,
                needs_clarification=False,
                label_source="manual_gold",
            ),
        ]

    def test_raw_and_fused_modes_are_separate(self) -> None:
        model = _FakeTensorFlowModel()

        raw = evaluate_tensorflow_model(model, self.records, mode="raw")
        fused = evaluate_tensorflow_model(model, self.records, mode="fused")

        self.assertEqual(raw["mode"], "raw")
        self.assertEqual(fused["mode"], "fused")
        self.assertEqual(model.raw_calls, 1)
        self.assertEqual(model.fused_calls, 1)
        self.assertIn("needs_clarification", raw["heads"])
        self.assertEqual(raw["heads"]["workload_type"]["accuracy"], 0.5)
        self.assertEqual(fused["heads"]["workload_type"]["accuracy"], 1.0)

    def test_unknown_mode_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            evaluate_tensorflow_model(_FakeTensorFlowModel(), self.records, mode="all")


if __name__ == "__main__":
    unittest.main()
