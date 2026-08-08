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
        self.inference_calls = 0
        self.raw_calls = 0

    def predict_many(self, prompts: list[str]) -> list[dict[str, object]]:
        self.inference_calls += 1
        return [
            _prediction(
                workload="text",
                intent="explain",
                complexity="low",
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
                observed_effort=True,
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
            TaskRecord(
                original_prompt="总结文档",
                workload_type="text",
                intent="summarize",
                complexity="low",
                effort_minutes=30,
                effort_bucket="S",
                needs_clarification=False,
                observed_effort=False,
                label_source="manual_bucket",
            ),
        ]

    def test_raw_and_inference_modes_share_probabilities(self) -> None:
        model = _FakeTensorFlowModel()

        raw = evaluate_tensorflow_model(model, self.records, mode="raw")
        inference = evaluate_tensorflow_model(model, self.records, mode="inference")

        self.assertEqual(raw["mode"], "raw")
        self.assertEqual(inference["mode"], "inference")
        self.assertEqual(model.raw_calls, 1)
        self.assertEqual(model.inference_calls, 1)
        self.assertIn("needs_clarification", raw["heads"])
        self.assertEqual(raw["heads"]["workload_type"]["accuracy"], 0.6667)
        self.assertEqual(inference["heads"]["workload_type"]["accuracy"], 0.6667)
        self.assertEqual(raw["effort"]["supervised_samples"], 1)
        self.assertEqual(raw["effort"]["bucket_supervised_samples"], 2)
        self.assertEqual(raw["effort"]["observed_samples"], 1)

    def test_legacy_fused_mode_is_an_inference_alias(self) -> None:
        result = evaluate_tensorflow_model(_FakeTensorFlowModel(), self.records, mode="fused")
        self.assertEqual(result["mode"], "inference")

    def test_unknown_mode_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            evaluate_tensorflow_model(_FakeTensorFlowModel(), self.records, mode="all")


if __name__ == "__main__":
    unittest.main()
