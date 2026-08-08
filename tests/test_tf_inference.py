from __future__ import annotations

import math
import unittest
from unittest.mock import patch

from trendmodel.ablation import evidence_only_prediction
from trendmodel.tf_inference import (
    TensorFlowTaskModel,
    _classification_payload,
    _effort_payload_from_raw,
    _ordinal_to_bucket_distribution,
)


class _FakeTensorFlow:
    float32 = object()

    @staticmethod
    def constant(value, dtype=None):
        return value


class _FakeKerasModel:
    output_names = [
        "workload_type",
        "intent",
        "complexity",
        "effort_bucket",
        "effort_log_minutes",
        "needs_clarification",
    ]

    def __init__(self) -> None:
        self.inputs = []

    def predict(self, inputs, **kwargs):
        self.inputs.append(inputs)
        return {
            "workload_type": [[0.8, 0.2]],
            "intent": [[0.7, 0.3]],
            "complexity": [[0.6, 0.4]],
            "effort_bucket": [[0.05, 0.10, 0.70, 0.10, 0.05]],
            "effort_log_minutes": [[math.log1p(90.0)]],
            "needs_clarification": [[0.1]],
        }


class TensorFlowInferenceContractTest(unittest.TestCase):
    def test_inference_returns_task_ast_evidence_without_post_hoc_fusion(self) -> None:
        model = _FakeKerasModel()
        metadata = {
            "model_id": "test-model",
            "model_version": "test",
            "task_ast_schema": "task-ast-v1",
            "structured_feature_dim": 56,
            "labels": {
                "workload_type": ["text", "code"],
                "intent": ["explain", "create"],
                "complexity": ["low", "high"],
                "effort_bucket": ["XS", "S", "M", "L", "XL"],
            },
            "effort_bucket_supervision_samples": 10,
            "observed_effort_samples": 0,
        }

        with patch("trendmodel.tf_inference.require_tensorflow", return_value=_FakeTensorFlow):
            classifier = TensorFlowTaskModel(model, metadata)
            result = classifier.predict("请实现一个 Python 程序，写入 ./output.py")

        self.assertEqual(result["workload_type"], "text")
        self.assertEqual(result["predictions"]["workload_type"]["probability"], 0.8)
        self.assertTrue(result["evidence"])
        self.assertEqual(result["task_ir"]["contract_version"], "task-ir-v1")
        self.assertEqual(result["task_ir"]["workload_type"]["label"], "text")
        self.assertIn("capability_demand", result["task_ir"])
        self.assertEqual(result["task_ir"]["fusion"]["post_hoc_probability_fusion"], False)
        self.assertEqual(result["fusion"]["task_ast_schema"], "task-ast-v1")
        self.assertFalse(result["fusion"]["post_hoc_probability_fusion"])
        self.assertNotIn("structural_teacher", result["fusion"])
        self.assertIn("structured_features", model.inputs[0])

    def test_raw_batch_and_inference_keep_the_same_head_probability(self) -> None:
        model = _FakeKerasModel()
        metadata = {
            "model_id": "test-model",
            "structured_feature_dim": 56,
            "labels": {
                "workload_type": ["text", "code"],
                "intent": ["explain", "create"],
                "complexity": ["low", "high"],
                "effort_bucket": ["XS", "S", "M", "L", "XL"],
            },
            "effort_bucket_supervision_samples": 10,
            "observed_effort_samples": 0,
        }

        with patch("trendmodel.tf_inference.require_tensorflow", return_value=_FakeTensorFlow):
            classifier = TensorFlowTaskModel(model, metadata)
            inference = classifier.predict("解释 ./output.py")
            raw = classifier.predict_raw_many(["解释 ./output.py"])[0]
            classifier.predict_raw_many(["解释 ./output.py"], feature_mode="text_only")

        self.assertEqual(
            inference["predictions"]["workload_type"]["probability"],
            raw["predictions"]["workload_type"]["probability"],
        )
        self.assertTrue(raw["evidence"])
        self.assertEqual(raw["task_ir"]["contract_version"], "task-ir-v1")
        self.assertFalse(raw["fusion"]["post_hoc_probability_fusion"])
        self.assertTrue(all(value == 0.0 for value in model.inputs[-1]["structured_features"][0]))

    def test_evidence_only_baseline_is_explicitly_rule_based(self) -> None:
        result = evidence_only_prediction(
            "请执行 kubectl delete pod api-123 -n production",
            workload_labels=["text", "code", "analysis", "operations", "planning", "mixed", "other"],
            intent_labels=[
                "create", "modify", "debug", "explain", "execute", "analyze", "review",
                "summarize", "translate", "plan", "extract", "classify", "unknown",
            ],
            complexity_labels=["low", "medium", "high"],
        )

        self.assertEqual(result["workload_type"], "operations")
        self.assertEqual(result["intent"], "execute")
        self.assertEqual(result["complexity"], "high")
        self.assertEqual(result["fusion"]["semantic_model"], "none")
        self.assertEqual(result["effort"]["label"], "unknown")

    def test_old_text_only_metadata_keeps_legacy_effort_fallback(self) -> None:
        payload = _effort_payload_from_raw(
            {"effort_log_minutes": [[math.log1p(90.0)]]},
            [0.8, 0.15, 0.05],
            {
                "effort_supervision_samples": 12,
                "observed_effort_samples": 0,
            },
            effort_bucket_labels=["XS", "S", "M", "L", "XL"],
        )

        self.assertEqual(payload["basis"], "bucket_seed_only")
        self.assertEqual(payload["label"], "M")
        self.assertLessEqual(payload["confidence"], 0.35)

    def test_new_bucket_head_reports_bucket_only_basis(self) -> None:
        payload = _effort_payload_from_raw(
            {
                "effort_bucket": [[0.02, 0.08, 0.75, 0.12, 0.03]],
                "effort_log_minutes": [[math.log1p(90.0)]],
            },
            [0.8, 0.15, 0.05],
            {
                "effort_bucket_supervision_samples": 48,
                "effort_supervision_samples": 0,
                "observed_effort_samples": 0,
            },
            effort_bucket_labels=["XS", "S", "M", "L", "XL"],
        )

        self.assertEqual(payload["basis"], "trained_effort_bucket_head")
        self.assertEqual(payload["label"], "M")
        self.assertEqual(payload["estimated_minutes"], 90)
        self.assertGreater(payload["probability"], 0.7)

    def test_observed_bucket_head_reports_numeric_basis(self) -> None:
        payload = _effort_payload_from_raw(
            {
                "effort_bucket": [[0.01, 0.03, 0.10, 0.80, 0.06]],
                "effort_log_minutes": [[math.log1p(300.0)]],
            },
            [0.1, 0.2, 0.7],
            {
                "effort_bucket_supervision_samples": 80,
                "effort_supervision_samples": 25,
                "effort_minutes_supervision_samples": 25,
                "observed_effort_samples": 25,
            },
            effort_bucket_labels=["XS", "S", "M", "L", "XL"],
        )

        self.assertEqual(payload["basis"], "trained_observed_effort_and_bucket_heads")
        self.assertEqual(payload["label"], "L")
        self.assertEqual(payload["estimated_minutes"], 300)

    def test_ordinal_effort_head_is_preferred_and_monotonic(self) -> None:
        distribution = _ordinal_to_bucket_distribution([0.9, 0.7, 0.8, 0.2])
        self.assertEqual(len(distribution), 5)
        self.assertAlmostEqual(sum(distribution), 1.0, places=6)
        self.assertTrue(all(value >= 0.0 for value in distribution))

        payload = _effort_payload_from_raw(
            {
                "effort_ordinal": [[0.8, 0.45, 0.3, 0.1]],
                "effort_log_minutes": [[math.log1p(300.0)]],
            },
            [0.1, 0.2, 0.7],
            {
                "effort_bucket_supervision_samples": 80,
                "effort_ordinal_supervision_samples": 80,
                "effort_supervision_samples": 25,
                "observed_effort_samples": 25,
            },
            effort_bucket_labels=["XS", "S", "M", "L", "XL"],
        )

        self.assertEqual(payload["basis"], "trained_observed_effort_and_ordinal_heads")
        self.assertEqual(payload["label"], "S")
        self.assertEqual(len(payload["candidates"]), 3)

    def test_classification_payload_exposes_uncertainty_diagnostics(self) -> None:
        payload = _classification_payload(
            "create",
            [0.51, 0.49],
            ["create", "modify"],
        )

        self.assertGreater(payload["entropy"], 0.9)
        self.assertLess(payload["margin"], 0.05)


if __name__ == "__main__":
    unittest.main()
