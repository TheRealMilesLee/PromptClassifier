from __future__ import annotations

import unittest
from types import SimpleNamespace

from trendmodel.calibration import (
    abstention_decision,
    expected_calibration_error,
    fit_head_calibration,
    fit_temperature,
    negative_log_likelihood,
    temperature_scale,
)
from trendmodel.tf_data import LabelMaps
from trendmodel.tf_train import _fit_calibration_metadata


class CalibrationTest(unittest.TestCase):
    def test_temperature_scaling_preserves_distribution_and_softens_logits(self) -> None:
        scaled = temperature_scale([0.8, 0.2], 2.0)

        self.assertAlmostEqual(sum(scaled), 1.0, places=6)
        self.assertGreater(scaled[0], scaled[1])
        self.assertLess(scaled[0], 0.8)

    def test_fit_temperature_reports_before_and_after_metrics(self) -> None:
        rows = [[0.95, 0.05], [0.9, 0.1], [0.55, 0.45], [0.52, 0.48]]
        targets = [0, 0, 1, 1]

        result = fit_temperature(rows, targets)

        self.assertGreater(result["samples"], 0)
        self.assertGreaterEqual(result["temperature"], 0.25)
        self.assertLessEqual(result["temperature"], 4.0)
        self.assertIn("ece_before", result)
        self.assertIn("ece_after", result)

    def test_head_calibration_reads_full_output_distribution(self) -> None:
        predictions = [
            {"predictions": {"intent": {"distribution": {"create": 0.8, "explain": 0.2}}}},
            {"predictions": {"intent": {"distribution": {"create": 0.3, "explain": 0.7}}}},
        ]

        result = fit_head_calibration(
            predictions,
            ["create", "explain"],
            head="intent",
            labels=["create", "explain"],
        )

        self.assertEqual(result["samples"], 2)
        self.assertGreater(result["temperature"], 0.0)

    def test_metrics_are_stable_for_empty_inputs(self) -> None:
        self.assertEqual(negative_log_likelihood([], []), 0.0)
        self.assertEqual(expected_calibration_error([], []), 0.0)

    def test_training_calibration_excludes_weak_labels(self) -> None:
        predictions = [
            {"predictions": {
                "workload_type": {"distribution": {"text": 0.8, "code": 0.2}},
                "intent": {"distribution": {"create": 0.8, "explain": 0.2}},
                "complexity": {"distribution": {"low": 0.8, "high": 0.2}},
                "effort": {"distribution": {"XS": 0.1, "S": 0.2, "M": 0.5, "L": 0.15, "XL": 0.05}},
            }},
            {"predictions": {
                "workload_type": {"distribution": {"text": 0.2, "code": 0.8}},
                "intent": {"distribution": {"create": 0.2, "explain": 0.8}},
                "complexity": {"distribution": {"low": 0.2, "high": 0.8}},
                "effort": {"distribution": {"XS": 0.05, "S": 0.15, "M": 0.5, "L": 0.2, "XL": 0.1}},
            }},
        ]
        records = [
            SimpleNamespace(
                workload_type="text", intent="create", complexity="low", effort_bucket="M",
                label_source="manual_gold",
            ),
            SimpleNamespace(
                workload_type="code", intent="explain", complexity="high", effort_bucket="M",
                label_source="task_tagger_pseudo",
            ),
        ]

        result = _fit_calibration_metadata(
            predictions,
            records,
            LabelMaps(("text", "code"), ("create", "explain"), ("low", "high")),
        )

        self.assertEqual(result["eligible_samples"], 1)
        self.assertEqual(result["heads"]["intent"]["samples"], 1)

    def test_abstention_is_explicit_and_does_not_mutate_labels(self) -> None:
        decision = abstention_decision(
            {
                "workload_type": {"label": "code", "probability": 0.8, "margin": 0.6, "entropy": 0.4},
                "intent": {"label": "create", "probability": 0.44, "margin": 0.02, "entropy": 0.99},
                "complexity": {"label": "high", "probability": 0.7, "margin": 0.4, "entropy": 0.5},
            },
            {"abstention": {"enabled": True, "min_probability": 0.45, "min_margin": 0.05, "max_entropy": 0.98}},
        )

        self.assertTrue(decision["abstain"])
        self.assertIn("intent:low_probability", decision["reasons"])
        self.assertIn("thresholds", decision)


if __name__ == "__main__":
    unittest.main()
