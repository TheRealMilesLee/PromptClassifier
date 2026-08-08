from __future__ import annotations

import math
import unittest

from trendmodel.tf_inference import _effort_payload_from_raw, _classification_payload


class TensorFlowInferenceContractTest(unittest.TestCase):
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
