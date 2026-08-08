from __future__ import annotations

import unittest

from trendmodel.evidence import extract_evidence
from trendmodel.tf_features import (
    STRUCTURED_FEATURE_NAMES,
    structured_feature_vector,
    structured_feature_vector_from_ast,
)


class TensorFlowFeatureTest(unittest.TestCase):
    def test_feature_width_is_stable(self) -> None:
        vector = structured_feature_vector(
            "请执行 kubectl rollout restart deployment/api，代码在 app.py 中。"
        )
        self.assertEqual(len(vector), len(STRUCTURED_FEATURE_NAMES))
        self.assertTrue(all(0.0 <= value <= 1.0 for value in vector))

    def test_explicit_structure_is_visible(self) -> None:
        vector = structured_feature_vector("```python\nprint('ok')\n```")
        by_name = dict(zip(STRUCTURED_FEATURE_NAMES, vector))
        self.assertEqual(by_name["has_code_block"], 1.0)
        self.assertEqual(by_name["language:python"], 1.0)

    def test_pre_extracted_task_ast_has_the_same_feature_encoding(self) -> None:
        prompt = "请执行 kubectl get pods -n production"
        self.assertEqual(
            structured_feature_vector(prompt),
            structured_feature_vector_from_ast(extract_evidence(prompt)),
        )


if __name__ == "__main__":
    unittest.main()
