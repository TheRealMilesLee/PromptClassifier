from __future__ import annotations

import json
import unittest

from trendmodel.evidence import extract_evidence
from trendmodel.fusion import FusionWeights, fuse_probabilities
from trendmodel.tagger import TaskTagger


class TaskTaggerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tagger = TaskTagger()

    def test_code_creation_extracts_language_and_operations(self) -> None:
        result = self.tagger.predict("帮我实现一个 C++ 多线程程序，计算笛卡尔积").to_dict()

        self.assertEqual(result["intent"]["label"], "create")
        self.assertEqual(result["domains"][0]["label"], "programming")
        self.assertEqual(result["workload_type"]["label"], "code")
        self.assertEqual(result["artifact"], "source_code")
        self.assertEqual(result["language"], "cpp")
        self.assertEqual(result["operations"], ["multithreading", "cartesian_product"])
        self.assertFalse(result["constraints"]["requires_tools"])
        self.assertGreater(result["capability_demand"]["coding"], 0.8)
        self.assertEqual(result["task_ir"]["contract_version"], "task-ir-v1")
        self.assertEqual(result["task_ir"]["workload_type"]["label"], "code")
        self.assertTrue(result["task_ir"]["fusion"]["post_hoc_probability_fusion"])
        json.dumps(result, ensure_ascii=False)

        python_result = self.tagger.predict("实现一个 Python 函数，把 CSV 转换成 JSON").to_dict()
        self.assertEqual(python_result["language"], "python")
        self.assertFalse(python_result["constraints"]["has_shell_commands"])

    def test_explanation_is_not_execute_even_when_it_mentions_execution(self) -> None:
        result = self.tagger.predict("请解释这段 C++ 多线程代码的执行过程").to_dict()

        self.assertEqual(result["intent"]["label"], "explain")
        self.assertEqual(result["artifact"], "technical_explanation")
        self.assertFalse(result["constraints"]["requires_tools"])
        self.assertEqual(result["workload_type"]["label"], "code")

    def test_networking_question_and_command_have_different_task_ir(self) -> None:
        explanation = self.tagger.predict("MANET 中的路由协议是什么，如何实现的？").to_dict()
        implementation = self.tagger.predict("帮我实现一个 MANET 路由协议模拟器").to_dict()
        operation = self.tagger.predict("请执行 kubectl delete pod api-123 -n production").to_dict()

        self.assertEqual(explanation["intent"]["label"], "explain")
        self.assertEqual(explanation["domains"][0]["label"], "networking")
        self.assertEqual(explanation["artifact"], "technical_explanation")
        self.assertEqual(explanation["risk_level"], "low")

        self.assertEqual(implementation["intent"]["label"], "create")
        self.assertEqual(implementation["artifact"], "source_code")
        self.assertEqual(implementation["workload_type"]["label"], "code")
        self.assertIn("networking", [item["label"] for item in implementation["domains"]])
        self.assertIn("programming", [item["label"] for item in implementation["domains"]])

        self.assertEqual(operation["intent"]["label"], "execute")
        self.assertIn("shell", [item["label"] for item in operation["domains"]])
        self.assertEqual(operation["artifact"], "command")
        self.assertTrue(operation["constraints"]["requires_tools"])
        self.assertEqual(operation["risk_level"], "high")

    def test_explicit_execution_wins_over_build_word_in_path(self) -> None:
        result = self.tagger.predict(
            "请执行 rm -rf /tmp/build-cache 并检查服务状态"
        ).to_dict()

        self.assertEqual(result["intent"]["label"], "execute")
        self.assertEqual(result["workload_type"]["label"], "operations")
        self.assertTrue(result["constraints"]["has_shell_commands"])
        self.assertEqual(result["risk_level"], "high")

    def test_structural_evidence_is_observable_but_not_a_direct_classifier(self) -> None:
        evidence = extract_evidence("请解释 ./src/main.cpp 中的 DNS 解析逻辑")

        self.assertTrue(evidence.constraints.has_file_paths)
        self.assertEqual(evidence.language, "cpp")
        self.assertIn("dns", evidence.operations)
        self.assertEqual(evidence.intent_scores["explain"], 0.95)
        self.assertTrue(any(item.kind == "domain_hint" for item in evidence.signals))

    def test_fusion_keeps_missing_evidence_neutral(self) -> None:
        result = fuse_probabilities(
            {"create": 0.8, "explain": 0.2},
            {"create": 0.0, "explain": 0.0},
            weights=FusionWeights(),
        )

        self.assertAlmostEqual(sum(result.posterior.values()), 1.0, places=6)
        self.assertGreater(result.posterior["create"], result.posterior["explain"])
        self.assertAlmostEqual(result.structural["create"], 0.5, places=6)


if __name__ == "__main__":
    unittest.main()
