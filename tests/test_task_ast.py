from __future__ import annotations

import unittest

from trendmodel import TaskAST
from trendmodel.evidence import StructuredEvidence, extract_evidence
from trendmodel.task_ir import TaskEvidence


class TaskASTTest(unittest.TestCase):
    def test_extractor_returns_formal_ast_and_legacy_alias(self) -> None:
        ast = extract_evidence(
            "请执行 kubectl rollout restart deployment/api，检查 ./deploy/app.yaml"
        )

        self.assertIsInstance(ast, TaskAST)
        self.assertIs(StructuredEvidence, TaskAST)
        self.assertEqual(ast.original_prompt, "请执行 kubectl rollout restart deployment/api，检查 ./deploy/app.yaml")
        self.assertEqual(ast.action_scores, ast.intent_scores)
        self.assertIn("deployment", ast.operations)
        self.assertTrue(ast.constraints.requires_files)
        self.assertTrue(ast.constraints.requires_tools)

    def test_ast_serialization_contains_root_candidates_and_risk(self) -> None:
        ast = extract_evidence("请执行 rm -rf /tmp/cache")
        payload = ast.to_dict()

        self.assertEqual(payload["schema_version"], "task-ast-v1")
        self.assertEqual(payload["root"]["type"], "task_request")
        self.assertEqual(payload["root"]["original_prompt"], "请执行 rm -rf /tmp/cache")
        self.assertTrue(payload["risk"]["high"])
        self.assertIn("destructive", payload["risk"]["reasons"])
        self.assertEqual(payload["evidence"], [item.to_dict() for item in ast.evidence])

    def test_ast_evidence_has_provenance_and_scores_are_immutable(self) -> None:
        ast = extract_evidence("请执行 kubectl get pods -n production")

        self.assertTrue(ast.evidence)
        first = ast.evidence[0]
        self.assertIsNotNone(first.source_span)
        self.assertTrue(first.rule_id)
        self.assertIn("source_span", first.to_dict())
        with self.assertRaises(TypeError):
            ast.intent_scores["create"] = 0.5
        with self.assertRaises(TypeError):
            del ast.domain_scores["operations"]

    def test_negated_and_quoted_commands_are_marked_without_active_execution(self) -> None:
        negated = extract_evidence("不要执行 kubectl delete pod api-123 -n production")
        quoted = extract_evidence("请解释命令 `rm -rf /tmp/cache` 的作用")

        self.assertNotIn("execute", negated.intent_scores)
        execute_evidence = [item for item in negated.evidence if item.label == "execute"]
        self.assertTrue(execute_evidence[0].negated)
        self.assertFalse(negated.high_risk)
        destructive = [item for item in quoted.evidence if item.label == "destructive"]
        self.assertTrue(destructive[0].quoted)
        self.assertFalse(quoted.high_risk)

    def test_task_evidence_validates_strength(self) -> None:
        with self.assertRaises(ValueError):
            TaskEvidence("structure", "bad", True, 1.1)

    def test_top_candidates_are_deterministic_and_bounded(self) -> None:
        ast = extract_evidence("请分析并解释 C++ 多线程原理")

        candidates = ast.top_candidates("intent", limit=2)
        self.assertEqual(len(candidates), 2)
        self.assertGreaterEqual(candidates[0][1], candidates[1][1])
        self.assertEqual(ast.top_candidates("action", limit=2), candidates)

        with self.assertRaises(ValueError):
            ast.top_candidates("unknown", limit=2)
        with self.assertRaises(ValueError):
            ast.top_candidates("intent", limit=0)

    def test_ast_rejects_empty_prompt_and_out_of_range_scores(self) -> None:
        with self.assertRaises(ValueError):
            TaskAST(
                original_prompt=" ",
                constraints=extract_evidence("ok").constraints,
                language=None,
                operations=(),
                signals=(),
                intent_scores={},
                domain_scores={},
                artifact_scores={},
                high_risk=False,
                risk_reasons=(),
            )

        with self.assertRaises(ValueError):
            TaskAST(
                original_prompt="ok",
                constraints=extract_evidence("ok").constraints,
                language=None,
                operations=(),
                signals=(),
                intent_scores={"create": 1.1},
                domain_scores={},
                artifact_scores={},
                high_risk=False,
                risk_reasons=(),
            )


if __name__ == "__main__":
    unittest.main()
