from __future__ import annotations

import unittest

from trendmodel import TaskAST
from trendmodel.evidence import StructuredEvidence, extract_evidence


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
