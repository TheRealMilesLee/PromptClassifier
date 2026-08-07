"""Hierarchical prompt tagger: embedding prototypes plus evidence fusion."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from typing import Any

from .embedding import Embedder, HashEmbedder
from .evidence import StructuredEvidence, extract_evidence
from .fusion import FusionResult, FusionWeights, confidence, fuse_probabilities, ranked, softmax
from .prototypes import PrototypeCatalog, default_catalog
from .task_ir import (
    CapabilityDemand,
    LabelCandidate,
    TaskIR,
    TaskPrediction,
    clamp_probability,
)


class TaskTagger:
    """Classify a prompt into a router-facing, explainable Task IR.

    The default backend is deliberately dependency-free.  Pass a
    ``SentenceTransformerEmbedder`` for real semantic coverage, or inject a
    custom embedder for training/evaluation experiments.
    """

    model_version = "task-ir-prototype-v1"

    def __init__(
        self,
        embedder: Embedder | None = None,
        *,
        catalog: PrototypeCatalog | None = None,
        fusion_weights: FusionWeights = FusionWeights(),
    ) -> None:
        self.embedder = embedder or HashEmbedder()
        self.catalog = catalog or default_catalog()
        self.fusion_weights = fusion_weights
        self._prototype_vectors: dict[str, dict[str, list[list[float]]]] = {}
        for head, groups in self.catalog.heads.items():
            vectors = self.embedder.encode(
                [phrase for group in groups for phrase in group.phrases]
            )
            cursor = 0
            self._prototype_vectors[head] = {}
            for group in groups:
                count = len(group.phrases)
                self._prototype_vectors[head][group.label] = vectors[cursor: cursor + count]
                cursor += count

    def predict(self, prompt: str, *, context: Mapping[str, Any] | None = None) -> TaskIR:
        original_prompt = str(prompt).strip()
        if not original_prompt:
            raise ValueError("prompt must not be empty")
        context = context or {}
        evidence = extract_evidence(original_prompt)
        prompt_vector = self.embedder.embed(original_prompt)

        intent_fusion = self._classify_head(
            "intent",
            prompt_vector,
            evidence.intent_scores,
            self._context_prior("intent", context),
        )
        intent_label = self._select_intent(intent_fusion, evidence, original_prompt)
        intent_prediction = self._prediction("intent", intent_label, intent_fusion.posterior)

        domain_structural = dict(evidence.domain_scores)
        self._add_domain_intent_hint(domain_structural, intent_label, evidence)
        domain_fusion = self._classify_head(
            "domain",
            prompt_vector,
            domain_structural,
            self._context_prior("domain", context),
        )
        domain_predictions = self._select_domains(domain_fusion, evidence)

        artifact_structural = dict(evidence.artifact_scores)
        self._add_artifact_hint(artifact_structural, intent_label, evidence)
        artifact_fusion = self._classify_head(
            "artifact",
            prompt_vector,
            artifact_structural,
            self._context_prior("artifact", context),
        )
        artifact_label, artifact_probability = self._select_artifact(
            artifact_fusion,
            intent_label,
            evidence,
        )

        mixed_request = self._is_mixed_request(original_prompt, evidence)

        workload_structural = self._workload_structural_scores(
            evidence,
            artifact_label,
            intent_label,
            domain_predictions,
            mixed_request,
        )
        workload_fusion = self._classify_head(
            "workload_type",
            prompt_vector,
            workload_structural,
            self._context_prior("workload_type", context),
        )
        workload_label = self._select_workload(
            workload_fusion,
            artifact_label,
            intent_label,
            domain_predictions,
            mixed_request,
            evidence,
            original_prompt,
        )
        workload_prediction = self._prediction(
            "workload_type",
            workload_label,
            workload_fusion.posterior,
        )

        capability = self._capability_demand(
            intent_label,
            artifact_label,
            domain_predictions,
            evidence,
            original_prompt,
        )
        complexity_score = self._complexity_score(capability, evidence, domain_predictions)
        complexity = self._complexity_prediction(complexity_score)
        risk_level = self._risk_level(intent_label, evidence)
        effort = self._effort_estimate(
            complexity_score,
            capability,
            evidence,
            risk_level,
        )

        intent_confidence = max(
            confidence(intent_fusion.posterior),
            max(evidence.intent_scores.values(), default=0.0),
        )
        domain_confidence = max(
            confidence(domain_fusion.posterior),
            max(domain_structural.values(), default=0.0),
        )
        evidence_clarity = min(1.0, 0.35 + 0.08 * len(evidence.signals))
        overall_confidence = (
            0.45 * intent_confidence
            + 0.35 * domain_confidence
            + 0.20 * evidence_clarity
        )
        if intent_label == "unknown" or not domain_predictions:
            overall_confidence *= 0.75
        needs_clarification = (
            overall_confidence < 0.46
            or intent_label == "unknown"
            or (domain_predictions and domain_predictions[0].label == "other" and overall_confidence < 0.62)
        )

        return TaskIR(
            original_prompt=original_prompt,
            intent=intent_prediction,
            domains=tuple(domain_predictions),
            workload_type=workload_prediction,
            artifact=artifact_label,
            artifact_probability=artifact_probability,
            operations=evidence.operations,
            language=evidence.language,
            constraints=evidence.constraints,
            capability_demand=capability,
            complexity=complexity,
            effort=effort,
            risk_level=risk_level,
            confidence=clamp_probability(overall_confidence),
            needs_clarification=needs_clarification,
            evidence=evidence.signals,
        )

    def predict_many(
        self,
        prompts: Sequence[str],
        *,
        context: Mapping[str, Any] | None = None,
    ) -> list[TaskIR]:
        return [self.predict(prompt, context=context) for prompt in prompts]

    def _classify_head(
        self,
        head: str,
        prompt_vector: list[float],
        structural_scores: Mapping[str, float],
        context_prior: Mapping[str, float] | None,
    ) -> FusionResult:
        semantic_scores: dict[str, float] = {}
        for label, vectors in self._prototype_vectors[head].items():
            similarities = [
                _safe_cosine(prompt_vector, vector)
                for vector in vectors
            ]
            if not similarities:
                semantic_scores[label] = 0.0
                continue
            # Max captures a close paraphrase; the mean prevents one accidental
            # prototype from dominating an entire category.
            semantic_scores[label] = 0.75 * max(similarities) + 0.25 * sum(similarities) / len(similarities)
        complete_structural = {
            label: float(structural_scores.get(label, 0.0))
            for label in self.catalog.labels(head)
        }
        return fuse_probabilities(
            semantic_scores,
            complete_structural,
            context_prior,
            weights=self.fusion_weights,
        )

    def _prediction(
        self,
        head: str,
        label: str,
        probabilities: Mapping[str, float],
    ) -> TaskPrediction:
        candidates = tuple(
            LabelCandidate(
                candidate_label,
                probability,
                self.catalog.name(head, candidate_label),
            )
            for candidate_label, probability in ranked(probabilities, limit=3)
        )
        return TaskPrediction(
            label=label,
            probability=float(probabilities.get(label, 0.0)),
            candidates=candidates,
            name=self.catalog.name(head, label),
        )

    def _select_intent(
        self,
        result: FusionResult,
        evidence: StructuredEvidence,
        prompt: str,
    ) -> str:
        if evidence.intent_scores:
            # An explicit execution request with a command/tool signal must
            # win over incidental words such as ``build`` in a file path or
            # ``check`` in a follow-up clause.  This is also the safety
            # boundary consumed by the router for shell/production actions.
            if evidence.intent_scores.get("execute", 0.0) >= 0.9 and (
                evidence.constraints.has_shell_commands
                or evidence.constraints.requires_tools
                or evidence.high_risk
            ):
                return "execute"
            if evidence.intent_scores.get("create", 0.0) >= 0.9:
                create_match = re.search(
                    r"写|生成|创建|开发|搭建|构建|补充|设计|建立|实现|\b(?:write|create|build|implement|generate)\b",
                    prompt,
                    re.I,
                )
                if create_match:
                    secondary_matches = [
                        match
                        for pattern in (
                            r"解释|说明|讲解|原理|\b(?:explain|why|how does)\b",
                            r"评估|评审|检查|审查|\breview\b",
                            r"分析|比较|对比|\b(?:analyze|compare|evaluate)\b",
                            r"提取|抽取|结构化|列出|\bextract\b",
                        )
                        if (match := re.search(pattern, prompt, re.I))
                    ]
                    if secondary_matches and create_match.start() < min(item.start() for item in secondary_matches):
                        return "create"
            if evidence.intent_scores.get("create", 0.0) >= 0.9 and evidence.intent_scores.get("execute", 0.0) >= 0.9:
                create_match = re.search(r"写|创建|开发|搭建|构建|设计|实现|\b(?:write|create|build|implement)\b", prompt, re.I)
                execute_match = re.search(r"执行|运行|部署|上线|\b(?:run|execute|deploy|apply)\b", prompt, re.I)
                if create_match and execute_match and create_match.start() < execute_match.start():
                    return "create"
            structural_label, structural_value = max(
                evidence.intent_scores.items(),
                key=lambda item: (item[1], item[0]),
            )
            # An explicit action phrase is a high-confidence observation.  It
            # resolves the known failure mode where a lightweight fallback
            # embedding spreads probability over many prototype classes.
            if structural_value >= 0.86:
                return structural_label
        top = ranked(result.posterior, limit=1)[0]
        if top[1] < 0.16 and not any(value > 0.55 for value in result.structural.values()):
            return "unknown"
        return top[0]

    def _select_domains(
        self,
        result: FusionResult,
        evidence: StructuredEvidence,
    ) -> list[TaskPrediction]:
        ordered = ranked(result.posterior, limit=4)
        if not ordered:
            return []
        top_label, top_probability = ordered[0]
        selected = [top_label]
        for label, probability in ordered[1:]:
            structural = float(evidence.domain_scores.get(label, 0.0))
            if len(selected) >= 2:
                break
            if structural >= 0.68 and probability >= max(0.12, top_probability * 0.38):
                selected.append(label)
        return [self._prediction("domain", label, result.posterior) for label in selected]

    def _select_artifact(
        self,
        result: FusionResult,
        intent_label: str,
        evidence: StructuredEvidence,
    ) -> tuple[str, float]:
        top_label, top_probability = ranked(result.posterior, limit=1)[0]
        if intent_label == "explain":
            preferred = "technical_explanation"
        elif intent_label == "execute":
            preferred = "command"
        elif intent_label in {"debug", "review"} and evidence.domain_scores.get("operations", 0.0) >= 0.7:
            preferred = "analysis_report"
        elif intent_label in {"create", "modify", "debug", "review"} and (
            (evidence.language and evidence.language != "shell")
            or evidence.constraints.has_code_block
            or evidence.domain_scores.get("programming", 0.0) >= 0.75
        ):
            preferred = "source_code"
        elif intent_label == "plan":
            preferred = "plan"
        elif intent_label in {"analyze", "review", "classify"}:
            preferred = "analysis_report"
        elif intent_label in {"summarize", "translate", "extract", "create", "modify"}:
            preferred = "document"
        else:
            preferred = top_label
        if preferred != top_label:
            # Explicit intent/structure is a high-confidence artifact signal;
            # retain the prototype probability as a lower bound for display.
            preferred_probability = max(
                float(result.posterior.get(preferred, 0.0)),
                float(result.structural.get(preferred, 0.0)) * 0.65,
            )
            return preferred, min(0.99, preferred_probability)
        return top_label, top_probability

    def _select_workload(
        self,
        result: FusionResult,
        artifact: str,
        intent: str,
        domains: Sequence[TaskPrediction],
        mixed_request: bool,
        evidence: StructuredEvidence,
        prompt: str,
    ) -> str:
        if mixed_request:
            return "mixed"
        if artifact not in {"source_code", "command"} and evidence.domain_scores.get("planning", 0.0) >= 0.65 and re.search(
            r"流程|清单|路线|排期|路线图|backlog|里程碑|roadmap|schedule|学习计划|优先级",
            prompt,
            re.I,
        ):
            return "planning"
        if evidence.domain_scores.get("operations", 0.0) >= 0.7 and artifact in {
            "document", "analysis_report", "technical_explanation", "plan"
        }:
            return "operations"
        if artifact == "source_code":
            return "code"
        if artifact == "command":
            return "operations"
        if artifact == "plan":
            return "planning"
        if artifact == "document":
            return "text"
        if artifact == "analysis_report":
            if any(domain.label in {"shell", "operations"} for domain in domains):
                return "operations"
            return "analysis"
        if artifact == "technical_explanation":
            if any(domain.label == "programming" for domain in domains):
                return "code"
            if any(domain.label in {"shell", "operations"} for domain in domains):
                return "operations"
            return "analysis"
        if len(domains) > 1 and artifact in {"unknown", "answer"}:
            return "mixed"
        top_label = ranked(result.posterior, limit=1)[0][0]
        if top_label == "other" and intent == "unknown":
            return "other"
        return top_label

    def _workload_structural_scores(
        self,
        evidence: StructuredEvidence,
        artifact: str,
        intent: str,
        domains: Sequence[TaskPrediction],
        mixed_request: bool,
    ) -> dict[str, float]:
        scores = {label: 0.0 for label in self.catalog.labels("workload_type")}
        mapping = {
            "programming": "code",
            "shell": "operations",
            "operations": "operations",
            "planning": "planning",
            "document": "text",
            "creative": "text",
            "data_analysis": "analysis",
            "mathematics": "analysis",
            "networking": "analysis",
            "conversation": "text",
        }
        for domain, score_value in evidence.domain_scores.items():
            target = mapping.get(domain)
            if target:
                scores[target] = max(scores[target], score_value)
        artifact_target = {
            "source_code": "code",
            "command": "operations",
            "document": "text",
            "analysis_report": "analysis",
            "technical_explanation": "analysis",
            "plan": "planning",
        }.get(artifact)
        if artifact_target:
            scores[artifact_target] = max(scores[artifact_target], 0.86)
        if artifact_target in {"analysis", "text", "planning"} and evidence.domain_scores.get("operations", 0.0) >= 0.7:
            scores["operations"] = max(scores["operations"], 0.97)
        domain_labels = {domain.label for domain in domains}
        if mixed_request:
            scores["mixed"] = max(scores["mixed"], 0.96)
        elif len(domain_labels) > 1 and artifact in {"unknown", "answer"}:
            scores["mixed"] = max(scores["mixed"], 0.72)
        if intent in {"analyze", "review", "explain", "classify"}:
            scores["analysis"] = max(scores["analysis"], 0.55)
        return scores

    def _is_mixed_request(self, prompt: str, evidence: StructuredEvidence) -> bool:
        """Detect multiple deliverables without flattening every conjunction."""

        connector = bool(
            re.search(
                r"并且|同时|以及|包括|一并|并|(?:后|之后)[，,\s]*(?:给出|输出|形成|设计|拆解)|\band\b|\balong with\b|\bplus\b",
                prompt,
                re.I,
            )
        )
        if not connector:
            return False
        if len(evidence.intent_scores) >= 2:
            # "translate ... and explain ..." or "summarize and extract ..."
            # really does expose two router-relevant outputs.
            explicit_second_action = bool(
                re.search(
                    r"(?:并且|同时|以及|一并|并|\band\b|\balong with\b|\bplus\b)\s*(?:写|创建|实现|修改|解释|说明|总结|提取|列出|翻译|分析|审查|\b(?:write|create|implement|modify|explain|summarize|extract|translate|analyze|review)\b)",
                    prompt,
                    re.I,
                )
            )
            if explicit_second_action:
                return True
        domain_groups = set()
        for domain in evidence.domain_scores:
            if domain in {"programming", "operations", "shell"}:
                domain_groups.add("technical")
            elif domain in {"document", "creative"}:
                domain_groups.add("prose")
            elif domain in {"data_analysis", "mathematics", "networking", "planning"}:
                domain_groups.add("analysis")
        if len(evidence.operations) and re.search(r"监控|告警|仪表盘|dashboard", prompt, re.I) and re.search(
            r"部署|生产|上线|代码|文档|测试|deploy|production|release|code|document|test",
            prompt,
            re.I,
        ):
            return True
        return len(domain_groups) >= 2 and bool(
            re.search(
                r"代码|部署|监控|文档|报告|方案|测试|接口|\bcode\b|\bdeploy\b|\bmonitor\b|\bdocument\b|\breport\b|\bplan\b",
                prompt,
                re.I,
            )
        )

    def _add_artifact_hint(
        self,
        scores: dict[str, float],
        intent: str,
        evidence: StructuredEvidence,
    ) -> None:
        if intent == "explain":
            scores["technical_explanation"] = max(scores.get("technical_explanation", 0.0), 0.82)
        elif intent == "execute" and evidence.constraints.has_shell_commands:
            scores["command"] = max(scores.get("command", 0.0), 0.92)
        elif intent == "plan":
            scores["plan"] = max(scores.get("plan", 0.0), 0.8)
        elif intent in {"analyze", "review", "classify"}:
            scores["analysis_report"] = max(scores.get("analysis_report", 0.0), 0.72)

    def _add_domain_intent_hint(
        self,
        scores: dict[str, float],
        intent: str,
        evidence: StructuredEvidence,
    ) -> None:
        """Use the action only as a weak domain prior when structure is sparse."""

        if intent in {"translate", "summarize", "extract"} and not (
            evidence.language or evidence.constraints.has_code_block or evidence.constraints.has_shell_commands
        ):
            scores["document"] = max(scores.get("document", 0.0), 0.78)
        elif intent == "plan":
            scores["planning"] = max(scores.get("planning", 0.0), 0.72)
        elif intent == "explain" and not scores:
            scores["conversation"] = 0.52
        elif intent in {"explain", "analyze", "review", "debug"} and scores.get("operations", 0.0) >= 0.7:
            # A Terraform/Kubernetes/DNS discussion is an operations task even
            # when the command-line tool name also creates a shell signal.
            scores["operations"] = max(scores.get("operations", 0.0), 0.99)

    def _capability_demand(
        self,
        intent: str,
        artifact: str,
        domains: Sequence[TaskPrediction],
        evidence: StructuredEvidence,
        prompt: str,
    ) -> CapabilityDemand:
        reasoning_base = {
            "explain": 0.62,
            "analyze": 0.72,
            "review": 0.7,
            "debug": 0.7,
            "create": 0.58,
            "modify": 0.58,
            "plan": 0.64,
            "classify": 0.42,
            "extract": 0.3,
            "summarize": 0.3,
            "translate": 0.2,
            "execute": 0.38,
            "unknown": 0.45,
        }.get(intent, 0.45)
        reasoning = reasoning_base + 0.08 * max(0, len(evidence.operations) - 1)
        if len(domains) > 1:
            reasoning += 0.08
        coding = 0.05
        if artifact == "source_code":
            coding = 0.9
        elif artifact == "command":
            coding = 0.35
        elif evidence.language and evidence.language != "shell":
            coding = 0.68
        elif evidence.language == "shell":
            coding = 0.35
        elif evidence.constraints.has_code_block:
            coding = 0.62
        context = 0.08
        if evidence.constraints.has_code_block:
            context += 0.16
        if evidence.constraints.has_file_paths:
            context += 0.22
        if evidence.constraints.requires_files:
            context += 0.16
        if len(prompt) > 160:
            context += 0.16
        if len(evidence.operations) >= 2:
            context += 0.12
        tool_use = 0.05
        if evidence.constraints.requires_tools:
            tool_use = 0.9
        elif evidence.constraints.has_shell_commands:
            tool_use = 0.52
        elif evidence.domain_scores.get("operations", 0.0) >= 0.7:
            tool_use = 0.42
        return CapabilityDemand(
            reasoning=clamp_probability(reasoning),
            coding=clamp_probability(coding),
            context=clamp_probability(context),
            tool_use=clamp_probability(tool_use),
        )

    def _complexity_score(
        self,
        capability: CapabilityDemand,
        evidence: StructuredEvidence,
        domains: Sequence[TaskPrediction],
    ) -> float:
        value = (
            0.35 * capability.reasoning
            + 0.25 * capability.coding
            + 0.20 * capability.context
            + 0.20 * capability.tool_use
        )
        if len(evidence.operations) >= 3:
            value += 0.08
        if len(domains) > 1:
            value += 0.06
        return max(0.0, min(1.0, value))

    def _complexity_prediction(self, score_value: float) -> TaskPrediction:
        probabilities = softmax(
            {
                "low": -abs(score_value - 0.2),
                "medium": -abs(score_value - 0.5),
                "high": -abs(score_value - 0.8),
            },
            temperature=0.11,
        )
        label = ranked(probabilities, limit=1)[0][0]
        candidates = tuple(
            LabelCandidate(item, probability)
            for item, probability in ranked(probabilities, limit=3)
        )
        return TaskPrediction(label, probabilities[label], candidates, label)

    def _effort_estimate(
        self,
        complexity_score: float,
        capability: CapabilityDemand,
        evidence: StructuredEvidence,
        risk_level: str,
    ) -> dict[str, Any]:
        score_value = (
            0.25 * complexity_score
            + 0.20 * capability.reasoning
            + 0.20 * capability.coding
            + 0.15 * capability.context
            + 0.20 * capability.tool_use
        )
        if len(evidence.operations) >= 2:
            score_value += 0.06
        if risk_level == "high":
            score_value += 0.08
        score_value = max(0.0, min(1.0, score_value))
        if score_value < 0.18:
            label, time_range = "XS", "5–15 分钟"
        elif score_value < 0.36:
            label, time_range = "S", "15–60 分钟"
        elif score_value < 0.58:
            label, time_range = "M", "1–4 小时"
        elif score_value < 0.78:
            label, time_range = "L", "4–16 小时"
        else:
            label, time_range = "XL", "16 小时以上"
        uncertainty = 0.55 + 0.35 * min(1.0, len(evidence.signals) / 8)
        if not evidence.signals:
            uncertainty -= 0.2
        return {
            "label": label,
            "estimated_score": round(score_value, 4),
            "confidence": round(max(0.25, min(0.9, uncertainty)), 4),
            "time_range": time_range,
            "basis": "heuristic; calibrate with observed active_minutes",
        }

    def _risk_level(self, intent: str, evidence: StructuredEvidence) -> str:
        if evidence.high_risk and intent in {"execute", "create", "modify", "debug"}:
            return "high"
        if evidence.high_risk or evidence.constraints.requires_tools or intent in {"execute", "modify", "debug"}:
            return "medium"
        return "low"

    def _context_prior(self, head: str, context: Mapping[str, Any]) -> dict[str, float]:
        labels = self.catalog.labels(head)
        prior = {label: 0.0 for label in labels}

        direct = context.get(f"{head}_prior")
        if isinstance(direct, Mapping):
            for label, value in direct.items():
                if str(label) in prior:
                    prior[str(label)] += float(value)
        preferred_key = {
            "domain": "preferred_domain",
            "intent": "preferred_intent",
            "artifact": "preferred_artifact",
            "workload_type": "preferred_workload",
        }.get(head)
        if preferred_key and str(context.get(preferred_key, "")) in prior:
            prior[str(context[preferred_key])] += 0.5

        if bool(context.get("in_project") or context.get("has_files") or context.get("has_attachments")):
            for label, value in {
                "programming": 0.35,
                "document": 0.25,
                "data_analysis": 0.2,
                "code": 0.25,
                "text": 0.15,
            }.items():
                if label in prior:
                    prior[label] += value
        if bool(context.get("allow_modify_files")):
            for label, value in {"programming": 0.25, "operations": 0.2, "source_code": 0.25}.items():
                if label in prior:
                    prior[label] += value
        if bool(context.get("tool_access")):
            for label, value in {"shell": 0.3, "operations": 0.3, "execute": 0.25, "command": 0.25}.items():
                if label in prior:
                    prior[label] += value
        return prior


def _safe_cosine(left: list[float], right: list[float]) -> float:
    try:
        from .fusion import cosine_similarity

        return cosine_similarity(left, right)
    except ValueError:
        return 0.0
