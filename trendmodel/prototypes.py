"""Semantic prototypes for the first hierarchical task taxonomy.

Prototypes are intentionally readable.  They are a starting vocabulary, not
training data pretending to be ground truth.  Once real annotations exist,
the same interface can compute centroids from labelled embeddings.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True)
class Prototype:
    label: str
    phrases: tuple[str, ...]
    name: str


@dataclass(frozen=True)
class PrototypeCatalog:
    heads: Mapping[str, tuple[Prototype, ...]]

    def labels(self, head: str) -> tuple[str, ...]:
        return tuple(item.label for item in self.heads[head])

    def groups(self, head: str) -> tuple[Prototype, ...]:
        return self.heads[head]

    def name(self, head: str, label: str) -> str:
        for item in self.heads[head]:
            if item.label == label:
                return item.name
        return label


def default_catalog() -> PrototypeCatalog:
    """Return the bilingual seed taxonomy used by :class:`TaskTagger`."""

    return PrototypeCatalog(
        heads={
            "intent": (
                _p("explain", "解释/讲解", "解释一个概念、原理或代码是如何工作的", "explain a concept, principle, or how code works"),
                _p("create", "创建/生成", "写出、生成、实现一个新的结果或程序", "create, write, build, or implement something new"),
                _p("modify", "修改/优化", "修改、重构、润色或优化已有内容", "modify, refactor, rewrite, or optimize existing content"),
                _p("review", "评审/检查", "审查质量、正确性、风险和可维护性", "review correctness, quality, risks, or maintainability"),
                _p("debug", "排障/修复", "定位错误、故障、异常或性能问题并修复", "debug, troubleshoot, diagnose, or fix a problem"),
                _p("execute", "执行/操作", "运行命令、部署服务或直接执行一个操作", "execute commands, deploy a service, or perform an operation"),
                _p("summarize", "总结/提炼", "总结长文本并提炼重点结论", "summarize a long text and extract its key points"),
                _p("translate", "翻译", "把内容从一种语言翻译成另一种语言", "translate content from one language to another"),
                _p("plan", "规划/拆解", "制定计划、路线图、步骤、排期或实施方案", "make a plan, roadmap, schedule, or implementation strategy"),
                _p("classify", "分类/归类", "判断内容属于哪个类别或进行归类", "classify or categorize items"),
                _p("extract", "提取/结构化", "从非结构化文本提取字段、行动项或表格", "extract fields, action items, or structured data"),
                _p("analyze", "分析/比较", "分析原因、趋势、差异、取舍或风险", "analyze causes, trends, trade-offs, differences, or risks"),
                _p("unknown", "未明确", "用户目标暂时不明确，需要更多上下文", "the user goal is unclear and needs more context"),
            ),
            "domain": (
                _p("conversation", "一般问答", "一般问题、知识问答和对话", "general questions and conversation"),
                _p("creative", "创意写作", "小说、故事、角色、诗歌和创意内容", "stories, fiction, characters, poetry, and creative writing"),
                _p("programming", "编程开发", "源代码、软件工程、算法、测试和 API 开发", "source code, software engineering, algorithms, tests, and APIs"),
                _p("mathematics", "数学", "数学公式、证明、计算和符号推理", "mathematical formulas, proofs, calculations, and symbolic reasoning"),
                _p("shell", "命令行", "Shell、终端命令、脚本和命令行操作", "shell, terminal commands, scripts, and command-line operations"),
                _p("networking", "网络", "网络协议、DNS、路由、HTTP、TCP 和防火墙", "network protocols, DNS, routing, HTTP, TCP, and firewalls"),
                _p("document", "文档", "邮件、报告、简历、技术文档和文字改写", "emails, reports, resumes, technical documents, and rewriting"),
                _p("data_analysis", "数据分析", "数据、日志、指标、统计和趋势分析", "data, logs, metrics, statistics, and trend analysis"),
                _p("operations", "运维基础设施", "云、Kubernetes、数据库、部署、监控和生产环境", "cloud, Kubernetes, databases, deployment, monitoring, and production systems"),
                _p("planning", "规划管理", "项目计划、路线图、里程碑、排期和优先级", "project plans, roadmaps, milestones, schedules, and prioritization"),
                _p("other", "其他", "不属于已定义领域的请求", "a request outside the defined domains"),
            ),
            "artifact": (
                _p("source_code", "源代码", "可运行的程序、函数、补丁或代码文件", "runnable source code, functions, patches, or code files"),
                _p("command", "命令/脚本", "Shell 命令、脚本或操作步骤", "shell commands, scripts, or operational commands"),
                _p("technical_explanation", "技术解释", "对概念、原理、代码或系统的说明", "an explanation of a concept, code, or system"),
                _p("document", "文档", "邮件、报告、文案、翻译或长文本结果", "an email, report, copy, translation, or prose document"),
                _p("analysis_report", "分析报告", "比较、评估、诊断或风险分析结果", "a comparison, evaluation, diagnosis, or risk analysis"),
                _p("plan", "计划方案", "路线图、实施计划、任务拆解或排期", "a roadmap, implementation plan, task breakdown, or schedule"),
                _p("answer", "直接回答", "一个直接的问答结果", "a direct answer to a question"),
                _p("unknown", "未确定", "暂时无法判断产物类型", "the expected artifact is not yet clear"),
            ),
            "workload_type": (
                _p("text", "文字", "写作、翻译、总结、润色和普通文字处理", "writing, translation, summarization, and prose editing"),
                _p("code", "代码", "编程、代码生成、调试、重构和测试", "programming, code generation, debugging, refactoring, and tests"),
                _p("analysis", "分析", "分析、解释、比较、研究和评估", "analysis, explanation, comparison, research, and evaluation"),
                _p("operations", "运维", "命令行、部署、基础设施和生产系统操作", "command-line work, deployment, infrastructure, and production operations"),
                _p("planning", "规划", "计划、路线图、任务拆解和项目安排", "plans, roadmaps, task breakdowns, and project organization"),
                _p("mixed", "混合", "同时涉及代码、文字、分析或运维多个交付面", "multiple delivery surfaces such as code, prose, analysis, and operations"),
                _p("other", "其他", "无法映射到主要工作类型", "a request outside the primary workload types"),
            ),
        }
    )


def _p(label: str, name: str, *phrases: str) -> Prototype:
    return Prototype(label=label, phrases=tuple(phrases), name=name)
