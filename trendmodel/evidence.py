"""Deterministic extraction of high-confidence structural prompt signals."""

from __future__ import annotations

import re
import unicodedata

from .task_ast import TaskAST
from .task_ir import TaskConstraints, TaskEvidence


# Keep the old import path working for downstream callers.  New code should
# use TaskAST to make the intermediate representation explicit.
StructuredEvidence = TaskAST


def extract_evidence(text: str) -> TaskAST:
    """Extract code/file/command/math and domain-specific evidence.

    A match contributes a bounded likelihood hint.  It never directly sets
    the final class; :mod:`trendmodel.fusion` combines these hints with the
    sentence embedding and optional context prior.
    """

    original = str(text)
    if not original.strip():
        raise ValueError("text must not be empty")
    normalized, normalized_spans = _normalize_with_spans(original)
    signals: list[TaskEvidence] = []
    intent_scores: dict[str, float] = {}
    domain_scores: dict[str, float] = {}
    artifact_scores: dict[str, float] = {}

    def signal(
        kind: str,
        label: str,
        value: str | bool | float,
        strength: float,
        detail: str,
        *,
        match: re.Match[str] | None = None,
        rule_id: str | None = None,
        negated: bool = False,
        quoted: bool = False,
    ) -> None:
        source_span = (
            _span_to_original(match.span(), normalized_spans)
            if match is not None
            else None
        )
        signals.append(
            TaskEvidence(
                kind,
                label,
                value,
                strength,
                detail,
                source_span=source_span,
                rule_id=rule_id or f"{kind}:{label}",
                negated=negated,
                quoted=quoted,
            )
        )

    def score(target: dict[str, float], label: str, value: float) -> None:
        target[label] = max(target.get(label, 0.0), min(1.0, value))

    has_code_block = "```" in original or bool(re.search(r"(?m)^\s{4,}\S+", original))
    if has_code_block:
        code_match = _search(re.compile(r"```|^\s{4,}\S+", re.M), normalized)
        signal(
            "structure",
            "code_block",
            True,
            0.98,
            "发现 fenced 或缩进代码块",
            match=code_match,
            rule_id="structure:code_block",
        )
        score(domain_scores, "programming", 0.95)
        score(artifact_scores, "source_code", 0.9)

    file_match = _search(_FILE_PATH_RE, normalized)
    has_file_paths = file_match is not None
    if file_match:
        value = _short_value(file_match.group(1) or file_match.group(0))
        signal(
            "structure",
            "file_path",
            value,
            0.93,
            "发现文件路径或常见代码/配置文件名",
            match=file_match,
            rule_id="structure:file_path",
            quoted=_is_quoted(normalized, file_match.span()),
        )
        score(artifact_scores, "source_code", 0.55)
        score(domain_scores, "programming", 0.5)

    url_match = _search(_URL_RE, normalized)
    has_urls = url_match is not None
    if url_match:
        signal(
            "structure",
            "url",
            _short_value(url_match.group(0)),
            0.8,
            "发现 URL",
            match=url_match,
            rule_id="structure:url",
            quoted=_is_quoted(normalized, url_match.span()),
        )

    math_match = _search(_MATH_RE, normalized)
    has_math_symbols = math_match is not None
    if math_match:
        signal(
            "structure",
            "math_symbols",
            _short_value(math_match.group(0)),
            0.96,
            "发现数学符号、公式或数学术语",
            match=math_match,
            rule_id="structure:math_symbols",
            quoted=_is_quoted(normalized, math_match.span()),
        )
        score(domain_scores, "mathematics", 0.95)
        score(artifact_scores, "technical_explanation", 0.3)

    language, language_strength, language_match = _detect_language(normalized)
    if language:
        signal(
            "language",
            language,
            _short_value(language_match.group(0) if language_match else language),
            language_strength,
            "发现编程语言或语言特有标记",
            match=language_match,
            rule_id=f"language:{language}",
            quoted=_is_quoted(normalized, language_match.span()) if language_match else False,
        )
        score(domain_scores, "programming", 0.92)
        score(artifact_scores, "source_code", 0.82)

    shell_match = _search(_SHELL_COMMAND_RE, normalized)
    command_line_match = _search(_COMMAND_LINE_RE, normalized)
    command_args_match = _search(_COMMAND_WITH_ARGS_RE, normalized)
    has_shell_commands = (
        command_line_match is not None
        or command_args_match is not None
        or (has_code_block and shell_match is not None)
    )
    if has_shell_commands:
        matched = (
            command_line_match.group(0)
            if command_line_match
            else command_args_match.group(0)
            if command_args_match
            else shell_match.group(0)
        )
        command_match = command_line_match or command_args_match or shell_match
        signal(
            "structure",
            "shell_command",
            _short_value(matched),
            0.94,
            "发现 Shell/CLI 命令或命令行结构",
            match=command_match,
            rule_id="structure:shell_command",
            negated=_is_negated(normalized, command_match.span()) if command_match else False,
            quoted=_is_quoted(normalized, command_match.span()) if command_match else False,
        )
        score(domain_scores, "shell", 0.96)
        score(domain_scores, "operations", 0.58)
        score(artifact_scores, "command", 0.9)
        if language is None:
            language = "shell"

    for label, pattern, strength in _INTENT_RULES:
        match = _search(pattern, normalized)
        if not match:
            continue
        negated = _is_negated(normalized, match.span())
        quoted = _is_quoted(normalized, match.span())
        if not negated:
            score(intent_scores, label, strength)
        signal(
            "intent_hint",
            label,
            _short_value(match.group(0)),
            strength,
            "发现高置信动作短语",
            match=match,
            rule_id=f"intent:{label}",
            negated=negated,
            quoted=quoted,
        )

    operations: list[str] = []
    for operation, pattern, strength in _OPERATION_RULES:
        match = _search(pattern, normalized)
        if not match:
            continue
        operations.append(operation)
        signal(
            "operation",
            operation,
            _short_value(match.group(0)),
            strength,
            "发现任务约束或技术操作",
            match=match,
            rule_id=f"operation:{operation}",
            negated=_is_negated(normalized, match.span()),
            quoted=_is_quoted(normalized, match.span()),
        )

    _score_domains_from_terms(normalized, domain_scores, signal)

    file_requirement_match = _search(_FILE_REQUIREMENT_RE, normalized)
    has_explicit_file_requirement = file_requirement_match is not None
    requires_files = has_file_paths or has_explicit_file_requirement
    if has_explicit_file_requirement and not has_file_paths:
        signal(
            "constraint",
            "requires_files",
            True,
            0.72,
            "请求显式引用文件、项目或附件",
            match=file_requirement_match,
            rule_id="constraint:requires_files",
            negated=_is_negated(normalized, file_requirement_match.span()),
            quoted=_is_quoted(normalized, file_requirement_match.span()),
        )

    explicit_execute = intent_scores.get("execute", 0.0) >= 0.65
    requires_tools = explicit_execute or bool(
        _search(_TOOL_REQUIREMENT_RE, normalized)
    ) and any(item in intent_scores for item in ("create", "modify", "debug", "execute"))
    if requires_tools:
        tool_match = _search(_TOOL_REQUIREMENT_RE, normalized)
        signal(
            "constraint",
            "requires_tools",
            True,
            0.82,
            "请求包含执行、部署或线上操作语义",
            match=tool_match,
            rule_id="constraint:requires_tools",
            negated=_is_negated(normalized, tool_match.span()) if tool_match else False,
            quoted=_is_quoted(normalized, tool_match.span()) if tool_match else False,
        )

    risk_reasons: list[str] = []
    for reason, pattern in _RISK_RULES:
        match = _search(pattern, normalized)
        if match:
            negated = _is_negated(normalized, match.span())
            quoted = _is_quoted(normalized, match.span())
            if not negated and not quoted:
                risk_reasons.append(reason)
            signal(
                "risk",
                reason,
                _short_value(match.group(0)),
                0.9,
                "发现可能影响外部系统或数据的操作",
                match=match,
                rule_id=f"risk:{reason}",
                negated=negated,
                quoted=quoted,
            )

    constraints = TaskConstraints(
        requires_files=requires_files,
        requires_tools=requires_tools,
        has_code_block=has_code_block,
        has_file_paths=has_file_paths,
        has_shell_commands=has_shell_commands,
        has_math_symbols=has_math_symbols,
        has_urls=has_urls,
    )
    return TaskAST(
        original_prompt=original,
        constraints=constraints,
        language=language,
        operations=tuple(dict.fromkeys(operations)),
        signals=tuple(signals),
        intent_scores=intent_scores,
        domain_scores=domain_scores,
        artifact_scores=artifact_scores,
        high_risk=bool(risk_reasons),
        risk_reasons=tuple(dict.fromkeys(risk_reasons)),
    )


_LANGUAGE_RULES: tuple[tuple[str, re.Pattern[str], float], ...] = (
    ("cpp", re.compile(r"c\+\+|\bcpp\b|\.cc\b|\.cpp\b|std::|#include\s*[<\"]", re.I), 0.98),
    ("python", re.compile(r"\bpython\s*3?\b|\.py\b|\bpip\b|\bpytest\b|\bpandas\b|\bdjango\b", re.I), 0.96),
    ("typescript", re.compile(r"\btypescript\b|\bts-node\b|\.tsx?\b", re.I), 0.96),
    ("javascript", re.compile(r"\bjavascript\b|\bnode(?:\.js)?\b|\bnpm\b|\.jsx?\b", re.I), 0.95),
    ("java", re.compile(r"\bjava\b|\bspring(?:\s+boot)?\b|\.java\b", re.I), 0.94),
    ("go", re.compile(r"\bgolang\b|\bgo\s+module\b|\.go\b", re.I), 0.94),
    ("rust", re.compile(r"\brust\b|\bcargo\b|\.rs\b", re.I), 0.94),
    ("swift", re.compile(r"\bswift\b|\bxcode\b|\.swift\b", re.I), 0.94),
    ("sql", re.compile(r"\bsql\b|\bpostgres(?:ql)?\b|\bmysql\b|\bselect\b.+\bfrom\b", re.I), 0.92),
    ("shell", re.compile(r"\bshell\b|\bbash\b|\bzsh\b|\bpowershell\b|\.sh\b", re.I), 0.94),
)

_INTENT_RULES: tuple[tuple[str, re.Pattern[str], float], ...] = (
    ("execute", re.compile(r"\b(?:run|execute|apply|deploy|rollout|install|restart|delete|remove)\b|(?:请|帮我|直接|立即)?(?:执行|运行|部署|上线|重启|删除|卸载)(?=\s*(?:命令|脚本|操作|这|该|一下|到|kubectl|terraform|docker|git|服务|程序|代码|文件|pod|[a-z0-9`./_-]))", re.I), 0.98),
    ("translate", re.compile(r"翻译|译成|翻成|\btranslate\b|translation", re.I), 0.97),
    ("summarize", re.compile(r"总结|摘要|提炼|概括|\bsummar(?:ize|y)\b", re.I), 0.96),
    ("extract", re.compile(r"提取|抽取|结构化|行动项|列出|\bextract\b", re.I), 0.94),
    ("classify", re.compile(r"分类|归类|判断属于|判断[^。！？]{0,12}归入|归入|\bclassif(?:y|ication)\b|categorize", re.I), 0.94),
    ("review", re.compile(r"审查|评审|检查|代码审阅|评估|\breview\b|maintainability|correctness risk|正确性风险", re.I), 0.96),
    ("debug", re.compile(r"修复|排查|排障|诊断|恢复|报错|异常|超时|失败|慢查询|性能问题|消费延迟|日志[^。！？]{0,18}(?:原因|错误|502|超时)|\bdebug\b|troubleshoot|bug", re.I), 0.97),
    ("modify", re.compile(r"修改|重构|润色|优化|改写|调整|升级|改成|\brefactor\b|\brewrite\b|\boptimize\b", re.I), 0.92),
    ("plan", re.compile(r"规划|计划|路线|路线图|拆解|排期|里程碑|\broadmap\b|\bschedule\b|(?:create|make|design|provide|give)\s+(?:a\s+)?(?:plan|roadmap)", re.I), 0.96),
    ("explain", re.compile(r"解释|说明|讲解|原理|为什么|如何工作|如何实现(?:的|原理|方式)?|是什么|什么是|\bexplain\b|\bhow does\b|\bwhat is\b|\bwhy\b", re.I), 0.95),
    ("analyze", re.compile(r"分析|对比|比较|评估|原因|风险|趋势|取舍|排优先级|\banaly[sz]e\b|\bcompare\b|\bevaluate\b|trade-?offs", re.I), 0.94),
    ("create", re.compile(r"写一|写个|写出|生成|创建|开发|搭建|构建|补充[^。！？]{0,12}(?:测试|配置|文档|ci)|设计(?:一个|一套)?|建立|自动检查|实现(?:一个|一套|程序|函数|代码|接口|模块)|\b(?:write|create|build|implement|generate)\b", re.I), 0.94),
)

_OPERATION_RULES: tuple[tuple[str, re.Pattern[str], float], ...] = (
    ("multithreading", re.compile(r"多线程|并发|并行|线程池|\bmultithread(?:ing)?\b|\bconcurren(?:cy|t)\b|\basync(?:hronous)?\b|\bawait\b|mutex|coroutine|lock", re.I), 0.95),
    ("cartesian_product", re.compile(r"笛卡尔积|笛卡尔乘积|\bcartesian\s+product\b", re.I), 0.95),
    ("routing", re.compile(r"路由协议|路由算法|\brouting\b|\brouter\b|\bmanet\b|\bbgp\b|\bosfp\b", re.I), 0.92),
    ("dns", re.compile(r"\bdns\b|域名解析|解析超时|\bbind9?\b|\bresolv", re.I), 0.92),
    ("http_api", re.compile(r"\bapi\b|\brest(?:ful)?\b|\bhttp\b|\bgrpc\b|接口", re.I), 0.78),
    ("database", re.compile(r"数据库|\bsql\b|\bpostgres(?:ql)?\b|\bmysql\b|\bmongodb\b|\bredis\b|索引|事务", re.I), 0.82),
    ("kubernetes", re.compile(r"\bkubernetes\b|\bk8s\b|\bkubectl\b|\bhelm\b", re.I), 0.92),
    ("deployment", re.compile(r"部署|发布|上线|回滚|\bdeploy(?:ment)?\b|\brollout\b|\bterraform\b|\bdocker\b", re.I), 0.9),
    ("testing", re.compile(r"测试|单元测试|集成测试|\btest(?:s|ing)?\b|\bpytest\b|\bci\b", re.I), 0.78),
)

_DOMAIN_TERM_RULES: tuple[tuple[str, re.Pattern[str], float], ...] = (
    ("programming", re.compile(r"代码|编程|函数|算法|程序|模块|类|接口|模拟器|仿真|重构|鉴权|分页|\bapi\b|\brest(?:ful)?\b|\bpull request\b|\bcode review\b|\bcode\b|\bprogram\b|\balgorithm\b|\bsoftware\b", re.I), 0.78),
    ("networking", re.compile(r"网络|协议|路由|\bnetwork\b|\bprotocol\b|\bdns\b|\btcp\b|\budp\b|\bhttp\b|\bmanet\b|\bfirewall\b", re.I), 0.78),
    ("operations", re.compile(r"生产|线上|集群|服务器|部署|监控|告警|云|运维|nginx|kafka|broker|\bdns\b|\bkubernetes\b|\bk8s\b|\bterraform\b|\bproduction\b|\bonline\b|\bcluster\b|\bserver\b|\bmonitoring\b|\bcloud\b|\bdevops\b", re.I), 0.8),
    ("shell", re.compile(r"命令行|终端|脚本|\bshell\b|\bterminal\b|\bscript\b|\bcli\b", re.I), 0.75),
    ("mathematics", re.compile(r"数学|公式|证明|方程|概率|积分|导数|矩阵|时间复杂度|\bmath(?:ematics)?\b|\bproof\b|\bequation\b|\btime complexity\b|\bbinary search\b", re.I), 0.78),
    ("data_analysis", re.compile(r"数据|日志|指标|统计|趋势|报表|研究报告|技术报告|技术方案|架构方案|\bdata\b|\blog(?:s)?\b|\bmetrics?\b|\bstatistics?\b|\bresearch report\b|\btechnical report\b|\btrade-offs?\b", re.I), 0.68),
    ("document", re.compile(r"邮件|简历|报告|文档|文案|段落|\bemail\b|\bresume\b|\breport\b|\bdocument\b|\bparagraph\b", re.I), 0.72),
    ("planning", re.compile(r"计划|路线|里程碑|排期|项目|优先级|流程|清单|方案|需求|开发拆解|摄影路线|\broadmap\b|\bmilestone\b|\bschedule\b|\bproject\b", re.I), 0.7),
    ("creative", re.compile(r"小说|故事|角色|诗歌|创意|\bstory\b|\bfiction\b|\bcharacter\b|\bcreative\b|\bpoem\b", re.I), 0.72),
)

_FILE_PATH_RE = re.compile(
    r"(?:^|[\s('" + '"' + r"`])((?:\.{1,2}/|~/|/(?:[a-z0-9_.-]+/)+|[a-z]:[\\/])[^\s'" + '"' + r"`,;)]+|[\w./~-]+\.(?:py|js|jsx|ts|tsx|java|go|rs|swift|cpp|cc|h|hpp|sql|tf|yaml|yml|json|csv|md|sh|toml|ini|conf)\b)",
    re.I,
)
_URL_RE = re.compile(r"https?://[^\s)\]}>]+", re.I)
_MATH_RE = re.compile(
    r"[∑∫∂√≈≤≥≠∞∈∀∃]|\\(?:frac|sum|int|sqrt|lim)|\b(?:equation|derivative|integral|matrix|proof|theorem)\b|(?<!\w)\d+\s*[+*/=]\s*\d+(?!\w)",
    re.I,
)
_SHELL_COMMAND_RE = re.compile(
    r"(?<![\w-])(?:kubectl|terraform|docker|git|curl|wget|ssh|scp|systemctl|journalctl|dig|nslookup|rm|mv|cp|chmod|chown|find|grep|sed|awk|npm|pip|make|helm|ansible|aws|az|gcloud|sudo)(?![\w-])|(?<![\w-])(?:python3?|go|cargo)\s+(?:[-A-Za-z0-9_./]+\.(?:py|go|rs)|-[a-z]|run\b)",
    re.I,
)
_COMMAND_LINE_RE = re.compile(
    r"(?m)^\s*(?:\$\s+|sudo\s+|(?:kubectl|terraform|docker|git|curl|wget|ssh|scp|systemctl|journalctl|dig|nslookup|rm|mv|cp|chmod|chown|find|grep|sed|awk|npm|pip|python3?|go|cargo|make|helm|ansible|aws|az|gcloud)\b)",
    re.I,
)
_COMMAND_WITH_ARGS_RE = re.compile(
    r"(?<![\w-])(?:kubectl|terraform|docker|git|curl|wget|ssh|scp|systemctl|journalctl|dig|nslookup|rm|mv|cp|chmod|chown|find|grep|sed|awk|npm|pip|make|helm|ansible|aws|az|gcloud|sudo|python3?|go|cargo)\s+[-A-Za-z0-9_./]+",
    re.I,
)
_FILE_REQUIREMENT_RE = re.compile(
    r"文件|附件|代码库|仓库|项目|模块|\b(?:file|attachment|repository|repo|project|module)\b", re.I
)
_TOOL_REQUIREMENT_RE = re.compile(
    r"线上|生产|集群|服务器|部署|发布|上线|执行|运行|\b(?:production|cluster|server|deploy|release|run|execute)\b", re.I
)
_RISK_RULES: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("production", re.compile(r"生产|线上|\bproduction\b|\bonline\b", re.I)),
    ("destructive", re.compile(r"删除|清空|销毁|\bdelete\b|\bdrop\b|\brm\s+-rf\b|\bremove\b", re.I)),
    ("privileged", re.compile(r"权限|提权|root|sudo|chmod\s+777|\bcredential|\bsecret|\btoken\b|密钥", re.I)),
    ("migration_or_release", re.compile(r"迁移|上线|发布|回滚|\bmigrat(?:e|ion)\b|\brelease\b|\brollback\b", re.I)),
)


def _detect_language(text: str) -> tuple[str | None, float, re.Match[str] | None]:
    for language, pattern, strength in _LANGUAGE_RULES:
        match = _search(pattern, text)
        if match:
            return language, strength, match
    return None, 0.0, None


def _score_domains_from_terms(text: str, scores: dict[str, float], signal) -> None:
    for domain, pattern, strength in _DOMAIN_TERM_RULES:
        match = _search(pattern, text)
        if not match:
            continue
        scores[domain] = max(scores.get(domain, 0.0), strength)
        signal(
            "domain_hint",
            domain,
            _short_value(match.group(0)),
            strength,
            "发现领域术语",
            match=match,
            rule_id=f"domain:{domain}",
            negated=_is_negated(text, match.span()),
            quoted=_is_quoted(text, match.span()),
        )


def _search(pattern: re.Pattern[str], text: str) -> re.Match[str] | None:
    return pattern.search(text)


def _normalize_with_spans(text: str) -> tuple[str, tuple[tuple[int, int], ...]]:
    """Normalize text while retaining a map back to original offsets."""

    normalized_parts: list[str] = []
    spans: list[tuple[int, int]] = []
    for index, character in enumerate(text):
        normalized = unicodedata.normalize("NFKC", character).casefold()
        normalized_parts.append(normalized)
        spans.extend((index, index + 1) for _ in normalized)
    return "".join(normalized_parts), tuple(spans)


def _span_to_original(
    span: tuple[int, int],
    normalized_spans: tuple[tuple[int, int], ...],
) -> tuple[int, int] | None:
    start, end = span
    if start < 0 or end > len(normalized_spans) or start >= end:
        return None
    return normalized_spans[start][0], normalized_spans[end - 1][1]


def _is_quoted(text: str, span: tuple[int, int]) -> bool:
    """Return whether a match is enclosed by common quote/code delimiters."""

    start, end = span
    before = text[max(0, start - 96):start]
    after = text[end:min(len(text), end + 96)]
    for left, right in (("`", "`"), ('"', '"'), ("'", "'"), ("“", "”"), ("「", "」"), ("『", "』")):
        if left == right:
            if before.count(left) % 2 == 1 and right in after:
                return True
            continue
        if before.rfind(left) > before.rfind(right) and right in after:
            return True
    return False


def _is_negated(text: str, span: tuple[int, int]) -> bool:
    """Return whether a nearby negation scopes over a matched signal."""

    start, _ = span
    prefix = text[max(0, start - 48):start]
    return bool(
        re.search(r"(?:不要|别|勿|禁止|不许|不可|不能|无需|不必|没必要|拒绝).{0,64}$", prefix)
        or re.search(r"(?:do\s+not|don't|never|not|without).{0,64}$", prefix, re.I)
    )


def _short_value(value: str, limit: int = 80) -> str:
    value = re.sub(r"\s+", " ", str(value)).strip()
    return value if len(value) <= limit else value[: limit - 1] + "…"
