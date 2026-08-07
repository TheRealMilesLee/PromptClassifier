from __future__ import annotations

import re
import unicodedata
from collections import Counter


# The model is intentionally vocabulary-light. Character n-grams make it work
# reasonably well for Chinese without requiring a separate segmentation package.
CJK_RUN_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]+")
LATIN_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_+#./-]*|\d+(?:\.\d+)?")


SIGNAL_PATTERNS: dict[str, tuple[str, ...]] = {
    "work_text": (
        r"文案",
        r"邮件",
        r"文章",
        r"润色",
        r"改写",
        r"翻译",
        r"摘要",
        r"总结",
        r"简历",
        r"报告",
        r"translate",
        r"translation",
        r"copywriting",
    ),
    "work_code": (
        r"代码",
        r"函数",
        r"脚本",
        r"程序",
        r"bug",
        r"报错",
        r"异常",
        r"接口",
        r"api",
        r"sql",
        r"python",
        r"javascript",
        r"typescript",
        r"java",
        r"c\+\+",
        r"仓库",
        r"编译",
        r"单元测试",
        r"重构",
        r"实现",
        r"refactor",
        r"repository",
    ),
    "work_operations": (
        r"部署",
        r"上线",
        r"监控",
        r"告警",
        r"服务器",
        r"集群",
        r"生产",
        r"dns",
        r"域名",
        r"证书",
        r"日志",
        r"故障",
        r"排障",
        r"kafka",
        r"kubernetes",
        r"k8s",
        r"容器",
        r"网络",
        r"nginx",
        r"terraform",
        r"cloudwatch",
        r"运维",
    ),
    "work_analysis": (
        r"分析",
        r"对比",
        r"比较",
        r"评估",
        r"研究",
        r"原因",
        r"为什么",
        r"原理",
        r"洞察",
        r"风险",
        r"趋势",
        r"explain",
        r"analy[sz]e",
        r"compare",
        r"evaluate",
    ),
    "work_planning": (
        r"规划",
        r"计划",
        r"路线",
        r"roadmap",
        r"拆解",
        r"排期",
        r"安排",
        r"方案",
        r"优先级",
        r"里程碑",
        r"plan",
        r"schedule",
        r"backlog",
    ),
    "action_create": (
        r"写",
        r"生成",
        r"创建",
        r"实现",
        r"搭建",
        r"开发",
        r"构建",
        r"produce",
        r"write",
        r"create",
        r"build",
        r"implement",
        r"generate",
    ),
    "action_modify": (
        r"修改",
        r"优化",
        r"润色",
        r"重构",
        r"调整",
        r"升级",
        r"修正",
        r"改成",
        r"refactor",
        r"rewrite",
        r"optimize",
        r"update",
    ),
    "action_debug": (
        r"修复",
        r"排查",
        r"排障",
        r"诊断",
        r"debug",
        r"troubleshoot",
        r"故障",
        r"超时",
        r"报错",
        r"失败",
        r"无法",
    ),
    "action_explain": (
        r"解释",
        r"说明",
        r"讲解",
        r"原理",
        r"如何工作",
        r"explain",
        r"how does",
        r"why",
    ),
    "action_analyze": (
        r"分析",
        r"比较",
        r"对比",
        r"评估",
        r"洞察",
        r"analy[sz]e",
        r"compare",
        r"evaluate",
    ),
    "action_summarize": (
        r"总结",
        r"摘要",
        r"提炼",
        r"概括",
        r"summarize",
        r"summary",
    ),
    "action_translate": (
        r"翻译",
        r"译成",
        r"翻成",
        r"translate",
        r"translation",
    ),
    "action_plan": (
        r"规划",
        r"计划",
        r"拆解",
        r"roadmap",
        r"plan",
        r"schedule",
    ),
    "action_review": (
        r"审查",
        r"评审",
        r"review",
        r"检查",
        r"质量",
        r"风险",
    ),
    "action_extract": (
        r"提取",
        r"抽取",
        r"结构化",
        r"列出",
        r"extract",
    ),
    "action_classify": (
        r"分类",
        r"归类",
        r"判断属于",
        r"classify",
        r"categorize",
    ),
    "scope_large": (
        r"多个",
        r"同时",
        r"并且",
        r"以及",
        r"完整",
        r"全套",
        r"从零",
        r"端到端",
        r"系统",
        r"跨",
        r"集成",
        r"迁移",
        r"end-to-end",
        r"multi",
        r"several",
    ),
    "external_system": (
        r"aws",
        r"云",
        r"服务器",
        r"数据库",
        r"api",
        r"github",
        r"gitlab",
        r"kubernetes",
        r"k8s",
        r"dns",
        r"生产",
        r"集群",
        r"网络",
        r"消息队列",
    ),
    "simple_request": (
        r"只需",
        r"简单",
        r"一句话",
        r"一个简单",
        r"单个文件",
        r"仅",
        r"only",
        r"just",
        r"one simple",
    ),
    "vague_request": (
        r"帮我弄一下",
        r"处理一下",
        r"优化一下",
        r"看看",
        r"搞一下",
        r"fix this",
        r"take a look",
    ),
}


SIGNAL_DISPLAY_NAMES = {
    "work_text": "文字内容",
    "work_code": "代码或软件开发",
    "work_operations": "基础设施或运行环境",
    "work_analysis": "分析、比较或解释",
    "work_planning": "计划、方案或排期",
    "work_mixed": "跨多个工作领域",
    "action_create": "创建或生成动作",
    "action_modify": "修改、优化或重构动作",
    "action_debug": "修复或排障动作",
    "action_explain": "解释或讲解动作",
    "action_analyze": "分析动作",
    "action_summarize": "总结或提炼动作",
    "action_translate": "翻译动作",
    "action_plan": "规划动作",
    "action_review": "评审或检查动作",
    "action_extract": "提取或结构化动作",
    "action_classify": "分类动作",
    "scope_large": "多个步骤或较大范围",
    "external_system": "外部系统或环境依赖",
    "simple_request": "单一或简单交付物",
    "vague_request": "目标描述偏模糊",
}


COMPILED_SIGNAL_PATTERNS = {
    name: tuple(re.compile(pattern, re.IGNORECASE) for pattern in patterns)
    for name, patterns in SIGNAL_PATTERNS.items()
}


def normalize_text(text: str) -> str:
    """Normalize width, case and whitespace while preserving useful punctuation."""

    value = unicodedata.normalize("NFKC", str(text)).lower()
    return re.sub(r"\s+", " ", value).strip()


def matched_signal_categories(text: str) -> set[str]:
    normalized = normalize_text(text)
    matched = {
        name
        for name, patterns in COMPILED_SIGNAL_PATTERNS.items()
        if any(pattern.search(normalized) for pattern in patterns)
    }
    primary_work_categories = matched & {
        "work_text",
        "work_code",
        "work_operations",
    }
    if len(primary_work_categories) >= 2:
        matched.add("work_mixed")
    return matched


def extract_feature_counts(text: str) -> Counter[str]:
    normalized = normalize_text(text)
    features: Counter[str] = Counter()

    for run in CJK_RUN_RE.findall(normalized):
        # Individual characters and short n-grams are robust to slightly
        # different Chinese wording and avoid a tokenizer dependency.
        for size in (1, 2, 3, 4):
            if len(run) < size:
                continue
            for index in range(len(run) - size + 1):
                features[f"cjk{size}:{run[index:index + size]}"] += 1
        if len(run) <= 12:
            features[f"cjkword:{run}"] += 1

    for token in LATIN_TOKEN_RE.findall(normalized):
        features[f"word:{token}"] += 1

    for category in matched_signal_categories(normalized):
        # Explicit domain/action signals are intentionally stronger than a
        # generic character n-gram. This is still a learned feature, but keeps
        # an unambiguous word such as "翻译" from disappearing in a tiny corpus.
        features[f"signal:{category}"] += 6 if category.startswith(("work_", "action_")) else 3

    compact = normalized.replace(" ", "")
    if "?" in normalized or "？" in normalized:
        features["meta:question"] += 1
    if "\n" in str(text) or "```" in str(text):
        features["meta:structured_input"] += 1
    if len(compact) <= 16:
        features["meta:short"] += 1
    elif len(compact) >= 120:
        features["meta:long"] += 1
    if len(re.findall(r"[。！？!?；;\n]", str(text))) >= 2:
        features["meta:multiple_sentences"] += 1

    return features
