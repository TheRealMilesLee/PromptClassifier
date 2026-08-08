# 数据源与标注说明

`examples.jsonl` 是当前用于跑通训练管线的种子数据。旧字段 `text`、`work_type`、`effort` 仍然被 TensorFlow loader 兼容，但真实数据建议逐步迁移到下面的格式：

```json
{
  "id": "task-000001",
  "original_prompt": "请把这个模块重构一下并补充单元测试。",
  "workload_type": "code",
  "intent": "modify",
  "complexity": "medium",
  "effort_minutes": 120,
  "effort_bucket": "M",
  "needs_clarification": false,
  "source": "manual_annotation"
}
```

新的 Task IR tagger 还建议补充 `domain`（可以是字符串或字符串数组）和 `artifact`：

```json
{
  "original_prompt": "帮我实现一个 C++ 多线程程序。",
  "intent": "create",
  "domain": ["programming"],
  "artifact": "source_code",
  "workload_type": "code",
  "complexity": "medium",
  "effort_minutes": 90
}
```

`TaskTagger` 当前使用可读的语义原型作为第一版分类器；人工样本的主要价值是后续计算类别 centroid、训练轻量分类头和校准概率/effort，而不是把 baseline 输出重新当成真值。

建议的数据来源按可信度排序：

1. 已完成的真实任务记录：保留原始 prompt、人工标签、实际 active minutes 和是否返工。
2. Issue、工单、项目管理系统中的任务描述：导出后先脱敏，再由人工确认标签和 effort。
3. 当前 baseline 生成的弱标签：只作为预标注，必须人工抽样纠正，不能当作 ground truth。
4. 模板化合成数据：只用于验证 TensorFlow 训练/保存/服务链路，不能用于证明模型真实效果。

真实 effort 至少应区分 `active_minutes` 和等待时间。没有实际耗时的记录可以使用 XS/S/M/L/XL 的区间中值跑通训练，但 metadata 会记录这类样本不是 observed effort。

## 独立 gold validation 集

`WildChat`、`Arena` 和 `TaskTagger` 产生的标签不能作为最终质量评估的 gold label。
请把真实 router prompt 脱敏后复制到 `data/gold_validation.jsonl`，模板在
`data/gold_validation.template.jsonl`。实际 gold 文件已加入 `.gitignore`，不会被提交。

每条样本至少需要人工确认：

```json
{
  "id": "gold-000001",
  "original_prompt": "请把这个模块重构一下并补充单元测试。",
  "workload_type": "code",
  "intent": "modify",
  "complexity": "medium",
  "effort_minutes": 120,
  "effort_bucket": "M",
  "needs_clarification": false,
  "source": "internal_gold",
  "label_source": "manual_gold"
}
```

`effort_minutes` 必须是实际 active minutes，不要把等待审批、CI、排队或外部团队
等待时间算进去。没有真实耗时的样本可以保留 `effort_minutes: null`，但它不能用于
验证 effort head。

建议先收集 300–1,000 条，且覆盖这些容易混淆的任务对：

- `explain` vs `create`
- `create` vs `modify`
- `create` vs `review`
- `explain` vs `execute`
- Shell 解释 vs Shell 执行
- 文档总结 vs 文档改写

Gold 集应与训练数据分开保存，评估时使用 validation split 或完整 gold 文件，不能
把包含训练记录的全量公共语料结果当成泛化指标。

不要把包含姓名、邮箱、客户信息、密钥、token、内部域名或生产敏感信息的原始聊天直接放进训练集。建议先脱敏，并保留一个不可逆的样本 ID 以便后续修正标签。

## 公共 prompt 语料

当前本地训练快照使用两套 Hugging Face 数据集：

| 数据集 | 主要用途 | 不能直接当作 |
| --- | --- | --- |
| [WildChat-4.8M](https://huggingface.co/datasets/allenai/WildChat-4.8M) | 广泛的真实 prompt 分布、语言和任务表达 | effort 真值 |
| [Arena Human Preference 140k](https://huggingface.co/datasets/lmarena-ai/arena-human-preference-140k) | `is_code`、`category_tag` 和偏好分析辅助信号 | `winner` 对应的任务 effort |

LMSYS-Chat-1M 的适配代码仍保留作兼容，但当前项目明确不下载、不清洗、不训练该数据集。

推荐的运行边界是先把原始 parquet 拉到本机，再让后续流程只读取本地文件。这样训练不会依赖 Hub 的 streaming 连接，也便于固定数据快照：

```bash
mkdir -p data/raw/huggingface/wildchat
mkdir -p data/raw/huggingface/arena

hf download allenai/WildChat-4.8M \
  --repo-type dataset --include '*.parquet' \
  --local-dir data/raw/huggingface/wildchat --max-workers 8

hf download lmarena-ai/arena-human-preference-140k \
  --repo-type dataset --include '*.parquet' \
  --local-dir data/raw/huggingface/arena --max-workers 8
```

WildChat 当前 parquet 总量约 14GB，Arena 约 1.6GB。下载完成后，训练准备不再访问 Hub：

```bash
python -m trendmodel.train prepare \
  --local-raw-dir data/raw/huggingface \
  --local-sources wildchat,arena \
  --wildchat-limit 50000 \
  --arena-limit 20000 \
  --output data/processed/prompts.cleaned.jsonl \
  --report data/processed/cleaning_report.json
```

原始 parquet 不应提交到代码仓库；清洗后的 JSONL 也建议按内部数据策略管理。

准备公共数据前，需要先安装可选依赖。Python 3.14 使用 `datasets` 4/5 系列，避免旧 3.x 与新 pickle runtime 的兼容问题：

```bash
python -m pip install -r requirements-data.txt
```

当前只从本地下载的 WildChat 和 Arena 生成训练数据；入口只导出每条 conversation 的第一个 user prompt：

```bash
python -m trendmodel.train prepare \
  --sources wildchat,arena \
  --wildchat-limit 50000 \
  --arena-limit 20000 \
  --output data/processed/prompts.cleaned.jsonl \
  --report data/processed/cleaning_report.json
```

入口会丢弃 assistant 回复、模型名、IP、header、country/state 和时间等字段，并对明显的密钥、邮箱和电话号码做防御性替换。这个替换不是完整匿名化，正式使用前仍需做隐私和 license 审核。

默认导出的公共记录没有 `workload_type`、`intent`、`complexity` 和 `effort` 真值；Arena 中 `is_code=true` 会作为一个明确标记的弱 `code` 标签。`trendmodel.train prepare` 默认使用 TaskTagger 产生弱 workload/intent/complexity 标签：

```bash
python -m trendmodel.train prepare \
  --sources wildchat,arena \
  --wildchat-limit 50000 \
  --arena-limit 20000 \
  --output data/processed/prompts.cleaned.jsonl \
  --report data/processed/cleaning_report.json
```

伪标签的 `label_source` 会被标记为 `task_tagger_pseudo`，不能用于最终评估。旧的 `trendmodel.public_data --pseudo-label-data` 入口仍保留用于兼容，但新训练流程不再依赖旧 Naive Bayes baseline。`effort_minutes` 永远不会由公共语料或 teacher 自动填充；真实 effort 只能来自内部任务记录的 active minutes 或人工确认的 bucket。
