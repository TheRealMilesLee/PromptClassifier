# 公共数据集与本地数据策略

TrendModel 不是聊天模型，也不是回答质量模型。它要从一段 prompt 判断路由所需的任务标签，因此数据集按“能补哪一种证据”选择，而不是按行数排序。

所有原始数据、清洗结果、训练过程和 artifact 都保留在本机。Hugging Face 只承担一次性下载；下载完成后，训练和评估只读取 `data/raw/huggingface/`，不使用在线 streaming，也不上传任何本地数据。

## 结论先说

现阶段最值得接入的是代码修改、工具执行、中文/混合语言表达和短指令意图这四类数据。没有任何一个已核验的公共数据集包含我们需要的真实 `active_minutes`，所以 effort 的核心训练数据必须来自内部脱敏任务记录。

当前建议分层：

| 层级 | 数据集 | 用途 |
| --- | --- | --- |
| 训练主料 | 内部人工 gold、`Code-Feedback`、`xLAM`、SWE-bench、少量 `MASSIVE`/`MTOP` | 学 workload、intent、复杂度 proxy 和工具/代码边界 |
| 训练辅助 | `COIG`、`OASST1`、`helpful-instructions`、少量 Dolly | 补中文、多语言和表达覆盖；标签必须标为 weak/unlabeled |
| 独立评估 | `BANKING77`、`CLINC150`、MTOP 的留出 split | 测试细粒度 intent、unknown/OOD 和短指令泛化 |
| 暂不接入 | LMSYS、无清晰许可证的数据集、只有单一领域的中文 instruction 大包 | 重复、许可证、领域偏差或标签收益不足 |

LMSYS-Chat-1M 的兼容读取代码仍保留，但本项目明确不下载、不清洗、不训练它。

## 已核验并有 adapter 的数据集

| 本地 source | Hugging Face 数据集 | 规模/关键字段 | 能补什么 | 许可证与边界 |
| --- | --- | --- | --- | --- |
| `wildchat` | [allenai/WildChat-4.8M](https://huggingface.co/datasets/allenai/WildChat-4.8M) | 约 4.8M；多轮 conversation | 广泛 prompt 分布、自然表达、中文/英文混用 | 体量大；只保留第一条 user prompt，不能当 effort gold |
| `arena` | [lmarena-ai/arena-human-preference-140k](https://huggingface.co/datasets/lmarena-ai/arena-human-preference-140k) | 约 140k；conversation、`is_code`、category、winner | 代码/非代码弱信号和困难 prompt 覆盖 | `is_code` 只是弱标签；winner 不是 effort |
| `xlam` | [Salesforce/xlam-function-calling-60k](https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k) | 约 60k；`query`、`tools`、`answers` | `execute`、工具调用、多步调用、工具数量 proxy | CC BY 4.0；数据卡要求登录/接受使用条件 |
| `swebench` | [SWE-bench/SWE-bench](https://huggingface.co/datasets/SWE-bench/SWE-bench) | 约 2.3k；`problem_statement`、`patch`、`test_patch`、repo | `code`、`modify`、debug、文件/仓库上下文 | 数据卡没有统一、充分的许可证说明；纳入产品前必须单独确认 |
| `swe_trajectories` | [nebius/SWE-agent-trajectories](https://huggingface.co/datasets/nebius/SWE-agent-trajectories) | 约 80k；`trajectory`、`generated_patch`、`target` | 多步工具轨迹、代码修改、复杂度 proxy | CC BY 4.0；agent 轨迹不等于人工工时 |
| `massive` | [AmazonScience/massive](https://huggingface.co/datasets/AmazonScience/massive) | 约 84 万；`utt`、`intent`、`scenario`、`locale` | 中文短指令、服务查询和执行边界 | CC BY 4.0；语音助手领域，不能当通用 workload gold |
| `oasst1` | [OpenAssistant/oasst1](https://huggingface.co/datasets/OpenAssistant/oasst1) | 约 88.8k；`text`、`role`、`lang` | 多语言 prompt 覆盖和无标签表示学习 | Apache-2.0；消息树需要按 prompt 去重 |
| `code_feedback` | [m-a-p/Code-Feedback](https://huggingface.co/datasets/m-a-p/Code-Feedback) | 约 66k；`messages` | 代码创建、编程表达和代码型 prompt | Apache-2.0；默认只取第一条 user message，并标为 `create` weak |
| `coig` | [BAAI/COIG](https://huggingface.co/datasets/BAAI/COIG) | 约 178k；`instruction`/`conversations`，中文 | 中文表达覆盖、混合指令和表示学习 | Apache-2.0；通用行没有可信 workload/intent 标签，默认不填标签 |
| `dolly` | [databricks/databricks-dolly-15k](https://huggingface.co/datasets/databricks/databricks-dolly-15k) | 约 15k；`instruction`、`context`、`category` | 由 category 提供粗粒度 summarize/extract/classify/explain/create | CC BY-SA 3.0；ShareAlike，先做合规确认，不作为默认主料 |
| `mtop` | [tasksource/mtop](https://huggingface.co/datasets/tasksource/mtop) | 约 104k；`question`、`intent`、`domain`、`lang`、logical form | 短指令 intent，尤其 `explain` vs `execute` | CC BY-SA 4.0；多语言任务型语音助手领域，不代表通用聊天分布 |
| `helpful_instructions` | [HuggingFaceH4/helpful-instructions](https://huggingface.co/datasets/HuggingFaceH4/helpful-instructions) | 约 148k；英文 instruction/demonstration | 英文任务表达覆盖、辅助 pseudo-label 语料 | Apache-2.0；没有可信 router workload/effort 标签，只能作为 unlabeled/weak 辅助 |

代码中的 adapter 会保留 `source`、`split_group`、`label_source`、`label_confidence` 和 `auxiliary_labels`。所有从公共字段映射出来的分类结果都带有 `weak` 标记，不能被报告为最终 gold 指标。

## 适合评估、暂不作为训练主料的数据集

这些数据集有明确的 intent 标签，但领域太窄或语言覆盖不符合主目标。它们更适合独立测试模型有没有把“短句意图分类”学会，而不是直接改变 TrendModel 的通用 workload 分布。

- [PolyAI/banking77](https://huggingface.co/datasets/PolyAI/banking77)：13,083 条、77 个银行客服 intent，CC BY 4.0。适合测细粒度 intent 和相似意图区分，不适合直接训练 `code`/`effort`。
- [DeepPavlov/clinc150](https://huggingface.co/datasets/DeepPavlov/clinc150)：约 23.8k、150 个助手 intent，并包含 OOD/unknown 设计。适合测 unknown 和开放域泛化；许可证和数据卡信息应在本地快照前再次确认。
- [tasksource/mtop](https://huggingface.co/datasets/tasksource/mtop)：已经有 adapter，但建议保留一个完全不参与训练的 validation/test 快照，用于检查短指令 intent 泛化。
- [HuggingFaceH4/helpful-instructions](https://huggingface.co/datasets/HuggingFaceH4/helpful-instructions)：已经有 adapter，适合补英文表达覆盖；默认不把它当成 gold，也不从它推导 effort。

`openai/gsm8k` 这类数学题数据可以作为高推理难例和独立评估集，但它没有我们的 workload/intent/active minutes 标注，暂不自动拼入主训练集。工具调用数据也要区分“存在函数调用”和“用户明确要求执行”：xLAM 已经覆盖前者，新的 tool-calling 混合数据在许可证、重复率和字段稳定性确认前不继续堆叠。

`Natural Instructions`、各种大规模中文 instruction/代码 instruction 镜像和新出现的 tool-calling 混合数据，虽然可能可下载，但很多数据卡只说明来源混合、许可证不完整或没有稳定的任务标签。许可证和字段来源没有确认前，不进入生产训练快照。

## Adapter 的标签边界

| source | 导出的 prompt | 当前弱标签 | 辅助证据 |
| --- | --- | --- | --- |
| `xlam` | `query` | `operations` + `execute` | 工具数、调用数、是否多步、`requires_tools` |
| `swebench` | `problem_statement` | `code` + `modify` | patch/test patch、repo、修改行数 |
| `swe_trajectories` | trajectory 中第一个 user prompt | `code` + `modify` | trajectory steps、target、exit status、patch |
| `massive` | `utt` | operations；按原始 intent 保守映射 | scenario、细粒度 intent、locale |
| `mtop` | `question` | `operations`；按 intent 映射 execute/explain | MTOP intent、domain、lang、logical form |
| `code_feedback` | `messages` 中第一条 user message | `code` + `create` | message count |
| `oasst1` | `role=prompter` 的 `text` | 无 | role、lang、质量字段 |
| `coig` | `instruction`/conversation 中的 instruction | 无 | 中文覆盖标记 |
| `dolly` | `instruction` | 按 category 映射粗粒度 intent | Dolly category |
| `helpful_instructions` | `instruction`/`prompt`/`text` | 无 | 英文 instruction 覆盖；可选 TaskTagger pseudo-label |

Adapter 永远不会从公共数据推导 `active_minutes`。工具调用次数、patch 行数、轨迹步数和 prompt 长度只能作为复杂度 proxy；它们不能伪装成人类实际投入时间。

## 本机下载

原始数据目录已被 `.gitignore` 排除，不应提交到代码仓库。下面的命令只把数据写到本机；如果某个数据卡要求先接受条款，需要先在浏览器完成确认。

```bash
mkdir -p data/raw/huggingface/{wildchat,arena,xlam,swebench,massive,mtop,code_feedback,coig,dolly,helpful_instructions}

hf download allenai/WildChat-4.8M \
  --repo-type dataset --include '*.parquet' \
  --local-dir data/raw/huggingface/wildchat --max-workers 8

hf download lmarena-ai/arena-human-preference-140k \
  --repo-type dataset --include '*.parquet' \
  --local-dir data/raw/huggingface/arena --max-workers 8

hf download Salesforce/xlam-function-calling-60k \
  --repo-type dataset --include '*.json*' \
  --local-dir data/raw/huggingface/xlam

hf download SWE-bench/SWE-bench \
  --repo-type dataset --revision refs/convert/parquet --include '*.parquet' \
  --local-dir data/raw/huggingface/swebench

hf download AmazonScience/massive \
  --repo-type dataset --revision refs/convert/parquet --include '*.parquet' \
  --local-dir data/raw/huggingface/massive

hf download tasksource/mtop \
  --repo-type dataset --include '*.parquet' \
  --local-dir data/raw/huggingface/mtop

hf download m-a-p/Code-Feedback \
  --repo-type dataset --include '*Code-Feedback*.jsonl*' \
  --local-dir data/raw/huggingface/code_feedback

hf download BAAI/COIG \
  --repo-type dataset --include '*.json*' --exclude '*.tar*' \
  --local-dir data/raw/huggingface/coig

hf download databricks/databricks-dolly-15k \
  --repo-type dataset --include '*dolly-15k*.jsonl*' \
  --local-dir data/raw/huggingface/dolly

hf download HuggingFaceH4/helpful-instructions \
  --repo-type dataset --include '*.json*' --exclude '*.tar*' \
  --local-dir data/raw/huggingface/helpful_instructions
```

OASST1 常用的本地文件是 `.jsonl.gz`，下载时应把该文件一并保留。大型 WildChat 不建议一开始全量进入清洗；先抽样验证字段、去重、脱敏和来源分布。

## 第一轮推荐混合比例

先做一个可解释的 `router-mix-v3`，不要把所有可下载数据直接拼接：

```bash
python3 -m trendmodel.train prepare \
  --local-raw-dir data/raw/huggingface \
  --local-sources wildchat,arena,xlam,swebench,massive,mtop,code_feedback,coig,helpful_instructions \
  --wildchat-limit 50000 \
  --arena-limit 20000 \
  --xlam-limit 15000 \
  --swebench-limit -1 \
  --massive-limit 20000 \
  --mtop-limit 10000 \
  --code-feedback-limit 20000 \
  --coig-limit 20000 \
  --helpful-instructions-limit 20000 \
  --languages en,zh \
  --output data/processed/router-mix-v3.jsonl \
  --report data/processed/router-mix-v3.report.json
```

准备后先检查报告里的 `source`、`label_source`、`label_confidence` 和每个字段的标签数量。仓库附带的 `data/hard_negatives.jsonl` 是一组可审计的人工边界样本，可以和种子数据一起作为训练 gold：

```bash
python3 -m trendmodel.train prepare \
  --input data/examples.jsonl data/hard_negatives.jsonl \
  --output data/processed/gold-plus-public.jsonl \
  --report data/processed/gold-plus-public.report.json
```

内部人工 gold 必须和公共语料一起输入，但不能被公共 weak 样本淹没；训练默认通过 `--gold-replay-multiplier 8` 做可信标签 replay。最终验证必须单独读取 `data/gold_validation.jsonl`，不能拿伪标签 holdout 当质量证明。

## Effort 的真实数据要求

公开语料可以帮助模型认识任务形态和复杂度 proxy，但不能提供 Router 真正需要的 effort 真值。要训练 effort head，至少要有下面这种内部脱敏记录：

```json
{
  "original_prompt": "请把这个模块重构一下并补充单元测试。",
  "workload_type": "code",
  "intent": "modify",
  "complexity": "medium",
  "active_minutes": 120,
  "effort_bucket": "M",
  "needs_clarification": false,
  "source": "internal_gold",
  "label_source": "manual_gold"
}
```

当前 TensorFlow 模型同时有 `effort_bucket` 分类头和 `effort_log_minutes` 回归头：bucket-only 样本只提供粗粒度路由先验；只有 `active_minutes` 才能训练真实分钟估计。训练时可信的人工/种子标签默认做 8 倍 replay weighting，避免几千条公共 weak/pseudo 样本淹没几十条内部 gold；这个倍率可以通过 `--gold-replay-multiplier` 调整，validation 不使用 replay。推理结果的 `basis` 会区分 `trained_observed_effort_and_bucket_heads`、`trained_effort_bucket_head` 和 `no_effort_supervision`。

`active_minutes` 应记录实际主动工作时间，不包括排队、审批、CI 或等待外部团队。第一批内部 gold 建议先收集 300–1,000 条，重点覆盖 `explain/create/modify/review/debug/execute` 的困难负样本。
