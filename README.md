# TrendModel

TrendModel 是给 Auto Router 使用的轻量级 prompt 标签模型。它只负责回答：

> 用户想做什么、涉及什么领域、需要什么能力、风险和 effort 大致如何？

它不负责生成最终答案，也不负责选择具体模型或执行工具。下游 router 消费这里产生的 Task IR，再根据质量、成本、上下文长度和权限选择模型。

## 给下游项目的集成入口

如果你要在另一个项目中引用最终 TensorFlow 模型，请先阅读
[下游集成手册](docs/INTEGRATION.md)。它包含：

- Python 直接加载 `model.keras` 的方式；
- OpenAI-compatible `/v1/chat/completions` 的启动和请求格式；
- 完整 JSON 字段、label、probability、confidence 和 effort 语义；
- 机器可读的 [task-analysis.schema.json](docs/task-analysis.schema.json)；
- artifact 版本、Metal/CUDA/CPU 行为和 Router fallback 建议。

想了解为什么采用当前的数据、模型结构、证据融合和 effort 监督方式，请阅读
[设计说明](docs/DESIGN.md)。

训练产物位于本机 `artifacts/trendmodel-text-v1/`，不提交到 Git。下游只需要复制整个 artifact 目录以及兼容的 Python/TensorFlow runtime。公共数据集的筛选、许可证和本机 adapter 见 [数据集说明](docs/DATASETS.md)。

## 当前架构

```text
用户 prompt
   ↓
句向量编码（可替换 backend）
   ↓
层级原型分类：intent / domain / artifact
   ↓
TaskAST：统一封装语言、文件、命令、数学符号、操作约束和风险 evidence
   ↓
概率融合：semantic 0.70 + structural 0.20 + context prior 0.10
   ↓
Task IR：标签、能力需求、风险、effort、置信度和证据
   ↓
另一个项目的 Auto Router
```

贝叶斯/概率层只做证据融合和不确定性管理，不承担中文语义理解。中文语义由句向量和原型完成；代码块、路径、Shell 命令、权限和生产环境等确定性信号由结构化提取器完成。

## 快速运行

默认使用无第三方依赖的 hash n-gram backend，可以直接在本机跑 smoke test：

```bash
python3 -m trendmodel tag "帮我实现一个 C++ 多线程程序，计算笛卡尔积"
```

也可以从标准输入读取：

```bash
echo "MANET 中的路由协议是什么，如何实现的？" | python3 -m trendmodel.tag_cli --compact
```

Python 调用：

```python
from trendmodel import TaskTagger

tagger = TaskTagger()
result = tagger.predict(
    "帮我实现一个 C++ 多线程程序，计算笛卡尔积",
    context={"in_project": True, "allow_modify_files": False},
)
payload = result.to_dict()
```

如果需要调试或审计结构化解析结果，可以直接读取 `TaskAST`；它是内部中间结构，不是最终 Router 输出：

```python
from trendmodel import TaskAST
from trendmodel.evidence import extract_evidence

ast: TaskAST = extract_evidence("请解释 ./src/main.cpp 中的 DNS 解析逻辑")
print(ast.to_dict())
```

输出包含这些核心字段；`predictions.*` 还提供 `entropy` 和 `margin`，便于 router 识别多个候选接近的请求：

```json
{
  "original_prompt": "帮我实现一个 C++ 多线程程序，计算笛卡尔积",
  "intent": {"label": "create", "probability": 0.27},
  "domains": [{"label": "programming", "probability": 0.27}],
  "workload_type": {"label": "code", "probability": 0.30},
  "artifact": "source_code",
  "operations": ["multithreading", "cartesian_product"],
  "language": "cpp",
  "capability_demand": {
    "reasoning": 0.66,
    "coding": 0.90,
    "context": 0.20,
    "tool_use": 0.05
  },
  "complexity": {"label": "medium"},
  "effort": {
    "label": "M",
    "time_range": "1–4 小时",
    "basis": "heuristic; calibrate with observed active_minutes"
  },
  "risk_level": "low",
  "confidence": 0.86,
  "needs_clarification": false,
  "evidence": [
    {"kind": "language", "label": "cpp", "strength": 0.98}
  ]
}
```

当前输出里的概率和 effort 是第一版 policy/heuristic，不能当作已经校准的准确率或真实工时。真正接入 router 前，应使用内部标注集校准概率、阈值和 effort 映射。

## 句向量 backend

默认 hash backend 只是离线 fallback，方便测试和开发，不具备真正的中文语义泛化能力。推荐在本地安装可选 embedding 依赖后使用中文或多语言 Sentence Transformer：

```bash
python3 -m pip install -r requirements-embedding.txt
python3 -m trendmodel tag \
  --embedding-backend sentence-transformers \
  --embedding-model BAAI/bge-small-zh-v1.5 \
  "帮我实现一个 C++ 多线程程序"
```

模型名称、许可证、下载来源和运行时内存需要在产品分发前单独确认。`TaskTagger` 只依赖 `Embedder` 接口，因此后续可以替换成 TensorFlow SavedModel、TFLite、Core ML 或项目自己的句向量 backend，而不用改 Task IR 和 router 协议。

## 标签层级

第一层是意图：

```text
explain / create / modify / review / debug / execute
summarize / translate / plan / classify / extract / analyze / unknown
```

第二层是领域：

```text
conversation / creative / programming / mathematics / shell
networking / document / data_analysis / operations / planning / other
```

第三层是预期产物：

```text
source_code / command / technical_explanation / document
analysis_report / plan / answer / unknown
```

另外会提取语言、并发、路由、DNS、数据库、Kubernetes、测试等 operations，并生成 `CapabilityDemand`：

```text
reasoning / coding / context / tool_use
```

这使得下游 router 可以区分“解释 C++ 多线程代码”和“生成 C++ 多线程程序”，而不必从一个扁平的 `code` 标签里猜测任务难度。

## 数据和训练边界

公共数据可以提供 prompt 分布、语言变化和部分弱标签，但不能凭空产生 effort 真值。当前推荐的数据组合、许可证和本地下载方式见 [数据集说明](docs/DATASETS.md)。第一轮可以从下面的受控混合开始：

- WildChat/Arena：广泛 prompt 分布和代码弱信号；
- Code-Feedback、xLAM、SWE-bench：代码创建、工具执行和代码修改；
- MASSIVE、MTOP、COIG：中文/多语言短指令和表达覆盖。

LMSYS-Chat-1M 的兼容读取代码仍保留，但本项目当前不下载、不清洗、不训练它。

一次性下载入口（后续训练只读本机文件）：

```bash
python3 -m pip install -r requirements-data.txt
hf download allenai/WildChat-4.8M \
  --repo-type dataset --include '*.parquet' \
  --local-dir data/raw/huggingface/wildchat
hf download lmarena-ai/arena-human-preference-140k \
  --repo-type dataset --include '*.parquet' \
  --local-dir data/raw/huggingface/arena
```

公共记录只保留第一个 user prompt，并丢弃 assistant 回复、模型名、IP、header 等 metadata；脚本会对明显密钥、邮箱和电话号码做防御性替换。正式训练前仍需做 license、隐私和重复样本审核。

正式的数据准备入口是：

```bash
# 先用仓库内 48 条种子样本验证清洗、弱标注和 JSONL 输出
python3 -m trendmodel.train prepare \
  --input data/examples.jsonl \
  --output data/processed/prompts.cleaned.jsonl \
  --report data/processed/cleaning_report.json

# 从已经落盘的 parquet 抽取本地 prompt；这里的数量只是示例，先小后大
python3 -m trendmodel.train prepare \
  --local-raw-dir data/raw/huggingface \
  --local-sources wildchat,arena \
  --wildchat-limit 50000 \
  --arena-limit 20000 \
  --languages en,zh \
  --output data/processed/prompts.cleaned.jsonl \
  --report data/processed/cleaning_report.json
```

`prepare` 会先清洗和去重，再用 `TaskTagger` 产生 workload/intent/complexity 的弱标签。报告会保留 reject reason、来源、teacher 版本和每个字段的 label provenance。它不会生成 `effort_minutes`；如果公共数据没有真实 effort，TensorFlow 的 effort head 会保持未监督。

真实的 effort 训练数据应来自已完成任务：

```json
{
  "original_prompt": "请把这个模块重构一下并补充单元测试。",
  "intent": "modify",
  "domain": ["programming"],
  "artifact": "source_code",
  "workload_type": "code",
  "complexity": "medium",
  "effort_minutes": 120,
  "needs_clarification": false,
  "source": "manual_annotation"
}
```

`effort_minutes` 应记录实际 active minutes，而不是等待审批、CI 或外部团队的墙上时间。公共数据和 baseline 伪标签只能用于预训练或预标注，不能混入最终评估集作为 gold label。

## TensorFlow 说明

TensorFlow 多任务模型现在是训练主路径：

```bash
python3 -m trendmodel.train train \
  --data data/processed/prompts.cleaned.jsonl \
  --artifact artifacts/trendmodel-text-v1 \
  --epochs 20

python3 -m trendmodel.train evaluate \
  --data data/processed/prompts.cleaned.jsonl \
  --artifact artifacts/trendmodel-text-v1

python3 -m trendmodel.tf_train predict \
  --artifact artifacts/trendmodel-text-v1 \
  "帮我实现一个 C++ 多线程程序"
```

模型是 `TextVectorization(character) + Embedding + BiGRU + character n-gram TF-IDF branch + integrated TaskAST feature branch + multi-task heads`，同时预测 workload、intent、complexity、effort bucket、effort minutes 和是否需要澄清。训练时对稀有类别使用 capped inverse-sqrt sample weighting，避免 `low` 或多数 workload 把 head 吸成单一类别；可信的人工/种子标签默认通过 `--gold-replay-multiplier 8` 做训练 replay，避免公共 weak/pseudo 数据淹没内部 gold。TaskAST 在训练和推理时都作为 TensorFlow 的结构化输入特征，推理 JSON 原样返回可审计 evidence，但不会再由 TaskTagger 或规则对 TensorFlow 概率做 post-hoc fusion。除兼容旧字段外，TensorFlow 还返回与 prototype `TaskIR.to_contract()` 对齐的 `task_ir` canonical object。缺失字段通过 sample weight mask；公共数据可以训练分类头，但没有真实 effort supervision 时不会伪造分钟真值。保存的 artifact 包括 `model.keras` 和 `metadata.json`，推理 JSON 在保留旧 scalar 字段的同时，新增 `predictions` 概率结构和 effort bucket candidates，方便 router 使用候选和置信度。旧的 text-only artifact 仍可加载，新训练产物使用模型版本 `0.3.0`。

当前仓库的默认解释器是 Python 3.14。它可以运行清洗、TaskTagger 和 Hugging Face `datasets` 4/5 系列，但仓库固定的 TensorFlow 2.16.2 没有 Python 3.14 wheel。本机真正训练需要单独准备 Python 3.11/3.12 环境，再按硬件安装对应 runtime；这不是代码训练失败，而是运行时 wheel 约束。

例如使用已安装的 Homebrew Python 3.12：

```bash
/opt/homebrew/bin/python3.12 -m venv .venv-tf
.venv-tf/bin/pip install -r requirements-tensorflow-macos.txt
.venv-tf/bin/python -m trendmodel.train train \
  --data data/processed/training.jsonl \
  --artifact artifacts/trendmodel-text-v1 \
  --epochs 10 \
  --require-gpu
```

`--require-gpu` 会在 TensorFlow 没有看到 Metal/CUDA GPU 时直接失败，避免训练过程
悄悄退回 CPU。省略这个参数则允许 CPU fallback。训练后的
`artifacts/trendmodel-text-v1/metadata.json` 会记录 `physical_devices`、
`gpu_devices`、`accelerator` 和 CUDA build 信息，可用来确认本次训练实际使用的设备。

### NVIDIA CUDA（RTX 4070）

RTX 4070 可以运行同一套 TensorFlow 模型。推荐使用 Linux 或 WSL2；当前
TensorFlow 版本不提供 Windows 原生 GPU 训练路径。先确认 NVIDIA 驱动可见：

```bash
nvidia-smi
python3.12 -m venv .venv-tf-cuda
.venv-tf-cuda/bin/python -m pip install --upgrade pip
.venv-tf-cuda/bin/pip install -r requirements-tensorflow-cuda.txt
```

检查 TensorFlow 是否识别 CUDA：

```bash
.venv-tf-cuda/bin/python - <<'PY'
import tensorflow as tf

print("tensorflow:", tf.__version__)
print("built_with_cuda:", tf.test.is_built_with_cuda())
print("devices:", tf.config.list_physical_devices())
print("gpus:", tf.config.list_physical_devices("GPU"))
PY
```

正式训练使用同一个 `trendmodel.train` 入口：

```bash
.venv-tf-cuda/bin/python -m trendmodel.train train \
  --data data/processed/training.jsonl \
  --artifact artifacts/trendmodel-text-v1-cuda \
  --epochs 10 \
  --batch-size 16 \
  --require-gpu
```

训练 metadata 中的 `runtime.accelerator` 应为 `cuda`。如果没有 GPU，
`--require-gpu` 会直接失败，不会把 CUDA 训练静默降级成 CPU。

### 评估口径

评估默认只看固定的 validation split，并同时输出 raw neural head 和 production
inference contract；两者都使用训练好的 TensorFlow heads 和内置 TaskAST feature branch：

```bash
.venv-tf/bin/python -m trendmodel.train evaluate \
  --data data/processed/training.jsonl \
  --artifact artifacts/trendmodel-text-v1 \
  --split validation \
  --mode both \
  --ablation all
```

其中：

- `--mode raw`：评估 TensorFlow heads 的结果；TaskAST 已经在模型输入中，不做 post-hoc fusion；
- `--mode inference`：评估生产 JSON，额外返回 TaskAST evidence，但不修改概率；
- `--mode fused`：兼容旧命令的别名，实际等同于 `inference`，不再执行旧的概率融合；
- `--ablation all`：额外输出 `text_only`、`text_ast` 和 `evidence_only` 三条诊断路径；
- `--split validation`：默认的留出集；
- `--split all`：只用于数据诊断，包含训练样本，不能当作泛化指标。

消融报告中，`text_only` 是把已训练模型的结构化输入置零，`text_ast` 是当前生产
输入，`evidence_only` 是不经过神经网络的确定性规则 baseline。它们用于判断语义
文本、TaskAST 输入和规则证据各自贡献多少，不代表三个可选生产模型。

独立 gold 集准备好后，使用：

```bash
.venv-tf/bin/python -m trendmodel.train evaluate \
  --data data/gold_validation.jsonl \
  --artifact artifacts/trendmodel-text-v1 \
  --split all \
  --mode both
```

`data/gold_validation.template.jsonl` 和标注要求见 [数据说明](data/README.md)。

项目同时提供本地 HTTP/OpenAI-compatible adapter。它不是远程大模型服务，而是对本地 TensorFlow artifact 的协议包装；启动方式、请求限制和响应解析见 [下游集成手册](docs/INTEGRATION.md)。如果下游与模型位于同一 Python 进程，优先直接调用 `TensorFlowTaskModel`，避免额外的 HTTP 和 JSON 字符串解析。

## 验证

```bash
python3 -m unittest discover -s tests -v
```

测试覆盖旧 baseline/TensorFlow 数据管线、公共数据脱敏和缺失 effort mask，以及新的中文/中英混合 prompt、意图冲突、结构化证据和概率融合。
