# TrendModel 下游集成手册

本文档是另一个项目接入 TrendModel 的主入口。TrendModel 只负责把用户 prompt 转换成路由标签和不确定性信息，不负责生成最终答案、选择具体供应商模型或执行工具。

当前推荐 artifact：`trendmodel-text-v1`。重新训练后模型版本为 `0.5.0`，默认
`metadata.architecture_family=bigru`，包含内置 TaskAST structured feature branch、
受控类别重加权、`effort_bucket` 分类头、ordinal effort head 和 abstention policy；
已有 text-only/旧 bucket artifact 仍可被兼容加载。仓库还提供
`architecture_family=transformer-challenger` 的可选对比 artifact；它使用相同的
TaskAST、heads 和 JSON contract，但不能因为架构不同就直接与主模型的指标混写。

## 1. 选择接入方式

有两种等价的本地接入方式：

| 方式 | 适用场景 | 入口 |
| --- | --- | --- |
| Python 直接调用 | Router 与 TrendModel 在同一个 Python 进程或同一台机器上 | `TensorFlowTaskModel.load(...).predict(...)` |
| OpenAI-compatible HTTP | Router 使用 OpenAI SDK、跨进程调用或希望隔离 TensorFlow runtime | `trendmodel.api` 的 `/v1/chat/completions` |

两种方式返回同一套分析 JSON。HTTP 方式只是把 JSON 放在 OpenAI response 的 `choices[0].message.content` 字符串里。

如果下游项目不需要 TensorFlow artifact，而只需要不带训练模型的可解释原型分类器，可以直接使用 `TaskTagger`。但它与最终 TensorFlow artifact 不是同一个模型，不能混用版本或把两者的概率直接比较。

## 2. Python 直接调用（推荐）

### 安装和路径

TensorFlow 当前固定使用 Python 3.11/3.12 runtime。Apple Silicon 使用 Metal
依赖，Linux/WSL2 NVIDIA 机器使用 CUDA 依赖。下游项目可以复用已有环境，也可以在自己的环境安装本项目：

```bash
cd /path/to/TrendModel
/opt/homebrew/bin/python3.12 -m venv .venv-tf
.venv-tf/bin/python -m pip install -r requirements-tensorflow-macos.txt
.venv-tf/bin/python -m pip install -e .
```

RTX 4070 等 NVIDIA GPU 请在 Linux 或 WSL2 中使用单独的环境：

```bash
python3.12 -m venv .venv-tf-cuda
.venv-tf-cuda/bin/python -m pip install --upgrade pip
.venv-tf-cuda/bin/pip install -r requirements-tensorflow-cuda.txt
.venv-tf-cuda/bin/python -m pip install -e .
```

当前 TensorFlow 版本不提供 Windows 原生 GPU 训练路径；Windows 主机请使用 WSL2。
训练 artifact 的 `metadata.json` 会记录 `runtime.accelerator`，其值通常为
`metal`、`cuda` 或 `cpu`。

推理只需要 artifact，不需要 Hugging Face 数据、原始 parquet 或训练 JSONL。默认 artifact 目录是：

```text
artifacts/trendmodel-text-v1/
├── model.keras
└── metadata.json
```

加载前建议检查 `metadata.json`：

```json
{
  "model_id": "trendmodel-text-v1",
  "model_version": "0.5.0",
  "architecture_family": "bigru",
  "post_hoc_probability_fusion": false
}
```

如果使用 challenger，`architecture_family` 应为
`transformer-challenger`。下游接口不需要根据这个字段分支解析 JSON，但应把它
记录到路由日志，便于比较延迟、质量和回滚来源。

### 最小调用

```python
import json

from trendmodel.tf_inference import TensorFlowTaskModel


classifier = TensorFlowTaskModel.load("artifacts/trendmodel-text-v1")
analysis = classifier.predict("帮我实现一个 C++ 多线程程序，计算笛卡尔积")

print(json.dumps(analysis, ensure_ascii=False, indent=2))
```

`predict` 接受一个非空字符串，返回普通 Python `dict`，可以直接交给 `json.dumps`、FastAPI、Pydantic adapter 或 Router 内部的 Task IR 转换器。

批量接口保持相同 JSON contract：

```python
analyses = classifier.predict_many([
    "解释 MANET 中的路由协议",
    "请把这段英文翻译成中文",
])
```

当前 `predict_many` 会一次执行 batched TensorFlow inference，并返回与单条 `predict` 相同的 contract；`batch_size` 可按本机内存和延迟目标调整。TaskAST 仍会逐条提取，这是确定性预处理，不会改变模型概率。

### 下游建议的最小适配器

```python
from typing import Any


def analyze_for_router(classifier: Any, prompt: str) -> dict[str, Any]:
    result = classifier.predict(prompt)
    workload = result["predictions"]["workload_type"]
    intent = result["predictions"]["intent"]
    complexity = result["predictions"]["complexity"]

    return {
        "prompt": result["original_prompt"],
        "workload_type": workload["label"],
        "workload_probability": workload["probability"],
        "intent": intent["label"],
        "intent_probability": intent["probability"],
        "complexity": complexity["label"],
        "complexity_probability": complexity["probability"],
        "effort": result["effort"],
        "needs_clarification": result["needs_clarification"],
        "overall_confidence": result["confidence"]["overall"],
        "evidence": result["evidence"],
    }
```

建议下游保留完整的 `predictions`、`confidence` 和 `evidence`，不要只保存三个 scalar label。Router 后续做阈值调整、校准或回放分析时需要这些信息。

## 3. OpenAI-compatible HTTP 调用

### 启动本地服务

在能够加载 TensorFlow artifact 的 Python 环境中安装 serving 依赖：

```bash
.venv-tf/bin/python -m pip install -r requirements-serving.txt
```

启动：

```bash
.venv-tf/bin/python -m trendmodel.api \
  --artifact artifacts/trendmodel-text-v1 \
  --host 127.0.0.1 \
  --port 8000
```

当前服务只绑定本机回环地址。除非明确配置反向代理、鉴权和网络 ACL，不要直接暴露到局域网或公网。

### 健康检查和模型列表

```bash
curl http://127.0.0.1:8000/healthz
curl http://127.0.0.1:8000/v1/models
```

`/healthz` 只表示进程和 artifact 已加载，不代表分类质量已经校准。

### Chat Completions 请求

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "trendmodel-text-v1",
    "messages": [
      {
        "role": "user",
        "content": "帮我实现一个 C++ 多线程程序，计算笛卡尔积"
      }
    ],
    "stream": false
  }'
```

响应遵循 OpenAI 外层结构，但分析结果是 JSON 字符串：

```json
{
  "id": "chatcmpl-...",
  "object": "chat.completion",
  "created": 1760000000,
  "model": "trendmodel-text-v1",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "{\"original_prompt\":\"...\",\"workload_type\":\"code\",...}"
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 25,
    "completion_tokens": 800,
    "total_tokens": 825
  }
}
```

下游必须对 `choices[0].message.content` 再执行一次 JSON parse：

```python
import json

response = openai_client.chat.completions.create(
    model="trendmodel-text-v1",
    messages=[{"role": "user", "content": prompt}],
    stream=False,
)
analysis = json.loads(response.choices[0].message.content)
```

当前 adapter 的边界：

- 只接受文本消息；多模态 content part 会返回 400；
- 只提取 `role=user` 的消息，并按换行拼接多个 user message；
- `system`、`developer`、`assistant` 消息不会进入分类文本；
- `stream=true` 不支持，会返回 400；
- `temperature`、`top_p`、`max_tokens` 等生成参数不会改变分类结果；
- `usage` 是字符数量的近似值，不是某个 LLM tokenizer 的真实 token 数；
- 当前没有鉴权、限流和多模型路由，这些属于下游服务边界。

## 4. TensorFlow 输出 contract

机器可读 schema 位于：[task-analysis.schema.json](./task-analysis.schema.json)。当前输出最外层字段如下：

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `original_prompt` | `string` | 原始输入文本；模型只在内部规范化换行，返回值保留原文本 |
| `workload_type` | `string` | 兼容旧 Router 的 scalar workload 标签 |
| `intent` | `string` | 兼容旧 Router 的 scalar intent 标签 |
| `complexity` | `string` | `low`、`medium` 或 `high` |
| `predictions` | `object` | 带 probability 和 Top 3 candidates 的推荐结构 |
| `effort` | `object` | effort bucket、粗略分钟数、区间和监督来源 |
| `evidence` | `array` | 可观察的结构化信号，不是模型自由生成的解释 |
| `task_ir` | `object` | 与 prototype `TaskIR.to_contract()` 相同的 canonical Router contract |
| `fusion` | `object` | TensorFlow 模型、TaskAST schema 和是否启用 post-hoc probability fusion |
| `needs_clarification` | `boolean` | 是否需要补充信息或人工确认 |
| `clarification_reasons` | `array[string]` | 触发澄清的原因 |
| `abstention` | `object` | 低置信度拒识状态、原因和阈值 |
| `confidence` | `object` | 各 head 最高分和整体路由分数 |
| `model_version` | `string` | artifact metadata 中的模型版本 |

### `predictions` 的分类 head

`predictions.workload_type`、`predictions.intent` 和 `predictions.complexity` 结构相同：

```json
{
  "label": "code",
  "probability": 0.81,
  "candidates": [
    {"label": "code", "probability": 0.81},
    {"label": "text", "probability": 0.10},
    {"label": "mixed", "probability": 0.04}
  ]
}
```

`probability` 是该 TensorFlow head 输出分布中选中类别的分数，范围为 `[0, 1]`。TaskAST 已经作为结构化输入参与模型计算，但推理阶段不会再把 evidence 作为第二次规则概率修正。新训练 artifact 会基于 validation split 做 temperature scaling；旧 artifact 或 validation 标签质量不足时，该分数仍不应直接解释成“模型有 81% 的真实正确率”。

`candidates` 只展示 Top 3。完整内部分布的总和为 1，但 Top 3 的分数加起来通常小于 1。

### `task_ir` canonical contract

`task_ir` 是 TensorFlow 与 prototype 的共同接口，版本由
`task_ir.contract_version == "task-ir-v1"` 标识。下游 Router 新代码应优先读取它，
而不是依赖 TensorFlow 为兼容旧 Router 保留的顶层 scalar 字段。

它包含：

- `intent`、`domains`、`workload_type`：统一的 `label + probability + candidates`；
- `artifact`、`operations`、`language`、`constraints`；
- `capability_demand`、`complexity`、`effort`、`risk_level`；
- `confidence`、`needs_clarification`、`evidence`、`fusion` 和 `model_version`。

TensorFlow 当前没有独立训练的 domain/artifact heads，因此 canonical contract 中
这两个字段明确来自 TaskAST evidence，不会伪装成神经 head 的输出。prototype 的
`TaskIR.to_dict()` 同时保留旧的直接字段，并增加同样的 `task_ir` canonical object。

分类标签集合：

```text
workload_type:
  text / code / analysis / operations / planning / mixed / other

intent:
  create / modify / debug / explain / execute / analyze / review
  summarize / translate / plan / extract / classify / unknown

complexity:
  low / medium / high
```

当请求过短、证据不足或需要确认时，scalar 字段和 `predictions.*.label` 可能是 `unknown`。Router 不应把 `unknown` 当成新的 workload 类型，而应把它视为“不确定，需要 fallback 或澄清”。

### `effort`

```json
{
  "bucket": "M",
  "estimated_minutes": 75,
  "range_minutes": [45, 120],
  "probability": 0.62,
  "confidence": 0.19,
  "basis": "trained_effort_bucket_head"
}
```

当前 bucket policy：

| Bucket | 当前区间 |
| --- | --- |
| `XS` | 5–15 分钟 |
| `S` | 15–45 分钟 |
| `M` | 45–120 分钟 |
| `L` | 120–480 分钟 |
| `XL` | 1,440–4,320 分钟 |

当前 artifact 的 `observed_effort_samples=0`，所以即使有 `effort_bucket`/ordinal effort head，`basis` 也只能是对应的 bucket-only basis 或旧 artifact 的 `bucket_seed_only`。这表示模型只学到了极粗的 bucket 先验，没有真实 active minutes 监督。下游应优先使用 `bucket` 做排序，不应把 `estimated_minutes` 当成承诺工时。

加入真实耗时后，basis 会变成 `trained_observed_effort_and_bucket_heads` 或 `trained_observed_effort_and_ordinal_heads`；如果完全没有 effort supervision，basis 会是 `no_effort_supervision`，分钟数为 `null`。

### `confidence`

每个 `predictions.<head>` 现在还会返回完整的 `distribution`。新训练 artifact 会用 validation split 拟合 temperature scaling，`metadata.calibration` 记录方法、样本数、NLL 和 ECE；旧 artifact 没有这个 metadata 时，distribution 仍然只是未校准的模型分数，不能解释为真实正确率。

```json
{
  "workload_type": 0.81,
  "intent": 0.83,
  "complexity": 0.75,
  "effort": 0.17,
  "overall": 0.70
}
```

这些值是路由不确定性信号，不是校准后的准确率。每个 `predictions.*` head 还会返回 `entropy` 和 `margin`：前者是归一化分布熵，越高越不确定；后者是 top-1 与 top-2 概率差，越低越容易混淆。特别是：

- `workload_type`、`intent`、`complexity` 通常是对应 head 的最大 posterior；
- `effort` 受监督数据质量影响，当前应明显低权重；
- `overall` 是多个 head 的加权汇总，不是独立概率事件；
- 低 confidence 应触发便宜模型 fallback、用户确认或保守路由。
- `entropy`/`margin` 可作为排序和拒识信号；新 artifact 的分布已做 validation calibration，但独立 gold 集仍应作为生产阈值的最终依据。

### `evidence`

典型信号如下：

```json
{
  "kind": "language",
  "label": "cpp",
  "value": "c++",
  "strength": 0.98,
  "detail": "发现编程语言或语言特有标记",
  "source_span": [7, 10],
  "rule_id": "language:cpp",
  "negated": false,
  "quoted": false
}
```

`strength` 是结构化 extractor 对该信号的强度，不是最终类别 probability。`source_span`
使用原始 prompt 的字符区间，`rule_id` 用于规则回放；`negated=true` 表示文本否定了
该动作，`quoted=true` 表示它更可能是被引用、解释或讨论的内容。常见 `kind` 包括
`language`、`intent_hint`、`operation`、`domain_hint`、`structure`、`constraint` 和 `risk`。

## 5. 推荐的 Router 决策方式

建议把 TrendModel 当作“分析器”，不要把它当作最终模型选择器：

```text
TrendModel JSON
  ↓
检查 abstention / needs_clarification / unknown / risk
  ↓
读取 workload、intent、complexity、effort bucket
  ↓
结合候选模型能力、价格、上下文长度和权限
  ↓
选择最终生成模型
```

一个可以先用于开发环境的 policy 示例：

```python
def should_fallback(analysis: dict) -> bool:
    predictions = analysis["predictions"]
    confidence = analysis["confidence"]

    if analysis["needs_clarification"]:
        return True
    if "unknown" in {
        predictions["workload_type"]["label"],
        predictions["intent"]["label"],
    }:
        return True
    if confidence["overall"] < 0.55:
        return True
    if predictions["workload_type"]["probability"] < 0.60:
        return True
    return False
```

这只是初始阈值，不是经过 calibration 的生产策略。真正接入前，应使用下游项目自己的 gold 集调节阈值，并记录：

- TrendModel 的完整 JSON；
- 最终选择的模型；
- 实际质量反馈；
- 实际 active minutes；
- 是否发生人工纠正或重新路由。

安全边界应优先于性价比：

- `intent=execute` 时必须检查工具权限；
- evidence 中出现 `risk` 或 `requires_tools` 时不能只靠低 confidence 自动执行；
- `unknown`、`needs_clarification=true` 时不应强行给出高风险自动路由；
- `effort.basis=bucket_seed_only` 时不要用分钟数做硬 SLA 承诺。

## 6. 训练、artifact 和版本管理

### 从本地数据训练

训练不访问 Hugging Face；Hub 只负责一次性下载原始 parquet。训练数据和模型均留在本机：

```bash
# 生成本地训练 JSONL
python -m trendmodel.train prepare \
  --input data/examples.jsonl \
  --local-raw-dir data/raw/huggingface \
  --local-sources wildchat,arena \
  --wildchat-limit 5000 \
  --arena-limit 2000 \
  --languages en,zh \
  --output data/processed/training.jsonl \
  --report data/processed/training_report.json

# M4 Metal 或 NVIDIA CUDA 训练；没有 GPU 时直接失败
.venv-tf/bin/python -m trendmodel.train train \
  --data data/processed/training.jsonl \
  --artifact artifacts/trendmodel-text-v1 \
  --epochs 10 \
  --batch-size 16 \
  --require-gpu

# 可选：训练 Transformer challenger，使用独立 artifact 进行对比
.venv-tf/bin/python -m trendmodel.train train \
  --data data/processed/training.jsonl \
  --artifact artifacts/trendmodel-text-v1-transformer \
  --architecture transformer-challenger \
  --epochs 10 \
  --batch-size 16 \
  --require-gpu

# 保存后的本地评估
.venv-tf/bin/python -m trendmodel.train evaluate \
  --data data/processed/training.jsonl \
  --artifact artifacts/trendmodel-text-v1 \
  --split validation \
  --mode both \
  --ablation all
```

评估报告会把两种口径分开：

- `raw`：TensorFlow heads 的输出；TaskAST 已作为模型输入，不做 post-hoc fusion；
- `inference`：生产 JSON contract，增加同一份 TaskAST evidence，但概率保持不变；
- `fused`：旧命令兼容别名，等同于 `inference`，不再表示第二次概率融合。

如需查看输入消融，增加 `--ablation all`。报告会分别给出：

- `text_only`：把已训练 artifact 的结构化特征向量置零，只保留文本分支；
- `text_ast`：正常使用文本分支和 TaskAST 56 维特征，是当前主路径；
- `evidence_only`：只用确定性 TaskAST 规则，不经过 TensorFlow，是诊断 baseline。

`text_only` 是输入消融，不是重新训练的纯文本模型；如果要证明删除结构分支后
架构仍然有效，还需要额外训练一个没有该输入的 challenger artifact。

默认 `--split validation` 使用与训练相同的确定性留出策略。`--split all` 会包含
训练样本，只适合做数据分布诊断，不能作为泛化指标。独立人工标注集准备好后，
建议这样评估：

```bash
.venv-tf/bin/python -m trendmodel.train evaluate \
  --data data/gold_validation.jsonl \
  --artifact artifacts/trendmodel-text-v1 \
  --split all \
  --mode both
```

公共数据的 intent 等弱标签可能来自同一个 `TaskTagger` teacher，因此这些标签本身
不能替代独立 gold 评估。训练时还会把 validation raw 指标与保存后
重新加载 artifact 的 raw 指标做 round-trip 对照；如果差异明显，metadata 的
`warnings` 会提示不要直接发布该 artifact。

### artifact 交付

下游只需要复制整个 artifact 目录：

```text
artifacts/trendmodel-text-v1/
├── model.keras       # TensorFlow/Keras 权重和网络结构
└── metadata.json     # label map、runtime、监督计数、验证指标和 warning
```

建议下游启动时检查：

1. `metadata.model_id` 是否为预期模型；
2. `metadata.model_version` 是否在兼容范围内；
3. `metadata.labels` 是否包含下游支持的 label；
4. `metadata.effort_bucket_supervision_samples`、`effort_minutes_supervision_samples`、`effort_supervision_samples` 和 `observed_effort_samples` 是否满足要求；
5. `metadata.architecture_family` 是否是经过评估和批准的 `bigru` 或 challenger；
6. 模型文件 checksum 是否与发布清单一致。

原始数据、清洗数据、模型 artifact 和 Python 虚拟环境都被 `.gitignore` 忽略，不应通过 Git 传递。建议使用内部 artifact storage、受控文件复制或带 checksum 的压缩包。

## 7. 常见故障

### artifact 加载失败

确认目录同时包含 `model.keras` 和 `metadata.json`，并使用与训练兼容的 TensorFlow/Keras runtime。不要只复制 `model.keras`。

### TensorFlow 没有看到 Metal 或 CUDA

训练时使用：

```bash
.venv-tf/bin/python -m trendmodel.train train ... --require-gpu
```

如果失败，先检查：

```bash
.venv-tf/bin/python - <<'PY'
import tensorflow as tf
print(tf.__version__)
print(tf.config.list_physical_devices())
print(tf.config.list_physical_devices("GPU"))
PY
```

当前 Apple Silicon 组合是 `tensorflow-macos==2.16.2` 和 `tensorflow-metal==1.2.0`。受限执行环境可能没有 Metal 设备访问权限；这种情况下 `--require-gpu` 应失败，而不是把 CPU 训练误认为 Metal 训练。

### TensorFlow 没有看到 CUDA

在 Linux 或 WSL2 中依次检查：

```bash
nvidia-smi
.venv-tf-cuda/bin/python - <<'PY'
import tensorflow as tf

print(tf.__version__)
print("built_with_cuda:", tf.test.is_built_with_cuda())
print(tf.config.list_physical_devices())
print(tf.config.list_physical_devices("GPU"))
PY
```

确认使用了 `requirements-tensorflow-cuda.txt`，而不是 Mac 的
`requirements-tensorflow-macos.txt`。如果 `nvidia-smi` 正常但 TensorFlow 没有
GPU，优先重建 CUDA 专用虚拟环境并确认 WSL2 已安装 NVIDIA GPU 驱动支持。

### HTTP 返回 400

重点检查：

- `messages` 是否为非空数组；
- 是否至少有一个非空 `role=user` 消息；
- `content` 是否是字符串或 text-only content parts；
- 是否错误地设置了 `stream=true`。

### 结果不稳定或概率过高

当前公共语料的分类标签主要是 TaskTagger 弱标签，probability 还没有在独立 gold 集上校准。不要只看单次输出；应保存完整候选、evidence 和下游反馈，建立回放评估集。
