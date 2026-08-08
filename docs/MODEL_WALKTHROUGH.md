# TrendModel 模型 Walkthrough

> 本文面向第一次阅读 TrendModel 代码的人。它解释当前模型究竟是什么、每一层为什么存在、输入如何变成输出，以及哪些组件属于生产主链路、哪些只是实验或兼容代码。
>
> 当前实现基线：model_version = 0.5.0；默认生产架构：BiGRU。

## 1. 先给结论

TrendModel 的核心任务很简单：

~~~text
输入一段用户 Prompt
输出：
  - 用户想做什么（intent）
  - 属于哪种 workload
  - 复杂度如何
  - 大致需要多少 effort
  - 当前判断有多确定
  - Prompt 中观察到了哪些结构化证据
~~~

当前真正的 TensorFlow 生产模型是一个轻量级、多任务、监督式文本分类器：

~~~text
字符级 TextVectorization
  → Embedding
  → BiGRU
  → 字符 n-gram TF-IDF 分支
  → TaskAST 56 维结构化特征分支
  → 共享表示
  → 多个分类和回归 heads
  → JSON / Task IR
~~~

它不是：

- 一个生成式大语言模型；
- 一个预训练中文句向量模型；
- 一个 CNN 模型；
- 一个用 Bayesian 方法完成中文语义理解的模型；
- 一个直接替下游 Router 选择具体供应商模型的系统。

当前仓库里有三条容易混淆的路径：

| 路径 | 作用 | 是否是当前 TensorFlow 生产模型 |
| --- | --- | --- |
| TensorFlowTaskModel | 字符级 BiGRU + n-gram + TaskAST + 多任务 heads | 是 |
| TaskTagger | 句向量/原型相似度 + 结构化证据 + 可选概率融合 | 否，主要是 prototype 和数据弱标注工具 |
| Transformer challenger | 用轻量 Transformer 替换 BiGRU，保持其他接口不变 | 否，只是实验对照组 |

因此，最准确的系统描述是：

> TrendModel 是一个带 TaskAST 结构化输入的轻量级 Prompt 多任务分类器。BiGRU 负责学习文本顺序和上下文，n-gram 分支负责捕捉技术短语，TaskAST 负责暴露代码、文件、命令、语言和风险等确定性信号。

## 2. 从输入到输出的完整链路

~~~mermaid
flowchart TD
    A["用户 Prompt"] --> B["TaskAST 规则提取"]
    A --> C["字符级 TextVectorization"]
    C --> D["Embedding + BiGRU"]
    A --> E["字符 2-5 gram TF-IDF"]
    B --> F["56 维结构化特征"]
    D --> G["文本表示"]
    E --> G
    G --> H["文本共享表示"]
    F --> I["结构化表示"]
    H --> J["拼接 + Dense"]
    I --> J
    J --> K["多任务 heads"]
    K --> L["workload / intent / complexity / effort"]
    K --> M["probability / entropy / margin"]
    B --> N["可审计 evidence"]
    L --> O["Task IR / JSON"]
    M --> O
    N --> O
    O --> P["下游 Auto Router"]
~~~

注意最后三类信息的来源不同：

~~~text
predictions：
  主要来自 TensorFlow 的神经网络 heads

evidence：
  来自 TaskAST 的确定性结构化提取器

task_ir：
  把神经预测和 TaskAST 字段组装成下游统一接口
~~~

TaskAST 在 TensorFlow 推理时已经作为输入参与模型计算。推理结束后返回 evidence，是为了审计和路由决策可解释；它不会再被拿出来对 TensorFlow probability 做第二次规则修正。

## 3. 第一步：读取和规范化 Prompt

输入可以是中文、英文、混合语言、代码、路径或命令，例如：

~~~text
帮我实现一个 C++ 多线程程序，计算两个集合的笛卡尔积
~~~

模型会对换行等文本形式做有限规范化，但会保留原始文本用于输出：

~~~json
{
  "original_prompt": "帮我实现一个 C++ 多线程程序，计算两个集合的笛卡尔积"
}
~~~

这里的规范化不是中文分词，也不会把 Prompt 改写成摘要。它只是让训练和推理使用相同的输入形式。

## 4. 第二步：TaskAST 提取结构化事实

### 4.1 TaskAST 是什么

普通 AST 是 Abstract Syntax Tree，即抽象语法树。编译器通常把源代码解析成 AST，节点表示变量、表达式、函数、循环等语法结构。

TrendModel 的 TaskAST 借用了“把一串字符转换成结构化中间表示”这个思想，但它不是编程语言 AST。它不理解 C++ 的完整语法，也不负责编译代码。

TrendModel 的 TaskAST 是：

~~~text
一个面向任务路由的、结构化的证据对象
~~~

实现位于：

- trendmodel/task_ast.py：TaskAST dataclass；
- trendmodel/evidence.py：确定性 evidence extractor。

### 4.2 TaskAST 保存什么

一个 TaskAST 大致包含：

~~~text
original_prompt
constraints
language
operations
intent_scores
domain_scores
artifact_scores
high_risk
risk_reasons
evidence / signals
~~~

例如对下面的 Prompt：

~~~text
请修改 ./deploy/dns.yaml，并运行 kubectl rollout status 检查生产发布结果。
~~~

TaskAST 可能提取出：

~~~json
{
  "language": "shell",
  "operations": ["dns", "kubernetes", "deployment"],
  "constraints": {
    "requires_files": true,
    "requires_tools": true,
    "has_file_paths": true,
    "has_shell_commands": true
  },
  "risk": {
    "high": true,
    "reasons": ["production"]
  },
  "evidence": [
    {
      "kind": "structure",
      "label": "file_path",
      "strength": 0.93,
      "rule_id": "structure:file_path"
    },
    {
      "kind": "structure",
      "label": "shell_command",
      "strength": 0.94,
      "rule_id": "structure:shell_command"
    }
  ]
}
~~~

其中 strength 表示规则对“这个事实存在”的判断强度，不等于最终 workload probability。

TaskAST 还会保留：

- source_span：证据在原始 Prompt 中的位置；
- rule_id：哪条规则命中了；
- negated：该信号是否被否定；
- quoted：该信号是否只是被引用或讨论。

例如：

~~~text
不要执行 rm -rf
~~~

里面虽然出现了 Shell 命令，但命令是被否定的。TaskAST 可以记录这个事实，同时避免把它直接解释为“用户要求执行删除操作”。

### 4.3 TaskAST 不是最终分类结果

TaskAST 中的 intent_scores、domain_scores 和 artifact_scores 都是候选证据或 hint，不是 TensorFlow 最终输出的 label。

例如，出现 C++ 可以提高 programming 和 source_code 的结构化分数，但不能仅凭 C++ 就断定用户是在创建代码。用户也可能是在解释、审查、调试、翻译或比较 C++ 方案。

这就是为什么 TaskAST 进入模型作为输入，而不是直接充当分类器。

## 5. 第三步：TaskAST 转换为 56 维特征

TaskAST 是可读的结构化对象，但神经网络通常接收固定形状的数值张量。因此代码会把它转换成固定宽度的 56 维向量。

当前特征分组如下：

| 特征组 | 数量 | 示例 |
| --- | ---: | --- |
| 文本形状 | 5 | 长度、行数、中文比例、英文比例、数字比例 |
| 结构信号 | 8 | 代码块、路径、Shell、数学符号、URL、文件需求、工具需求、风险 |
| 语言 | 10 | C++、Python、Java、Go、Rust、SQL、Shell 等 |
| intent hint | 13 | create、modify、debug、explain、execute 等 |
| domain hint | 11 | programming、networking、document、operations 等 |
| operations | 9 | DNS、Kubernetes、数据库、部署、测试、多线程等 |
| 合计 | 56 | 供结构化 Dense 分支使用 |

这个过程可以抽象为：

~~~text
TaskAST
  → [0.0, 1.0, 0.42, ..., 0.0]
  → Dense(32)
  → 结构化表示
~~~

数值特征的好处是，模型可以直接看到“是否有文件路径”“是否需要工具”“是否有生产风险”，不必完全依靠文本分支自己从字符序列中重新发现这些事实。

## 6. 第四步：文本分支

当前生产文本分支由两个互补部分组成：

~~~text
字符级 BiGRU
字符 n-gram TF-IDF
~~~

### 6.1 Token 是什么

Token 是模型处理文本时的基本单位。不同模型的 token 可以是一个词、一个子词、一个字符、一个标点或特殊符号。

TrendModel 当前使用字符级切分。比如：

~~~text
C++ 多线程
~~~

可以近似看成：

~~~text
C、+、+、 、多、线、程
~~~

实际 TensorFlow TextVectorization 会维护一个字符 vocabulary，并把每个字符映射成整数 id。

字符级切分的工程取舍是：

优点：

- 不需要额外的中文分词器；
- 可以保留 C++、路径、文件扩展名和 Shell flag；
- 对中英文混合输入比较自然；
- 模型和依赖较小。

缺点：

- 一个 Prompt 可能产生很多 token；
- 字符本身缺少词级语义；
- 模型需要更多训练数据才能学到稳定的短语含义；
- 它不是预训练中文语义模型。

### 6.2 Embedding 是什么

Embedding 是把离散 id 映射成连续向量的表。

例如：

~~~text
字符 id 42
  → [0.12, -0.41, 0.08, ...]
~~~

当前字符 Embedding 的维度是 96。一个 Prompt 经过向量化后，会变成最多 192 个字符、每个字符 96 维的向量序列。较短的 Prompt 会 padding，过长的 Prompt 会截断。

Embedding 本身不理解语言。它只是可训练的参数表。只有在训练过程中，模型才会逐渐把相似任务中的字符和片段映射到更有用的向量位置。

### 6.3 RNN 是什么

RNN 是 Recurrent Neural Network，即循环神经网络。

普通前馈网络通常一次接收一个固定向量，而 RNN 会按顺序读取序列，并维护一个 hidden state：

~~~text
x1 → state1
x2 + state1 → state2
x3 + state2 → state3
...
~~~

因此 RNN 适合处理文本、时间序列、语音和其他有顺序的数据。

它的缺点是序列较长时容易遇到梯度消失或梯度爆炸，而且必须按顺序处理 token，训练并行度不如 Transformer。

### 6.4 GRU 是什么

GRU 是 Gated Recurrent Unit，即门控循环单元。它是 RNN 的一种改进结构。

普通 RNN 每一步都直接更新 hidden state。GRU 使用门来决定：

~~~text
哪些旧信息应该保留
哪些旧信息应该遗忘
哪些新信息应该写入
~~~

可以把它理解成一个带有“记忆管理机制”的 RNN。GRU 通常比 LSTM 结构更简单、参数更少，但仍能处理比普通 RNN 更长的依赖关系。

GRU 的核心思想是 update gate 和 reset gate：

~~~text
update gate：
  这一步要保留多少旧记忆？

reset gate：
  计算新内容时要忘掉多少旧记忆？
~~~

TensorFlow 的 GRU layer 会完成这些计算，项目代码不需要手写门控公式。

### 6.5 BiGRU 是什么

BiGRU 是 Bidirectional GRU，即双向 GRU。

普通单向 GRU 只从左到右读取：

~~~text
第一个字符 → 最后一个字符
~~~

BiGRU 同时使用两个方向：

~~~text
正向：  第一个字符 → 最后一个字符
反向：  最后一个字符 → 第一个字符
~~~

最后把两个方向的结果拼接起来。

例如下面两句话共享很多词：

~~~text
请解释这段 C++ 多线程代码
帮我实现一个 C++ 多线程程序
~~~

但它们的动作和交付物不同。双向序列编码可以同时利用动作前后的上下文。

当前主模型中的文本分支是：

~~~text
Character TextVectorization
  → Embedding(96)
  → Bidirectional(GRU(64))
  → Dense(128)
~~~

BiGRU 的限制也很明确：

- 它不是 ChatGPT 一类的生成模型；
- 它不会生成答案；
- 它不是预训练模型；
- 它不会自动获得完整中文语义；
- 它需要从我们的标注数据中学习任务边界。

BiGRU 适合当前场景的原因是参数量和推理开销相对可控，且对离线 Prompt 分类足够自然。相关的 GRU 和双向循环网络思想可参考 Cho 等人（2014）和 Schuster 与 Paliwal（1997）。

### 6.6 字符 n-gram 是什么

n-gram 表示连续的 n 个单位。在当前模型中单位是字符。

例如 kubectl rollout 可以产生很多连续的字符片段。代码使用 2 到 5 字符的 n-gram，因此能够捕捉 C++、api.py、kubectl、DNS、SELECT、docker compose、文件扩展名和命令片段。

n-gram 不理解句子深层含义，但对于短 Prompt、代码符号和稳定技术表达非常有效。

### 6.7 TF-IDF 是什么

TF-IDF 是 Term Frequency–Inverse Document Frequency，一种经典的词法加权方法。

它的基本直觉是：

~~~text
一个片段在当前文本中出现得越重要，TF 越高；
一个片段在所有文本中越少见，IDF 越高；
两者相乘后，该片段的区分度越高。
~~~

简化表示为：

~~~text
TF-IDF(g, d) = TF(g, d) × log(N / (DF(g) + 1))
~~~

其中 g 是一个 n-gram，d 是当前 Prompt，N 是训练文档数量，DF 是包含该 n-gram 的文档数量。

在 TrendModel 中，TF-IDF 只负责提供词法 shortcut。它不是语义理解器，也不会取代 BiGRU。

## 7. 第五步：共享表示和多任务 heads

### 7.1 Shared representation 是什么

两个文本分支和 TaskAST 分支分别得到向量后，会被拼接起来：

~~~text
BiGRU representation
  + n-gram representation
  + TaskAST representation
  → shared representation
~~~

Shared representation 可以理解为模型对当前 Prompt 的一份综合内部表示。

### 7.2 Head 是什么

Head 是接在共享表示后面的一个具体预测输出层。

本模型不是只输出一个 label，而是有多个 head：

~~~text
shared representation
  ├── workload_type head
  ├── intent head
  ├── complexity head
  ├── effort_bucket head
  ├── effort_ordinal head
  ├── effort_log_minutes head
  └── needs_clarification head
~~~

每个 head 负责一个相关但不同的问题。

### 7.3 多任务学习是什么

多任务学习（Multi-Task Learning）是让一个模型同时学习多个相关任务，而不是为每个任务训练一个完全独立的模型。

在 TrendModel 中：

~~~text
同一个 Prompt
  → 同一个共享表示
  → 同时预测 workload、intent、complexity、effort
~~~

这样做的原因是这些任务有关联：

- create 往往比 explain 需要更多 effort；
- 生产环境和文件操作往往增加 complexity；
- code workload 往往伴随语言、路径或代码结构；
- modify 和 debug 可能共享很多技术词汇，但动作不同。

共享表示可以让不同任务互相提供训练信号，同时避免为每一个 label 单独维护一套文本编码器。多任务学习的经典讨论见 Caruana（1997）。

## 8. 输出层里的概率和数学对象

### 8.1 Logit 是什么

神经网络最后通常先输出一组未归一化的数，这些数称为 logits。

~~~text
workload logits = [1.8, 0.4, -0.7, ...]
~~~

Logit 本身不是概率，也不要求总和为 1。

### 8.2 Softmax 是什么

对于 workload、intent、complexity 和 effort bucket 这种互斥分类，模型使用 softmax：

~~~text
logits
  → softmax
  → [0.72, 0.18, 0.10]
~~~

softmax 会把一组数转换成非负分布，所有类别概率加起来为 1。

probability 是模型分布中的概率分量，不自动等于真实世界中 72% 的正确率。只有在独立数据上完成 calibration 后，才可以更谨慎地解释为置信度。

### 8.3 Sigmoid 是什么

对于二分类或独立的顺序阈值，模型使用 sigmoid：

~~~text
任意实数
  → sigmoid
  → 0 到 1 之间的数
~~~

effort ordinal head 的每一个输出都是一个 sigmoid 概率，例如 P(Y > XS)、P(Y > S)、P(Y > M) 和 P(Y > L)。

### 8.4 Ordinal effort 是什么

XS、S、M、L、XL 是有顺序的类别。普通 softmax 只知道五个类别，不知道它们之间有距离关系。

Ordinal head 把问题转换为四个顺序问题：

~~~text
任务是否超过 XS？
任务是否超过 S？
任务是否超过 M？
任务是否超过 L？
~~~

例如 M 的理想目标是：

~~~text
[P(Y > XS), P(Y > S), P(Y > M), P(Y > L)]
= [1, 1, 0, 0]
~~~

推理时，代码会把四个阈值转换为 XS 到 XL 的 bucket distribution，并强制分布满足单调顺序。

这比单独使用普通 bucket softmax 更符合 effort 的有序性质，但它仍然不能替代真实 active minutes。

## 9. 第六步：推理后处理

模型产生原始 head 输出后，推理包装器会做一些不改变核心分类器的后处理。

### 9.1 Temperature scaling

Temperature scaling 是一种概率校准方法。

模型可能经常输出 0.98，但实际上并没有那么确定。Temperature scaling 会使用 validation 数据学习一个温度参数，调整 logits 的尖锐程度：

~~~text
原始 logits
  → 除以 temperature
  → softmax
  → 更合理的 distribution
~~~

它通常会改变 probability、entropy 和 margin，但不会改变 argmax label。

它不是 Bayesian inference，也不是第二个分类器。

### 9.2 Entropy 是什么

Entropy 用来表示一个概率分布有多分散。

~~~text
[0.99, 0.005, 0.005]
~~~

这个分布的 entropy 很低，模型比较集中。

~~~text
[0.34, 0.33, 0.33]
~~~

这个分布的 entropy 很高，模型几乎无法区分候选。

Router 可以利用 entropy 判断是否应该选择更强的模型或走 fallback。

### 9.3 Margin 是什么

Margin 是 top-1 probability 和 top-2 probability 的差：

~~~text
margin = top1 - top2
~~~

例如：

~~~text
[0.80, 0.12, 0.08] → margin = 0.68
[0.42, 0.39, 0.19] → margin = 0.03
~~~

第二种情况说明两个候选很接近，即使 top-1 仍然是第一名。

### 9.4 Abstention 是什么

Abstention 可以翻译为拒识策略。

它不是第五个分类标签，而是模型在不确定时承认：

~~~text
当前判断不够可靠，需要澄清、fallback 或更强模型
~~~

当前 policy 会观察 top probability、top-1 和 top-2 margin、以及 entropy。

触发 abstention 时：

- 不会把原来的 label 强行改成 unknown；
- 不会修改 TensorFlow head 的 probability；
- 会设置 needs_clarification；
- 会记录 abstention reasons。

## 10. Task IR 是什么

Task IR 是 Task Intermediate Representation，即任务中间表示。

它不是神经网络层，也不是额外的分类器。它是一个稳定的、面向 Router 的数据结构。

可以把它理解为：

~~~text
TaskAST = 模型内部和证据层使用的表示
Task IR = 给下游 Router 使用的接口表示
~~~

TensorFlow 当前输出的主要神经预测是 workload_type、intent、complexity 和 effort。

TaskAST 提供 domain、artifact、operations、language、constraints、risk 和 evidence。

Task IR 将两者组装起来。例如：

~~~json
{
  "workload_type": {
    "label": "code",
    "probability": 0.81
  },
  "intent": {
    "label": "create",
    "probability": 0.76
  },
  "complexity": {
    "label": "medium",
    "probability": 0.68
  },
  "artifact": "source_code",
  "language": "cpp",
  "operations": ["multithreading"],
  "risk_level": "low",
  "evidence": []
}
~~~

这里 artifact 和 language 不应该被误读成当前 TensorFlow 独立训练出来的 head；它们主要是 TaskAST-derived fields。

## 11. Bayesian fusion 在当前系统中到底在哪里

这是整个项目里最容易产生误解的地方。

### 当前 TensorFlow 主链路

当前 TensorFlow 主链路不做：

~~~text
TensorFlow probability
  + TaskAST rule probability
  + context prior
  → second posterior
~~~

TaskAST 只在模型内部作为输入特征，并在输出中作为 evidence 返回。

### TaskTagger prototype 路径

TaskTagger 仍然保留了：

~~~text
句向量相似度
  + TaskAST structural scores
  + optional context prior
  → weighted / Bayesian-style fusion
~~~

这条路径适合作为早期 prototype、可解释 baseline、公共数据的弱标签 teacher，或没有 TensorFlow artifact 时的 fallback。

### Bayesian 方法的正确定位

Bayesian 方法更适合表达：

~~~text
已有先验
  + 当前 Prompt 的多种证据
  → 更新后的不确定性
~~~

它不应该被当成中文语义理解器。当前主模型已经按照这个边界，把确定性证据作为输入，把概率校准和拒识作为输出管理层。

## 12. 训练阶段发生什么

训练入口是：

~~~text
python -m trendmodel.train train
~~~

完整训练链路如下：

~~~mermaid
flowchart LR
    A["本地 JSONL / Parquet adapter"] --> B["清洗、脱敏、去重"]
    B --> C["TaskRecord"]
    C --> D["group-aware train/validation split"]
    D --> E["TaskAST + 56 维特征"]
    D --> F["label maps + sample weights"]
    E --> G["TensorFlow dataset"]
    F --> G
    G --> H["BiGRU multi-task model"]
    H --> I["validation evaluation"]
    I --> J["temperature scaling"]
    J --> K["model.keras + metadata.json"]
~~~

### 12.1 数据标签

公共数据通常只有 Prompt 和对话元数据，没有我们的 workload、intent、complexity 和真实 effort 标签。

因此准备阶段可以使用 TaskTagger 产生弱标签，但这些标签不能被当成最终 gold truth。

训练记录会区分：

~~~text
manual_gold：
  人工确认的标签

seed：
  仓库内的受控种子样本

pseudo / weak：
  规则或 teacher 产生的弱标签

observed：
  来自真实任务记录的 active minutes
~~~

### 12.2 缺失标签和 sample-weight mask

不是每条记录都拥有全部标签。例如公共 Prompt 可能有 intent，却没有 effort_minutes。

这时数据管线不会伪造缺失值，而是：

~~~text
目标值可以填充占位值
对应 sample weight = 0
该记录不参与这个 head 的 loss
~~~

因此同一个数据集可以同时训练多个 head，而不会因为缺少 effort 真值而制造错误监督。

### 12.3 Group-aware split

同一个会话、同一个任务的多个变体不能随机分散到训练集和验证集，否则会产生 data leakage。

当前切分会尽量保证：

~~~text
同一个 split_group
  → 只出现在 train 或 validation 的一侧
~~~

同时会按 workload、intent、complexity 和 effort 的 dominant label 做近似分层。

### 12.4 当前训练目标

当前总 loss 可以概括为：

~~~text
L_total =
  1.0 × workload loss
  + 1.0 × intent loss
  + 0.8 × complexity loss
  + 0.8 × effort bucket loss
  + 0.8 × effort ordinal loss
  + 1.0 × effort minutes loss
  + 0.7 × clarification loss
~~~

其中缺失标签对应的 sample weight 为 0。

## 13. 输出 JSON 如何被下游使用

一个简化的输出结构如下：

~~~json
{
  "original_prompt": "帮我实现一个 C++ 多线程程序",
  "predictions": {
    "workload_type": {
      "label": "code",
      "probability": 0.81,
      "distribution": {}
    },
    "intent": {
      "label": "create",
      "probability": 0.76,
      "distribution": {}
    },
    "complexity": {
      "label": "medium",
      "probability": 0.68,
      "distribution": {}
    },
    "effort": {}
  },
  "effort": {
    "bucket": "M",
    "estimated_minutes": 75,
    "basis": "bucket_only"
  },
  "evidence": [
    {
      "kind": "language",
      "label": "cpp",
      "strength": 0.98
    }
  ],
  "needs_clarification": false,
  "confidence": {
    "overall": 0.72
  }
}
~~~

这个示例中的数值只是说明结构，不代表当前模型的实测结果。

下游 Router 可以据此决定：

~~~text
code + create + high complexity
  → 倾向选择代码能力更强的模型

text + translate + low complexity
  → 可以选择便宜模型

低 probability / 高 entropy / abstention
  → 选择 fallback 或请求澄清
~~~

TrendModel 不负责直接选择 Model-A、Model-B，也不负责执行命令或修改文件。

## 14. 当前真正的限制

### 14.1 生产 TensorFlow 模型没有预训练中文语义

TaskTagger prototype 可以接入 Sentence Transformer，但当前 TensorFlow 主模型使用的是字符级 BiGRU。

这意味着模型的中文泛化能力主要取决于训练数据量、标签质量、中文和中英混合样本覆盖、hard-negative 样本以及真实 workload 分布。

如果只有少量 seed 或大量 teacher pseudo-label，模型很可能只是学习规则和数据分布，而不是真正获得通用语义能力。

### 14.2 Effort 目前只能算粗粒度

没有真实 active minutes 时，effort 主要是 bucket prior。

~~~text
effort.bucket：
  可以用于任务排序和模型能力分层

effort.estimated_minutes：
  不应该被当作承诺工时或 SLA
~~~

### 14.3 标签集合是固定的

当前模型只能在训练时定义的 workload、intent、complexity 和 effort label 集合中选择。

它所谓的“通用”是可以接受各种自然语言 Prompt，并映射到固定的任务标签体系；它不是可以无限发现新任务类型的开放世界模型。

### 14.4 TaskAST 依赖规则覆盖

TaskAST 对代码块、路径、命令、语言和风险等结构事实很有用，但规则本身可能漏报或误报。

所以 evidence 的 strength 不能直接当成最终分类概率。

### 14.5 Transformer 不是当前生产模型

Transformer challenger 只替换 BiGRU 编码器，其他输入和 heads 不变。

它是否更好必须在同一份 gold/hard-negative 数据上比较：

- workload、intent、complexity 的 macro-F1；
- effort bucket accuracy 和 observed minutes MAE；
- NLL、ECE 和 abstention coverage；
- 延迟、内存和 artifact 大小。

在这些结果出来之前，BiGRU 是默认生产模型。

## 15. 当前系统应该如何理解

可以用下面这张表记住每个组件的责任：

| 组件 | 负责什么 | 不负责什么 |
| --- | --- | --- |
| 字符 TextVectorization | 把 Prompt 转成字符 id | 不理解语义 |
| Embedding | 把离散 id 转成向量 | 不单独完成分类 |
| BiGRU | 建模字符顺序和上下文 | 不生成最终答案 |
| n-gram TF-IDF | 捕捉技术短语和词法信号 | 不理解深层语义 |
| TaskAST | 提取可观察结构事实 | 不直接决定最终 label |
| 56 维特征分支 | 把 AST 事实交给神经网络 | 不等于 Bayesian posterior |
| Multi-task heads | 预测多个任务字段 | 不负责模型供应商选择 |
| Calibration | 调整概率可信度 | 不提高模型本身的语义能力 |
| Abstention | 标记不确定请求 | 不自动修正预测 label |
| Task IR | 输出稳定 Router contract | 不是额外模型 |
| Auto Router | 结合模型能力、成本和权限做选择 | 不属于 TrendModel |

## 16. 推荐的下一步 Task

当前最重要的下一步不是继续增加模型结构，而是建立独立的 gold evaluation loop。

建议的第一个闭环是：

~~~text
1. 冻结当前 BiGRU 架构和标签定义
2. 从真实 Router prompt 中抽取并脱敏 300 条样本
3. 人工标注 workload、intent、complexity、needs_clarification
4. 为至少 100 条有可靠记录的任务补 active_minutes
5. 专门加入 explain/create、create/modify、modify/debug、explain/execute hard negatives
6. 保留一份完全不参与训练和调参的 gold_validation 集
7. 在 M4 或 RTX 4070 上训练当前 BiGRU
8. 记录 macro-F1、混淆矩阵、ECE、abstention coverage、effort MAE 和延迟
9. 只有 baseline 有稳定结果后，再决定是否运行 Transformer challenger
~~~

仓库已有标注模板：

~~~text
data/gold_validation.template.jsonl
~~~

这个 Task 的产出应该是一个可重复的 baseline report，而不是又一个模型架构。

如果 baseline 显示主要错误集中在 explain vs create、create vs modify、code vs operations 或 effort bucket，就针对错误类型补数据和 hard negatives。只有当错误明显来自长距离文本依赖，而不是标签不足或规则误报时，Transformer challenger 才值得投入。

## 17. 代码入口

| 代码 | 作用 |
| --- | --- |
| trendmodel/tf_model.py | 构建 BiGRU 主模型和 Transformer challenger |
| trendmodel/tf_features.py | TaskAST 到 56 维特征 |
| trendmodel/task_ast.py | TaskAST dataclass |
| trendmodel/evidence.py | 结构化证据提取 |
| trendmodel/tf_data.py | TaskRecord、切分、targets 和 sample weights |
| trendmodel/tf_train.py | 训练、校准、保存和评估 |
| trendmodel/tf_inference.py | 加载 artifact、预测和 JSON contract |
| trendmodel/task_ir.py | Task IR 适配和序列化 |
| trendmodel/tagger.py | prototype/teacher 路径，不是 TensorFlow 主模型 |

## 18. 参考文献

Caruana, R. (1997). Multitask learning. *Machine Learning, 28*, 41–75. https://doi.org/10.1023/A:1007379606734

Cho, K., van Merriënboer, B., Gulcehre, C., Bahdanau, D., Bougares, F., Schwenk, H., & Bengio, Y. (2014). Learning phrase representations using RNN encoder–decoder for statistical machine translation. *arXiv preprint arXiv:1406.1078*. https://arxiv.org/abs/1406.1078

Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). On calibration of modern neural networks. *Proceedings of the 34th International Conference on Machine Learning*. https://proceedings.mlr.press/v70/guo17a.html

Reimers, N., & Gurevych, I. (2019). Sentence-BERT: Sentence embeddings using Siamese BERT-networks. *Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing*. https://aclanthology.org/D19-1410/

Salton, G., & Buckley, C. (1988). Term-weighting approaches in automatic text retrieval. *Information Processing & Management, 24*(5), 513–523. https://doi.org/10.1016/0306-4573(88)90021-0

Schuster, M., & Paliwal, K. K. (1997). Bidirectional recurrent neural networks. *IEEE Transactions on Signal Processing, 45*(11), 2673–2681. https://doi.org/10.1109/78.650093

Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., & Polosukhin, I. (2017). Attention is all you need. *Advances in Neural Information Processing Systems, 30*. https://papers.nips.cc/paper/7181-attention-is-all-you-need
