# TrendModel：面向 Auto Router 的轻量级任务分析模型

> 论文式系统设计文档；实现基线：`model_version = 0.5.0`

## 摘要

本文定义 TrendModel 的问题边界、数据流、模型结构和推理算法。TrendModel 的输入是一段用户 prompt，输出是面向 Auto Router 的结构化任务分析，而不是最终答案。系统需要同时判断用户意图、workload 类型、复杂度、粗粒度 effort、是否需要澄清，以及代码、文件、命令、工具和风险等可观测约束。

本文采用“语义表示 + TaskAST 结构化证据 + Bayesian-style 证据管理 + Task IR”的分层设计。这里的 `TaskAST` 是正式的 Python dataclass，但它不是编程语言语法树，而是面向任务路由的结构化中间层。文本编码器负责理解自然语言整体语义；结构化解析器负责提取代码块、语言、路径、命令、操作和风险等确定性信号；TaskAST 在当前 TensorFlow 主模型中直接编码成输入特征，同时作为 evidence 原样返回。贝叶斯方法在本系统中不是中文语义理解器，也不是当前 TensorFlow 推理阶段的第二次概率修正；它保留在 prototype `TaskTagger` 的可选证据融合、以及未来的校准/上下文更新层。

当前 TensorFlow 主模型使用字符级 `BiGRU`、字符 n-gram `TF-IDF`、56 维 TaskAST 结构化特征分支和多任务 heads。系统还保留了一个可插拔的句向量 + 原型分类路径，作为解释性 TaskTagger 和独立 prototype。文档同时区分当前已实现能力、概念架构和待补齐的数据/标注能力，避免把弱标签、规则分数或未经校准的 probability 误写成真实准确率。

## 关键词

任务路由；文本分类；Task AST；Task IR；结构化证据；Bayesian-style evidence fusion；多任务学习；effort estimation；TensorFlow；不确定性估计

## 相关工作与本文定位

本系统吸收了几个成熟方向的思想：TF-IDF 用于提供高效的词法/字符片段信号（Salton & Buckley, 1988）；GRU 和双向循环网络用于建模序列上下文（Cho et al., 2014; Schuster & Paliwal, 1997）；多任务学习通过共享表示同时优化相关任务（Caruana, 1997）；句向量模型可以把整句映射到适合相似度比较的语义空间（Reimers & Gurevych, 2019）；概率校准则用于把模型分数转化为更可信的置信度（Guo et al., 2017）。TrendModel 的区别在于目标不是通用问答或单一 intent 分类，而是把用户请求转换为 Auto Router 可消费的任务能力描述，并且显式保留结构化证据、标签来源和不确定性。

本文不是一份声称已经达到某个 benchmark 的实验论文，而是一份和当前代码同步的系统设计论文。所有“当前实现”描述均以仓库中的 TensorFlow、数据管线和推理代码为准；所有“未来”描述都单独标注。

## 1. 文档目的

TrendModel 是一个部署在 Auto Router 前面的轻量级任务分析模型。它接收一段用户 prompt，输出结构化的任务判断，帮助下游 Router 在质量、成本、延迟和上下文能力之间做模型选择。

它的职责是：

- 判断用户想做什么；
- 判断请求属于哪一种 workload；
- 判断任务复杂度和粗粒度 effort；
- 识别代码、文件、命令、工具、语言和风险等确定性信号；
- 输出候选、概率和不确定性，供 Router 决定是否需要 fallback 或澄清。

它不负责：

- 生成最终答案；
- 选择具体的外部大模型；
- 执行 Shell、文件修改或其他工具；
- 从公共聊天数据中伪造真实工时；
- 代替下游 Router 的权限和安全判断。

最终边界可以概括为：

```text
TrendModel = prompt understanding + task tagging + uncertainty estimation
Auto Router = capability matching + cost/quality decision + permission enforcement
```

## 2. 为什么需要单独的任务分析模型

直接把原始 prompt 交给 Router，让 Router 自己猜测该使用哪个模型，会把三个不同问题混在一起：

1. 用户到底想做什么；
2. 这个任务需要哪些能力；
3. 哪个模型以更低成本提供足够质量。

如果这三个问题全部交给生成模型完成，会出现几个问题：

- 每次请求都要额外调用一次分析模型，增加延迟和成本；
- 分析模型和最终生成模型之间形成循环依赖；
- JSON 输出和分类结果受模型生成风格影响，不够稳定；
- 工具权限、文件权限和风险信号不应该依赖自由生成的文本；
- Router 很难复用、回放和校准模型选择依据。

因此，TrendModel 把“理解任务”和“选择模型”解耦。TrendModel 提供可解释的 Task IR，下游 Router 只根据 Task IR、候选模型能力、价格、上下文窗口和权限做决策。

### 2.1 问题定义和符号

把一次用户请求记为字符串 `x`。模型不直接生成答案，而是计算一个结构化分析结果 `y`：

```text
x = original user prompt
A = TaskAST extracted from x
φ(A) = fixed-width structured feature vector
h = neural representation of x and φ(A)
p = multi-task probability distributions produced by h
y = Task IR normalized from A, p, and optional context
```

用更紧凑的形式表示：

```text
A = E(x)
p = Fθ(x, φ(A))
y = N(A, p, z)
```

其中：

- `E` 是确定性结构化提取器，负责从文本中找出路径、命令、代码块、语言、操作和风险等可观察信号；
- `φ` 把 TaskAST 编码成当前 TensorFlow 模型使用的 56 维向量；
- `Fθ` 是参数为 θ 的轻量级多任务模型；
- `z` 是可选的 Project、附件、工具权限等上下文。当前 TensorFlow 公开 `predict(prompt)` 接口还没有把 `z` 接进来；
- `N` 是归一化、证据融合、不确定性计算和 JSON 序列化过程。

这个问题本质上是一个多输出预测问题，而不是单一的“分类一个标签”：

```text
workload_type       ∈ {text, code, analysis, operations, planning, mixed, other}
intent              ∈ {create, modify, debug, explain, execute, ...}
complexity          ∈ {low, medium, high}
effort_bucket       ∈ {XS, S, M, L, XL}
effort_ordinal      ∈ {P(Y > XS), P(Y > S), P(Y > M), P(Y > L)}
needs_clarification ∈ {false, true}
```

这些输出之间有关联，但不完全是父子关系。例如，“C++”可以出现在解释、创建、修改或审阅任务中。因此当前 TensorFlow 模型使用共享表示和多个 heads，而不是强迫所有请求沿一棵单一路径分类。`TaskTagger` 原型路径则会先用 intent 影响 domain、artifact 和 workload 的选择，这是一种更显式的层级推理。

## 3. 总体设计

当前设计采用四类证据的组合：

```mermaid
flowchart TD
    A["用户 prompt"] --> B["清洗与规范化"]
    B --> C["文本语义编码"]
    B --> D["结构化证据提取"]
    C --> E["多任务分类 heads"]
    D --> F["语言、文件、命令、工具、风险特征"]
    E --> G["概率与不确定性"]
    F --> G
    G --> H["受限证据融合"]
    H --> I["Task IR / JSON contract"]
    I --> J["Auto Router"]
    J --> K["模型选择、成本策略与权限检查"]
```

先记住三句话：

1. **Task AST 是证据层**：它记录 prompt 中观察到的事实和多个候选解释；
2. **Task IR 是接口层**：它把融合后的任务判断整理成 Router 可以消费的稳定结构；
3. **Bayesian-style fusion 是更新层**：它把语义模型、结构证据和上下文分数合在一起，并把冲突暴露为不确定性。

### 3.1 Task AST：把 prompt 转成可计算的任务证据

#### 3.1.1 先理解普通 AST 是什么

AST 是 **Abstract Syntax Tree** 的缩写，中文通常译为“抽象语法树”。它是编译器前端把源代码从一串字符转换成结构化节点的一种表示（Aho et al., 2006; Cooper & Torczon, 2012）。

例如，源代码表达式：

```text
a + b * c
```

不能只看成五个平铺的字符或 token。编译器需要知道乘法优先级高于加法，因此可以把它表示为：

```text
        +
       / \
      a   *
         / \
        b   c
```

这棵树表达了两个重要事实：

1. `+` 是根操作，左右子节点分别是 `a` 和 `b * c`；
2. `*` 是加法右侧的子操作，所以先计算 `b * c`。

AST 和原始文本的区别在于：原始文本强调“字符顺序”，AST 强调“结构和关系”。编译器可以在 AST 上进行类型检查、作用域分析、优化和代码生成，而不必每一步重新从字符串猜测结构。

这里的“树”也不是简单的 JSON 嵌套。它通常包含：

- 节点类型，例如表达式、调用、变量或操作；
- 节点之间的父子关系；
- 节点属性，例如名称、位置和字面量；
- 由解析过程得到的结构约束。

#### 3.1.2 TrendModel 为什么也叫 Task AST

TrendModel 借用了“先把原始输入变成结构化中间层”的思想，但 **Task AST 不是编程语言 AST**。它不会解析完整的中文语法，也不会为每个词建立严格的句法树。更准确地说，TrendModel 的 Task AST 是一个 **router-oriented, AST-like evidence representation**：面向路由的、具有树状组织方式的任务证据表示。

它的根节点代表一次 `TaskRequest`，下面按任务分析所需的维度组织子节点：

```text
TaskRequest
├── action candidates: create | modify | debug | explain | execute | ...
├── domain candidates: programming | networking | operations | document | ...
├── artifact candidates: source_code | config | command | explanation | ...
├── operations: dns | kubernetes | database | testing | ...
├── language: cpp | python | shell | sql | ...
├── constraints
│   ├── requires_files
│   ├── requires_tools
│   ├── has_code_block
│   └── has_shell_commands
├── risk: production | destructive | privileged | ...
└── evidence signals: kind, label, value, strength, detail
```

因此，Task AST 这个名字强调的是 **组织方式和处理阶段**，不是宣称我们实现了一个通用中文句法分析器。它回答的问题是：

```text
在送入轻量分类模型之前，用户请求中有哪些可观察的任务事实和候选解释？
```

它同时保存两种信息：

1. **可观察事实**：例如文本中确实出现了 `kubectl`、`./deploy/dns.yaml`、代码块或数学符号；
2. **候选证据**：例如出现 `kubectl` 后，`operations` 候选得到较高支持，但还没有成为最终 workload label。

这一区分很重要。一个 prompt 可以同时包含“解释原因”“修改文件”和“运行命令”三个动作。Task AST 不应该在规则提取阶段假装这些冲突不存在；它应该把它们都保留下来，交给后面的语义模型、融合策略和安全规则处理。

#### 3.1.3 当前实现的正式定义

这里的 Task AST 不是编程语言的语法树，也不是让模型生成一棵完整的自然语言 parse tree。它是一个面向任务路由的、AST-like 的中间结构：根节点代表一次用户请求，子节点代表动作、领域、交付物、操作、约束、风险和证据。它的目的不是直接给出最终标签，而是把“文本中明确观察到的事实”和“由语义模型推断出的候选”分开保存。

概念上的 Task AST 可以表示为：

```text
TaskRequest
├── action: create | modify | debug | explain | execute | ...
├── domains: programming | networking | operations | document | ...
├── artifact: source_code | config | command | explanation | ...
├── operations: dns | kubernetes | database | testing | ...
├── language: cpp | python | shell | sql | ...
├── constraints
│   ├── requires_files
│   ├── requires_tools
│   ├── has_code_block
│   └── has_shell_commands
├── risk: production | destructive | privileged | ...
└── evidence: observable signals with strength and detail
```

当前 `TaskAST` dataclass 统一封装 `StructuredEvidence` 过去分散承载的内容：`TaskConstraints`、`TaskEvidence` 序列、`intent_scores`、`domain_scores`、`artifact_scores`、语言、操作和风险原因。实现见 [`trendmodel/task_ast.py`](../trendmodel/task_ast.py) 和 [`trendmodel/evidence.py`](../trendmodel/evidence.py)。它仍然不是一个通用 AST parser；`extract_evidence()` 输出的是候选分数和可观测信号，不是最终分类结果。同一个请求可以同时拥有 `debug`、`modify` 和 `execute` 的证据，最终选择仍由语义模型、规则优先级和融合策略共同决定。原来的 `StructuredEvidence` 名称保留为 `TaskAST` 的兼容别名，避免破坏已有调用方。

当前代码中的 `TaskAST` 是一个不可变的 Python `dataclass`。它把原来分散在 `StructuredEvidence`、规则返回值和若干局部变量里的信息放进一个明确契约中（实现见 [`trendmodel/task_ast.py`](../trendmodel/task_ast.py) 和 [`trendmodel/evidence.py`](../trendmodel/evidence.py)）。其字段可以按下表理解：

| 字段 | 类型/示例 | 含义 | 是否是最终结论 |
| --- | --- | --- | --- |
| `original_prompt` | `str` | 原始请求文本 | 否，是审计输入 |
| `constraints` | `TaskConstraints` | 文件、工具、代码块、路径、命令、数学符号、URL 等布尔事实 | 否 |
| `language` | `"python"`、`"cpp"`、`"shell"` | 规则检测到的语言或命令环境 | 否 |
| `operations` | `("dns", "kubernetes")` | 检测到的技术操作或对象 | 否 |
| `signals` / `evidence` | `TaskEvidence` 序列 | 每个可观察信号的 kind、label、value、strength、detail、source span、rule id、否定/引用状态 | 否 |
| `intent_scores` | `{"debug": 0.97}` | 动作候选的规则证据分数 | 否 |
| `domain_scores` | `{"operations": 0.80}` | 领域候选的规则证据分数 | 否 |
| `artifact_scores` | `{"command": 0.90}` | 交付物候选的规则证据分数 | 否 |
| `high_risk` | `True` / `False` | 是否发现生产、删除、提权、发布等风险信号 | 否，是安全提示 |
| `risk_reasons` | `("production", "migration_or_release")` | 风险原因列表 | 否 |

这里必须区分三个词：

- **事实（fact）**：文本里确实出现了某个路径、命令或结构；
- **候选分数（candidate evidence score）**：某条规则对一个候选标签的支持程度；
- **最终概率（model probability）**：模型经过 softmax 和融合后，对最终类别给出的分布分量。

TaskAST 中的 `strength` 和 `*_scores` 属于前两类，不等于最终 probability。它们表示“某条规则匹配得有多强”，而不是“最终分类正确的概率”。同一个请求可以同时有 `debug`、`modify` 和 `execute` 三种非零候选；TaskAST 的职责是保存这种不确定性和冲突，而不是提前强行选一个。

TaskAST 的几个不变量是：

1. `original_prompt` 必须是非空字符串；
2. 候选分数必须在 `[0, 1]`；
3. `operations` 和 `risk_reasons` 不允许出现空标签；
4. evidence 是可观察信号，不是模型自由生成的解释；
5. 每条 evidence 的 `strength` 必须在 `[0, 1]`，`source_span` 必须是原始 prompt 的有序字符区间；
6. `negated=true` 或 `quoted=true` 的 signal 可以被审计，但不能自动等同于用户要求执行；
7. `intent_scores`、`domain_scores` 和 `artifact_scores` 在构造后是深层只读 mapping，避免推理链路中的隐式改写；
8. `requires_tools=true` 和 `high_risk=true` 只是路由/安全输入，绝不等于已经获得工具权限；
9. `schema_version` 当前为 `task-ast-v1`，用于未来演进和回放。

当前保留 `signals` 字段是为了兼容旧调用方；新代码可以使用语义更清楚的 `evidence` 属性。`action_scores` 是 `intent_scores` 的 AST-oriented 别名。两者都指向候选证据，不是最终 intent label。

例如输入：

```text
请在生产 Kubernetes 集群里排查 DNS 超时，修改 ./deploy/dns.yaml，运行 kubectl rollout status 并给出回滚方案。
```

其 Task AST-like 结构应包含：

```text
action candidates: debug, modify, execute, plan
domains: operations, networking
artifact candidates: command, analysis_report, plan, source_code_or_config
operations: dns, kubernetes, deployment
language: shell (explicit command); file_extension: yaml (path evidence)
constraints: requires_files=true, requires_tools=true, has_file_paths=true,
             has_shell_commands=true
risk: production, migration_or_release
```

这些字段是“证据和候选”，不是权限授权。尤其是 `requires_tools=true` 或 `risk=production` 只能提醒下游 Router 进入安全检查，不能触发执行。

### 3.2 IR：先理解“中间表示”这个概念

IR 是 **Intermediate Representation** 的缩写，中文通常译为“中间表示”或“中间代码”。在编译器里，源代码 AST 经过语义分析后，会被转换成一种比源代码更规整、又比最终机器码更抽象的表示。编译器优化、不同硬件后端和代码生成阶段可以共同消费这种表示（Aho et al., 2006; Lattner & Adve, 2004）。

可以把编译器粗略想象成：

```text
source code
    ↓ parse
AST
    ↓ type checking / semantic analysis
IR
    ↓ optimization / code generation
machine code
```

IR 的价值不是“把前一步的对象换一个名字”，而是定义一个稳定的阶段间接口：后续模块不必理解所有源语言细节，只需要理解 IR 的字段和语义。

### 3.3 Task IR：把证据压缩成 Router 合同

在 TrendModel 中，**Task IR 是证据提取、语义分类和不确定性处理之后，供 Auto Router 消费的规范化任务表示**。它不是编译器意义上的机器码，也不是一个具体供应商模型的调用参数。它把“这个 prompt 可能是什么”转换为“Router 可以稳定读取哪些能力需求、风险和置信度”。

可以把 Task IR 写成一个函数：

```text
TaskIR = Normalize(
    semantic_predictions,
    structural_evidence,
    optional_context_prior,
    uncertainty_diagnostics,
    effort_supervision_basis
)
```

Task IR 不是以下任何东西：

- 不是原始 prompt 的 JSON 包装；
- 不是 TaskAST 的逐字段 dump；TaskAST 保留了更多中间证据和冲突；
- 不是最终模型选择结果；它不说“必须调用 Model-A”；
- 不是最终生成答案；
- 不是工具权限或安全审批；`execute` 和 `high_risk` 仍需下游策略确认。

原型 `TaskTagger` 直接构造 `TaskIR` dataclass，字段包括 `intent`、`domains`、`workload_type`、`artifact`、`operations`、`language`、`constraints`、`capability_demand`、`complexity`、`effort`、`risk_level`、`confidence`、`needs_clarification`、`abstention` 和 `evidence`（实现见 [`trendmodel/task_ir.py`](../trendmodel/task_ir.py) 和 [`trendmodel/tagger.py`](../trendmodel/tagger.py)）。`TaskIR.to_dict()` 保留 prototype 的旧直接字段，并增加 `task_ir` canonical object。

TensorFlow 主模型的 `TensorFlowTaskModel.predict()` 为兼容现有 Router 保留最外层的 `workload_type`、`intent`、`complexity`、`predictions` 等字段，同时通过 `TaskIR.from_tensorflow()` 生成 `task_ir` canonical object（实现见 [`trendmodel/task_ir.py`](../trendmodel/task_ir.py) 和 [`trendmodel/tf_inference.py`](../trendmodel/tf_inference.py)）。它与 prototype 共享 `task_ir.contract_version == "task-ir-v1"` 的字段集合；TensorFlow 当前没有独立的 domain/artifact heads，因此这两类字段在 canonical object 中明确标记为 TaskAST-derived 的候选，而不是伪造神经输出。

#### 3.3.1 TaskAST 与 Task IR 的区别

| 维度 | Task AST | Task IR |
| --- | --- | --- |
| 类比的处理阶段 | 前端提取/结构化分析 | 语义融合后的阶段间接口 |
| 主要目的 | 保存可观察事实、规则信号和多个候选 | 提供 Router 可以消费的稳定结论 |
| 信息形态 | 详细、可能有冲突、保留中间状态 | 规范化、包含选中标签和候选概率 |
| 是否必须有最终标签 | 不必须；可能只有候选 | 通常有 label，也允许 `unknown` |
| 分数含义 | 规则匹配强度或候选支持度 | head 输出/融合后的 posterior-like score |
| 是否保存风险事实 | 是，保存 `high_risk` 和原因 | 是，转换成 Router 可检查的风险字段或 evidence |
| 是否直接给 Router | 通常不直接给 | 是，Task IR/JSON 是接口 |
| 当前代码 | `TaskAST`，由 `extract_evidence()` 返回 | `TaskIR.to_contract()`；TensorFlow 和 prototype 都通过 `task_ir` 暴露 |
| 能否代替权限系统 | 不能 | 不能 |

最容易混淆的地方是：TaskAST 和 Task IR 都可以出现 `intent`、`domain` 或 `evidence` 这些词，但它们在不同阶段含义不同。举例：

```text
TaskAST.intent_scores["execute"] = 0.98
    = 规则发现了强执行语义

TaskIR.intent.probability = 0.61
    = 该路径对 execute 候选的归一化输出分数
```

前者说明“文本里有什么信号”，后者说明“该路径当前准备把它当成什么”。两者不能直接当作同一个 probability 比较。

#### 3.3.2 当前代码中的数据流

```text
original prompt
      │
      ▼
extract_evidence(prompt)
      │
      ▼
TaskAST
  ├── constraints / language / operations / risk
  ├── evidence signals
  └── intent/domain/artifact candidate scores
      │
      ├──────────────► structured_feature_vector ──► 56-dim TensorFlow input
      │                                      │
      │                                      ▼
      │                               TensorFlow multi-task heads
      │                                      │
      │                                      ▼
      │                               raw predictions
      │
      └──────────────► TaskTagger prototype + fusion ──► prototype TaskIR

raw predictions + TaskAST evidence
      │
      ▼
TaskIR.from_tensorflow() / TaskIR.to_contract()
      │
      ▼
shared task_ir canonical contract
```

有三个实现边界需要明确：

1. `TaskAST` 现在已经是正式 dataclass，不再是只存在于文档中的概念；
2. prototype `TaskTagger` 和 TensorFlow adapter 都生成同一版本的 canonical `task_ir`；
3. TensorFlow 顶层仍保留旧字段，但新 Router 应读取 canonical object，而不是依赖某一模型的兼容包装。

因此，TaskAST 是“证据层”，Task IR 是“路由接口层”。前者帮助我们知道模型为什么有这样的候选，后者帮助下游稳定地使用结果。

### 3.4 完整例子：从中文 prompt 到 Task IR

下面用一个混合自然语言、代码、文件和生产操作的请求走完整链路：

```text
请在生产 Kubernetes 集群里排查 DNS 超时，修改 ./deploy/dns.yaml，
运行 kubectl rollout status 并给出回滚方案。
```

下面的概率数字是为了展示算法如何运作的教学示例，不是当前 artifact 的 benchmark 输出。

#### 第一步：保留原始输入并做模型规范化

系统保留 `original_prompt` 作为审计和最终输出字段；模型分支只做 Unicode/换行规范化，不改变任务语义：

```text
x_raw   = 原始两行 prompt
x_model = 规范化换行后的 prompt
```

这一步的原因是：训练时需要稳定的输入，回放和下游审计又需要知道用户原话。原始文本和模型输入不应混为一谈。

#### 第二步：提取 TaskAST

```text
TaskAST.root.type = task_request

action candidates:
  debug   ← “排查 DNS 超时”
  modify  ← “修改 ./deploy/dns.yaml”
  execute ← “运行 kubectl rollout status”
  plan    ← “给出回滚方案”

domain candidates:
  operations
  networking
  programming (弱候选，因为存在配置文件/命令，不代表写程序)

artifact candidates:
  command
  source_code_or_config
  analysis_report
  rollback_plan

operations:
  dns, kubernetes, deployment

language:
  shell (来自 kubectl 命令)

constraints:
  requires_files=true
  requires_tools=true
  has_file_paths=true
  has_shell_commands=true
  has_code_block=false

risk:
  high_risk=true
  reasons=(production, migration_or_release)
```

这一步没有“授权 kubectl”，也没有“决定必须使用某个大模型”。它只报告文本中观察到的事实和候选解释。`requires_tools=true` 的真正含义是：下游应该检查是否允许工具调用；如果没有权限，Router 必须选择解释/规划路径，而不是执行路径。对于“不要执行 kubectl…”或“解释 `rm -rf`”这样的 prompt，相关 command/risk evidence 仍会返回，但会带 `negated` 或 `quoted` 标记，active intent/risk 不会被错误激活。

#### 第三步：编码成 TensorFlow 输入

同一个 TaskAST 被两路使用：

```text
字符文本分支:
  x_model → character TextVectorization → Embedding → BiGRU

结构化分支:
  TaskAST → 56-dimensional feature vector → Dense(32)

词法 shortcut 分支:
  x_model → character 2–5 gram TF-IDF → Dense(128)
```

这样做的直觉是：BiGRU 负责“整句话的顺序和上下文”，n-gram 负责“kubectl rollout、dns.yaml 这类稳定片段”，TaskAST 分支负责“文本中是否确实存在文件、命令、生产和 Kubernetes 等结构事实”。

#### 第四步：神经模型输出 raw heads

共享表示经过多任务 heads 后，得到：

```text
workload_type probabilities:
  code=0.40, operations=0.35, text=0.25

intent probabilities:
  debug=0.46, modify=0.31, execute=0.18, plan=0.05

complexity probabilities:
  low=0.05, medium=0.37, high=0.58

effort_bucket probabilities:
  XS=0.02, S=0.08, M=0.29, L=0.50, XL=0.11
```

再次强调，这些是教学用的 raw probability 分布；真实输出由当前 artifact 的权重决定。

#### 第五步：对不同 head 使用不同证据

结构证据不能无差别地覆盖所有 head：

| Head | 主要证据 | 原因 |
| --- | --- | --- |
| workload | `operations`、`kubernetes`、`dns`、`shell` | 这些事实能支持 operations/networking 方向 |
| intent | `排查`、`修改`、`运行` 等动作短语 | 动作短语比“代码/网络”等名词更能区分意图 |
| complexity | 生产、文件、工具、部署、回滚、多步骤 | 这些信号增加上下文和验证需求 |
| effort | bucket head、complexity、真实 observed minutes（如有） | effort 需要独立监督，不能由规则硬猜 |

因此，“出现 Python”不应直接等于 `create`；“出现 `kubectl`”也不应直接等于“可以执行”。

#### 第六步：生成 Task IR/JSON

融合和不确定性处理后，Router 看到的结果可以是：

```json
{
  "original_prompt": "请在生产 Kubernetes 集群里排查 DNS 超时，修改 ./deploy/dns.yaml，运行 kubectl rollout status 并给出回滚方案。",
  "workload_type": "operations",
  "intent": "debug",
  "complexity": "high",
  "predictions": {
    "workload_type": {
      "label": "operations",
      "probability": 0.7712,
      "candidates": [
        {"label": "operations", "probability": 0.7712},
        {"label": "code", "probability": 0.1408},
        {"label": "text", "probability": 0.088}
      ]
    },
    "intent": {
      "label": "debug",
      "probability": 0.61,
      "candidates": [
        {"label": "debug", "probability": 0.61},
        {"label": "modify", "probability": 0.20},
        {"label": "execute", "probability": 0.12}
      ]
    },
    "complexity": {
      "label": "high",
      "probability": 0.74,
      "candidates": [
        {"label": "high", "probability": 0.74},
        {"label": "medium", "probability": 0.22},
        {"label": "low", "probability": 0.04}
      ]
    }
  },
  "effort": {
    "bucket": "L",
    "estimated_minutes": 240,
    "range_minutes": [120, 480],
    "basis": "trained_effort_bucket_head"
  },
  "needs_clarification": false,
  "evidence": [
    {"kind": "operation", "label": "kubernetes", "strength": 0.92},
    {"kind": "risk", "label": "production", "strength": 0.90}
  ]
}
```

这个结果告诉 Router：任务更像生产运维排障，需要高能力和上下文的候选模型，effort 粗粒度为 `L`，但它没有告诉 Router “选择 Model-X”，也没有获得执行权限。下游仍需要检查工具授权、生产变更审批、成本预算和候选模型能力。

Task IR 的设计原则是：

1. 标签和候选可机器读取；
2. evidence 可回放、可审计，不依赖模型自由生成解释；
3. confidence、entropy、margin 与标签分离，允许 Router 自己设阈值；
4. effort 带有 `basis`，明确区分真实分钟监督、bucket 监督和无监督 fallback；
5. Task IR 不包含模型选择结果，Router 仍然负责能力匹配、成本/质量权衡和权限执行。

### 3.5 从输入到输出的算法链路

下面的伪代码描述当前 TensorFlow 主链路。`TaskAST` 保存结构化 evidence；Bayesian-style 融合只属于可选 prototype/未来校准层，不在当前 TensorFlow `predict` 中二次修改分类概率。

```text
Algorithm 1: TrendModelAnalyze(prompt, optional_context)

Input:
    x_raw       = original user prompt
    z           = optional project/file/tool context
Output:
    y           = router-facing Task IR / JSON

1.  Validate x_raw is non-empty.
2.  x_model ← normalize line separators for model tokenization;
    preserve x_raw for output and audit.
3.  A ← ExtractTaskAST(x_raw)
    where A contains language, actions, domains, artifact hints,
    operations, constraints, risk and observable evidence.
4.  s_struct ← EncodeStructuredFeatures(A)        # current width: 56
5.  h_char ← BiGRU(Embedding(CharacterVectorize(x_model)))
6.  h_ngram ← Dense(CharacterNgramTFIDF(x_model))
7.  h ← Fuse(h_char, h_ngram, Dense(s_struct))
8.  p_raw ← MultiTaskHeads(h)
    where heads = workload, intent, complexity, effort_bucket,
    effort_ordinal, effort_log_minutes and needs_clarification.
9.  evidence ← Serialize(A.evidence)
    # return the observable TaskAST signals; do not alter p_raw.
10. effort ← ConvertEffortHeads(p_raw, complexity, observed_effort_metadata)
11. uncertainty ← Compute(probability, entropy, margin, overall_confidence)
12. abstention ← ApplyAbstentionPolicy(predictions, metadata)
13. clarification ← ApplyClarificationPolicy(p_raw, x_raw, abstention)
14. y ← BuildTaskIRJSON(x_raw, predictions=p_raw, effort, evidence,
                         abstention, fusion_metadata, uncertainty)
14. return y
```

步骤 3 先生成一份 TaskAST；步骤 4 把这份 AST 编码成 56 维结构化输入，步骤 9 返回同一份 AST 的 evidence，避免训练输入和审计输出来自两套规则。步骤 10–13 都不会把证据再混入分类概率；abstention 只读取已经产生的概率和不确定性，不修改 label。`needs_clarification` 是独立的路由安全信号。旧的 `TaskTagger` 仍可作为 prototype 运行，但不再是 TensorFlow 主推理的第二个概率来源。

核心思想不是用一个算法解决所有问题，而是给不同类型的信号分配不同的责任：

| 信号类型 | 负责什么 | 适合的算法 |
| --- | --- | --- |
| 自然语言语义 | “这句话整体想做什么” | 字符级编码、句向量、分类 head |
| 代码/路径/命令/数学符号 | “文本里明确出现了什么” | 确定性规则和结构化 extractor |
| 历史上下文 | “当前场景下哪些任务更可能” | context prior |
| 人工反馈与任务记录 | “这个判断过去是否可靠、实际花了多久” | gold supervision、校准、回放 |
| 多种证据的冲突 | “当前结论有多确定” | 概率融合、entropy、margin、fallback |

当前代码中有两条互补的实现路径，不能把它们当成同一个模型：

- `TaskTagger` 是可选的 prototype 分类器。它可以接入真正的中文/多语言句向量模型，也可以使用默认的无依赖 `HashEmbedder`；它负责原型相似度、可选的 Bayesian-style 证据融合和解释性 Task IR。
- `TensorFlowTaskModel` 是当前主要训练和部署的 artifact。它使用字符级 BiGRU、字符 n-gram TF-IDF、TaskAST 结构化特征和多任务 heads，输出下游约定的 JSON。推理时，TaskAST 只作为模型输入并以 evidence 返回，TensorFlow 概率不再经过 TaskTagger 或规则的 post-hoc 修改。

因此，“句向量 + 原型分类”是系统的语义 teacher/演进方向；“字符级轻量模型 + 结构化分支”是目前可在 M4 和 RTX 4070 本地训练、保存和跨平台加载的 TensorFlow 主模型。这样既保留了未来替换语义 encoder 的空间，也不会让第一版训练依赖一个额外的在线或大型预训练模型。

`context prior` 已在通用 `TaskTagger` 融合接口中预留，但当前 `TensorFlowTaskModel.predict(prompt)` 的公开接口不接收 project、附件或权限上下文；这类上下文暂时由下游 Router 处理，后续可以在不改变 JSON contract 的情况下接入。

## 4. 为什么不把贝叶斯当作中文语义模型

中文和混合语言 prompt 中经常出现：

- 分词歧义；
- 中英文、代码和自然语言混用；
- 主语和动作省略；
- 同一个词在不同语境下含义不同；
- “实现”可能意味着写代码，也可能只是解释实现思路；
- 文件路径、Shell 命令和自然语言出现在同一个请求中。

朴素贝叶斯或词袋统计适合处理词频证据，但不适合承担完整的句子级语义理解。例如：

```text
解释 MANET 中的路由协议如何实现
实现一个 MANET 路由协议模拟器
```

两个请求共享大量词语，但前者更接近技术解释，后者更接近代码创建。词频统计很容易被“实现”“路由”“协议”等共享词汇吸引，难以稳定区分动作和交付物。

所以 TrendModel 的分工是：

```text
文本编码器       负责语义表示
结构化 extractor 负责高置信确定性信号
概率/贝叶斯层     负责证据融合和不确定性管理
```

### 4.1 先理解 Bayes 在这里解决什么问题

贝叶斯定理回答的是一个很具体的问题：当我们先有一个类别的先验倾向，又观察到文本和结构证据之后，应该如何更新对这个类别的相信程度（Bayes, 1763）？

在分类场景里，可以把几个词先记住：

- **先验（prior）**：还没看当前 prompt 之前，某个类别本来有多常见；
- **似然（likelihood）**：如果这个类别是真的，当前证据出现的可能性有多大；
- **后验（posterior）**：看完当前证据后，对这个类别的更新倾向；
- **证据（evidence）**：当前 prompt、TaskAST 信号和上下文。

贝叶斯形式可以写成：

```text
P(class | text, evidence, context)
  ∝ P(text | class) × P(evidence | class) × P(class | context)
```

左边是我们想知道的“在所有证据下，属于某个 class 的后验倾向”。右边分成三个来源：

1. `P(text | class)`：文本整体语义和这个类别的匹配程度；
2. `P(evidence | class)`：TaskAST 发现的路径、命令、语言、操作等结构信号对这个类别的支持程度；
3. `P(class | context)`：Project、附件、当前工具权限等上下文带来的先验倾向。

在当前系统中，`P(text | class)` 不由朴素贝叶斯直接估计，而由神经分类 head 或句向量与原型的相似度间接提供；`P(evidence | class)` 来自结构化 extractor；`P(class | context)` 是可选的上下文先验。

这个公式的工程意义是“把不同来源的证据放在同一套更新逻辑里”，而不是要求系统真的知道每个条件概率的真实统计值。它也解释了为什么贝叶斯层应该放在语义模型之后：如果文本语义本身还没有被编码，规则证据只能告诉我们“出现了什么”，却不能可靠地告诉我们“用户想完成什么”。

### 4.2 一个具体的数值例子

假设 TensorFlow workload head 对一个 prompt 的 raw 分布是：

```text
p_neural(code)       = 0.40
p_neural(operations) = 0.35
p_neural(text)       = 0.25
```

TaskAST 发现了高置信的 Kubernetes、Shell 命令和生产环境信号，因此可以在 evidence 层记录 `operations` 候选，并估计 `evidence_strength = 0.90`。下面的数值只用于解释“如果未来启用 Bayesian-style post-hoc fusion 会发生什么”；当前 TensorFlow 推理不执行这一步：

```text
α = min(0.72, 0.72 × evidence_strength)
  = 0.72 × 0.90
  = 0.648

p_fused = (1 - α) × p_neural + α × one_hot(structural_label)
```

于是：

```text
p_fused(operations) = 0.352 × 0.35 + 0.648 × 1.00 = 0.7712
p_fused(code)       = 0.352 × 0.40 + 0.648 × 0.00 = 0.1408
p_fused(text)       = 0.352 × 0.25 + 0.648 × 0.00 = 0.0880
```

这说明了 prototype/未来设计中的两个决策：

- 结构化证据能够把“明显的运维请求”从 `code=0.40` 的模糊 raw 输出拉向 `operations`；
- 结构证据最多占有限比例，仍然保留神经模型的 35.2% 权重，因此规则不会无条件覆盖语义模型。

这里的 `0.7712` 是融合后的 posterior-like 分数，不是经过独立人工 gold 集校准后的真实正确率。只有当我们收集独立 gold 数据并进行 calibration 后，才可以讨论“预测为 0.8 的样本大约有多少比例正确”。

### 4.3 可选 prototype：Bayesian-style，而不是严格 Bayes 模型

当前实现并没有假设所有概率都已经严格校准，也没有假设文本、规则和上下文之间相互独立，而是采用两种受限的 Bayesian-style 近似。

在原型 `TaskTagger` 路径中，`fusion.py` 默认使用概率混合：

```text
p_mix = 0.70 × p_semantic
      + 0.20 × p_structural
      + 0.10 × p_context_prior
```

缺失的 evidence source 使用均匀分布作为中性输入，避免“没有检测到 Shell 命令”被错误地解释成“肯定不是 Shell”。代码中还保留了乘法型的 `P(class | evidence) ∝ likelihood × prior` 实验实现，但默认没有启用，因为稀疏规则证据可能造成过度收缩。

历史设计曾考虑在 TensorFlow 对外推理路径使用更保守的结构化证据修正：

```text
p_fused = (1 - α) × p_neural + α × one_hot(structural_label)
α = min(0.72, 0.72 × evidence_strength)
```

P0 方案 A 已经关闭这条 post-hoc 路径。当前 TensorFlow 对外行为是：`TaskAST → 56 维输入 → TensorFlow heads`，然后返回 `TaskAST.evidence`；`fusion.post_hoc_probability_fusion` 固定为 `false`。上式和后续融合示例仅保留用于解释 prototype `TaskTagger` 与未来实验，不代表当前 artifact 的输出公式。

这意味着结构化证据最多可以占据有限比例的后验，不会无条件覆盖神经模型。这样做有两个原因：

1. 明确的 Shell、代码块或语言标记应该影响分类；
2. extractor 也可能误判，不能把规则直接变成绝对硬标签。

高风险操作仍然需要下游权限检查，不能因为融合分数高就自动执行。

需要特别区分下面四种数字：

| 数字 | 来源 | 实际含义 | 能否直接解释为真实正确率 |
| --- | --- | --- | --- |
| `TaskEvidence.strength` | 规则 extractor | 某条结构规则匹配得有多强 | 不能 |
| `TaskAST.intent_scores` 等 | 规则候选分数 | 某个候选标签获得多少结构支持 | 不能 |
| `predictions.*.probability` | TensorFlow softmax/ordinal distribution | 选中 label 的模型分量；新 artifact 可经过 temperature scaling | 只有新 artifact 的 validation calibration 后才可近似解释 |
| `confidence.overall` | 多个 head 的加权组合 | 供 Router 使用的总体确定性分数 | 不能 |

当前训练流程已经在 validation split 上执行 temperature scaling，并把 NLL 与 ECE 写入 artifact metadata；如果要把它作为生产级概率使用，仍应在没有参与训练、规则调整和阈值选择的独立 gold validation 集上复核，并报告 ECE、Brier score 和 reliability diagram（Guo et al., 2017）。

## 5. 任务标签体系

### 5.1 Workload

当前 TensorFlow 模型使用以下 workload 标签：

```text
text
code
analysis
operations
planning
mixed
other
```

它回答的是“这是什么类型的工作”，不是“具体做哪一个动作”。例如：

| Prompt | Workload | Intent |
| --- | --- | --- |
| 解释 C++ 多线程原理 | `code` | `explain` |
| 写一个 C++ 多线程程序 | `code` | `create` |
| 排查 Kubernetes DNS 超时 | `operations` | `debug` |
| 制定数据库迁移方案 | `planning` | `plan` |
| 总结一份技术报告 | `text` | `summarize` |

### 5.2 Intent

当前 intent 标签为：

```text
create / modify / debug / explain / execute / analyze / review
summarize / translate / plan / extract / classify / unknown
```

Intent 比关键词更重要，因为 Router 通常需要区分同一个领域中的不同动作：

```text
代码 + explain   → 需要理解和讲解能力
代码 + create    → 需要生成代码能力
代码 + modify    → 需要阅读上下文和修改能力
代码 + review    → 需要审阅和风险分析能力
Shell + execute  → 需要工具权限和执行能力
Shell + explain  → 只需要解释命令，不应自动执行
```

### 5.3 Complexity

复杂度是任务范围和推理需求的粗粒度判断：

```text
low / medium / high
```

它不是简单的 prompt 长度。影响复杂度的因素包括：

- 是否需要多步骤推理；
- 是否需要读写文件或理解上下文；
- 是否涉及多个系统或组件；
- 是否需要代码、测试、部署和验证闭环；
- 是否具有生产环境、风险或回滚要求；
- 是否存在多个交付物。

### 5.4 Effort bucket

Effort 用于 Router 的成本和模型能力排序，当前使用：

```text
XS / S / M / L / XL
```

初始区间是一个可解释的 policy：

| Bucket | 初始范围 |
| --- | --- |
| `XS` | 5–15 分钟 |
| `S` | 15–45 分钟 |
| `M` | 45–120 分钟 |
| `L` | 120–480 分钟 |
| `XL` | 1,440–4,320 分钟 |

这些区间不是承诺，也不是当前已经校准的真实工时。没有 `active_minutes` 时，模型只能学习 bucket 先验；只有内部真实任务记录才能训练分钟回归 head。

## 6. 当前 TensorFlow 模型结构

当前版本使用一个轻量级多任务模型，而不是为每个标签训练一个完全独立的模型。

```mermaid
flowchart LR
    A["original_prompt"] --> B["Character TextVectorization\nmax 12,000 / length 192"]
    B --> C["Embedding 96"]
    C --> D["Bidirectional GRU\n64 units each direction"]
    D --> E["Dense 128"]

    A --> F["Character n-gram TF-IDF\n2-gram to 5-gram / max 24,000"]
    F --> G["Dense 128"]

    A --> H["Structured feature extractor\n56 fixed-width features"]
    H --> I["Dense 32"]

    E --> J["Text representation"]
    G --> J
    J --> K["Dense 160"]
    K --> L["Concatenate text + structure"]
    I --> L
    L --> M["Dense 128 + Dropout 0.2"]

    M --> N["workload_type"]
    M --> O["intent"]
    M --> P["complexity"]
    M --> Q["effort_bucket"]
    M --> R["effort_ordinal"]
    M --> S["effort_log_minutes"]
    M --> T["needs_clarification"]
```

### 6.1 字符级文本分支

字符级编码是当前中文和混合语言场景下的工程折中：

- 不依赖中文分词器；
- 能保留 `C++`、文件扩展名、路径、Shell flag 和代码符号；
- 对中英文混用更稳健；
- 模型尺寸和本地推理成本可控。

字符级 BiGRU 用于捕捉顺序和局部上下文。它能够区分“解释某段代码”和“实现某个程序”中动作与对象的组合，而不仅仅统计关键词出现次数。GRU 的门控序列建模来自 Cho 等人（2014），双向循环结构参考 Schuster 与 Paliwal（1997）。

### 6.2 字符 n-gram TF-IDF 分支

字符 n-gram 分支提供一种高效的 lexical shortcut。TF-IDF 的基本思想和 term weighting 传统见 Salton 与 Buckley（1988）：

- `kubectl rollout`、`api.py`、`C++`、`SELECT ...` 等片段可以形成稳定信号；
- 对短 prompt 比单纯依赖 recurrent encoder 更友好；
- 不需要引入额外的 tokenizer 或句向量服务；
- 与 BiGRU 的语义表示互补。

这个分支不是最终语义理解器，而是帮助小模型快速捕捉任务形状和高频表达。

### 6.3 结构化特征分支

当前固定宽度为 56 维，分为：

| 特征组 | 数量 | 示例 |
| --- | ---: | --- |
| 文本形状 | 5 | prompt 长度、行数、中文/拉丁字母/数字比例 |
| 结构信号 | 8 | 代码块、文件路径、Shell、数学符号、URL、文件/工具需求、风险 |
| 语言 | 10 | C++、Python、JavaScript、SQL、Shell 等 |
| intent hint | 13 | create、modify、debug、explain、execute 等 |
| domain hint | 11 | programming、networking、document、operations 等 |
| operations | 9 | DNS、Kubernetes、数据库、部署、测试等 |

结构化特征的作用是把高置信事实显式暴露给模型。例如，模型不需要只靠 192 个字符 token 自己猜测某个 prompt 是否包含代码块或 Shell 命令。

这不是把规则硬编码成最终答案。结构化特征进入训练模型，同时在推理阶段作为受限 evidence 参与融合，最终仍然保留神经模型的不确定性。

## 7. 多任务学习与损失函数

共享表示后接多个任务 head：

| Head | 输出 | Loss |
| --- | --- | --- |
| `workload_type` | 多分类概率 | Sparse Categorical Cross Entropy |
| `intent` | 多分类概率 | Sparse Categorical Cross Entropy |
| `complexity` | 多分类概率 | Sparse Categorical Cross Entropy |
| `effort_bucket` | 多分类概率 | Sparse Categorical Cross Entropy |
| `effort_ordinal` | 四个顺序阈值概率 `P(Y > XS)` … `P(Y > L)` | Binary Cross Entropy |
| `effort_log_minutes` | `log1p(minutes)` 回归值 | Huber Loss |
| `needs_clarification` | 0–1 概率 | Binary Cross Entropy |

初始 loss 权重为：

```text
L_total =
    1.0 × L_workload
  + 1.0 × L_intent
  + 0.8 × L_complexity
  + 0.8 × L_effort_bucket
  + 0.8 × L_effort_ordinal
  + 1.0 × L_effort_log_minutes
  + 0.7 × L_needs_clarification
```

选择多任务学习有几个原因。共享表示、同时学习相关任务的思想见 Caruana（1997）：

- workload、intent 和 complexity 共享大量语义信息；
- 结构化信号可以同时帮助多个 head；
- effort 与 complexity 有关联，但可以单独监督和校准；
- 缺失标签时可以通过 sample weight mask 只训练存在的 head；
- 共享 encoder 比为每个任务训练一套模型更适合本地轻量部署。

## 8. Effort 监督的边界

这是整个系统最容易产生错误假设的地方。

### 8.1 Bucket-only 样本

如果样本只有：

```json
{
  "effort_bucket": "M"
}
```

它只能监督 `effort_bucket` 分类 head。代码为了兼容旧数据，会在内部保留一个 midpoint，但这个 midpoint 不是实际工时，不能训练分钟回归。

### 8.2 Observed effort 样本

只有包含真实主动工作时间的记录才可以监督分钟回归：

```json
{
  "active_minutes": 120,
  "effort_bucket": "M"
}
```

训练时回归目标是：

```text
y_effort = log1p(active_minutes)
```

使用 `log1p` 的原因是工时通常是长尾分布。直接回归分钟数会让少数极大任务主导 loss，log 空间可以压缩长尾，同时保留任务大小顺序。

### 8.3 推理 basis

输出中的 `effort.basis` 表示 effort 结论的来源：

| Basis | 含义 |
| --- | --- |
| `trained_effort_bucket_head` | 有 bucket head，但没有真实分钟监督 |
| `trained_effort_ordinal_head` | 使用 ordinal effort head，但没有真实分钟监督 |
| `trained_observed_effort_and_bucket_heads` | 同时有真实分钟和 bucket 监督 |
| `trained_observed_effort_and_ordinal_heads` | 同时有真实分钟和 ordinal bucket 监督 |
| `trained_observed_effort_head` | 旧 artifact 的分钟回归模型 |
| `bucket_seed_only` | 旧 text-only artifact 的 bucket fallback |
| `no_effort_supervision` | 没有可用 effort supervision |

如果 basis 不是 observed effort 类型，下游只应该把 effort 当作排序特征，不应该把 `estimated_minutes` 当作 SLA 或承诺。

## 9. 标签质量、类别不平衡与 Gold Replay

### 9.1 标签 provenance

每条记录保留：

- `source`：来自哪个数据源；
- `split_group`：对话、session 或任务组；
- `label_source`：manual、gold、weak、pseudo 等来源；
- `label_confidence`：弱标签的保守置信度；
- `label_confidence_by_field`：不同 head 的标签质量。

这使得训练报告可以按 source 和 label quality 分组，而不是只输出一个被弱标签污染的总准确率。

### 9.2 类别平衡

公共数据通常存在明显长尾，例如 `low` complexity 或某个 workload 占绝大多数。当前使用 capped inverse-square-root weighting：

```text
raw_weight(c) = sqrt(total_weight / (number_of_active_classes × count(c)))
weight(c) = clip(raw_weight(c), 1 / cap, cap)
```

当前 `cap=3.0`，并对权重做归一化，使整体梯度尺度保持接近 1。使用平方根而不是完整 inverse frequency，是为了避免少量噪声标签被过度放大。

### 9.3 Gold Replay

即使每条 weak label 的权重低，几千条公共样本仍然可能淹没几十条人工样本。因此训练阶段对可信的 manual/gold/seed/observed label 额外使用 replay multiplier，当前默认值为 8：

```text
effective_weight = label_weight × confidence_weight × gold_replay_multiplier
```

这个 multiplier 只影响训练 sample weight，不改变 validation 数据，也不会重复写入数据行。

它不是最终超参数。后续应在独立 gold validation 上比较 4、8、16 等值，选择泛化最好的配置。

## 10. 数据处理和本地化边界

数据流水线为：

```mermaid
flowchart LR
    A["Hugging Face / internal source"] --> B["local raw files"]
    B --> C["source adapter"]
    C --> D["normalize + redact + deduplicate"]
    D --> E["weak labels / manual labels"]
    E --> F["group-aware train/validation split"]
    F --> G["local TensorFlow training"]
    G --> H["model.keras + metadata.json"]
```

系统明确采用本地数据边界：

- Hugging Face 只负责一次性下载；
- 原始数据保存在 `data/raw/`；
- 清洗结果保存在 `data/processed/`；
- artifact 保存在 `artifacts/`；
- 训练和评估不依赖在线 streaming；
- 本地数据、训练数据和模型 artifact 不提交到 Git。

清洗阶段会执行：

- Unicode 和换行规范化；
- 空 prompt、过短 prompt、过长 prompt 和异常控制字符过滤；
- 明显密钥、Bearer token、邮箱和电话号码的防御性替换；
- 规范化后的 prompt 去重；
- 语言过滤；
- source/session provenance 保留。

对于多轮数据，默认只提取第一条有效 user prompt，并丢弃 assistant 回复、模型名、IP、header 等无关字段。这样可以减少隐私风险和数据泄漏，也避免模型学习“某个模型如何回答”而不是“用户想做什么”。

## 11. 为什么要做 Group-aware Split

普通随机行切分可能把同一个对话、同一个 session 或同一个任务的近似重复样本同时放进训练集和 validation。这样得到的高分不代表模型真的泛化，只代表模型看过相同表达。

因此只要数据提供 `split_group`，切分时会把整个 group 放在同一侧：

```text
同一 conversation/session/task group
  → 全部 train 或全部 validation
```

没有 `split_group` 时，loader 会把每条记录当作独立 group；这在效果上等同于 record-level deterministic split，但仍然保持同一套切分逻辑。

## 12. 推理阶段的 TaskAST 输入与 evidence 输出

TensorFlow `predict` 先从原始 prompt 提取 TaskAST，然后把同一份 AST 编码为结构化输入，与字符序列和 n-gram 分支一起送入模型。模型输出神经 heads 后，contract 返回同一份 TaskAST 的 observable evidence：

- 语言；
- intent hint；
- domain hint；
- 代码块；
- 文件路径；
- Shell 命令；
- 数学符号；
- 文件和工具需求；
- 高风险信号；
- DNS、Kubernetes、数据库、部署、测试等 operations。

这些 evidence 不会在推理阶段被再次混入概率。它们的作用是让 Router 能审计模型看到了哪些高置信结构信号；不同 head 的最终概率由训练时学到的多任务 heads 产生。effort 仍由 effort bucket/regression heads 和 metadata supervision basis 共同解释。

历史 prototype 的 head-specific evidence policy 如下，仅用于 `TaskTagger` 或后续实验：

| Head | 参与融合的 evidence |
| --- | --- |
| workload | language、operation、domain、constraint |
| intent | intent hint |
| complexity | language、operation、constraint、intent hint |
| effort | effort bucket head、complexity posterior、observed effort metadata |

当前主模型不再执行上述 post-hoc 修改；这也避免了同一份 TaskAST 在训练输入和推理后处理阶段被重复计权。

## 13. 概率和不确定性

### 13.1 Probability

`probability` 是对应 head 的 posterior 分量，通常是 softmax/ordinal distribution 中选中 label 的概率。旧 artifact 没有独立校准；新训练 artifact 的 distribution 会经过 validation temperature scaling。

旧 artifact 没有 `metadata.calibration` 时，仍然只能把它当作未校准的模型分数。新训练 artifact 在 validation split 上拟合 temperature scaling，并在 metadata 记录每个 head 的 temperature、NLL 和 ECE；这一步不改变 argmax label，只调整 distribution、entropy、margin 和 probability 的可信度。

因此：

```text
probability ≠ 真实准确率
probability ≠ 模型选择成功率
```

### 13.2 Entropy

对一个 head 的概率分布 `p`，使用归一化 entropy：

```text
H(p) = -Σ p_i log(p_i) / log(K)
```

其中 `K` 是该 head 的类别数量。

- entropy 接近 0：分布集中，候选较明确；
- entropy 接近 1：分布分散，候选之间难以区分。

### 13.3 Margin

```text
margin = p_top1 - p_top2
```

margin 越低，说明两个或多个候选接近。Router 可以把低 margin 的请求交给更强的 fallback 模型，或者保留多个候选后再结合成本策略决策。

### 13.4 Overall confidence

当前整体 confidence 是一个路由分数，不是概率事件：

```text
overall =
    0.30 × workload_confidence
  + 0.30 × intent_confidence
  + 0.25 × complexity_confidence
  + 0.15 × effort_confidence
```

effort 的权重较低，是因为当前真实 `active_minutes` 仍然稀缺。

## 14. 澄清与 Unknown 策略

以下情况可以触发 `needs_clarification`：

- prompt 过短；
- 只有“帮我弄一下”“看看这个”之类模糊动作；
- clarification head 概率达到阈值；
- 缺少任务目标、交付物或操作范围。

### 14.1 Abstention（拒识）

`abstention` 是一个独立的 uncertainty policy，不是第五个分类标签。新训练 artifact 的默认配置是：

```json
{
  "enabled": true,
  "min_probability": 0.45,
  "min_margin": 0.05,
  "max_entropy": 0.98
}
```

它检查 workload、intent 和 complexity 三个 head：任一 head 的 top-1 probability 低于下限、top-1/top-2 margin 过小、entropy 过高，便返回：

```json
{
  "enabled": true,
  "abstain": true,
  "reasons": ["intent:low_margin"],
  "thresholds": {
    "min_probability": 0.45,
    "min_margin": 0.05,
    "max_entropy": 0.98
  }
}
```

拒识只会把 `needs_clarification` 置为 `true` 并暴露原因，不会把原来的预测 label 改成 `unknown`，也不会修改 TensorFlow head 的 probability。Router 可以据此选择低成本 fallback、请求用户补充信息，或在高风险场景停止自动路由。旧 artifact 没有 `abstention` metadata 时，输出 `enabled=false`，保持兼容。

如果请求需要澄清且没有结构化证据，模型会把 workload 和 intent 降级为 `unknown`。这是为了避免 Router 把不明确的请求误送到高成本或高风险路径。

`unknown` 不应该被下游当成新的业务 workload。它表示：

```text
当前证据不足，需要 fallback、澄清或保守路由
```

## 15. 训练和评估口径

训练入口是：

```bash
python -m trendmodel.train prepare ...
python -m trendmodel.train train ...
python -m trendmodel.train evaluate ...
```

### Raw evaluation

Raw evaluation 评估 TensorFlow heads 的输出；这些 heads 本身已经接收 TaskAST 结构化输入。它用于回答：

```text
保存下来的 TensorFlow 模型本身学到了什么？
```

### Input ablation

为了避免把“模型学到的语义”和“规则显式提供的结构事实”混成一个数字，评估 CLI
支持 `--ablation all`。三条路径定义如下：

| 路径 | TensorFlow 文本输入 | TaskAST 输入 | 目的 |
| --- | --- | --- | --- |
| `text_only` | 保留 | 置零 | 测量没有结构特征时的文本能力 |
| `text_ast` | 保留 | 保留 | 当前主模型的真实输入 |
| `evidence_only` | 不使用 | 只使用 | 测量确定性规则 baseline，不代表生产模型 |

这里的 `text_only` 是同一个已训练 artifact 的输入消融，不是重新训练的纯文本模型，
因此它适合做贡献诊断，不能替代独立 challenger 的架构比较。

### Inference evaluation

Inference evaluation 调用生产 `predict` contract。它返回与 raw 相同的 head 概率，并附带同一份 TaskAST evidence，用于回答：

```text
下游 Router 实际会看到什么？
```

旧命令 `--mode fused` 仍作为兼容别名，但实际等同于 inference，不再代表概率融合。如果公共数据的标签本身来自同一个 TaskTagger teacher，那么这些标签仍不能作为独立质量证明，因为 teacher 可能参与了标签生成。

### 当前指标

分类 head：

- accuracy；
- macro-F1；
- weighted accuracy；
- actual/predicted distribution；
- 按 source 分组；
- 按 label quality 分组。

Effort：

- bucket accuracy：所有有 bucket 标签的样本；
- observed MAE：只有真实 active minutes 的样本；
- 不把 bucket midpoint 当成真实分钟误差。

最终发布前必须使用独立的 `data/gold_validation.jsonl`，不能只看公共数据的 pseudo-label holdout。

## 16. 硬件和部署选择

当前模型的目标是同一套 Keras artifact 跨平台运行：

| 环境 | Runtime | 作用 |
| --- | --- | --- |
| Apple Silicon M4 | TensorFlow + `tensorflow-metal` | 本地开发、低延迟推理、轻量训练 |
| NVIDIA RTX 4070 | TensorFlow `and-cuda` extra | CUDA 训练、数据混合实验和速度测试 |
| CPU | TensorFlow CPU | CI、兼容性 smoke、无 GPU fallback |

训练产物包含：

```text
artifacts/trendmodel-text-v1/
├── model.keras
└── metadata.json
```

`metadata.json` 会记录：

- 模型版本；
- label maps；
- structured feature schema；
- TensorFlow 版本；
- physical devices；
- GPU devices；
- accelerator 类型；
- source 和 label quality 分布；
- effort supervision 数量；
- validation metrics；
- training warning。

`--require-gpu` 用于防止 GPU 训练静默退回 CPU。M4 和 4070 的浮点结果可能存在小幅差异，但同一 commit、同一数据、同一 seed 下，输出 schema 和核心指标应该可比较。

## 17. 当前限制

当前版本仍有几个明确限制：

1. TensorFlow 主模型目前没有独立的 domain head；domain 主要来自 TaskAST evidence 和后续的统一 IR 映射。
2. `predict_many` 当前是逐条推理，高吞吐场景还需要真正的 batched inference。
3. 新训练 artifact 的 workload、intent、complexity 和 effort distribution 会基于 validation split 做 temperature scaling；如果 validation 标签质量不足，校准结果仍不能替代独立 gold calibration。旧 artifact 的 probability、entropy 和 margin 仍未校准。
4. 没有真实 active minutes 时，effort 只能作为 bucket 先验，不能当作工时承诺。
5. 公共数据的弱标签可能继承 teacher 偏差，必须和人工 gold 分开报告。
6. 当前 unknown/OOD 主要依赖澄清规则和不确定性信号，还不是专门训练的开放集识别 head。
7. 代码、Shell、文件和工具信号覆盖了高频场景，但不代表所有领域的结构化证据。

## 18. 后续演进路线

推荐按下面的顺序演进：

### 阶段一：运行时闭环

- 在 RTX 4070 上完成 CUDA device check；
- 用 curated hard-negative 数据完成 2 epoch smoke；
- 验证 train、save、load、raw/inference evaluate、predict 全链路；
- 对比 M4 Metal 和 CUDA 的速度及 metadata。

### 阶段二：数据和标签质量

- 收集 300–1,000 条内部 gold；
- 每个困难边界至少准备一组正负样本；
- 收集 100–200 条真实 active minutes；
- 把 production 真实纠正反馈回流为 hard-negative；
- 防止同一任务或同一用户 session 跨 train/validation 泄漏。

### 阶段三：概率和开放集能力

- 在独立 gold 集上复核 validation temperature scaling，必要时比较 isotonic calibration；
- 计算 ECE、Brier score 和按 confidence 分桶的准确率；
- 增加 unknown/OOD validation；
- 为低 confidence 和高风险任务设计保守 fallback policy。

### 阶段四：推理和模型能力

- 实现真正的 batched inference；
- 根据 gold 数据判断是否需要更强的中文/多语言 sentence embedding；
- 在不破坏结构化 evidence 的前提下尝试轻量 MLP、Linear head 或对比学习；
- 如果下游确实需要，再增加独立 domain/artifact head。

公开数据规模只有在上述数据质量和评估闭环完成后才值得继续扩大。对于 Router，区分少量高价值困难边界，通常比继续增加大量同质化聊天 prompt 更重要。

## 19. 数据集、数据缺口与监督来源

### 19.1 数据选择原则

TrendModel 的训练数据不是“越多越好”，而是要覆盖 Router 真正需要区分的边界。一个数据源只有在至少满足以下一项时才值得接入：

1. 提供新的语言或 prompt 表达分布；
2. 提供 `create / modify / debug / review / execute / explain` 等困难 intent 边界；
3. 提供代码、文件、命令、工具调用或多步轨迹等结构化证据；
4. 提供可以独立验证的标签，而不是只提供模型回答偏好。

所有 Hugging Face 数据只在本机下载和读取，Hugging Face 不参与训练服务、不接收本地数据。详细下载命令、adapter 字段和许可证边界见 [数据集策略](DATASETS.md)。

### 19.2 当前数据集分层

下表区分“已有 adapter”“推荐训练角色”和“不能从中推导出的标签”。数据集卡或论文是数据来源引用，不代表该数据集的标签可以直接当作 TrendModel gold。

| 数据源 | 当前状态 | 可提供的监督/证据 | 主要缺口 |
| --- | --- | --- | --- |
| WildChat-4.8M（Allen Institute for AI, n.d.） | 已有 adapter；用于 prompt 分布补充 | 多轮自然表达、中文/英文混用、任务形态 | 没有可信 workload/intent/active minutes；只取第一条 user prompt |
| Arena Human Preference 140K（LMSYS Organization, n.d.） | 已有 adapter；有限使用 | `is_code`、category、偏好字段提供代码/非代码弱信号 | 偏好 winner 不是任务类型，也不是 effort；`is_code=false` 不能直接当 text gold |
| xLAM Function Calling 60K（Salesforce AI Research, n.d.） | 已有 adapter；训练辅助/主料候选 | `execute`、工具调用、工具数量和多步调用 proxy | 函数调用存在不等于用户明确授权执行；没有人工工时 |
| SWE-bench（Jimenez et al., 2024） | 已有 adapter；代码修改主料 | issue、patch、test patch、repository context，支持 `code + modify` | patch 行数不是人工 active minutes；许可证需产品化前单独确认 |
| SWE-agent trajectories（Nebius, n.d.） | 已有 adapter；可选辅助 | 多步 agent 轨迹、工具调用和修改过程 | agent trajectory 是过程数据，不等于真实人类 effort |
| Code-Feedback（M-A-P, n.d.） | 已有 adapter；代码表达辅助 | 代码创建类 user prompt，弱映射 `code + create` | label 主要来自数据形态，不是独立人工 Router 标注 |
| MASSIVE（Amazon Science, n.d.）、MTOP（Amazon, n.d.） | 已有 adapter；短指令和 intent 辅助/评估 | 多语言短句、服务查询、`execute/explain` 边界 | 语音助手领域偏差，不能代表开放域 workload |
| COIG（BAAI, n.d.）、OASST1（OpenAssistant, n.d.）、Helpful Instructions（Hugging Face H4, n.d.） | 已有 adapter；表达覆盖辅助 | 中文、英文和多语言自然表达 | 通用行没有可靠 workload/intent/effort gold |
| Dolly（Databricks, n.d.） | adapter 可用；默认不作为主料 | category 提供粗粒度 intent 辅助 | CC BY-SA 约束和领域/指令偏差，需要单独合规确认 |

当前推荐的第一轮混合数据是 WildChat、Arena、xLAM、SWE-bench、MASSIVE、MTOP、Code-Feedback、COIG 和 Helpful Instructions；SWE-agent trajectories、OASST1 和 Dolly 的 adapter 已保留，但不代表每次训练都应加入。LMSYS-Chat-1M 的兼容读取代码仍在仓库中，但项目决策是“不下载、不清洗、不训练”。

### 19.3 当前缺少的数据

真正限制模型上限的不是公开 prompt 数量，而是下面几类尚未充分标注的数据：

| 缺口 | 为什么影响 Router | 目标数据 |
| --- | --- | --- |
| 真实内部任务 gold | 公共数据的 label 多数是 weak/pseudo，无法代表实际路由边界 | 真实脱敏 prompt + 人工 workload/intent/complexity |
| 真实 `active_minutes` | 没有它就无法把 effort 输出解释成工作量估计 | 任务开始/结束或主动操作时长记录 |
| 困难负样本 | `explain`/`create`、`modify`/`debug`、`execute`/`explain` 仅靠大规模同质 prompt 很难拉开 | 成对或成组的最小语义差异样本 |
| 中文和中英混合的工程请求 | 当前公开 intent 数据多偏英语或语音助手 | 中文代码、Shell、Kubernetes、网络、文档和混合表达 |
| 上下文与权限样本 | 同一个 prompt 在有附件、Project 或工具权限时，路由含义不同 | prompt + 文件摘要/项目状态/权限上下文；不保存敏感原文 |
| unknown/OOD 与高风险样本 | Router 不能把模糊请求或生产删除请求误送到自动执行路径 | 模糊 prompt、越权请求、生产变更、破坏性命令和明确拒绝样本 |
| 路由结果反馈 | 目前模型只学习任务标签，不知道某次低成本路由是否真的达到质量要求 | 选择的模型、成本、延迟、用户纠正、质量回退和最终成功信号 |

因此，下一轮数据工作优先级应是内部 gold、hard negatives、active minutes 和安全/unknown 集，而不是继续堆叠未经标注的聊天数据。`BANKING77`（PolyAI, n.d.）、`CLINC150`（DeepPavlov, n.d.）和 `GSM8K`（OpenAI, n.d.）更适合做独立 intent/OOD/高推理难例评估，不应直接改变通用 workload 分布。

## 20. 标注协议与数据质量控制

### 20.1 标注对象和字段

每条内部 gold 至少包含以下字段：

| 字段 | 取值 | 标注原则 |
| --- | --- | --- |
| `original_prompt` | 脱敏后的原始 prompt | 保留用户意图，不保留 token、邮箱、IP、私密路径等敏感值 |
| `workload_type` | `text / code / analysis / operations / planning / mixed / other` | 按主要交付物和操作域判断；跨域任务可标 `mixed` |
| `intent` | `create / modify / debug / explain / execute / analyze / review / summarize / translate / plan / extract / classify / unknown` | 按用户要求的动作和交付物判断，不按关键词单独决定 |
| `complexity` | `low / medium / high` | 考虑步骤数、上下文、系统数量、验证闭环和风险，不等于文字长度 |
| `effort_bucket` | `XS / S / M / L / XL` | 只表示粗粒度工作量等级；不能代替真实分钟 |
| `active_minutes` | 正数或空 | 只填写实际主动工作时间；不包含排队、CI、审批和等待外部团队 |
| `needs_clarification` | `true / false` | 目标、交付物、范围或权限不足时为 true |
| `split_group` | task/session/conversation id | 同一任务的变体必须留在同一 train 或 validation 侧 |
| `label_source`、`label_confidence` | provenance metadata | 区分人工 gold、seed、weak、pseudo 和 observed |

`domain`、`artifact`、`language`、`operations`、`risk` 和 `capability_demand` 是 Task AST/Task IR 的重要字段；当前 TensorFlow 训练主 head 尚未全部独立建模，因此它们可以作为结构化 evidence、辅助标注和未来扩展字段保存，不应伪装成当前已训练的独立 head。

### 20.2 推荐的人工标注流程

```text
原始任务日志
  ↓
脱敏、截断、去重、生成 split_group
  ↓
两名标注者独立标注 workload / intent / complexity / effort bucket
  ↓
有实际记录时补 active_minutes；没有则留空
  ↓
对冲突样本进行 adjudication，记录最终版本和原因
  ↓
自动生成困难负样本对、标签分布和 agreement 报告
  ↓
按 split_group 固定划分 train / validation / gold test
```

具体执行规则：

1. 标注者先判断动作，再判断 workload，避免看到“代码”“实现”等词就直接标成 `create`；
2. 对 `explain` vs `create`、`create` vs `modify`、`modify` vs `debug`、`explain` vs `execute` 强制建立 hard-negative 对；
3. 若 prompt 本身没有目标或交付物，优先标 `unknown`/`needs_clarification=true`，不要靠猜测补齐；
4. `active_minutes` 只能来自真实任务记录，不能由 prompt 长度、patch 行数、工具调用次数或 bucket midpoint 推算；
5. 每轮标注后统计每个字段的 inter-annotator agreement。对 intent/workload 可使用 Cohen's kappa 或 macro-F1，对多值 domain/operation 可使用 per-label F1/Jaccard；agreement 低时先修订标签定义，再扩大数据量；
6. gold validation/test 永远不能参与 pseudo-label 生成、prototype 调整、阈值选择或 gold replay 超参数选择。

第一批建议收集 300–1,000 条内部 gold，其中 100–200 条同时具有可靠 `active_minutes`。分布不应只按自然流量采样，还要对困难边界、中文/英文、代码/运维、低风险/高风险和 unknown 进行定额采样。

### 20.3 标签 provenance 和训练权重

每条记录保留 `source`、`label_source`、字段级 confidence 和 `split_group`。训练时：

```text
effective_weight(field, record)
  = field_label_confidence
  × class_balance_weight(field, label)
  × gold_replay_multiplier(if trusted gold/seed/observed)
```

这是训练策略的概念表达；实际代码对 `effort_bucket`/`effort_ordinal` 和 `effort_log_minutes` 使用独立的监督 mask，bucket-only 记录不会进入分钟回归，不能简单套用普通分类 head 的 confidence 权重。weak/pseudo 标签只作为覆盖和表示学习信号；人工 gold 和真实 observed effort 才是最终质量判断的依据。当前默认 `gold_replay_multiplier=8`，只作用于训练 sample weight，不复制数据行，也不作用于 validation。

## 21. 训练流程与优化目标

### 21.1 数据准备

入口是 `python -m trendmodel.train prepare`。它执行：

1. 读取本地 JSONL 或 `data/raw/huggingface/` 下的 Parquet/JSON 文件；
2. 通过 source adapter 只提取第一条有效 user prompt，并保留来源/组别 metadata；
3. 进行 Unicode、换行、长度、控制字符、敏感信息和重复记录清洗；
4. 可选地调用 `TaskTagger` 生成 workload/intent/complexity 的 pseudo labels；
5. 明确禁止 teacher 生成 `active_minutes` 或任何 effort 真值；
6. 写出本地 `data/processed/*.jsonl` 和清洗/标签报告。

### 21.2 TensorFlow 训练

入口是 `python -m trendmodel.train train`，内部调用 `tf_train.train_model()`。训练逻辑为：

```mermaid
flowchart LR
    A["local JSONL"] --> B["TaskRecord.from_mapping"]
    B --> C["group-aware split"]
    C --> D["label maps + class weights"]
    D --> E["structured_feature_vector"]
    E --> F["tf.data with masked sample weights"]
    F --> G["build_text_model"]
    G --> H["Adam + multi-task fit"]
    H --> I["save model.keras"]
    I --> J["reload + raw round-trip evaluation"]
    J --> K["metadata.json"]
```

当前默认训练参数如下：

| 参数 | 当前值 | 作用 |
| --- | ---: | --- |
| optimizer | Adam | 多任务共享表示的默认优化器 |
| learning rate | `1e-3` | 初始学习率 |
| batch size | `16` | 本地训练默认 batch |
| epochs | `20` | 由 EarlyStopping 提前结束 |
| validation ratio | `0.2` | group-aware validation 比例 |
| random seed | `42` | 可复现实验初始化和切分 |
| EarlyStopping | `patience=4`, restore best | 监控 `val_loss` |
| ReduceLROnPlateau | factor `0.5`, patience `2`, min `1e-5` | 验证集停滞时降学习率 |

### 21.3 多任务目标函数

对于有标签的样本，模型优化：

```text
L_total =
    1.0 × L_workload
    + 1.0 × L_intent
    + 0.8 × L_complexity
    + 0.8 × L_effort_bucket
    + 0.8 × L_effort_ordinal
    + 1.0 × L_effort_log_minutes
    + 0.7 × L_needs_clarification
```

其中 workload、intent、complexity 和 `effort_bucket` 使用 sparse categorical cross-entropy，`effort_ordinal` 和 `needs_clarification` 使用 binary cross-entropy，`effort_log_minutes` 使用 Huber loss。缺失的标签对应 sample weight 为 0，因此公共 prompt 可以训练 workload/intent 等字段，而不会因为没有 `active_minutes` 被强行填充成伪真值（实现见 [`trendmodel/tf_data.py`](../trendmodel/tf_data.py) 和 [`trendmodel/tf_train.py`](../trendmodel/tf_train.py)）。

`effort_bucket`、`effort_ordinal` 和 `effort_log_minutes` 是三个不同监督目标：

```text
bucket-only record:
    trains effort_bucket + ordinal effort only

record with observed active_minutes:
    trains effort_bucket + ordinal effort + log1p(active_minutes) regression
```

训练结束后保存 `model.keras` 和 `metadata.json`。metadata 记录 labels、feature schema、runtime、数据来源分布、supervision counts、validation metrics、warnings 和 artifact round-trip metrics。

### 21.4 评估协议

每次训练至少报告两条路径：

- `raw`：运行保存后的 TensorFlow heads，回答“神经模型本身学到了什么”；
- `inference`：返回生产 JSON 和 TaskAST evidence，回答“下游 Router 实际看到什么”；概率与 raw 保持一致。

两者不是两套分类器，也不应期待 inference 比 raw 自动更准；它们主要用于检查输出 contract 和 evidence 是否完整。若训练标签本身由 TaskTagger 生成，仍然存在 teacher overlap，不能作为独立泛化证明。最终发布需要独立的 `data/gold_validation.jsonl`，报告 accuracy、macro-F1、按 source/label quality 分组的指标、effort bucket accuracy、observed MAE 和 unknown/OOD 指标。

## 22. 代码组成与运行时调用关系

### 22.1 核心模块

| 模块 | 责任 | 是否属于当前主链路 |
| --- | --- | --- |
| `trendmodel/train.py` | prepare/train/evaluate CLI 编排，连接本地数据、清洗和 TensorFlow 训练 | 是 |
| `trendmodel/public_data.py` | Hugging Face source adapter、字段映射、弱标签边界 | 是，只有准备数据时使用 |
| `trendmodel/local_data.py` | 读取已经下载的本地 Parquet/JSON/JSONL | 是 |
| `trendmodel/data_cleaning.py` | 规范化、脱敏、去重、拒绝原因和报告 | 是 |
| `trendmodel/embedding.py` | `HashEmbedder` 和可选 SentenceTransformer backend | TaskTagger/准备数据辅助 |
| `trendmodel/evidence.py` | 规则提取语言、intent/domain/artifact、operation、constraint、risk | 是，训练和推理共用 |
| `trendmodel/task_ast.py` | 正式 `TaskAST` dataclass、候选分数、风险和 evidence 的统一封装 | 是 |
| `trendmodel/task_ir.py` | prototype 路径的 Task IR dataclass 和 JSON 序列化 | TaskTagger 路径 |
| `trendmodel/prototypes.py` + `trendmodel/tagger.py` | 原型相似度、层级选择、结构化融合和解释性 Task IR | evidence teacher / 可选独立路径 |
| `trendmodel/fusion.py` | cosine、softmax、加权融合、乘法 Bayesian-style 实验和不确定性 | 是 |
| `trendmodel/tf_features.py` | 把 AST-like evidence 转成固定 56 维向量 | 是 |
| `trendmodel/tf_data.py` | TaskRecord、label map、split、类别权重、sample-weight mask、tf.data | 是 |
| `trendmodel/tf_model.py` | Keras 模型结构、loss、optimizer 和 head | 是 |
| `trendmodel/tf_train.py` | TensorFlow train/evaluate、保存、reload、metadata | 是 |
| `trendmodel/tf_inference.py` | artifact load、raw/inference predict、TaskAST evidence、effort 转换和 JSON contract | 是 |
| `trendmodel/api.py` | 本地 OpenAI-compatible HTTP 包装 | 可选 serving 层 |
| `trendmodel/model.py`、`nb.py`、`vectorizer.py` | 旧版 TF-IDF + 多头 Naive Bayes baseline | 兼容保留，不是当前主 artifact |

### 22.2 训练与推理时的代码边界

训练阶段，`tf_features.structured_feature_vector()` 调用 `evidence.extract_evidence()` 得到 `TaskAST`，再把其中的规则事实编码进 56 维结构化输入；`tf_model.build_text_model()` 再把文本分支和结构分支融合，训练六个 heads。推理阶段，`tf_inference.TensorFlowTaskModel` 提取一份 TaskAST，将它编码进模型输入，并把同一份 AST 的 evidence 放进 JSON；不会再调用 `TaskTagger` 修改概率。实现细节见 [`trendmodel/tf_features.py`](../trendmodel/tf_features.py)、[`trendmodel/tf_model.py`](../trendmodel/tf_model.py)、[`trendmodel/tf_inference.py`](../trendmodel/tf_inference.py) 和 [`trendmodel/tagger.py`](../trendmodel/tagger.py)。

这条边界有三个工程含义：

1. 规则 extractor 不是一个绕过模型的硬编码分类器；
2. TaskTagger 的 prototype probability 不是 TensorFlow head 的 probability；
3. Router 应保存完整的 `predictions`、`candidates`、`evidence`、`fusion` 和 `confidence`，而不是只保存最终 label。

### 22.3 推理输出和 Router 使用方式

对外 JSON 的机器可读约束位于 [`task-analysis.schema.json`](task-analysis.schema.json)。推荐 Router 按以下顺序使用：

```text
1. 检查 needs_clarification 和 risk/evidence
2. 检查 workload、intent、complexity 的 label + probability + margin
3. 读取 effort.bucket、range_minutes 和 basis
4. 将 Task IR 与候选模型能力、价格、上下文窗口和延迟做匹配
5. 低 confidence、低 margin 或高风险时走 fallback/确认
6. 保存原始 prompt hash、Task IR、最终路由和用户反馈用于回放
```

TrendModel 不应根据自己的输出直接执行 Shell、修改文件或授予权限。

## 23. 参考文献与代码引用

### 23.1 算法与评估文献


Aho, A. V., Lam, M. S., Sethi, R., & Ullman, J. D. (2006). *Compilers: Principles, techniques, and tools* (2nd ed.). Pearson.

Bayes, T. (1763). An essay towards solving a problem in the doctrine of chances. *Philosophical Transactions of the Royal Society of London, 53*, 370–418. <https://doi.org/10.1098/rstl.1763.0053>

Caruana, R. (1997). Multitask learning. *Machine Learning, 28*, 41–75. <https://doi.org/10.1023/A:1007379606734>

Cho, K., van Merriënboer, B., Gulcehre, C., Bahdanau, D., Bougares, F., Schwenk, H., & Bengio, Y. (2014). Learning phrase representations using RNN encoder–decoder for statistical machine translation. *arXiv preprint arXiv:1406.1078*. <https://arxiv.org/abs/1406.1078>

Cooper, K. D., & Torczon, L. (2012). *Engineering a compiler* (2nd ed.). Morgan Kaufmann.

Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). On calibration of modern neural networks. *Proceedings of the 34th International Conference on Machine Learning*. <https://proceedings.mlr.press/v70/guo17a.html>

Lattner, C., & Adve, V. S. (2004). LLVM: A compilation framework for lifelong program analysis & transformation. *Proceedings of the International Symposium on Code Generation and Optimization*. <https://doi.org/10.1109/CGO.2004.1281665>

Reimers, N., & Gurevych, I. (2019). Sentence-BERT: Sentence embeddings using Siamese BERT-networks. *Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing*. <https://aclanthology.org/D19-1410/>

Salton, G., & Buckley, C. (1988). Term-weighting approaches in automatic text retrieval. *Information Processing & Management, 24*(5), 513–523. <https://doi.org/10.1016/0306-4573(88)90021-0>

Schuster, M., & Paliwal, K. K. (1997). Bidirectional recurrent neural networks. *IEEE Transactions on Signal Processing, 45*(11), 2673–2681. <https://doi.org/10.1109/78.650093>

### 23.2 数据集与数据卡


Allen Institute for AI. (n.d.). *WildChat-4.8M* [Data set]. Hugging Face. Retrieved August 8, 2026, from <https://huggingface.co/datasets/allenai/WildChat-4.8M>

Amazon. (n.d.). *MTOP* [Data set]. Hugging Face. Retrieved August 8, 2026, from <https://huggingface.co/datasets/tasksource/mtop>

Amazon Science. (n.d.). *MASSIVE* [Data set]. Hugging Face. Retrieved August 8, 2026, from <https://huggingface.co/datasets/AmazonScience/massive>

BAAI. (n.d.). *COIG* [Data set]. Hugging Face. Retrieved August 8, 2026, from <https://huggingface.co/datasets/BAAI/COIG>

Databricks. (n.d.). *databricks-dolly-15k* [Data set]. Hugging Face. Retrieved August 8, 2026, from <https://huggingface.co/datasets/databricks/databricks-dolly-15k>

DeepPavlov. (n.d.). *CLINC150* [Data set]. Hugging Face. Retrieved August 8, 2026, from <https://huggingface.co/datasets/DeepPavlov/clinc150>

Hugging Face H4. (n.d.). *Helpful instructions* [Data set]. Hugging Face. Retrieved August 8, 2026, from <https://huggingface.co/datasets/HuggingFaceH4/helpful-instructions>

Jimenez, C. E., Yang, C., Wettig, A., Yao, K., Pei, R., Press, O., & Narasimhan, K. (2024). SWE-bench: Can language models resolve real-world GitHub issues? *Proceedings of the International Conference on Learning Representations*. <https://arxiv.org/abs/2310.06770>

LMSYS Organization. (n.d.). *Arena human preference 140K* [Data set]. Hugging Face. Retrieved August 8, 2026, from <https://huggingface.co/datasets/lmarena-ai/arena-human-preference-140k>

M-A-P. (n.d.). *Code-Feedback* [Data set]. Hugging Face. Retrieved August 8, 2026, from <https://huggingface.co/datasets/m-a-p/Code-Feedback>

Nebius. (n.d.). *SWE-agent trajectories* [Data set]. Hugging Face. Retrieved August 8, 2026, from <https://huggingface.co/datasets/nebius/SWE-agent-trajectories>

OpenAI. (n.d.). *GSM8K* [Data set]. Hugging Face. Retrieved August 8, 2026, from <https://huggingface.co/datasets/openai/gsm8k>

OpenAssistant. (n.d.). *OASST1* [Data set]. Hugging Face. Retrieved August 8, 2026, from <https://huggingface.co/datasets/OpenAssistant/oasst1>

PolyAI. (n.d.). *BANKING77* [Data set]. Hugging Face. Retrieved August 8, 2026, from <https://huggingface.co/datasets/PolyAI/banking77>

Salesforce AI Research. (n.d.). *xLAM function calling 60K* [Data set]. Hugging Face. Retrieved August 8, 2026, from <https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k>

### 23.3 Implementation pointers（实现指针，非学术引用）

以下链接用于说明本文中的“当前实现”如何落在仓库代码里。它们不是外部学术来源，不使用 APA 文献编号，也不用于证明算法在学术意义上的有效性。

- [`trendmodel/task_ast.py`](../trendmodel/task_ast.py)：正式 `TaskAST` dataclass、候选访问、序列化和兼容性边界。
- [`trendmodel/evidence.py`](../trendmodel/evidence.py)：结构化 evidence extractor、语言/动作/领域/操作/风险规则。
- [`trendmodel/task_ir.py`](../trendmodel/task_ir.py)：`TaskEvidence`、`TaskConstraints` 和 prototype 路径 `TaskIR` 数据结构。
- [`trendmodel/tagger.py`](../trendmodel/tagger.py)：句向量/原型相似度、层级选择、context prior 和 Task IR 组装。
- [`trendmodel/fusion.py`](../trendmodel/fusion.py)：加权概率混合、乘法 Bayesian-style 实验、entropy/margin/confidence。
- [`trendmodel/tf_features.py`](../trendmodel/tf_features.py)：56 维结构化特征向量。
- [`trendmodel/tf_model.py`](../trendmodel/tf_model.py)：字符向量化、BiGRU、n-gram TF-IDF、结构分支和多任务 heads。
- [`trendmodel/tf_data.py`](../trendmodel/tf_data.py)：记录解析、label map、group split、类别平衡、replay 和 sample-weight mask。
- [`trendmodel/tf_inference.py`](../trendmodel/tf_inference.py)：raw/inference 推理、TaskAST evidence、`TaskIR.from_tensorflow()` 适配、effort basis、uncertainty diagnostics 和 JSON contract。
- [`trendmodel/tf_train.py`](../trendmodel/tf_train.py)：训练、评估、保存、reload round-trip 和 metadata。
- [`trendmodel/train.py`](../trendmodel/train.py)、[`trendmodel/public_data.py`](../trendmodel/public_data.py)、[`trendmodel/local_data.py`](../trendmodel/local_data.py)、[`trendmodel/data_cleaning.py`](../trendmodel/data_cleaning.py)：本地数据准备、adapter 和清洗边界。

### 23.4 APA 7 引用说明

正文使用 APA 7 的作者—年份形式：

- 单一作者：`（Bayes, 1763）`；
- 两位作者：`（Salton & Buckley, 1988）`；
- 三位及以上作者：`（Cho et al., 2014）`；
- 没有稳定出版年份的数据卡：`（Allen Institute for AI, n.d.）`。

数据集引用只说明数据来源和版本追溯时间，不代表该数据集提供了 TrendModel 所需的 gold workload、intent 或 effort 标签。公共数据的弱标签、数据卡和论文都不能替代内部人工标注。

### 23.5 相关文档

- [README](../README.md)：快速运行、训练和 CUDA/Metal 环境；
- [数据集策略](DATASETS.md)：数据源、许可证、adapter 和本地下载；
- [下游集成手册](INTEGRATION.md)：Python、OpenAI-compatible API 和 JSON contract；
- [Task Analysis Schema](task-analysis.schema.json)：机器可读输出 schema；
- [Gold validation template](../data/gold_validation.template.jsonl)：内部人工标注模板。
