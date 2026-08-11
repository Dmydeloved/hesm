# HESM

[English](README.md)

HESM 是一个按照 **Experience（长期经历）→ Segment（阶段片段）→ QA（原始证据）**
组织的层级长期记忆系统。系统将对话轮次转换为可追踪的结构化记忆，在不同层级维护摘要，
并为问题检索规模受控的层级上下文。本仓库同时提供完整的 LoCoMo 实验流程，包括基线对比、
消融实验、断点续跑、四阶段日志和结果表生成。

## 系统架构

```text
原始对话轮次
      │
      ▼
主题 / 实体 / 意图抽取
      │
      ▼
Experience ── 长期主题与状态
      └── Segment ── 连贯的意图或事件阶段
            └── QA ── 可追踪的原始证据轮次
      │
      ▼
结构化召回 + 向量召回
      │
      ▼
可选的 LLM 层级重排序
      │
      ▼
受预算约束的 Experience + Segment + QA 上下文
```

核心实现位于 `memory/`：

- `manager.py`：创建并维护 Experience、Segment、QA 三层记忆；
- `storage.py`：SQLite 持久化和关系检索；
- `vector_store.py`：兼容 Chroma 的向量存储及本地后备实现；
- `extractor.py`：抽取主题、实体、意图和置信度；
- `summarizer.py`：模板摘要或 LLM 摘要；
- `retriever.py`：层级召回、回退路由和重排序。

## LoCoMo 基准实验

一键实验默认比较四种方法：

| 方法 | 说明 |
|---|---|
| `vector_rag` | 对原始对话轮次进行扁平向量检索 |
| `mem0` | Mem0 记忆与检索适配器 |
| `amem` | A-MEM 风格的原子记忆与关系链接 |
| `hesm` | 完整的 Experience → Segment → QA 层级记忆 |

`full_context` 仍可通过 `run_main` 单独运行，但已从 `run_all` 一键流程中删除，
一键流程只对比实际记忆系统。

默认数据路径为 `data/locomo10.json`。LoCoMo 类别 1—4 是正式评价集合；类别 5
为对抗问题，除非定义专门的评价规则，否则不应和普通问答混合计算总分。

### 评价指标

- 答案词元级 Precision、Recall 和 F1；
- 二元 LLM-as-a-Judge 正确率；
- K=1、3、5 时的证据 Recall、Precision、F1 和 Accuracy；
- 检索上下文 token 数和压缩比；
- 检索、LLM 作答和 Judge 的逐问题耗时。

## 环境要求

- Python 3.10 或更高版本；
- OpenAI 兼容的聊天补全接口；
- 与配置服务兼容的 Embedding 接口；
- SQLite；
- 对应基线所需的 ChromaDB 和 Mem0。

安装运行依赖：

```bash
pip install openai pyyaml tiktoken chromadb mem0ai
```

在 `configs/config.yaml` 中配置模型角色。请勿把真实 API Key 提交到版本库。
主要配置段包括：

- `topic_extraction`：主题抽取；
- `answer_generation`：最终答案生成；
- `embedding`：向量模型；
- `retrieval`：检索和重排序；
- `summarization`：层级摘要；
- `evaluation`：Judge 模型。

实验路径、启用方法、Top-K 和消融变体位于
`experiments/locomo/config/experiment.yaml`。

### Mem0、A-MEM 与 HESM 使用的模型

当前配置如下：

| 组件 | 配置段 | 当前模型 |
|---|---|---|
| Mem0 内部记忆抽取/更新 LLM | `memory_methods.mem0.llm` | `gpt-5.4-nano-high` |
| Mem0 Embedding | `memory_methods.mem0.embedding` | `text-embedding-v4` |
| A-MEM 关键词与上下文生成 LLM | `memory_methods.amem.llm` | `gpt-5.4-nano-high` |
| A-MEM Embedding | `memory_methods.amem.embedding` | `text-embedding-v4` |
| HESM 主题抽取 | `memory_methods.hesm.topic_extraction` | `gpt-5.4-nano-high` |
| HESM 层级摘要 | `memory_methods.hesm.summarization` | `gpt-5.4-nano-high` |
| HESM 检索重排 | `memory_methods.hesm.retrieval` | `gpt-5.4-nano-high` |
| HESM Embedding | `memory_methods.hesm.embedding` | `text-embedding-v4` |
| 各方法共用的最终答案模型 | `answer_generation` | `gpt-5.4-mini` |
| 各方法共用的 Judge 模型 | `evaluation` | `gpt-5.4-mini` |

三个适配器分别读取自己的 `memory_methods` 子配置。YAML Profile 统一维护公共连接、
模型、重试和 Embedding 参数，同时保留代码原有的角色配置路径。修改 Profile 可同步
更新所有引用它的角色；如需单独调整某种方法，也可将对应别名替换为独立配置。

## 运行实验

以下命令均在仓库根目录执行。

### 一键运行

```bash
python -m experiments.locomo.run_all
```

该命令依次运行：

1. Vector RAG、Mem0、A-MEM、HESM 主实验；
2. HESM 消融实验；
3. 缓存延迟实验。

常用参数：

```bash
# 只运行第一个对话，用于快速检查
python -m experiments.locomo.run_all --max-conversations 1

# 只运行指定主实验方法
python -m experiments.locomo.run_all --methods vector_rag hesm

# 跳过耗时部分
python -m experiments.locomo.run_all --skip-parts ablation cache

# 推荐配置：方法串行，每种方法内部并发执行 4 个 QA
python -m experiments.locomo.run_all --method-workers 1 --qa-workers 4
```

`method_workers` 控制同时运行的方法数，默认值为 1，避免不同方法竞争资源而干扰
延迟指标。`qa_workers` 控制每个对话完成记忆构建后并发处理的问题数，默认值为 4。
每个 QA 工作线程独占检索读取器、答案模型客户端和 Judge 客户端，checkpoint 仍由
父线程串行写入。如果更关心总完成时间而不是隔离后的单方法延迟，可使用
`--method-workers 4 --qa-workers 1`。不建议同时把两者设得很大，因为峰值请求并发量
近似为两者的乘积。

### 只运行主实验

```bash
python -m experiments.locomo.run_main
python -m experiments.locomo.run_main --methods mem0 hesm
python -m experiments.locomo.run_main --max-conversations 2
```

如需单独运行 Full Context 上界：

```bash
python -m experiments.locomo.run_main --methods full_context
```

### 消融实验

```bash
python -m experiments.locomo.run_ablation
```

完整消融集合如下：

| 变体 | 查询时主题抽取 | 使用层级 | LLM重排序 |
|---|---:|---|---:|
| `flat_memory` | 否 | 原始对话轮次 | 否 |
| `qa_only` | 否 | QA | 否 |
| `qa_segment` | 否 | QA + Segment | 否 |
| `full_hesm_no_reranker` | 是 | Experience + Segment + QA | 否 |
| `full_hesm` | 是 | Experience + Segment + QA | 是 |

运行部分变体：

```bash
python -m experiments.locomo.run_ablation \
  --variants qa_only qa_segment full_hesm_no_reranker full_hesm
```

QA 和层级变体会复用主实验生成的 HESM 存储。如果存储不存在，独立运行消融实验时
会自动先构建 HESM 记忆，然后继续执行所有变体。

### 缓存实验

```bash
python -m experiments.locomo.run_cache --num-samples 50
```

## 四阶段方法日志

每个主实验方法和消融变体都会生成一个独立日志文件：

```text
outputs/locomo/logs/<method>.log
```

日志为 UTF-8 JSON Lines 格式，每行是一个 JSON 对象，通过 `stage` 字段划分为四部分：

1. `BUILD`：构建数据或连接已有记忆存储；
2. `RETRIEVAL`：检索 ID、token 数、上下文预览和耗时；
3. `ANSWER`：LLM 生成答案和耗时；
4. `JUDGE`：标准答案、预测答案、判分和耗时。

每个阶段都会输出 `STARTED` 和 `SUCCESS`；失败时输出 `FAILED` 以及具体错误原因。
构建完成但存在部分轮次失败时，会输出 `PARTIAL` 和捕获到的失败原因。
如果前置阶段失败，后续阶段会输出 `SKIPPED` 及跳过原因。日志不会记录 API Key 或完整敏感配置。

示例：

```json
{
  "method": "hesm",
  "stage": "RETRIEVAL",
  "status": "SUCCESS",
  "conv_id": "conv-26",
  "query_index": 0,
  "retrieved_ids": ["D1:3"],
  "retrieved_tokens": 1082,
  "elapsed_ms": 18320.4
}
```

## 断点续跑和失败重试

每个方法、每个对话维护一个检查点：

```text
outputs/locomo/answers/<method>_<conv_id>.json
```

只有检索、LLM 作答和 Judge 三个查询阶段全部成功，query 才会标记为完成。再次运行时：

- 已成功 query 直接跳过；
- 检索失败、空答案和 Judge 失败的 query 自动重试；
- 失败记录和阶段错误原因保留在检查点及日志中；
- 旧检查点只有在答案非空且 `judge_score >= 0` 时才视为成功。
- Mem0 与 A-MEM 会分别在 `memory/mem0_<conv_id>/build_state.json` 和
  `memory/amem_<conv_id>/build_state.json` 保存已完成的源 `dia_id`；完整记忆会跳过
  全部构建调用，部分完成的记忆只补建缺失轮次。
- HESM 会核对 SQLite 中保存的 `dia_id`：存储完整时跳过重构，部分完成时仅补建
  缺失轮次。

因此，实验中断后可以直接重新执行原命令，不会重复支付已成功 query 的调用成本。

## 输出目录

```text
outputs/locomo/
├── answers/   # 按方法和对话保存的检查点、逐问题结果
├── logs/      # 每个方法或消融变体的四阶段日志
├── memory/    # 隔离的 SQLite 与向量存储
├── metrics/   # 按方法汇总的指标
└── tables/    # Markdown、CSV、JSON 结果表
```

主实验表为 `main_results.{md,csv,json}`，消融和缓存结果分别为
`ablation_results.*` 与 `cache_results.*`。

## 测试

断点恢复与日志测试只依赖 Python 标准库：

```bash
python -m unittest tests.locomo.test_runner_resume_logging -v
```

运行所有 unittest：

```bash
python -m unittest discover -s tests -v
```

## 可复现性注意事项

- 所有对比方法应使用相同的配置快照和模型版本；
- 不要比较 Judge 成功样本集合不同的部分运行；
- 标准类别 1—4 总分中不要混入类别 5；
- 正式实验应记录 Git commit、配置快照和运行环境；
- 摘要证据若要参与 Evidence@K，应保留到原始 `dia_id` 的证据链接。

## 数据许可

LoCoMo 数据集遵循其上游许可。分发或发表数据集相关产物时，需要遵守上游署名和
非商业使用要求。仓库代码与数据集文件不应被默认视为采用同一种许可证。
