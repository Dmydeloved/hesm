# HESM 与 LoCoMo 实验实现说明

本文档对应当前仓库代码，共分为两部分：

1. HESM 实现文档：说明系统目标、总体设计、数据结构、记忆构建和检索的逐步实现；
2. 实验实现文档：说明 LoCoMo 实验组织、对比方法、消融设计、运行流程以及每项指标的计算方式。

文档描述“当前代码实际执行的逻辑”。模型名称、服务地址和参数以运行时配置为准；本文不记录任何 API Key。

---

# 第一部分：HESM 设计与实现文档

## 1. 实现边界

HESM 核心实现全部位于 hesm/，由三条主线组成：

1. 主题提取：hesm/extractor.py；
2. 记忆管理：hesm/manager.py；
3. 记忆检索：hesm/retriever.py。

以下模块是三条主线使用的内部基础设施：

- hesm/storage.py：结构化存储；
- hesm/vector_store.py：向量索引；
- hesm/embedder.py：三层向量表示；
- hesm/summarizer.py：Segment 和 Experience 摘要；
- hesm/service.py：对外记忆添加与检索接口；
- hesm/prompts/：主题提取、摘要和检索 Prompt。

生产 HESM 只读取 `config/hesm.yaml`，并将 SQLite/Chroma 数据写入根目录
`memory/`。LoCoMo 实验只读取 `experiments/config/locomo.yaml`，所有实验记忆、
答案、指标、日志和表格均写入 `experiments/outputs/`。两套配置和存储不互相回退。

## 2. HESM 设计

### 2.1 设计目标

长期记忆需要同时满足：

- 稳定聚合：同一长期主题的多轮交互能够归入同一记忆；
- 阶段划分：同一主题下不同意图和子任务能够分段；
- 事实追溯：高层摘要可以回到原始交互证据；
- 分层检索：先定位长期主题，再定位阶段和原始证据；
- 上下文压缩：在有限 token 预算内保留高价值信息；

### 2.2 三层结构

HESM 把长期记忆设计为：

~~~text
Experience（长期主题或目标）
└── Segment（该主题下的阶段或子目标）
    └── QA（原始交互证据）
~~~

三层职责：

| 层级 | 设计职责 |
|---|---|
| Experience | 保存稳定 topic、核心对象、长期摘要和整体状态 |
| Segment | 保存某个 intent 对应的阶段过程、结论和关键事实 |
| QA | 保存原始输入、输出、时间、实体和来源信息 |

QA 解决可追溯问题，Segment 解决过程压缩问题，Experience 解决长期聚合问题。

### 2.3 核心设计原则

#### 证据与摘要分离

QA 保留事实原文；Segment 和 Experience 提供不同粒度的语义压缩。检索上下文可以同时提供摘要和原始证据，避免“只有原文难理解”和“只有摘要难追溯”两个极端。

#### 稳定 topic，变化 intent

topic 用于长期聚合，不是当前输入的标题或摘要。

core_entity 表示长期关注对象。

intent 表示当前动作或阶段子目标。

概念模型为：

~~~text
Experience 身份 = topic + core_entity
Segment 边界    = intent 的语义变化
QA              = 一次原始交互及其结构化语义
~~~

#### 结构关系优先

SQLite 中的父子关系是真实关系。向量相似度只用于召回候选，不能用来重新定义 QA 的父 Segment 或 Segment 的父 Experience。

#### 高层导航，底层救援

正常检索从 Experience 向下展开。查询主题置信度低时，允许直接全局检索 QA，再沿真实关系补齐父节点。

#### 检索结果保持树结构

最终候选和返回结果必须是合法的 Experience → Segment → QA 子树，而不是三个互不相关的 Top-K 列表。

### 2.4 总体数据流

~~~text
写入：
当前输入 + 历史上下文 + 可选领域知识
        → TopicExtractor
        → topic/core_entity/intent/entities/confidence
        → MemoryManager.add_qa()
        → Experience 路由
        → Segment 边界判断
        → QA 写入、摘要更新、向量同步

检索：
自然语言查询
        → TopicExtractor（可复用）
        → HybridRetriever.recall()
        → Experience 根召回
        → 加载 Segment/QA 后代
        → 低置信度 QA 救援
        → 候选树裁剪
        → LLM 联合重排或本地回退
        → Experience + Segment + QA 上下文
~~~

## 3. 主题提取实现

### 3.1 主题提取目的

TopicExtractor 将非结构化输入转换为 HESM 可以路由和检索的结构字段：

- topic：稳定的长期讨论领域；
- core_entity：唯一、具体、可长期追踪的核心对象；
- intent：当前业务动作或阶段子目标；
- entities：相关实体集合；
- confidence：本次抽取可信度；
- reasoning：抽取依据。

其中 topic 决定长期聚合稳定性，intent 决定 Segment 是否需要切换，confidence 决定检索时是否启用全局 QA 救援。

### 3.2 输入

TopicExtractor.extract() 接收：

~~~text
user_input            当前输入
context               历史上下文，可为空
domain_knowledge      领域知识图谱或领域约束，可为空
~~~

build_extractor_prompt() 将它们填入 hesm/prompts/extractor_prompt.txt：

~~~text
{user_input}
{conversation_context}
{domain_knowledge}
~~~

TopicExtractor 本身不规定历史窗口长度，历史上下文的选择由上层业务调用方决定。

### 3.3 输出 Schema

每个主题记录必须符合：

~~~json
{
  "topic": "稳定的长期讨论领域",
  "core_entity": "核心对象",
  "intent": "中文业务意图",
  "entities": ["实体1", "实体2"],
  "confidence": 0.0,
  "reasoning": "topic来源 + core_entity来源 + intent判断依据"
}
~~~

输入可以包含多个独立长期主题。此时输出多个记录，而不是在一个记录中使用 topics 数组。每个记录只能有一个 topic、一个 core_entity 和一个 intent。

### 3.4 Prompt 设计

完整主题提取 Prompt 位于 [hesm/prompts/extractor_prompt.txt](hesm/prompts/extractor_prompt.txt)。其关键约束如下。

#### Topic 规则

- topic 是 Discussion Domain，不是当前问题摘要；
- 同一讨论对象的 topic 应在多轮中保持稳定；
- topic 必须能聚合未来同类问题；
- 抽象程度不能过细，也不能宽泛到“技术”“金融”；
- 当前输入明显承接历史时优先继承历史 topic；
- 输入同时包含独立领域或任务时拆分多个记录。

#### Core Entity 规则

- 唯一、具体、可长期追踪；
- 优先选择人物、产品、组织、框架、模型或具体业务对象；
- 多个核心对象属于独立任务时拆分多个记录；
- 有知识图谱时优先映射 Entity Node。

#### Intent 规则

- 表示用户当前希望执行的动作；
- 必须输出中文；
- 优先使用领域业务意图，如“方案设计”“原因分析”“代码实现”；
- 无明确领域意图时使用查询、分析、比较、推理、推荐、执行、总结、解释、评估、设计、实现、优化等通用意图；
- 相同语义应归一化为稳定表达。

#### Entities、Confidence 和 Reasoning

- entities 为去重、标准化的名词短语，不包含代词和完整句；
- confidence 位于 0.0—1.0，根据明确程度、知识图谱匹配、上下文依赖和歧义评估；
- reasoning 简洁记录 topic、core_entity 和 intent 的判断依据；
- 所有字段必须可追溯到当前输入、上下文或领域知识，禁止编造。

### 3.5 Prompt 核心版

~~~text
# Role

你是 Topic-CoreEntity-Intent Extractor。
基于当前输入、历史上下文和可选领域知识，提取适用于
Experience → Segment → QA 长期记忆的结构化语义。

# Input

【当前输入】
{user_input}

【历史上下文】
{conversation_context}

【领域知识图谱（可选）】
{domain_knowledge}

# Rules

- topic 是稳定、可聚合的长期讨论领域，不是当前问题摘要。
- core_entity 是唯一、具体、可长期追踪的核心对象。
- intent 是当前希望执行的动作，优先输出中文领域业务意图。
- entities 是去重、标准化、可追溯的名词短语。
- 延续历史主题时继承 topic。
- 多个独立领域、对象或任务拆成多个语义记录。
- domain_knowledge 存在时优先映射 Topic/Entity/Intent。
- 禁止编造不存在的主题、实体和关系。
- confidence 范围为 0.0—1.0。
- reasoning 简述判断依据。

# Output

只输出非空 JSON 数组：

[
  {
    "topic": "",
    "core_entity": "",
    "intent": "",
    "entities": [],
    "confidence": 0.0,
    "reasoning": ""
  }
]
~~~

### 3.6 调用、解析与重试

TopicExtractor.extract() 的实现流程：

1. 加载 Prompt 模板并替换输入；
2. 调用 OpenAI-compatible chat completions；
3. temperature 固定为 0；
4. 剥离可能存在的 Markdown code fence；
5. 使用 json.loads() 解析对象或数组；
6. 校验六个字段是否齐全；
7. 校验字符串非空、entities 非空、confidence 位于 [0,1]；
8. 失败时按 retry_delay × attempt 退避重试；
9. 全部失败后抛出 RuntimeError。

Prompt 要求输出数组；解析器为兼容模型也接受单个 JSON 对象。单元素数组会规范化为对象，多元素数组保持列表。

### 3.7 多主题处理

TopicExtractor 只负责返回一个或多个主题记录。调用方逐项交给 MemoryManager：

~~~python
result = extractor.extract(
    user_input=current_input,
    context=history_context,
    domain_knowledge=domain_knowledge,
)
records = result if isinstance(result, list) else [result]

for record in records:
    manager.add_qa(
        topic_result=record,
        user_input=current_input,
        assistant_output=current_output,
        tools=tool_records,
        timestamp=timestamp,
        state_key=state_key,
    )
~~~

每个主题记录独立执行 Experience 路由和 Segment 边界判断。

## 4. 记忆管理实现

MemoryManager 是 HESM 写入侧核心。它负责三层节点生命周期，并协调 storage、summarizer、embedder 和 vector_store。

### 4.1 内部组件

| 组件 | 在记忆管理中的职责 |
|---|---|
| MemoryStorage | 保存三层节点、真实父子关系和 runtime_state |
| Summarizer | 更新 Segment/Experience 摘要 |
| TextEmbedder | 生成节点向量 |
| ChromaVectorStore | upsert 和查询三层语义索引 |

### 4.2 SQLite 数据结构

MemoryStorage 初始化：

- qa_memory；
- segment_memory；
- experience_memory；
- runtime_state。

真实父子关系：

~~~text
qa_memory.segment_id
segment_memory.experience_id
~~~

父节点也保存 qa_ids_json 或 segment_ids_json，用于聚合和顺序恢复。

runtime_state 按 state_key 保存当前 Experience 和 Segment，作为连续写入游标。

### 4.3 add_qa() 总流程

~~~text
校验主题字段
  → 读取 runtime_state
  → Experience 路由
  → Segment 边界判断
  → 创建 QA
  → 写入 QA 向量
  → 更新 Segment、摘要和向量
  → 更新 Experience、摘要和向量
  → 更新 runtime_state
  → commit
~~~

异常时 SQLite rollback。

### 4.4 Experience 路由

当前实现：

1. 当前 Experience 存在且 current.topic == new.topic：直接复用；
2. topic 改变时先总结旧 Segment 和旧 Experience；
3. 用“主题 + 核心实体”构造向量查询文本；
4. 在 Experience 向量中查询 Top-5；
5. 相似度达到 experience_similarity_threshold 时复用；
6. Embedding 或向量查询失败时回退 topic 精确匹配；
7. 未命中时创建新 Experience。

默认 experience_similarity_threshold=0.82。

需要注意设计与当前实现的差异：

- Prompt 和概念设计把 topic + core_entity 定义为 Experience 身份；
- 当前 _same_experience() 快速路径只硬比较 topic；
- core_entity 参与向量查询和节点内容，但不是硬路由键。

因此当前实现倾向于减少同 topic 的重复 Experience。若业务要求不同核心对象严格隔离，需要把快速路径改为 topic + core_entity，或增加可配置路由策略。

### 4.5 Segment 边界判断

创建新 Segment 需要同时满足：

1. 新 intent 与当前 intent 字符串不同；
2. 两个 intent 的本地词袋余弦相似度小于 0.8；
3. 当前 Segment 已有 QA 数达到 min_segment_qas，默认 2。

本地 intent 相似度不调用模型。最小 QA 数用于避免产生大量单条 Segment。

需要切换时先总结旧 Segment，再创建新 Segment。

### 4.6 QA 创建

QA 使用 qa_<uuid> 作为 ID，保存：

- timestamp；
- user_input、assistant_output、tools；
- topic、core_entity、intent、entities；
- confidence、reasoning；
- segment_id；
- status=active。

QA 插入 SQLite 后立即构造向量并 upsert。

### 4.7 Segment 更新

新 qa_id 加入 segment.qa_ids，更新 updated_at。

摘要触发条件：

~~~text
当前 QA 数 - last_summarized_qa_count
>= segment_summary_qa_threshold
~~~

默认每新增 5 条 QA 更新摘要。intent 或 Experience 切换时也会总结当前 Segment。

摘要后更新 summary、version、last_summarized_qa_count 和向量。

### 4.8 Experience 更新

当前 Segment 加入 experience.segment_ids，新 intent 加入 intents_link，并更新 state。

摘要触发条件：

~~~text
当前 Segment 数 - last_summarized_segment_count
>= experience_summary_segment_threshold
~~~

默认每新增 5 个 Segment 更新摘要。Experience 切换时也会总结旧 Experience。

### 4.9 Segment 摘要 Prompt

完整模板：[hesm/prompts/segment_summary_prompt.txt](hesm/prompts/segment_summary_prompt.txt)。

它要求同时提炼：

- 过程记忆：阶段发生了什么、方案如何演进、形成哪些决策和约束；
- 事实记忆：已确认且未来有检索价值的具体事实。

固定格式：

~~~text
阶段概述：
...
关键过程：
...
阶段结论：
...
关键事实：
类型：事实内容
~~~

猜测、被否定内容、未确认信息和临时讨论不进入关键事实。

### 4.10 Experience 摘要 Prompt

完整模板：[hesm/prompts/experience_summary_prompt.txt](hesm/prompts/experience_summary_prompt.txt)。

它要求保持长期目标不变，聚合阶段推进、整体状态、长期信息类别和基于证据的下一步：

~~~text
目标：...
总体状态：进行中 / 已暂停 / 已阻塞 / 已完成
阶段总结：...
当前推进：...
长期信息：...
下一步：...
~~~

具体事实应留在 Segment/QA，Experience 只保留高层长期表示。

### 4.11 摘要调用和降级

LLMSummarizer：

- Segment 最多输入最近 20 条规范化 QA；
- Experience 最多输入最近 20 个规范化 Segment；
- 长字段进入 Prompt 前截断；
- temperature=0；
- 按配置重试；
- 空响应或全部失败时回退 TemplateSummarizer。

### 4.12 三层向量文本

QA 向量文本：

~~~text
主题、核心实体、用户意图、相关实体、
用户输入、助手回答摘要
~~~

Segment 向量文本：

~~~text
主题、核心实体、阶段意图、阶段摘要、
片段状态、最近 3 条 QA 输入
~~~

Experience 向量文本：

~~~text
主题、核心实体、相关意图、长期摘要、
当前状态、最近 3 个 Segment
~~~

QA 创建后写入一次；Segment 和 Experience 在新增子节点、状态或摘要变化后 upsert。相同 memory_type + memory_id 覆盖旧向量。

### 4.13 双存储一致性

SQLite 是权威结构库，Chroma 是语义候选索引。

MemoryManager 对 SQLite 使用 commit/rollback，但 SQLite 与 Chroma 不属于同一事务。SQLite rollback 无法撤销已完成的向量 upsert。稳定 ID 和重复 upsert 可以修复多数部分失败，当前尚无跨存储原子事务。

## 5. 检索实现

HybridRetriever 是 HESM 读取侧核心，采用 Experience-first 策略。

### 5.1 输入与查询结构

recall() 接收：

~~~text
topic
core_entity
intent
entities
query
query_confidence
top_experience
top_segment
top_qa
~~~

自然语言查询通常先复用 TopicExtractor，生成检索结构。查询文本把结构字段和原 query 拼接后计算 embedding。

### 5.2 Experience 根召回

根候选来自：

- Chroma Experience 向量召回；
- SQLite topic/core_entity 关系召回。

去重后用 RRF 风格融合：

~~~text
vector_rrf   = 61 / (60 + vector_rank)
relation_rrf = 61 / (60 + relation_rank)

root_score = 0.4 × vector_rrf
           + 0.4 × relation_rrf
           + 0.2 × root_local_score
~~~

root_local_score 综合向量相似度和 topic/core_entity/intent 结构匹配。

### 5.3 加载真实后代

对初始 Experience：

1. SQLite 批量加载未删除 Segment；
2. SQLite 批量加载 active QA；
3. Segment 向量查询限制 experience_id；
4. QA 向量查询限制 segment_id；
5. 计算子节点局部相关性。

正常高置信度路径不会直接全库搜索 QA。

### 5.4 局部评分

Segment 和 QA 的基础评分：

~~~text
local_score = 0.35 × vector_similarity
            + 0.35 × keyword_score
            + 0.30 × structure_score
~~~

structure_score 对 topic、core_entity、intent 等权匹配。

QA 再融合写入时的抽取置信度：

~~~text
qa_score = 0.95 × local_score + 0.05 × qa.confidence
~~~

### 5.5 低置信度 QA 救援

query_confidence 低于默认阈值 0.8 时：

1. 全局 QA 向量召回；
2. 使用 topic、core_entity、entities 和关键词执行 SQLite 关系召回；
3. 合并、去重并评分；
4. 为新增 QA 加载真实 Segment；
5. 为 Segment 加载真实 Experience；
6. 删除无法形成完整父链的节点。

这条路径只补真实父链，不按相似度虚构关系。

### 5.6 候选树构造与校验

候选统一组织为 Experience → Segment → QA，并要求：

- 各层 ID 唯一；
- 每个 Experience 至少有一个 Segment；
- 每个 Segment 至少有一个 QA；
- QA/Segment 使用 SQLite 中的真实父 ID；
- 不存在孤儿节点和空父节点。

### 5.7 上下文预算

候选树在 LLM 重排前序列化并计数。默认 max_context_tokens=30000。

超限时依次：

1. 删除最低分 QA，但每个 Segment 保留最强 QA；
2. 删除最低分 Segment，并保护结构最小值；
3. 删除最低分 Experience；
4. 压缩摘要并应用软字符上限；
5. 使用更严格的摘要上限；
6. 仍超限时截断最长 QA payload。

默认至少保留 1 个 Experience 和 2 个 Segment。最终仍超限时跳过 LLM reranker，改用本地选择。

### 5.8 LLM 联合层级重排

新版检索器把完整候选树一次性交给模型，联合选择 Experience、Segment 和 QA。

运行时 Prompt 由 hesm/retriever.py 的 build_hierarchical_retrieval_prompt() 动态生成。核心规则：

~~~text
1. 联合判断完整 Experience → Segment → QA 路径。
2. 向量、关系、关键词分数只是召回提示，不是最终答案。
3. 优先选择具体、可追踪、能够帮助回答查询的 QA。
4. 遵守三层全局数量上限。
5. 每个子节点必须位于其真实父节点下。
6. 只能选择候选树中的 ID，不得编造。
~~~

返回结构：

~~~json
{
  "experiences": [
    {
      "id": "experience id",
      "score": 0.0,
      "reason": "brief reason",
      "segments": [
        {
          "id": "segment id",
          "score": 0.0,
          "reason": "brief reason",
          "qas": [
            {"id": "qa id", "score": 0.0, "reason": "brief reason"}
          ]
        }
      ]
    }
  ]
}
~~~

解析后再次检查 ID、去重、父子归属和 top_experience/top_segment/top_qa。

完整实现引用见第 6 节。

### 5.9 本地回退

关闭 reranker、模型异常、无有效 QA 或候选仍超限时，使用本地分支评分：

~~~text
segment_path_score = 0.7 × 分支最高 QA 分
                   + 0.3 × Segment 分

experience_branch_score = 0.6 × 分支最高 QA 分
                        + 0.3 × 分支最高 Segment 路径分
                        + 0.1 × Experience 分
~~~

本地回退同样保持真实父子关系和三层数量上限。

### 5.10 结果组装

最终结果移除没有已选后代的父节点，并生成：

~~~text
【长期记忆 Experience】
主题、核心实体、相关意图、长期摘要

【相关片段 Segment】
阶段意图、阶段摘要

【原始问答 QA】
时间、用户输入、助手输出
~~~

recall() 返回 query、experiences、segments、qas、candidate_tree、context_text 和 debug。

debug 包含候选来源、低置信度救援、裁剪前后数量、token 预算、删除节点和 LLM 调用次数。

## 6. Prompt 与源码引用

### 6.1 主题提取

- 完整 Prompt：[hesm/prompts/extractor_prompt.txt](hesm/prompts/extractor_prompt.txt)
- Prompt 构建：[hesm/prompts/topic_memory.py](hesm/prompts/topic_memory.py) 的 build_extractor_prompt()
- 调用、解析与校验：[hesm/extractor.py](hesm/extractor.py)

### 6.2 Segment 摘要

- 完整 Prompt：[hesm/prompts/segment_summary_prompt.txt](hesm/prompts/segment_summary_prompt.txt)
- Prompt 构建：[hesm/prompts/topic_memory.py](hesm/prompts/topic_memory.py) 的 build_segment_summary_prompt()
- 调用与降级：[hesm/summarizer.py](hesm/summarizer.py)

### 6.3 Experience 摘要

- 完整 Prompt：[hesm/prompts/experience_summary_prompt.txt](hesm/prompts/experience_summary_prompt.txt)
- Prompt 构建：[hesm/prompts/topic_memory.py](hesm/prompts/topic_memory.py) 的 build_experience_summary_prompt()
- 调用与降级：[hesm/summarizer.py](hesm/summarizer.py)

### 6.4 层级检索

当前实际使用的联合层级 Prompt：

- 构建：[hesm/retriever.py](hesm/retriever.py) 的 build_hierarchical_retrieval_prompt()
- 调用：LLMRetrievalReranker.rerank_hierarchy()
- 解析：parse_hierarchical_rerank_response()

兼容保留的旧版逐层 Prompt：

- 模板：[hesm/prompts/retrieval_prompt.txt](hesm/prompts/retrieval_prompt.txt)
- 构建：[hesm/prompts/topic_memory.py](hesm/prompts/topic_memory.py)
- 调用：LLMRetrievalReranker.rerank()

新版 HybridRetriever.recall() 正常使用联合层级 Prompt。

### 6.5 核心实现索引

| 核心过程 | 文件 |
|---|---|
| 主题提取 | [hesm/extractor.py](hesm/extractor.py) |
| 记忆管理 | [hesm/manager.py](hesm/manager.py) |
| 检索 | [hesm/retriever.py](hesm/retriever.py) |
| 结构存储 | [hesm/storage.py](hesm/storage.py) |
| 向量索引 | [hesm/vector_store.py](hesm/vector_store.py) |
| 向量表示 | [hesm/embedder.py](hesm/embedder.py) |
| 摘要 | [hesm/summarizer.py](hesm/summarizer.py) |

# 第二部分：实验实现文档

## 10. 实验目的和总体思路

实验使用 LoCoMo 长对话问答数据，比较不同记忆系统在以下方面的差异：

- 最终答案质量；
- 原始证据检索质量；
- 上下文 token 成本；
- 检索、回答和评判延迟；
- HESM 各层结构和 reranker 的独立贡献。

所有方法实现统一的 MemorySystem 接口：

~~~text
reset()
build_memory(conv_id, sessions, speaker_a, speaker_b)
retrieve(question, top_k) -> RetrievalResult
~~~

每种方法只负责“怎样构建和检索记忆”。最终答案生成器、Judge、数据加载器、指标代码、checkpoint 和报告生成逻辑在方法之间共享。

## 11. 数据加载

默认数据文件为 data/locomo10.json。加载器将每个样本解析为：

- conv_id 和两位说话人；
- 按时间排序的 Session；
- 每个 Session 内的 Turn；
- 问题、标准答案、Evidence dia_id 和类别。

Turn 的关键字段为 speaker、text、dia_id、session_num 和 timestamp。问题证据使用相同 dia_id 格式，因此各方法必须尽可能把召回内容映射回来源 turn。

正式分析通常把类别 1—4 作为普通问答集合，把类别 5 作为 Adversarial 单独报告。当前自动 aggregate() 不会自动排除类别 5，具体口径见第 19 节。

## 12. 主实验方法

### 12.1 Vector RAG

Vector RAG 是原始 turn 级稠密检索基线：

1. 把每条 turn 格式化为 [speaker]: text；
2. 逐条计算 embedding；
3. 写入每个 conversation 独立的 Chroma；
4. 查询时 embedding 问题；
5. 按余弦相似度返回 Top-K 原始 turn。

它不使用主题抽取、摘要、图结构或 LLM 重排。

### 12.2 Mem0

Mem0 适配器调用 mem0ai：

1. 每个 conversation 使用独立 user_id 和 Chroma collection；
2. 每个 turn 调用 Memory.add()，由 Mem0 内部抽取或更新记忆；
3. 原始 dia_id、speaker、session 和 timestamp 放入 metadata；
4. 查询时调用 Memory.search(query, filters={user_id}, top_k)；
5. 返回 Mem0 memory 文本，并从 metadata 恢复 dia_id。

构建进度写入 build_state.json；旧版存储可以从 Chroma metadata 恢复已完成 dia_id。

### 12.3 A-MEM

A-MEM 按原子笔记和关系链接思路实现。

写入阶段：

1. LLM 提取最多 8 个关键词；
2. LLM 为原始 turn 生成上下文描述；
3. embedding “context + keywords”；
4. 查询最相近的已有笔记；
5. 相似度达到阈值时建立双向链接；
6. SQLite 保存 note 和链接，Chroma 保存向量。

查询阶段：

1. LLM 提取问题关键词；
2. 对全部 note 计算关键词覆盖分；
3. 为问题生成上下文并进行向量召回；
4. 使用 0.4 × keyword + 0.6 × vector 合并分数；
5. 展开首批候选的图邻居；
6. 按合并分数返回 Top-K 原始 note。

当前代码注释中提到可选 LLM reranking，但实际 _search() 没有执行独立的查询结果 reranker。

### 12.4 HESM

HESM 使用第一部分描述的三层构建和 Experience-first 检索，返回层级摘要与原始 QA 的混合上下文，同时只用最终 QA 的 dia_id 参与证据指标。

## 13. 主实验执行流程

主入口为 experiments/locomo/run_main.py，一键入口为 experiments/locomo/run_all.py。

### 13.1 初始化

1. 读取 `experiments/config/locomo.yaml`；
2. 设置随机种子；
3. 创建 answers、metrics、tables、logs 目录；
4. 加载 LoCoMo conversations；
5. 构造启用的 MemorySystem；
7. 为每个方法构造同配置的答案模型和 Judge。

### 13.2 BUILD 阶段

Runner 对每个 conversation 调用：

~~~text
method.reset()
method.build_memory(...)
~~~

持久化方法会连接已有存储并跳过已完成 turn。构建中的单条失败以 warning 记录；如果有部分构建警告，该 conversation 的 QA 会降为串行执行，以降低并发读取部分存储的风险。

### 13.3 RETRIEVAL 阶段

Runner 用 max(top_k_values) 作为统一方法的查询 Top-K。默认 top_k_values=[1,3,5]，因此 Vector RAG、Mem0 和 A-MEM 返回 Top-5；HESM 使用自己的三层上限。

记录内容：

- 有序 retrieved_ids；
- 检索上下文；
- 检索 token 数；
- 检索耗时；
- 方法特有 raw_result。

raw_result.error 非空时本次检索判为失败，后续 Answer 和 Judge 跳过。

### 13.4 ANSWER 阶段

所有方法使用同一个 LLMAnswerGenerator 和同一 Prompt。Prompt 要求模型：

- 只能使用检索上下文；
- 无法从上下文推出时回答 Unknown；
- 避免 yesterday、recently 等相对表达；
- 尽量明确人物、日期和事件；
- 答案保持简洁、事实化。

模型温度为 0，并按配置重试。空答案视为失败。

### 13.5 JUDGE 阶段

Judge 接收问题、标准答案和预测答案，不接收检索上下文。它按事实正确性输出 CORRECT 或 WRONG，并转换为：

- CORRECT → 1；
- WRONG → 0；
- 调用或解析完全失败 → -1。

### 13.6 逐题记录和方法汇总

每题完成后立即计算答案 F1、Evidence@K、token 和压缩比，并与三阶段耗时、状态一起写入 checkpoint。

一个方法的全部 conversation 完成后，aggregate() 对逐题指标做宏平均并生成：

~~~text
experiments/outputs/locomo/metrics/<method>_metrics.json
experiments/outputs/locomo/tables/main_results.{md,csv,json}
~~~

## 14. 并发实现

实验支持两层并发：

- method_workers：同时运行多个方法；
- qa_workers：同一方法完成记忆构建后并发处理问题。

每个 QA 工作线程拥有独立的方法读取器、答案模型客户端和 Judge 客户端。线程首次处理某个 conversation 时调用 build_memory()，持久化方法通常只连接和验证已有存储。Checkpoint 由 conversation 所在线程串行写入，避免多个 worker 同时修改同一个 JSON。

高并发会让不同方法竞争模型服务和网络资源，因此比较延迟时应固定并发配置，并记录 method_workers × qa_workers。

## 15. Checkpoint 和失败重试

每个 (method, conv_id) 对应：

~~~text
experiments/outputs/locomo/answers/<method>_<conv_id>.json
~~~

只有以下条件全部满足，问题才标记为成功：

- Retrieval 成功；
- Answer 成功且非空；
- Judge 成功，或实验明确关闭 Judge。

失败记录仍保存在 records 中用于诊断，但不会加入 completed_indices，下次运行会自动重试。旧 checkpoint 只有在预测非空且 judge_score >= 0 时才被视为成功。

## 16. 四阶段结构化日志

每种方法写入：

~~~text
experiments/outputs/locomo/logs/<method>.log
~~~

事件阶段包括 BUILD、RETRIEVAL、ANSWER、JUDGE。每个阶段记录 STARTED、SUCCESS、FAILED 或 SKIPPED，以及 conversation、问题下标、耗时和错误原因。日志用于区分“指标低”和“API/构建失败”。

## 17. 消融实验实现

消融入口为 experiments/locomo/run_ablation.py。五种变体共享 Runner、答案模型、Judge 和指标实现。

| 变体 | 记忆来源 | 查询主题抽取 | 检索方式 | LLM reranker |
|---|---|---:|---|---:|
| flat_memory | 独立原始 turn 向量库 | 否 | 问题向量直接查 Top-K turn | 否 |
| qa_only | HESM QA 向量层 | 否 | 问题向量直接查 Top-K QA | 否 |
| qa_segment | HESM QA + Segment 向量层 | 否 | 两层各查 Top-K，拼接上下文 | 否 |
| full_hesm_no_reranker | 完整 HESM 存储 | 是 | Experience-first + 本地选择 | 否 |
| full_hesm | 完整 HESM 存储 | 是 | Experience-first + 联合重排 | 是 |

除 flat_memory 外，其余变体复用主实验 HESM SQLite 和 Chroma。存储不存在时，消融入口会先构建 HESM 记忆。

qa_segment 的 retrieved_ids 只来自 QA；Segment 只增加回答上下文，不增加可直接计算的 Evidence ID。因此它可能提高答案质量而不改变 Evidence@K。

## 18. 评测指标及计算方法

### 18.1 答案 Token Precision、Recall 和 F1

预测和标准答案先做归一化：

1. 转为小写；
2. 删除 string.punctuation 中的英文标点；
3. 合并多余空白；
4. 按空格切分 token。

使用词频多重集合计算共同 token 数：

~~~text
common = Σ_token min(count_pred(token), count_gold(token))

Answer Precision = common / 预测 token 数
Answer Recall    = common / 标准答案 token 数
Answer F1        = 2PR / (P + R)
~~~

边界规则：

- 预测和标准答案都为空：P=R=F1=1；
- 仅一方为空：P=R=F1=0；
- 没有公共 token：F1=0。

方法级 avg_f1、avg_precision、avg_recall 是逐问题宏平均，不是把所有问题 token 合并后的微平均。

实现：experiments/locomo/evaluation/f1.py。

### 18.2 LLM-as-a-Judge

Judge 判断预测是否包含全部关键事实、没有额外不受支持事实，并且人物、实体和时间正确。每题得分为 0 或 1：

~~~text
Judge = Σ valid_judge_score / 有效 Judge 数
~~~

judge_score=-1 表示调用或解析失败，聚合时排除。比较不同方法 Judge 时必须同时报告有效样本数，否则分母变化可能造成误导。

实现：experiments/locomo/evaluation/judge.py 和 aggregator.py。

### 18.3 Evidence Recall@K

Evidence 匹配采用 dia_id 精确字符串相等，不做文本、语义或摘要匹配。令：

~~~text
R_K = 前 K 个 retrieved_ids 的集合
E   = 标准 evidence_ids 的集合
H   = R_K ∩ E

Evidence Recall@K = |H| / |E|
~~~

标准 evidence 为空时 Recall 为 0。多证据问题必须找回全部 evidence 才能达到 1。

### 18.4 Evidence Precision@K

~~~text
Evidence Precision@K = |H| / K
~~~

当前实现分母固定为 K，而不是实际返回条数。方法返回少于 K 条时，未返回位置仍相当于错误项。这便于统一固定检索预算，但与“实际返回数作分母”的版本不同。

### 18.5 Evidence F1@K

~~~text
Evidence F1@K = 2 × Precision@K × Recall@K
                / (Precision@K + Recall@K)
~~~

两者均为 0 时 Evidence F1 为 0。结果表通常标记为 EvF1@K，避免与答案 F1 混淆。

### 18.6 Retrieval Accuracy@K

~~~text
Accuracy@K = 1，若 |H| > 0
Accuracy@K = 0，若 |H| = 0
~~~

它表示前 K 条中是否至少命中一条证据，不要求召回全部证据。方法级 Accuracy@K 是逐题 0/1 平均，即至少命中一条 evidence 的问题比例。

### 18.7 检索指标宏平均

Recall、Precision、Evidence F1 和 Accuracy 先逐题计算，再取算术平均：

~~~text
MacroMetric@K = (1/N) × Σ_i Metric_i@K
~~~

默认 K 为 1、3、5。实现：experiments/locomo/evaluation/retrieval_metrics.py。

### 18.8 上下文 token 数

retrieved_tokens 是交给答案模型的 context_text token 数。默认使用 cl100k_base；tiktoken 不可用时回退到空白分词。avg_retrieved_tokens 是逐题平均上下文 token 数。

### 18.9 压缩比

先计算整个 conversation 原始 turn 文本 token 数，然后逐题计算：

~~~text
compression_ratio = total_conversation_tokens / retrieved_tokens
~~~

结果至少为 1。retrieved_tokens=0 时，完整对话非空则返回 total_conversation_tokens，否则返回 1。

方法级结果是逐题压缩比平均，不一定等于“平均总 token / 平均检索 token”。

实现：experiments/locomo/evaluation/token_metrics.py。

### 18.10 延迟

每题分别记录：

- latency_ms：只包含 method.retrieve()；
- answer_latency_ms：只包含答案模型调用；
- judge_latency_ms：只包含 Judge 调用。

BUILD 时间记录在四阶段日志中，不计入 latency_ms。主结果自动表当前不汇总延迟列，需要从 checkpoint 或日志重新聚合。

缓存 Runner 的 percentile 实现：

~~~text
index = max(0, int(N × p / 100) - 1)
percentile = sorted_latency[index]
~~~

这是离散下取整式百分位，不是线性插值百分位。

### 18.11 分类指标和类别 5

category_f1 和 category_judge 在类别内部宏平均；Judge 仍排除 -1。

当前 aggregate() 会聚合 checkpoint 中存在的全部类别。若正式报告规定类别 1—4 为主集合、类别 5 单独报告，应在聚合前过滤类别 5，或者从逐题记录重新计算。仅在文字中声明排除类别 5，不会改变自动 metrics.json 和表格。

### 18.12 失败记录对指标的影响

Runner 会保留失败问题记录：

- Retrieval/Answer 失败时 prediction 通常为空，答案 F1 按空预测计算；
- Judge 失败时为 -1，Judge 平均排除该题；
- 失败问题不进入 completed_indices，下次运行会重试；
- 方法级聚合使用 checkpoint 中当前存在的记录，包括尚未修复的失败记录。

正式比较应同时报告总问题数、成功问题数、Judge 有效数和失败原因。

## 19. 输出文件

~~~text
experiments/outputs/locomo/
├── memory/                 # 各方法、各 conversation 的持久记忆
├── answers/                # 逐问题 checkpoint 和完整上下文
├── logs/                   # 四阶段 JSONL 日志
├── metrics/                # 方法级聚合 JSON
└── tables/                 # Markdown、CSV、JSON 对比表
~~~

主实验输出 answers/<method>_<conv_id>.json、metrics/<method>_metrics.json 和 tables/main_results.*。

消融实验输出 answers/ablation_<variant>_<conv_id>.json、metrics/ablation_<variant>_metrics.json 和 tables/ablation_results.*。

缓存实验输出 metrics/cache_raw.json 和 tables/cache_results.*。

## 20. 运行入口

在仓库根目录执行：

~~~bash
# 一键运行主实验、消融和缓存部分
python -m experiments.locomo.run_all

# 只运行主实验
python -m experiments.locomo.run_main

# 只运行指定方法
python -m experiments.locomo.run_main --methods vector_rag mem0 amem hesm

# 快速运行前两个 conversation
python -m experiments.locomo.run_main --max-conversations 2

# 运行全部或部分消融
python -m experiments.locomo.run_ablation
python -m experiments.locomo.run_ablation --variants qa_only full_hesm

# 当前缓存延迟采样入口
python -m experiments.locomo.run_cache --num-samples 50
~~~
