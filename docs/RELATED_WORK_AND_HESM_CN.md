# 大语言模型长期记忆系统相关工作与 HESM 项目定位

## 1. 研究问题与相关工作脉络

长时交互中的记忆问题不仅是“从历史文本中找到相似片段”。系统还要决定哪些信息值得保留、以何种粒度组织、如何处理新旧信息变化、怎样为不同问题组合证据，并控制检索延迟和上下文成本。现有工作大致从四个方向展开：从对话中抽取并更新事实（Mem0）、让记忆单元动态建立联系并演化（A-MEM）、以分层存储与迁移管理短中长期记忆（MemoryOS）、以时间知识图谱维护事实关系及其历史（Zep）。OpenClaw 则提供了一个持续演进的开源智能体记忆工程案例，强调可读文件、工具化检索、索引与后台整理。

这些工作关注点不同，不能只按单一准确率排名。HESM 的论文围绕**记忆组织粒度、写入与更新、检索决策、来源可追溯性、生命周期和运行成本**展开。

## 2. 代表性系统的设计理念

### 2.1 Mem0：从交互中提取、整合并按需召回

Mem0 将长期记忆视为对话中的显著信息，而非持续拼接全部历史。其基本流程是从新交互提取候选记忆，与已有记忆比较后执行新增、更新或删除等整合操作；需要回答时，再根据语义相关性及用户、会话等元数据约束召回内容。论文还提出 Mem0g，以图结构表达实体间关系，扩展普通记忆项难以直接表示的关系信息。其核心思想是以持续维护的紧凑记忆替代全量上下文，从而在个性化、延迟与 token 成本之间取得折中。

### 2.2 A-MEM：Zettelkasten 启发的动态笔记网络

A-MEM 将一次经历整理为包含上下文描述、关键词、标签等属性的记忆笔记，并依据语义相似性及共享属性为新笔记建立链接。它还提出“记忆演化”：新信息可以促使既有笔记的上下文描述或属性调整。因此，记忆不是固定 schema 下互不相干的向量条目，而是逐渐形成关联网络的知识单元。该设计强调自组织能力和跨记忆关系发现。

### 2.3 MemoryOS：短期—中期—长期的分层记忆管理

MemoryOS 将操作系统中的层级管理思想用于智能体记忆，划分短期记忆、中期记忆和长期个人记忆，并围绕存储、更新、检索、生成设计模块。对话链及 FIFO 机制用于短期到中期的迁移，分段页式组织用于中期到长期的整理。设计目标是让近期上下文、阶段性信息和稳定个人记忆承担不同职责，通过迁移与整合控制长期记忆规模并保留个性化信息。

### 2.4 Zep / Graphiti：带时间与来源信息的知识图谱

Zep 将 Graphiti 作为其智能体记忆核心，目标是持续整合对话事件与结构化业务数据。Graphiti 以 episode 保存输入事件，并抽取实体与事实关系；事实关系带有时间有效性信息，系统可在新事实到来时处理旧关系失效，同时保留历史和来源关联。检索将图结构信号与关键词、向量等信号组合，用于回答实体关系、时间变化和跨事件问题。相较于只保存当前事实，这一思路将“某事实何时成立、由何处得出”纳入记忆表示。

### 2.5 OpenClaw：文件优先、工具检索与后台整理的工程体系

OpenClaw 的默认记忆体系以工作区 Markdown 文件作为可读的规范记忆：用户偏好、长期记忆和按日记录分层存放；后台索引为文件提供检索入口，`memory_search` 与 `memory_get` 分别用于查找和读取。其文档还描述了会话压缩前的记忆写回、后台整理，以及内置 SQLite 索引和混合检索等机制。设计理念偏向**人可检查的持久化材料 + 机器检索索引 + 生命周期钩子**，并允许不同记忆插件替换默认实现。

## 3. 设计维度对照

| 系统 | 记忆基本单元 | 组织方式 | 更新/生命周期重点 | 主要检索思路 |
|---|---|---|---|---|
| Mem0 | 从交互提取的记忆项 | 向量记忆；可选关系图 | 新增、更新、删除与整合 | 相关记忆召回并受用户/会话元数据约束 |
| A-MEM | 结构化原子笔记 | 动态链接的笔记网络 | 新记忆触发关联与既有笔记演化 | 笔记相关性及网络关联 |
| MemoryOS | 对话链、分段页和个人记忆 | 短期—中期—长期层级 | 在层级间迁移、压缩和更新 | 按层级管理与检索后生成 |
| Zep / Graphiti | episode、实体及带时间属性的事实 | 时间知识图谱 | 新事实更新有效关系，保留历史来源 | 图检索与语义/关键词信号结合 |
| OpenClaw | Markdown 文件及其索引片段 | 可读文件分层、SQLite 派生索引 | 会话压缩前写回、后台整理与索引维护 | 工具化文件读取与混合搜索 |
| HESM（当前实现） | QA、Segment、Experience | 经历—阶段—问答三级树状组织 | 状态推进、增量摘要与异步派生索引 | 运行时路由、层级摘要读取及 QA 混合回退 |

该表是架构维度的定性整理，不是同一数据集、同一模型和同一评测协议下的性能比较。

## 4. HESM 项目本身的特点

HESM 的中心设计是以 **Experience → Segment → QA** 组织交互记忆。Experience 聚合较长周期内的主题与核心实体；Segment 表达其中的阶段/意图；QA 保存具体用户输入、助手输出、工具信息、抽取出的主题/实体/意图、置信度及时间等原始证据。该结构将可读摘要和可追溯对话证据放在同一层级关系中：摘要用于低成本概览，QA 用于回溯和细节检索。

写入路径先由主题抽取器生成结构化记录，再由管理器路由到当前或可复用的 Experience 和 Segment，并将 QA 持久化到 SQLite。Segment 与 Experience 的摘要、向量等派生内容通过 SQLite outbox 交给后台 worker 增量生成或更新。该安排使主存储与派生索引分离，并为异步任务提供可恢复状态。当前代码也包含基于 SQLite FTS5 的 QA 全文检索，以及 Chroma 向量检索。

检索路径采用层级入口：优先利用会话运行状态和主题/核心实体定位 Experience，读取其摘要及相关 Segment；当路由置信不足或层级结果不足时，再进入 QA 混合检索回退。存在目标 Experience 时也支持按指定 ID 检索。实验说明记录了只读检索适配器，检索本身不新建 Experience。其目标是在常见查询中利用层级摘要控制证据规模，同时保留从摘要下探到原始 QA 的能力。

因此，HESM 当前较鲜明的定位是**面向连续用户任务的层级化经历记忆与证据回溯**：它用经历和阶段组织内容，而不是把所有记忆仅视为独立事实、平面笔记或时间图谱节点；并把摘要作为分层导航与压缩表示、把 QA 作为证据底层。相较于 A-MEM 的动态笔记网络、Zep 的显式时间关系图，HESM 选择较明确的父子层级和关系型主存储；相较于 MemoryOS 的多级缓存迁移，HESM 的层次围绕任务经历与意图阶段定义；相较于 OpenClaw 的可读文件工作区，HESM 的核心记忆以 SQLite 结构化存储并提供服务/API。

## 5. 参考文献与项目资料

以下条目给出可核对的正式入口。提交论文前请按目标期刊/会议的参考文献格式导出完整作者、卷期、页码及 DOI，并再次核验版本信息。

1. Chhikara, P., Khant, D., Aryan, S., Singh, T., & Yadav, D. (2025). *Mem0: Building Production-Ready AI Agents with Scalable Long-Term Memory*. arXiv:2504.19413. https://arxiv.org/abs/2504.19413
2. Xu, W., Liang, Z., Mei, K., Gao, H., Tan, J., & Zhang, Y. (2025). *A-MEM: Agentic Memory for LLM Agents*. NeurIPS 2025. https://papers.neurips.cc/paper_files/paper/2025/file/19909c36f51abc4856b4560aff3d36d6-Paper-Conference.pdf
3. Kang, J., Ji, M., Zhao, Z., & Bai, T. (2025). *Memory OS of AI Agent*. EMNLP 2025, 25961–25970. https://aclanthology.org/2025.emnlp-main.1318/
4. Li, Z., et al. (2025). *MemOS: An Operating System for Memory-Augmented Generation (MAG) in Large Language Models*. arXiv:2505.22101. https://arxiv.org/abs/2505.22101
5. Rasmussen, P., Paliychuk, P., Beauvais, T., Ryan, J., & Chalef, D. (2025). *Zep: A Temporal Knowledge Graph Architecture for Agent Memory*. arXiv:2501.13956. https://arxiv.org/abs/2501.13956
6. OpenClaw. *Memory architecture; Memory overview; Memory configuration reference*. 官方文档，访问日期：2026-10-09. https://docs.openclaw.ai/concepts/memory-architecture ; https://docs.openclaw.ai/concepts/memory ; https://docs.openclaw.ai/reference/memory-config
7. HESM 项目内部资料：`docs/HESM_MEMORY_DESIGN_V2_CN.md`（设计审查/提案，须区分已实现与拟议内容）；`experiments/USER_MEMORY_GUIDE_CN.md`（当前实验适配和实现边界）。

