# HESM

[English](README.md)

HESM 是一个面向对话智能体的分层情景记忆系统。它把结构化主题抽取结果组织成 Experience、Segment、QA 三层记忆，并结合结构化过滤、向量召回、LLM 重排和运行时缓存，为后续对话检索可追溯的历史证据。

## 记忆层级

```text
Experience（长期 topic + core_entity / 目标）
└── Segment（intent / 子目标）
    └── QA（原始用户输入—助手输出证据）
```

- `Experience`：按 `topic` 和 `core_entity` 聚合同一长期目标。
- `Segment`：按 `intent` 聚合同一目标下连续推进的阶段。
- `QA`：保存原始用户输入、助手输出、实体、置信度、时间戳和推理信息。

## 检索流程

`HybridRetriever.recall()` 按 Experience → Segment → QA 的顺序分层检索：

1. 构造一次查询文本。
2. 每次 `recall()` 只计算一次 query embedding。
3. Experience、Segment、QA 三层向量检索复用同一个查询向量。
4. 将 SQLite 中的结构化候选与 Chroma 向量召回结果结合。
5. 使用检索 LLM 在每一层选择候选并打分。
6. 返回结构化结果以及可直接提供给生成模型的 `context_text`。

## 项目结构

| 路径 | 说明 |
| --- | --- |
| `memory/` | 主题抽取、分层记忆管理、embedding、存储、检索与摘要 |
| `prompts/` | 主题抽取、检索和摘要提示词模板 |
| `configs/config.yaml` | 默认路径、模型、接口地址、重试策略和检索数量配置 |
| `scripts/` | MultiWOZ 预处理、主题抽取、记忆构建和交互流程脚本 |
| `experiments/` | AAAI 实验框架、正式 API 配置、指标和聚合脚本 |
| `evaluation/` | 评测辅助代码与测试 |
| `tests/` | 单元测试和本地集成测试 |
| `data/` | 原始/处理后数据、记忆存储和 benchmark 输出 |

## 正式 AAAI 实验流程

正式实验入口在 `experiments/runner.py`。当前流程以 dialogue 为单位推进，每个 dialogue 内按轮次执行：

1. 用户输入进入在线主题提取。
2. 主题提取时携带最近前 5 轮上下文，每轮包含原始 `user_input`、原始 `assistant_output` 和该轮 `topic_extraction` 结果。
3. 用当前轮在线抽取的主题结果进行 HESM 检索。
4. 调用生成模型回答。
5. 调用裁判模型，将模型回答与数据集原答案对比。
6. 将当前轮写入该 dialogue 的结构化数据库和向量库，供后续轮次使用。

这意味着第 6 轮提取主题时，会带上最近前 5 轮的主题结果以及原始输入输出。

## 正式跑一个 dialogue 的测试命令

下面命令是真实 API 流程，不是离线流程：

```powershell
cd D:\code\hesm
$env:DASHSCOPE_API_KEY="你的百炼Embedding Key"
$env:EVALUATION_API_KEY="你的评测/生成/主题抽取模型Key"
python -m experiments.runner --config experiments\configs\multiwoz_api_smoke.yaml --reset-output
```

`multiwoz_api_smoke.yaml` 已设置为只跑 1 个完整 dialogue：

- `dialogue_limit: 1`
- 在线主题提取：`topic_extraction.enabled: true`
- 在线 embedding：`embedder: bailian`
- 在线 LLM 重排：`rerank: llm`
- 在线回答生成：`generation.enabled: true`
- 在线裁判：`judge.enabled: true`

第一次正式测试建议加 `--reset-output`，它会清除该配置对应的旧输出目录，避免历史实验数据造成干扰。

## 中断恢复

如果实验中途意外中断，恢复时不要加 `--reset-output`：

```powershell
cd D:\code\hesm
$env:DASHSCOPE_API_KEY="你的百炼Embedding Key"
$env:EVALUATION_API_KEY="你的评测/生成/主题抽取模型Key"
python -m experiments.runner --config experiments\configs\multiwoz_api_smoke.yaml
```

恢复粒度是 dialogue：

- 已完整完成的 dialogue 会写入 `completed_dialogues.jsonl`。
- 恢复时会跳过已完成 dialogue。
- 如果某个 dialogue 中途断掉，它不会被标记完成；恢复时从这个 dialogue 开始重新跑。
- 每个 dialogue 的 SQLite 和 Chroma 状态库都隔离在 `state/<dialogue_id>/` 下。

## 正式全量实验命令

单 dialogue 测试确认正常后，运行全量 API 配置：

```powershell
cd D:\code\hesm
$env:DASHSCOPE_API_KEY="你的百炼Embedding Key"
$env:EVALUATION_API_KEY="你的评测/生成/主题抽取模型Key"
python -m experiments.runner --config experiments\configs\multiwoz_api.yaml --reset-output
```

如果全量实验中断，恢复时去掉 `--reset-output`：

```powershell
python -m experiments.runner --config experiments\configs\multiwoz_api.yaml
```

## 实验产出

单 dialogue 正式测试默认输出目录：

```text
results/experiments/multiwoz_api_smoke_hesm/
```

全量正式实验默认输出目录：

```text
results/experiments/multiwoz_api_hesm/
```

主要产出包括：

- `predictions.jsonl`：每轮主题结果、检索结果、模型回答、原答案、裁判结果和 debug 信息。
- `metrics.json`：整体和分类型指标，包括 `answer_f1`、`evidence_recall`、`mrr`、延迟、检索 token 数、`judge_score`。
- `run.log`：进度日志，包括 dialogue 开始/完成、主题抽取、检索进度和最终产物路径。
- `node_inputs.jsonl`：关键节点输入输出，包括主题提取输入、主题提取输出、检索输入、生成输入、裁判输入和单轮输出。
- `completed_dialogues.jsonl`：已完成 dialogue 清单，用于中断恢复。
- `state/<dialogue_id>/memory.sqlite3`：该 dialogue 的结构化 SQLite 记忆库。
- `state/<dialogue_id>/chroma/`：该 dialogue 的 Chroma 向量库。
- `config.yaml`：本次运行使用的配置快照。
- `manifest.json`：Python、平台、Git commit、seed 等复现实验信息。

## 清理旧实验数据

推荐方式是在正式运行时使用 `--reset-output`：

```powershell
python -m experiments.runner --config experiments\configs\multiwoz_api_smoke.yaml --reset-output
python -m experiments.runner --config experiments\configs\multiwoz_api.yaml --reset-output
```

如果需要手工删除对应输出目录：

```powershell
Remove-Item -LiteralPath results\experiments\multiwoz_api_smoke_hesm -Recurse -Force
Remove-Item -LiteralPath results\experiments\multiwoz_api_hesm -Recurse -Force
```

注意：恢复中断实验时不要清理输出目录，也不要加 `--reset-output`，否则已完成 dialogue 的 checkpoint 会被删除。

## 环境要求

- Python 3.10+
- `openai`
- `PyYAML`
- `chromadb`

安装依赖示例：

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install openai PyYAML chromadb
```
