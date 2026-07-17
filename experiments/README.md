# HESM 正式实验说明

本目录是 AAAI 实验部分的运行入口。当前正式流程按 dialogue 顺序执行，每个 dialogue 内部逐轮推进，确保第 N 轮只能看到前面已经发生的记忆。

## 当前正式实验流程

每一轮用户输入都会按下面顺序处理：

1. 在线主题提取：调用配置中的 `topic_extraction` API。
2. 主题提取上下文：最多携带最近前 5 轮，每轮包含：
   - 原始 `user_input`
   - 原始 `assistant_output`
   - 该轮 `topic_extraction` 结果
3. 分层检索：用当前轮在线抽取出的主题结果进入 HESM 检索。
4. 模型回答：把检索上下文交给生成模型回答。
5. 自动裁判：把模型回答和数据集原答案进行对比评分。
6. 写入结构化数据库：当前轮原始输入、原始输出和主题结果写入该 dialogue 对应的 SQLite 与 Chroma 状态库，供后续轮次检索。

因此，第 6 轮做主题提取时，会携带最近前 5 轮的原始输入、原始输出和主题提取结果；不会只传当前用户输入。

## 正式跑一个 dialogue 的测试命令

在 PowerShell 中进入项目目录：

```powershell
cd D:\code\hesm
$env:DASHSCOPE_API_KEY="你的百炼Embedding Key"
$env:EVALUATION_API_KEY="你的评测/生成/主题抽取模型Key"
python -m experiments.runner --config experiments\configs\multiwoz_api_smoke.yaml --reset-output
```

这个命令是真实 API 流程，不是离线流程。`multiwoz_api_smoke.yaml` 已设置：

- `dialogue_limit: 1`：只跑 1 个完整 dialogue。
- `topic_extraction.enabled: true`：在线调用主题提取模型。
- `embedder: bailian`：在线调用 embedding。
- `rerank: llm`：在线调用 LLM 检索重排。
- `generation.enabled: true`：在线生成答案。
- `judge.enabled: true`：在线裁判模型答案。

第一次正式测试建议加 `--reset-output`，它会先删除该配置对应的输出目录，避免旧实验数据干扰。

## 中断后的恢复方式

如果正式实验中途意外中断，恢复时不要加 `--reset-output`：

```powershell
cd D:\code\hesm
$env:DASHSCOPE_API_KEY="你的百炼Embedding Key"
$env:EVALUATION_API_KEY="你的评测/生成/主题抽取模型Key"
python -m experiments.runner --config experiments\configs\multiwoz_api_smoke.yaml
```

恢复逻辑以 dialogue 为单位：

- 已完整跑完的 dialogue 会写入 `completed_dialogues.jsonl`，恢复时会直接跳过。
- 如果某个 dialogue 中途断掉，它不会被标记为完成；恢复时会从这个 dialogue 开始重新跑。
- HESM 的 SQLite 和 Chroma 状态库也按 dialogue 隔离，路径在 `输出目录/state/<dialogue_id>/`。

## 正式全量实验命令

确认单 dialogue 测试正常后，再跑全量配置：

```powershell
cd D:\code\hesm
$env:DASHSCOPE_API_KEY="你的百炼Embedding Key"
$env:EVALUATION_API_KEY="你的评测/生成/主题抽取模型Key"
python -m experiments.runner --config experiments\configs\multiwoz_api.yaml --reset-output
```

如果全量实验中断，恢复时同样去掉 `--reset-output`：

```powershell
python -m experiments.runner --config experiments\configs\multiwoz_api.yaml
```

## 主要产出

以单 dialogue 测试为例，默认输出目录是：

```text
results/experiments/multiwoz_api_smoke_hesm/
```

主要文件包括：

- `predictions.jsonl`：每轮的主题结果、检索结果、模型回答、原答案、裁判结果和调试信息。
- `metrics.json`：整体和分类型指标，包括 `answer_f1`、`evidence_recall`、`mrr`、延迟、检索 token 数、`judge_score`。
- `run.log`：正式进度日志，包括 dialogue 开始/完成、主题抽取、检索进度和最终产物路径。
- `node_inputs.jsonl`：关键节点输入输出，包括主题提取输入、主题提取输出、检索输入、生成输入、裁判输入和单轮输出。
- `completed_dialogues.jsonl`：已完整完成的 dialogue 清单，用于中断恢复。
- `state/<dialogue_id>/memory.sqlite3`：该 dialogue 的结构化 SQLite 记忆库。
- `state/<dialogue_id>/chroma/`：该 dialogue 的 Chroma 向量库。
- `config.yaml`：本次运行使用的配置快照。
- `manifest.json`：Python、平台、Git commit、seed 等复现实验信息。

## 清理旧实验数据

推荐使用 `--reset-output` 清理当前配置对应的输出目录。比如单 dialogue 正式测试：

```powershell
python -m experiments.runner --config experiments\configs\multiwoz_api_smoke.yaml --reset-output
```

如果需要手工清理，可以删除对应输出目录：

```powershell
Remove-Item -LiteralPath results\experiments\multiwoz_api_smoke_hesm -Recurse -Force
Remove-Item -LiteralPath results\experiments\multiwoz_api_hesm -Recurse -Force
```

注意：恢复中断实验时不要清理输出目录，也不要加 `--reset-output`，否则已完成 dialogue 的 checkpoint 会被删除。
