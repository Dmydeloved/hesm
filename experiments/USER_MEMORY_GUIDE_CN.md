# HESM × OmniMemEval User Memory 实验说明

## 1. 实验范围

本实验只包含三个基准：

- LoCoMo
- LongMemEval-S
- BEAM（100k、500k、1m、10m）

每个基准均执行 OmniMemEval 原有的六阶段流水线：ingestion、search、responses、eval、metric、report。HESM Adapter 只负责满足 add/search 接口、数据隔离、请求记录和错误传播；记忆添加与检索仍由 HESM Core 完成。

## 2. 固定模型

| 用途 | 模型 |
|---|---|
| HESM topic extraction | `gpt-4.1-mini-2025-04-14` |
| HESM summarization | `gpt-4.1-mini-2025-04-14` |
| HESM chat | `gpt-4.1-mini-2025-04-14` |
| HESM embedding | `text-embedding-v4` |
| OmniMemEval answer | `gpt-4.1-mini-2025-04-14` |
| OmniMemEval judge | `gpt-4o-mini-2024-07-18` |

启动器会拒绝与上表不一致的模型配置，并把实际模型写入 `hesm_manifest.json`。

## 3. 配置边界

`experiments/config/user_memory.yaml` 是独立实验配置，不继承 `config/hesm.yaml`。它控制 HESM 模型、检索参数、记忆管理参数和实验存储根目录。

YAML 使用 `${ENV_NAME}` 读取 `experiments/.env.user_memory` 中的凭据：

- `HESM_EVAL_TOKEN`
- `HESM_EVAL_MEMORY_API_KEY`
- `HESM_EVAL_MEMORY_BASE_URL`
- `HESM_EVAL_EMBEDDING_API_KEY`
- `HESM_EVAL_EMBEDDING_BASE_URL`

OmniMemEval 的 Answer/Judge 模型和凭据也放在 `.env.user_memory`，但不属于 HESM 配置。参考模板为 `experiments/env.user_memory.example`。

## 4. Embedding 输入上限

`text-embedding-v4` 的单条上限按 8192 Token 管理：

```yaml
embedding:
  max_input_tokens: 8192
  chunk_tokens: 7800
```

短内容保持原样调用。超过上限的 embedding 文本按字符边界拆成不超过 7800 Token 的块，分别编码后进行均值归一化，避免单次请求超过 8192；原始记忆正文不会被截断或改写。日志只记录字符数、Token 数和块数，不记录正文。

## 5. 数据与并发隔离

```text
experiments/.runtime/user_memory/<dataset_id>/
├── requests.sqlite3
└── users/
    └── <sha256(user_id)>/
        ├── hesm.sqlite3
        └── chroma/
```

- 不同数据集/版本使用不同 `dataset_id`。
- 同一数据集内按 `user_id` 隔离 HESM SQLite 和 Chroma。
- 写锁按 `user_id` 建立：同一用户串行，不同用户可并行。
- `requests.sqlite3/request_log` 只记录请求哈希、状态、耗时、错误类型和幂等命中，不保存请求正文。
- add 调用原生 `HESMService.add_memory()`；search 调用只读版原生层级检索，不在检索时创建 Experience。

## 6. 准备数据

在 `D:\code\OmniMemEval` 下执行：

```powershell
# LongMemEval-S，约 265 MB
python data\longmemeval\prepare_longmemeval.py

# BEAM 先准备 100k；正式四档实验再下载 all
python data\beam\prepare_beam.py --scale 100k
python data\beam\prepare_beam.py --scale all
```

默认正式数据位置：

- `data/locomo/locomo10.json`
- `data/longmemeval/longmemeval_s_cleaned.json`
- `data/beam/beam_100k.json`
- `data/beam/beam_500k.json`
- `data/beam/beam_1m.json`
- `data/beam/beam_10m_10m.json`

## 7. 只检查配置

在 `D:\code\hesm` 下执行；不加 `--live` 不会访问模型接口：

```powershell
python -B -m experiments.preflight `
  --config experiments\config\user_memory.yaml `
  --env experiments\.env.user_memory
```

需要真实检测模型端点时才添加 `--live --direct-network`。

## 8. 小数据验证

先生成开发用数据。LoCoMo 保留官方数据中的一条完整 conversation；LME 和 BEAM 是只验证接口/字段/六阶段流程的合成样本，不能作为正式分数：

```powershell
python -B -m experiments.smoke_data `
  --locomo-source D:\code\OmniMemEval\data\locomo\locomo10.json `
  --conversation-index 0
```

公共参数：

```text
--omnimemeval D:\code\OmniMemEval
--eval-python C:\ProgramData\anaconda3\envs\omnimemeval\python.exe
--service-python C:\ProgramData\anaconda3\python.exe
--workers 1 --llm-workers 1 --direct-network
```

LoCoMo：

```powershell
python -B -m experiments.run_user_memory `
  --benchmark locomo --version smoke_v1 --smoke `
  --data D:\code\hesm\experiments\.runtime\smoke_data\locomo.json `
  --omnimemeval D:\code\OmniMemEval `
  --eval-python C:\ProgramData\anaconda3\envs\omnimemeval\python.exe `
  --service-python C:\ProgramData\anaconda3\python.exe `
  --workers 1 --llm-workers 1 --direct-network
```

LongMemEval：

```powershell
python -B -m experiments.run_user_memory `
  --benchmark lme --version smoke_v1 --smoke `
  --data D:\code\hesm\experiments\.runtime\smoke_data\lme.json `
  --omnimemeval D:\code\OmniMemEval `
  --eval-python C:\ProgramData\anaconda3\envs\omnimemeval\python.exe `
  --service-python C:\ProgramData\anaconda3\python.exe `
  --workers 1 --llm-workers 1 --direct-network
```

BEAM：

```powershell
python -B -m experiments.run_user_memory `
  --benchmark beam --scale 100k --version smoke_v1 --smoke `
  --data D:\code\hesm\experiments\.runtime\smoke_data\beam `
  --omnimemeval D:\code\OmniMemEval `
  --eval-python C:\ProgramData\anaconda3\envs\omnimemeval\python.exe `
  --service-python C:\ProgramData\anaconda3\python.exe `
  --workers 1 --llm-workers 1 --direct-network
```

## 9. 正式实验

小数据的六阶段均成功后，换新 `--version`，移除 `--smoke` 和 `--data`。

```powershell
# LoCoMo
python -B -m experiments.run_user_memory --benchmark locomo --version full_v1 `
  --omnimemeval D:\code\OmniMemEval `
  --eval-python C:\ProgramData\anaconda3\envs\omnimemeval\python.exe `
  --service-python C:\ProgramData\anaconda3\python.exe `
  --workers 4 --llm-workers 4 --direct-network

# LongMemEval-S
python -B -m experiments.run_user_memory --benchmark lme --version full_v1 `
  --omnimemeval D:\code\OmniMemEval `
  --eval-python C:\ProgramData\anaconda3\envs\omnimemeval\python.exe `
  --service-python C:\ProgramData\anaconda3\python.exe `
  --workers 4 --llm-workers 4 --direct-network

# BEAM：分别执行 100k、500k、1m、10m
python -B -m experiments.run_user_memory --benchmark beam --scale 100k --version full_v1 `
  --omnimemeval D:\code\OmniMemEval `
  --eval-python C:\ProgramData\anaconda3\envs\omnimemeval\python.exe `
  --service-python C:\ProgramData\anaconda3\python.exe `
  --workers 4 --llm-workers 4 --direct-network
```

失败后检查对应 `hesm_step_<N>.log`。确认原因并保持同一配置后，可用 `--from-step N` 继续；如果数据、模型、参数或源码发生变化，应使用新的 `--version`，避免混合结果。

## 10. 指标与结果

| 基准 | 主指标 | 效率指标 |
|---|---|---|
| LoCoMo | LLM-as-a-Judge Accuracy | 平均 Answer 阶段 Context Tokens |
| LongMemEval | LLM-as-a-Judge Accuracy | 平均 Answer 阶段 Context Tokens |
| BEAM | Nugget Score | 平均 Answer 阶段 Context Tokens |

结果位于 `D:\code\OmniMemEval\results\<benchmark>\hesm-<version>\`。重点文件包括：

- `hesm_manifest.json`：数据哈希、配置指纹、固定模型、阶段命令和执行状态
- `hesm_execution.json`：每阶段退出码、耗时和日志位置
- `hesm_step_<N>.log`：阶段日志
- `hesm_*_grades.json` / `hesm_*_results.xlsx`：指标结果
- Markdown report：最终报告
