# HESM 实验报告模板

状态：`not_run`。所有 `—` 均为未测；不得解释为 0 或不支持。

## 复现信息

| 字段 | 值 |
|---|---|
| study_id / run_id | — |
| HESM / OmniMemEval commit | — |
| 数据 revision / SHA-256 / task IDs | — |
| Agent 协议（固定五域 / 最新四域） | — |
| OpenClaw / 插件版本 | — |
| Answer / Judge / Memory 模型完整 ID | — |
| Prompt hash / tokenizer / 检索预算 | — |
| 训练与测试重复次数 / 顺序 | — |
| 基础设施 / 向量后端 / 并发 / 超时 | — |
| 实际费用 / 训练与写入消耗 | — |

## User Memory 主实验

| 方法 | LoCoMo Acc | Context Tokens | LME Overall | Knowledge Update | Multi-Session | Temporal Reasoning | LME Context Tokens |
|---|---:|---:|---:|---:|---:|---:|---:|
| HESM | — | — | — | — | — | — | — |
| Flat RAG | — | — | — | — | — | — | — |
| Mem0 | — | — | — | — | — | — | — |
| Supermemory | — | — | — | — | — | — | — |
| Hindsight | — | — | — | — | — | — | — |

精度统一用百分比；Context Tokens 指完整回答模型输入。每个指标附 CI、预定 / 已运行 / 有效题数。公开历史分数放到另一张带来源与协议差异的表。

## BEAM 规模实验

每个方法填写四行，补充按能力维度分解的结果。

| 方法 | 规模 / CLI 标签 | Nugget Score | Context Tokens | 截断率 | 检索 p95 ms | 写入 tokens / 时间 | 峰值 RAM | 索引体积 | 完整历史写入率 |
|---|---|---:|---:|---:|---:|---|---:|---:|---:|
| HESM | 128K / 100k | — | — | — | — | — | — | — | — |
| HESM | 500K / 500k | — | — | — | — | — | — | — | — |
| HESM | 1M / 1m | — | — | — | — | — | — | — | — |
| HESM | 10M / 10m | — | — | — | — | — | — | — | — |

官方不同规模集合与固定证据的受控扩容分别报告，不混用配对统计。

## AgentBench 主实验

每种方法填写五域，并保留 reward、token、错误率和成功子集轮数。

| 方法 | domain | Acc | Avg Turns 全任务 | Avg Turns 双方均成功 | Avg Chars | Reward | Token / task | 有效 / 预定任务 | Infra failures |
|---|---|---:|---:|---:|---:|---:|---:|---|---:|
| No Memory | — | — | — | — | — | — | — | — | — |
| HESM w/o Experience | — | — | — | — | — | — | — | — | — |
| HESM Full | — | — | — | — | — | — | — | — | — |
| Raw Trajectory Memory | — | — | — | — | — | — | — | — | — |

Acc 是多次单次执行平均成功率，不是 pass@3。宏平均与按 task 数加权的微平均分别展示。

## 消融与假设判定

| 假设 | 对照与数据 | 差值 / 比值 | CI | 预注册边界 | 判定 |
|---|---|---|---|---|---|
| 相近精度、更少上下文 | — | — | — | — | not_run |
| State 改善更新识别 | — | — | — | — | not_run |
| Hierarchical 保持规模质量 | — | — | — | — | not_run |
| Experience 提升 Acc、减少 Turns | — | — | — | — | not_run |

判定只用 supported / not_supported / inconclusive / not_run，并解释适用范围。补充样本隔离、只读校验、快照一致性、开发集使用、缺失样本、异常处理与协议偏离；结论指向原始记录和报告路径。
