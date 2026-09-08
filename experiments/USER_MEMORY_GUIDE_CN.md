# HESM × OmniMemEval User Memory 实验说明

## 1. 当前交付范围

本次已经完成 HESM 对 OmniMemEval User Memory 的接入，覆盖 LoCoMo、LongMemEval 和 BEAM。OmniMemEval 仍负责数据加载、问答生成、LLM Judge、指标汇总和报告生成；HESM 只负责写入长期记忆和返回检索上下文。

本轮没有运行正式基准，因此仓库内没有可报告的 HESM Accuracy、Nugget Score 或 Context Tokens。公开结果只作为后续横向比较的参照，不能标记为本次复现结果。OpenClaw / EvoAgentBench 不在本轮完成范围内。

## 2. 已补齐的能力

### 独立实验配置

- 唯一运行配置为 `experiments/config/user_memory.yaml`。
- HESM 数据写入 `experiments/.runtime/user_memory`，不会读写原有 `memory/`、生产 SQLite 或生产 Chroma 数据。
- 密钥只从被 Git 忽略的 `experiments/.env.user_memory` 读取。运行时不会隐式加载生产配置。
- 每个外部 `user_id` 会映射到独立的哈希 namespace；`delete_all` 只递增该用户的实验 generation，不会清理其他用户或生产数据。
- 服务端和 OmniMemEval client 双向校验 `config_fingerprint`，避免连到其他 HESM 实例。

### OmniMemEval Adapter

OmniMemEval 已注册 `--lib hesm`，并在原有三个 benchmark runner 中完成以下适配：

- LoCoMo：保留双说话人独立 memory user，传递 session ID 和会话日期。
- LongMemEval：按真实 session 日期写入，向检索层传递 `question_date`，支持 Knowledge Update、Multi-Session 和 Temporal Reasoning。
- BEAM：支持 `100k`、`500k`、`1m`、`10m`，数据目录可以显式覆盖。
- 三个 benchmark 均沿用 OmniMemEval 原始 responses、eval、metric 和 report 脚本，没有另写评分逻辑。

### HESM 评测后端

- 写入具备稳定 source ID 和 session 幂等性，失败的 session 会隔离，避免半成功结果被当成完整写入。
- 只读检索不调用会创建 Experience 的 `route_experience()` 路径。
- 检索按 Experience → Segment → QA 组织证据，并应用独立的上下文 token 上限。
- HESM 只接收对话正文、说话人、session 和时间。gold answer、证据定位标签等评分信息不会进入记忆。
- 使用真实 Chroma 持久化向量库；模型或 embedding 调用失败会直接暴露，不会静默退化为模板摘要或伪向量。
- 服务提供健康检查、能力声明、逻辑清空、finalize、模式切换、快照和恢复接口，供后续 Agent 生命周期复用。

## 3. 文件与职责

| 文件 | 职责 |
|---|---|
| `experiments/config/user_memory.yaml` | HESM User Memory 独立配置 |
| `experiments/settings.py` | 严格加载配置、环境变量和隔离检查 |
| `experiments/components.py` | 真实模型、embedding、Chroma 和评测专用组件 |
| `experiments/backend.py` | namespace、幂等写入、层级只读检索和生命周期 |
| `experiments/server.py` | 独立 FastAPI 服务 |
| `experiments/bootstrap.py` | 一次性生成独立环境文件，不覆盖已有文件 |
| `experiments/preflight.py` | 可选的依赖和服务探测 |
| `experiments/run_user_memory.py` | 调用 OmniMemEval 原始六阶段流水线 |
| `experiments/REPORT_TEMPLATE_CN.md` | 正式结果填写模板 |

OmniMemEval 侧的核心接入点是 `scripts/client_factory/hesm_client.py`，其余修改仅用于注册 client、传递 session/date 和允许显式数据路径。

## 4. 准备配置

在 HESM 根目录执行一次：

```powershell
python -B -m experiments.bootstrap --source-credentials config/hesm.yaml
```

该命令只在 `experiments/.env.user_memory` 不存在时创建它。也可以手工从 `experiments/env.user_memory.example` 创建独立环境文件。不得把密钥写入 YAML、manifest、报告或提交记录。

正式运行前，固定以下内容：

1. `user_memory.yaml` 中的 memory model、embedding model、检索预算和 HESM 开关。
2. `.env.user_memory` 中的 `ANSWER_MODEL` 和 `EVAL_MODEL`，使其与所比较的 OmniMemEval 公开结果协议一致。
3. 数据文件版本与 SHA-256；启动器会把实际 hash 写入结果 manifest。
4. 唯一的 `--version`。配置、数据或命令发生变化时使用新版本名，启动器拒绝在旧目录上混合续跑。

## 5. 正式运行入口

以下命令均从 HESM 根目录运行。`--eval-python` 指向安装了 OmniMemEval 依赖的解释器；`--service-python` 指向安装了 HESM、FastAPI、Chroma 和模型 SDK 的解释器。启动器会自动启动并关闭独立 HESM 服务。

```powershell
python -B -m experiments.run_user_memory --benchmark locomo --version main_v1 --omnimemeval D:\code\OmniMemEval --eval-python <OMNI_PYTHON> --service-python <HESM_PYTHON> --direct-network

python -B -m experiments.run_user_memory --benchmark lme --version main_v1 --omnimemeval D:\code\OmniMemEval --eval-python <OMNI_PYTHON> --service-python <HESM_PYTHON> --direct-network

python -B -m experiments.run_user_memory --benchmark beam --scale 100k --version main_v1 --omnimemeval D:\code\OmniMemEval --eval-python <OMNI_PYTHON> --service-python <HESM_PYTHON> --direct-network

python -B -m experiments.run_user_memory --benchmark beam --scale 10m --version main_v1 --omnimemeval D:\code\OmniMemEval --eval-python <OMNI_PYTHON> --service-python <HESM_PYTHON> --direct-network
```

`--direct-network` 仅用于主机存在无效代理时忽略继承的代理设置。正常网络环境可以去掉。先用 OmniMemEval 原始数据准备脚本生成数据；启动器不会下载或改写数据。

每个 benchmark 依次复用六个原始阶段：

```text
ingestion → search → responses → eval → metric → report
```

可以用 `--from-step` 和 `--to-step` 恢复某一段。`--smoke` 产生的目录会带 `smoke_` 前缀，只能用于功能检查，不能纳入论文或正式表格。

## 6. 结果与复现记录

结果写入 OmniMemEval：

```text
results/locomo/hesm-<version>_locomo/
results/lme/hesm-<version>_lme/
results/beam/hesm-<version>_beam_<scale>/
```

除 OmniMemEval 原始中间文件和报告外，每个目录还包含：

- `hesm_manifest.json`：数据 hash、有效配置 fingerprint、模型 ID、阶段命令和源码 hash。
- `experiment_config.sh`：OmniMemEval report 可直接读取的无密钥参数快照。
- `snapshot_eval.env`：只含允许公开的模型与服务标识，不含密钥和 base URL。
- `hesm_config_snapshot.json`：解析后的 HESM 实验配置，不含凭据值。
- `hesm_execution.json` 和 `hesm_step_*.log`：阶段退出码、耗时和日志。
- `hesm_service.log`：评测服务日志。

若 manifest 标记为 `partial` 或 `failed`，该目录不能作为正式结果。正式报告只使用六阶段全部完成的目录。

## 7. 指标解释与公开基线

| Benchmark | 主指标 | HESM 设计对应关系 |
|---|---|---|
| LoCoMo | Overall Accuracy、Context Tokens | 长对话记忆质量与层级压缩 |
| LongMemEval | Knowledge Update、Multi-Session、Temporal Reasoning | State 的更新与时间演化 |
| BEAM | 100K / 10M Nugget Score、Context Tokens | Hierarchical 在规模扩大时的质量与检索开销 |

横向比较直接引用 OmniMemEval 的已发布 reproduced 表，不重跑 Mem0、Supermemory、Hindsight 等产品。报告中应把两类结果分开标注：

- `HESM — this run`：本次完整六阶段生成的结果。
- `Published reference — OmniMemEval`：公开页面中的历史结果，并记录页面链接和读取日期。

只有数据版本、问题集合、回答模型、judge、prompt 和统计口径一致时，才能把 HESM 与公开分数放入同一主表。若公开页面随后更新，应引用固定 commit 的结果文件；不要把不同 revision 的数字静默替换进旧报告。

OmniMemEval 主表中的 `Context Tokens` 是回答模型的完整输入 token，包含提示模板、问题和检索记忆。HESM 服务内部记录的检索文本 token 与模型处理 token 应作为成本附表，不能替代主表口径。

公开基线来源：

- User Memory Results：<https://github.com/MemTensor/OmniMemEval/blob/main/docs/user_memory/results.md>
- OmniMemEval User Memory 指南：<https://github.com/MemTensor/OmniMemEval/blob/main/docs/user_memory/README.md>

## 8. 实验结论边界

LoCoMo 的有效结论是“在相近 Accuracy 下是否减少 Context Tokens”。LongMemEval 应优先报告三项 State 切片，不能只用 Overall 替代。BEAM 的 100K 与 10M 官方集合若不是同题配对数据，只能做分层比较；还需要同时确认完整历史已写入，避免通过截断数据制造“tokens 不涨”。

目前只能声明 User Memory 接入和实验执行能力已经完成。正式数值结论必须等对应 benchmark 的完整运行目录和报告生成后再填写。
