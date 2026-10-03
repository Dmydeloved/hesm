# Evidence 主题/实体替换实验

这是一版检索消融实验：检索请求必须携带数据集提供的 `evidence`，HESM 不在 query time 调用主题抽取模型，而是从本次实验用户自己的 `hesm.sqlite3.qa_memory` 中读取 evidence 对应的 ingestion-time `topic/core_entity/entities/intent`，再进入原有 HybridRetriever。向量召回、SQLite 召回、RRF 和层级组装均不变。

实验使用独立配置和端口：

- 配置：`experiments/config/evidence_topic_retrieval.yaml`
- 存储根目录：`experiments/.runtime/evidence_topic_retrieval/<dataset_id>/`
- 用户库：`users/<sha256(user_id)>/hesm.sqlite3` 和对应 Chroma 目录
- 结果目录：OmniMemEval 的 `results/<benchmark>/hesm-<version>/`

必须从 ingestion 阶段重新运行一次，不能复用普通 HESM 版本的数据库：LoCoMo ingestion 会把 `D{session}:{turn}` 写入每条 HESM QA 的 `source_id`，搜索阶段再把题目的 `evidence` 传入两个 speaker store。缺少 evidence 或映射时实验会失败，不会退回 LLM 主题抽取。

运行方式与普通 User Memory 实验相同，只替换配置文件和版本名：

```powershell
python -B -m experiments.run_user_memory `
  --benchmark locomo `
  --config experiments/config/evidence_topic_retrieval.yaml `
  --env experiments/.env.user_memory `
  --version evidence_topic_lookup_v1 `
  --omnimemeval D:\code\OmniMemEval `
  --workers 1 --llm-workers 2
```

服务健康信息会包含 `evidence_topic_lookup`、`evidence_required` 和 `native_qa_topic_entity_source` 能力标记；每条 search 响应也记录 `evidence` 和 `topic_extraction_source`，便于审计是否真的绕过了 query-time 抽取。
