# HESM Memory Manager 前端原型

该目录包含 HESM 存量记忆管理界面和真实记忆检索界面。

## 页面功能

- 浏览完整 Experience 列表，查看主题、核心实体、版本、Segment 数量和 QA 数量；
- 按主题、实体、ID 搜索 Experience，并筛选当前 Experience 或有总结的 Experience；
- 查看 Experience 总结、运行状态、意图演进和 Segment 序列；
- 选择 Segment 后查看阶段总结、状态、版本、时间及其 QA 列表；
- 搜索 QA 内容和实体标签，并打开 QA 详情查看原始输入、助手输出、Topic、Intent、实体、置信度和判断依据；
- 一键定位 `runtime_state` 中的当前 Experience 和 Segment。

## 数据快照

页面使用 [memory-data.js](memory-data.js) 中的完整静态快照，来源为：

```text
memory/hesm.sqlite3
```

当前快照包含：

- 61 个 Experience
- 250 个 Segment
- 489 条 QA

页面只读取前端快照，不会修改 SQLite 数据库。

## 启动

需要使用真实检索时，在 HESM 项目根目录启动本地 API 与静态服务：

```powershell
python -m frontend.server
```

然后访问：

- 记忆管理：`http://127.0.0.1:8080/index.html`
- 记忆检索：`http://127.0.0.1:8080/retrieval.html`

检索接口会真实执行：

```text
用户问题
→ TopicExtractor.extract
→ HybridRetriever.recall
→ Experience → Segment → QA 来源树
```

服务对外提供两个写实接口，均使用 `config/hesm.yaml` 和 `memory/`：

- `POST /api/memories`：添加一条交互记忆；
- `POST /api/retrieve`：检索层级记忆。

添加记忆示例：

```json
{
  "user_input": "Alice 搬到了巴黎。",
  "assistant_output": "已记录。",
  "state_key": "alice"
}
```

仅浏览离线管理快照时，也可以直接打开 `index.html`，或启动普通静态服务：

可以直接打开 `index.html`，也可以在 HESM 项目根目录启动静态服务：

```powershell
python -m http.server 8080 -d frontend
```

普通静态服务不提供 `/api/retrieve`，因此不能执行真实检索。

## 重新生成数据快照

SQLite 数据变化后，在项目根目录运行：

```powershell
python frontend/export_memory_snapshot.py memory/hesm.sqlite3 frontend/memory-data.js
```

该脚本以 SQLite 只读模式打开数据库，并重新生成前端可直接加载的完整嵌套数据：

```text
Experience
└── Segment
    └── QA
```

## 后续接入后端

接入真实管理接口时，可以保留当前页面结构和渲染逻辑，将 `memory-data.js` 替换为 API 数据源。若需要支持新增、修改、合并或删除记忆，应在后端增加相应写接口，并在界面中加入权限确认和操作审计。
