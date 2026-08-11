# HESM Memory Manager 前端原型

这是一个不连接后端接口的 HESM 存量记忆管理原型，用于查看已经写入的三级记忆结构，而不是执行记忆检索。

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
outputs/locomo/memory/hesm_conv-26/memory.sqlite3
```

当前快照包含：

- 61 个 Experience
- 250 个 Segment
- 489 条 QA

页面只读取前端快照，不会修改 SQLite 数据库。

## 启动

可以直接打开 `index.html`，也可以在 HESM 项目根目录启动静态服务：

```powershell
python -m http.server 8080 -d frontend
```

然后访问 `http://localhost:8080`。

## 重新生成数据快照

SQLite 数据变化后，在项目根目录运行：

```powershell
python frontend/export_memory_snapshot.py outputs/locomo/memory/hesm_conv-26/memory.sqlite3 frontend/memory-data.js
```

该脚本以 SQLite 只读模式打开数据库，并重新生成前端可直接加载的完整嵌套数据：

```text
Experience
└── Segment
    └── QA
```

## 后续接入后端

接入真实管理接口时，可以保留当前页面结构和渲染逻辑，将 `memory-data.js` 替换为 API 数据源。若需要支持新增、修改、合并或删除记忆，应在后端增加相应写接口，并在界面中加入权限确认和操作审计。
