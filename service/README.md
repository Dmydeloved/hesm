# HESM Service

该目录是独立的后端服务层：

- `hesm_service.py`：组装核心 HESM 组件，提供写入、检索和对话应用服务。
- `server.py`：FastAPI 接口、管理仓储和静态前端托管入口。
- `export_memory_snapshot.py`：从 SQLite 导出只读前端快照的工具。

核心记忆逻辑位于 `core/`，页面与静态资源位于 `frontend/`。

从项目根目录启动：

```powershell
python -m service.server
```

运行日志统一写入项目根目录的 `logs/`，文件名格式为
`hesm-YYYY-MM-DD.log`。服务跨过午夜后会自动切换到新日期文件，并保留最近
30 天日志。每条日志包含时间、等级、源文件名、行号、logger 名称和消息。
