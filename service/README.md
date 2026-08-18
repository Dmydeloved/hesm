# HESM Service

该目录是独立的后端服务层：

- `hesm_service.py`：组装核心 HESM 组件，提供写入、检索和对话应用服务。
- `server.py`：FastAPI 接口、管理仓储和静态前端托管入口。
- `export_memory_snapshot.py`：从 SQLite 导出只读前端快照的工具。

核心记忆逻辑位于 `hesm/`，页面与静态资源位于 `frontend/`。

从项目根目录启动：

```powershell
python -m service.server
```
