# HESM Web Console

启动：

```powershell
python -m service.server
```

页面：

- `/index.html`：Experience、Segment、QA 数据管理
- `/retrieval.html`：检索分析
- `/chat.html`：智能对话
- `/operations.html`：Chat、Retriever、Add 以及 Segment/Experience 蒸馏轨迹

聊天页面使用 `chat_session.chats` 恢复会话，并从 `chat` 表读取消息。每轮 Chat 固定关联一个 Retriever 和一个 Add；异步记忆更新通过 `memory_outbox` 关联到 `segment_update`、`experience_update`。

主要接口：

- `GET /api/sessions`
- `POST /api/sessions`
- `GET /api/sessions/{session_id}`
- `POST /api/chat/stream`
- `GET /api/chats/{chat_id}/trajectory?session_id={session_id}`
- `GET /api/operations`
- `GET /api/operations/{type}/{id}`
