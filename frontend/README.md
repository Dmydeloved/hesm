# HESM Web Console

这是 HESM 的纯静态前端。FastAPI 后端位于独立的 `service/` 包，并负责提供 API 和托管本目录中的页面资源。

## 启动

在项目根目录执行：

```powershell
python -m service.server
```

然后访问：

- 数据管理：<http://127.0.0.1:8080/index.html>
- 检索分析：<http://127.0.0.1:8080/retrieval.html>
- 智能对话：<http://127.0.0.1:8080/chat.html>
- OpenAPI：<http://127.0.0.1:8080/docs>

## 页面能力

数据管理页面直接读取 `config/hesm.yaml` 指向的 SQLite 数据库，不再依赖静态快照。支持 Experience、Segment、QA 的搜索、分页、父子范围筛选、完整详情与状态管理。

检索分析页面执行真实的 HESM 流程：

1. `TopicExtractor` 提取主题、核心实体、意图和相关实体；
2. `HybridRetriever` 按主题和核心实体定位当前 Experience，并读取最近的 Segment 与 QA；
3. 展示最终证据树和注入模型的上下文。

耗时面板展示主题提取、Experience/上下文查询和响应组装的真实耗时。

智能对话页面执行“主题提取 → HESM 检索 → Prompt 拼接 → LLM 回答 → 记忆存储”。前端只提交 `session_id` 和当前消息，服务端会从 `chat_session` 中读取最近 5 轮历史用于主题提取；最终回答 Prompt 仍只使用 HESM 检索内容和当前问题。每轮成功回答会作为 QA 写入 HESM；`tools_json` 只保存真实工具调用，不混入检索诊断。

聊天历史由独立的 `chat_session` 表管理，并保留浏览器本地缓存作为离线回退。刷新页面后会通过 `session_id` 自动恢复；“新建会话”会生成独立的会话标识，页面左侧可以切换或归档已保存会话。该表只负责会话历史，不改变 Experience、Segment、QA 的业务逻辑。

## 主要接口

- `GET /api/health`
- `GET /api/stats`
- `GET /api/{experience|segment|qa}`
- `GET /api/{experience|segment|qa}/{id}`
- `PATCH /api/{experience|segment|qa}/{id}/status`
- `POST /api/memories`
- `POST /api/retrieve`
- `POST /api/chat`
- `POST /api/chat/stream`
- `GET /api/sessions`
- `POST /api/sessions`
- `GET/PATCH/DELETE /api/sessions/{session_id}`

服务端对页面与接口统一使用 `Cache-Control: no-store`，静态服务不会返回协商缓存 304；页面资源使用绝对路径，并提供 `/favicon.ico`，避免从嵌套路由访问时产生静态资源 404。

状态管理使用软状态更新，不物理删除层级数据。
