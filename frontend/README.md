# HESM Web Console

这是 HESM 的本地 Web 管理与检索界面，由同一个 FastAPI WebServer 提供 API 和静态页面。

## 启动

在项目根目录执行：

```powershell
python -m frontend.server
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
2. `HybridRetriever` 执行 Experience → Segment → QA 层级召回；
3. 展示最终证据树和注入模型的上下文。

耗时面板展示主题提取、查询向量、SQLite 召回、Chroma 召回、候选评分、候选树构建、上下文裁剪、LLM 重排、结果处理和响应组装等真实指标。

智能对话页面执行“主题提取 → HESM 检索 → Prompt 拼接 → LLM 回答 → 记忆存储”。每轮成功回答会作为 QA 写入 HESM，并在 `tools_json` 中保留完整 Prompt、会话历史、主题结果、召回树、检索诊断、生成模型和耗时数据。

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
- `GET /api/sessions`
- `POST /api/sessions`
- `GET/PATCH/DELETE /api/sessions/{session_id}`

服务端对页面与接口统一使用 `Cache-Control: no-store`，静态服务不会返回协商缓存 304；页面资源使用绝对路径，并提供 `/favicon.ico`，避免从嵌套路由访问时产生静态资源 404。

状态管理使用软状态更新，不物理删除层级数据。
