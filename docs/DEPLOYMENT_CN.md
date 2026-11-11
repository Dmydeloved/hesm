# HESM 项目部署指南

本文说明如何在单机 Windows 或 Linux 主机上部署 HESM Web 服务。服务由 FastAPI 提供接口，SQLite 保存权威记忆数据，Chroma 保存向量索引，前端静态文件由同一服务托管。

## 1. 部署前准备

- Python 3.11 或更新的 3.x 版本。
- 可访问模型服务的网络，以及主题抽取、摘要/对话和向量模型对应的 API 凭据。
- 项目目录所在磁盘具有写入权限；SQLite、Chroma、日志和配置文件需要持久化存储。
- 生产环境应由进程管理器（如 systemd、Windows 服务管理器或容器编排器）守护进程，并通过反向代理提供 HTTPS。

SQLite 需要支持 FTS5 才能启用 QA 混合全文检索；核心记忆读写和向量检索仍以 SQLite 与 Chroma 为主。部署前可运行 `python -c "import sqlite3; print('ENABLE_FTS5' in [row[0] for row in sqlite3.connect(':memory:').execute('pragma compile_options')])"` 检查当前 SQLite 构建。

## 2. 安装运行依赖

依赖版本已固定在项目根目录的 requirements.txt 中。在项目根目录创建隔离环境并按清单安装：

```powershell
# Windows PowerShell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

```bash
# Linux
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

依赖清单已包含 tiktoken，用于文本分块计数。清单中的依赖项若与目标操作系统不兼容，应先调整该项或使用对应平台的依赖清单，再执行安装。

## 3. 准备独立配置

不要把真实 API 密钥写入 Git 跟踪的默认配置。复制 `config/hesm.production.example.yaml` 到受限访问的部署目录，填入三类模型服务的凭据，并设置数据库与向量库的持久化路径。例如：

```powershell
$env:HESM_CONFIG = 'D:\hesm-data\hesm.yaml'
```

```bash
export HESM_CONFIG=/srv/hesm-data/hesm.yaml
```

`HESM_CONFIG` 应指向可读的 YAML 文件。配置中的相对 `paths.memory_db` 和 `paths.chroma` 路径相对于项目根目录解析；生产环境建议使用绝对路径，避免工作目录变化导致数据写入其他位置。确保目标目录由运行 HESM 的账户创建并可写。

配置中的三个凭据分别用于 `topic_extraction`、`summarization`（也作为未单独配置时的 `chat` 默认值）和 `embedding`。如聊天模型使用不同凭据或模型，可单独设置 `chat` 节。API 密钥应通过受限权限的配置文件或部署平台密钥管理功能注入，不要写入日志、文档或代码仓库。

## 4. 启动服务

在项目根目录运行：

```powershell
python -m service.server
```

直接运行入口默认监听 `http://127.0.0.1:8080`。如需让反向代理或容器访问，可使用 Uvicorn 参数覆盖监听地址与端口：

```powershell
python -m uvicorn service.server:app --host 0.0.0.0 --port 8080
```

服务启动时会创建或迁移 SQLite schema，并启动后台派生任务处理器；SQLite 文件、Chroma 数据目录和 `logs/` 会在运行时生成。请将 Web Console 放在可信网络或受认证的反向代理之后；服务管理接口不应直接暴露到不受信任的公网。

## 5. 验证部署

浏览器访问以下页面检查静态资源是否可用：

- `http://127.0.0.1:8080/`：记忆管理控制台。
- `http://127.0.0.1:8080/retrieval.html`：检索调试页面。
- `http://127.0.0.1:8080/chat.html`：对话页面。
- `http://127.0.0.1:8080/operations.html`：操作轨迹页面。

首次启动只验证服务和数据库初始化时，不要求先创建记忆数据。真实写入/对话请求会调用配置的模型供应商，并产生 API 费用。运行日志写入项目根目录 `logs/hesm-YYYY-MM-DD.log`，保留最近 30 天；检查该日志中的启动异常、模型调用错误和后台任务失败信息。

## 6. 数据备份与恢复

- 停止服务后，备份配置的 SQLite 数据库文件和整个 Chroma 持久化目录；两者需要成对备份以保持向量索引与关系记录一致。
- 同时备份部署配置和所需密钥（使用安全的密钥管理流程），不要将密钥与普通数据库备份放在公开位置。
- 恢复时先停止服务，将 SQLite 与 Chroma 恢复到配置指定位置，再启动服务。启动期间会执行兼容性迁移。
- 定期在隔离环境验证备份可读且服务可启动；不要在生产数据库上试验恢复流程。

## 7. 升级与运维

1. 记录当前版本并完整备份 SQLite、Chroma 和配置。
2. 在新版本代码对应的隔离环境安装依赖。
3. 先在备份副本上启动服务，确认数据库迁移、页面访问和一次受控读写正常。
4. 切换到新版本并重启服务；保留可回退的旧代码和升级前数据备份。
5. 监控 `logs/`、磁盘空间和模型服务可用性，并定期清理或归档过期日志与备份。

数据库表结构、配置字段和接口行为详见[系统实现说明](HESM_SYSTEM_IMPLEMENTATION_CN.md)。
