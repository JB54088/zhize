# 知择 V0.1

知择是一个以来源和证据为基础的 AI 信息决策平台。V0.1 包含搜索首页、FastAPI 搜索任务 API、可配置的 Tavily 搜索引擎，以及两份 OpenAPI 契约。

## Monorepo 结构

```text
apps/web/                  Next.js + TypeScript + Tailwind CSS
services/api/              搜索任务、状态查询与持久化
services/ai-engine/        小采、小洗、小验、小判及搜索源适配
contracts/                 后端与 AI 引擎 OpenAPI 契约
qa/                        跨模块契约检查与验收清单
```

## 环境要求

- Node.js 20.9+、pnpm 11
- Python 3.10+
- 实时搜索需要 Tavily API key；大模型注释可选
- 后端默认用本地 SQLite。只有主动配置本地 PostgreSQL URL 时才连接 PostgreSQL

## 安装

从仓库根目录执行：

```bash
pnpm install
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env
cp apps/web/.env.example apps/web/.env.local
```

若系统没有 `python3.12`，请使用 Python 3.10+。在 `.env` 中配置本地开发用 `TAVILY_API_KEY` 才能发起实时搜索；密钥不要提交到 Git。未配置搜索源时，引擎会返回明确错误，不会生成演示搜索结果。

要使用本地 PostgreSQL，请先启动自己的开发数据库，再取消注释并调整 `.env` 中的 `DATABASE_URL`。示例只指向本机开发库。V0.1 不需要配置数据库，默认 SQLite 文件为 `zhize-api-local.db`。

## 启动

在三个终端中从仓库根目录分别运行。后端与 AI 引擎终端先加载本地环境变量：

```bash
set -a; source .env; set +a
uvicorn app.main:app --app-dir services/ai-engine --host 127.0.0.1 --port 8001 --reload
```

```bash
set -a; source .env; set +a
uvicorn services.api.app.main:app --host 127.0.0.1 --port 8000 --reload
```

```bash
pnpm dev:web
```

前端运行于 <http://127.0.0.1:3000>。后端和 AI 引擎健康检查分别是 `/health`。前端提交 `POST /api/v1/search`，然后轮询 `GET /api/v1/search/{task_id}`。

## 检查

```bash
python -m pytest -q qa services/api/tests services/ai-engine/tests
python -m compileall -q services/api/app services/ai-engine/app
pnpm --filter @zhize/web typecheck
pnpm --filter @zhize/web lint
pnpm --filter @zhize/web build
```

自动化测试使用 mock 搜索提供方和本地 SQLite，不要求真实密钥或 PostgreSQL。真实 Tavily 搜索需配置本地开发 key 后单独验证；不要将生产服务或密钥用于测试。

## 契约

- [`contracts/api.openapi.yaml`](contracts/api.openapi.yaml)：搜索任务创建和状态查询
- [`contracts/ai-engine.openapi.yaml`](contracts/ai-engine.openapi.yaml)：真实来源搜索与证据结果

