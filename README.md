# 知择 V0.1

知择是一个 AI 信息决策平台。当前仓库提供前端、API、AI 核验服务和接口契约的最小本地开发骨架。

## 技术栈

- 前端：Next.js App Router、TypeScript、Tailwind CSS
- 后端：FastAPI、Python
- AI 引擎：FastAPI、Python，通过可配置的 OpenAI 兼容 Chat Completions API 分析给定来源
- 数据库：PostgreSQL（仅保留本地开发配置；当前骨架不会建立数据库连接）
- 包管理：pnpm workspace + Python `requirements.txt`

## 目录

```text
frontend/       Next.js 用户界面
backend/        面向前端的 FastAPI 服务
ai-engine/      基于来源证据的大模型核验服务
contracts/      后端 API 与 AI 引擎 OpenAPI 契约
qa/             测试与质量保障说明
```

## 环境要求

- Node.js 20.9 或更新版本，以及 pnpm 11
- Python 3.10 或更新版本
- 本地 PostgreSQL 仅在后续接入数据库功能时需要

## 安装

在仓库根目录执行：

```bash
pnpm install
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env
```

若系统没有 `python3.12`，请使用 Python 3.10+ 创建虚拟环境。不要把真实密钥提交到 Git。

## 启动

分别在三个终端中、从仓库根目录运行：

```bash
# 终端 1：AI 引擎
cd ai-engine && ../.venv/bin/uvicorn app.main:app --reload --port 8001
```

```bash
# 终端 2：后端 API
cd backend && ../.venv/bin/uvicorn app.main:app --reload --port 8000
```

```bash
# 终端 3：前端
pnpm dev:web
```

前端地址为 <http://localhost:3000>，后端健康检查为 <http://127.0.0.1:8000/health>，AI 引擎健康检查为 <http://127.0.0.1:8001/health>。

当前 AI 接口只分析调用方提供的来源；搜索源尚未接入。没有来源时会返回“证据不足”，不调用大模型。提供来源后，需要在本地 `.env` 配置 `LLM_BASE_URL`、`LLM_API_KEY` 和 `LLM_MODEL`。数据库配置也是本地占位值，目前没有数据库启动或连接逻辑。

## 接口契约

- [`contracts/api.openapi.yaml`](contracts/api.openapi.yaml)：前端调用的后端 API
- [`contracts/ai-engine.openapi.yaml`](contracts/ai-engine.openapi.yaml)：后端调用的 AI 引擎 API

服务启动后也可在 FastAPI 的 `/docs` 查看自动生成的交互式 API 文档。

