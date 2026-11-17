# 知择 V0.4

知择是面向 C 端与 B 端的通用 AI 信息搜索与决策底座。用户用自然语言提出需求，系统依次执行小采、小洗、小验、小判，输出来源、关键条件、字段级证据和判断理由。DeepSeek 模型协助理解意图、字段抽取、官方证据分析、语义匹配和结论组织；确定性规则仍负责来源与硬性条件的最终裁决。

## Monorepo 结构

```text
apps/web/                                      Next.js 搜索与决策 UI
services/api/                                  搜索任务编排、阶段轮询、持久化与分页
services/ai-engine/app/stages/collect.py       查询意图与并行来源采集
services/ai-engine/app/stages/clean.py         结构规范化、去重与字段来源保留
services/ai-engine/app/stages/verify.py        官方来源与字段级证据核验
services/ai-engine/app/stages/judge.py         硬/软条件匹配与解释性分类
services/ai-engine/app/agents/                 四 Agent 结构化工作流与执行日志
services/ai-engine/app/llm/                    OpenAI 兼容 DeepSeek 客户端
services/ai-engine/app/research/               有预算的多轮研究规划与工具调用
services/ai-engine/app/search_tools/            搜索来源、官网发现和安全网页证据读取
contracts/                                     API 与 AI Engine OpenAPI 契约
qa/                                            跨模块契约检查与验收清单
```

## 环境要求

- Node.js 20.9+、pnpm 11
- Python 3.10+
- 实时搜索默认使用百度千帆搜索 API，需要本地配置 `BAIDU_SEARCH_API_KEY`；Tavily 与 Brave Search 可作为可选来源
- 启用模型能力需要 DeepSeek 或兼容服务的 API key 和模型名
- 公开 PDF 读取使用 pypdf，限制文件大小与页数
- 后端默认用本地 SQLite。只有主动配置本地 PostgreSQL URL 时才连接 PostgreSQL

## 安装

从仓库根目录执行：

```bash
pnpm install
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
test -f .env || cp .env.example .env
test -f apps/web/.env.local || cp apps/web/.env.example apps/web/.env.local
```

若系统没有 `python3.12`，请使用 Python 3.10+。在 `.env` 中配置本地开发用 `BAIDU_SEARCH_API_KEY` 才能发起实时搜索；密钥不要提交到 Git。未配置搜索源时，引擎会返回明确错误，不会生成演示搜索结果。

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

前端运行于 <http://127.0.0.1:3000>。后端和 AI 引擎健康检查分别是 `/health`。前端提交 `POST /api/v1/search`，然后轮询 `GET /api/v1/search/{task_id}`。Backend 读取 AI Engine 的 NDJSON 阶段事件并持久化快照；每个阶段的状态、数量和耗时仅在执行时写入，未执行的数据保持空。任务读取支持 `page` 和 `page_size`。

## 检查

```bash
python -m pytest -q qa services/api/tests services/ai-engine/tests
python -m compileall -q services/api/app services/ai-engine/app
pnpm --filter @zhize/web typecheck
pnpm --filter @zhize/web lint
pnpm --filter @zhize/web build --webpack
```

自动化测试使用 mock 搜索提供方和本地 SQLite，不要求真实密钥或 PostgreSQL。真实百度搜索需配置本地开发 key 后单独验证；不要将生产服务或密钥用于测试。

## 契约

- [`contracts/api.openapi.yaml`](contracts/api.openapi.yaml)：搜索任务创建和状态查询
- [`contracts/ai-engine.openapi.yaml`](contracts/ai-engine.openapi.yaml)：四阶段事件流与来源证据结果

AI Engine 的 `/api/v1/search` 旧 JSON 接口保持兼容；Backend 使用 `/api/v1/search/stream` 接收 NDJSON 阶段快照、四 Agent 的 action 记录及最终结果。采集查询和搜索可以并行，清洗、核验和判断按依赖顺序执行。模型未配置、调用失败或引用校验失败时均有单独状态，不会冒充模型完成。

## 大模型配置与安全边界

在本地 `.env` 中设置 `LLM_BASE_URL=https://api.deepseek.com`、`LLM_API_KEY` 和 `LLM_MODEL=deepseek-flash`。真实密钥不要放进 `.env.example`、Git 或日志。只有同时配置 Key 和模型后才会发出模型请求；缺少配置时，Agent 日志会显示 `skipped_unconfigured`。若请求失败，日志保留脱敏错误代码、耗时和重试信息。

`deepseek-chat` 是 DeepSeek 已停用的旧模型标识；当前示例使用 `deepseek-flash`。成本估算需要显式设置 `LLM_INPUT_USD_PER_MILLION` 和 `LLM_OUTPUT_USD_PER_MILLION`；未设置时只记录 API 返回的 Token 用量，不估算费用。

## V0.4 多轮研究

搜索先返回真实来源候选，再由 DeepSeek 在固定工具白名单中提出补充检索、官方来源定向检索、公开网页读取或证据检查。每条工具建议均由本地代码校验和执行；网页访问会校验 DNS、重定向、robots.txt、协议和下载上限，不访问私有/本机地址，也不绕过登录、验证码或访问限制。模型建议不能把未知字段升级为已核实字段。

研究最多 3 轮，默认上限为 8 次搜索请求、8 次研究规划模型调用、12,000 个研究模型 Token、30 个候选和 90 秒；可通过 `.env` 的 `RESEARCH_MAX_*` 设置降低或调整（服务端还会执行代码中约束的最大值）。`ResearchOutcome`、逐轮日志、工具调用日志、搜索源请求量和阶段性候选会进入任务状态。阶段性候选始终标记为待核实。

可选启用多个来源时，在 `.env` 配置相应密钥并设置 `SEARCH_PROVIDERS=baidu,tavily`（或其他已配置组合）。未配置的来源会跳过并保留未配置状态；新增来源只在真实 API 成功响应时计数。公开 PDF 读取需先安装 AI Engine requirements 中的 `pypdf` 依赖。
