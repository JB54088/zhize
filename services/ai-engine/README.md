# 知择 AI 引擎

FastAPI 搜索与信息核验服务，默认监听 `127.0.0.1:8001`。

## 本地运行

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r services/ai-engine/requirements.txt
SEARCH_PROVIDER=baidu BAIDU_SEARCH_API_KEY=your-local-key \
  uvicorn app.main:app --app-dir services/ai-engine --host 127.0.0.1 --port 8001 --reload
```

引擎也可由仓库根目录的 Python 环境启动；依赖仅列在本服务目录下。`GET /health` 不依赖外部密钥。实际搜索使用 `POST /api/v1/search`，请求示例：

```json
{"query":"某项决策相关的问题","limit":8}
```

## 配置

| 变量 | 用途 |
| --- | --- |
| `SEARCH_PROVIDER` | 单一搜索源，缺省时搜索返回明确的 503 配置错误 |
| `SEARCH_PROVIDERS` | 可选，逗号分隔的已注册搜索适配器；缺省时沿用 `SEARCH_PROVIDER`。内置适配器为 `baidu`、`tavily`、`brave`；每个选中的源必须有本地密钥，否则会有显式配置状态 |
| `BAIDU_SEARCH_API_KEY` | 百度千帆 AppBuilder API Key；仅存放在本地 `.env` |
| `TAVILY_API_KEY` | 可选的 Tavily 搜索 API 密钥 |
| `SEARCH_BASE_URL` | 可选，Tavily 搜索 API 地址覆盖；默认 `https://api.tavily.com/search` |
| `SEARCH_TIMEOUT_SECONDS` | 可选，搜索请求超时；默认 30 秒 |
| `OFFICIAL_SOURCE_DOMAINS` | 可选，逗号分隔的官方域名白名单；默认仅识别政府域名后缀 |
| `LLM_BASE_URL` | OpenAI 兼容 API 根地址；DeepSeek 默认 `https://api.deepseek.com` |
| `LLM_API_KEY` | 本地配置的大模型 API 密钥；不会写入日志或仓库 |
| `LLM_MODEL` | 可配置模型名；当前 DeepSeek 示例为 `deepseek-flash` |
| `LLM_THINKING` | `auto` 时仅对 DeepSeek API 发送 `thinking: disabled`，避免结构化任务默认消耗高强度推理 Token；其他兼容服务不发送该厂商字段。可明确设为 `enabled` 或 `disabled` |
| `LLM_REQUEST_TIMEOUT_SECONDS` | 模型调用超时；默认 45 秒 |
| `LLM_MAX_RETRIES` | 可重试错误的上限；默认 2，最大 5 |
| `LLM_INPUT_USD_PER_MILLION` / `LLM_OUTPUT_USD_PER_MILLION` | 可选计价；未配置时费用保持未知 |
| `RESEARCH_MAX_ROUNDS` | 自主研究最多轮次；上限 3 |
| `RESEARCH_MAX_SEARCH_REQUESTS` | 单任务搜索 API 请求硬上限，包含初始采集 |
| `RESEARCH_MAX_MODEL_CALLS` | 研究规划模型调用上限 |
| `RESEARCH_MAX_TOTAL_TOKENS` | 自主研究规划环节的总 Token 硬上限；其他 Agent 另按调用记录 Token |
| `RESEARCH_MAX_ELAPSED_SECONDS` | 自主研究总耗时上限 |
| `RESEARCH_MAX_CANDIDATES` | 规范化候选数量上限 |
| `RESEARCH_MAX_RESULTS_PER_QUERY` | 每条搜索式返回上限 |
| `RESEARCH_MAX_PROMPT_TOKENS` / `RESEARCH_MAX_COMPLETION_TOKENS` | 单次研究规划请求的输入/输出预算；输出仍受总 Token 上限约束 |
| `RESEARCH_MAX_TOOL_CALLS_PER_ROUND` | 每轮允许的模型工具调用数 |
| `RESEARCH_MAX_CONTEXT_CANDIDATES` / `RESEARCH_MAX_CONTEXT_SOURCES` / `RESEARCH_MAX_TOOL_OUTPUT_CHARS` | 发送给研究规划器的候选、来源和工具输出上下文上限 |

`baidu` 使用百度千帆官方 Web Search API（`qianfan.baidubce.com/v2/ai_search/web_search`），返回网页引用后仍由知择逐字段读取和核验证据。无需 Playwright 浏览器；搜索网页自动化不作为数据源。

搜索服务没有配置时不会使用 fixtures、演示结果或模型生成的结果。返回来源始终来自真实搜索提供方响应；证据从该响应的 `raw_content` 或 `content` 中提取。模型只能辅助处理已有来源：小洗仅能补充带精确原文引用且规则未提取的字段；小验仅分析已经逐字段核验的官方来源；小判和最终结论只能引用确定性核验过的事实。引用逐字校验失败时会丢弃模型文本，硬性条件始终由规则裁决。

`LLM_API_KEY` 或 `LLM_MODEL` 缺失时，模型动作会标记为 `skipped_unconfigured`，不发出匿名/虚构请求。模型调用失败会标记 `failed` 并保留可用的确定性结果作为显式部分结果，不会记录为模型成功。每次执行记录包含实际模型名、请求 ID、Token、客户端请求耗时和整段 action 耗时；模型没有返回 Token 时保持空值。

## V0.4 自主研究

研究规划器使用 DeepSeek Chat Completions Tool Calling 提出白名单工具调用；程序校验参数后才执行真实搜索或读取公开网页。工具回合最多 3 轮，按上述配置限制请求数、候选数、模型 Token 和耗时。初始采集计入搜索请求总数。网页读取仅允许公开 HTTP(S)、遵守 `robots.txt`，对 DNS/IP、重定向和下载大小设限；不绕过登录、验证码或访问限制。`ResearchOutcome.total_tokens` 统计研究规划器调用，其他小采/小洗/小验/小判调用的 Token 分别出现在 Agent 执行记录中。

## V0.3 四阶段与 Agent 工作流

- **小采** (`app/agents/collect.py`)：LLM 理解意图并生成搜索式，规则保留硬条件；所选搜索源并发运行并记录 provider 追溯。默认使用百度千帆 Web Search API，返回的网页标题、链接、摘要及日期仍需进入后续证据核验。
- **小洗** (`app/agents/clean.py`、`app/stages/clean.py`)：确定性规范化和 URL/语义去重后，LLM 只补充有逐字原文、规则尚未提取的字段。
- **小验** (`app/agents/verify.py`、`app/stages/verify.py`)：规则寻找和核验官方来源，逐字段区分 `verified`、`unverified`、`conflict` 或缺失；LLM 只解释已核验官方正文中的字段证据，不改变规则状态。
- **小判** (`app/agents/judge.py`、`app/stages/judge.py`)：规则依据已核实字段输出 `recommended`、`needs_verification` 或 `not_matched`；LLM 只提供带精确证据的语义匹配解释和最终结论，不能把未知条件当作满足。

`POST /api/v1/search/stream` 以 NDJSON 输出四阶段快照、六种 action 的执行记录以及最终结果；action 包含需求理解、多路搜索、字段抽取、来源核验、语义匹配和结论生成。模型执行记录展示实际模型、请求 ID、Token、客户端请求耗时、action 耗时和错误代码；提示词、完整网页正文和密钥不会进入日志。`POST /api/v1/search` 保留兼容字段，并添加结论及其来源引用。`application_url` 只接受匹配职位的官方原文明确标记并逐字核验的链接；没有证据时返回 `null`，不会回退到搜索结果 URL。

## 检查

在安装服务依赖后，从仓库根目录运行：

```sh
python -m unittest discover -s services/ai-engine/tests -v
python -m compileall -q services/ai-engine/app
```
