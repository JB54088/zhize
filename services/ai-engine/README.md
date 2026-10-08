# 知择 AI 引擎

FastAPI 搜索与信息核验服务，默认监听 `127.0.0.1:8001`。

## 本地运行

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r services/ai-engine/requirements.txt
SEARCH_PROVIDER=tavily TAVILY_API_KEY=your-local-key \
  uvicorn app.main:app --app-dir services/ai-engine --host 127.0.0.1 --port 8001 --reload
```

引擎也可由仓库根目录的 Python 环境启动；依赖仅列在本服务目录下。`GET /health` 不依赖外部密钥。实际搜索使用 `POST /api/v1/search`，请求示例：

```json
{"query":"某项决策相关的问题","limit":8}
```

## 配置

| 变量 | 用途 |
| --- | --- |
| `SEARCH_PROVIDER` | 当前支持 `tavily`；缺省时搜索返回明确的 503 配置错误 |
| `TAVILY_API_KEY` | Tavily 搜索 API 密钥 |
| `SEARCH_BASE_URL` | 可选，Tavily 搜索 API 地址覆盖；默认 `https://api.tavily.com/search` |
| `SEARCH_TIMEOUT_SECONDS` | 可选，搜索请求超时；默认 30 秒 |
| `OFFICIAL_SOURCE_DOMAINS` | 可选，逗号分隔的官方域名白名单；默认仅识别政府域名后缀 |
| `LLM_BASE_URL` | 可选的大模型 OpenAI 兼容 API 根地址 |
| `LLM_API_KEY` | 可选的大模型 API 密钥；本地兼容服务可留空 |
| `LLM_MODEL` | 可选的大模型名称；未配置模型时使用确定性词项匹配 |

搜索服务没有配置时不会使用 fixtures、演示结果或模型生成的结果。返回来源始终来自真实搜索提供方响应；证据从该响应的 `raw_content` 或 `content` 中提取。若配置了 LLM，它只能对搜索已返回的来源 ID 进行标注，且其证据引用必须是提供方正文中的逐字子串，否则该条标记为 `unverified`。LLM 不负责创建或补充搜索结果。

## 流程

- **小采**：调用配置的 Tavily Web Search API。
- **小洗**：校验 URL、规范化字段、去重，并从提供方正文截取证据。
- **小验**：检查查询词项是否出现在来源正文；配置 LLM 后可对现有来源做受约束的辅助核验。
- **小判**：输出 `supports`、`contradicts`、`context` 或 `unverified`、置信度及可解释匹配理由。无 LLM 时只给出 `context` 或 `unverified`，不把关键词重合夸大成支持结论。
- **状态提示**：截止日期只在原文明确写出截止词和邻近日期时判断；无日期则标为未提及，含糊日期标为未核实。非政府来源只有在 `OFFICIAL_SOURCE_DOMAINS` 白名单中才标为官方已核实；无可引用正文会明确标注证据缺失。

## 检查

在安装服务依赖后，从仓库根目录运行：

```sh
python -m unittest discover -s services/ai-engine/tests -v
python -m compileall -q services/ai-engine/app
```
