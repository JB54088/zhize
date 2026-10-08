# 知择 V0.1 QA

API 契约一致性测试位于 `test_search_contracts.py`，覆盖后端任务创建/状态查询契约和 AI 引擎搜索响应契约。服务单测和前端检查由各模块自行维护。

## 本地运行

```bash
python -m pip install -r qa/requirements.txt
python -m pytest qa
```

## 验收清单

- 后端创建搜索任务并返回 `task_id`；任务状态能从 queued/running 到 completed 或 failed。
- 搜索结果由 AI 引擎传回，后端轮询响应的 `SearchResult` 与两个 OpenAPI 契约一致。
- AI 引擎未配置搜索源时返回明确错误，不返回示例或合成搜索结果。
- 每条结果的 URL 和摘要来自配置的真实搜索源；证据能在来源内容中定位，判断理由引用对应来源。
- LLM 未配置、超时或响应不合约时，结果标记为未验证或返回明确错误；不得补造来源、证据或结论。
- 前端覆盖提交、排队/运行、完成、无结果、失败和移动端布局。
- 自动化测试使用 mock/本地测试数据库，不访问生产服务；有凭据的实时搜索验证需显式配置本地开发密钥后人工执行。
