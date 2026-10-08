# QA 与质量保障

V0.1 的自动化测试由后续 QA 工作流补充。接口契约位于 `../contracts/`。

首批质量门禁建议包括：

- 前端 TypeScript 检查、ESLint 与 Next.js production build
- FastAPI 健康检查、请求校验、AI 引擎代理和无来源返回路径
- 后端与 AI 引擎的契约一致性及错误映射
- 全链路本地冒烟流程；不要求生产密钥或数据库

