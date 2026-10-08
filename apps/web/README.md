# 知择 Web

Next.js + TypeScript + Tailwind CSS front end for the 知择 V0.1 search experience.

## Local development

From this directory, install dependencies with `pnpm install --ignore-workspace`, then run `pnpm dev`.
The app expects the API at `http://127.0.0.1:8000` by default. Set
`NEXT_PUBLIC_API_BASE_URL` to override it. Search requests use the task API:

- `POST /api/v1/search` with `{ "query": "...", "limit": 10 }`
- `GET /api/v1/search/{task_id}` until the task is completed or failed

Search results are rendered only when returned by the API; this app does not
include sample or seeded result data.
