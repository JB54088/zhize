# 知择 Web

Next.js + TypeScript + Tailwind CSS front end for the 知择 V0.1 search experience.

## Local development

From the repository root, run `pnpm install`, then `pnpm dev:web`.
The app expects the API at `http://127.0.0.1:8000` by default. To override it,
copy `.env.example` to `.env.local` in this directory and set
`NEXT_PUBLIC_API_BASE_URL`. Search requests use the task API:

- `POST /api/v1/search` with `{ "query": "...", "limit": 10 }`
- `GET /api/v1/search/{task_id}` until the task is completed or failed

Search results are rendered only when returned by the API; this app does not
include sample or seeded result data.
