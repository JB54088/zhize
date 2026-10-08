# 知择 API service

FastAPI service for creating and polling V0.1 search tasks. It forwards searches
to the configured AI engine and persists task status and engine-returned results
with SQLAlchemy. It does not generate search results itself.

## Local setup

From the repository root, install this service's dependencies and run it:

```sh
python -m venv .venv
. .venv/bin/activate
pip install -r services/api/requirements.txt
uvicorn services.api.app.main:app --reload --port 8000
```

The default local database is `sqlite:///./zhize-api-local.db`. Set
`DATABASE_URL` to a local PostgreSQL URL to use PostgreSQL, for example
`postgresql+psycopg://zhize:zhize@localhost:5432/zhize_dev`. Use only a local
development database. Set `AI_ENGINE_URL` (default `http://127.0.0.1:8001`) to
the local AI-engine service. No production credentials or database are needed.

## API

- `POST /api/v1/search` accepts `{ "query": "...", "limit": 10 }` and returns
  HTTP 202 with a task ID and `queued` status.
- `GET /api/v1/search/{task_id}` returns the stored query, status, engine results,
  and an optional safe error message.
- `GET /health` reports service health.

The API contract is maintained at `contracts/api.openapi.yaml`.

## V0.1 limits

Task execution uses FastAPI in-process background work. Tasks survive process
restarts because records are persisted, but queued/running work is not resumed
after a restart, and there is no distributed worker, retry policy, task expiry,
or migration framework yet. `create_all` is suitable for the initial local
schema only; use managed migrations before evolving a shared database schema.
