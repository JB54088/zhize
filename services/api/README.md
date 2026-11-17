# 知择 API service

FastAPI service for creating and polling search tasks. It forwards requests to
the configured AI engine, persists task status, source records and compact
cross-domain result cards with SQLAlchemy. It does not generate search results
itself.

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

- `POST /api/v1/search` accepts `{ "query": "...", "candidate_limit": 50 }`
  and returns HTTP 202 with a task ID and `queued` status. Candidate collection
  is bounded from 1 to 100 and is independent from display pagination. The old
  `limit` request field is accepted during migration but is deprecated and no
  longer truncates the candidate pool.
- `GET /api/v1/search/{task_id}?page=1&page_size=10` returns ranked
  `ResultCard` objects, with a default page size of 10 and a `total_results`
  count covering all retained candidates. Full bounded source records remain in
  local task storage and are not included in the public task response.
- `GET /health` reports service health.

The API contract is maintained at `contracts/api.openapi.yaml`.

## V0.1 limits

Task execution uses FastAPI in-process background work. Tasks survive process
restarts because records are persisted, but queued/running work is not resumed
after a restart, and there is no distributed worker, retry policy, or task
expiry. Startup applies additive nullable columns for local compatibility and
backfills old result JSON into source-record storage. Use managed migrations
before evolving a shared database schema.
