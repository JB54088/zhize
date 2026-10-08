from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from typing import AsyncIterator
from uuid import uuid4

from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .engine_client import EngineClientError, HttpEngineClient
from .repository import SQLAlchemyTaskRepository, TaskRepository, task_to_values
from .schemas import SearchAccepted, SearchRequest, SearchResult, SearchTaskResponse

logger = logging.getLogger(__name__)


def create_app(
    *,
    repository: TaskRepository | None = None,
    engine_client: HttpEngineClient | None = None,
    database_url: str | None = None,
) -> FastAPI:
    """Build the API app with injectable persistence and engine boundaries."""
    task_repository = repository or SQLAlchemyTaskRepository(
        database_url
        or os.getenv("DATABASE_URL", "sqlite:///./zhize-api-local.db")
    )
    search_engine = engine_client or HttpEngineClient(
        os.getenv("AI_ENGINE_URL", "http://127.0.0.1:8001"),
        timeout_seconds=float(os.getenv("AI_ENGINE_TIMEOUT_SECONDS", "45")),
    )

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        task_repository.ensure_schema()
        yield

    app = FastAPI(title="知择 API", version="0.1.0", lifespan=lifespan)
    origins = [
        origin.strip()
        for origin in os.getenv(
            "CORS_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000"
        ).split(",")
        if origin.strip()
    ]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
    )

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok", "service": "zhize-api"}

    async def execute_search(task_id: str, query: str, limit: int) -> None:
        task_repository.set_status(task_id, "running")
        try:
            results = await search_engine.search(query, limit)
        except EngineClientError as exc:
            task_repository.fail(task_id, str(exc))
        except Exception:
            logger.exception("Unexpected error while executing search task %s", task_id)
            task_repository.fail(task_id, "Search processing failed.")
        else:
            # These results are accepted only from the configured search engine;
            # the API never attempts to synthesize or fill in missing results.
            task_repository.complete(task_id, results)

    @app.post(
        "/api/v1/search",
        status_code=202,
        response_model=SearchAccepted,
    )
    async def create_search(
        payload: SearchRequest, background_tasks: BackgroundTasks
    ) -> SearchAccepted:
        task_id = str(uuid4())
        task_repository.create(task_id, payload.query)
        background_tasks.add_task(
            execute_search, task_id, payload.query, payload.limit
        )
        return SearchAccepted(task_id=task_id, status="queued")

    @app.get(
        "/api/v1/search/{task_id}",
        response_model=SearchTaskResponse,
        response_model_exclude_none=True,
    )
    async def get_search(task_id: str) -> SearchTaskResponse:
        row = task_repository.get(task_id)
        if row is None:
            raise HTTPException(status_code=404, detail="Search task not found.")
        return SearchTaskResponse.model_validate(task_to_values(row))

    return app


app = create_app()
