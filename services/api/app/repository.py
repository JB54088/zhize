from __future__ import annotations

from datetime import datetime, timezone
from typing import Protocol

from sqlalchemy import JSON, DateTime, String, Text, create_engine, select
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker
from sqlalchemy.pool import StaticPool

from .schemas import SearchResult, TaskStatus


class Base(DeclarativeBase):
    pass


class SearchTaskRecord(Base):
    __tablename__ = "search_tasks"

    task_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    query: Mapped[str] = mapped_column(String(4000), nullable=False)
    status: Mapped[str] = mapped_column(String(16), nullable=False)
    results: Mapped[list[dict[str, object]]] = mapped_column(JSON, nullable=False)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )


class SearchTask(Protocol):
    task_id: str
    query: str
    status: TaskStatus
    results: list[SearchResult]
    error: str | None


class TaskRepository(Protocol):
    def ensure_schema(self) -> None: ...

    def create(self, task_id: str, query: str) -> SearchTask: ...

    def get(self, task_id: str) -> SearchTask | None: ...

    def set_status(self, task_id: str, status: TaskStatus) -> SearchTask | None: ...

    def complete(
        self, task_id: str, results: list[SearchResult]
    ) -> SearchTask | None: ...

    def fail(self, task_id: str, error: str) -> SearchTask | None: ...


class SQLAlchemyTaskRepository:
    """Persist search tasks in PostgreSQL or a local SQLite database."""

    def __init__(self, database_url: str) -> None:
        engine_options: dict[str, object] = {"pool_pre_ping": True}
        if database_url.startswith("sqlite"):
            engine_options["connect_args"] = {"check_same_thread": False}
            if database_url in {"sqlite://", "sqlite:///:memory:"}:
                engine_options["poolclass"] = StaticPool

        self.engine: Engine = create_engine(database_url, **engine_options)
        self._sessions = sessionmaker(self.engine, expire_on_commit=False)

    def ensure_schema(self) -> None:
        Base.metadata.create_all(self.engine)

    def create(self, task_id: str, query: str) -> SearchTaskRecord:
        now = datetime.now(timezone.utc)
        row = SearchTaskRecord(
            task_id=task_id,
            query=query,
            status="queued",
            results=[],
            error=None,
            created_at=now,
            updated_at=now,
        )
        with self._sessions() as session:
            session.add(row)
            session.commit()
            session.refresh(row)
            return row

    def get(self, task_id: str) -> SearchTaskRecord | None:
        with self._sessions() as session:
            return session.scalar(
                select(SearchTaskRecord).where(SearchTaskRecord.task_id == task_id)
            )

    def set_status(
        self, task_id: str, status: TaskStatus
    ) -> SearchTaskRecord | None:
        with self._sessions() as session:
            row = session.get(SearchTaskRecord, task_id)
            if row is None:
                return None
            row.status = status
            row.updated_at = datetime.now(timezone.utc)
            session.commit()
            session.refresh(row)
            return row

    def complete(
        self, task_id: str, results: list[SearchResult]
    ) -> SearchTaskRecord | None:
        with self._sessions() as session:
            row = session.get(SearchTaskRecord, task_id)
            if row is None:
                return None
            row.status = "completed"
            row.results = [result.model_dump(mode="json") for result in results]
            row.error = None
            row.updated_at = datetime.now(timezone.utc)
            session.commit()
            session.refresh(row)
            return row

    def fail(self, task_id: str, error: str) -> SearchTaskRecord | None:
        with self._sessions() as session:
            row = session.get(SearchTaskRecord, task_id)
            if row is None:
                return None
            row.status = "failed"
            row.error = error
            row.updated_at = datetime.now(timezone.utc)
            session.commit()
            session.refresh(row)
            return row


def task_to_values(row: SearchTask) -> dict[str, object]:
    """Normalize ORM and in-memory task records for the response model."""
    raw_results = row.results
    results = [
        result if isinstance(result, SearchResult) else SearchResult.model_validate(result)
        for result in raw_results
    ]
    return {
        "task_id": row.task_id,
        "query": row.query,
        "status": row.status,
        "results": results,
        "error": row.error,
    }
