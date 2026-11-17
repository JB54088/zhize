from __future__ import annotations

from datetime import datetime, timezone
from typing import Protocol

from sqlalchemy import JSON, DateTime, String, Text, create_engine, inspect, select, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker
from sqlalchemy.pool import StaticPool

from .schemas import (
    AgentRun,
    AnswerClaim,
    ResearchOutcome,
    ResearchRound,
    ResultCard,
    SearchProviderMetric,
    SearchResult,
    SourceRecord,
    StageKey,
    StageProgress,
    TaskStatus,
    ToolCallRecord,
)


STAGE_KEYS: tuple[StageKey, ...] = ("collect", "clean", "verify", "judge")


def pending_stages() -> list[dict[str, object]]:
    return [StageProgress(key=key).model_dump(mode="json") for key in STAGE_KEYS]


class Base(DeclarativeBase):
    pass


class SearchTaskRecord(Base):
    __tablename__ = "search_tasks"

    task_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    query: Mapped[str] = mapped_column(String(4000), nullable=False)
    status: Mapped[str] = mapped_column(String(16), nullable=False)
    results: Mapped[list[dict[str, object]]] = mapped_column(JSON, nullable=False)
    # Full engine records remain stored separately when the API serves compact
    # cards, so evidence excerpts and source provenance are not lost.
    source_records: Mapped[list[dict[str, object]] | None] = mapped_column(JSON, nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    stages: Mapped[list[dict[str, object]] | None] = mapped_column(JSON, nullable=True)
    errors: Mapped[list[str] | None] = mapped_column(JSON, nullable=True)
    agent_runs: Mapped[list[dict[str, object]] | None] = mapped_column(JSON, nullable=True)
    answer: Mapped[str | None] = mapped_column(Text, nullable=True)
    answer_claims: Mapped[list[dict[str, object]] | None] = mapped_column(JSON, nullable=True)
    research: Mapped[dict[str, object] | None] = mapped_column(JSON, nullable=True)
    research_rounds: Mapped[list[dict[str, object]] | None] = mapped_column(JSON, nullable=True)
    tool_calls: Mapped[list[dict[str, object]] | None] = mapped_column(JSON, nullable=True)
    search_metrics: Mapped[list[dict[str, object]] | None] = mapped_column(JSON, nullable=True)
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
    results: list[ResultCard]
    source_records: list[dict[str, object]] | None
    error: str | None
    stages: list[dict[str, object]] | None
    errors: list[str] | None
    agent_runs: list[dict[str, object]] | None
    answer: str | None
    answer_claims: list[dict[str, object]] | None
    research: dict[str, object] | None
    research_rounds: list[dict[str, object]] | None
    tool_calls: list[dict[str, object]] | None
    search_metrics: list[dict[str, object]] | None


class TaskRepository(Protocol):
    def ensure_schema(self) -> None: ...

    def create(self, task_id: str, query: str) -> SearchTask: ...

    def get(self, task_id: str) -> SearchTask | None: ...

    def set_status(self, task_id: str, status: TaskStatus) -> SearchTask | None: ...

    def complete(
        self,
        task_id: str,
        results: list[SearchResult | ResultCard],
        errors: list[str] | None = None,
        agent_runs: list[AgentRun] | None = None,
        answer: str | None = None,
        answer_claims: list[AnswerClaim] | None = None,
        research: ResearchOutcome | None = None,
        search_metrics: list[SearchProviderMetric] | None = None,
        source_records: list[SourceRecord | SearchResult | ResultCard] | None = None,
    ) -> SearchTask | None: ...

    def fail(self, task_id: str, error: str) -> SearchTask | None: ...

    def update_stage(
        self, task_id: str, stage: StageProgress
    ) -> SearchTask | None: ...

    def update_agent_run(
        self, task_id: str, run: AgentRun
    ) -> SearchTask | None: ...

    def update_research_round(
        self, task_id: str, research_round: ResearchRound
    ) -> SearchTask | None: ...

    def update_tool_call(
        self, task_id: str, tool_call: ToolCallRecord
    ) -> SearchTask | None: ...

    def update_partial_results(
        self, task_id: str, results: list[SearchResult | ResultCard]
    ) -> SearchTask | None: ...


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
        # create_all does not alter tables created by an earlier app version.
        # Additive nullable columns keep existing local/Postgres task rows readable.
        columns = {column["name"] for column in inspect(self.engine).get_columns("search_tasks")}
        with self.engine.begin() as connection:
            if "stages" not in columns:
                connection.execute(text("ALTER TABLE search_tasks ADD COLUMN stages JSON"))
            if "errors" not in columns:
                connection.execute(text("ALTER TABLE search_tasks ADD COLUMN errors JSON"))
            if "agent_runs" not in columns:
                connection.execute(text("ALTER TABLE search_tasks ADD COLUMN agent_runs JSON"))
            if "answer" not in columns:
                connection.execute(text("ALTER TABLE search_tasks ADD COLUMN answer TEXT"))
            if "answer_claims" not in columns:
                connection.execute(text("ALTER TABLE search_tasks ADD COLUMN answer_claims JSON"))
            if "research" not in columns:
                connection.execute(text("ALTER TABLE search_tasks ADD COLUMN research JSON"))
            if "research_rounds" not in columns:
                connection.execute(text("ALTER TABLE search_tasks ADD COLUMN research_rounds JSON"))
            if "tool_calls" not in columns:
                connection.execute(text("ALTER TABLE search_tasks ADD COLUMN tool_calls JSON"))
            if "search_metrics" not in columns:
                connection.execute(text("ALTER TABLE search_tasks ADD COLUMN search_metrics JSON"))
            if "source_records" not in columns:
                connection.execute(text("ALTER TABLE search_tasks ADD COLUMN source_records JSON"))
            # Preserve each pre-card result verbatim before it is reserialized as
            # a compact ResultCard by a later task update.
            connection.execute(
                text("UPDATE search_tasks SET source_records = results WHERE source_records IS NULL")
            )

    def create(self, task_id: str, query: str) -> SearchTaskRecord:
        now = datetime.now(timezone.utc)
        row = SearchTaskRecord(
            task_id=task_id,
            query=query,
            status="queued",
            results=[],
            source_records=[],
            error=None,
            stages=pending_stages(),
            errors=[],
            agent_runs=[],
            answer=None,
            answer_claims=[],
            research=None,
            research_rounds=[],
            tool_calls=[],
            search_metrics=[],
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
        self,
        task_id: str,
        results: list[SearchResult | ResultCard],
        errors: list[str] | None = None,
        agent_runs: list[AgentRun] | None = None,
        answer: str | None = None,
        answer_claims: list[AnswerClaim] | None = None,
        research: ResearchOutcome | None = None,
        search_metrics: list[SearchProviderMetric] | None = None,
        source_records: list[SourceRecord | SearchResult | ResultCard] | None = None,
    ) -> SearchTaskRecord | None:
        with self._sessions() as session:
            row = session.get(SearchTaskRecord, task_id)
            if row is None:
                return None
            from .result_cards import legacy_results_to_cards

            legacy_results = [result for result in results if isinstance(result, SearchResult)]
            cards = [result for result in results if isinstance(result, ResultCard)]
            if legacy_results:
                cards.extend(legacy_results_to_cards(legacy_results, query=row.query))
            has_partial_stage = any(
                isinstance(stage, dict) and stage.get("status") == "partial"
                for stage in (row.stages or [])
            )
            has_partial_research = research is not None and research.status == "partial"
            row.status = (
                "partial"
                if (errors and results) or has_partial_stage or has_partial_research
                else "failed"
                if errors
                else "completed"
            )
            row.results = [result.model_dump(mode="json") for result in cards]
            if source_records is not None:
                row.source_records = [
                    result.model_dump(mode="json") for result in source_records
                ]
            elif results:
                row.source_records = [result.model_dump(mode="json") for result in results]
            row.errors = errors or []
            row.error = "; ".join(row.errors) if row.errors else None
            if answer is not None:
                row.answer = answer
            if answer_claims is not None:
                row.answer_claims = [
                    claim.model_dump(mode="json", exclude={"sources"})
                    for claim in answer_claims
                ]
            if research is not None:
                row.research = research.model_dump(mode="json")
                # Preserve streamed events if the terminal outcome intentionally
                # omits them, while preferring the validated final snapshot.
                row.research_rounds = [
                    item.model_dump(mode="json") for item in research.rounds
                ]
                row.tool_calls = [
                    item.model_dump(mode="json") for item in research.tool_calls
                ]
            if search_metrics is not None:
                row.search_metrics = [item.model_dump(mode="json") for item in search_metrics]
            for run in agent_runs or []:
                self._upsert_agent_run(row, run)
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
            row.errors = [error]
            now = datetime.now(timezone.utc)
            agent_runs: list[dict[str, object]] = []
            for item in row.agent_runs or []:
                try:
                    run = AgentRun.model_validate(item)
                except (TypeError, ValueError):
                    continue
                if run.status == "running":
                    duration_ms = run.duration_ms
                    if run.started_at:
                        try:
                            started_at = datetime.fromisoformat(
                                run.started_at.replace("Z", "+00:00")
                            )
                            duration_ms = max(
                                0, int((now - started_at).total_seconds() * 1000)
                            )
                        except ValueError:
                            pass
                    run = run.model_copy(
                        update={
                            "status": "failed",
                            "completed_at": now.isoformat(),
                            "duration_ms": duration_ms,
                            "error_code": "engine_failed",
                        }
                    )
                agent_runs.append(run.model_dump(mode="json"))
            row.agent_runs = agent_runs
            row.updated_at = now
            session.commit()
            session.refresh(row)
            return row

    def update_stage(
        self, task_id: str, stage: StageProgress
    ) -> SearchTaskRecord | None:
        with self._sessions() as session:
            row = session.get(SearchTaskRecord, task_id)
            if row is None:
                return None
            # SQLAlchemy's JSON type does not track nested in-place updates;
            # assign a fresh outer list so every received snapshot is committed.
            stages = [dict(item) for item in (row.stages or pending_stages())]
            serialized = stage.model_dump(mode="json")
            index = next(
                (i for i, current in enumerate(stages) if current.get("key") == stage.key),
                None,
            )
            if index is None:
                stages.append(serialized)
            else:
                stages[index] = serialized
            row.stages = stages
            row.updated_at = datetime.now(timezone.utc)
            session.commit()
            session.refresh(row)
            return row

    @staticmethod
    def _upsert_agent_run(row: SearchTaskRecord, run: AgentRun) -> None:
        runs = [dict(item) for item in (row.agent_runs or [])]
        serialized = run.model_dump(mode="json")
        index = next(
            (
                i
                for i, current in enumerate(runs)
                if current.get("agent") == run.agent
                and current.get("action") == run.action
            ),
            None,
        )
        if index is None:
            runs.append(serialized)
        else:
            runs[index] = serialized
        row.agent_runs = runs

    def update_agent_run(
        self, task_id: str, run: AgentRun
    ) -> SearchTaskRecord | None:
        with self._sessions() as session:
            row = session.get(SearchTaskRecord, task_id)
            if row is None:
                return None
            # SQLAlchemy JSON columns don't track nested mutations. Assign a new
            # list so each streamed snapshot is durable and pollable.
            self._upsert_agent_run(row, run)
            row.updated_at = datetime.now(timezone.utc)
            session.commit()
            session.refresh(row)
            return row

    def update_research_round(
        self, task_id: str, research_round: ResearchRound
    ) -> SearchTaskRecord | None:
        with self._sessions() as session:
            row = session.get(SearchTaskRecord, task_id)
            if row is None:
                return None
            rounds = [dict(item) for item in (row.research_rounds or [])]
            serialized = research_round.model_dump(mode="json")
            index = next(
                (i for i, current in enumerate(rounds)
                 if current.get("round_number") == research_round.round_number),
                None,
            )
            if index is None:
                rounds.append(serialized)
            else:
                rounds[index] = serialized
            row.research_rounds = rounds
            row.updated_at = datetime.now(timezone.utc)
            session.commit()
            session.refresh(row)
            return row

    @staticmethod
    def _upsert_tool_call(row: SearchTaskRecord, tool_call: ToolCallRecord) -> None:
        calls = [dict(item) for item in (row.tool_calls or [])]
        serialized = tool_call.model_dump(mode="json")
        match_index = next(
            (
                i for i, current in enumerate(calls)
                if current.get("round_number") == tool_call.round_number
                and current.get("name") == tool_call.name
                and current.get("input_summary") == tool_call.input_summary
                and current.get("provider") == tool_call.provider
                and current.get("status") == "running"
            ),
            None,
        )
        if match_index is None and tool_call.status == "running":
            calls.append(serialized)
        elif match_index is None:
            calls.append(serialized)
        else:
            calls[match_index] = serialized
        row.tool_calls = calls[-100:]

    def update_tool_call(
        self, task_id: str, tool_call: ToolCallRecord
    ) -> SearchTaskRecord | None:
        with self._sessions() as session:
            row = session.get(SearchTaskRecord, task_id)
            if row is None:
                return None
            self._upsert_tool_call(row, tool_call)
            row.updated_at = datetime.now(timezone.utc)
            session.commit()
            session.refresh(row)
            return row

    def update_partial_results(
        self, task_id: str, results: list[SearchResult | ResultCard]
    ) -> SearchTaskRecord | None:
        """Expose current candidates without ending the running task."""
        with self._sessions() as session:
            row = session.get(SearchTaskRecord, task_id)
            if row is None:
                return None
            from .result_cards import legacy_results_to_cards

            legacy_results = [result for result in results if isinstance(result, SearchResult)]
            cards = [result for result in results if isinstance(result, ResultCard)]
            if legacy_results:
                cards.extend(
                    legacy_results_to_cards(legacy_results, query=row.query)
                )
            row.source_records = [result.model_dump(mode="json") for result in results]
            cards = [
                card.model_copy(update={"matching_status": "needs_verification"})
                for card in cards
            ]
            row.results = [card.model_dump(mode="json") for card in cards]
            row.updated_at = datetime.now(timezone.utc)
            session.commit()
            session.refresh(row)
            return row


def task_to_values(row: SearchTask) -> dict[str, object]:
    """Normalize ORM and in-memory task records for the response model."""
    from .result_cards import legacy_results_to_cards

    raw_results = row.results
    legacy_results: list[SearchResult] = []
    cards: list[ResultCard] = []
    for raw_result in raw_results:
        if isinstance(raw_result, ResultCard):
            cards.append(raw_result)
        elif isinstance(raw_result, SearchResult):
            legacy_results.append(raw_result)
        elif isinstance(raw_result, dict) and "category" in raw_result and "matching_status" in raw_result:
            cards.append(ResultCard.model_validate(raw_result))
        else:
            legacy_results.append(SearchResult.model_validate(raw_result))
    if legacy_results:
        cards.extend(legacy_results_to_cards(legacy_results, query=row.query))
    return {
        "task_id": row.task_id,
        "query": row.query,
        "status": row.status,
        "results": cards,
        # Kept internal for support/reprocessing; not included in public task JSON.
        "source_records": getattr(row, "source_records", None) or [],
        "error": row.error,
        # Legacy records predate stage tracking. Keep their snapshot empty
        # rather than implying stages are still pending or inventing timings.
        "stages": row.stages or [],
        "agent_runs": [
            run if isinstance(run, AgentRun) else AgentRun.model_validate(run)
            for run in (row.agent_runs or [])
        ],
        "answer": row.answer,
        "answer_claims": [
            claim if isinstance(claim, AnswerClaim) else AnswerClaim.model_validate(claim)
            for claim in (row.answer_claims or [])
        ],
        "research": (
            row.research if isinstance(getattr(row, "research", None), ResearchOutcome)
            else ResearchOutcome.model_validate(row.research)
            if getattr(row, "research", None) else None
        ),
        "research_rounds": [
            item if isinstance(item, ResearchRound) else ResearchRound.model_validate(item)
            for item in (getattr(row, "research_rounds", None) or [])
        ],
        "tool_calls": [
            item if isinstance(item, ToolCallRecord) else ToolCallRecord.model_validate(item)
            for item in (getattr(row, "tool_calls", None) or [])
        ],
        "search_metrics": [
            item if isinstance(item, SearchProviderMetric) else SearchProviderMetric.model_validate(item)
            for item in (getattr(row, "search_metrics", None) or [])
        ],
        "errors": row.errors or ([row.error] if row.error else []),
    }
