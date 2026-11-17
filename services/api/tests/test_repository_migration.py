import json
import sqlite3

from services.api.app.repository import SQLAlchemyTaskRepository, task_to_values


def test_additive_migration_preserves_legacy_rows_and_initializes_new_snapshots(tmp_path) -> None:
    database_path = tmp_path / "legacy.sqlite"
    connection = sqlite3.connect(database_path)
    connection.execute(
        """
        CREATE TABLE search_tasks (
            task_id VARCHAR(36) PRIMARY KEY,
            query VARCHAR(4000) NOT NULL,
            status VARCHAR(16) NOT NULL,
            results JSON NOT NULL,
            error TEXT,
            created_at DATETIME NOT NULL,
            updated_at DATETIME NOT NULL
        )
        """
    )
    old_result = {
        "id": "legacy-result",
        "title": "Legacy result",
        "url": "https://example.org/legacy",
        "source_name": "Example",
        "snippet": "old snippet",
        "evidence": "old evidence",
        "match_reason": "old reason",
        "verdict": "unverified",
        "confidence": 0.2,
    }
    connection.execute(
        "INSERT INTO search_tasks VALUES (?, ?, ?, ?, ?, ?, ?)",
        (
            "legacy-task",
            "legacy query",
            "completed",
            json.dumps([old_result]),
            None,
            "2026-10-01 00:00:00",
            "2026-10-01 00:00:00",
        ),
    )
    connection.commit()
    connection.close()

    repository = SQLAlchemyTaskRepository(f"sqlite:///{database_path}")
    repository.ensure_schema()
    row = repository.get("legacy-task")
    values = task_to_values(row)

    assert values["query"] == "legacy query"
    assert values["status"] == "completed"
    assert values["results"][0].id == "legacy-result"
    assert row.source_records[0]["id"] == "legacy-result"
    assert row.source_records[0]["evidence"] == "old evidence"
    assert values["stages"] == []
    assert values["errors"] == []


def test_new_task_starts_with_four_pending_stages_without_invented_metrics() -> None:
    repository = SQLAlchemyTaskRepository("sqlite:///:memory:")
    repository.ensure_schema()
    row = repository.create("new-task", "new query")
    values = task_to_values(row)

    assert [stage["key"] for stage in values["stages"]] == [
        "collect", "clean", "verify", "judge"
    ]
    for stage in values["stages"]:
        assert stage["status"] == "pending"
        assert stage["started_at"] is None
        assert stage["completed_at"] is None
        assert stage["duration_ms"] is None
        assert stage["input_count"] is None
        assert stage["output_count"] is None
