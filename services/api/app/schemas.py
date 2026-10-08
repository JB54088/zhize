from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


TaskStatus = Literal["queued", "running", "completed", "failed"]
Verdict = Literal["supports", "contradicts", "context", "unverified"]


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=4000)
    limit: int = Field(default=10, ge=1, le=50)

    @field_validator("query")
    @classmethod
    def normalize_query(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("query must contain non-whitespace characters")
        return normalized


class SearchAccepted(BaseModel):
    task_id: str
    status: TaskStatus


class SearchResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    url: str = Field(min_length=1)
    source_name: str = Field(min_length=1)
    snippet: str
    evidence: str
    match_reason: str
    verdict: Verdict
    confidence: float = Field(ge=0, le=1)
    published_at: datetime | None = None


class EngineSearchResponse(BaseModel):
    query: str
    results: list[SearchResult] = Field(default_factory=list)


class SearchTaskResponse(BaseModel):
    task_id: str
    query: str
    status: TaskStatus
    results: list[SearchResult] = Field(default_factory=list)
    error: str | None = None
