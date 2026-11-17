"""Shared, bounded event/data contract for V0.4 autonomous research.

The model may propose one of the declared tools. Application code validates
the arguments, enforces the budget, executes the tool, and validates evidence.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


ResearchStatus = Literal[
    "planned", "running", "completed", "partial", "stopped", "failed"
]
ToolName = Literal[
    "search_web", "search_official", "read_webpage", "inspect_evidence"
]
ToolStatus = Literal["running", "completed", "partial", "rejected", "failed"]


class ResearchBudget(BaseModel):
    """Hard ceilings; reaching any ceiling stops tool execution safely."""

    model_config = ConfigDict(extra="forbid")

    max_rounds: int = Field(default=3, ge=1, le=3)
    max_search_requests: int = Field(default=12, ge=1, le=30)
    max_model_calls: int = Field(default=8, ge=1, le=30)
    max_total_tokens: int = Field(default=16_000, ge=256, le=100_000)
    # Reserve room for evidence-driven follow-up sources after initial search.
    max_candidates: int = Field(default=60, ge=1, le=100)
    max_elapsed_seconds: float = Field(default=90, gt=0, le=600)
    max_results_per_query: int = Field(default=8, ge=1, le=20)


class ToolCallRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    round_number: int = Field(ge=1)
    name: ToolName
    status: ToolStatus
    input_summary: str = Field(default="", max_length=500)
    output_summary: str = Field(default="", max_length=500)
    provider: str | None = Field(default=None, max_length=80)
    source_count: int | None = Field(default=None, ge=0)
    duration_ms: int | None = Field(default=None, ge=0)
    error_code: str | None = Field(default=None, max_length=80)
    model: str | None = Field(default=None, max_length=120)
    request_id: str | None = Field(default=None, max_length=180)
    prompt_tokens: int | None = Field(default=None, ge=0)
    completion_tokens: int | None = Field(default=None, ge=0)
    total_tokens: int | None = Field(default=None, ge=0)


class ResearchRound(BaseModel):
    model_config = ConfigDict(extra="forbid")

    round_number: int = Field(ge=1, le=3)
    status: ResearchStatus
    reason: str = Field(default="", max_length=500)
    planned_queries: list[str] = Field(default_factory=list, max_length=12)
    search_requests: int = Field(default=0, ge=0)
    sources_found: int = Field(default=0, ge=0)
    new_sources: int = Field(default=0, ge=0)
    candidates_before: int = Field(default=0, ge=0)
    candidates_after: int = Field(default=0, ge=0)
    evidence_gaps: list[str] = Field(default_factory=list, max_length=20)
    duration_ms: int | None = Field(default=None, ge=0)


class SearchProviderMetric(BaseModel):
    model_config = ConfigDict(extra="forbid")

    provider: str = Field(min_length=1, max_length=80)
    configured: bool
    requests: int = Field(default=0, ge=0)
    returned_sources: int = Field(default=0, ge=0)
    failures: int = Field(default=0, ge=0)
    latency_ms: int = Field(default=0, ge=0)
    error_code: str | None = Field(default=None, max_length=80)


class ResearchOutcome(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Literal["completed", "partial", "stopped", "failed"]
    rounds: list[ResearchRound] = Field(default_factory=list, max_length=3)
    tool_calls: list[ToolCallRecord] = Field(default_factory=list, max_length=100)
    providers: list[SearchProviderMetric] = Field(default_factory=list, max_length=10)
    model_calls: int = Field(default=0, ge=0)
    total_tokens: int | None = Field(default=None, ge=0)
    elapsed_ms: int = Field(default=0, ge=0)
    stop_reason: str = Field(default="", max_length=500)
    partial_results: bool = False
