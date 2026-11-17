"""Structured inputs, outputs, and logs for the four business agents."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


AgentName = Literal["collect", "clean", "verify", "judge"]
AgentAction = Literal[
    "intent_understanding",
    "search",
    "extraction",
    "source_verification",
    "semantic_match",
    "final_answer",
]
AgentStatus = Literal[
    "pending",
    "running",
    "completed",
    "partial",
    "skipped_unconfigured",
    "skipped_no_evidence",
    "failed",
]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class IntentAnalysis(StrictModel):
    domain: Literal["employment", "procurement", "general"] = "general"
    intent_summary: str = Field(default="", max_length=600)
    search_queries: list[str] = Field(default_factory=list, max_length=6)
    soft_preferences: list[str] = Field(default_factory=list, max_length=12)


class ExtractedField(StrictModel):
    field: Literal[
        "organization",
        "role",
        "graduate_cohort",
        "majors",
        "deadline",
        "application_url",
        "qualifications",
        "business_attributes",
        "region",
    ]
    value: str = Field(min_length=1, max_length=240)
    quote: str = Field(min_length=1, max_length=320)


class ExtractionItem(StrictModel):
    id: str
    fields: list[ExtractedField] = Field(default_factory=list, max_length=1)


class ExtractionResponse(StrictModel):
    items: list[ExtractionItem] = Field(default_factory=list, max_length=30)


class VerificationFact(StrictModel):
    field: Literal[
        "project_name",
        "organization",
        "role",
        "graduate_cohort",
        "major",
        "deadline",
        "open_status",
        "application_url",
        "qualifications",
        "region",
        "purchasing_unit",
        "budget",
        "published_at",
        "business_attributes",
    ]
    value: str = Field(min_length=1, max_length=240)
    quote: str = Field(min_length=1, max_length=320)
    evidence_url: str = Field(min_length=1, max_length=2048)
    assessment: Literal["supports", "contradicts", "context", "uncertain"]


class VerificationItem(StrictModel):
    id: str
    facts: list[VerificationFact] = Field(default_factory=list, max_length=1)


class VerificationResponse(StrictModel):
    items: list[VerificationItem] = Field(default_factory=list, max_length=20)


class MatchItem(StrictModel):
    id: str
    semantic_fit: Literal["strong", "partial", "weak", "uncertain"]
    score: float = Field(ge=0, le=1)
    reason: str = Field(default="", max_length=1000)
    evidence_quotes: list[str] = Field(default_factory=list, max_length=10)


class MatchResponse(StrictModel):
    items: list[MatchItem] = Field(default_factory=list, max_length=30)


class AnswerClaim(StrictModel):
    result_id: str
    text: str = Field(min_length=1, max_length=1000)
    evidence_quotes: list[str] = Field(min_length=1, max_length=8)


class AnswerResponse(StrictModel):
    summary: str = Field(min_length=1, max_length=1600)
    claims: list[AnswerClaim] = Field(default_factory=list, max_length=20)


class AgentRunLog(BaseModel):
    agent: AgentName
    action: AgentAction
    status: AgentStatus
    started_at: str
    completed_at: str
    duration_ms: int = Field(ge=0)
    actual_model: str | None = None
    request_id: str | None = None
    latency_ms: int | None = Field(default=None, ge=0)
    prompt_tokens: int | None = Field(default=None, ge=0)
    completion_tokens: int | None = Field(default=None, ge=0)
    total_tokens: int | None = Field(default=None, ge=0)
    estimated_cost_usd: float | None = Field(default=None, ge=0)
    retry_count: int | None = Field(default=None, ge=0)
    input_summary: str
    output_summary: str
    error_code: str | None = None
    skip_reason: str | None = None


@dataclass(frozen=True)
class AgentExecution:
    """JSON-safe action output and its private-safe execution summary."""

    output: dict[str, Any]
    run_log: dict[str, Any]
