from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


TaskStatus = Literal["queued", "running", "completed", "partial", "failed"]
StageKey = Literal["collect", "clean", "verify", "judge"]
StageStatus = Literal["pending", "running", "completed", "partial", "failed", "skipped"]
AgentKey = Literal["collect", "clean", "verify", "judge"]
AgentAction = Literal[
    "intent_understanding",
    "search",
    "extraction",
    "source_verification",
    "semantic_match",
    "final_answer",
]
AgentRunStatus = Literal[
    "pending", "running", "completed", "partial", "skipped_unconfigured",
    "skipped_no_evidence", "failed"
]
ResearchStatus = Literal["planned", "running", "completed", "partial", "stopped", "failed"]
ResearchOutcomeStatus = Literal["completed", "partial", "stopped", "failed"]
ResearchToolName = Literal["search_web", "search_official", "read_webpage", "inspect_evidence"]
ResearchToolStatus = Literal["running", "completed", "partial", "rejected", "failed"]
Verdict = Literal["supports", "contradicts", "context", "unverified"]
DeadlineStatus = Literal["expired", "not_expired", "not_stated", "unverified"]
OfficialSourceStatus = Literal[
    "verified", "third_party", "insufficient_evidence", "unverified"
]
EvidenceStatus = Literal["available", "missing"]
VerificationField = Literal[
    "project_name",
    "purchasing_unit",
    "published_at",
    "budget",
    "deadline",
    "evidence",
    "graduate_cohort",
    "major",
    "open_status",
    "application_url",
    "role",
    "organization",
    "qualifications",
    "business_attributes",
    "region",
]
ProcurementCategory = Literal[
    "ai_training",
    "ai_system",
    "digital_transformation",
    "compute_hardware",
    "other",
    "unclear",
]
MatchStatus = Literal[
    "matched", "partial", "keyword_only", "not_matched", "unverified"
]
Decision = Literal["recommended", "needs_verification", "not_matched"]
ConditionStatus = Literal["verified", "unverified", "conflict", "missing"]
EvidenceSourceType = Literal["official", "third_party", "unknown"]
EvidenceSourceStatus = Literal["verified", "unverified", "conflict"]


class StageProgress(BaseModel):
    key: StageKey
    status: StageStatus = "pending"
    started_at: str | None = None
    completed_at: str | None = None
    duration_ms: int | None = Field(default=None, ge=0)
    input_count: int | None = Field(default=None, ge=0)
    output_count: int | None = Field(default=None, ge=0)
    verified_count: int | None = Field(default=None, ge=0)
    unverified_count: int | None = Field(default=None, ge=0)
    conflict_count: int | None = Field(default=None, ge=0)
    error: str | None = None


class AgentRun(BaseModel):
    """A redacted execution snapshot for one action owned by a business agent."""

    agent: AgentKey
    action: AgentAction
    status: AgentRunStatus
    started_at: str | None = None
    completed_at: str | None = None
    duration_ms: int | None = Field(default=None, ge=0)
    latency_ms: int | None = Field(default=None, ge=0)
    model: str | None = Field(default=None, max_length=120)
    request_id: str | None = Field(default=None, max_length=180)
    prompt_tokens: int | None = Field(default=None, ge=0)
    completion_tokens: int | None = Field(default=None, ge=0)
    total_tokens: int | None = Field(default=None, ge=0)
    estimated_cost_usd: float | None = Field(default=None, ge=0)
    input_summary: str | None = Field(default=None, max_length=500)
    output_summary: str | None = Field(default=None, max_length=500)
    error_code: str | None = Field(default=None, max_length=80)


class KeyCondition(BaseModel):
    label: str
    value: str
    status: ConditionStatus
    evidence_url: str | None = None
    quote: str | None = None


class EvidenceSource(BaseModel):
    title: str
    url: str
    source_type: EvidenceSourceType = "unknown"
    status: EvidenceSourceStatus = "unverified"


CardAttributeStatus = Literal["verified", "pending", "unknown", "conflict"]
CardVerificationStatus = Literal["verified", "pending", "conflict", "unknown"]
CardMatchingStatus = Literal[
    "recommended", "needs_verification", "not_matched", "unknown"
]
CardFreshnessStatus = Literal["fresh", "stale", "unknown"]
ResultCategory = Literal[
    "dining", "employment", "procurement", "product", "travel", "policy", "other"
]


class ResultCardAttribute(BaseModel):
    model_config = ConfigDict(extra="forbid")

    key: str = Field(min_length=1, max_length=120)
    label: str = Field(min_length=1, max_length=120)
    value: str | None
    status: CardAttributeStatus
    evidence_url: str | None
    evidence_quote: str | None


class ResultCardSource(BaseModel):
    model_config = ConfigDict(extra="forbid")

    url: str = Field(min_length=1)
    name: str = Field(min_length=1)
    type: EvidenceSourceType
    status: CardAttributeStatus


class ResultCard(BaseModel):
    """Cross-domain, evidence-linked result consumed by the C-side UI."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    category: ResultCategory
    summary: str
    rank: int = Field(ge=1)
    recommendation_reason: str
    attributes: list[ResultCardAttribute]
    source_url: str | None
    source_name: str | None
    additional_sources: list[ResultCardSource]
    published_at: str | None
    last_verified_at: str | None
    verification_status: CardVerificationStatus
    matching_status: CardMatchingStatus
    freshness_status: CardFreshnessStatus


class SourceRecord(BaseModel):
    """Bounded internal source body/provenance retained outside the public card."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1, max_length=80)
    title: str = Field(min_length=1, max_length=1000)
    url: str = Field(min_length=1, max_length=2048)
    source_name: str | None = Field(default=None, max_length=240)
    provider: str | None = Field(default=None, max_length=80)
    captured_at: str | None = Field(default=None, max_length=80)
    published_at: str | None = Field(default=None, max_length=120)
    search_query_index: int | None = Field(default=None, ge=0)
    search_query: str | None = Field(default=None, max_length=4000)
    raw_text: str = Field(default="", max_length=12000)
    raw_text_truncated: bool = False
    metadata: dict[str, str | int | float | bool | None] = Field(
        default_factory=dict, max_length=20
    )


class LLMVerificationFact(BaseModel):
    field: str
    value: str
    assessment: Literal["supports", "contradicts", "context", "uncertain"]
    evidence_url: str | None = None
    quote: str | None = None


class LLMVerification(BaseModel):
    status: Literal["analyzed", "rejected", "not_analyzed"]
    facts: list[LLMVerificationFact] = Field(default_factory=list)


class LLMSemanticMatch(BaseModel):
    status: Literal["analyzed", "rejected", "not_analyzed"]
    fit: Literal["strong", "partial", "weak", "uncertain"] | None = None
    score: float | None = Field(default=None, ge=0, le=1)
    reason: str | None = None
    evidence_quotes: list[str] = Field(default_factory=list)


class AnswerClaim(BaseModel):
    result_id: str
    text: str
    evidence_quotes: list[str] = Field(default_factory=list)
    # The API enriches claims with links from the matching verified result;
    # this is not accepted as an engine-supplied source of truth.
    sources: list[EvidenceSource] = Field(default_factory=list)


class ResearchRound(BaseModel):
    """A concise, user-safe summary of one bounded search/research round."""

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


class ToolCallRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    round_number: int = Field(ge=1)
    name: ResearchToolName
    status: ResearchToolStatus
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

    status: ResearchOutcomeStatus
    rounds: list[ResearchRound] = Field(default_factory=list, max_length=3)
    tool_calls: list[ToolCallRecord] = Field(default_factory=list, max_length=100)
    providers: list[SearchProviderMetric] = Field(default_factory=list, max_length=10)
    model_calls: int = Field(default=0, ge=0)
    total_tokens: int | None = Field(default=None, ge=0)
    elapsed_ms: int = Field(default=0, ge=0)
    stop_reason: str = Field(default="", max_length=500)
    partial_results: bool = False


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=4000)
    # Candidate collection is deliberately independent from response pagination.
    # The old `limit` remains accepted for clients during migration, but it no
    # longer truncates the candidate pool or controls the page size.
    candidate_limit: int = Field(default=50, ge=1, le=100)
    limit: int = Field(default=10, ge=1, le=20, deprecated=True)

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
    published_at: str | None = None
    official_source_url: str | None = None
    verified_fields: list[VerificationField] = Field(default_factory=list)
    unverified_fields: list[VerificationField] = Field(default_factory=list)
    purchasing_unit: str | None = None
    budget: str | None = None
    deadline_at: str | None = None
    deadline_status: DeadlineStatus = "unverified"
    official_source_status: OfficialSourceStatus = "insufficient_evidence"
    evidence_status: EvidenceStatus = "missing"
    procurement_category: ProcurementCategory = "unclear"
    match_status: MatchStatus = "unverified"
    match_score: float = Field(default=0.0, ge=0, le=1)
    mismatch_reason: str | None = None
    decision: Decision = "needs_verification"
    summary: str = ""
    key_conditions: list[KeyCondition] = Field(default_factory=list)
    evidence_sources: list[EvidenceSource] = Field(default_factory=list)
    evidence_detail: str = ""
    application_url: str | None = None
    last_verified_at: str | None = None
    result_kind: Literal["specific_opportunity", "organization_lead", "unknown"] | None = None
    llm_verification: LLMVerification | None = None
    llm_semantic_match: LLMSemanticMatch | None = None


class EngineSearchResponse(BaseModel):
    query: str
    results: list[SearchResult] = Field(default_factory=list)
    cards: list[ResultCard] = Field(default_factory=list)
    source_records: list[SourceRecord] = Field(default_factory=list, max_length=500)
    errors: list[str] = Field(default_factory=list)
    agent_runs: list[AgentRun] = Field(default_factory=list)
    answer: str | None = Field(default=None, max_length=10000)
    answer_claims: list[AnswerClaim] = Field(default_factory=list)
    research: ResearchOutcome | None = None
    search_metrics: list[SearchProviderMetric] = Field(default_factory=list, max_length=10)


class SearchTaskResponse(BaseModel):
    task_id: str
    query: str
    status: TaskStatus
    results: list[ResultCard] = Field(default_factory=list)
    stages: list[StageProgress] = Field(default_factory=list)
    agent_runs: list[AgentRun] = Field(default_factory=list)
    research: ResearchOutcome | None = None
    research_rounds: list[ResearchRound] = Field(default_factory=list, max_length=3)
    tool_calls: list[ToolCallRecord] = Field(default_factory=list, max_length=100)
    search_metrics: list[SearchProviderMetric] = Field(default_factory=list, max_length=10)
    answer: str | None = Field(default=None, max_length=10000)
    answer_claims: list[AnswerClaim] = Field(default_factory=list)
    total_results: int = 0
    page: int = 1
    page_size: int = 10
    errors: list[str] = Field(default_factory=list)
    error: str | None = None
