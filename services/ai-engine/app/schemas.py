from typing import Literal, Optional

from pydantic import BaseModel, Field

from .research.contracts import ResearchOutcome, SearchProviderMetric


Verdict = Literal["supports", "contradicts", "context", "unverified"]
DeadlineStatus = Literal["expired", "not_expired", "not_stated", "unverified"]
OfficialSourceStatus = Literal[
    "verified", "third_party", "insufficient_evidence", "unverified"
]
EvidenceStatus = Literal["available", "missing"]
VerificationField = Literal[
    "project_name", "purchasing_unit", "published_at", "budget", "deadline", "evidence",
    "graduate_cohort", "major", "open_status", "application_url", "role",
    "organization", "qualifications", "business_attributes",
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
ResultKind = Literal["specific_opportunity", "organization_lead", "unknown"]
Decision = Literal["recommended", "needs_verification", "not_matched"]
ConditionStatus = Literal["verified", "unverified", "conflict", "missing"]
SourceType = Literal["official", "third_party", "unknown"]
StageKey = Literal["collect", "clean", "verify", "judge"]
StageStatus = Literal["pending", "running", "completed", "partial", "failed", "skipped"]
AgentAction = Literal[
    "intent_understanding", "search", "extraction", "source_verification",
    "semantic_match", "final_answer",
]
AgentStatus = Literal[
    "pending", "running", "completed", "partial", "skipped_unconfigured",
    "skipped_no_evidence", "failed",
]
ModelAssessment = Literal["supports", "contradicts", "context", "uncertain"]
ModelVerificationStatus = Literal["analyzed", "rejected", "not_analyzed"]
SemanticFit = Literal["strong", "partial", "weak", "uncertain"]


class KeyCondition(BaseModel):
    label: str
    value: str
    status: ConditionStatus
    evidence_url: Optional[str] = None
    quote: Optional[str] = None


class EvidenceSource(BaseModel):
    title: str
    url: str
    source_type: SourceType = "unknown"
    status: Literal["verified", "unverified", "conflict"] = "unverified"


class ModelVerifiedFact(BaseModel):
    field: VerificationField
    value: str
    assessment: ModelAssessment
    evidence_url: Optional[str] = None
    quote: Optional[str] = None


class LLMVerification(BaseModel):
    """Non-authoritative model reading of already verified official evidence."""

    status: ModelVerificationStatus
    facts: list[ModelVerifiedFact] = Field(default_factory=list)


class LLMSemanticMatch(BaseModel):
    """Semantic explanation kept separate from deterministic rule decisions."""

    status: ModelVerificationStatus
    fit: Optional[SemanticFit] = None
    score: Optional[float] = Field(default=None, ge=0, le=1)
    reason: str = ""
    evidence_quotes: list[str] = Field(default_factory=list)


class AnswerClaim(BaseModel):
    result_id: str
    text: str
    evidence_quotes: list[str] = Field(default_factory=list)


class StageSnapshot(BaseModel):
    key: StageKey
    status: StageStatus
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    duration_ms: Optional[int] = None
    input_count: Optional[int] = None
    output_count: Optional[int] = None
    verified_count: Optional[int] = None
    unverified_count: Optional[int] = None
    conflict_count: Optional[int] = None
    error: Optional[str] = None


class AgentRunSnapshot(BaseModel):
    """A redacted execution record for one four-agent workflow action."""

    agent: StageKey
    action: AgentAction
    status: AgentStatus
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    duration_ms: Optional[int] = Field(default=None, ge=0)
    latency_ms: Optional[int] = Field(default=None, ge=0)
    model: Optional[str] = None
    request_id: Optional[str] = None
    prompt_tokens: Optional[int] = Field(default=None, ge=0)
    completion_tokens: Optional[int] = Field(default=None, ge=0)
    total_tokens: Optional[int] = Field(default=None, ge=0)
    estimated_cost_usd: Optional[float] = Field(default=None, ge=0)
    input_summary: str = ""
    output_summary: str = ""
    error_code: Optional[str] = None


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=4000)
    # Candidate collection is independent from response/UI pagination. The old
    # `limit` remains accepted during migration but never truncates candidates.
    candidate_limit: int = Field(default=50, ge=1, le=100)
    limit: Optional[int] = Field(default=None, ge=1, le=20, deprecated=True)


CardAttributeStatus = Literal["verified", "pending", "unknown", "conflict"]
CardVerificationStatus = Literal["verified", "pending", "conflict", "unknown"]
CardMatchingStatus = Literal["recommended", "needs_verification", "not_matched", "unknown"]
CardFreshnessStatus = Literal["fresh", "stale", "unknown"]
ResultCategory = Literal["dining", "employment", "procurement", "product", "travel", "policy", "other"]


class ResultCardAttribute(BaseModel):
    key: str = Field(min_length=1, max_length=120)
    label: str = Field(min_length=1, max_length=120)
    value: Optional[str]
    status: CardAttributeStatus
    evidence_url: Optional[str]
    evidence_quote: Optional[str]


class ResultCardSource(BaseModel):
    url: str = Field(min_length=1)
    name: str = Field(min_length=1)
    type: SourceType
    status: CardAttributeStatus


class ResultCard(BaseModel):
    id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    category: ResultCategory
    summary: str
    rank: int = Field(ge=1)
    recommendation_reason: str
    attributes: list[ResultCardAttribute]
    source_url: Optional[str]
    source_name: Optional[str]
    additional_sources: list[ResultCardSource]
    published_at: Optional[str]
    last_verified_at: Optional[str]
    verification_status: CardVerificationStatus
    matching_status: CardMatchingStatus
    freshness_status: CardFreshnessStatus


class SourceRecord(BaseModel):
    """Bounded, internal provenance record passed from engine to API storage."""

    id: str = Field(min_length=1, max_length=80)
    title: str = Field(min_length=1, max_length=1000)
    url: str = Field(min_length=1, max_length=2048)
    source_name: Optional[str] = Field(default=None, max_length=240)
    provider: Optional[str] = Field(default=None, max_length=80)
    captured_at: Optional[str] = Field(default=None, max_length=80)
    published_at: Optional[str] = Field(default=None, max_length=120)
    search_query_index: Optional[int] = Field(default=None, ge=0)
    search_query: Optional[str] = Field(default=None, max_length=4000)
    raw_text: str = Field(default="", max_length=12000)
    raw_text_truncated: bool = False
    metadata: dict[str, str | int | float | bool | None] = Field(default_factory=dict, max_length=20)


class SearchResult(BaseModel):
    id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    url: str = Field(min_length=1)
    source_name: str = Field(min_length=1)
    snippet: str
    evidence: str
    match_reason: str
    verdict: Verdict
    confidence: float = Field(ge=0, le=1)
    published_at: Optional[str] = None
    official_source_url: Optional[str] = None
    verified_fields: list[VerificationField] = Field(default_factory=list)
    unverified_fields: list[VerificationField] = Field(default_factory=list)
    purchasing_unit: Optional[str] = None
    budget: Optional[str] = None
    deadline_at: Optional[str] = None
    deadline_status: DeadlineStatus = "unverified"
    official_source_status: OfficialSourceStatus = "insufficient_evidence"
    evidence_status: EvidenceStatus = "missing"
    procurement_category: ProcurementCategory = "unclear"
    match_status: MatchStatus = "unverified"
    match_score: float = Field(default=0.0, ge=0, le=1)
    mismatch_reason: Optional[str] = None
    # V0.2 generic decision fields are additive so V0.1 clients keep working.
    decision: Decision = "needs_verification"
    summary: str = ""
    key_conditions: list[KeyCondition] = Field(default_factory=list)
    evidence_sources: list[EvidenceSource] = Field(default_factory=list)
    evidence_detail: str = ""
    application_url: Optional[str] = None
    llm_verification: Optional[LLMVerification] = None
    llm_semantic_match: Optional[LLMSemanticMatch] = None
    result_kind: ResultKind = "unknown"


class SearchResponse(BaseModel):
    query: str
    results: list[SearchResult]
    cards: list[ResultCard] = Field(default_factory=list)
    source_records: list[SourceRecord] = Field(default_factory=list, max_length=500)
    answer: str = ""
    answer_claims: list[AnswerClaim] = Field(default_factory=list)
    agent_runs: list[AgentRunSnapshot] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    research: ResearchOutcome | None = None
    search_metrics: list[SearchProviderMetric] = Field(default_factory=list)
