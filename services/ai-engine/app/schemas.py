from typing import Literal, Optional

from pydantic import BaseModel, Field


Verdict = Literal["supports", "contradicts", "context", "unverified"]
DeadlineStatus = Literal["expired", "not_expired", "not_stated", "unverified"]
OfficialSourceStatus = Literal["verified", "unverified"]
EvidenceStatus = Literal["available", "missing"]


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=4000)
    limit: int = Field(default=10, ge=1, le=20)


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
    deadline_status: DeadlineStatus = "unverified"
    official_source_status: OfficialSourceStatus = "unverified"
    evidence_status: EvidenceStatus = "missing"


class SearchResponse(BaseModel):
    query: str
    results: list[SearchResult]
