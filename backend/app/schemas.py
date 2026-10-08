from typing import List, Literal, Optional

from pydantic import BaseModel, Field


class Source(BaseModel):
    id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    url: Optional[str] = None
    snippet: str
    published_at: Optional[str] = None


class VerificationRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    sources: List[Source] = Field(default_factory=list, max_length=50)


class Claim(BaseModel):
    text: str
    verdict: Literal["supported", "mixed", "refuted", "insufficient_evidence"]
    confidence: float = Field(ge=0, le=1)


class Citation(BaseModel):
    source_id: str
    quote: str
    url: Optional[str] = None
    title: Optional[str] = None


class VerificationResponse(BaseModel):
    verdict: Literal["supported", "mixed", "refuted", "insufficient_evidence"]
    summary: str
    confidence: float = Field(ge=0, le=1)
    claims: List[Claim] = Field(default_factory=list)
    citations: List[Citation] = Field(default_factory=list)

