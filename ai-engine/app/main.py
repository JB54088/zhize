from pathlib import Path
from typing import Dict

from dotenv import load_dotenv
from fastapi import FastAPI

from .provider import verify_with_model
from .schemas import VerificationRequest, VerificationResponse


load_dotenv(Path(__file__).resolve().parents[2] / ".env")

app = FastAPI(title="知择 AI 引擎", version="0.1.0")


@app.get("/health")
async def health() -> Dict[str, str]:
    return {"status": "ok", "service": "zhize-ai-engine"}


@app.post("/api/v1/verify", response_model=VerificationResponse)
async def verify(payload: VerificationRequest) -> VerificationResponse:
    if not payload.sources:
        return VerificationResponse(
            verdict="insufficient_evidence",
            summary="当前请求没有提供可核验来源；搜索源接入后才能基于证据分析。",
            confidence=0,
        )
    return await verify_with_model(payload)

