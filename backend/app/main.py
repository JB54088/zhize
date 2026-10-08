import os
from pathlib import Path
from typing import Dict

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .schemas import VerificationRequest, VerificationResponse


load_dotenv(Path(__file__).resolve().parents[2] / ".env")

app = FastAPI(title="知择 API", version="0.1.0")
origins = [
    origin.strip()
    for origin in os.getenv(
        "CORS_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000"
    ).split(",")
    if origin.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


@app.get("/health")
async def health() -> Dict[str, str]:
    return {"status": "ok", "service": "zhize-api"}


@app.post("/api/v1/verify", response_model=VerificationResponse)
async def verify(payload: VerificationRequest) -> VerificationResponse:
    engine_url = os.getenv("AI_ENGINE_URL", "http://127.0.0.1:8001").rstrip("/")
    try:
        async with httpx.AsyncClient(timeout=65) as client:
            response = await client.post(
                f"{engine_url}/api/v1/verify", json=payload.model_dump()
            )
    except httpx.RequestError as exc:
        raise HTTPException(
            status_code=503, detail="AI 引擎暂不可用，请确认本地服务已启动。"
        ) from exc

    if response.is_error:
        try:
            detail = response.json().get("detail", "AI 引擎处理失败。")
        except ValueError:
            detail = "AI 引擎处理失败。"
        raise HTTPException(status_code=response.status_code, detail=detail)

    return VerificationResponse.model_validate(response.json())

