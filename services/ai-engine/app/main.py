from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException

from .annotator import annotate_results
from .pipeline import apply_grounded_annotations, clean_and_ground_results
from .provider import ProviderError, create_search_provider
from .schemas import SearchRequest, SearchResponse
from .settings import Settings


load_dotenv(Path(__file__).resolve().parents[3] / ".env")

app = FastAPI(title="知择 AI 搜索与信息核验引擎", version="0.1.0")


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "service": "zhize-ai-engine"}


@app.post("/api/v1/search", response_model=SearchResponse)
async def search(payload: SearchRequest) -> SearchResponse:
    query = payload.query.strip()
    if not query:
        raise HTTPException(status_code=422, detail="query must contain non-whitespace text")
    settings = Settings.from_env()
    try:
        provider = create_search_provider(settings)
        raw_results = await provider.search(query, payload.limit)
    except ProviderError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc

    # 小采 (provider) -> 小洗 (normalization and evidence extraction).
    results = clean_and_ground_results(query, raw_results, payload.limit)
    # 小验 / 小判 are optional LLM annotations; they are constrained to IDs and
    # literal quotes from provider records. Deterministic labels remain available
    # when no model credentials are configured.
    annotations = await annotate_results(settings, query, results)
    apply_grounded_annotations(results, annotations)
    return SearchResponse(query=query, results=results)
