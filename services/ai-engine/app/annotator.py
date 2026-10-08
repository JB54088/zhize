import json
from typing import Any

import httpx
from fastapi import HTTPException

from .schemas import SearchResult
from .settings import Settings


async def annotate_results(
    settings: Settings, query: str, results: list[SearchResult]
) -> list[dict[str, Any]]:
    """Optionally classify retrieved sources; never returns or creates sources."""
    if not results:
        return []
    if not settings.llm_model and not settings.llm_api_key:
        return []
    if not settings.llm_model:
        raise HTTPException(
            status_code=503,
            detail="LLM_API_KEY is set but LLM_MODEL is not configured.",
        )

    candidates = [
        {
            "id": result.id,
            "title": result.title,
            "url": result.url,
            "snippet": result.snippet,
            "evidence": result.evidence,
        }
        for result in results
    ]
    messages = [
        {
            "role": "system",
            "content": (
                "You classify only the supplied web-search records. Never create, "
                "rename, or infer a source. Return JSON with an annotations array; "
                "each item has id, verdict, confidence, evidence. id must be copied "
                "from an input record. evidence must be an exact quote copied from "
                "that record's evidence field. If no exact quote supports a judgment, "
                "use verdict unverified, confidence 0, and empty evidence. Verdict "
                "must be supports, contradicts, context, or unverified."
            ),
        },
        {
            "role": "user",
            "content": json.dumps(
                {"query": query, "sources": candidates}, ensure_ascii=False
            ),
        },
    ]
    headers = {"Content-Type": "application/json"}
    if settings.llm_api_key:
        headers["Authorization"] = f"Bearer {settings.llm_api_key}"
    try:
        async with httpx.AsyncClient(timeout=45) as client:
            response = await client.post(
                f"{settings.llm_base_url}/chat/completions",
                headers=headers,
                json={
                    "model": settings.llm_model,
                    "messages": messages,
                    "response_format": {"type": "json_object"},
                    "temperature": 0,
                },
            )
            response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        raise HTTPException(
            status_code=502, detail="Configured LLM API returned an error response."
        ) from exc
    except httpx.RequestError as exc:
        raise HTTPException(
            status_code=502, detail="Could not connect to the configured LLM API."
        ) from exc

    try:
        body = response.json()["choices"][0]["message"]["content"]
        payload = json.loads(body)
        annotations = payload["annotations"]
        if not isinstance(annotations, list):
            raise TypeError("annotations must be a list")
        return [item for item in annotations if isinstance(item, dict)]
    except (ValueError, KeyError, IndexError, TypeError) as exc:
        raise HTTPException(
            status_code=502, detail="Configured LLM API returned invalid annotations."
        ) from exc
