import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    search_provider: str
    search_api_key: str
    search_base_url: str
    search_timeout_seconds: float
    llm_api_key: str
    llm_model: str
    llm_base_url: str

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            search_provider=os.getenv("SEARCH_PROVIDER", "").strip().lower(),
            search_api_key=os.getenv("TAVILY_API_KEY", "").strip(),
            search_base_url=os.getenv(
                "SEARCH_BASE_URL",
                os.getenv("TAVILY_BASE_URL", "https://api.tavily.com/search"),
            ).strip(),
            search_timeout_seconds=_positive_float(
                os.getenv("SEARCH_TIMEOUT_SECONDS", "30"), 30.0
            ),
            llm_api_key=os.getenv("LLM_API_KEY", "").strip(),
            llm_model=os.getenv("LLM_MODEL", "").strip(),
            llm_base_url=os.getenv(
                "LLM_BASE_URL", "https://api.openai.com/v1"
            ).strip().rstrip("/"),
        )


def _positive_float(value: str, fallback: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return fallback
    return parsed if parsed > 0 else fallback

