import os
from dataclasses import dataclass
from urllib.parse import urlparse


@dataclass(frozen=True)
class Settings:
    search_provider: str
    search_api_key: str
    search_base_url: str
    search_timeout_seconds: float
    official_source_domains: tuple[str, ...]
    llm_api_key: str
    llm_model: str
    llm_base_url: str
    llm_thinking_mode: str | None = None
    llm_request_timeout_seconds: float = 45.0
    llm_max_retries: int = 2
    llm_input_usd_per_million: float | None = None
    llm_output_usd_per_million: float | None = None
    search_providers: tuple[str, ...] = ()
    brave_search_api_key: str = ""
    brave_search_base_url: str = "https://api.search.brave.com/res/v1/web/search"
    baidu_search_api_key: str = ""

    @classmethod
    def from_env(cls) -> "Settings":
        search_provider = os.getenv("SEARCH_PROVIDER", "").strip().lower()
        configured_providers = tuple(
            item.strip().lower()
            for item in os.getenv("SEARCH_PROVIDERS", "").split(",")
            if item.strip()
        )
        llm_base_url = os.getenv(
            "LLM_BASE_URL", "https://api.deepseek.com"
        ).strip().rstrip("/")
        return cls(
            search_provider=search_provider,
            search_api_key=os.getenv("TAVILY_API_KEY", "").strip(),
            search_base_url=os.getenv(
                "SEARCH_BASE_URL",
                os.getenv("TAVILY_BASE_URL", "https://api.tavily.com/search"),
            ).strip(),
            search_timeout_seconds=_positive_float(
                os.getenv("SEARCH_TIMEOUT_SECONDS", "30"), 30.0
            ),
            official_source_domains=tuple(
                domain.strip().lower().removeprefix("www.").strip(".")
                for domain in os.getenv("OFFICIAL_SOURCE_DOMAINS", "").split(",")
                if domain.strip()
            ),
            llm_api_key=os.getenv("LLM_API_KEY", "").strip(),
            llm_model=os.getenv("LLM_MODEL", "").strip(),
            llm_base_url=llm_base_url,
            llm_thinking_mode=_llm_thinking_mode(llm_base_url),
            llm_request_timeout_seconds=_positive_float(
                os.getenv("LLM_REQUEST_TIMEOUT_SECONDS", "45"), 45.0
            ),
            llm_max_retries=_bounded_int(os.getenv("LLM_MAX_RETRIES", "2"), 2, 0, 5),
            llm_input_usd_per_million=_optional_nonnegative_float(
                os.getenv("LLM_INPUT_USD_PER_MILLION", "")
            ),
            llm_output_usd_per_million=_optional_nonnegative_float(
                os.getenv("LLM_OUTPUT_USD_PER_MILLION", "")
            ),
            search_providers=configured_providers,
            brave_search_api_key=os.getenv("BRAVE_SEARCH_API_KEY", "").strip(),
            brave_search_base_url=os.getenv(
                "BRAVE_SEARCH_BASE_URL",
                "https://api.search.brave.com/res/v1/web/search",
            ).strip(),
            baidu_search_api_key=os.getenv("BAIDU_SEARCH_API_KEY", "").strip(),
        )


def _positive_float(value: str, fallback: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return fallback
    return parsed if parsed > 0 else fallback


def _bounded_int(value: str, fallback: int, minimum: int, maximum: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return fallback
    return min(maximum, max(minimum, parsed))


def _optional_nonnegative_float(value: str) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed >= 0 else None


def _llm_thinking_mode(base_url: str) -> str | None:
    """Default DeepSeek to non-thinking for bounded structured/API tasks."""
    requested = os.getenv("LLM_THINKING", "auto").strip().lower()
    if requested in {"enabled", "disabled"}:
        return requested
    if requested != "auto":
        return None
    hostname = (urlparse(base_url).hostname or "").lower()
    if hostname == "deepseek.com" or hostname.endswith(".deepseek.com"):
        return "disabled"
    return None
