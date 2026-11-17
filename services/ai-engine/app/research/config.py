"""Environment-backed, bounded configuration for autonomous research."""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from .contracts import ResearchBudget


@dataclass(frozen=True)
class ResearchLoopConfig:
    budget: ResearchBudget = field(default_factory=ResearchBudget)
    max_prompt_tokens: int = 8_000
    max_completion_tokens: int = 2_400
    max_tool_calls_per_round: int = 2
    max_context_candidates: int = 6
    max_context_sources: int = 3
    max_tool_output_chars: int = 8_000

    @classmethod
    def from_env(cls) -> "ResearchLoopConfig":
        """Read bounded local settings; malformed values use safe defaults."""
        return cls(
            budget=research_budget_from_env(),
            max_prompt_tokens=_bounded_int("RESEARCH_MAX_PROMPT_TOKENS", 8_000, 256, 24_000),
            max_completion_tokens=_bounded_int(
                "RESEARCH_MAX_COMPLETION_TOKENS", 2_400, 64, 8_192
            ),
            max_tool_calls_per_round=_bounded_int(
                "RESEARCH_MAX_TOOL_CALLS_PER_ROUND", 2, 1, 8
            ),
            max_context_candidates=_bounded_int(
                "RESEARCH_MAX_CONTEXT_CANDIDATES", 6, 1, 100
            ),
            max_context_sources=_bounded_int("RESEARCH_MAX_CONTEXT_SOURCES", 3, 1, 40),
            max_tool_output_chars=_bounded_int(
                "RESEARCH_MAX_TOOL_OUTPUT_CHARS", 8_000, 512, 30_000
            ),
        )


def research_budget_from_env() -> ResearchBudget:
    """Parse the shared RESEARCH_MAX_* settings using contract-safe clamps."""
    return ResearchBudget(
        max_rounds=_bounded_int("RESEARCH_MAX_ROUNDS", 3, 1, 3),
        max_search_requests=_bounded_int("RESEARCH_MAX_SEARCH_REQUESTS", 12, 1, 30),
        max_model_calls=_bounded_int("RESEARCH_MAX_MODEL_CALLS", 8, 1, 30),
        max_total_tokens=_bounded_int("RESEARCH_MAX_TOTAL_TOKENS", 16_000, 256, 100_000),
        max_candidates=_bounded_int("RESEARCH_MAX_CANDIDATES", 60, 1, 100),
        max_elapsed_seconds=_bounded_float(
            "RESEARCH_MAX_ELAPSED_SECONDS", 90.0, 0.1, 600.0
        ),
        max_results_per_query=_bounded_int(
            "RESEARCH_MAX_RESULTS_PER_QUERY", 8, 1, 20
        ),
    )


def _bounded_int(name: str, default: int, minimum: int, maximum: int) -> int:
    try:
        parsed = int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default
    return min(maximum, max(minimum, parsed))


def _bounded_float(name: str, default: float, minimum: float, maximum: float) -> float:
    try:
        parsed = float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default
    if parsed != parsed:  # NaN
        return default
    return min(maximum, max(minimum, parsed))
