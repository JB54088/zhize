# 知择 V0.4 implementation contract

## Runtime boundary

- Search adapters implement `search(query, limit, *, include_domains=None) -> list[dict]` and report provider identity and request/result counts. A provider without credentials is explicitly unconfigured and must never return fabricated results.
- Research tool names are fixed: `search_web`, `search_official`, `read_webpage`, `inspect_evidence`. The model proposes calls; application code validates arguments against the tool schema, enforces budgets, invokes the implementation, and checks returned evidence.
- The research loop returns `ResearchOutcome` from `services/ai-engine/app/research/contracts.py`. Streaming events use `{type: "research_round", round: ResearchRound}` and `{type: "tool_call", call: ToolCallRecord}`. Existing `stage`, `agent`, and terminal `complete` events remain compatible; terminal completion adds `research` and `search_metrics`.
- `ResearchBudget` defaults to at most 3 rounds, 12 search requests, 8 model calls, 16,000 total tokens, 60 candidates (leaving room for follow-up search after the initial result batch), 90 seconds, and 8 results per query. Every ceiling is configurable locally and enforced in application code.
- LLM-assisted verification is capped at one priority official source and one auxiliary fact per task call; deterministic field verification still runs over all candidates, and unselected records remain explicitly not analyzed by the LLM.
- A round records a concise, user-safe reason for continuing/stopping, queries, evidence gaps, and before/after source/candidate counts. No hidden chain-of-thought or credentials are persisted.
- Existing result and verification schemas remain authoritative. Model suggestions cannot create a source, result, deadline, eligibility, or verified field. A quote must occur in the fetched evidence before it can support a field.
- Public webpage reads must validate each redirect target, reject private/local/reserved IP destinations, honor robots.txt, cap redirects/time/bytes, and avoid authentication or access-control bypass. PDF parsing is bounded and only for publicly accessible files.

## File ownership for this parallel pass

- Main agent: this contract, `contracts/**`, `qa/**`, `services/ai-engine/app/main.py`, `schemas.py`, `settings.py`, final integration and acceptance.
- Agent 1, Search tools: `services/ai-engine/app/provider.py`, new `services/ai-engine/app/search_tools/**`, and only `services/ai-engine/tests/test_search_tools_v04.py`.
- Agent 2, Tool calling/research loop: `services/ai-engine/app/llm/client.py`, new `services/ai-engine/app/research/loop.py` and `config.py`, and only `services/ai-engine/tests/test_research_loop_v04.py`. Do not edit `research/contracts.py` or the AI Engine HTTP entrypoint.
- Agent 3, UX/task observability: `apps/web/**`, `services/api/app/**`, and only new API/frontend tests under their existing test directories. Do not edit either OpenAPI contract or AI Engine files.

## Coordination

All agents share the existing checkout so current uncommitted work remains present. The ownership boundaries above are disjoint; no reset, checkout, or rebase is allowed. The OpenAPI contracts are maintained by the main agent and will be updated before integration.
