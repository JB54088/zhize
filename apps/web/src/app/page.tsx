"use client";

import { FormEvent, useEffect, useRef, useState } from "react";
import Link from "next/link";

type TaskStatus = "queued" | "running" | "completed" | "partial" | "failed";
type StageKey = "collect" | "clean" | "verify" | "judge";
type StageStatus = "pending" | "running" | "completed" | "partial" | "failed" | "skipped";
type AgentKey = "collect" | "clean" | "verify" | "judge";
type AgentAction = "intent_understanding" | "search" | "extraction" | "source_verification" | "semantic_match" | "final_answer";
type AgentRunStatus = "pending" | "running" | "completed" | "partial" | "skipped_unconfigured" | "skipped_no_evidence" | "failed";
type Decision = "recommended" | "needs_verification" | "not_matched";
type ResultKind = "specific_opportunity" | "organization_lead" | "unknown";
type ConditionStatus = "verified" | "unverified" | "conflict" | "missing";
type EvidenceSourceStatus = "verified" | "unverified" | "pending" | "unknown" | "conflict";
type SourceType = "official" | "third_party" | "unknown";
type SortMode = "comprehensive" | "match" | "evidence";

type StageSnapshot = {
  key: StageKey;
  status: StageStatus;
  started_at: string | null;
  completed_at: string | null;
  duration_ms: number | null;
  input_count: number | null;
  output_count: number | null;
  verified_count: number | null;
  unverified_count: number | null;
  conflict_count: number | null;
  error: string | null;
};

type AgentRun = {
  agent: AgentKey;
  action: AgentAction;
  status: AgentRunStatus;
  started_at: string | null;
  completed_at: string | null;
  duration_ms: number | null;
  latency_ms: number | null;
  model: string | null;
  request_id: string | null;
  prompt_tokens: number | null;
  completion_tokens: number | null;
  total_tokens: number | null;
  estimated_cost_usd: number | null;
  input_summary: string | null;
  output_summary: string | null;
  error_code: string | null;
};

type KeyCondition = {
  label: string;
  value: string;
  status: ConditionStatus;
  evidence_url: string | null;
  quote: string | null;
};

type EvidenceSource = {
  title: string;
  url: string;
  source_type: SourceType;
  status: EvidenceSourceStatus;
};

type AttributeStatus = "verified" | "pending" | "unknown" | "conflict";
type ResultAttribute = {
  key: string;
  label: string;
  value: string | null;
  status: AttributeStatus;
  evidence_url: string | null;
  evidence_quote: string | null;
};

type AdditionalSource = {
  title?: string;
  name?: string;
  url?: string | null;
  source_url?: string | null;
  source_type?: SourceType;
  type?: SourceType;
  status?: EvidenceSourceStatus | AttributeStatus;
  verification_status?: EvidenceSourceStatus | AttributeStatus;
};

type LLMVerificationFact = {
  field: string;
  value: string;
  assessment: "supports" | "contradicts" | "context" | "uncertain";
  evidence_url: string | null;
  quote: string | null;
};

type LLMVerification = {
  status: "analyzed" | "rejected" | "not_analyzed";
  facts: LLMVerificationFact[];
};

type LLMSemanticMatch = {
  status: "analyzed" | "rejected" | "not_analyzed";
  fit: "strong" | "partial" | "weak" | "uncertain" | null;
  score: number | null;
  reason: string | null;
  evidence_quotes: string[];
};

type AnswerClaim = {
  result_id: string;
  text: string;
  evidence_quotes: string[];
  sources: EvidenceSource[];
};

type ResearchRound = {
  round_number: number;
  status: "planned" | "running" | "completed" | "partial" | "stopped" | "failed";
  reason: string;
  planned_queries: string[];
  search_requests: number;
  sources_found: number;
  new_sources: number;
  candidates_before: number;
  candidates_after: number;
  evidence_gaps: string[];
  duration_ms: number | null;
};

type ResearchToolCall = {
  round_number: number;
  name: "search_web" | "search_official" | "read_webpage" | "inspect_evidence";
  status: "running" | "completed" | "partial" | "rejected" | "failed";
  input_summary: string;
  output_summary: string;
  provider: string | null;
  source_count: number | null;
  duration_ms: number | null;
  error_code: string | null;
  model: string | null;
  request_id: string | null;
  prompt_tokens: number | null;
  completion_tokens: number | null;
  total_tokens: number | null;
};

type SearchProviderMetric = {
  provider: string;
  configured: boolean;
  requests: number;
  returned_sources: number;
  failures: number;
  latency_ms: number;
  error_code: string | null;
};

type ResearchOutcome = {
  status: "completed" | "partial" | "stopped" | "failed";
  rounds: ResearchRound[];
  tool_calls: ResearchToolCall[];
  providers: SearchProviderMetric[];
  model_calls: number;
  total_tokens: number | null;
  elapsed_ms: number;
  stop_reason: string;
  partial_results: boolean;
};

type SearchResult = {
  id?: string;
  title: string;
  category?: string | null;
  rank?: number | null;
  recommendation_reason?: string | null;
  attributes?: ResultAttribute[];
  source_url?: string | null;
  additional_sources?: AdditionalSource[];
  last_verified_at?: string | null;
  verification_status?: string | null;
  matching_status?: string | null;
  freshness_status?: string | null;
  url?: string;
  source_name?: string;
  snippet?: string;
  summary?: string;
  evidence?: string;
  evidence_detail?: string;
  match_reason?: string;
  mismatch_reason?: string | null;
  application_url?: string | null;
  result_kind?: ResultKind | null;
  confidence?: number;
  match_score?: number;
  decision?: Decision;
  key_conditions?: KeyCondition[];
  evidence_sources?: EvidenceSource[];
  published_at?: string | null;
  official_source_url?: string | null;
  verified_fields?: string[];
  unverified_fields?: string[];
  purchasing_unit?: string | null;
  budget?: string | null;
  deadline_at?: string | null;
  deadline_status?: string;
  official_source_status?: string;
  evidence_status?: string;
  procurement_category?: string;
  match_status?: string;
  llm_verification?: LLMVerification | null;
  llm_semantic_match?: LLMSemanticMatch | null;
};

type SearchTask = {
  task_id: string;
  query: string;
  status: TaskStatus;
  results: SearchResult[];
  stages: StageSnapshot[];
  agent_runs: AgentRun[];
  research: ResearchOutcome | null;
  research_rounds: ResearchRound[];
  tool_calls: ResearchToolCall[];
  search_metrics: SearchProviderMetric[];
  answer: string | null;
  answer_claims: AnswerClaim[];
  total_results: number | null;
  page: number | null;
  page_size: number | null;
  errors: string[];
  error?: string;
};

type AcceptedTask = { task_id: string; status: TaskStatus };

const API_BASE_URL = (process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://127.0.0.1:8000").replace(/\/$/, "");
const EXAMPLE_QUERIES = [
  "我是 2027 届毕业生，电子信息工程专业，帮我寻找符合条件且正在招聘的校招公司。",
  "寻找福建省近期有明确 AI 系统建设需求且未截止的政府采购项目。",
  "比较适合小团队的项目管理工具，重点看价格和数据安全。",
];

const STAGE_INFO: { key: StageKey; title: string; subtitle: string }[] = [
  { key: "collect", title: "小采", subtitle: "采集来源" },
  { key: "clean", title: "小洗", subtitle: "清洗去重" },
  { key: "verify", title: "小验", subtitle: "核对证据" },
  { key: "judge", title: "小判", subtitle: "判断匹配" },
];

const AGENT_TITLE: Record<AgentKey, string> = {
  collect: "小采",
  clean: "小洗",
  verify: "小验",
  judge: "小判",
};

const ACTION_TITLE: Record<AgentAction, string> = {
  intent_understanding: "需求理解",
  search: "多路搜索",
  extraction: "信息抽取",
  source_verification: "来源核验",
  semantic_match: "需求判断",
  final_answer: "结论生成",
};

const AGENT_RUN_STATUS_LABEL: Record<AgentRunStatus, string> = {
  pending: "等待",
  running: "运行中",
  completed: "已完成",
  partial: "部分完成",
  skipped_unconfigured: "未配置，未调用",
  skipped_no_evidence: "无核验证据，未调用",
  failed: "失败",
};

const RESEARCH_STATUS_LABEL: Record<ResearchRound["status"], string> = {
  planned: "待执行",
  running: "进行中",
  completed: "已完成",
  partial: "部分完成",
  stopped: "已停止",
  failed: "失败",
};

const TOOL_STATUS_LABEL: Record<ResearchToolCall["status"], string> = {
  running: "进行中",
  completed: "已完成",
  partial: "部分完成",
  rejected: "未执行",
  failed: "失败",
};

const TOOL_NAME_LABEL: Record<ResearchToolCall["name"], string> = {
  search_web: "联网搜索",
  search_official: "官网定向搜索",
  read_webpage: "读取网页",
  inspect_evidence: "检查证据",
};

const TASK_STATUS_LABEL: Record<TaskStatus, string> = {
  queued: "排队中",
  running: "处理中",
  completed: "已完成",
  partial: "部分完成",
  failed: "未完成",
};

const STAGE_STATUS_LABEL: Record<StageStatus, string> = {
  pending: "等待",
  running: "处理中",
  completed: "已完成",
  partial: "部分完成",
  failed: "失败",
  skipped: "未执行",
};

const DECISION_LABEL: Record<Decision, string> = {
  recommended: "推荐",
  needs_verification: "待核实",
  not_matched: "不匹配",
};

const SOURCE_TYPE_LABEL: Record<SourceType, string> = {
  official: "官方来源",
  third_party: "第三方来源",
  unknown: "来源类型未确认",
};

const LLM_ASSESSMENT_LABEL = {
  supports: "证据支持",
  contradicts: "证据相反",
  context: "相关上下文",
  uncertain: "无法确定",
} as const;

const LLM_FIT_LABEL = {
  strong: "较强匹配",
  partial: "部分匹配",
  weak: "匹配较弱",
  uncertain: "尚不确定",
} as const;

function isRecord(value: unknown): value is Record<string, unknown> {
  return Boolean(value) && typeof value === "object" && !Array.isArray(value);
}

function nullableNumber(value: unknown): value is number | null {
  return value === null || (typeof value === "number" && Number.isFinite(value));
}

function isStage(value: unknown): value is StageSnapshot {
  if (!isRecord(value)) return false;
  return ["collect", "clean", "verify", "judge"].includes(String(value.key))
    && ["pending", "running", "completed", "partial", "failed", "skipped"].includes(String(value.status))
    && nullableNumber(value.duration_ms)
    && nullableNumber(value.input_count)
    && nullableNumber(value.output_count)
    && nullableNumber(value.verified_count)
    && nullableNumber(value.unverified_count)
    && nullableNumber(value.conflict_count)
    && (value.error === null || typeof value.error === "string");
}

function isAgentRun(value: unknown): value is AgentRun {
  if (!isRecord(value)) return false;
  return ["collect", "clean", "verify", "judge"].includes(String(value.agent))
    && ["intent_understanding", "search", "extraction", "source_verification", "semantic_match", "final_answer"].includes(String(value.action))
    && ["pending", "running", "completed", "partial", "skipped_unconfigured", "skipped_no_evidence", "failed"].includes(String(value.status))
    && nullableNumber(value.duration_ms)
    && nullableNumber(value.latency_ms)
    && nullableNumber(value.prompt_tokens)
    && nullableNumber(value.completion_tokens)
    && nullableNumber(value.total_tokens)
    && nullableNumber(value.estimated_cost_usd)
    && (value.model === null || typeof value.model === "string")
    && (value.input_summary === null || typeof value.input_summary === "string")
    && (value.output_summary === null || typeof value.output_summary === "string")
    && (value.error_code === null || typeof value.error_code === "string");
}

function isResearchRound(value: unknown): value is ResearchRound {
  return isRecord(value)
    && typeof value.round_number === "number" && Number.isInteger(value.round_number) && value.round_number >= 1 && value.round_number <= 3
    && ["planned", "running", "completed", "partial", "stopped", "failed"].includes(String(value.status))
    && typeof value.reason === "string"
    && Array.isArray(value.planned_queries) && value.planned_queries.every((query) => typeof query === "string")
    && [value.search_requests, value.sources_found, value.new_sources, value.candidates_before, value.candidates_after].every((count) => typeof count === "number" && Number.isInteger(count) && count >= 0)
    && Array.isArray(value.evidence_gaps) && value.evidence_gaps.every((gap) => typeof gap === "string")
    && nullableNumber(value.duration_ms);
}

function isResearchToolCall(value: unknown): value is ResearchToolCall {
  return isRecord(value)
    && typeof value.round_number === "number" && Number.isInteger(value.round_number) && value.round_number >= 1
    && ["search_web", "search_official", "read_webpage", "inspect_evidence"].includes(String(value.name))
    && ["running", "completed", "partial", "rejected", "failed"].includes(String(value.status))
    && typeof value.input_summary === "string" && typeof value.output_summary === "string"
    && (value.provider === null || typeof value.provider === "string")
    && nullableNumber(value.source_count) && nullableNumber(value.duration_ms)
    && (value.error_code === null || typeof value.error_code === "string")
    && (value.model === null || typeof value.model === "string")
    && (value.request_id === null || typeof value.request_id === "string")
    && nullableNumber(value.prompt_tokens) && nullableNumber(value.completion_tokens) && nullableNumber(value.total_tokens);
}

function isSearchProviderMetric(value: unknown): value is SearchProviderMetric {
  return isRecord(value)
    && typeof value.provider === "string" && typeof value.configured === "boolean"
    && [value.requests, value.returned_sources, value.failures, value.latency_ms].every((count) => typeof count === "number" && Number.isInteger(count) && count >= 0)
    && (value.error_code === null || typeof value.error_code === "string");
}

function isResearchOutcome(value: unknown): value is ResearchOutcome {
  return isRecord(value)
    && ["completed", "partial", "stopped", "failed"].includes(String(value.status))
    && Array.isArray(value.rounds) && value.rounds.every(isResearchRound)
    && Array.isArray(value.tool_calls) && value.tool_calls.every(isResearchToolCall)
    && Array.isArray(value.providers) && value.providers.every(isSearchProviderMetric)
    && typeof value.model_calls === "number" && Number.isInteger(value.model_calls) && value.model_calls >= 0
    && nullableNumber(value.total_tokens)
    && typeof value.elapsed_ms === "number" && Number.isFinite(value.elapsed_ms) && value.elapsed_ms >= 0
    && typeof value.stop_reason === "string" && typeof value.partial_results === "boolean";
}

function isCondition(value: unknown): value is KeyCondition {
  if (!isRecord(value)) return false;
  return typeof value.label === "string"
    && typeof value.value === "string"
    && ["verified", "unverified", "conflict", "missing"].includes(String(value.status))
    && (value.evidence_url === null || typeof value.evidence_url === "string")
    && (value.quote === null || typeof value.quote === "string");
}

function isEvidenceSource(value: unknown): value is EvidenceSource {
  if (!isRecord(value)) return false;
  return typeof value.title === "string"
    && typeof value.url === "string"
    && ["official", "third_party", "unknown"].includes(String(value.source_type))
    && ["verified", "unverified", "pending", "unknown", "conflict"].includes(String(value.status));
}

function isResultAttribute(value: unknown): value is ResultAttribute {
  return isRecord(value)
    && typeof value.key === "string"
    && typeof value.label === "string"
    && (value.value === null || typeof value.value === "string")
    && ["verified", "pending", "unknown", "conflict"].includes(String(value.status))
    && (value.evidence_url === null || typeof value.evidence_url === "string")
    && (value.evidence_quote === null || typeof value.evidence_quote === "string");
}

function isAdditionalSource(value: unknown): value is AdditionalSource {
  if (!isRecord(value)) return false;
  return ["url", "source_url"].every((key) => value[key] === undefined || value[key] === null || typeof value[key] === "string")
    && ["title", "name"].every((key) => value[key] === undefined || typeof value[key] === "string")
    && ["source_type", "type"].every((key) => value[key] === undefined || ["official", "third_party", "unknown"].includes(String(value[key])))
    && ["status", "verification_status"].every((key) => value[key] === undefined || ["verified", "unverified", "pending", "unknown", "conflict"].includes(String(value[key])));
}

function isLLMVerification(value: unknown): value is LLMVerification {
  if (!isRecord(value) || !["analyzed", "rejected", "not_analyzed"].includes(String(value.status))) return false;
  return Array.isArray(value.facts) && value.facts.every((fact) => isRecord(fact)
    && typeof fact.field === "string"
    && typeof fact.value === "string"
    && ["supports", "contradicts", "context", "uncertain"].includes(String(fact.assessment))
    && (fact.evidence_url === null || typeof fact.evidence_url === "string")
    && (fact.quote === null || typeof fact.quote === "string"));
}

function isLLMSemanticMatch(value: unknown): value is LLMSemanticMatch {
  return isRecord(value)
    && ["analyzed", "rejected", "not_analyzed"].includes(String(value.status))
    && (value.fit === undefined || value.fit === null || ["strong", "partial", "weak", "uncertain"].includes(String(value.fit)))
    && (value.score === undefined || value.score === null || (typeof value.score === "number" && Number.isFinite(value.score) && value.score >= 0 && value.score <= 1))
    && (value.reason === undefined || value.reason === null || typeof value.reason === "string")
    && (value.evidence_quotes === undefined || (Array.isArray(value.evidence_quotes) && value.evidence_quotes.every((quote) => typeof quote === "string")));
}

function isAnswerClaim(value: unknown): value is AnswerClaim {
  return isRecord(value)
    && typeof value.result_id === "string"
    && typeof value.text === "string"
    && Array.isArray(value.evidence_quotes) && value.evidence_quotes.every((quote) => typeof quote === "string")
    && Array.isArray(value.sources) && value.sources.every(isEvidenceSource);
}

function isSearchResult(value: unknown): value is SearchResult {
  if (!isRecord(value) || typeof value.title !== "string") return false;
  if (value.category !== undefined && value.category !== null && typeof value.category !== "string") return false;
  if (value.rank !== undefined && !nullableNumber(value.rank)) return false;
  if (value.recommendation_reason !== undefined && value.recommendation_reason !== null && typeof value.recommendation_reason !== "string") return false;
  if (value.attributes !== undefined && (!Array.isArray(value.attributes) || !value.attributes.every(isResultAttribute))) return false;
  if (value.source_url !== undefined && value.source_url !== null && typeof value.source_url !== "string") return false;
  if (value.additional_sources !== undefined && (!Array.isArray(value.additional_sources) || !value.additional_sources.every(isAdditionalSource))) return false;
  if (["last_verified_at", "verification_status", "matching_status", "freshness_status"].some((key) => value[key] !== undefined && value[key] !== null && typeof value[key] !== "string")) return false;
  if (value.decision !== undefined && !["recommended", "needs_verification", "not_matched"].includes(String(value.decision))) return false;
  if (value.result_kind !== undefined && value.result_kind !== null && !["specific_opportunity", "organization_lead", "unknown"].includes(String(value.result_kind))) return false;
  if (value.application_url !== undefined && value.application_url !== null && typeof value.application_url !== "string") return false;
  if (value.llm_verification !== undefined && value.llm_verification !== null && !isLLMVerification(value.llm_verification)) return false;
  if (value.llm_semantic_match !== undefined && value.llm_semantic_match !== null && !isLLMSemanticMatch(value.llm_semantic_match)) return false;
  if (value.key_conditions !== undefined && (!Array.isArray(value.key_conditions) || !value.key_conditions.every(isCondition))) return false;
  if (value.evidence_sources !== undefined && (!Array.isArray(value.evidence_sources) || !value.evidence_sources.every(isEvidenceSource))) return false;
  return true;
}

function isSearchTask(value: unknown): value is SearchTask {
  if (!isRecord(value)) return false;
  return typeof value.task_id === "string"
    && typeof value.query === "string"
    && ["queued", "running", "completed", "partial", "failed"].includes(String(value.status))
    && Array.isArray(value.results)
    && value.results.every(isSearchResult)
    && (value.stages === undefined || (Array.isArray(value.stages) && value.stages.every(isStage)))
    && (value.agent_runs === undefined || (Array.isArray(value.agent_runs) && value.agent_runs.every(isAgentRun)))
    && (value.research === undefined || value.research === null || isResearchOutcome(value.research))
    && (value.research_rounds === undefined || (Array.isArray(value.research_rounds) && value.research_rounds.every(isResearchRound)))
    && (value.tool_calls === undefined || (Array.isArray(value.tool_calls) && value.tool_calls.every(isResearchToolCall)))
    && (value.search_metrics === undefined || (Array.isArray(value.search_metrics) && value.search_metrics.every(isSearchProviderMetric)))
    && (value.answer === undefined || value.answer === null || typeof value.answer === "string")
    && (value.answer_claims === undefined || (Array.isArray(value.answer_claims) && value.answer_claims.every(isAnswerClaim)));
}

function isAcceptedTask(value: unknown): value is AcceptedTask {
  return isRecord(value)
    && typeof value.task_id === "string"
    && ["queued", "running", "completed", "partial", "failed"].includes(String(value.status));
}

function normalizeTask(value: SearchTask): SearchTask {
  const raw = value as SearchTask & { stages?: StageSnapshot[]; agent_runs?: AgentRun[]; answer?: string | null; answer_claims?: AnswerClaim[]; total_results?: number; page?: number; page_size?: number; errors?: unknown[]; research?: ResearchOutcome | null; research_rounds?: ResearchRound[]; tool_calls?: ResearchToolCall[]; search_metrics?: SearchProviderMetric[] };
  return {
    ...value,
    stages: Array.isArray(raw.stages) ? raw.stages : [],
    agent_runs: Array.isArray(raw.agent_runs) ? raw.agent_runs : [],
    research: raw.research ?? null,
    research_rounds: Array.isArray(raw.research_rounds) ? raw.research_rounds : raw.research?.rounds ?? [],
    tool_calls: Array.isArray(raw.tool_calls) ? raw.tool_calls : raw.research?.tool_calls ?? [],
    search_metrics: Array.isArray(raw.search_metrics) ? raw.search_metrics : raw.research?.providers ?? [],
    answer: typeof raw.answer === "string" && raw.answer.trim() ? raw.answer : null,
    answer_claims: Array.isArray(raw.answer_claims) ? raw.answer_claims : [],
    total_results: typeof raw.total_results === "number" && Number.isFinite(raw.total_results) ? raw.total_results : null,
    page: typeof raw.page === "number" && Number.isFinite(raw.page) ? raw.page : null,
    page_size: typeof raw.page_size === "number" && Number.isFinite(raw.page_size) ? raw.page_size : null,
    errors: Array.isArray(raw.errors) ? raw.errors.map((item) => typeof item === "string" ? item : "").filter(Boolean) : [],
  };
}

function formatDate(value?: string | null) {
  if (!value) return null;
  const chineseDate = value.match(/^(\d{4})年\s*(\d{1,2})月\s*(\d{1,2})日(?:\s*(\d{1,2}):(\d{2}))?/);
  if (chineseDate) {
    const [, yearText, monthText, dayText, hourText, minuteText] = chineseDate;
    const year = Number(yearText);
    const month = Number(monthText);
    const day = Number(dayText);
    const date = new Date(year, month - 1, day);
    if (date.getFullYear() !== year || date.getMonth() !== month - 1 || date.getDate() !== day) return null;
    const formatted = new Intl.DateTimeFormat("zh-CN", { year: "numeric", month: "short", day: "numeric" }).format(date);
    return hourText && minuteText ? formatted + " " + hourText.padStart(2, "0") + ":" + minuteText : formatted;
  }
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return null;
  return new Intl.DateTimeFormat("zh-CN", { year: "numeric", month: "short", day: "numeric" }).format(date);
}

function safeSourceUrl(value?: string | null) {
  if (!value) return null;
  try {
    const parsed = new URL(value);
    if ((parsed.protocol !== "https:" && parsed.protocol !== "http:") || parsed.username || parsed.password) return null;
    const hostname = parsed.hostname.toLowerCase().replace(/^\[|\]$/g, "");
    if (!hostname || hostname === "localhost" || hostname.endsWith(".localhost") || hostname.endsWith(".local") || !hostname.includes(".")) return null;
    const ipv4 = hostname.match(/^(\d{1,3})\.(\d{1,3})\.(\d{1,3})\.(\d{1,3})$/);
    if (ipv4) {
      const octets = ipv4.slice(1).map(Number);
      if (octets.some((part) => part > 255)
        || octets[0] === 0 || octets[0] === 10 || octets[0] === 127
        || (octets[0] === 169 && octets[1] === 254)
        || (octets[0] === 172 && octets[1] >= 16 && octets[1] <= 31)
        || (octets[0] === 192 && octets[1] === 168)) return null;
    }
    if (hostname === "::1" || hostname.startsWith("fc") || hostname.startsWith("fd") || hostname.startsWith("fe80:")) return null;
    return parsed.href;
  } catch {
    return null;
  }
}

function durationLabel(duration?: number | null) {
  if (typeof duration !== "number" || !Number.isFinite(duration)) return "未提供";
  if (duration < 1000) return Math.max(0, Math.round(duration)) + " 毫秒";
  return (duration / 1000).toFixed(1) + " 秒";
}

function agentTokens(run: AgentRun) {
  if (typeof run.total_tokens === "number") return run.total_tokens;
  if (typeof run.prompt_tokens === "number" && typeof run.completion_tokens === "number") {
    return run.prompt_tokens + run.completion_tokens;
  }
  return null;
}

function formatCost(value?: number | null) {
  if (typeof value !== "number" || !Number.isFinite(value)) return null;
  return "$" + value.toFixed(value < 0.01 ? 6 : 4);
}

function countLabel(value?: number | null) {
  return typeof value === "number" && Number.isFinite(value) ? String(value) : "未提供";
}

function taskDecision(result: SearchResult): Decision {
  if (result.decision) return result.decision;
  const status = (result.matching_status || result.match_status || "").toLowerCase();
  if (["matched", "recommended", "match", "符合"].includes(status)) return "recommended";
  if (["not_matched", "unmatched", "not_recommended", "不匹配"].includes(status)) return "not_matched";
  return "needs_verification";
}

function resultIdentity(result: SearchResult) {
  return result.id || result.source_url || result.url || (result.category || "") + "-" + result.title;
}

function decisionRank(result: SearchResult) {
  const decision = taskDecision(result);
  return decision === "recommended" ? 0 : decision === "needs_verification" ? 1 : 2;
}

function evidenceRank(result: SearchResult) {
  const sources = resultSources(result);
  const verifiedSources = sources.filter((source) => source.status === "verified").length;
  const verifiedConditions = result.attributes?.filter((attribute) => attribute.status === "verified").length
    ?? (result.key_conditions ?? []).filter((condition) => condition.status === "verified").length;
  return verifiedSources * 3 + verifiedConditions;
}

function resultMatchScore(result: SearchResult) {
  if (typeof result.match_score === "number" && Number.isFinite(result.match_score)) return result.match_score;
  return -1;
}

type ResultDomain = "restaurant" | "recruitment" | "procurement" | "product" | "travel" | "policy" | "other";

function resultDomain(result: SearchResult, query: string): ResultDomain {
  const category = (result.category || "").toLowerCase();
  if (/restaurant|food|dining|餐厅|餐饮/.test(category)) return "restaurant";
  if (/employ|recruit|hiring|job|career|招聘|校招/.test(category)) return "recruitment";
  if (/procurement|tender|government|采购|招标/.test(category)) return "procurement";
  if (/product|shopping|商品|产品/.test(category)) return "product";
  if (/travel|hotel|trip|旅游|旅行|酒店/.test(category)) return "travel";
  if (/policy|regulation|政策|法规/.test(category)) return "policy";
  const text = query.toLowerCase();
  if (/restaurant|food|dining|餐厅|餐饮|午餐|晚餐/.test(text)) return "restaurant";
  if (/employ|recruit|hiring|job|career|校招|招聘|岗位|毕业生/.test(text)) return "recruitment";
  if (/procurement|tender|government|采购|招标|投标/.test(text)) return "procurement";
  if (/product|shopping|商品|产品|购物/.test(text)) return "product";
  if (/travel|hotel|trip|旅游|旅行|酒店/.test(text)) return "travel";
  if (/policy|regulation|政策|法规/.test(text)) return "policy";
  if (result.purchasing_unit || result.budget || result.deadline_at || result.procurement_category) return "procurement";
  return "other";
}

function attributeStatus(status: ResultAttribute["status"] | ConditionStatus): ResultAttribute["status"] {
  if (status === "unverified") return "pending";
  if (status === "missing") return "unknown";
  return status;
}

function forbiddenAttribute(domain: ResultDomain, field: string) {
  const text = field.toLowerCase();
  const groups: Partial<Record<ResultDomain, RegExp[]>> = {
    restaurant: [/采购|招标|预算|投标|purchase|tender|procurement/, /招聘|校招|毕业届|学历|专业|薪资|岗位|recruit|salary|degree|major/, /deadline|响应截止|投标截止|报名截止|截止时间|截止日期/],
    recruitment: [/采购|招标|预算|投标|purchase|tender|procurement/, /人均|餐厅|营业|菜品|近期评价|restaurant|dining|rating/],
    procurement: [/招聘|校招|毕业届|学历|专业|薪资|岗位|recruit|salary|degree|major/, /人均|餐厅|营业|菜品|近期评价|restaurant|dining|rating/],
    product: [/采购单位|采购预算|响应截止|采购项目|tender|procurement/, /招聘|校招|毕业届|学历|专业|薪资|岗位|recruit|salary|degree|major/, /人均|餐厅|营业|菜品|近期评价|restaurant|dining|rating/, /deadline|响应截止|投标截止|报名截止|截止时间|截止日期/],
  };
  return groups[domain]?.some((pattern) => pattern.test(text)) ?? false;
}

function resultAttributes(result: SearchResult, query: string): ResultAttribute[] {
  const domain = resultDomain(result, query);
  if (result.attributes?.length) {
    return result.attributes.filter((attribute) => !forbiddenAttribute(domain, attribute.key + " " + attribute.label));
  }
  if (result.key_conditions?.length) {
    return result.key_conditions
      .filter((condition) => !forbiddenAttribute(domain, condition.label))
      .map((condition) => ({
        key: condition.label,
        label: condition.label,
        value: condition.value || null,
        status: attributeStatus(condition.status),
        evidence_url: condition.evidence_url,
        evidence_quote: condition.quote,
      }));
  }
  // Legacy procurement fields are only meaningful for an explicitly inferred procurement query.
  if (domain !== "procurement") return [];
  const verified = result.verified_fields ?? [];
  const unverified = result.unverified_fields ?? [];
  const legacy: { key: string; label: string; value?: string | null }[] = [
    { key: "purchasing_unit", label: "采购单位", value: result.purchasing_unit },
    { key: "budget", label: "预算", value: result.budget },
    { key: "published_at", label: "发布日期", value: formatDate(result.published_at) },
    { key: "deadline", label: "截止时间", value: formatDate(result.deadline_at) },
  ];
  return legacy.filter((item) => Boolean(item.value) || verified.includes(item.key) || unverified.includes(item.key))
    .map((item) => ({
      key: item.key,
      label: item.label,
      value: item.value || null,
      status: verified.includes(item.key) ? "verified" : unverified.includes(item.key) ? "pending" : "unknown",
      evidence_url: result.official_source_url || result.url || null,
      evidence_quote: null,
    }));
}

function resultSources(result: SearchResult): EvidenceSource[] {
  const sources: EvidenceSource[] = [...(result.evidence_sources ?? [])];
  for (const item of result.additional_sources ?? []) {
    const url = safeSourceUrl(item.url || item.source_url);
    if (!url || sources.some((source) => safeSourceUrl(source.url) === url)) continue;
    const status = item.status || item.verification_status || "unknown";
    sources.push({
      title: item.title || item.name || "补充来源",
      url,
      source_type: item.source_type || item.type || "unknown",
      status: status === "verified" ? "verified" : status === "conflict" ? "conflict" : status === "pending" || status === "unverified" ? "pending" : "unknown",
    });
  }
  const primaryUrl = safeSourceUrl(result.source_url || result.url);
  const officialUrl = safeSourceUrl(result.official_source_url);
  const officialVerified = result.official_source_status === "verified";
  const primaryIsVerified = Boolean(primaryUrl && (
    (result.source_url === undefined && officialVerified && primaryUrl === officialUrl)
    || sources.some((source) => safeSourceUrl(source.url) === primaryUrl && source.status === "verified")
  ));
  if (primaryUrl && !sources.some((source) => safeSourceUrl(source.url) === primaryUrl)) {
    sources.push({
      title: result.source_name || "主要来源",
      url: primaryUrl,
      source_type: officialVerified && primaryUrl === officialUrl
        ? "official"
        : result.official_source_status === "third_party" ? "third_party" : "unknown",
      status: primaryIsVerified ? "verified" : "unverified",
    });
  }
  if (officialUrl && officialUrl !== primaryUrl) {
    sources.push({
      title: officialVerified ? "官方公告" : "待核实的来源候选",
      url: officialUrl,
      source_type: officialVerified
        ? "official"
        : result.official_source_status === "third_party" ? "third_party" : "unknown",
      status: officialVerified ? "verified" : "unverified",
    });
  }
  return sources;
}

function primarySource(result: SearchResult) {
  const sources = resultSources(result);
  const explicitPrimary = safeSourceUrl(result.source_url);
  const isVerifiedSource = (url: string | null) => Boolean(url && sources.some((source) => safeSourceUrl(source.url) === url && source.status === "verified"));
  const verifiedExplicit = isVerifiedSource(explicitPrimary) ? explicitPrimary : null;
  const verifiedSource = sources.find((source) => source.status === "verified" && source.source_type === "official" && safeSourceUrl(source.url))
    || sources.find((source) => source.status === "verified" && safeSourceUrl(source.url));
  const legacyOfficial = result.official_source_status === "verified" ? safeSourceUrl(result.official_source_url) : null;
  const url = verifiedExplicit || legacyOfficial || (verifiedSource ? safeSourceUrl(verifiedSource.url) : null) || explicitPrimary || safeSourceUrl(result.url) || safeSourceUrl(result.official_source_url);
  const verified = Boolean(url && (
    (result.official_source_status === "verified" && url === safeSourceUrl(result.official_source_url))
    || isVerifiedSource(url)
  ));
  return { url: verified ? url : null, available: Boolean(url), verified, name: result.source_name || "主要来源" };
}

function categoryLabel(result: SearchResult, query: string) {
  if (result.category?.trim()) {
    const labels: Record<string, string> = {
      dining: "餐饮", restaurant: "餐饮", restaurants: "餐饮", food: "餐饮",
      employment: "招聘", recruitment: "招聘", job: "招聘", jobs: "招聘",
      government: "政府采购", government_procurement: "政府采购", procurement: "政府采购",
      product: "商品", travel: "旅行", policy: "政策", other: "其他信息",
    };
    return labels[result.category.toLowerCase()] || result.category;
  }
  const labels: Record<ResultDomain, string> = {
    restaurant: "餐饮", recruitment: "招聘", procurement: "政府采购", product: "商品",
    travel: "旅行", policy: "政策", other: "信息",
  };
  return labels[resultDomain(result, query)];
}

function attributeStatusLabel(status: ResultAttribute["status"]) {
  if (status === "verified") return "已核实";
  if (status === "pending") return "待核实";
  if (status === "conflict") return "有冲突";
  return "未知";
}

function matchingStatusLabel(result: SearchResult) {
  const raw = (result.matching_status || "").toLowerCase();
  if (raw.includes("recommend") || raw === "matched") return "推荐";
  if (raw.includes("not_match") || raw === "unmatched") return "不匹配";
  return DECISION_LABEL[taskDecision(result)];
}

function verificationStatusLabel(result: SearchResult) {
  const status = (result.verification_status || "").toLowerCase();
  if (["verified", "official_verified", "已核实"].includes(status)) return "信息已核实";
  if (status === "conflict" || status === "存在冲突") return "信息存在冲突";
  if (status === "pending" || status === "unverified" || status === "待核实") return "信息待核实";
  if (status === "unknown" || status === "未知") return "信息核验状态未知";
  return null;
}

function freshnessStatusLabel(result: SearchResult) {
  const status = (result.freshness_status || "").toLowerCase();
  if (["fresh", "current", "active", "近期", "有效"].includes(status)) return "信息较新";
  if (["stale", "expired", "outdated", "过期", "已截止"].includes(status)) return "信息可能已过期";
  if (["pending", "unknown", "待核实", "未知"].includes(status)) return "时效待核实";
  return null;
}

function shortSummary(result: SearchResult) {
  const text = (result.summary || result.snippet || "").trim();
  if (!text) return "暂无简要摘要；请查看来源与证据详情。";
  if (text.length <= 180) return text;
  return text.slice(0, 177).trimEnd() + "…";
}

function sourceStatusLabel(status: EvidenceSourceStatus) {
  if (status === "verified") return "已核实";
  if (status === "conflict") return "有冲突";
  if (status === "unknown") return "未知";
  return "待核实";
}

function resultKindLabel(kind?: ResultKind | null) {
  if (kind === "specific_opportunity") return "具体岗位 / 项目";
  if (kind === "organization_lead") return "机构主页线索";
  return "机会类型待核实";
}

function ResultCardView({ result, query, active }: { result: SearchResult; query: string; active: boolean }) {
  const decision = taskDecision(result);
  const attributes = resultAttributes(result, query);
  const sources = resultSources(result);
  const primary = primarySource(result);
  const reason = result.recommendation_reason || (decision === "not_matched" ? result.mismatch_reason : null) || result.match_reason || "当前结果未提供匹配理由。";
  const details = (result.evidence_detail || result.evidence || "").trim();
  const resultId = resultIdentity(result);
  const published = formatDate(result.published_at);
  const verifiedAt = formatDate(result.last_verified_at);
  const cardContent = (
    <>
      <div className="result-card-heading">
        <span className={"decision-badge decision-" + decision}>{matchingStatusLabel(result)}</span>
        {active && <span className="candidate-badge">候选线索 · 研究继续中</span>}
        <span className="result-kind">{categoryLabel(result, query)}</span>
        {result.result_kind && <span className={"result-kind result-kind-" + result.result_kind}>{resultKindLabel(result.result_kind)}</span>}
      </div>
      <h4>{result.title}</h4>
      <p className="result-summary">{shortSummary(result)}</p>
      <div className="result-reason"><span>判断理由</span><p>{reason}</p></div>
      {attributes.length > 0 && (
        <dl className="card-attributes" aria-label="结果关键信息">
          {attributes.map((attribute, attributeIndex) => (
            <div className="card-attribute" key={attribute.key + "-" + attributeIndex}>
              <dt>{attribute.label}</dt>
              <dd>{attribute.value?.trim() || "未提供"}<span className={"attribute-status attribute-status-" + attribute.status}>{attributeStatusLabel(attribute.status)}</span></dd>
            </div>
          ))}
        </dl>
      )}
      {(published || verifiedAt || verificationStatusLabel(result) || freshnessStatusLabel(result)) && (
        <div className="card-meta">
          {published && <span>发布时间：{published}</span>}
          {verifiedAt && <span>最近核验：{verifiedAt}</span>}
          {verificationStatusLabel(result) && <span>{verificationStatusLabel(result)}</span>}
          {freshnessStatusLabel(result) && <span>{freshnessStatusLabel(result)}</span>}
        </div>
      )}
      <div className="card-primary-source">
        <span>{primary.name}</span>
        {primary.url
          ? <strong>已核实 · 打开原始来源 ↗</strong>
          : <strong className="card-source-disabled">{primary.available ? "来源尚未核实，暂不可跳转" : "没有可用的安全来源链接"}</strong>}
      </div>
    </>
  );

  return (
    <article className="result-card" data-result-id={resultId}>
      {primary.url
        ? <a className="result-card-primary" href={primary.url} target="_blank" rel="noopener noreferrer" aria-label={`打开已核实来源：${result.title}`}>{cardContent}</a>
        : <div className="result-card-primary result-card-primary-disabled">{cardContent}</div>}
      <details className="evidence-details card-evidence">
        <summary>查看证据与其他来源</summary>
        {attributes.length > 0 && (
          <section className="card-evidence-section">
            <span className="block-label">字段核验</span>
            <ul className="card-evidence-list">
              {attributes.map((attribute, attributeIndex) => {
                const href = safeSourceUrl(attribute.evidence_url);
                return <li key={attribute.key + "-evidence-" + attributeIndex}><div><strong>{attribute.label}</strong><span>{attributeStatusLabel(attribute.status)}</span></div><p>{attribute.value || "未提供"}</p>{attribute.evidence_quote && <blockquote>“{attribute.evidence_quote}”</blockquote>}{href && <a href={href} target="_blank" rel="noopener noreferrer">查看字段来源 ↗</a>}</li>;
              })}
            </ul>
          </section>
        )}
        <section className="card-evidence-section">
          <span className="block-label">来源与核验状态</span>
          {sources.length > 0 ? (
            <ul className="source-list">
              {sources.map((source, sourceIndex) => {
                const href = safeSourceUrl(source.url);
                return <li key={source.url + "-" + sourceIndex}>{href ? <a href={href} target="_blank" rel="noopener noreferrer">{source.title || "查看来源原文"} <span aria-hidden="true">↗</span></a> : <span className="source-invalid">{source.title || "来源链接不可用"}</span>}<span className={"source-kind source-kind-" + source.source_type}>{SOURCE_TYPE_LABEL[source.source_type]}</span><span className={"source-trust source-trust-" + source.status}>{sourceStatusLabel(source.status)}</span></li>;
              })}
            </ul>
          ) : <p className="missing-data">没有提供可追溯的补充来源。</p>}
          {result.application_url && (safeSourceUrl(result.application_url)
            ? <a className="application-link" href={safeSourceUrl(result.application_url) ?? undefined} target="_blank" rel="noopener noreferrer">打开投递链接 ↗</a>
            : <p className="application-link-missing">投递链接无效或不安全，已禁用。</p>)}
        </section>
        {details && <section className="card-evidence-section"><span className="block-label">原始网页内容</span><div className="evidence-detail-body">{details}</div></section>}
        {(result.llm_verification || result.llm_semantic_match) && (
          <details className="model-analysis-details">
            <summary>查看模型辅助分析</summary>
            <section className="llm-analysis" aria-label="模型辅助分析">
              <div className="llm-analysis-heading"><strong>模型辅助分析</strong><span>参考信息，不替代规则核验与最终裁决</span></div>
              {result.llm_verification && <div className="llm-verification"><span className="block-label">证据理解 · {result.llm_verification.status === "analyzed" ? "已分析" : result.llm_verification.status === "rejected" ? "未通过核验" : "未分析"}</span>{result.llm_verification.facts.length ? <ul className="llm-fact-list">{result.llm_verification.facts.map((fact, factIndex) => { const href = safeSourceUrl(fact.evidence_url); return <li key={fact.field + factIndex}><div><strong>{fact.field}</strong><span>{LLM_ASSESSMENT_LABEL[fact.assessment]}</span></div><p>{fact.value}</p>{fact.quote && <blockquote>{fact.quote}</blockquote>}{href && <a href={href} target="_blank" rel="noopener noreferrer">查看证据 ↗</a>}</li>; })}</ul> : <p className="llm-empty">没有可展示的字段分析。</p>}</div>}
              {result.llm_semantic_match && <div className="llm-match"><div><span className="block-label">语义匹配 · {result.llm_semantic_match.status === "analyzed" ? "已分析" : result.llm_semantic_match.status === "rejected" ? "未通过校验" : "未分析"}</span>{result.llm_semantic_match.fit && <span>{LLM_FIT_LABEL[result.llm_semantic_match.fit]}</span>}</div>{result.llm_semantic_match.reason && <p>{result.llm_semantic_match.reason}</p>}{result.llm_semantic_match.evidence_quotes.length > 0 && <ul>{result.llm_semantic_match.evidence_quotes.map((quote, quoteIndex) => <li key={quoteIndex}>“{quote}”</li>)}</ul>}</div>}
            </section>
          </details>
        )}
      </details>
    </article>
  );
}

export default function Home() {
  const [query, setQuery] = useState("");
  const [task, setTask] = useState<SearchTask | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const [loadingPage, setLoadingPage] = useState(false);
  const [requestError, setRequestError] = useState("");
  const [sortMode, setSortMode] = useState<SortMode>("comprehensive");
  const [visibleCount, setVisibleCount] = useState(10);
  const requestSequence = useRef(0);
  const submittedQuery = useRef<string | null>(null);
  const taskId = task?.task_id;
  const taskStatus = task?.status;

  useEffect(() => {
    if (taskStatus !== "completed" && taskStatus !== "partial" && taskStatus !== "failed") return;
    if (submittedQuery.current && query.trim() === submittedQuery.current) {
      setQuery("");
    }
    submittedQuery.current = null;
  }, [query, taskStatus]);

  useEffect(() => {
    const activeTaskId = taskId ?? "";
    if (!activeTaskId || (taskStatus !== "queued" && taskStatus !== "running")) return;
    let cancelled = false;
    const sequence = requestSequence.current;

    async function refreshTask() {
      try {
        const response = await fetch(API_BASE_URL + "/api/v1/search/" + encodeURIComponent(activeTaskId), { cache: "no-store" });
        if (!response.ok) throw new Error("任务状态请求失败（" + response.status + "）");
        const payload: unknown = await response.json();
        if (!isSearchTask(payload)) throw new Error("服务返回了无法识别的任务状态。");
        if (!cancelled && sequence === requestSequence.current) {
          setTask(normalizeTask(payload));
          setRequestError("");
        }
      } catch (error) {
        if (!cancelled && sequence === requestSequence.current) {
          setRequestError(error instanceof Error ? error.message : "暂时无法获取任务状态。");
        }
      }
    }

    void refreshTask();
    const interval = window.setInterval(() => void refreshTask(), 1600);
    return () => {
      cancelled = true;
      window.clearInterval(interval);
    };
  }, [taskId, taskStatus]);

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const normalizedQuery = query.trim();
    if (!normalizedQuery || submitting) return;
    requestSequence.current += 1;
    setSubmitting(true);
    setRequestError("");
    setTask(null);
    setSortMode("comprehensive");
    setVisibleCount(10);

    try {
      const response = await fetch(API_BASE_URL + "/api/v1/search", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: normalizedQuery, limit: 10 }),
      });
      if (!response.ok) {
        const message = response.status === 422
          ? "问题内容不符合要求，请调整后再试。"
          : "无法创建搜索任务（" + response.status + "）。请确认服务已启动后重试。";
        throw new Error(message);
      }
      const acceptedPayload: unknown = await response.json();
      if (!isAcceptedTask(acceptedPayload)) throw new Error("服务返回了无法识别的任务信息。");
      const detailResponse = await fetch(API_BASE_URL + "/api/v1/search/" + encodeURIComponent(acceptedPayload.task_id), { cache: "no-store" });
      if (!detailResponse.ok) throw new Error("任务状态请求失败（" + detailResponse.status + "）");
      const detailPayload: unknown = await detailResponse.json();
      if (!isSearchTask(detailPayload)) throw new Error("服务返回了无法识别的任务状态。");
      submittedQuery.current = normalizedQuery;
      setTask(normalizeTask(detailPayload));
    } catch (error) {
      setRequestError(error instanceof Error ? error.message : "暂时无法连接知择服务。");
    } finally {
      setSubmitting(false);
    }
  }

  async function loadPage(nextPage: number) {
    if (!task || loadingPage || nextPage < 1) return;
    const pageSize = task.page_size ?? Math.max(task.results.length, 1);
    setLoadingPage(true);
    setRequestError("");
    try {
      const search = new URLSearchParams({ page: String(nextPage), page_size: String(pageSize) });
      const response = await fetch(API_BASE_URL + "/api/v1/search/" + encodeURIComponent(task.task_id) + "?" + search.toString(), { cache: "no-store" });
      if (!response.ok) throw new Error("结果分页请求失败（" + response.status + "）");
      const payload: unknown = await response.json();
      if (!isSearchTask(payload)) throw new Error("服务返回了无法识别的任务结果。");
      const nextTask = normalizeTask(payload);
      setTask((current) => {
        if (!current || current.task_id !== nextTask.task_id) return nextTask;
        const seen = new Set(current.results.map(resultIdentity));
        const appended = nextTask.results.filter((result) => {
          const key = resultIdentity(result);
          if (seen.has(key)) return false;
          seen.add(key);
          return true;
        });
        return { ...nextTask, results: [...current.results, ...appended] };
      });
      setVisibleCount((count) => count + 10);
    } catch (error) {
      setRequestError(error instanceof Error ? error.message : "暂时无法加载下一页。");
    } finally {
      setLoadingPage(false);
    }
  }

  const active = task?.status === "queued" || task?.status === "running";
  const showingEarlyResults = Boolean(active && task?.results.length);
  const hasTask = Boolean(task);
  const stageByKey = new Map((task?.stages ?? []).map((stage) => [stage.key, stage]));
  const agentRuns = task?.agent_runs ?? [];
  const researchRounds = task?.research_rounds ?? [];
  const researchToolCalls = task?.tool_calls ?? [];
  const searchMetrics = task?.search_metrics ?? [];
  const returnedSourceCount = searchMetrics.length
    ? searchMetrics.reduce((sum, metric) => sum + metric.returned_sources, 0)
    : null;
  const tokenRuns = agentRuns.map(agentTokens).filter((value): value is number => value !== null);
  const totalTokens = tokenRuns.length ? tokenRuns.reduce((sum, value) => sum + value, 0) : null;
  const pageDecisionCounts = (task?.results ?? []).reduce<Record<Decision, number>>((counts, result) => {
    counts[taskDecision(result)] += 1;
    return counts;
  }, { recommended: 0, needs_verification: 0, not_matched: 0 });
  const resultRank = (result: SearchResult) => typeof result.rank === "number" ? result.rank : Number.POSITIVE_INFINITY;
  const results = [...(task?.results ?? [])].sort((a, b) => {
    if (sortMode === "match") return decisionRank(a) - decisionRank(b) || resultMatchScore(b) - resultMatchScore(a) || resultRank(a) - resultRank(b);
    if (sortMode === "evidence") return evidenceRank(b) - evidenceRank(a) || decisionRank(a) - decisionRank(b) || resultRank(a) - resultRank(b);
    return resultRank(a) - resultRank(b) || decisionRank(a) - decisionRank(b) || evidenceRank(b) - evidenceRank(a) || resultMatchScore(b) - resultMatchScore(a);
  });
  const visibleResults = results.slice(0, visibleCount);
  const currentPage = task?.page ?? 1;
  const pageSize = task?.page_size ?? (task?.results.length ? task.results.length : null);
  const totalResults = task?.total_results;
  const pageCount = typeof totalResults === "number" && pageSize && pageSize > 0 ? Math.max(1, Math.ceil(totalResults / pageSize)) : null;
  const hasMoreLoaded = results.length > visibleCount;
  const canLoadMorePage = pageCount !== null && currentPage < pageCount;

  return (
    <main className={"app-shell" + (hasTask ? " has-task" : "")}>
      <header className="topbar">
        <Link className="brand" href="/" onClick={() => { setTask(null); setRequestError(""); }} aria-label="知择首页">
          <span className="brand-mark" aria-hidden="true">知</span>
          <span>知择</span>
        </Link>
        <span className="brand-promise">信息万千，知而善择</span>
      </header>

      <section className={"search-hero" + (hasTask ? " search-hero-compact" : "")}>
        {!hasTask && (
          <div className="hero-copy">
            <h1>知择</h1>
            <p>信息万千，知而善择</p>
            <span>让重要的信息，更清晰、更可信。</span>
          </div>
        )}
        <form className="search-form" onSubmit={handleSubmit}>
          <label className="sr-only" htmlFor="search-query">描述你正在寻找或判断的事情</label>
          <textarea
            id="search-query"
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            placeholder="描述你正在寻找或判断的事情…"
            rows={hasTask ? 2 : 3}
            maxLength={2000}
            required
          />
          <div className="search-form-footer">
            <span>自然语言提问</span>
            <button type="submit" disabled={submitting || !query.trim()}>
              {submitting ? <><span className="button-spinner" /> 正在开始</> : <>搜索 <span aria-hidden="true">↗</span></>}
            </button>
          </div>
        </form>
        {!hasTask && !requestError && (
          <div className="suggestions">
            <span>试着搜索</span>
            {EXAMPLE_QUERIES.map((example) => (
              <button key={example} type="button" onClick={() => setQuery(example)}>{example}</button>
            ))}
          </div>
        )}
      </section>

      {requestError && (
        <div className="request-error" role="alert">
          <span aria-hidden="true">!</span>
          <div><strong>暂时无法获取分析结果</strong><p>{requestError}</p></div>
          <button type="button" onClick={() => setRequestError("")} aria-label="关闭提示">×</button>
        </div>
      )}

      {task && (
        <section className="results-section" aria-live="polite" aria-busy={Boolean(active || loadingPage)}>
          <div className="results-heading">
            <div className="query-label">搜索内容</div>
            <h2>{task.query}</h2>
            <span className={"task-badge task-" + task.status}>{active && <span className="badge-spinner" />}{TASK_STATUS_LABEL[task.status]}</span>
          </div>

          <details className="execution-details" open={Boolean(active)}>
            <summary>
              <span>查看执行过程</span>
              <span>四阶段状态、真实调用记录与 Token 用量</span>
            </summary>
            <section className="pipeline" aria-label="信息处理阶段状态">
            <div className="pipeline-heading">
              <div><h3>信息决策流程</h3><p>根据服务返回的阶段状态与处理记录更新</p></div>
              {!task.stages.length && <span className="pipeline-unavailable">阶段数据未提供</span>}
            </div>
            <ol className="stage-list">
              {STAGE_INFO.map((info, index) => {
                const stage = stageByKey.get(info.key);
                const stageStatus = stage?.status;
                const unfinishedAfterTerminal = Boolean(stageStatus === "running" && !active);
                return (
                  <li className={"stage-item" + (stageStatus ? " stage-" + stageStatus : " stage-unknown") + (unfinishedAfterTerminal ? " stage-unfinished" : "")} key={info.key}>
                    <div className="stage-topline">
                      <span className="stage-index">{String(index + 1).padStart(2, "0")}</span>
                      <span className="stage-name">{info.title}</span>
                      <span className="stage-status">{unfinishedAfterTerminal ? "任务终态未收尾" : stageStatus ? STAGE_STATUS_LABEL[stageStatus] : "未提供"}</span>
                    </div>
                    <span className="stage-subtitle">{info.subtitle}</span>
                    <div className="stage-metrics">
                      <span>输入 {countLabel(stage?.input_count)} · 输出 {countLabel(stage?.output_count)}</span>
                      <span>{durationLabel(stage?.duration_ms)}</span>
                    </div>
                    {info.key === "verify" && stage && (stage.verified_count !== null || stage.unverified_count !== null || stage.conflict_count !== null) && (
                      <div className="stage-verification-counts">
                        已核实 {countLabel(stage.verified_count)} · 待核实 {countLabel(stage.unverified_count)} · 冲突 {countLabel(stage.conflict_count)}
                      </div>
                    )}
                    {stage?.error && <p className="stage-error">{stage.error}</p>}
                  </li>
                );
              })}
            </ol>
            </section>

            {(task.research || researchRounds.length > 0 || researchToolCalls.length > 0 || searchMetrics.length > 0 || active) && (
              <section className="research-observability" aria-label="自主研究轮次与搜索来源">
                <div className="research-heading">
                  <div><h3>自主研究记录</h3><p>仅展示后端实际发送的轮次、工具调用和搜索源统计</p></div>
                  {task.research && <span className={"research-outcome research-outcome-" + task.research.status}>{task.research.status === "completed" ? "研究完成" : task.research.status === "partial" ? "部分完成" : task.research.status === "stopped" ? "已停止" : "研究失败"}</span>}
                </div>

                {(returnedSourceCount !== null || task.research) && (
                  <div className="research-summary-metrics">
                    {returnedSourceCount !== null && <div><span>搜索提供方原始返回</span><strong>{returnedSourceCount} 条</strong></div>}
                    {task.research && <div><span>研究轮次</span><strong>{researchRounds.length}</strong></div>}
                    {task.research && <div><span>模型调用</span><strong>{task.research.model_calls} 次</strong></div>}
                    {task.research && <div><span>总耗时</span><strong>{durationLabel(task.research.elapsed_ms)}</strong></div>}
                  </div>
                )}

                {task.research?.stop_reason && <p className="research-stop-reason">停止原因：{task.research.stop_reason}</p>}

                {searchMetrics.length > 0 && (
                  <div className="provider-metrics">
                    <strong>搜索提供方</strong>
                    <ul>
                      {searchMetrics.map((metric) => (
                        <li key={metric.provider}>
                          <span>{metric.provider}</span>
                          <span>{metric.configured ? "已配置" : "未配置"}</span>
                          <span>请求 {metric.requests}</span>
                          <span>来源 {metric.returned_sources}</span>
                          <span>失败 {metric.failures}</span>
                          <span>{durationLabel(metric.latency_ms)}</span>
                          {metric.error_code && <code>{metric.error_code}</code>}
                        </li>
                      ))}
                    </ul>
                  </div>
                )}

                {researchRounds.length > 0 ? (
                  <ol className="research-round-list">
                    {researchRounds.map((round) => (
                      <li key={round.round_number} className={"research-round research-round-" + round.status}>
                        <div className="research-round-topline">
                          <strong>第 {round.round_number} 轮</strong>
                          <span>{RESEARCH_STATUS_LABEL[round.status]}</span>
                          {round.duration_ms !== null && <span>{durationLabel(round.duration_ms)}</span>}
                        </div>
                        {round.reason && <p>{round.reason}</p>}
                        <div className="research-round-counts">
                          <span>搜索请求 {round.search_requests}</span>
                          <span>来源返回 {round.sources_found}</span>
                          <span>新增来源 {round.new_sources}</span>
                          <span>候选 {round.candidates_before} → {round.candidates_after}</span>
                        </div>
                        {round.planned_queries.length > 0 && <details><summary>本轮查询 ({round.planned_queries.length})</summary><ul>{round.planned_queries.map((queryText, index) => <li key={index}>{queryText}</li>)}</ul></details>}
                        {round.evidence_gaps.length > 0 && <div className="research-gaps"><span>仍待补证</span>{round.evidence_gaps.map((gap, index) => <span key={index}>{gap}</span>)}</div>}
                      </li>
                    ))}
                  </ol>
                ) : <p className="research-empty">{active ? "尚未收到研究轮次事件。" : "本任务未提供研究轮次记录。"}</p>}

                {researchToolCalls.length > 0 && (
                  <div className="research-tool-calls">
                    <strong>工具调用记录</strong>
                    <ol>
                      {researchToolCalls.map((call, index) => (
                        <li key={call.round_number + "-" + call.name + "-" + index} className={"tool-call tool-call-" + call.status}>
                          <div><span>第 {call.round_number} 轮 · {TOOL_NAME_LABEL[call.name]}</span><span>{TOOL_STATUS_LABEL[call.status]}</span></div>
                          <p>{call.input_summary || "调用参数摘要未提供"}</p>
                          {call.output_summary && <p>{call.output_summary}</p>}
                          <small>
                            {call.provider && <span>提供方 {call.provider}</span>}
                            {call.source_count !== null && <span>来源 {call.source_count}</span>}
                            {call.duration_ms !== null && <span>{durationLabel(call.duration_ms)}</span>}
                            {call.model && <span>模型 {call.model}</span>}
                            {call.total_tokens !== null && <span>Token {call.total_tokens}</span>}
                            {call.error_code && <span>错误 {call.error_code}</span>}
                          </small>
                        </li>
                      ))}
                    </ol>
                  </div>
                )}
              </section>
            )}

            <section className="agent-runs" aria-label="大模型与智能 Agent 执行记录">
            <div className="agent-runs-heading">
              <div>
                <h3>Agent 执行记录</h3>
                <p>仅展示服务返回的实际调用状态和用量</p>
              </div>
              <span className="agent-usage-total">
                Token {totalTokens === null ? "未提供" : totalTokens.toLocaleString("zh-CN")}
              </span>
            </div>
            {agentRuns.length ? (
              <ol className="agent-run-list">
                {agentRuns.map((run, index) => {
                  const cost = formatCost(run.estimated_cost_usd);
                  return (
                    <li className={"agent-run agent-run-" + run.status} key={run.agent + "-" + run.action + "-" + index}>
                      <div className="agent-run-topline">
                        <strong>{AGENT_TITLE[run.agent]} · {ACTION_TITLE[run.action]}</strong>
                        <span className={"agent-run-status agent-run-status-" + run.status}>{AGENT_RUN_STATUS_LABEL[run.status]}</span>
                      </div>
                      <div className="agent-run-meta">
                        <span>模型：{run.model || (run.status === "skipped_unconfigured" ? "未调用" : "未提供")}</span>
                        <span>耗时：{durationLabel(run.duration_ms)}</span>
                        {run.latency_ms !== null && <span>模型请求 {durationLabel(run.latency_ms)}</span>}
                        <span>Token：{agentTokens(run) === null ? "未提供" : agentTokens(run)?.toLocaleString("zh-CN")}</span>
                        {typeof run.prompt_tokens === "number" && <span>输入 {run.prompt_tokens.toLocaleString("zh-CN")}</span>}
                        {typeof run.completion_tokens === "number" && <span>输出 {run.completion_tokens.toLocaleString("zh-CN")}</span>}
                        {cost && <span>估算费用 {cost}</span>}
                        {run.request_id && <span>请求 ID：{run.request_id}</span>}
                      </div>
                      {run.input_summary && <p className="agent-run-summary"><span>输入</span>{run.input_summary}</p>}
                      {run.output_summary && <p className="agent-run-summary"><span>输出</span>{run.output_summary}</p>}
                      {run.error_code && <p className="agent-run-error">错误代码：{run.error_code}</p>}
                    </li>
                  );
                })}
              </ol>
            ) : (
              <p className="agent-runs-empty">
                {active ? "尚未收到 Agent 执行记录。" : "本任务未提供模型调用记录。"}
              </p>
            )}
            </section>
          </details>

          {task.answer && (
            <section className="final-answer" aria-label="知择生成的结论">
              <span>知择结论</span>
              <p>{task.answer}</p>
              {task.answer_claims.length > 0 && (
                <div className="answer-claims">
                  <strong>可追溯依据</strong>
                  <ul>
                    {task.answer_claims.map((claim, claimIndex) => (
                      <li key={claim.result_id + "-" + claimIndex}>
                        <p>{claim.text}</p>
                        {claim.evidence_quotes.length > 0 && (
                          <ul className="answer-claim-quotes">
                            {claim.evidence_quotes.map((quote, quoteIndex) => <li key={quoteIndex}>“{quote}”</li>)}
                          </ul>
                        )}
                        {claim.sources.length > 0 ? (
                          <div className="answer-claim-sources">
                            {claim.sources.map((source, sourceIndex) => {
                              const href = safeSourceUrl(source.url);
                              return href ? <a key={source.url + sourceIndex} href={href} target="_blank" rel="noopener noreferrer">{source.title} ↗</a> : null;
                            })}
                          </div>
                        ) : <span className="answer-claim-unlinked">该依据对应的已核验来源链接未提供</span>}
                      </li>
                    ))}
                  </ul>
                </div>
              )}
            </section>
          )}

          {active && <div className="working-note"><span className="progress-pulse" />任务仍在处理中，阶段数据会随服务状态更新。</div>}

          {task.errors.length > 0 && (
            <div className="task-errors" role="status">
              {task.errors.map((message, index) => <p key={index}>{message}</p>)}
            </div>
          )}

          {task.status === "failed" && (
            <div className="empty-state">
              <h3>{task.error || "搜索任务未能完成"}</h3>
              <p>你可以调整问题后重新搜索。</p>
            </div>
          )}

          {(task.status === "completed" || task.status === "partial") && task.results.length === 0 && (
            <div className="empty-state">
              <h3>暂时没有可展示的结果</h3>
              <p>当前搜索没有返回可用来源。可以换一种问法再试一次。</p>
            </div>
          )}

          {showingEarlyResults && (
            <div className="partial-results-note" role="status">
              <span className="progress-pulse" />
              <div><strong>研究继续中</strong><p>以下是当前已采集的候选线索，尚未完成来源核验和最终匹配判断。</p></div>
            </div>
          )}

          {(task.status === "completed" || task.status === "partial" || showingEarlyResults) && task.results.length > 0 && (
            <>
              <div className="result-toolbar">
                <div>
                  <h3>搜索结果</h3>
                  <span>{typeof totalResults === "number" ? "共 " + totalResults + " 条" : task.results.length + " 条结果"} · 已载入：推荐 {pageDecisionCounts.recommended} · 待核实 {pageDecisionCounts.needs_verification} · 不匹配 {pageDecisionCounts.not_matched}</span>
                </div>
                <label className="sort-control">
                  <span>排序</span>
                  <select value={sortMode} onChange={(event) => setSortMode(event.target.value as SortMode)}>
                    <option value="comprehensive">综合排序</option>
                    <option value="match">匹配度优先</option>
                    <option value="evidence">证据优先</option>
                  </select>
                </label>
              </div>
              <div className="result-list">
                {visibleResults.map((result) => (
                  <ResultCardView
                    key={resultIdentity(result)}
                    result={result}
                    query={task.query}
                    active={showingEarlyResults}
                  />
                ))}
              </div>
              {(hasMoreLoaded || canLoadMorePage) && (
                <nav className="pagination" aria-label="搜索结果分页">
                  <span>显示 {Math.min(visibleCount, results.length)} 条{totalResults !== null ? " / 共 " + totalResults + " 条" : ""}</span>
                  <button
                    type="button"
                    onClick={() => hasMoreLoaded ? setVisibleCount((count) => count + 10) : void loadPage(currentPage + 1)}
                    disabled={loadingPage}
                  >
                    {loadingPage ? "正在加载" : "查看更多"}
                  </button>
                </nav>
              )}
            </>
          )}
        </section>
      )}

      <footer className="footer"><span>知择 · 信息万千，知而善择</span><span>判断之前，请回到来源核对重要信息。</span></footer>
    </main>
  );
}
