"use client";

import { FormEvent, useEffect, useRef, useState } from "react";

type TaskStatus = "queued" | "running" | "completed" | "failed";
type Verdict = "supports" | "contradicts" | "context" | "unverified";

type SearchResult = {
  id: string;
  title: string;
  url: string;
  source_name: string;
  snippet: string;
  evidence: string;
  match_reason: string;
  verdict: Verdict;
  confidence: number;
  published_at?: string | null;
};

type SearchTask = {
  task_id: string;
  query: string;
  status: TaskStatus;
  results: SearchResult[];
  error?: string;
};

type AcceptedTask = { task_id: string; status: TaskStatus };

const API_BASE_URL = (process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://127.0.0.1:8000").replace(/\/$/, "");
const EXAMPLE_QUERIES = ["AI 如何改变知识工作？", "新能源汽车电池寿命受什么影响？", "远程办公对团队效率有何影响？"];

const statusCopy: Record<TaskStatus, { title: string; detail: string }> = {
  queued: { title: "已加入分析队列", detail: "正在等待开始处理。" },
  running: { title: "正在查找与核对", detail: "我们正在整理来源，并逐条检查相关证据。" },
  completed: { title: "分析完成", detail: "以下信息来自检索到的来源，请结合原文判断。" },
  failed: { title: "分析未能完成", detail: "任务暂时无法完成，请稍后重试。" },
};

const verdictCopy: Record<Verdict, string> = {
  supports: "支持判断",
  contradicts: "与判断相悖",
  context: "提供背景",
  unverified: "尚未核实",
};

function isSearchTask(value: unknown): value is SearchTask {
  if (!value || typeof value !== "object") return false;
  const task = value as Partial<SearchTask>;
  return typeof task.task_id === "string"
    && typeof task.query === "string"
    && ["queued", "running", "completed", "failed"].includes(task.status ?? "")
    && Array.isArray(task.results);
}

function isAcceptedTask(value: unknown): value is AcceptedTask {
  if (!value || typeof value !== "object") return false;
  const task = value as Partial<AcceptedTask>;
  return typeof task.task_id === "string"
    && ["queued", "running", "completed", "failed"].includes(task.status ?? "");
}

function formatDate(value?: string | null) {
  if (!value) return null;
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return null;
  return new Intl.DateTimeFormat("zh-CN", { year: "numeric", month: "short", day: "numeric" }).format(date);
}

function safeSourceUrl(value: string) {
  try {
    const parsed = new URL(value);
    return parsed.protocol === "https:" || parsed.protocol === "http:" ? parsed.href : null;
  } catch {
    return null;
  }
}

export default function Home() {
  const [query, setQuery] = useState("");
  const [task, setTask] = useState<SearchTask | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const [requestError, setRequestError] = useState("");
  const requestSequence = useRef(0);

  useEffect(() => {
    if (!task || (task.status !== "queued" && task.status !== "running")) return;

    let cancelled = false;
    const taskId = task.task_id;
    const sequence = requestSequence.current;

    async function refreshTask() {
      try {
        const response = await fetch(`${API_BASE_URL}/api/v1/search/${encodeURIComponent(taskId)}`, {
          cache: "no-store",
        });
        if (!response.ok) throw new Error(`任务状态请求失败（${response.status}）`);
        const payload: unknown = await response.json();
        if (!isSearchTask(payload)) throw new Error("服务返回了无法识别的任务状态。");
        if (!cancelled && sequence === requestSequence.current) {
          setTask(payload);
          setRequestError("");
        }
      } catch (error) {
        if (!cancelled && sequence === requestSequence.current) {
          setRequestError(error instanceof Error ? error.message : "暂时无法获取任务状态。");
        }
      }
    }

    const interval = window.setInterval(() => void refreshTask(), 1800);
    return () => {
      cancelled = true;
      window.clearInterval(interval);
    };
  }, [task?.task_id, task?.status]);

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const normalizedQuery = query.trim();
    if (!normalizedQuery || submitting) return;

    requestSequence.current += 1;
    setSubmitting(true);
    setRequestError("");
    setTask(null);

    try {
      const response = await fetch(`${API_BASE_URL}/api/v1/search`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: normalizedQuery, limit: 10 }),
      });
      if (!response.ok) {
        const message = response.status === 422
          ? "问题内容不符合要求，请调整后再试。"
          : `无法创建搜索任务（${response.status}）。请确认服务已启动后重试。`;
        throw new Error(message);
      }
      const acceptedPayload: unknown = await response.json();
      if (!isAcceptedTask(acceptedPayload)) throw new Error("服务返回了无法识别的任务信息。");
      const detailResponse = await fetch(`${API_BASE_URL}/api/v1/search/${encodeURIComponent(acceptedPayload.task_id)}`, {
        cache: "no-store",
      });
      if (!detailResponse.ok) throw new Error(`任务状态请求失败（${detailResponse.status}）`);
      const detailPayload: unknown = await detailResponse.json();
      if (!isSearchTask(detailPayload)) throw new Error("服务返回了无法识别的任务状态。");
      setTask(detailPayload);
    } catch (error) {
      setRequestError(error instanceof Error ? error.message : "暂时无法连接知择服务。");
    } finally {
      setSubmitting(false);
    }
  }

  const status = task ? statusCopy[task.status] : null;
  const active = task?.status === "queued" || task?.status === "running";
  const showResults = task?.status === "completed" && task.results.length > 0;

  return (
    <main className="site-shell">
      <header className="topbar">
        <a className="brand" href="/" aria-label="知择首页">
          <span className="brand-mark" aria-hidden="true"><span /></span>
          <span>知择</span>
        </a>
        <div className="topbar-note"><span className="status-dot" /> 基于来源的 AI 信息决策</div>
      </header>

      <section className={`hero ${task ? "hero-compact" : ""}`}>
        <div className="eyebrow"><span className="eyebrow-line" /> THINK CLEARER, CHOOSE BETTER</div>
        <h1>先看清信息，<br className="mobile-break" />再做重要决定。</h1>
        <p className="intro">提出你正在判断的问题。知择会查找相关信息、呈现证据来源，<br className="desktop-break" />并说明每条内容与你的问题有什么关系。</p>

        <form className="search-box" onSubmit={handleSubmit}>
          <label htmlFor="search-query" className="sr-only">输入你想了解的问题</label>
          <textarea
            id="search-query"
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            placeholder="输入你想了解的问题…"
            rows={2}
            maxLength={2000}
            required
          />
          <div className="search-actions">
            <span className="input-hint">尽量描述清楚你的问题</span>
            <button type="submit" disabled={submitting || !query.trim()}>
              {submitting ? <><span className="button-spinner" /> 正在提交</> : <>开始查找 <span aria-hidden="true">↗</span></>}
            </button>
          </div>
        </form>

        {!task && !requestError && (
          <div className="examples">
            <span>试试这样问</span>
            {EXAMPLE_QUERIES.map((example) => (
              <button key={example} type="button" onClick={() => setQuery(example)}>{example}</button>
            ))}
          </div>
        )}
      </section>

      {requestError && (
        <div className="request-error" role="alert">
          <span className="error-icon" aria-hidden="true">!</span>
          <div><strong>暂时无法获取分析结果</strong><p>{requestError}</p></div>
          <button type="button" onClick={() => setRequestError("")} aria-label="关闭提示">×</button>
        </div>
      )}

      {task && status && (
        <section className="results-section" aria-live="polite" aria-busy={Boolean(active)}>
          <div className="section-heading">
            <div>
              <div className="section-kicker">你的问题</div>
              <h2>{task.query}</h2>
            </div>
            <span className={`task-badge task-${task.status}`}>
              {active && <span className="badge-spinner" />}
              {status.title}
            </span>
          </div>

          {active && (
            <div className="progress-panel">
              <div className="progress-mark"><span className="progress-pulse" /></div>
              <div><strong>{status.title}</strong><p>{status.detail}</p></div>
            </div>
          )}

          {task.status === "failed" && (
            <div className="empty-panel failed-panel">
              <span className="empty-mark">↻</span>
              <h3>{task.error || status.title}</h3>
              <p>{status.detail}</p>
              <button className="retry-button" type="button" onClick={() => { setTask(null); setRequestError(""); }}>重新提问</button>
            </div>
          )}

          {task.status === "completed" && task.results.length === 0 && (
            <div className="empty-panel">
              <span className="empty-mark">⌕</span>
              <h3>暂时没有找到相关来源</h3>
              <p>当前搜索没有返回可展示的结果。可以换一种问法再试一次。</p>
            </div>
          )}

          {showResults && (
            <>
              <div className="results-intro">
                <span className="result-count">找到 {task.results.length} 条来源</span>
                <span className="results-caveat">信息摘要供参考，判断前请查看原始来源。</span>
              </div>
              <div className="result-list">
                {task.results.map((result, index) => {
                  const publishedAt = formatDate(result.published_at);
                  const sourceUrl = safeSourceUrl(result.url);
                  const confidence = Number.isFinite(result.confidence) ? Math.round(Math.max(0, Math.min(1, result.confidence)) * 100) : 0;
                  return (
                    <article className="result-card" key={result.id}>
                      <div className="result-index">{String(index + 1).padStart(2, "0")}</div>
                      <div className="result-content">
                        <div className="result-meta">
                          <span className="source-name">{result.source_name || "未知来源"}</span>
                          {publishedAt && <><span className="meta-separator">·</span><time>{publishedAt}</time></>}
                          <span className={`verdict verdict-${result.verdict}`}>{verdictCopy[result.verdict] ?? verdictCopy.unverified}</span>
                        </div>
                        <h3>
                          {sourceUrl ? <a href={sourceUrl} target="_blank" rel="noreferrer">{result.title} <span aria-hidden="true">↗</span></a> : result.title}
                        </h3>
                        {result.snippet && <p className="result-snippet">{result.snippet}</p>}
                        {result.evidence && (
                          <div className="evidence-block">
                            <span className="detail-label">证据摘录</span>
                            <p>“{result.evidence}”</p>
                          </div>
                        )}
                        <div className="match-row">
                          <span className="detail-label">与问题的关联</span>
                          <p>{result.match_reason || "服务暂未提供关联说明。"}</p>
                        </div>
                        <div className="confidence-row">
                          <span>置信度参考</span>
                          <div className="confidence-track" aria-label={`置信度 ${confidence}%`}>
                            <span style={{ width: `${confidence}%` }} />
                          </div>
                          <span className="confidence-value">{confidence}%</span>
                        </div>
                      </div>
                    </article>
                  );
                })}
              </div>
            </>
          )}
        </section>
      )}

      <footer className="footer">
        <span>知择 <span className="footer-dot">·</span> 让判断有据可依</span>
        <span>请结合来源原文与具体情境独立判断</span>
      </footer>
    </main>
  );
}
