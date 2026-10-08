"use client";

import { FormEvent, useState } from "react";

type VerificationResult = {
  verdict: "supported" | "mixed" | "refuted" | "insufficient_evidence";
  summary: string;
  confidence: number;
};

const verdictLabels: Record<VerificationResult["verdict"], string> = {
  supported: "证据支持",
  mixed: "证据不一",
  refuted: "证据不支持",
  insufficient_evidence: "证据不足",
};

export default function Home() {
  const [question, setQuestion] = useState("");
  const [result, setResult] = useState<VerificationResult | null>(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setError("");
    setResult(null);
    setLoading(true);

    try {
      const apiBase = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://127.0.0.1:8000";
      const response = await fetch(`${apiBase}/api/v1/verify`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question, sources: [] }),
      });

      if (!response.ok) {
        throw new Error(`请求失败（${response.status}），请确认后端服务已启动。`);
      }

      setResult((await response.json()) as VerificationResult);
    } catch (requestError) {
      setError(requestError instanceof Error ? requestError.message : "暂时无法连接服务。");
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="min-h-screen px-5 py-8 sm:px-10 sm:py-12">
      <div className="mx-auto flex max-w-5xl flex-col">
        <header className="flex items-center justify-between">
          <a className="flex items-center gap-3 font-semibold tracking-wide" href="/">
            <span className="grid size-10 place-items-center rounded-2xl bg-[#16745c] text-lg text-white">知</span>
            <span className="text-lg">知择</span>
          </a>
          <span className="rounded-full border border-[var(--line)] bg-white px-3 py-1 text-xs text-[var(--muted)]">V0.1 本地开发版</span>
        </header>

        <section className="mx-auto mt-20 w-full max-w-3xl sm:mt-28">
          <p className="text-sm font-medium tracking-[0.16em] text-[var(--green)]">THINK CLEARER, CHOOSE BETTER</p>
          <h1 className="mt-5 text-4xl font-semibold leading-tight tracking-tight sm:text-6xl">让重要判断，<br className="sm:hidden" />有据可依。</h1>
          <p className="mt-5 max-w-2xl text-base leading-7 text-[var(--muted)] sm:text-lg">
            把你正在判断的问题交给知择。我们会围绕证据整理信息，帮你看清事实、分歧与不确定性。
          </p>

          <form onSubmit={handleSubmit} className="mt-10 rounded-3xl border border-[var(--line)] bg-white p-3 shadow-[0_18px_55px_-40px_rgba(21,44,39,0.4)] sm:p-4">
            <label className="sr-only" htmlFor="question">输入问题</label>
            <textarea
              id="question"
              className="min-h-32 w-full resize-y rounded-2xl border-0 bg-transparent p-4 text-base leading-7 outline-none placeholder:text-[#9aa59f] focus:ring-2 focus:ring-[#b5d7ca]"
              placeholder="例如：这项政策会怎样影响我的选择？"
              value={question}
              onChange={(event) => setQuestion(event.target.value)}
              required
            />
            <div className="flex flex-col gap-3 border-t border-[var(--line)] px-2 pt-3 sm:flex-row sm:items-center sm:justify-between">
              <p className="px-2 text-xs leading-5 text-[var(--muted)]">当前骨架尚未接入搜索来源，结果会明确标记证据不足。</p>
              <button
                className="rounded-xl bg-[var(--green)] px-5 py-3 text-sm font-medium text-white transition hover:bg-[#105d4a] disabled:cursor-wait disabled:opacity-60"
                type="submit"
                disabled={loading || !question.trim()}
              >
                {loading ? "正在整理…" : "开始分析"}
              </button>
            </div>
          </form>

          {error && <p role="alert" className="mt-4 rounded-xl bg-red-50 px-4 py-3 text-sm text-red-700">{error}</p>}
          {result && (
            <section aria-live="polite" className="mt-5 rounded-2xl border border-[var(--line)] bg-white p-5">
              <div className="flex items-center justify-between gap-4">
                <h2 className="font-semibold">{verdictLabels[result.verdict]}</h2>
                <span className="text-sm text-[var(--muted)]">置信度 {Math.round(result.confidence * 100)}%</span>
              </div>
              <p className="mt-3 text-sm leading-6 text-[var(--muted)]">{result.summary}</p>
            </section>
          )}
        </section>

        <footer className="mt-20 border-t border-[var(--line)] pt-5 text-xs text-[var(--muted)] sm:mt-28">
          知择 V0.1 · 结论应结合来源与上下文判断
        </footer>
      </div>
    </main>
  );
}

