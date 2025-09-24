import React, { useState } from "react";
import ResultCard from "./ResultCard";

export default function Chat({ apiBase }: { apiBase: string }) {
  const [q, setQ] = useState("");
  const [answer, setAnswer] = useState("");
  const [pendingResults, setPendingResults] = useState<any[]>([]);
  const [results, setResults] = useState<any[]>([]);
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [done, setDone] = useState(false);

  function uniqueBy<T>(arr: T[], key: (x: T) => string) {
    const seen = new Set<string>();
    return arr.filter((x) => {
      const k = key(x);
      if (seen.has(k)) return false;
      seen.add(k);
      return true;
    });
  }

  async function runQuery(e?: React.FormEvent) {
    e?.preventDefault();
    setLoading(true);
    setErr(null);
    setAnswer("");
    setResults([]);
    setPendingResults([]);
    setDone(false);

    const base = apiBase.replace(/\/+$/, "");
    const url = new URL(`${base}/answers/stream`);
    url.searchParams.set("q", q);
    url.searchParams.set("top_k", "12");          // retrieval breadth
    url.searchParams.set("alpha", "0.6");
    url.searchParams.set("use_reranker", "true"); // better ordering

    const es = new EventSource(url.toString());

    es.onmessage = (evt) => {
      try {
        const msg = JSON.parse(evt.data);

        if (typeof msg.delta === "string" && msg.delta) {
          setAnswer((prev) => prev + msg.delta);
        }

        if (Array.isArray(msg.results)) {
          setPendingResults(msg.results);
        }
        if (Array.isArray(msg.citations)) {
          setPendingResults(msg.citations);
        }

        if (msg.event === "done") {
          es.close();
          setLoading(false);
          setDone(true);

          // fetch citations after streaming
          (async () => {
            try {
              const base = apiBase.replace(/\/+$/, "");
              const res = await fetch(`${base}/query/`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ q, top_k: 12, alpha: 0.6, use_reranker: true })
              });
              const j = await res.json();
              // dedup + cap for tidy UI
              const seen = new Set<string>();
              const deduped = (j.results || []).filter((h: any) => {
                const k = `${h.doc_id}:${h.page}:${h.chunk_idx ?? "x"}`;
                if (seen.has(k)) return false; seen.add(k); return true;
              });
                setResults(deduped.slice(0, 2));
            } catch (_) { /* ignore */ }
          })();
        }
      } catch {
        // ignore malformed frames
      }
    };

    es.onerror = () => {
      es.close();
      setLoading(false);
      setErr("Stream error. Please try again.");
    };
  }

  return (
    // responsive 12-col grid; middle grows, right stays reasonable
    <div className="grid gap-4 md:grid-cols-12">
    {/* MIDDLE: prompt + streamed answer (dominant) */}
    <main className="md:col-span-8 lg:col-span-9 space-y-3">
    <form onSubmit={runQuery} className="flex gap-2">
    <input
    className="border px-2 py-1 rounded w-full"
    placeholder="Ask your docs…"
    value={q}
    onChange={(e) => setQ(e.target.value)}
    />
    <button
    className="px-3 py-1 rounded bg-black text-white disabled:opacity-50"
    disabled={loading || !q}
    >
    {loading ? "Searching…" : "Ask"}
    </button>
    </form>

    {err && <div className="text-sm text-red-500">{err}</div>}

    {answer && (
      <div className="whitespace-pre-wrap border rounded p-3 text-sm bg-gray-50">
      {answer}
      </div>
    )}

    {/* Mobile citations (stacked under answer) */}
    <section className="md:hidden space-y-3">
    {done && results.length > 0 && (
      <>
      <h3 className="text-sm font-semibold text-gray-700">Citations</h3>
      <div className="space-y-3">
      {results.map((hit: any) => (
        <ResultCard
        key={`m:${hit.doc_id}:${hit.page}:${hit.chunk_idx ?? "x"}`}
        apiBase={apiBase}
        hit={hit}
        query={q}
        />
      ))}
      </div>
      </>
    )}
    </section>
    </main>

    {/* RIGHT: citations sidebar (sticks, shrinks/grows with window) */}
    <aside className="hidden md:block md:col-span-4 lg:col-span-3">
    <div className="sticky top-4 space-y-3">
    <h3 className="text-sm font-semibold text-gray-700">Citations</h3>

    {!done && loading && (
      <div className="text-xs text-gray-500">Collecting sources…</div>
    )}

    {done && results.length === 0 && (
      <div className="text-xs text-gray-500">No citations.</div>
    )}

    {done && results.length > 0 && (
      <div className="space-y-3">
      {results.map((hit: any) => (
        <ResultCard
        key={`${hit.doc_id}:${hit.page}:${hit.chunk_idx ?? "x"}`}
        apiBase={apiBase}
        hit={hit}
        query={q}
        />
      ))}
      </div>
    )}
    </div>
    </aside>
    </div>
  );
}
