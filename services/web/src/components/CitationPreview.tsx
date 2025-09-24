import React, { useState } from "react";
import ResultCard from "./ResultCard";

type Hit = {
    doc_id: string;
    page: number;
    chunk_idx?: number;
    text: string;
    // …whatever else you already use in ResultCard
};

export default function Chat({ apiBase }: { apiBase: string }) {
    const [q, setQ] = useState("");
    const [answer, setAnswer] = useState("");       // streamed model text
    const [results, setResults] = useState<Hit[]>([]);
    const [loading, setLoading] = useState(false);
    const [err, setErr] = useState<string | null>(null);

    // ---- STREAMING IMPLEMENTATION ----
    async function runQuery(e?: React.FormEvent) {
        e?.preventDefault();
        setLoading(true);
        setErr(null);
        setAnswer("");
        setResults([]);

        // Build /answers/stream URL
        const base = apiBase.replace(/\/+$/, "");
        const url = new URL(`${base}/answers/stream`);
        url.searchParams.set("q", q);
        url.searchParams.set("top_k", "8");
        url.searchParams.set("alpha", "0.6");
        url.searchParams.set("use_reranker", "false");

        // Start SSE
        const es = new EventSource(url.toString());

        es.onmessage = (evt) => {
            try {
                const msg = JSON.parse(evt.data);

                // Server can send different frames; handle flexibly.
                // Common ones we'll support:
                // { event: "results", results: Hit[] }
                // { event: "citations", results: Hit[] }  // sometimes named citations
                // { delta: "partial text" }
                // { event: "done" }
                if (Array.isArray(msg.results)) {
                    // results / citations frame
                    setResults(msg.results);
                }
                if (typeof msg.delta === "string" && msg.delta.length) {
                    setAnswer((prev) => prev + msg.delta);
                }
                if (msg.event === "done") {
                    es.close();
                    setLoading(false);
                }
            } catch (e) {
                // If a frame isn't JSON (shouldn't happen), ignore it
            }
        };

        es.onerror = () => {
            es.close();
            setLoading(false);
            setErr("Stream error. Please try again.");
        };
    }
}
