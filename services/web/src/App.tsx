import React from "react";
import Upload from "./components/Upload";
import Chat from "./components/Chat";

export default function App() {
  const apiBase = "http://localhost:8000";

  return (
    <div className="min-h-screen max-w-screen-2xl mx-auto px-4 py-4">
    {/* Header */}
    <header className="mb-4">
    <h1 className="text-xl font-semibold">RAG Micro</h1>
    <div className="text-xs text-gray-500">API Base <span className="font-mono">{apiBase}</span></div>
    </header>

    {/* Page grid: fixed 320px left, flexible right */}
    <div className="grid gap-4 lg:grid-cols-[320px_minmax(0,1fr)]">
    {/* LEFT: uploader */}
    <aside className="space-y-3">
    <Upload apiBase={apiBase} />
    </aside>

    {/* RIGHT: chat (let it shrink/grow) */}
    <section className="min-w-0">
    <Chat apiBase={apiBase} />
    </section>
    </div>
    </div>
  );
}
