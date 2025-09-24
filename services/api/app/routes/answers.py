# services/api/app/routes/answers.py
import os, json, asyncio, time, hashlib, re
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, HTTPException, Query
from starlette.responses import StreamingResponse
from pydantic import BaseModel, Field

# Reuse retrieval pipeline + model
from .query import (
    _vector_search, _fts_search, _fuse, _dedup_chunks, _maybe_rerank,
    ContextChunk,
)

from ..db import engine
from ..utils.embed import ollama_stream  # async generator that streams model tokens

router = APIRouter()

# -----------------------
# Request / Response models
# -----------------------

class AnswerRequest(BaseModel):
    q: str = Field(..., description="User query")
    top_k: int = Field(8, ge=1, le=50)
    alpha: float = Field(0.5, ge=0.0, le=1.0)
    doc_id: Optional[str] = None
    use_reranker: bool = False
    model: Optional[str] = None
    system_prompt: Optional[str] = None

class AnswerResponse(BaseModel):
    query: str
    answer: str
    citations: List[Dict[str, Any]]
    results: List[Dict[str, Any]]   # <-- make serializable

# -----------------------
# Helpers
# -----------------------

_ws = re.compile(r"\s+")
_punct = re.compile(r"[^\w\s]", re.UNICODE)

def _normalize_text(t: str) -> str:
    t = t.lower()
    t = _punct.sub(" ", t)
    t = _ws.sub(" ", t).strip()
    return t

def _chunk_to_result(c: ContextChunk) -> Dict[str, Any]:
    """Map a ContextChunk to a plain dict the UI expects."""
    return {
        "doc_id": getattr(c, "doc_id", None),
        "page": getattr(c, "page", None),
        "chunk_idx": getattr(c, "chunk_idx", None),
        "text": getattr(c, "text", "") or "",
        "score": getattr(c, "score", None),
        "where": getattr(c, "where", {}) or {},
    }

def _collapse_and_dedupe_chunks(chunks: List[ContextChunk], keep: int = 5, per_page_topn: int = 1, sim_threshold: float = 0.88,) -> List[ContextChunk]:
    """
    1) Group by (doc_id, page) and avoid neighboring chunk_idx (same paragraph)
    2) Text-hash dedupe
    3) Cap to `keep`
    """
    if not chunks:
        return []

    # Group by doc_id + page; assume chunks already scored/reranked desc
    grouped: Dict[tuple, List[ContextChunk]] = {}
    for h in chunks:
        key = (getattr(h, "doc_id", None), getattr(h, "page", None))
        grouped.setdefault(key, []).append(h)

    collapsed: List[ContextChunk] = []
    for key, group in grouped.items():
        # within a page, drop neighbors within +/-1 chunk
        seen_idxs: set[int] = set()
        for h in group:
            idx = getattr(h, "chunk_idx", -10**9)
            if any(abs(idx - s) <= 1 for s in seen_idxs):
                continue
            seen_idxs.add(idx)
            collapsed.append(h)

    # Text-hash dedupe (normalized prefix)
    seen_hash: set[str] = set()
    uniq: List[ContextChunk] = []
    for h in collapsed:
        txt = _normalize_text((getattr(h, "text", "") or "")[:1200])
        hh = hashlib.md5(txt.encode("utf-8")).hexdigest()
        if hh in seen_hash:
            continue
        seen_hash.add(hh)
        uniq.append(h)

    return uniq[:keep]

def _guess_kind(filename: str | None, url: str | None) -> str:
    name = (filename or url or "" ).lower()
    if name.endswith(".pdf"):
        return "pdf"
    if name.endswith((".htm", ".html")):
        return "html"
    return "text"

def _fallback_citation_id(doc_id: str, page: Optional[int], chunk_idx: Optional[int]) -> str:
    base = f"{doc_id}-{page}-{chunk_idx}"
    return "c_" + hashlib.sha1(base.encode("utf-8")).hexdigest()[:16]

def _build_citations(chunks: List[ContextChunk], limit: int) -> List[Dict[str, Any]]:
    """
    Prefer chunk.citation (from /query fusion). Fallback to filename/url heuristics.
    """
    out: List[Dict[str, Any]] = []
    seen: set[str] = set()

    for c in chunks[:limit]:
        # Prefer existing citation object produced by /query
        cit = getattr(c, "citation", None)

        if cit and getattr(cit, "id", None):
            cid = str(cit.id)
            if cid in seen:
                continue
            seen.add(cid)
            out.append({
                "id": cid,
                "title": getattr(cit, "title", None) or getattr(c, "where", {}).get("filename") or f"doc {c.doc_id}",
                "url": getattr(cit, "url", "") or getattr(c, "where", {}).get("url") or "",
                "ref_id": str(c.doc_id),
                "kind": getattr(cit, "kind", None) or _guess_kind(getattr(c, "where", {}).get("filename"), getattr(c, "where", {}).get("url")),
                "locator": getattr(cit, "locator", None) or ({"type": "pdf", "page": c.page} if c.page else {"type": "text"}),
                "snippet": (c.text or "")[:240],
                "preview_available": True,
            })
            continue

        # Fallback path (no chunk.citation)
        where = getattr(c, "where", {}) or {}
        filename = where.get("filename")
        url = where.get("url")
        kind = _guess_kind(filename, url)
        cid = _fallback_citation_id(str(c.doc_id), c.page, c.chunk_idx)
        if cid in seen:
            continue
        seen.add(cid)

        title_parts: List[str] = []
        if filename: title_parts.append(filename)
        if c.page is not None: title_parts.append(f"p.{c.page}")
        title = " • ".join(title_parts) if title_parts else f"doc {c.doc_id}"

        locator = {"type": "pdf", "page": c.page} if (kind == "pdf" and c.page) else {"type": "text"}

        out.append({
            "id": cid,
            "title": title,
            "url": url or "",
            "ref_id": str(c.doc_id),
            "kind": kind,
            "locator": locator,
            "snippet": (c.text or "")[:240],
            "preview_available": bool(url),
        })

    return out

def _persist_citations(citations: List[Dict[str, Any]]) -> None:
    """
    Upsert citations into 'sources' so /preview can resolve them.
    Safe no-op if the table doesn't exist.
    """
    if not citations:
        return

    sql_exists = "SELECT to_regclass('public.sources') IS NOT NULL AS present"
    upsert = """
      INSERT INTO public.sources (id, url, kind, locator, title)
      VALUES (%s,%s,%s,%s,%s)
      ON CONFLICT (id) DO UPDATE
        SET url=EXCLUDED.url,
            kind=EXCLUDED.kind,
            locator=EXCLUDED.locator,
            title=EXCLUDED.title
    """
    with engine.begin() as conn:
        try:
            present = conn.execute(sql_exists).scalar()
        except Exception:
            present = False
        if not present:
            return
        for c in citations:
            conn.execute(upsert, (
                c["id"], c.get("url"), c.get("kind"),
                json.dumps(c.get("locator") or {}),
                c.get("title"),
            ))

async def _retrieve(q: str, top_k: int, alpha: float,
                    doc_id: Optional[str], use_reranker: bool) -> List[ContextChunk]:
    vec_hits = _vector_search(q, limit=top_k, doc_id=doc_id)
    fts_hits = _fts_search(q, limit=max(1, top_k) * 4, doc_id=doc_id)
    fused = _fuse(vec_hits, fts_hits, top_k=top_k, alpha=alpha)
    fused = _dedup_chunks(fused)
    reranked = _maybe_rerank(q, fused, top_k=top_k, enabled=use_reranker)
    return reranked

# -----------------------
# POST /answers (non-streaming)
# -----------------------

@router.post("/", response_model=AnswerResponse)
async def answer(req: AnswerRequest):
    q = (req.q or "").strip()
    if not q:
        raise HTTPException(status_code=400, detail="q is required")

    # 1) retrieve context
    chunks = await _retrieve(q, req.top_k, req.alpha, req.doc_id, req.use_reranker)
    tidy_chunks = _collapse_and_dedupe_chunks(chunks, keep=min(5, req.top_k), per_page_topn=1, sim_threshold=0.88,)

    # 2) build + persist citations for preview
    citations = _build_citations(tidy_chunks, limit=len(tidy_chunks))
    _persist_citations(citations)

    # 3) run Ollama and collect full answer text
    parts: List[str] = []
    try:
        async for piece in ollama_stream(
            q,
            context_chunks=[c.text for c in chunks[:req.top_k]],  # keep full context for answer
            system_prompt=req.system_prompt,
            model=req.model,
        ):
            parts.append(piece)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Ollama error: {e}")

    tidy_results = [_chunk_to_result(c) for c in tidy_chunks]

    return AnswerResponse(
        query=q,
        answer="".join(parts).strip(),
        citations=citations,
        results=tidy_results,
    )

# -----------------------
# GET /answers/stream (SSE)
# -----------------------

@router.get("/stream")
async def stream_answer(
    q: str = Query(..., description="User query"),
    top_k: int = Query(8, ge=1, le=50),
    alpha: float = Query(0.5, ge=0.0, le=1.0),
    doc_id: Optional[str] = Query(None),
    use_reranker: bool = Query(False),
    model: Optional[str] = Query(None, description="Override OLLAMA_MODEL"),
    system_prompt: Optional[str] = Query(None, description="Optional system prompt"),
):
    # 1) retrieve
    chunks = await _retrieve(q, top_k, alpha, doc_id, use_reranker)
    tidy_chunks = _collapse_and_dedupe_chunks(chunks, keep=min(5, top_k))
    citations = _build_citations(tidy_chunks, limit=len(tidy_chunks))
    _persist_citations(citations)

    async def gen():
        # emit citations up-front (and include results for compatibility)
        results_json = [_chunk_to_result(c) for c in tidy_chunks]
        yield "data: " + json.dumps(
            {"event": "citations", "citations": citations, "results": results_json}
        ) + "\n\n"

        buf: List[str] = []
        last_flush = time.monotonic()
        last_heartbeat = last_flush
        FLUSH_EVERY = 0.03
        HEARTBEAT_EVERY = 20.0

        try:
            async for piece in ollama_stream(
                q,
                context_chunks=[c.text for c in chunks[:top_k]],  # use full retrieved context
                system_prompt=system_prompt,
                model=model,
            ):
                buf.append(piece)
                now = time.monotonic()
                if (now - last_heartbeat) >= HEARTBEAT_EVERY:
                    yield "data: " + json.dumps({"event": "ping"}) + "\n\n"
                    last_heartbeat = now
                if "\n" in piece or (now - last_flush) >= FLUSH_EVERY:
                    if buf:
                        out = "".join(buf); buf.clear(); last_flush = now
                        yield "data: " + json.dumps({"delta": out}) + "\n\n"
                await asyncio.sleep(0)
        except Exception as e:
            yield "data: " + json.dumps({"event": "error", "message": f"{type(e).__name__}: {e}"}) + "\n\n"

        if buf:
            yield "data: " + json.dumps({"delta": "".join(buf)}) + "\n\n"
        yield "data: " + json.dumps({"event": "done"}) + "\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream")
