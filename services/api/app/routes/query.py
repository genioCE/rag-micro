# app/routes/query.py
import os
import re
import hashlib
import inspect
from typing import Dict, List, Optional, Tuple, Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import text

from ..db import SessionLocal, engine
from ..utils.embed import get_embedder, search_qdrant  # we'll adapt to its signature dynamically

router = APIRouter()

# ===========================
# Request / Response Schemas
# ===========================

class QueryRequest(BaseModel):
    q: str = Field(..., description="User query")
    top_k: int = Field(8, ge=1, le=50)
    alpha: float = Field(0.5, ge=0.0, le=1.0, description="Fusion weight: 1.0=vector only, 0.0=FTS only")
    doc_id: Optional[str] = Field(None, description="Optional hard filter to a single document_id")
    min_score: Optional[float] = Field(None, description="Optional min fused score filter (0..1 after normalization)")
    use_reranker: bool = Field(False, description="Use cross-encoder reranker on top candidates")
    debug: bool = Field(False, description="If true, include raw counts and debug info")

class Citation(BaseModel):
    id: Optional[str] = None
    title: Optional[str] = None
    url: Optional[str] = None
    kind: Optional[str] = None      # 'pdf' | 'html' | 'text'
    locator: Optional[dict] = None  # e.g., {"page": 3, "bbox": [...]}


class ContextChunk(BaseModel):
    doc_id: str
    page: Optional[int]
    chunk_idx: Optional[int]
    text: str
    scores: Dict[str, float]  # vector / fts / fused / (optional) rerank
    where: Dict[str, str]     # filename, composite_id (if available)
    citation: Optional[Citation] = None

class QueryResponse(BaseModel):
    query: str
    top_k: int
    alpha: float
    results: List[ContextChunk]
    debug: Optional[Dict] = None

# ============
# Config
# ============

QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "docs")
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-base")
RERANKER_MAX_CANDIDATES = int(os.getenv("RERANKER_MAX_CANDIDATES", "50"))

DEDUP_MIN_CHARS = int(os.getenv("DEDUP_MIN_CHARS", "40"))
DEDUP_NORM_WHITESPACE = True
_WS_RE = re.compile(r"\s+")

# ============
# Helpers
# ============

def _normalize_scores(vals: Dict[Tuple[str, Optional[int], Optional[int]], float]) -> Dict[Tuple[str, Optional[int], Optional[int]], float]:
    if not vals:
        return {}
    mn = min(vals.values())
    mx = max(vals.values())
    if mx <= mn:
        return {k: 1.0 for k in vals}
    return {k: (v - mn) / (mx - mn) for k, v in vals.items()}

def _key(doc_id: str, page: Optional[int], chunk_idx: Optional[int]) -> Tuple[str, Optional[int], Optional[int]]:
    return (str(doc_id), page, chunk_idx)

def _get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# ========================
# Postgres FTS search
# ========================

def _fts_search(q: str, limit: int, doc_id: Optional[str]) -> List[dict]:
    params: Dict[str, Any] = {"q": q, "limit": limit}
    if doc_id:
        params["doc_id"] = str(doc_id)

    # Detect if chunks_fts has source_id column
    has_source_id = False
    with engine.begin() as conn:
        try:
            has_source_id = bool(conn.execute(text("""
                SELECT EXISTS(
                    SELECT 1
                    FROM information_schema.columns
                    WHERE table_schema='public'
                      AND table_name='chunks_fts'
                      AND column_name='source_id'
                )
            """)).scalar())
        except Exception:
            has_source_id = False

    if has_source_id:
        sql = text(f"""
            WITH q AS (SELECT websearch_to_tsquery('english', :q) AS query)
            SELECT
              c.id,
              c.document_id,
              c.page,
              c.chunk_idx,
              c.text,
              jsonb_extract_path_text(c.meta, 'filename')     AS filename,
              jsonb_extract_path_text(c.meta, 'composite_id') AS composite_id,
              c.source_id                                      AS source_id,
              s.title                                          AS source_title,
              s.url                                            AS source_url,
              s.kind                                           AS source_kind,
              s.locator                                        AS source_locator,
              ts_rank(
                to_tsvector('english', unaccent(coalesce(c.text,''))),
                (SELECT query FROM q)
              ) AS rank
            FROM public.chunks_fts c
            LEFT JOIN public.sources s ON s.id = c.source_id
            WHERE to_tsvector('english', unaccent(coalesce(c.text,''))) @@ (SELECT query FROM q)
            {"AND c.document_id = :doc_id" if doc_id else ""}
            ORDER BY rank DESC
            LIMIT :limit
        """)
    else:
        sql = text(f"""
            WITH q AS (SELECT websearch_to_tsquery('english', :q) AS query)
            SELECT
              id,
              document_id,
              page,
              chunk_idx,
              text,
              jsonb_extract_path_text(meta, 'filename')     AS filename,
              jsonb_extract_path_text(meta, 'composite_id') AS composite_id,
              NULL::text  AS source_id,
              NULL::text  AS source_title,
              NULL::text  AS source_url,
              NULL::text  AS source_kind,
              NULL::jsonb AS source_locator,
              ts_rank(
                to_tsvector('english', unaccent(coalesce(text,''))),
                (SELECT query FROM q)
              ) AS rank
            FROM public.chunks_fts
            WHERE to_tsvector('english', unaccent(coalesce(text,''))) @@ (SELECT query FROM q)
            {"AND document_id = :doc_id" if doc_id else ""}
            ORDER BY rank DESC
            LIMIT :limit
        """)

    # IMPORTANT: run the chosen SQL in a fresh transaction
    with engine.begin() as conn:
        rows = conn.execute(sql, params).mappings().all()

    return [dict(r) for r in rows]

# ==========================================
# Qdrant vector search (signature-adaptive)
# ==========================================

def _supports_kw(param_map: Dict[str, inspect.Parameter], *names: str) -> bool:
    return any(n in param_map for n in names)

def _first_required_positional_is(params: Dict[str, inspect.Parameter], name: str) -> bool:
    idx = 0
    for p in params.values():
        if p.default is p.empty and p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD):
            return p.name == name and idx == 0
        if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD):
            idx += 1
    return False

def _call_search_qdrant(raw_query: str, vec: List[float], top_k: int, doc_id: Optional[str], collection: str, embedder: Any):
    """
    Call utils.search_qdrant with the correct argument shape.

    Supports:
      - search_qdrant(query: str, top_k: int, collection: str, embedder: ..., *, doc_id=..., include_text=...)
      - search_qdrant(vector, top_k, collection, embedder, [doc_id])
      - search_qdrant(query_vector=..., top_k=..., collection|collection_name=..., embedder=..., doc_id=..., include_text=...)
      - legacy search_qdrant(vector, top_k)
    """
    sig = inspect.signature(search_qdrant)
    params = sig.parameters

    # Decide whether it wants raw text query or a vector as first arg
    wants_query = _supports_kw(params, "query") or _first_required_positional_is(params, "query")

    # Prefer kw call; it's more robust to ordering
    kw: Dict[str, Any] = {}
    if wants_query:
        if _supports_kw(params, "query"):
            kw["query"] = raw_query
    else:
        if _supports_kw(params, "vector"):
            kw["vector"] = vec
        elif _supports_kw(params, "query_vector"):
            kw["query_vector"] = vec

    if _supports_kw(params, "top_k"):
        kw["top_k"] = top_k

    if _supports_kw(params, "collection"):
        kw["collection"] = collection
    elif _supports_kw(params, "collection_name"):
        kw["collection_name"] = collection

    if _supports_kw(params, "embedder"):
        kw["embedder"] = embedder

    if _supports_kw(params, "include_text"):
        kw["include_text"] = True

    if doc_id is not None and _supports_kw(params, "doc_id"):
        try:
            kw["doc_id"] = int(doc_id)
        except Exception:
            kw["doc_id"] = doc_id

    if kw:
        try:
            return search_qdrant(**kw)
        except TypeError:
            # fall through to positional attempts
            pass

    # Build positional args map
    required_pos = [
        p for p in params.values()
        if p.default is p.empty and p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]

    args: List[Any] = []
    # arg1: query or vector
    args.append(raw_query if wants_query else vec)
    # arg2: top_k
    if len(required_pos) >= 2:
        args.append(top_k)
    # arg3: collection
    if len(required_pos) >= 3:
        args.append(collection)
    # arg4: embedder
    if len(required_pos) >= 4:
        args.append(embedder)
    # arg5: doc_id (positional)
    if len(required_pos) >= 5:
        try:
            args.append(int(doc_id) if doc_id is not None else None)
        except Exception:
            args.append(doc_id)

    try:
        return search_qdrant(*args)
    except TypeError as e:
        # Final legacy fallback
        try:
            return search_qdrant(vec, top_k)
        except TypeError as e2:
            raise TypeError(
                f"Adaptive call to search_qdrant failed. Signature={sig}. "
                f"Tried kwargs (wants_query={wants_query}) and positional variants. "
                f"Last errors: {e!r} | {e2!r}"
            )

def _vector_search(q: str, limit: int, doc_id: Optional[str]) -> List[dict]:
    embedder = get_embedder()
    vec = embedder.encode([q])[0]
    vec = vec.tolist() if hasattr(vec, "tolist") else list(vec)

    k = max(limit, 16)
    hits = _call_search_qdrant(q, vec, k, doc_id, QDRANT_COLLECTION, embedder)

    out: List[dict] = []
    for h in hits or []:
        payload = h.get("payload", {}) if isinstance(h, dict) else getattr(h, "payload", {}) or {}
        score = float(h.get("score", 0.0) if isinstance(h, dict) else getattr(h, "score", 0.0))
        out.append({
            "score": score,
            "document_id": str(payload.get("doc_id", "")),
            "page": payload.get("page"),
            "chunk_idx": payload.get("chunk_idx"),
            "text": payload.get("text") or "",
            "filename": payload.get("filename") or "",
            "composite_id": payload.get("composite_id") or "",
            "url": payload.get("url") or "",
            # NEW optional citation hints from payload (safe if missing)
            "source_id": payload.get("source_id"),
            "source_title": payload.get("source_title"),
            "source_url": payload.get("source_url"),
            "source_kind": payload.get("source_kind"),
            "source_locator": payload.get("source_locator"),
        })

    return [h for h in out if (h["document_id"] and h["text"].strip())]

# ========================
# Fusion (late weighted)
# ========================

def _fuse(vec_hits: List[dict], fts_hits: List[dict], top_k: int, alpha: float) -> List[ContextChunk]:
    vmap, fmap, meta = {}, {}, {}

    # vector path
    for h in vec_hits:
        k = _key(h["document_id"], h.get("page"), h.get("chunk_idx"))
        vmap[k] = max(vmap.get(k, 0.0), float(h.get("score", 0.0)))
        m = meta.setdefault(k, {})
        m.update({
            "text": h.get("text") or m.get("text", ""),
            "filename": h.get("filename") or m.get("filename", ""),
            "composite_id": h.get("composite_id") or m.get("composite_id", ""),
            "url": h.get("url") or m.get("url", ""),
            "doc_id": h.get("document_id"),
            "page": h.get("page"),
            "chunk_idx": h.get("chunk_idx"),
            # citation from vector payload (if present)
            "source_id": h.get("source_id") or m.get("source_id"),
            "source_title": h.get("source_title") or m.get("source_title"),
            "source_url": h.get("source_url") or m.get("source_url"),
            "source_kind": h.get("source_kind") or m.get("source_kind"),
            "source_locator": h.get("source_locator") or m.get("source_locator"),
        })

    # fts path
    for h in fts_hits:
        k = _key(str(h["document_id"]), h.get("page"), h.get("chunk_idx"))
        fmap[k] = max(fmap.get(k, 0.0), float(h.get("rank", 0.0)))
        m = meta.setdefault(k, {})
        m.update({
            "text": m.get("text") or (h.get("text") or ""),
            "filename": m.get("filename") or (h.get("filename") or ""),
            "composite_id": m.get("composite_id") or (h.get("composite_id") or ""),
            "doc_id": m.get("doc_id") or str(h.get("document_id")),
            "page": m.get("page") if m.get("page") is not None else h.get("page"),
            "chunk_idx": m.get("chunk_idx") if m.get("chunk_idx") is not None else h.get("chunk_idx"),
            # citation from FTS join (if available)
            "source_id": m.get("source_id") or h.get("source_id"),
            "source_title": m.get("source_title") or h.get("source_title"),
            "source_url": m.get("source_url") or h.get("source_url"),
            "source_kind": m.get("source_kind") or h.get("source_kind"),
            "source_locator": m.get("source_locator") or h.get("source_locator"),
        })

    vnorm, fnorm = _normalize_scores(vmap), _normalize_scores(fmap)

    fused: List[ContextChunk] = []
    for k_ in set(vnorm.keys()) | set(fnorm.keys()):
        vs, fs = vnorm.get(k_, 0.0), fnorm.get(k_, 0.0)
        score = alpha * vs + (1.0 - alpha) * fs
        m = meta.get(k_, {})

        citation = None
        if m.get("source_id"):
            citation = Citation(
                id=str(m.get("source_id")),
                title=m.get("source_title"),
                url=m.get("source_url") or m.get("url"),
                kind=m.get("source_kind"),
                locator=m.get("source_locator") if isinstance(m.get("source_locator"), dict) else None,
            )

        fused.append(
            ContextChunk(
                doc_id=m.get("doc_id",""),
                page=m.get("page"),
                chunk_idx=m.get("chunk_idx"),
                text=m.get("text",""),
                scores={"vector": vs, "fts": fs, "fused": score},
                where={"filename": m.get("filename",""), "composite_id": m.get("composite_id",""), "url": m.get("url","")},
                citation=citation,
            )
        )

    fused.sort(key=lambda x: x.scores["fused"], reverse=True)
    return fused[: top_k * 3]

# ========================
# Deduplication (hash)
# ========================

def _normalize_for_hash(s: str) -> str:
    if not s:
        return ""
    s2 = s.strip().lower()
    if DEDUP_NORM_WHITESPACE:
        s2 = _WS_RE.sub(" ", s2)
    return s2

def _hash_text(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()

def _dedup_chunks(chunks: List[ContextChunk]) -> List[ContextChunk]:
    seen: Dict[str, ContextChunk] = {}
    for c in chunks:
        txt = c.text or ""
        if len(txt) < DEDUP_MIN_CHARS:
            key = f"short::{txt}"
        else:
            key = "sha1::" + _hash_text(_normalize_for_hash(txt))
        if key in seen:
            prev = seen[key]
            prev_score = prev.scores.get("rerank", prev.scores.get("fused", 0.0))
            cur_score = c.scores.get("rerank", c.scores.get("fused", 0.0))
            if cur_score > prev_score:
                seen[key] = c
        else:
            seen[key] = c
    out = list(seen.values())
    out.sort(key=lambda x: x.scores.get("rerank", x.scores.get("fused", 0.0)), reverse=True)
    return out

# ==================================
# Optional Reranker (CrossEncoder)
# ==================================

_reranker = None
_reranker_name_loaded: Optional[str] = None

def _get_reranker(model_name: str):
    global _reranker, _reranker_name_loaded
    if _reranker is not None and _reranker_name_loaded == model_name:
        return _reranker
    try:
        from sentence_transformers import CrossEncoder
    except Exception as e:
        raise RuntimeError(f"Reranker requested but sentence-transformers not available: {e}")
    _reranker = CrossEncoder(model_name, max_length=512)
    _reranker_name_loaded = model_name
    return _reranker

def _maybe_rerank(query: str, chunks: List[ContextChunk], top_k: int, enabled: bool) -> List[ContextChunk]:
    if not enabled or not chunks:
        return chunks[:top_k]

    # Prefer non-empty text for reranking
    nonempty = [c for c in chunks if c.text and c.text.strip()]
    cand_pool = nonempty if nonempty else chunks
    cand = cand_pool[: min(len(cand_pool), RERANKER_MAX_CANDIDATES)]

    try:
        reranker = _get_reranker(RERANKER_MODEL)
    except Exception:
        return cand[:top_k]

    pairs = [(query, c.text) for c in cand]
    try:
        scores = reranker.predict(pairs)
    except Exception:
        return cand[:top_k]

    if len(scores) > 0:
        s_min = float(min(scores))
        s_max = float(max(scores))
        for c, s in zip(cand, scores):
            c.scores["rerank_raw"] = float(s)
            c.scores["rerank"] = (float(s) - s_min) / (s_max - s_min) if s_max > s_min else 1.0
        cand.sort(key=lambda x: x.scores.get("rerank_raw", 0.0), reverse=True)

    return cand[:top_k]

# ============
# Route
# ============

@router.post("/", response_model=QueryResponse)
def hybrid_query(req: QueryRequest):
    q = (req.q or "").strip()
    if not q:
        raise HTTPException(status_code=400, detail="q is required")

    # 1) retrieve
    vec_hits = _vector_search(q, limit=req.top_k, doc_id=req.doc_id)
    fts_hits = _fts_search(q, limit=max(1, req.top_k) * 4, doc_id=req.doc_id)

    # 2) fuse (keep extra headroom for reranker & dedup)
    fused = _fuse(vec_hits, fts_hits, top_k=req.top_k, alpha=req.alpha)

    # 3) dedup BEFORE rerank to avoid wasting reranker calls on clones
    fused = _dedup_chunks(fused)

    # 4) optional rerank (request flag OR env default)
    use_reranker = req.use_reranker or os.getenv("DEFAULT_USE_RERANKER", "false").lower() in ("1", "true", "yes")
    reranked = _maybe_rerank(q, fused, top_k=req.top_k, enabled=use_reranker)

    # 5) optional min_score post-filter on fused score
    if req.min_score is not None:
        reranked = [c for c in reranked if c.scores.get("fused", 0.0) >= req.min_score]

    debug = None
    if req.debug:
        debug = {
            "vector_candidates": len(vec_hits),
            "fts_candidates": len(fts_hits),
            "fused_after_merge": len(fused),
            "reranker": RERANKER_MODEL if use_reranker else None,
            "reranker_candidates": min(len(fused), RERANKER_MAX_CANDIDATES) if use_reranker else 0,
            "kept_after_all": len(reranked),
        }

    return QueryResponse(
        query=req.q,
        top_k=req.top_k,
        alpha=req.alpha,
        results=reranked,
        debug=debug,
    )
