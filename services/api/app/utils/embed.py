# services/api/app/utils/embed.py
import os
import uuid
import json
import asyncio
from typing import List, Dict, Any, Optional, Iterable, Tuple, AsyncGenerator

import httpx
from qdrant_client import QdrantClient
from qdrant_client.http import models as qm
from sentence_transformers import SentenceTransformer

_QDRANT: Optional[QdrantClient] = None
_EMBEDDER: Optional[SentenceTransformer] = None

# -----------------------------
# Ollama streaming configuration
# -----------------------------
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://ollama:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1")
OLLAMA_TIMEOUT = float(os.getenv("OLLAMA_TIMEOUT_S", "30"))

# =====================================================================================
# Qdrant helpers
# =====================================================================================

def get_qdrant() -> QdrantClient:
    """
    Prefer QDRANT_URL if set (e.g., http://qdrant:6333), otherwise fall back to host/port.
    """
    global _QDRANT
    if _QDRANT is None:
        url = os.getenv("QDRANT_URL")
        if url:
            _QDRANT = QdrantClient(url=url, timeout=30.0)
        else:
            host = os.getenv("QDRANT_HOST", "qdrant")
            port = int(os.getenv("QDRANT_PORT", "6333"))
            _QDRANT = QdrantClient(host=host, port=port, timeout=30.0)
    return _QDRANT


def get_embedder() -> SentenceTransformer:
    global _EMBEDDER
    if _EMBEDDER is None:
        model_name = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
        _EMBEDDER = SentenceTransformer(model_name)
    return _EMBEDDER


def ensure_qdrant(collection: str, vector_size: int, distance: qm.Distance = qm.Distance.COSINE):
    """
    Create the collection if it does not exist.
    """
    client = get_qdrant()
    existing = {c.name for c in client.get_collections().collections}
    if collection not in existing:
        client.create_collection(
            collection_name=collection,
            vectors_config=qm.VectorParams(size=vector_size, distance=distance),
        )


def _sanitize_point_id(pid) -> str | int:
    """
    Qdrant allows: unsigned int OR UUID string.
    - If int-like -> return int
    - If valid UUID string -> normalized UUID str
    - Otherwise -> generate a new UUID
    """
    if isinstance(pid, int):
        return pid
    s = str(pid) if pid is not None else ""
    if s.isdigit():
        return int(s)
    try:
        return str(uuid.UUID(s))
    except Exception:
        return str(uuid.uuid4())


def _chunked(iterable: List[Any], size: int) -> Iterable[Tuple[int, List[Any]]]:
    for i in range(0, len(iterable), size):
        yield i, iterable[i : i + size]


def upsert_embeddings(
    texts: List[str],
    ids: List[Any],
    payloads: List[Dict[str, Any]],
    collection: str,
    embedder: SentenceTransformer,
):
    """
    Upsert (vector, payload, id) triplets into Qdrant.
    - IDs are sanitized to valid Qdrant IDs (UUID or int).
    - Upserts happen in batches for reliability on large uploads.
    """
    if not texts:
        return

    if payloads and len(payloads) != len(texts):
        raise ValueError("payloads length must match texts length")

    # Ensure one id per text (generate UUIDs if missing/misaligned)
    if not ids or len(ids) != len(texts):
        ids = [str(uuid.uuid4()) for _ in texts]
    else:
        ids = [_sanitize_point_id(pid) for pid in ids]

    client = get_qdrant()

    # Encode in one pass
    vectors = embedder.encode(texts, normalize_embeddings=True).tolist()

    # Batch settings
    batch_size = int(os.getenv("UPSERT_BATCH_SIZE", "256"))
    wait_flag = os.getenv("QDRANT_WAIT_UPSERT", "true").lower() in ("1", "true", "yes")

    for start, _ in _chunked(texts, batch_size):
        end = min(start + batch_size, len(texts))
        points = [
            qm.PointStruct(
                id=ids[i],
                vector=vectors[i],
                payload={"text": texts[i], **(payloads[i] if payloads else {})},
            )
            for i in range(start, end)
        ]
        client.upsert(collection_name=collection, points=points, wait=wait_flag)


def search_qdrant(
    query: str,
    top_k: int,
    collection: str,
    embedder: SentenceTransformer,
    *,
    doc_id: Optional[int] = None,
    min_score: Optional[float] = None,
    include_text: bool = True,
):
    """
    Search with optional filtering and quality controls:
    - doc_id: restrict to a specific document
    - min_score: drop weak matches
    - overfetch + dedup on (doc_id, page) to reduce repetitive neighbors
    """
    client = get_qdrant()
    qvec = embedder.encode([query], normalize_embeddings=True)[0].tolist()

    qfilter = None
    if doc_id is not None:
        qfilter = qm.Filter(
            must=[qm.FieldCondition(key="doc_id", match=qm.MatchValue(value=doc_id))]
        )

    # Overfetch to allow post-filtering and dedup
    res = client.search(
        collection_name=collection,
        query_vector=qvec,
        limit=max(top_k * 2, top_k),
        with_payload=True,
        query_filter=qfilter,
    )

    hits = []
    seen = set()  # (doc_id, page)
    for r in res:
        if min_score is not None and r.score < min_score:
            continue
        payload = r.payload or {}
        key = (payload.get("doc_id"), payload.get("page"))
        if key in seen:
            continue
        seen.add(key)

        hits.append(
            {
                "id": r.id,
                "score": r.score,
                "text": (payload.get("text", "") if include_text else None),
                "filename": payload.get("filename"),
                "page": payload.get("page"),
                "chunk_idx": payload.get("chunk_idx"),
                "doc_id": payload.get("doc_id"),
                "composite_id": payload.get("composite_id"),
            }
        )
        if len(hits) >= top_k:
            break

    return hits

# =====================================================================================
# Ollama streaming (for SSE/WebSocket)
# =====================================================================================

async def _stream_chat(client: httpx.AsyncClient, model: str, messages: list) -> AsyncGenerator[str, None]:
    url = f"{OLLAMA_HOST.rstrip('/')}/api/chat"
    async with client.stream("POST", url, json={"model": model, "messages": messages, "stream": True}) as r:
        # If chat isn’t supported, signal the caller to fall back
        if r.status_code in (404, 405, 501):
            raise RuntimeError("CHAT_UNSUPPORTED")
        r.raise_for_status()
        async for line in r.aiter_lines():
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            msg = obj.get("message")
            if msg and isinstance(msg, dict):
                content = msg.get("content")
                if content:
                    yield content

async def _stream_generate(client: httpx.AsyncClient, model: str, prompt: str) -> AsyncGenerator[str, None]:
    url = f"{OLLAMA_HOST.rstrip('/')}/api/generate"
    async with client.stream("POST", url, json={"model": model, "prompt": prompt, "stream": True}) as r:
        r.raise_for_status()
        async for line in r.aiter_lines():
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            # /api/generate emits {"response":"...", "done":false}
            resp = obj.get("response")
            if resp:
                yield resp

async def ollama_stream(
    prompt: str,
    *,
    context_chunks: Optional[List[str]] = None,
    system_prompt: Optional[str] = None,
    model: Optional[str] = None,
) -> AsyncGenerator[str, None]:
    """
    Stream text from Ollama. Tries /api/chat first; if unsupported, falls back to /api/generate.
    """
    model_name = model or OLLAMA_MODEL
    timeout = httpx.Timeout(OLLAMA_TIMEOUT, connect=10.0)

    # Build chat-style messages
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    if context_chunks:
        ctx = "\n\n".join([f"[{i+1}] {t}" for i, t in enumerate(context_chunks)])
        messages.append({"role": "system", "content": f"Use the following context when helpful:\n{ctx}"})
    messages.append({"role": "user", "content": prompt})

    # Build plain prompt for /api/generate fallback
    if system_prompt or context_chunks:
        header = (system_prompt or "You are a helpful assistant.") + "\n\n"
        if context_chunks:
            header += "Context:\n" + "\n\n".join([f"[{i+1}] {t}" for i, t in enumerate(context_chunks)]) + "\n\n"
        gen_prompt = header + "User: " + prompt + "\nAssistant:"
    else:
        gen_prompt = prompt

    async with httpx.AsyncClient(timeout=timeout) as client:
        try:
            async for delta in _stream_chat(client, model_name, messages):
                yield delta
            return
        except RuntimeError as e:
            if str(e) != "CHAT_UNSUPPORTED":
                raise
        except httpx.HTTPStatusError as e:
            if e.response.status_code not in (404, 405, 501):
                raise
        # Fallback
        async for delta in _stream_generate(client, model_name, gen_prompt):
            yield delta
        await asyncio.sleep(0)
