import io
import os
import hashlib
from typing import List
from uuid import uuid4
from pathlib import Path
from datetime import datetime

from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from sqlalchemy.orm import Session
from sqlalchemy import bindparam, Integer, text as sa_text
from sqlalchemy.types import JSON
from sqlalchemy.dialects.postgresql import UUID as PG_UUID, TEXT as PG_TEXT

from ..db import SessionLocal, init_db, init_fts, engine
from ..models import Document, Chunk
from ..utils.pdf import extract_text_by_pages
from ..utils.chunk import chunk_texts
from ..utils.embed import get_embedder, ensure_qdrant, upsert_embeddings

router = APIRouter()

# ------------------------------------------------------------------------------
# Config
# ------------------------------------------------------------------------------
MAX_FILES = int(os.getenv("MAX_UPLOAD_FILES", "5"))
COLLECTION = os.getenv("QDRANT_COLLECTION", "docs")
PREVIEW_CACHE_DIR = Path(os.getenv("PREVIEW_CACHE_DIR", "/cache/preview")).resolve()
PREVIEW_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------------------------
# DB helpers
# ------------------------------------------------------------------------------
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Ensure minimal schema for sources (idempotent)
ENSURE_SOURCES_SQL = sa_text("""
CREATE TABLE IF NOT EXISTS public.sources (
  id TEXT PRIMARY KEY,
  url TEXT NOT NULL UNIQUE,
  kind TEXT NOT NULL CHECK (kind IN ('html','pdf','text')),
  locator JSONB NOT NULL,
  title TEXT,
  raw_text TEXT,
  pdf_blob_path TEXT,
  raw_fetched_at TIMESTAMPTZ
);
CREATE INDEX IF NOT EXISTS idx_sources_url ON public.sources(url);
""")

# Proper upsert using JSON bind param (no ::jsonb in the SQL text)
UPSERT_SOURCE_SQL = sa_text("""
INSERT INTO public.sources (id, url, kind, locator, title, pdf_blob_path, raw_fetched_at)
VALUES (:id, :url, 'pdf', :locator, :title, :pdf_path, :ts)
ON CONFLICT (id) DO UPDATE
  SET url = EXCLUDED.url,
      kind = EXCLUDED.kind,
      locator = EXCLUDED.locator,
      title = COALESCE(EXCLUDED.title, public.sources.title),
      pdf_blob_path = EXCLUDED.pdf_blob_path,
      raw_fetched_at = COALESCE(public.sources.raw_fetched_at, EXCLUDED.raw_fetched_at)
""").bindparams(
    bindparam("locator", type_=JSON)
)

def _cid_for_url(url: str) -> str:
    """Stable citation id based on URL-ish identifier."""
    h = hashlib.sha1((url or "").encode("utf-8")).hexdigest()
    return f"c_{h[:16]}"

def _ensure_cached_pdf(cid: str, pdf_bytes: bytes) -> str:
    """Write uploaded PDF once to preview cache and return absolute path."""
    cdir = PREVIEW_CACHE_DIR / "pdf" / cid
    cdir.mkdir(parents=True, exist_ok=True)
    pdf_path = cdir / "doc.pdf"
    if not pdf_path.exists():
        pdf_path.write_bytes(pdf_bytes)
    return pdf_path.as_posix()

def _upsert_source_for_pdf(conn, *, url: str, title: str | None, pdf_path: str, default_page: int = 1) -> str:
    """
    Upsert into public.sources and return cid.
    - kind='pdf'
    - locator stores a default page; client can override with ?page=
    - url can be a local scheme (local://...) for uploaded docs
    """
    cid = _cid_for_url(url)
    # ensure table exists (idempotent, cheap)
    conn.execute(ENSURE_SOURCES_SQL)

    conn.execute(
        UPSERT_SOURCE_SQL,
        {
            "id": cid,
            "url": url,
            "locator": {"page": int(default_page)},
            "title": title,
            "pdf_path": pdf_path,
            "ts": datetime.utcnow(),
        },
    )
    return cid

# ------------------------------------------------------------------------------
# App startup: ensure DB + FTS + Qdrant
# ------------------------------------------------------------------------------
@router.on_event("startup")
def _startup():
    init_db()
    init_fts()
    embedder = get_embedder()
    ensure_qdrant(
        collection=COLLECTION,
        vector_size=embedder.get_sentence_embedding_dimension(),
    )

# ------------------------------------------------------------------------------
# Mirror chunks into FTS table
# ------------------------------------------------------------------------------
def _mirror_chunks_to_fts(doc_id: str, filename: str, records: List[Chunk]) -> int:
    """
    Mirror chunk text into the Postgres FTS mirror table 'chunks_fts'.
    Ensures correct UUID binding and explicit param types to avoid executemany ambiguities.
    """
    if not records:
        return 0

    insert_sql = sa_text("""
        INSERT INTO public.chunks_fts (id, document_id, page, chunk_idx, text, meta)
        VALUES (
            :id, :document_id, :page, :chunk_idx, :text,
            jsonb_build_object(
                'filename', :filename,
                'composite_id', :composite_id
            )
        )
        ON CONFLICT (id) DO NOTHING
    """).bindparams(
        bindparam("id", type_=PG_UUID(as_uuid=True)),
        bindparam("document_id", type_=PG_TEXT),
        bindparam("page", type_=Integer),
        bindparam("chunk_idx", type_=Integer),
        bindparam("text", type_=PG_TEXT),
        bindparam("filename", type_=PG_TEXT),
        bindparam("composite_id", type_=PG_TEXT),
    )

    params = []
    for c in records:
        params.append({
            "id": uuid4(),  # real UUID object
            "document_id": str(doc_id),
            "page": int(c.page) if c.page is not None else None,
            "chunk_idx": int(c.chunk_idx) if c.chunk_idx is not None else None,
            "text": c.text or "",
            "filename": filename or "",
            "composite_id": f"{doc_id}-{c.page}-{c.chunk_idx}",
        })

    BATCH = 500
    inserted = 0
    with engine.begin() as conn:
        for i in range(0, len(params), BATCH):
            batch = params[i:i + BATCH]
            conn.execute(insert_sql, batch)  # executemany
            inserted += len(batch)
    return inserted

# ------------------------------------------------------------------------------
# Upload route
# ------------------------------------------------------------------------------
@router.post("/upload")
def upload_pdfs(files: List[UploadFile] = File(...), db: Session = Depends(get_db)):
    if len(files) > MAX_FILES:
        raise HTTPException(status_code=400, detail=f"Max {MAX_FILES} files allowed.")

    embedder = get_embedder()
    ingested = []

    for f in files:
        if not f.filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail=f"{f.filename} is not a PDF")

        content = f.file.read()
        if not content:
            raise HTTPException(status_code=400, detail=f"{f.filename} is empty")

        # 1) Parse PDF pages and create Document
        pages = extract_text_by_pages(io.BytesIO(content))
        doc = Document(filename=f.filename, size_bytes=len(content), pages=len(pages))
        db.add(doc)
        db.commit()
        db.refresh(doc)

        # 2) Create a local:// source + cache the exact bytes we got
        local_url = f"local://pdf/{doc.id}:{doc.filename}"
        cid_local = _cid_for_url(local_url)
        pdf_path = _ensure_cached_pdf(cid_local, content)

        with engine.begin() as conn:
            source_id = _upsert_source_for_pdf(
                conn,
                url=local_url,
                title=doc.filename,
                pdf_path=pdf_path,
                default_page=1,
            )

        # 3) Chunk per page to keep page mapping
        records: List[Chunk] = []
        for page_i, text_page in enumerate(pages, start=1):
            for idx, chunk in enumerate(chunk_texts(text_page)):
                rec = Chunk(document_id=doc.id, chunk_idx=idx, page=page_i, text=chunk)
                db.add(rec)
                records.append(rec)
        db.commit()

        # 4) Prepare Qdrant payloads
        payloads, texts, ids = [], [], []
        for c in records:
            texts.append(c.text)
            ids.append(str(uuid4()))  # Qdrant uses string UUIDs for point IDs
            payloads.append({
                "doc_id": str(doc.id),
                "filename": doc.filename,
                "page": c.page,
                "chunk_idx": c.chunk_idx,
                "composite_id": f"{doc.id}-{c.page}-{c.chunk_idx}",
                "text": c.text,  # optional: helps in Qdrant UI debugging
                # helpful for UI/preview/debugging
                "source_id": source_id,
                "source_title": doc.filename,
                "source_url": local_url,
                "source_kind": "pdf",
                "source_locator": {"page": c.page},
            })

        # 5) Embed & upsert to Qdrant
        upsert_embeddings(
            texts=texts,
            ids=ids,
            payloads=payloads,
            collection=COLLECTION,
            embedder=embedder,
        )

        # 6) Mirror chunks into Postgres FTS table
        mirrored = _mirror_chunks_to_fts(str(doc.id), doc.filename, records)

        # 7) Stamp FTS rows with the source_id (so /query join -> citation works)
        with engine.begin() as conn:
            conn.execute(
                sa_text("""
                    ALTER TABLE public.chunks_fts
                      ADD COLUMN IF NOT EXISTS source_id TEXT NULL
                """)
            )
            conn.execute(
                sa_text("""
                    UPDATE public.chunks_fts
                       SET source_id = :cid
                     WHERE document_id = :doc_id
                """),
                {"cid": source_id, "doc_id": str(doc.id)},
            )

        ingested.append({
            "document_id": str(doc.id),
            "filename": doc.filename,
            "pages": doc.pages,
            "chunks": len(texts),
            "fts_mirrored": mirrored,
            "citation_id": source_id,
        })

    return {"status": "ok", "ingested": ingested}
