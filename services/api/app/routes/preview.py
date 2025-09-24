from fastapi import APIRouter, HTTPException, Query
from ..db import engine
import os, json, re
from datetime import datetime, timedelta
from pathlib import Path
import httpx
from readability import Document
from lxml.html.clean import Cleaner
import lxml.html as LH
import fitz  # PyMuPDF
import html
from sqlalchemy import text as sa_text


router = APIRouter()

CACHE_DIR = Path(os.getenv("PREVIEW_CACHE_DIR", "/cache/preview")).resolve()
CACHE_DIR.mkdir(parents=True, exist_ok=True)
REVALIDATE_HOURS = int(os.getenv("HTML_REVALIDATE_HOURS", "12"))

# -------------------- utils --------------------

def _table_exists(conn) -> bool:
    return bool(conn.execute(sa_text("SELECT to_regclass('public.sources')")).scalar())

def _get_source(citation_id: str):
    sql = sa_text("""
        SELECT id, url, kind, locator, title, raw_text, raw_fetched_at, pdf_blob_path
        FROM public.sources
        WHERE id = :id
    """)
    with engine.begin() as conn:
        if not _table_exists(conn):
            return None
        row = conn.execute(sql, {"id": citation_id}).fetchone()
        return row

async def _fetch_html_text(url: str) -> str:
    async with httpx.AsyncClient(
        follow_redirects=True,
        timeout=12.0,
        headers={"User-Agent": "RAG-Micro/preview"}
    ) as client:
        r = await client.get(url)
        r.raise_for_status()
        doc = Document(r.text)
        html_part = doc.summary(html_partial=True)
        cleaner = Cleaner(
            scripts=True, javascript=True, style=True, comments=True,
            links=False, meta=False, page_structure=False, safe_attrs_only=False
        )
        cleaned = cleaner.clean_html(html_part)
        text = LH.fromstring(cleaned).text_content()
        return " ".join(text.split())

def _make_preview(text: str, query: str | None, window: int = 180, max_frags: int = 2):
    """Return (preview_html, preview_text)."""
    text = text or ""
    safe = html.escape(text)

    if not query:
        snippet = safe[:window] + ("…" if len(safe) > window else "")
        return snippet, text[:window] + ("…" if len(text) > window else "")

    toks = [re.escape(t) for t in re.split(r"\s+", query.strip()) if t]
    if not toks:
        return _make_preview(text, None, window, max_frags)

    pat = re.compile(r"(" + "|".join(toks) + r")", flags=re.IGNORECASE)
    matches = list(pat.finditer(safe))
    if not matches:
        return _make_preview(text, None, window, max_frags)

    frags = []
    for m in matches[:max_frags]:
        start = max(0, m.start() - window // 2)
        end = min(len(safe), start + window)
        frag = safe[start:end]
        frag = pat.sub(r"<mark>\1</mark>", frag)
        prefix = "…" if start > 0 else ""
        suffix = "…" if end < len(safe) else ""
        frags.append(prefix + frag + suffix)

    html_out = " … ".join(frags)
    text_out = re.sub(r"</?mark>", "", html_out)
    # best-effort unescape for plain text
    text_out = LH.fromstring(f"<div>{text_out}</div>").text_content()
    return html_out, text_out

# -------------------- route --------------------

@router.get("/{citation_id}")
async def get_preview(
    citation_id: str,
    q: str | None = Query(default=None, description="Optional terms for highlighting"),
    page: int | None = Query(default=None, description="Override page for PDFs"),
):
    row = _get_source(citation_id)
    if not row:
        raise HTTPException(status_code=404, detail="Preview not available (no sources table or citation not found)")

    cid, url, kind, locator, title, raw_text, raw_fetched_at, pdf_blob_path = row
    if not url:
        raise HTTPException(status_code=404, detail="No URL stored for this citation")

    locator = locator if isinstance(locator, dict) else json.loads(locator or "{}")
    title = title or url

    # HTML / TEXT: cache & highlight
    if kind in ("html", "text"):
        must_refetch = True
        if raw_text and raw_fetched_at:
            try:
                must_refetch = (datetime.utcnow() - raw_fetched_at) > timedelta(hours=REVALIDATE_HOURS)
            except Exception:
                pass

        if must_refetch:
            try:
                text = await _fetch_html_text(url)
                with engine.begin() as conn:
                    conn.execute(
                        sa_text("UPDATE public.sources SET raw_text = :text, raw_fetched_at = :ts WHERE id = :id"),
                        {"text": text, "ts": datetime.utcnow(), "id": cid},
                    )
                raw_text = text
            except Exception as e:
                if not raw_text:
                    raise HTTPException(status_code=502, detail=f"fetch failed: {e}")

        html_snip, text_snip = _make_preview(raw_text or "", q)
        return {
            "citation_id": cid,
            "title": title,
            "kind": kind,
            "snippet_html": html_snip,
            "snippet_text": text_snip,
            "full_available": bool(raw_text),
            "url": url,
        }

    # PDF: render/cached page image (optionally override page with ?page=)
    if kind == "pdf":
        pgnum = int(page or locator.get("page", 1))
        if pgnum < 1:
            raise HTTPException(status_code=400, detail="Invalid PDF page")

        cdir = CACHE_DIR / "pdf" / cid
        cdir.mkdir(parents=True, exist_ok=True)
        png_path = cdir / f"p{pgnum}@2x.png"
        pdf_path = cdir / "doc.pdf"  # local cache target

        # Prefer locally cached blob (set during ingest)
        src_pdf_path = None
        if pdf_blob_path:
            p = Path(pdf_blob_path)
            if p.exists():
                src_pdf_path = p

        # Ensure we have a PDF file to open under cdir
        if not pdf_path.exists():
            if src_pdf_path and src_pdf_path.exists():
                try:
                    pdf_path.write_bytes(src_pdf_path.read_bytes())
                except Exception:
                    src_pdf_path = None  # fall through to URL fetch

            if not pdf_path.exists():
                # last resort: fetch from URL
                if not url or not url.startswith(("http://", "https://")):
                    raise HTTPException(status_code=502, detail="No cached PDF and URL is not fetchable")
                async with httpx.AsyncClient(follow_redirects=True, timeout=30.0) as client:
                    r = await client.get(url)
                    r.raise_for_status()
                    pdf_path.write_bytes(r.content)

        # Render page image to cache
        if not png_path.exists():
            try:
                doc = fitz.open(pdf_path.as_posix())
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Unable to open PDF: {e}")

            try:
                pg = doc.load_page(pgnum - 1)
            except Exception:
                doc.close()
                raise HTTPException(status_code=400, detail="PDF page out of bounds")

            pix = pg.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
            pix.save(png_path.as_posix())
            doc.close()

        return {
            "citation_id": cid,
            "title": title or url,
            "kind": kind,
            "page_num": pgnum,
            "page_image_url": f"/static/preview/pdf/{cid}/p{pgnum}@2x.png",
            "highlight_bbox": locator.get("bbox"),  # optional [x0,y0,x1,y1]
            "url": url,
        }

    # Unknown kind
    raise HTTPException(status_code=400, detail=f"Unknown kind: {kind}")
