# RAG Micro (FastAPI + Qdrant + Postgres + Vite/React/Tailwind)

A minimal, dockerized RAG scaffold designed for a **2‑hour micro‑sprint**:
- **Backend:** FastAPI, PDF ingestion, chunking, sentence-transformer embeddings, Qdrant vector store, Postgres metadata.
- **Frontend:** Vite + React + Tailwind for PDF upload and simple question UI.
- **Infra:** Docker Compose with `postgres`, `qdrant`, `api`, and `web` services.

## Quick Start

```bash
cp .env.example .env
docker compose up --build
```

- API: http://localhost:8000 (health: `/health`)
- Web: http://localhost:5173

## Flow

1. Upload up to 5 PDFs via the web UI (or `POST /ingest/upload`).
2. The API extracts text per page (pdfplumber → PyMuPDF fallback), chunks text, embeds with `all-MiniLM-L6-v2`, and upserts to Qdrant.
3. Ask a question via the web UI (or `POST /query`). You’ll get top hits and a context-only draft (no LLM yet).

## Notable Files

- `services/api/app/routes/ingest.py` — file upload + ingest → Qdrant.
- `services/api/app/routes/query.py` — semantic search over Qdrant.
- `services/api/app/utils/embed.py` — embedding + Qdrant helpers.
- `services/api/app/utils/pdf.py` — page-level text extraction (basic OCR fallback uses PyMuPDF text only; add OCR later).
- `services/api/app/utils/chunk.py` — adjustable chunking env vars.

## Next Steps (suggested, ~1–2 sprints)

- Add **LLM answer synthesis** (NVIDIA NeMo, OpenAI/Groq, or local models).
- Add **source-citation formatting** (filename, page, score).
- Implement **auth** and **rate limiting** for the API.
- Add **alembic** migrations; expand metadata (doc hashes, MIME types).
- Add **OCR** (PaddleOCR or Tesseract) for scanned PDFs.
- Implement **batch status** & job queue (RQ/Celery) for large uploads.
- Replace sentence-transformers with your **NeMo embedding head** when ready.
- Frontend: show hits with document/page chips and copy-to-clipboard.
- Observability: Prometheus/Grafana or OpenTelemetry.

---

*Scaffold generated on 2025-09-20T14:58:20.546532Z.*
