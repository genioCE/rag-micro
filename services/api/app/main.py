# services/api/app/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.staticfiles import StaticFiles
from pathlib import Path
import os
import logging

from .db import init_db, init_fts
from .routes.ingest import router as ingest_router
from .routes.query import router as query_router
from .routes.answers import router as answers_router   # <- import once
from .routes.preview import router as preview_router   # <- import once

logger = logging.getLogger(__name__)

app = FastAPI(title="RAG Micro API", version="0.1.0")

# CORS
ALLOW_ORIGINS = os.getenv("ALLOW_ORIGINS", "*")
allow_origins = [o.strip() for o in ALLOW_ORIGINS.split(",")] if ALLOW_ORIGINS else ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static mounts
PREVIEW_CACHE_DIR = Path(os.getenv("PREVIEW_CACHE_DIR", "/cache/preview")).resolve()
PREVIEW_CACHE_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static/preview", StaticFiles(directory=str(PREVIEW_CACHE_DIR)), name="preview")

FILES_DIR = os.getenv("FILES_DIR")
if FILES_DIR and os.path.isdir(FILES_DIR):
    app.mount("/files", StaticFiles(directory=FILES_DIR), name="files")
    logger.info("Mounted /files from %s", FILES_DIR)

# Startup
@app.on_event("startup")
def _startup():
    init_db()
    init_fts()
    logger.info("Startup OK. Preview cache: %s", PREVIEW_CACHE_DIR)

@app.on_event("shutdown")
def _shutdown():
    pass

# Routers (mount each exactly once)
app.include_router(ingest_router,  prefix="/ingest",  tags=["ingest"])
app.include_router(query_router,   prefix="/query",   tags=["query"])
app.include_router(answers_router, prefix="/answers", tags=["answers"])
app.include_router(preview_router, prefix="/preview", tags=["preview"])

# Health
@app.get("/health")
def health():
    return {
        "status": "ok",
        "version": app.version,
        "preview_static": str(PREVIEW_CACHE_DIR),
        "files_dir": FILES_DIR if FILES_DIR else None,
        "cors_origins": allow_origins,
    }
