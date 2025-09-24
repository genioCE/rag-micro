import os
from contextlib import contextmanager

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, DeclarativeBase

# ---- Env / URL (same defaults you had) ---------------------------------------
DB_USER = os.getenv("POSTGRES_USER", "rag")
DB_PASS = os.getenv("POSTGRES_PASSWORD", "ragpass")
DB_NAME = os.getenv("POSTGRES_DB", "ragdb")
DB_HOST = os.getenv("POSTGRES_HOST", "postgres")
DB_PORT = os.getenv("POSTGRES_PORT", "5432")

# SQLAlchemy 2.0 style URL using psycopg (psycopg3)
DATABASE_URL = f"postgresql+psycopg://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# ---- Engine / Session --------------------------------------------------------
engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
    pool_size=int(os.getenv("DB_POOL_SIZE", "5")),
    max_overflow=int(os.getenv("DB_MAX_OVERFLOW", "10")),
    future=True,
)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)

class Base(DeclarativeBase):
    pass

@contextmanager
def session_scope():
    """
    Context manager for a transactional session.
    Usage:
        with session_scope() as s:
            s.execute(...)
    """
    s = SessionLocal()
    try:
        yield s
        s.commit()
    except Exception:
        s.rollback()
        raise
    finally:
        s.close()

# ---- Bootstrap: create tables from SQLAlchemy models -------------------------
def init_db():
    # Import models so metadata is populated, then create tables if missing.
    from .models import Document, Chunk  # noqa: F401
    Base.metadata.create_all(bind=engine)

# ---- Optional: ensure FTS plumbing on 'chunks' table -------------------------
_FTS_BOOTSTRAP_SQL = """
CREATE EXTENSION IF NOT EXISTS unaccent;

CREATE TABLE IF NOT EXISTS chunks_fts (
  id UUID PRIMARY KEY,
  document_id TEXT NOT NULL,
  page INTEGER,
  chunk_idx INTEGER,
  text TEXT NOT NULL,
  meta JSONB DEFAULT '{}'::jsonb
);

DO $$
BEGIN
  IF NOT EXISTS (
    SELECT 1 FROM information_schema.columns
    WHERE table_name='chunks_fts' AND column_name='tsv'
  ) THEN
    ALTER TABLE chunks_fts ADD COLUMN tsv tsvector;
  END IF;
END$$;

UPDATE chunks_fts
SET tsv = to_tsvector('english', unaccent(coalesce(text, '')))
WHERE tsv IS NULL;

CREATE OR REPLACE FUNCTION chunks_fts_tsv_update() RETURNS trigger AS $$
BEGIN
  NEW.tsv := to_tsvector('english', unaccent(coalesce(NEW.text, '')));
  RETURN NEW;
END
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_chunks_fts_tsv_update ON chunks_fts;
CREATE TRIGGER trg_chunks_fts_tsv_update
BEFORE INSERT OR UPDATE OF text ON chunks_fts
FOR EACH ROW EXECUTE FUNCTION chunks_fts_tsv_update();

CREATE INDEX IF NOT EXISTS idx_chunks_fts_tsv ON chunks_fts USING GIN (tsv);
CREATE INDEX IF NOT EXISTS idx_chunks_fts_document ON chunks_fts (document_id);
CREATE INDEX IF NOT EXISTS idx_chunks_fts_document_page ON chunks_fts (document_id, page);
"""
def init_fts():
    """
    Ensures the 'tsv' column, trigger, and GIN index exist on the 'chunks' table.
    Call once on startup after init_db().
    """
    with engine.begin() as conn:
        conn.exec_driver_sql(_FTS_BOOTSTRAP_SQL)
