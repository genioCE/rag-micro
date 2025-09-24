-- Enable useful extensions (safe to re-run)
CREATE EXTENSION IF NOT EXISTS unaccent;
-- If you prefer server-side UUID defaults later:
-- CREATE EXTENSION IF NOT EXISTS pgcrypto;

-- Chunks table for FTS (id generated client-side; vectors stay in Qdrant)
CREATE TABLE IF NOT EXISTS chunks (
  id UUID PRIMARY KEY,
  doc_id TEXT NOT NULL,
  page INTEGER,
  chunk_id INTEGER,
  text TEXT NOT NULL,
  meta JSONB DEFAULT '{}'::jsonb,
  created_at TIMESTAMP DEFAULT NOW()
  -- If you want server-side UUIDs later:
  -- id UUID PRIMARY KEY DEFAULT gen_random_uuid()
);

-- Fast filters
CREATE INDEX IF NOT EXISTS idx_chunks_doc      ON chunks (doc_id);
CREATE INDEX IF NOT EXISTS idx_chunks_doc_page ON chunks (doc_id, page);

-- Generated tsvector column (auto-kept in sync)
ALTER TABLE chunks
  ADD COLUMN IF NOT EXISTS tsv tsvector
  GENERATED ALWAYS AS (
    to_tsvector('english', unaccent(coalesce(text, '')))
  ) STORED;

-- GIN index for FTS
CREATE INDEX IF NOT EXISTS idx_chunks_tsv ON chunks USING GIN (tsv);
