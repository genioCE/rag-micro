-- 1) Extension
CREATE EXTENSION IF NOT EXISTS unaccent;

-- 2) FTS mirror table (separate from your ORM 'chunks' table)
CREATE TABLE IF NOT EXISTS chunks_fts (
  id UUID PRIMARY KEY,
  document_id TEXT NOT NULL,
  page INTEGER,
  chunk_idx INTEGER,
  text TEXT NOT NULL,
  meta JSONB DEFAULT '{}'::jsonb
);

-- 3) Add/refresh tsv column
DO $$
BEGIN
  IF NOT EXISTS (
    SELECT 1 FROM information_schema.columns
    WHERE table_name='chunks_fts' AND column_name='tsv'
  ) THEN
    ALTER TABLE chunks_fts ADD COLUMN tsv tsvector;
  END IF;
END$$;

-- 4) Backfill tsv for any existing rows
UPDATE chunks_fts
SET tsv = to_tsvector('english', unaccent(coalesce(text, '')))
WHERE tsv IS NULL;

-- 5) Trigger to keep tsv in sync
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

-- 6) Indexes
CREATE INDEX IF NOT EXISTS idx_chunks_fts_tsv ON chunks_fts USING GIN (tsv);
CREATE INDEX IF NOT EXISTS idx_chunks_fts_document ON chunks_fts (document_id);
CREATE INDEX IF NOT EXISTS idx_chunks_fts_document_page ON chunks_fts (document_id, page);
