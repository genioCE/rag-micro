-- Ensure extension (safe to re-run)
CREATE EXTENSION IF NOT EXISTS unaccent;

-- 1) Add the tsv column (plain), if missing
DO $$
BEGIN
  IF NOT EXISTS (
    SELECT 1
    FROM information_schema.columns
    WHERE table_name='chunks' AND column_name='tsv'
  ) THEN
    ALTER TABLE chunks ADD COLUMN tsv tsvector;
  END IF;
END$$;

-- 2) Backfill existing rows
UPDATE chunks
SET tsv = to_tsvector('english', unaccent(coalesce(text, '')))
WHERE tsv IS NULL;

-- 3) Trigger function to keep tsv in sync
CREATE OR REPLACE FUNCTION chunks_tsv_update() RETURNS trigger AS $$
BEGIN
  NEW.tsv := to_tsvector('english', unaccent(coalesce(NEW.text, '')));
  RETURN NEW;
END
$$ LANGUAGE plpgsql;

-- 4) Replace trigger (idempotent)
DROP TRIGGER IF EXISTS trg_chunks_tsv_update ON chunks;
CREATE TRIGGER trg_chunks_tsv_update
BEFORE INSERT OR UPDATE OF text ON chunks
FOR EACH ROW EXECUTE FUNCTION chunks_tsv_update();

-- 5) GIN index on tsv (safe to re-run)
CREATE INDEX IF NOT EXISTS idx_chunks_tsv ON chunks USING GIN (tsv);
