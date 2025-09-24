-- Add nullable FK to sources
ALTER TABLE public.chunks_fts
  ADD COLUMN IF NOT EXISTS source_id TEXT NULL;

-- (optional but recommended) FK + index
DO $$
BEGIN
  IF NOT EXISTS (
    SELECT 1 FROM pg_constraint
    WHERE conname = 'chunks_fts_source_id_fkey'
  ) THEN
    ALTER TABLE public.chunks_fts
      ADD CONSTRAINT chunks_fts_source_id_fkey
      FOREIGN KEY (source_id) REFERENCES public.sources(id)
      ON DELETE SET NULL;
  END IF;
END$$;

CREATE INDEX IF NOT EXISTS idx_chunks_fts_source_id
  ON public.chunks_fts(source_id);
