-- sources table for citation previews
CREATE TABLE IF NOT EXISTS sources (
  id TEXT PRIMARY KEY,                  -- citation id (e.g. "c1")
  url TEXT NOT NULL UNIQUE,
  kind TEXT NOT NULL CHECK (kind IN ('html','pdf','text')),
  locator JSONB NOT NULL,               -- {"type":"pdf","page":37,"bbox":[...]} or {"type":"text","start":123,"end":456}
  title TEXT,
  raw_text TEXT,                        -- cached for html/text
  pdf_blob_path TEXT,                   -- cache location if you persist PDFs
  raw_fetched_at TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_sources_url ON sources(url);
