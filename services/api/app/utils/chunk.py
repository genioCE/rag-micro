import os
from typing import List

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "120"))

def chunk_texts(text: str) -> List[str]:
    text = (text or "").strip().replace("\x00", "")
    if not text:
        return []
    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(n, start + CHUNK_SIZE)
        chunk = text[start:end]
        chunks.append(chunk)
        start += CHUNK_SIZE - CHUNK_OVERLAP if CHUNK_SIZE > CHUNK_OVERLAP else CHUNK_SIZE
    return chunks
