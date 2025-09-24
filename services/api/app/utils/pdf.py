import pdfplumber
import fitz  # PyMuPDF
from typing import List, IO

def extract_text_by_pages(fileobj: IO[bytes]) -> List[str]:
    # Try pdfplumber
    pages_text = []
    try:
        with pdfplumber.open(fileobj) as pdf:
            for page in pdf.pages:
                pages_text.append(page.extract_text() or "")
    except Exception:
        pages_text = []

    if not pages_text or all(t.strip() == "" for t in pages_text):
        # fallback to OCR with PyMuPDF (for scanned PDFs you'd need Tesseract or PaddleOCR—placeholder here)
        fileobj.seek(0)
        doc = fitz.open(stream=fileobj.read(), filetype="pdf")
        pages_text = []
        for page in doc:
            text = page.get_text()
            pages_text.append(text or "")
    return pages_text
