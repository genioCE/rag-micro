# rag-micro

**rag-micro** is a lightweight microservice implementation of a Retrieval-Augmented Generation (RAG) system.  
It combines semantic vector search with keyword search, deduplication, and optional reranking, then streams answers back from a local LLM (via Ollama).  

This project is designed to be **fast, modular, and easy to run locally** while remaining extensible for production use.

---

## Features

- 🔎 **Hybrid Retrieval**: Combines vector embeddings + full-text search (BM25 style).  
- 🧹 **Deduplication**: Removes near-identical chunks from results.  
- 🔄 **Optional Reranker**: Cross-encoder reranker support for improved result ordering.  
- 📑 **Citation Persistence**: Each streamed answer includes consistent document/page citations.  
- 📤 **Streaming Answers**: Uses Server-Sent Events (SSE) for real-time model responses.  
- 📂 **File Upload**: Upload and index PDF/Markdown/TXT documents via API or frontend.  
- 🧩 **Microservice Ready**: Clean separation of retrieval, embedding, and answer generation.  

---

## Architecture

```
rag-micro/
├── services/
│   ├── api/              # FastAPI backend
│   │   ├── routes/       # Query + ingest endpoints
│   │   ├── db.py         # Database/Index connections
│   │   ├── utils/        # Embedding + helper logic
│   │   └── ...
│   ├── worker/           # Async ingestion / indexing jobs (optional)
│   └── ...
├── cache/                # Preview files
├── docker-compose.yml    # Local dev stack
└── README.md
```

- **Database**: Postgres (metadata), Qdrant (vector store).  
- **Embeddings**: Ollama, or external APIs.  
- **LLM**: Ollama local models (e.g., `llama3`), streamed to client.  

---

## Quickstart

### Prerequisites
- Docker & Docker Compose  
- Ollama installed locally (with at least one model pulled, e.g. `llama3`)  

### Run Locally
```bash
git clone https://github.com/genioCE/rag-micro.git
cd rag-micro
docker compose up --build
```

The API will be available at [http://localhost:8000](http://localhost:8000).

---

## API Endpoints

### Health Check
```bash
GET /health
```

### Upload Documents
```bash
POST /ingest/upload
```
Form-data:  
- `files=@/path/to/document.pdf`

### Query
```bash
POST /query/
```
JSON body:
```json
{
  "q": "<choose a word>",
  "top_k": 5,
  "alpha": 0.6,
  "use_reranker": true
}
```

Returns:
- Retrieved chunks (doc/page/chunk info)  
- Streamed LLM answer with citations  

---

## Example Query (curl)
```bash
curl -s -X POST "http://localhost:8000/query/" \
  -H "Content-Type: application/json" \
  -d '{"q":"<use word chosen above>","top_k":3,"alpha":0.6}' | jq
```

---

## Development

- Python 3.11+  
- FastAPI, SQLAlchemy, Qdrant, Ollama  
- For local dev: use `uvicorn` to run API service  

---

## Roadmap

- [ ] Improve reranker integration (bge-reranker)  
- [ ] Frontend preview + conversation interface  
- [ ] Multi-user doc collections  
- [ ] Production deployment (Kubernetes, GPU-enabled)  

---

## License

MIT License. See [LICENSE](./LICENSE) for details.  
