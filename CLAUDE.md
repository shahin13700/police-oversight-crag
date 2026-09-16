# Police Oversight CRAG — Developer Cheat Sheet

Corrective RAG (CRAG) pipeline for Ontario police oversight legislation (CSPA 2019, Regulations, LECA guidelines).

---

## Quick Commands

### Setup & Ingestion
```bash
python -m venv .venv
source .venv/bin/activate        # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
cp .env.example .env             # Configure GROQ_API_KEY, OPENROUTER_API_KEY, COHERE_API_KEY
python scripts/index_all.py      # Chunk and index statutes into ChromaDB
```

### Running Locally
```bash
# Option A: Single terminal local UI (direct pipeline)
streamlit run src/ui/app.py

# Option B: Two-service microservice (API + UI)
uvicorn src.api.main:app --reload --port 8000
streamlit run src/ui/app_prod.py

# Option C: Docker Compose
docker compose up --build
```

### Testing & Quality
```bash
# Run test suite offline (mocked API calls)
pytest tests/

# Run fast syntax and undefined-name checks
ruff check --select E9,F63,F7,F82 .

# Run LLM-as-a-Judge benchmark
python evaluation/run_eval.py
```

---

## Architecture Summary
- **CRAG Workflow:** `Router` -> `Hybrid Retriever` (BM25 + ChromaDB Cosine via RRF $k=60$) -> `Cohere Reranker` -> `Relevance Grader` -> `Query Rewriter` (max 2 retries) -> `Generator` (citation enforcement).
- **Core Models:**
  - Embeddings: `openai/text-embedding-3-small` (1536-d) via OpenRouter
  - LLM Nodes: `llama-3.3-70b-versatile` via Groq API
  - Reranker: `rerank-v4.0-fast` via Cohere API
  - Vectorstore: ChromaDB (`ontario_legislation` collection)