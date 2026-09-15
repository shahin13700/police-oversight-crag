# Ontario Oversight CRAG — System Architecture & Technical Reference

**Project**: Legal AI Engineering Portfolio  
**Domain**: Ontario Police Oversight Legislation (CSPA 2019 & Regulations)  
**Purpose**: Agentic Corrective RAG (CRAG) pipeline for answering compliance and QA questions about Ontario police oversight legislation with strict, verifiable section citations.

This document is the technical architecture reference for developers and reviewers. It mirrors the layout of `src/` and explains how ingestion, retrieval, the LangGraph agent, API, and UI fit together.

### Table of contents

1. [Problem statement](#problem-statement)
2. [High-level architecture](#high-level-architecture)
3. [Agent nodes](#agent-nodes)
4. [Ingestion pipeline](#1-ingestion-pipeline)
5. [Embeddings](#2-embeddings)
6. [Vector store](#3-vector-store)
7. [Retrieval pipeline](#4-retrieval-pipeline)
8. [Agent state](#5-agent-state)
9. [LangGraph wiring](#6-langgraph-graph-wiring)
10. [LLM prompts](#7-llm-prompts)
11. [API layer](#8-api-layer)
12. [Evaluation](#9-evaluation)
13. [Deployment](#10-deployment-architecture)
14. [Models and APIs](#11-models--apis-summary)
15. [Technology stack](#12-technology-stack)
16. [Design decisions](#13-key-design-decisions)
17. [UI features](#14-ui-features)
18. [File reference](#15-file-reference)
19. [NotebookLM slide prompt](#notebooklm--prompt-for-slide-deck-generation)

---

## Problem Statement

Legal compliance and oversight QA professionals need fast, accurate answers grounded in Ontario police oversight legislation (CSPA, O.Regs, LECA). Manual lookup across hundreds of pages of legislation is slow and error-prone. Ontario Oversight CRAG automates this with an agentic AI pipeline that always cites the specific legislative section behind every answer.

---

## High-Level Architecture

```
User Question
      │
      ▼
┌─────────────┐
│   Router    │  Classifies: needs retrieval? or direct answer?
└─────────────┘
      │ retrieval
      ▼
┌─────────────┐
│  Retriever  │  Hybrid BM25 + ChromaDB (each top-10) → RRF → top_k fused (default 5)
└─────────────┘
      │
      ▼
┌─────────────┐
│  Reranker   │  Cohere Rerank API → top 5 chunks scored for relevance
└─────────────┘
      │
      ▼
┌─────────────┐
│   Grader    │  LLM judges each chunk: relevant or not?
└─────────────┘
      │ fail (max 2x)        │ pass
      ▼                      ▼
┌─────────────┐        ┌─────────────┐
│  Rewriter   │──────▶ │  Generator  │  Produces cited answer
└─────────────┘        └─────────────┘
                              │
                              ▼
                     Answer + Citations + Confidence
```

**Pattern**: Corrective RAG (CRAG) implemented with LangGraph  
**Rewrite loop**: Max 2 retries. Each retry goes through Retriever → Reranker → Grader again.

---

## Agent Nodes

| Node | Role | LLM / Service |
|---|---|---|
| **Router** | Decides retrieval vs. direct answer | Groq llama-3.3-70b (temp=0) |
| **Retriever** | Hybrid BM25 + vector search, RRF fusion | — |
| **Reranker** | Scores (question, chunk) relevance | Cohere rerank-v4.0-fast |
| **Grader** | Judges chunk relevance, pass/retry decision | Groq llama-3.3-70b (temp=0) |
| **Rewriter** | Rewrites query using formal legislative language | Groq llama-3.3-70b (temp=0.3) |
| **Generator** | Produces cited plain-language answer | Groq llama-3.3-70b (temp=0.1) |

---

## 1. Ingestion Pipeline

### 1.1 Master Pipeline (`src/ingestion/pipeline.py`)

`run_ingestion()` loads all three corpora in sequence:

1. `data/raw/CSPA_2019.docx` → `chunk_document()` → CSPA chunks
2. `data/raw/Regulations/*.docx` → `chunk_reg()` per file → O.Regs chunks
3. `data/raw/pdfs/*.pdf` → `chunk_pdf()` per file → LECA chunks
4. All lists merged → single `list[LegislativeChunk]` returned

### 1.2 Data Model

Every chunk is a `LegislativeChunk` dataclass:

```python
@dataclass
class LegislativeChunk:
    text: str            # Full section text
    section_number: str  # e.g. "11", "11(1)", "11(1)(a)"
    section_title: str   # e.g. "Adequate and effective policing"
    part_name: str       # e.g. "PART III — PROVISION OF POLICING"
    source_doc: str      # e.g. "CSPA_2019", "O. Reg. 392/23", "LECA Rules"
```

`chunk.citation()` returns e.g. `"CSPA s.11(1)"` — used in every LLM prompt and displayed in UI.

### 1.3 CSPA Chunker (`src/ingestion/chunker.py`)

Single-pass through all Word paragraphs, tracking styles:

| Style | Action |
|---|---|
| `partnum` | Start new Part (e.g. "PART I") |
| `headnote` | Start new chunk, set section title |
| `section` | Extract section number via regex `r'^(\d+)\s*(\(\d+\))?'` |
| `subsection` | Inherit parent section number |
| `paragraph` | Accumulate as sub-items (a), (b), (c) |
| `definition` | Accumulate as definitions |
| Skipped styles | `shorttitle`, `chapter`, `toc`, `Normal`, `footnoteLeft` |

Chunk saved only if `> 5 words` and non-empty title.

### 1.4 Ontario Regulations Chunker (`src/ingestion/reg_chunker.py`)

- Searches first 20 paragraphs for regex `r'ONTARIO REGULATION\s+([0-9]+/[0-9]+)'` to extract regulation number
- Heading detection via `check_bold_heading()`: True if style contains "Heading"/"Headnote"/"RegTitle", OR text is all-uppercase with length > 3
- Section number extracted with two fallback patterns: `r'^(\d+(?:\.\d+)?)\s*\.'` or `r'^(\d+)\s+'`
- Chunk saved if `> 3 words`

### 1.5 LECA PDF Chunker (`src/ingestion/pdf_chunker.py`)

- Pages extracted by pdfplumber, concatenated, split by line
- Heading detection heuristic:
  - Line is all-uppercase AND length < 100, OR
  - Line doesn't end with `.`/`:` AND length 2–100 AND word count < 10 AND all words (> 3 chars) start uppercase
- `section_number` is always empty for PDFs — sections named by heading only
- Document name extracted from filename (e.g. `"LECA Guideline 3 — Use of Force"`)

---

## 2. Embeddings

**File**: `src/embeddings/embedder.py`

| Parameter | Value |
|---|---|
| Model | `openai/text-embedding-3-small` (via OpenRouter) |
| Dimension | 1536d |
| Batch size | 20 texts per API call |
| Retry logic | 3 attempts, exponential backoff (1s, 2s, 4s) |
| Rate limiting | 0.5s sleep after each successful batch |

**Key functions:**
- `embed(texts: list[str]) -> list[list[float]]` — batch embedding
- `embed_query(query: str) -> list[float]` — single query embedding

Response format is OpenAI-compatible: `{"data": [{"embedding": [...], "index": i}]}`  
Results sorted by index before extraction.

> **Known issue**: `get_embedding_dimension()` hardcoded to return `1536`. Comment added to warn future devs — update if switching back to bge-m3 (1024d).

---

## 3. Vector Store

**File**: `src/vectorstore/store.py`

| Parameter | Value |
|---|---|
| Database | ChromaDB (persistent, disk-backed) |
| Directory | `CHROMA_PERSIST_DIR` env var (default: `chroma_db/`) |
| Collection | `CHROMA_COLLECTION_NAME` env var (default: `ontario_legislation`) |
| Distance metric | Cosine similarity (`hnsw:space: cosine`) |

**Chunk ID format**: `"{source_doc}__{section_number}__{title_slug}__{index}"`

**Metadata stored per chunk**: `section_number`, `section_title`, `part_name`, `source_doc`, `citation`

**Score conversion**: `score = 1 - distance` (converts cosine distance to similarity)

**Startup optimization**: `index_chunks()` only runs if collection is empty. Subsequent app starts skip re-embedding.

---

## 4. Retrieval Pipeline

### Step 1 — BM25 Keyword Search (`src/retrieval/bm25_retriever.py`)

**Tokenizer**: lowercase → strip punctuation (`r'[^\w\s]' → ' '`) → split on whitespace  
Short tokens like "s", "OPP", "SIU" intentionally kept for legal relevance.

**Algorithm**: `BM25Okapi` (standard Okapi BM25 with document length normalization)  
Index built over all 1,868 chunks at startup.  
Returns top_k results with raw BM25 score and rank.

### Step 2 — Vector Search (`src/vectorstore/store.py`)

Query embedded via `embed_query()` → cosine similarity search in ChromaDB  
Returns top_k results with similarity score.

### Step 3 — RRF Fusion (`src/retrieval/hybrid_retriever.py`)

**Formula**:
```
RRF_score(doc) = Σ 1 / (k + rank(doc, retriever))
k = 60  (standard value from original RRF paper)
```

- BM25 results and vector results each assigned ranks 1..top_k
- If doc appears in both: scores summed (rewarded for appearing in both)
- If doc appears in one only: single contribution
- Merged results sorted by RRF score descending
- Top 5 passed to Reranker

**Why k=60?** Prevents top-ranked documents from dominating; smooths score distribution.

### Step 4 — Cohere Reranker (`src/retrieval/reranker.py`)

Single API call to Cohere `rerank-v4.0-fast`:
- Scores all (question, chunk) pairs simultaneously
- Returns top_n results with `relevance_score` (typically 0.70–0.95 for strong matches)
- Adds `rerank_score` field to each chunk dict
- Client is a module-level singleton — created once, reused for all queries

### Confidence Scoring (`src/retrieval/confidence.py`)

Computed from RRF top score (max theoretical ~0.033):

| Label | Threshold | UI |
|---|---|---|
| High | RRF ≥ 0.020 | 🟢 |
| Medium | RRF ≥ 0.010 | 🟡 |
| Low | RRF < 0.010 | 🔴 |

Displayed above every answer in the Streamlit UI.

---

## 5. Agent State

**File**: `src/agent/state.py`

```python
class AgentState(TypedDict):
    question: str                          # Original question (unchanged)
    retrieved_chunks: list[dict]           # Reranker overwrites this each retrieval
    relevance_passed: Optional[bool]       # Grader verdict (None → grader not run yet)
    rewrite_count: int                     # Rewrite attempts (0 initially, max 2)
    answer: Optional[str]                  # Final answer (None until generator)
    route: Optional[str]                   # "retrieval" or "conversational"
    confidence_label: Annotated[str, take_latest]   # "High"/"Medium"/"Low"
    confidence_score: Annotated[float, take_latest] # Top RRF score
    rrf_scores: list[float]               # All RRF scores from retrieval
```

`take_latest` merge function: returns updated value if not None, else keeps current.

---

## 6. LangGraph Graph Wiring

**File**: `src/agent/graph.py`

```
START → router
router → retriever          (if route == "retrieval")
router → generator          (if route == "conversational")
retriever → reranker        (fixed edge)
reranker → grader           (fixed edge)
grader → generator          (if relevance_passed == True)
grader → rewriter           (if relevance_passed == False)
rewriter → retriever        (retry — goes through reranker → grader again)
rewriter → generator        (if rewrite_count >= 2, forced pass)
generator → END
```

`chunks` bound to retriever node via `functools.partial()` at graph build time.

`run_query(graph, question)` initializes state and returns:
```python
{
    "answer": str,
    "retrieved_chunks": list[dict],
    "confidence": str,
    "confidence_score": float
}
```

---

## 7. LLM Prompts

### Router System Prompt
```
Classify question into ONE of:
1. "retrieval" — Question about Ontario police oversight, CSPA, duties, complaints, oversight bodies
2. "conversational" — Greeting, thank you, or does NOT require legislation lookup

Respond with ONLY the single word: retrieval or conversational
```
Default to "retrieval" if output unexpected (safer than missing a retrieval).

### Grader System Prompt
```
Given user question and legislative section, decide if section is relevant.
Respond with ONLY "yes" or "no".
```
Minimum 2 relevant chunks required to pass. Early exit after 2 found (performance optimization).  
Chunk truncated to first 500 chars per grader call.

### Rewriter System Prompt
```
Original query failed to retrieve relevant sections. Rewrite to:
1. Use formal, legislative language
2. Reference specific legal concepts (duties, obligations, powers)
3. Be more specific about which aspect of policing is asked

Return ONLY the rewritten query. No explanation.
```
Temperature 0.3 — slight creativity allowed for rephrasing.

### Generator System Prompt
```
You are an AI Quality Assurance and Legal Compliance assistant specialized in Ontario Police Oversight Legislation (CSPA 2019 & Regulations).
Answer based ONLY on provided legislative sections.

STRICT RULES:
1. Every claim MUST cite the specific section: e.g. "CSPA s.11(1)"
2. NEVER answer from own knowledge
3. If sections don't address question, say: "The provided sections do not address this question directly."
4. Format checklists as numbered lists
5. Keep answers professional

FORMAT:
- Start with direct answer
- Support with citations in brackets: [CSPA s.X(Y)]
- End with "Sources" section listing all cited sections
```
Temperature 0.1 — minimal randomness, fact-focused.

---

## 8. API Layer

**File**: `src/api/main.py`

### Startup lifecycle
1. `run_ingestion()` → load all chunks from CSPA, O.Regs, LECA
2. `get_collection()` → open ChromaDB
3. If collection empty → `index_chunks(chunks)` (embed + upsert all chunks)
4. `build_graph(chunks)` → compile LangGraph
5. Store in global `pipeline_state` dict

### Pydantic Models
```python
class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    answer: str
    sources: list[dict]
    route: str
    confidence: str = "Low"
    confidence_score: float = 0.0
```

### Endpoints

| Endpoint | Method | Purpose |
|---|---|---|
| `/health` | GET | Returns `{"status": "ok", "chunks": N}` — liveness check |
| `/query` | POST | Takes `QueryRequest`, returns `QueryResponse` with answer + sources + confidence |

---

## 9. Evaluation

**File**: `evaluation/run_eval.py`  
**Test set**: `evaluation/qa_pairs.json` — 12 QA pairs

### Categories
1. Foundational Oversight and QA (2 questions)
2. Inspectorate of Policing (2 questions)
3. LECA / Complaints and Professional Standards (2 questions)
4. Risk, Monitoring, and Continuous Improvement (2 questions)
5. QA Checklists (4 questions)

### Scoring

**Citation score (50% weight)**: Binary — 1.0 if any citation found matching regex, else 0.0  
Citation regex covers `CSPA s.X(Y)`, `O. Reg. NNN/NN s.X`, `LECA Guideline N`, `LECA Rules`

**Topic score (50% weight)**: `found_topics / expected_topics` — case-insensitive substring match

**Overall**: `(citation_score × 0.5) + (topic_score × 0.5) ≥ 0.5` → PASS

**LLM Judge (Qwen via OpenRouter)**: Optional deeper evaluation on:
- `faithfulness` — answer grounded in retrieved chunks?
- `relevance` — directly answers the question?
- `completeness` — covers expected topics?

**Phase 2 result**: 100% (12/12 passed)

Results saved to `evaluation/results_YYYY-MM-DD.json` with full per-question breakdown.

---

## 10. Deployment Architecture

### Local Development (one terminal)
```
streamlit run src/ui/app.py
```
Pipeline runs in-process. No backend server needed.

### Production / Azure (two containers)
```
Container 1 — FastAPI backend:
  uvicorn src.api.main:app --host 0.0.0.0 --port 8000

Container 2 — Streamlit frontend:
  streamlit run src/ui/app_prod.py --server.port 8501
  (env: API_URL=http://<fastapi-container-url>:8000)
```

**Azure gotcha**: ChromaDB is file-based — containers are ephemeral. Azure Blob Storage must be mounted to `chroma_db/` to persist the index across restarts. Without this, the pipeline re-indexes on every cold start.

---

## 11. Models & APIs Summary

| Component | Model / Service | Provider | Notes |
|---|---|---|---|
| LLM — all nodes | llama-3.3-70b-versatile | Groq | LPU hardware — ~10x faster than GPU inference |
| Embeddings | openai/text-embedding-3-small | OpenRouter | 1536d, temp swap (bge-m3 provider down) |
| Reranker | rerank-v4.0-fast | Cohere | Dec 2025 release, 32k context, cloud API |
| Eval Judge | qwen/qwen3.5-plus-02-15 | OpenRouter | Eval only — never used in prod pipeline |

### Why Groq?
Groq LPUs (Language Processing Units) are purpose-built inference chips. The pipeline makes 3–5 sequential LLM calls per query. Groq delivers sub-second inference per call (~300ms), keeping total pipeline latency under 5 seconds.

### Why Cohere Reranker?
- Cloud API — no GPU required on demo laptop (CPU-only)
- One API call scores all chunks simultaneously (faster than local sequential inference)
- `rerank-v4.0-fast` released Dec 2025: 32k token context, multilingual, purpose-built for relevance
- Free tier: 1,000 calls/month — sufficient for dev and demo

### Why Hybrid Retrieval?
Legal text contains specific identifiers (`s.11(1)`, `OPP`, `SIU`) that dense embeddings handle poorly. BM25 catches exact term matches; vector search catches semantic matches. RRF fusion gets the best of both without tuning weights.

---

## 12. Technology Stack

| Layer | Technology |
|---|---|
| Agent framework | LangGraph |
| LLM inference | Groq API (llama-3.3-70b-versatile) |
| Embedding API | OpenRouter (openai/text-embedding-3-small) |
| Reranking | Cohere Rerank API (rerank-v4.0-fast) |
| Vector store | ChromaDB (persistent, cosine similarity) |
| Keyword search | rank_bm25 (BM25Okapi) |
| Fusion algorithm | Reciprocal Rank Fusion (k=60) |
| Backend API | FastAPI |
| Frontend UI | Streamlit |
| Document parsing | python-docx, pdfplumber |
| Language | Python 3.11+ |
| Cloud | Azure Container Apps |

---

## 13. Key Design Decisions

**1. CRAG over naive RAG**  
Naive RAG always passes retrieved chunks to the generator — even irrelevant ones. CRAG adds a Grader that validates relevance and rewrites the query if needed. Critical for legal text where a wrong citation is worse than no answer.

**2. Hybrid retrieval over vector-only**  
Legal text contains specific section numbers (`s.11(1)`) and acronyms (`OPP`, `SIU`, `LECA`) that dense embeddings handle poorly. BM25 catches exact keyword matches; vector search catches semantic matches. RRF fusion without weight tuning.

**3. Cohere reranker between RRF and Grader**  
RRF ranks by retrieval score, not question relevance. The cross-encoder reranker scores each (question, chunk) pair directly — the Grader sees the most relevant chunks first, reducing false negatives.

**4. Groq for all LLM nodes**  
3–5 LLM calls per query (router + grader × N + generator). Speed is critical for a live demo and production UX. Groq LPUs deliver ~300ms per call vs ~2–3s on standard GPU inference.

**5. Citation enforcement in Generator**  
Every claim must cite a specific legislative section. System prompt is strict: never answer from own knowledge, always end with a Sources section. This is the #1 compliance requirement for legislative QA.

**6. Separate local vs. prod UI**  
`app.py` runs pipeline directly (one terminal, local dev). `app_prod.py` calls FastAPI over HTTP (two containers, Azure). Keeps deployment concerns out of the pipeline code entirely.

---

## 14. UI Features

### Local UI (`src/ui/app.py`) — one terminal
- Chat interface with dark/light theme toggle
- Confidence indicator — 🟢 High / 🟡 Medium / 🔴 Low above each answer
- Source expander — shows citation, section title, and RRF score for each retrieved chunk
- **Copy answer button** — native HTML clipboard button per answer; shows "Copied!" for 1.5s then resets
- **Export Session to Word** — sidebar button downloads full Q&A session as formatted `.docx`

### Production UI (`src/ui/app_prod.py`) — two terminals (FastAPI + Streamlit)
- Identical UI to app.py
- All pipeline calls replaced with HTTP requests to FastAPI backend
- `API_URL` env var points to backend (default: `http://localhost:8000`)

### Export Module (`src/ui/export.py`)

`build_session_docx(messages, sources) -> bytes`

- Iterates `st.session_state.messages` as user/assistant pairs
- Sources passed as separate dict keyed by message index (`st.session_state.sources`)
- Output includes: Report header, export timestamp, each Q&A numbered, confidence label, answer text, cited sources as bullet list
- Returns raw bytes for `st.download_button`
- No new dependencies — uses `python-docx` already in requirements

---

## 15. File Reference

| File | Purpose |
|---|---|
| `src/ingestion/pipeline.py` | `run_ingestion()` — master ingestion orchestrator |
| `src/ingestion/chunker.py` | CSPA .docx chunker, `LegislativeChunk` dataclass |
| `src/ingestion/reg_chunker.py` | Ontario Regulations .docx chunker |
| `src/ingestion/pdf_chunker.py` | LECA PDF chunker |
| `src/ingestion/loader.py` | `load_docx()`, `load_pdf()` |
| `src/embeddings/embedder.py` | OpenRouter embedding API, batch + retry logic |
| `src/vectorstore/store.py` | ChromaDB: `get_collection()`, `index_chunks()`, `query()` |
| `src/retrieval/bm25_retriever.py` | `BM25Retriever`, legal-aware tokenizer |
| `src/retrieval/hybrid_retriever.py` | `HybridRetriever`, `reciprocal_rank_fusion()` |
| `src/retrieval/confidence.py` | `compute_confidence()` — RRF score → label |
| `src/retrieval/reranker.py` | Cohere API singleton, `rerank()` |
| `src/agent/state.py` | `AgentState` TypedDict, `take_latest` merge |
| `src/agent/nodes/router.py` | `route_question()` |
| `src/agent/nodes/retriever.py` | `retrieve_chunks()`, HybridRetriever singleton |
| `src/agent/nodes/reranker.py` | `rerank_chunks()` LangGraph node |
| `src/agent/nodes/grader.py` | `grade_chunks()`, min 2 relevant threshold |
| `src/agent/nodes/rewriter.py` | `rewrite_query()`, max 2 rewrites |
| `src/agent/nodes/generator.py` | `generate_answer()`, citation enforcement |
| `src/agent/graph.py` | `build_graph()`, `run_query()`, all edges |
| `src/api/main.py` | FastAPI: `/health`, `/query`, startup lifecycle |
| `src/ui/app.py` | Streamlit local UI (direct pipeline, one terminal) |
| `src/ui/app_prod.py` | Streamlit prod UI (calls FastAPI over HTTP) |
| `src/ui/export.py` | `build_session_docx()` — exports Q&A session to Word |
| `evaluation/run_eval.py` | LLM-as-a-Judge eval harness |
| `evaluation/qa_pairs.json` | 12 ground-truth QA pairs |

---


When using NotebookLM, upload **this file** and **`README.md`** as sources; add `CONTRIBUTING.md` only if the deck should cover team process and branching.
