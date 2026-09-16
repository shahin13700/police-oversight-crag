"""
src/api/main.py
------------------
FastAPI backend for Ontario Oversight CRAG. Loads the NLP pipeline at startup and serves HTTP
so the UI can decouple its rendering loop from the heavy model memory footprints.
"""

import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from pydantic import BaseModel

from src.ingestion.pipeline import run_ingestion
from src.agent.graph import build_graph, run_query

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global state to hold the initialized pipeline components
pipeline_state = {
    "graph": None,
    "chunks": None
}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the entire legislative chunk corpus and compile LangGraph on boot
    logger.info("Initializing Ontario Oversight CRAG ingestion pipeline...")
    chunks = run_ingestion()
    from src.vectorstore.store import get_collection, index_chunks
    collection = get_collection()
    if collection.count() == 0:
        logger.info("ChromaDB collection empty — indexing chunks now...")
        index_chunks(chunks)
        logger.info(f"Indexed {len(chunks)} chunks into ChromaDB.")
    else:
        logger.info(f"ChromaDB collection already has {collection.count()} chunks — skipping re-index.")
    graph = build_graph(chunks)
    
    pipeline_state["chunks"] = chunks
    pipeline_state["graph"] = graph
    logger.info(f"Pipeline initialized successfully with {len(chunks)} chunks.")
    
    yield
    
    # Teardown logic
    logger.info("Shutting down Police Oversight CRAG API...")
    pipeline_state["chunks"] = None
    pipeline_state["graph"] = None


app = FastAPI(
    title="Police Oversight CRAG API",
    description="Backend inference and retrieval service for the Ontario Oversight CRAG UI.",
    lifespan=lifespan
)

class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    answer: str
    sources: list[dict]
    route: str
    confidence: str = "Low"
    confidence_score: float = 0.0


@app.get("/health")
def health_check():
    """Verify the backend is running and output the chunk count."""
    chunks = pipeline_state.get("chunks")
    count = len(chunks) if chunks else 0
    return {"status": "ok", "chunks": count}

@app.post("/query", response_model=QueryResponse)
def query_pipeline(request: QueryRequest):
    """
    Accepts string queries, runs them against the compiled LangGraph setup,
    and returns exact matches mapped to sources and their routing classification.
    """
    graph = pipeline_state.get("graph")
    if not graph:
        return QueryResponse(
            answer="Error: Pipeline is not initialized.", 
            sources=[], 
            route="error"
        )
        
    try:
        # run_query takes graph and string, and returns the state dict
        result = run_query(graph, request.question)
        
        answer = result.get("answer", "I could not generate an answer.")
        retrieved_chunks = result.get("retrieved_chunks", [])
        route = result.get("route", "unknown")
        
        # Pydantic validates and returns strictly as JSON
        return QueryResponse(
            answer=answer,
            sources=retrieved_chunks,
            route=route,
            confidence=result.get("confidence", "Low"),
            confidence_score=result.get("confidence_score", 0.0),
        )
        
    except Exception as e:
        logger.error(f"Error executing graph query: {e}")
        return QueryResponse(
            answer=f"Error generating answer: {str(e)}",
            sources=[],
            route="error"
        )
