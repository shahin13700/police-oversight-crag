"""
src/agent/nodes/retriever.py
-----------------------------
The Retriever node runs hybrid search (ChromaDB + BM25 + RRF) to find
the most relevant CSPA legislative sections for the user's question.

"""

import os
import logging
from src.agent.state import AgentState
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.confidence import compute_confidence

logger = logging.getLogger(__name__)

# Module-level retriever singleton — built once, reused across queries
# Building the BM25 index takes ~1 second for ~1,800 chunks across CSPA, Regulations, and LECA, so we don't
# want to rebuild it on every query
_retriever: HybridRetriever | None = None


def get_retriever(chunks) -> HybridRetriever:
    """Return the singleton HybridRetriever, building it if necessary."""
    global _retriever
    if _retriever is None:
        _retriever = HybridRetriever(chunks)
    return _retriever


def retrieve_chunks(state: AgentState, chunks) -> AgentState:
    """
    Run hybrid retrieval and store results in state.

    Args:
        state:  Current AgentState with the user's question.
        chunks: Full list of LegislativeChunks for BM25 indexing.

    Returns:
        Updated AgentState with "retrieved_chunks" populated.
    """
    question = state["question"]
    top_k_final = int(os.getenv("TOP_K_FINAL", "5"))

    logger.info(f"[retriever] Retrieving chunks for: '{question[:60]}'")

    retriever = get_retriever(chunks)
    results = retriever.retrieve(question, top_k=top_k_final)
    scores = [doc.get("rrf_score", 0.0) if isinstance(doc, dict) else getattr(doc, 'rrf_score', 0.0) 
              for doc in results]


    confidence_label, confidence_score = compute_confidence(scores)

    logger.info(f"[retriever] Retrieved {len(results)} chunks.")
    logger.info(f"[retriever] Confidence: {confidence_label} ({confidence_score:.2f})")


    return {
        "retrieved_chunks": results,
        "confidence_label": confidence_label,
        "confidence_score": confidence_score,
    }
