"""
src/agent/nodes/reranker.py
----------------------------
The Reranker node refines the list of retrieved chunks using a cross-encoder
before the Grader sees them.

WHY A RERANKER NODE:
Hybrid retrieval (BM25 + ChromaDB + RRF) optimises for recall — it retrieves
a wider set of candidates than we ultimately need. The cross-encoder acts as a
precision filter: it reads each (question, chunk) pair jointly and produces a
score far more reliable than any RRF rank.

Inserting this node between Retriever and Grader means:
- The Grader sees only the most relevant chunks (fewer LLM calls)
- The Generator has better context
- The pipeline is more robust when the user's phrasing differs from the
  legislation's phrasing

POSITION IN GRAPH:
    Retriever → Reranker → Grader → ...

"""

import os
import logging
from src.agent.state import AgentState
from src.retrieval.reranker import rerank

logger = logging.getLogger(__name__)


def rerank_chunks(state: AgentState) -> dict:
    """
    Rerank retrieved chunks using the cross-encoder and overwrite
    state["retrieved_chunks"] with the reranked results.

    The Grader node downstream will see only the top-scored chunks, making
    its relevance decisions faster and more accurate.

    Args:
        state: Current AgentState with question and retrieved_chunks populated.

    Returns:
        Partial state update — {"retrieved_chunks": reranked_chunks}.
    """
    question = state["question"]
    chunks = state.get("retrieved_chunks", [])
    top_n = int(os.getenv("RERANKER_TOP_N", "5"))

    logger.info(
        f"[reranker_node] Reranking {len(chunks)} chunks "
        f"for: '{question[:60]}'"
    )

    if not chunks:
        logger.warning("[reranker_node] No chunks to rerank — skipping.")
        return {"retrieved_chunks": []}

    reranked = rerank(question, chunks, top_n=top_n)

    # Log the top chunk score for debugging
    if reranked:
        top_score = reranked[0].get("rerank_score", 0.0)
        top_cite = reranked[0].get("citation", "?")
        logger.info(
            f"[reranker_node] Top chunk after reranking: "
            f"{top_cite} (score={top_score:.4f})"
        )


    return {"retrieved_chunks": reranked}
