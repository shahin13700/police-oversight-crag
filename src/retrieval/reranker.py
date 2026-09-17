"""
src/retrieval/reranker.py
--------------------------
Reranks retrieved chunks using Cohere Rerank API (rerank-v4.0-fast).
One API call reranks all chunks simultaneously — faster than local CPU inference.

WHY COHERE RERANK:
Unlike a local cross-encoder which scores pairs one-by-one on CPU,
the Cohere Rerank API processes all candidates in a single call with
a model trained specifically for passage relevance ranking.

"""

import os
import logging
import cohere

logger = logging.getLogger(__name__)

# Module-level singleton — one client reused across all queries
_client: cohere.ClientV2 | None = None


def _get_client() -> cohere.ClientV2:
    """Return the singleton Cohere client, creating it on first call."""
    global _client
    if _client is None:
        _client = cohere.ClientV2(api_key=os.getenv("COHERE_API_KEY"))
    return _client


def rerank(question: str, chunks: list[dict], top_n: int = 5) -> list[dict]:
    """
    Rerank retrieved chunks using the Cohere Rerank API.

    Sends all (question, chunk_text) pairs in a single API call and
    returns the top_n chunks sorted by relevance score descending.

    Args:
        question: The user's question as a plain string.
        chunks:   List of chunk dicts from the hybrid retriever. Each dict
                  must have a "text" key.
        top_n:    Number of chunks to return after reranking.

    Returns:
        List of up to top_n chunk dicts, each with an added
        "rerank_score" field, sorted by that score descending.
    """
    if not chunks:
        return chunks

    client = _get_client()
    model = os.getenv("COHERE_RERANK_MODEL", "rerank-v4.0-fast")
    documents = [c["text"] for c in chunks]

    response = client.rerank(
        model=model,
        query=question,
        documents=documents,
        top_n=top_n,
    )

    reranked = []
    for result in response.results:
        chunk = chunks[result.index].copy()
        chunk["rerank_score"] = result.relevance_score
        reranked.append(chunk)

    logger.info(
        f"[reranker] {len(chunks)} → {len(reranked)} chunks. "
        f"Top score: {reranked[0]['rerank_score']:.4f}"
    )
    return reranked
