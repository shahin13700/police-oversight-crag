"""
src/retrieval/hybrid_retriever.py
----------------------------------
Combines ChromaDB vector search and BM25 keyword search using
Reciprocal Rank Fusion (RRF) to produce a single ranked result list.

WHY HYBRID SEARCH:
Neither vector search nor BM25 alone is sufficient for legal RAG:

- Vector search finds semantically similar chunks but misses exact legal
  identifiers like section numbers ("s.11(1)"), acronyms ("OPP", "SIU"),
  and precise legal phrases.

- BM25 finds exact keyword matches but misses semantically related content
  when different words are used (e.g. "duties" vs "responsibilities").

Hybrid search gives us both: semantic understanding AND exact matching.
For a QA system over police oversight legislation, this is critical.

WHY RRF (RECIPROCAL RANK FUSION):
The scores from vector search (cosine similarity, range ~0-1) and BM25
(term frequency scores, unbounded) are on completely different scales.
We CANNOT simply add them together.

RRF uses ranks instead of scores, making it scale-agnostic:
    RRF_score(document) = Σ 1 / (k + rank(document, retriever))

Where k=60 is a constant that dampens the impact of very high ranks.
A document ranked #1 by both retrievers gets the highest RRF score.
A document only found by one retriever still gets a partial score.

This is the standard approach for hybrid search in production RAG systems.

Branch: feature/hybrid-retrieval-rrf
Issue:  #7 — Hybrid retrieval with RRF fusion
"""

import os
import logging
from dotenv import load_dotenv

from src.ingestion.chunker import LegislativeChunk
from src.retrieval.bm25_retriever import BM25Retriever
from src.vectorstore.store import query as vector_query
from src.embeddings.embedder import embed_query

load_dotenv()
logger = logging.getLogger(__name__)

# RRF constant k=60 is the standard value from the original RRF paper.
# Higher k reduces the impact of rank differences at the top of the list.
# Lower k makes the top ranks matter more. 60 works well in practice.
RRF_K = 60


def reciprocal_rank_fusion(
    vector_results: list[dict],
    bm25_results: list[dict],
    top_k: int = 5,
) -> list[dict]:
    """
    Merge two ranked result lists using Reciprocal Rank Fusion (RRF).

    RRF assigns each document a score based on its rank in each list:
        score(doc) = Σ 1 / (k + rank(doc))

    Documents appearing in both lists get contributions from both,
    making them rank higher in the fused list. Documents only in one
    list still appear but with a lower score.

    Args:
        vector_results: Ranked list of dicts from ChromaDB vector search.
                        Each dict must have an "id" key.
        bm25_results:   Ranked list of dicts from BM25 search.
                        Each dict must have an "id" key.
        top_k:          Number of results to return after fusion.

    Returns:
        List of top_k result dicts, sorted by RRF score descending.
        Each dict has an added "rrf_score" field.

    Example:
        >>> fused = reciprocal_rank_fusion(vec_results, bm25_results, top_k=5)
        >>> for r in fused:
        ...     print(r["citation"], r["rrf_score"])
    """
    # Accumulate RRF scores in a dict keyed by chunk ID
    rrf_scores: dict[str, float] = {}

    # Also store the full result dict so we can return metadata
    # We prefer vector result metadata when a chunk appears in both lists
    result_store: dict[str, dict] = {}

    # Process vector search results
    for rank, result in enumerate(vector_results, start=1):
        chunk_id = result["id"]
        rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0) + 1 / (RRF_K + rank)
        result_store[chunk_id] = result

    # Process BM25 results
    for rank, result in enumerate(bm25_results, start=1):
        chunk_id = result["id"]
        rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0) + 1 / (RRF_K + rank)
        # Only store if not already stored from vector results
        if chunk_id not in result_store:
            result_store[chunk_id] = result

    # Sort by RRF score descending and take top_k
    sorted_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
    top_ids = sorted_ids[:top_k]

    # Build the final result list with RRF scores added
    fused_results = []
    for chunk_id in top_ids:
        result = result_store[chunk_id].copy()
        result["rrf_score"] = rrf_scores[chunk_id]
        fused_results.append(result)

    return fused_results


class HybridRetriever:
    """
    Combines ChromaDB vector search and BM25 using RRF fusion.

    This is the main retrieval component used by the LangGraph agent.
    It must be initialised with the full list of chunks so BM25 can
    build its in-memory index.

    Usage:
        retriever = HybridRetriever(chunks)
        results = retriever.retrieve("duties of chief of police", top_k=5)
    """

    def __init__(self, chunks: list[LegislativeChunk]) -> None:
        """
        Initialise the hybrid retriever and build the BM25 index.

        Args:
            chunks: Full list of LegislativeChunk objects — typically the
                    1260 chunks from the CSPA document. BM25 builds its
                    index from these in memory.
        """
        # Read retrieval config from .env
        self._top_k_vector = int(os.getenv("TOP_K_VECTOR", "10"))
        self._top_k_bm25 = int(os.getenv("TOP_K_BM25", "10"))

        # Build the BM25 index from the chunks
        self._bm25 = BM25Retriever()
        self._bm25.build(chunks)

        print(f"[hybrid] Retriever ready. "
              f"BM25 corpus: {self._bm25.corpus_size} chunks, "
              f"Vector top-k: {self._top_k_vector}, "
              f"BM25 top-k: {self._top_k_bm25}")

    def retrieve(self, query: str, top_k: int = 5) -> list[dict]:
        """
        Retrieve the most relevant legislative chunks for a query.

        Runs vector search and BM25 in parallel, then fuses the results
        using RRF to produce a single ranked list.

        Args:
            query:  The user's question as a plain string.
            top_k:  Number of final results to return after RRF fusion.
                    This is TOP_K_FINAL from .env, defaulting to 5.

        Returns:
            List of top_k result dicts, each with:
            {
                "id", "text", "rrf_score",
                "section_number", "section_title",
                "part_name", "source_doc", "citation"
            }

        Example:
            >>> results = retriever.retrieve("duties of chief of police")
            >>> for r in results:
            ...     print(r["citation"], r["rrf_score"])
        """
        # Step 1: Embed the query for vector search
        query_vector = embed_query(query)

        # Step 2: Run vector search against ChromaDB
        vector_results = vector_query(query_vector, top_k=self._top_k_vector)

        # Step 3: Run BM25 keyword search
        bm25_results = self._bm25.search(query, top_k=self._top_k_bm25)

        # Step 4: Fuse results using RRF
        fused = reciprocal_rank_fusion(vector_results, bm25_results, top_k=top_k)

        logger.info(
            f"[hybrid] Query: '{query[:60]}' → "
            f"vector: {len(vector_results)}, "
            f"bm25: {len(bm25_results)}, "
            f"fused: {len(fused)}"
        )

        return fused
