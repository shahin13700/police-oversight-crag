"""
src/retrieval/bm25_retriever.py
--------------------------------
Implements a BM25 keyword-based retriever over the CSPA legislation chunks.

WHY BM25 IN ADDITION TO VECTOR SEARCH:
Vector search (ChromaDB) excels at semantic similarity — finding chunks that
mean the same thing even with different words. But it struggles with exact
matches for legal identifiers like section numbers ("s.11(1)"), specific
legal terms ("adequate and effective policing"), or acronyms ("OPP", "SIU").

BM25 is a classic keyword ranking algorithm that scores documents based on
term frequency and inverse document frequency. It excels at exactly the cases
where vector search struggles — precise legal terminology and exact phrases.

By combining both (hybrid search), we get the best of both worlds:
- Vector search finds semantically related sections
- BM25 finds sections with exact keyword matches
- RRF fusion (in hybrid_retriever.py) merges and re-ranks the results

WHY WE REBUILD THE INDEX IN MEMORY:
Unlike ChromaDB, BM25 doesn't need to persist to disk. The index rebuilds
from the chunk list in under 1 second, so we rebuild it fresh each time
the app starts. This avoids cache invalidation issues if chunks change.

Branch: feature/hybrid-retrieval-rrf
Issue:  #6 — BM25 index builder
"""

import re
import logging
from rank_bm25 import BM25Okapi

from src.ingestion.chunker import LegislativeChunk

logger = logging.getLogger(__name__)


def tokenize(text: str) -> list[str]:
    """
    Tokenize text into lowercase words for BM25 indexing.

    We use a simple but effective tokenization strategy:
    1. Lowercase everything for case-insensitive matching
    2. Split on whitespace and punctuation
    3. Filter out empty tokens

    We intentionally keep short tokens like "s" and numbers because
    legal text contains meaningful short terms like section references
    ("s.11", "OPP", "SIU") that we don't want to discard.

    Args:
        text: Raw text string to tokenize.

    Returns:
        List of lowercase token strings.

    Example:
        >>> tokenize("Section 11(1) — Adequate and effective policing")
        ['section', '11', '1', 'adequate', 'and', 'effective', 'policing']
    """
    # Replace punctuation with spaces, then split on whitespace
    # This handles legal text patterns like "11(1)", "s.11", "OPP/SIU"
    cleaned = re.sub(r'[^\w\s]', ' ', text.lower())
    tokens = [t for t in cleaned.split() if t]
    return tokens


class BM25Retriever:
    """
    BM25-based keyword retriever for legislative chunks.

    Build the index once with build(), then call search() as many times
    as needed. The index lives in memory — no disk persistence required.

    Usage:
        retriever = BM25Retriever()
        retriever.build(chunks)
        results = retriever.search("duties of chief of police", top_k=10)
    """

    def __init__(self) -> None:
        """Initialise an empty retriever. Call build() before search()."""
        self._bm25: BM25Okapi | None = None
        self._chunks: list[LegislativeChunk] = []
        self._is_built: bool = False

    def build(self, chunks: list[LegislativeChunk]) -> None:
        """
        Build the BM25 index from a list of LegislativeChunks.

        Tokenizes each chunk's text and builds a BM25Okapi index.
        BM25Okapi is the standard Okapi BM25 variant — it adds a
        normalisation factor for document length, which is important
        here because CSPA sections vary widely in length (6 to 2787 words).

        Args:
            chunks: List of LegislativeChunk objects to index.
                    Typically the full list of 1260 CSPA chunks.

        Raises:
            ValueError: If chunks list is empty.

        Example:
            >>> retriever = BM25Retriever()
            >>> retriever.build(chunks)
            >>> print(retriever.is_built)  # True
        """
        if not chunks:
            raise ValueError(
                "BM25Retriever.build() received an empty list. "
                "Pass the full list of LegislativeChunk objects."
            )

        self._chunks = chunks

        # Tokenize each chunk's text for BM25 indexing.
        # We index the full chunk text (including the section title as the
        # first line) so that title keywords also contribute to BM25 scores.
        tokenized_corpus = [tokenize(chunk.text) for chunk in chunks]

        # Build the BM25 index.
        # BM25Okapi takes a list of token lists (the corpus).
        self._bm25 = BM25Okapi(tokenized_corpus)
        self._is_built = True

        print(f"[bm25] Index built over {len(chunks)} chunks.")
        logger.info(f"[bm25] Index built over {len(chunks)} chunks.")

    def search(self, query: str, top_k: int = 10) -> list[dict]:
        """
        Search the BM25 index and return the top-k ranked chunks.

        Tokenizes the query the same way as the corpus, then uses BM25
        to rank all chunks by relevance. Returns the top-k as dicts
        with the same format as ChromaDB query results so the hybrid
        retriever can merge them without special casing.

        Args:
            query:  The user's question or search query as a plain string.
            top_k:  Number of top results to return. Defaults to 10.

        Returns:
            List of dicts, each representing one result:
            {
                "id":             chunk ID (source_doc__section__title),
                "text":           full text of the chunk,
                "score":          BM25 relevance score (higher = more relevant),
                "rank":           1-based rank position,
                "section_number": e.g. "11(1)",
                "section_title":  e.g. "Adequate and effective policing",
                "part_name":      e.g. "PART III — PROVISION OF POLICING",
                "source_doc":     e.g. "CSPA_2019",
                "citation":       e.g. "CSPA s.11(1)",
            }

        Raises:
            RuntimeError: If search() is called before build().

        Example:
            >>> results = retriever.search("duties of chief of police", top_k=5)
            >>> for r in results:
            ...     print(r["citation"], r["score"])
        """
        if not self._is_built:
            raise RuntimeError(
                "BM25Retriever.search() called before build(). "
                "Call retriever.build(chunks) first."
            )

        # Tokenize the query the same way we tokenized the corpus.
        # Using the same tokenizer for both is critical — mismatched
        # tokenization leads to poor retrieval quality.
        query_tokens = tokenize(query)

        # Get BM25 scores for all chunks in the corpus.
        # scores is a numpy array of length len(chunks), one score per chunk.
        scores = self._bm25.get_scores(query_tokens)

        # Get the indices of the top-k highest scores.
        # argsort() sorts ascending, so we reverse with [::-1] and take top_k.
        top_k_clamped = min(top_k, len(self._chunks))
        top_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True
        )[:top_k_clamped]

        # Build the result list in the same format as ChromaDB results.
        # Consistent format means the hybrid retriever can treat both
        # retrievers identically when doing RRF fusion.
        results = []
        for rank, idx in enumerate(top_indices, start=1):
            chunk = self._chunks[idx]
            title_slug = chunk.section_title[:50].replace(" ", "_").replace("/", "-")
            section_part = chunk.section_number if chunk.section_number else "no_section"
            chunk_id = f"{chunk.source_doc}__{section_part}__{title_slug}__{idx}"

            results.append({
                "id": chunk_id,
                "text": chunk.text,
                "score": float(scores[idx]),
                "rank": rank,
                "section_number": chunk.section_number,
                "section_title": chunk.section_title,
                "part_name": chunk.part_name,
                "source_doc": chunk.source_doc,
                "citation": chunk.citation(),
            })

        return results

    @property
    def is_built(self) -> bool:
        """Return True if the index has been built and is ready to search."""
        return self._is_built

    @property
    def corpus_size(self) -> int:
        """Return the number of chunks in the index."""
        return len(self._chunks)
