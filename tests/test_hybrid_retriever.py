"""
tests/test_hybrid_retriever.py
--------------------------------
Unit tests for src/retrieval/hybrid_retriever.py.

We test:
1. reciprocal_rank_fusion() correctly merges two result lists
2. RRF scores are higher for docs appearing in both lists
3. HybridRetriever.retrieve() returns correct number of results
4. Results have all required metadata fields
5. RRF scores are present in fused results

Note: HybridRetriever tests require ChromaDB to be indexed.
We use the isolated_chroma fixture to create a fresh test collection.

Branch: feature/hybrid-retrieval-rrf
Issue:  #7 — Hybrid retrieval with RRF fusion
"""

import os
import pytest
from unittest.mock import patch

from src.ingestion.chunker import LegislativeChunk
from src.retrieval.hybrid_retriever import reciprocal_rank_fusion, HybridRetriever


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_mock_chunks(n: int = 5) -> list[LegislativeChunk]:
    """Create mock chunks for testing."""
    return [
        LegislativeChunk(
            text=f"Section {i} about policing duties and responsibilities in Ontario.",
            section_number=str(i),
            section_title=f"Test Section {i}",
            part_name="PART I — TEST",
            source_doc="TEST_DOC",
        )
        for i in range(1, n + 1)
    ]


def make_mock_result(chunk_id: str, rank: int) -> dict:
    """Create a mock result dict as returned by vector or BM25 search."""
    return {
        "id": chunk_id,
        "text": f"Text for {chunk_id}",
        "score": 1.0 / rank,
        "section_number": str(rank),
        "section_title": f"Section {rank}",
        "part_name": "PART I",
        "source_doc": "TEST_DOC",
        "citation": f"CSPA s.{rank}",
    }


@pytest.fixture(autouse=True)
def isolated_chroma(tmp_path):
    """Redirect ChromaDB to a temp directory for each test."""
    import src.vectorstore.store as store_module
    store_module._client = None
    store_module._collection = None

    def mock_embed(texts):
        return [[(float(i + 1) % 50.0) / 100.0] * 1536 for i, _ in enumerate(texts)]

    def mock_embed_query(text):
        return [0.01] * 1536

    with patch.dict(os.environ, {
        "CHROMA_PERSIST_DIR": str(tmp_path / "test_chroma"),
        "CHROMA_COLLECTION_NAME": "test_hybrid",
        "TOP_K_VECTOR": "5",
        "TOP_K_BM25": "5",
    }), patch("src.vectorstore.store.embed", side_effect=mock_embed), patch("src.retrieval.hybrid_retriever.embed_query", side_effect=mock_embed_query), patch("src.embeddings.embedder.embed_query", side_effect=mock_embed_query):
        yield

    store_module._client = None
    store_module._collection = None


# ---------------------------------------------------------------------------
# Tests for reciprocal_rank_fusion()
# ---------------------------------------------------------------------------

class TestReciprocalRankFusion:
    """Tests for the RRF fusion function."""

    def test_returns_correct_number_of_results(self):
        """RRF should return exactly top_k results."""
        vec = [make_mock_result(f"doc_{i}", i) for i in range(1, 6)]
        bm25 = [make_mock_result(f"doc_{i}", i) for i in range(1, 6)]
        fused = reciprocal_rank_fusion(vec, bm25, top_k=3)
        assert len(fused) == 3

    def test_doc_in_both_lists_scores_higher(self):
        """
        A document appearing in both lists should score higher than
        one appearing in only one list. This is the core RRF property.
        """
        # doc_1 appears in both lists at rank 1
        # doc_unique appears only in vector results at rank 2
        vec = [
            make_mock_result("doc_1", 1),
            make_mock_result("doc_unique", 2),
        ]
        bm25 = [
            make_mock_result("doc_1", 1),
            make_mock_result("doc_bm25_only", 2),
        ]
        fused = reciprocal_rank_fusion(vec, bm25, top_k=3)

        # doc_1 should be ranked first
        assert fused[0]["id"] == "doc_1"

    def test_rrf_scores_added_to_results(self):
        """Each fused result should have an rrf_score field."""
        vec = [make_mock_result(f"doc_{i}", i) for i in range(1, 4)]
        bm25 = [make_mock_result(f"doc_{i}", i) for i in range(1, 4)]
        fused = reciprocal_rank_fusion(vec, bm25, top_k=3)
        for r in fused:
            assert "rrf_score" in r
            assert r["rrf_score"] > 0

    def test_handles_disjoint_lists(self):
        """RRF should work when the two lists have no documents in common."""
        vec = [make_mock_result("vec_doc_1", 1), make_mock_result("vec_doc_2", 2)]
        bm25 = [make_mock_result("bm25_doc_1", 1), make_mock_result("bm25_doc_2", 2)]
        fused = reciprocal_rank_fusion(vec, bm25, top_k=4)
        assert len(fused) == 4

    def test_handles_empty_bm25_results(self):
        """RRF should work if BM25 returns no results."""
        vec = [make_mock_result(f"doc_{i}", i) for i in range(1, 4)]
        fused = reciprocal_rank_fusion(vec, [], top_k=3)
        assert len(fused) == 3

    def test_results_sorted_by_rrf_score(self):
        """Fused results should be sorted by rrf_score descending."""
        vec = [make_mock_result(f"doc_{i}", i) for i in range(1, 6)]
        bm25 = [make_mock_result(f"doc_{i}", i) for i in range(1, 6)]
        fused = reciprocal_rank_fusion(vec, bm25, top_k=5)
        scores = [r["rrf_score"] for r in fused]
        assert scores == sorted(scores, reverse=True)


# ---------------------------------------------------------------------------
# Tests for HybridRetriever
# ---------------------------------------------------------------------------

class TestHybridRetriever:
    """Tests for the HybridRetriever class."""

    def setup_indexed_collection(self, chunks: list[LegislativeChunk]) -> None:
        """Helper to index chunks into ChromaDB for testing."""
        from src.vectorstore.store import index_chunks
        index_chunks(chunks)

    def test_retrieve_returns_correct_number_of_results(self):
        """retrieve() should return exactly top_k results."""
        chunks = make_mock_chunks(10)
        self.setup_indexed_collection(chunks)

        retriever = HybridRetriever(chunks)
        results = retriever.retrieve("policing duties", top_k=3)
        assert len(results) == 3

    def test_retrieve_results_have_required_fields(self):
        """Each result should have all required metadata fields."""
        chunks = make_mock_chunks(5)
        self.setup_indexed_collection(chunks)

        retriever = HybridRetriever(chunks)
        results = retriever.retrieve("policing", top_k=3)

        required_fields = {
            "id", "text", "rrf_score",
            "section_number", "section_title",
            "part_name", "source_doc", "citation"
        }
        for r in results:
            assert required_fields.issubset(set(r.keys()))

    def test_retrieve_rrf_scores_are_positive(self):
        """All RRF scores should be positive."""
        chunks = make_mock_chunks(5)
        self.setup_indexed_collection(chunks)

        retriever = HybridRetriever(chunks)
        results = retriever.retrieve("policing duties", top_k=5)
        for r in results:
            assert r["rrf_score"] > 0
