"""
tests/test_store.py
-------------------
Unit tests for src/vectorstore/store.py.

We test:
1. get_collection() returns a valid ChromaDB collection
2. index_chunks() successfully indexes mock chunks
3. query() returns the correct number of results
4. query() returns expected metadata fields
5. index_chunks() raises ValueError for empty input
6. Re-indexing (upsert) doesn't create duplicates

We use a temporary ChromaDB directory for each test so tests don't
interfere with the real database in chroma_db/.

"""

import os
from unittest import result
import pytest
from unittest.mock import patch

from src.ingestion.chunker import LegislativeChunk


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def isolated_chroma(tmp_path):
    """
    Redirect ChromaDB to a temporary directory for each test.
    This ensures tests don't pollute the real chroma_db/ and
    each test starts with a clean empty collection.
    """
    import src.vectorstore.store as store_module
    store_module._client = None
    store_module._collection = None

    def mock_embed(texts):
        return [[(float(i + 1) % 50.0) / 100.0] * 1536 for i, _ in enumerate(texts)]

    def mock_embed_query(text):
        return [0.01] * 1536

    with patch.dict(os.environ, {
        "CHROMA_PERSIST_DIR": str(tmp_path / "test_chroma"),
        "CHROMA_COLLECTION_NAME": "test_collection",
    }), patch("src.vectorstore.store.embed", side_effect=mock_embed), patch("src.embeddings.embedder.embed_query", side_effect=mock_embed_query):
        yield

    store_module._client = None
    store_module._collection = None


def make_mock_chunks(n: int = 5) -> list[LegislativeChunk]:
    """Create n mock LegislativeChunk objects for testing."""
    return [
        LegislativeChunk(
            text=f"Section {i} text about policing duties and responsibilities in Ontario.",
            section_number=str(i),
            section_title=f"Test Section {i}",
            part_name="PART I — TEST",
            source_doc="TEST_DOC",
        )
        for i in range(1, n + 1)
    ]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestGetCollection:
    """Tests for get_collection()."""

    def test_returns_collection(self):
        """get_collection() should return a ChromaDB Collection object."""
        import chromadb
        from src.vectorstore.store import get_collection
        collection = get_collection()
        assert collection is not None
        assert isinstance(collection, chromadb.Collection)

    def test_collection_starts_empty(self):
        """A freshly created collection should have 0 documents."""
        from src.vectorstore.store import get_collection
        collection = get_collection()
        assert collection.count() == 0

    def test_get_collection_idempotent(self):
        """Calling get_collection() twice should return the same collection."""
        from src.vectorstore.store import get_collection
        col1 = get_collection()
        col2 = get_collection()
        assert col1.name == col2.name


class TestIndexChunks:
    """Tests for index_chunks()."""

    def test_indexes_chunks_successfully(self):
        """index_chunks() should add chunks to the collection."""
        from src.vectorstore.store import index_chunks, get_collection
        chunks = make_mock_chunks(5)
        index_chunks(chunks)
        assert get_collection().count() == 5

    def test_raises_for_empty_chunks(self):
        """index_chunks() should raise ValueError for an empty list."""
        from src.vectorstore.store import index_chunks
        with pytest.raises(ValueError):
            index_chunks([])

    def test_upsert_no_duplicates(self):
        """Running index_chunks() twice should not create duplicates."""
        from src.vectorstore.store import index_chunks, get_collection
        chunks = make_mock_chunks(5)
        index_chunks(chunks)
        index_chunks(chunks)  # Run again — should update, not duplicate
        # Should still be 5, not 10
        assert get_collection().count() == 5

    def test_metadata_stored_correctly(self):
        """Metadata fields should be retrievable after indexing."""
        from src.vectorstore.store import index_chunks, get_collection
        chunks = make_mock_chunks(1)
        index_chunks(chunks)

        collection = get_collection()
        result = collection.get(include=["metadatas"])
        meta = result["metadatas"][0]

        assert meta["section_number"] == "1"
        assert meta["section_title"] == "Test Section 1"
        assert meta["source_doc"] == "TEST_DOC"
        assert meta["part_name"] == "PART I — TEST"
        assert "citation" in meta


class TestQuery:
    """Tests for query()."""

    def test_returns_correct_number_of_results(self):
        """query() should return exactly top_k results."""
        from src.vectorstore.store import index_chunks, query
        from src.embeddings.embedder import embed_query

        chunks = make_mock_chunks(10)
        index_chunks(chunks)

        vec = embed_query("policing duties in Ontario")
        results = query(vec, top_k=5)

        assert len(results) == 5

    def test_results_have_required_fields(self):
        """Each result should have text, metadata, distance, and id."""
        from src.vectorstore.store import index_chunks, query
        from src.embeddings.embedder import embed_query

        chunks = make_mock_chunks(5)
        index_chunks(chunks)

        vec = embed_query("policing duties")
        results = query(vec, top_k=3)

        for result in results:
            assert "text" in result
            assert "section_number" in result
            assert "citation" in result
            assert "distance" in result
            assert "id" in result

    def test_metadata_fields_present_in_results(self):
        """Result metadata should contain all required fields."""
        from src.vectorstore.store import index_chunks, query
        from src.embeddings.embedder import embed_query

        chunks = make_mock_chunks(5)
        index_chunks(chunks)

        vec = embed_query("policing duties")
        results = query(vec, top_k=1)
        assert "section_number" in results[0]
        assert "section_title" in results[0]
        assert "part_name" in results[0]
        assert "source_doc" in results[0]
        assert "citation" in results[0]
