"""
tests/test_bm25_retriever.py
-----------------------------
Unit tests for src/retrieval/bm25_retriever.py.

We test:
1. tokenize() correctly processes legal text
2. build() creates a working index
3. search() returns correct number of results
4. search() returns results with required fields
5. search() ranks keyword matches higher than unrelated chunks
6. search() raises RuntimeError if called before build()
7. build() raises ValueError for empty input
8. is_built and corpus_size properties work correctly

Branch: feature/hybrid-retrieval-rrf
Issue:  #6 — BM25 index builder
"""

import pytest
from src.ingestion.chunker import LegislativeChunk
from src.retrieval.bm25_retriever import BM25Retriever, tokenize


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_mock_chunks(n: int = 10) -> list[LegislativeChunk]:
    """Create diverse mock chunks for testing BM25 ranking."""
    chunks = [
        LegislativeChunk(
            text="Duties of chief of police. The chief of police is responsible for "
                 "the administration of the police service and the management of its members.",
            section_number="41",
            section_title="Duties of chief of police",
            part_name="PART IV — POLICE SERVICES",
            source_doc="CSPA_2019",
        ),
        LegislativeChunk(
            text="Adequate and effective policing means all of the following functions "
                 "provided in accordance with the standards set out in the regulations.",
            section_number="11(1)",
            section_title="Adequate and effective policing",
            part_name="PART III — PROVISION OF POLICING",
            source_doc="CSPA_2019",
        ),
        LegislativeChunk(
            text="Inspector General of Policing. The Lieutenant Governor in Council "
                 "shall appoint an Inspector General of Policing.",
            section_number="gloss1",
            section_title="Inspector General",
            part_name="PART VIII — INSPECTOR GENERAL",
            source_doc="CSPA_2019",
        ),
        LegislativeChunk(
            text="Community safety and well-being plan. Every municipality shall prepare "
                 "a community safety and well-being plan in consultation with local agencies.",
            section_number="3(1)",
            section_title="Community safety plan",
            part_name="PART II — COMMUNITY SAFETY",
            source_doc="CSPA_2019",
        ),
        LegislativeChunk(
            text="Complaints about police officer conduct. Any person may make a complaint "
                 "about the conduct of a police officer to the Law Enforcement Complaints Agency.",
            section_number="76",
            section_title="Complaints about conduct",
            part_name="PART VII — COMPLAINTS",
            source_doc="CSPA_2019",
        ),
    ]
    # Return only n chunks
    return chunks[:min(n, len(chunks))]


# ---------------------------------------------------------------------------
# Tests for tokenize()
# ---------------------------------------------------------------------------

class TestTokenize:
    """Tests for the tokenize() helper function."""

    def test_lowercases_text(self):
        """tokenize() should return all lowercase tokens."""
        result = tokenize("Chief of Police")
        assert all(t == t.lower() for t in result)

    def test_splits_on_punctuation(self):
        """tokenize() should split on punctuation like brackets and dots."""
        result = tokenize("section 11(1) of the Act")
        assert "11" in result
        assert "1" in result
        assert "section" in result

    def test_filters_empty_tokens(self):
        """tokenize() should not return empty strings."""
        result = tokenize("  multiple   spaces  ")
        assert "" not in result
        assert all(len(t) > 0 for t in result)

    def test_handles_empty_string(self):
        """tokenize() should return empty list for empty input."""
        result = tokenize("")
        assert result == []

    def test_handles_legal_abbreviations(self):
        """tokenize() should handle OPP, SIU and similar abbreviations."""
        result = tokenize("The OPP and SIU have jurisdiction")
        assert "opp" in result
        assert "siu" in result


# ---------------------------------------------------------------------------
# Tests for BM25Retriever
# ---------------------------------------------------------------------------

class TestBM25RetrieverBuild:
    """Tests for BM25Retriever.build()."""

    def test_build_succeeds_with_valid_chunks(self):
        """build() should set is_built to True."""
        retriever = BM25Retriever()
        chunks = make_mock_chunks(5)
        retriever.build(chunks)
        assert retriever.is_built is True

    def test_build_sets_corpus_size(self):
        """corpus_size should equal the number of chunks after build()."""
        retriever = BM25Retriever()
        chunks = make_mock_chunks(5)
        retriever.build(chunks)
        assert retriever.corpus_size == 5

    def test_build_raises_for_empty_chunks(self):
        """build() should raise ValueError for empty input."""
        retriever = BM25Retriever()
        with pytest.raises(ValueError) as exc_info:
            retriever.build([])
        assert "empty" in str(exc_info.value).lower()

    def test_is_built_false_before_build(self):
        """is_built should be False before build() is called."""
        retriever = BM25Retriever()
        assert retriever.is_built is False

    def test_corpus_size_zero_before_build(self):
        """corpus_size should be 0 before build() is called."""
        retriever = BM25Retriever()
        assert retriever.corpus_size == 0


class TestBM25RetrieverSearch:
    """Tests for BM25Retriever.search()."""

    def test_search_raises_before_build(self):
        """search() should raise RuntimeError if called before build()."""
        retriever = BM25Retriever()
        with pytest.raises(RuntimeError) as exc_info:
            retriever.search("policing duties")
        assert "build" in str(exc_info.value).lower()

    def test_search_returns_correct_number_of_results(self):
        """search() should return exactly top_k results."""
        retriever = BM25Retriever()
        retriever.build(make_mock_chunks(5))
        results = retriever.search("policing duties", top_k=3)
        assert len(results) == 3

    def test_search_returns_all_results_when_top_k_exceeds_corpus(self):
        """search() should return all chunks if top_k > corpus size."""
        retriever = BM25Retriever()
        retriever.build(make_mock_chunks(3))
        results = retriever.search("policing", top_k=10)
        assert len(results) == 3

    def test_search_results_have_required_fields(self):
        """Each result should have all required metadata fields."""
        retriever = BM25Retriever()
        retriever.build(make_mock_chunks(5))
        results = retriever.search("chief of police", top_k=1)

        required_fields = {
            "id", "text", "score", "rank",
            "section_number", "section_title",
            "part_name", "source_doc", "citation"
        }
        assert required_fields.issubset(set(results[0].keys()))

    def test_search_ranks_keyword_match_first(self):
        """
        A chunk containing the exact query keywords should rank higher
        than unrelated chunks. This validates BM25 is working correctly.
        """
        retriever = BM25Retriever()
        retriever.build(make_mock_chunks(5))

        # Search for a term that appears in only one chunk
        results = retriever.search("inspector general policing", top_k=5)

        # The Inspector General chunk should be ranked first
        top_result = results[0]
        assert "Inspector General" in top_result["section_title"] or \
               "inspector" in top_result["text"].lower()

    def test_search_scores_are_non_negative(self):
        """BM25 scores should be non-negative floats."""
        retriever = BM25Retriever()
        retriever.build(make_mock_chunks(5))
        results = retriever.search("policing duties", top_k=5)
        for r in results:
            assert r["score"] >= 0
            assert isinstance(r["score"], float)

    def test_search_ranks_are_sequential(self):
        """Ranks should be sequential integers starting from 1."""
        retriever = BM25Retriever()
        retriever.build(make_mock_chunks(5))
        results = retriever.search("policing", top_k=5)
        ranks = [r["rank"] for r in results]
        assert ranks == list(range(1, len(results) + 1))

    def test_search_source_doc_preserved(self):
        """source_doc metadata should be preserved in results."""
        retriever = BM25Retriever()
        retriever.build(make_mock_chunks(5))
        results = retriever.search("policing", top_k=3)
        for r in results:
            assert r["source_doc"] == "CSPA_2019"
