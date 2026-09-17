"""
tests/test_embedder.py
----------------------
Unit tests for src/embeddings/embedder.py.

We test:
1. embed() returns correct output shape
2. embed() returns consistent results for the same input
3. embed_query() returns a single vector
4. get_embedding_dimension() returns the correct dimension
5. embed() raises ValueError for empty input

"""

import os
import pytest
from unittest.mock import patch, MagicMock


def mock_openrouter_response(num_vectors, dim=1536):
    """A helper to create fake OpenAI-format JSON response from OpenRouter"""
    data = []
    for i in range(num_vectors):
        # Generate some dummy floats for vectors
        embedding = [(float(j) + i) * 0.001 for j in range(dim)]
        data.append({"index": i, "embedding": embedding})
    
    response = MagicMock()
    response.json.return_value = {"data": data}
    response.raise_for_status = MagicMock()
    return response


class TestEmbed:
    """Tests for the embed() function."""

    @patch("src.embeddings.embedder.requests.post")
    @patch.dict(os.environ, {"OPENROUTER_API_KEY": "fake_key"})
    def test_embed_returns_correct_number_of_vectors(self, mock_post):
        """embed() should return one vector per input string."""
        mock_post.return_value = mock_openrouter_response(3)
        from src.embeddings.embedder import embed
        texts = ["First sentence.", "Second sentence.", "Third sentence."]
        result = embed(texts)
        assert len(result) == 3

    @patch("src.embeddings.embedder.requests.post")
    @patch.dict(os.environ, {"OPENROUTER_API_KEY": "fake_key"})
    def test_embed_returns_correct_dimension(self, mock_post):
        """Each vector should have 1536 dimensions for openai/text-embedding-3-small."""
        mock_post.return_value = mock_openrouter_response(1)
        from src.embeddings.embedder import embed
        result = embed(["Test sentence."])
        assert len(result[0]) == 1536

    @patch("src.embeddings.embedder.requests.post")
    @patch.dict(os.environ, {"OPENROUTER_API_KEY": "fake_key"})
    def test_embed_returns_list_of_lists(self, mock_post):
        """embed() should return a list of lists."""
        mock_post.return_value = mock_openrouter_response(1)
        from src.embeddings.embedder import embed
        result = embed(["Test sentence."])
        assert isinstance(result, list)
        assert isinstance(result[0], list)
        assert isinstance(result[0][0], float)

    @patch("src.embeddings.embedder.requests.post")
    @patch.dict(os.environ, {"OPENROUTER_API_KEY": "fake_key"})
    def test_embed_single_string(self, mock_post):
        """embed() should work with a single string in a list."""
        mock_post.return_value = mock_openrouter_response(1)
        from src.embeddings.embedder import embed
        result = embed(["What are the duties of a police chief?"])
        assert len(result) == 1
        assert len(result[0]) == 1536

    def test_embed_raises_for_empty_list(self):
        """embed() should raise ValueError when given an empty list."""
        from src.embeddings.embedder import embed
        with pytest.raises(ValueError) as exc_info:
            embed([])
        assert "empty" in str(exc_info.value).lower()

    @patch("src.embeddings.embedder.requests.post")
    @patch.dict(os.environ, {"OPENROUTER_API_KEY": "fake_key"})
    def test_embed_consistent_results(self, mock_post):
        """
        The mock should return the same result.
        """
        response1 = mock_openrouter_response(1)
        mock_post.side_effect = [response1, response1]
        
        from src.embeddings.embedder import embed
        text = ["Policing shall be provided throughout Ontario."]
        
        # We need to test sequential calls without caching effects
        result1 = embed(text)
        result2 = embed(text)
        assert result1 == result2

    @patch("src.embeddings.embedder.requests.post")
    @patch.dict(os.environ, {"OPENROUTER_API_KEY": "fake_key"})
    def test_different_texts_produce_different_vectors(self, mock_post):
        """Different texts should produce different vectors. Handled by mock."""
        mock_post.return_value = mock_openrouter_response(2)
        from src.embeddings.embedder import embed
        result = embed([
            "The duties of the chief of police.",
            "The weather in Toronto today."
        ])
        assert result[0] != result[1]


class TestEmbedQuery:
    """Tests for the embed_query() convenience function."""

    @patch("src.embeddings.embedder.requests.post")
    @patch.dict(os.environ, {"OPENROUTER_API_KEY": "fake_key"})
    def test_returns_single_vector(self, mock_post):
        """embed_query() should return a single flat vector, not a list of lists."""
        mock_post.return_value = mock_openrouter_response(1)
        from src.embeddings.embedder import embed_query
        result = embed_query("What are police duties?")
        assert isinstance(result, list)
        assert isinstance(result[0], float)  # flat list, not nested
        assert len(result) == 1536

    @patch("src.embeddings.embedder.requests.post")
    @patch.dict(os.environ, {"OPENROUTER_API_KEY": "fake_key"})
    def test_matches_embed_output(self, mock_post):
        """embed_query(text) should equal embed([text])[0]."""
        response = mock_openrouter_response(1)
        # We need two responses, one for embed_query (which calls embed), and one for explicit embed
        mock_post.side_effect = [response, response]
        
        from src.embeddings.embedder import embed, embed_query
        query = "What are the duties of a police chief?"
        assert embed_query(query) == embed([query])[0]


class TestGetEmbeddingDimension:
    """Tests for get_embedding_dimension()."""

    def test_returns_correct_dimension(self):
        """Should return 1536."""
        from src.embeddings.embedder import get_embedding_dimension
        dim = get_embedding_dimension()
        assert dim == 1536

    def test_returns_integer(self):
        """Dimension should be an integer."""
        from src.embeddings.embedder import get_embedding_dimension
        assert isinstance(get_embedding_dimension(), int)
