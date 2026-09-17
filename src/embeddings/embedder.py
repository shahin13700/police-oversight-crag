"""
src/embeddings/embedder.py
--------------------------
Provides a reusable embedding module that calls OpenRouter's embeddings API
and exposes a simple embed() function for the rest of the pipeline.

WHY OPENROUTER API:
Offloading embeddings to an API saves local memory and compute time.
We use openai/text-embedding-3-small (1536d) for embeddings.

"""

import os
import logging
import requests
import time
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Set up logging so we can see what's happening without using print()
# In production, logs can be redirected to a file or monitoring system
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def embed(texts: list[str]) -> list[list[float]]:
    """
    Convert a list of text strings into a list of embedding vectors.

    Each text is converted to a fixed-size float vector (1536 dimensions
    for openai/text-embedding-3-small). Vectors that are closer together in this space
    represent texts that are semantically similar.

    This function is the main interface for the rest of the pipeline:
    - The vectorstore module calls it to embed chunks during indexing
    - The retrieval module calls it to embed user queries at search time

    Args:
        texts: List of strings to embed. Can be a single string in a list,
               a batch of document chunks, or a single query.
               Empty strings are handled gracefully.

    Returns:
        List of embedding vectors. Each vector is a list of floats with
        length equal to 1536.
        The output list has the same length as the input list.

    Raises:
        ValueError: If texts is empty.
    """
    if not texts:
        raise ValueError(
            "embed() received an empty list. "
            "Pass at least one string to embed."
        )

    model_name = os.getenv("EMBEDDING_MODEL", "openai/text-embedding-3-small")
    api_key = os.getenv("OPENROUTER_API_KEY")
    api_url = os.getenv("OPENROUTER_API_URL", "https://openrouter.ai/api/v1")
    
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY environment variable is not set.")

    endpoint = f"{api_url.rstrip('/')}/embeddings"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    all_embeddings = []
    
    # Batch documents before sending (reduced to 20 to prevent token limits)
    batch_size = 20
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        payload = {
            "model": model_name,
            "input": batch
        }
        
        max_retries = 3
        data = None
        for attempt in range(max_retries):
            response = requests.post(endpoint, headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()
            
            if "error" in data:
                logger.warning(f"OpenRouter API error on attempt {attempt+1}: {data['error']}")
                time.sleep(2 ** attempt)
                continue
            else:
                break
                
        if data is None or "error" in data:
             raise ValueError(f"OpenRouter API returned an error after retries: {data.get('error', 'unknown error')}")
            
        # OpenRouter returns data in the standard OpenAI format:
        # { "data": [ { "embedding": [...] }, ... ] }
        data_list = data.get("data", [])
        if len(data_list) != len(batch):
            raise ValueError(f"OpenRouter API returned {len(data_list)} embeddings for a batch of {len(batch)} texts.")
            
        sorted_data = sorted(data_list, key=lambda x: x.get("index", 0))
        batch_embeddings = [item["embedding"] for item in sorted_data]
        all_embeddings.extend(batch_embeddings)
        
        # Minor sleep to avoid rate limiting
        if i + batch_size < len(texts):
            time.sleep(0.5)

    return all_embeddings


def embed_query(query: str) -> list[float]:
    """
    Convenience function to embed a single query string.

    This is a thin wrapper around embed() that handles the common case
    of embedding a single user query at retrieval time. It saves callers
    from having to wrap their query in a list and unwrap the result.

    Args:
        query: The user's question or search query as a plain string.

    Returns:
        A single embedding vector as a list of floats.

    Example:
        >>> vector = embed_query("What are police duties?")
        >>> print(len(vector))  # 1536
    """
    return embed([query])[0]


def get_embedding_dimension() -> int:
    """
    Return the embedding dimension.

    This is useful for configuring ChromaDB's collection, which needs
    to know the vector size upfront.

    Returns:
        Integer dimension of the embedding vectors (1536 for openai/text-embedding-3-small).
    """
    # Hardcoded — update this if EMBEDDING_MODEL changes (text-embedding-3-small = 1536)
    return 1536
