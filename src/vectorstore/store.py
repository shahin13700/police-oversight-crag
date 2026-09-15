"""
src/vectorstore/store.py
------------------------
Sets up and manages a persistent ChromaDB vector store for the CSPA legislation.

WHY CHROMADB:
ChromaDB is a lightweight, open-source vector database that runs locally with
no server setup required. It persists embeddings to disk so we don't have to
re-embed the entire CSPA document every time the app starts — only when the
document changes or is first indexed.

HOW IT WORKS:
1. get_collection() opens (or creates) a ChromaDB collection on disk
2. index_chunks() takes LegislativeChunk objects, embeds them, and upserts
   them into the collection with full metadata
3. query() takes an embedding vector and returns the top-k most similar chunks

WHY WE MANAGE EMBEDDINGS MANUALLY:
ChromaDB has a built-in embedding function slot, but we pass
embedding_function=None and supply our own embeddings. This keeps all
embedding logic in src/embeddings/embedder.py and avoids tight coupling.

Branch: feature/chromadb-vectorstore
Issue:  #5 — ChromaDB vectorstore setup and indexing
"""

import os
import logging
from typing import Optional
from dotenv import load_dotenv
import chromadb
from chromadb.config import Settings

from src.ingestion.chunker import LegislativeChunk
from src.embeddings.embedder import embed

load_dotenv()
logger = logging.getLogger(__name__)

_client: Optional[chromadb.PersistentClient] = None
_collection: Optional[chromadb.Collection] = None


def get_collection() -> chromadb.Collection:
    """
    Return the singleton ChromaDB collection, creating it if necessary.

    The collection is persisted to disk at CHROMA_PERSIST_DIR (from .env).
    If the collection already exists it is opened and returned.
    If not, it is created empty and ready for index_chunks().

    Returns:
        chromadb.Collection: The opened or created collection.
    """
    global _client, _collection

    if _collection is not None:
        return _collection

    persist_dir = os.getenv("CHROMA_PERSIST_DIR", "chroma_db")
    collection_name = os.getenv("CHROMA_COLLECTION_NAME", "ontario_legislation")

    os.makedirs(persist_dir, exist_ok=True)

    _client = chromadb.PersistentClient(
        path=persist_dir,
        settings=Settings(anonymized_telemetry=False)
    )

    _collection = _client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"}
    )

    count = _collection.count()
    print(f"[vectorstore] Opened collection '{collection_name}' "
          f"({count} chunks indexed, persisted at '{persist_dir}')")

    return _collection


def index_chunks(chunks: list[LegislativeChunk]) -> None:
    """
    Embed a list of LegislativeChunks and upsert them into ChromaDB.

    Uses upsert so it is safe to call multiple times without duplicates.
    Each chunk is stored with its embedding vector, full text, and all
    metadata fields needed for citation and filtering.

    Args:
        chunks: List of LegislativeChunk objects from the chunker.

    Raises:
        ValueError: If chunks list is empty.
    """
    if not chunks:
        raise ValueError("index_chunks() received an empty list of chunks.")

    collection = get_collection()

    print(f"[vectorstore] Embedding {len(chunks)} chunks — this may take a minute...")

    texts = [chunk.text for chunk in chunks]
    embeddings = embed(texts)

    ids = []
    metadatas = []

    for i, chunk in enumerate(chunks):
        title_slug = chunk.section_title[:50].replace(" ", "_").replace("/", "-")
        section_part = chunk.section_number if chunk.section_number else "no_section"
        chunk_id = f"{chunk.source_doc}__{section_part}__{title_slug}__{i}"

        ids.append(chunk_id)
        metadatas.append({
            "section_number": chunk.section_number,
            "section_title": chunk.section_title,
            "part_name": chunk.part_name,
            "source_doc": chunk.source_doc,
            "citation": chunk.citation(),
        })

    collection.upsert(
        ids=ids,
        embeddings=embeddings,
        documents=texts,
        metadatas=metadatas,
    )

    print(f"[vectorstore] Successfully indexed {len(chunks)} chunks. "
          f"Collection now has {collection.count()} total chunks.")


def query(query_embedding: list[float], top_k: int = 5) -> list[dict]:
    """
    Search the vector store for the top-k most similar chunks.

    Args:
        query_embedding: A single embedding vector from embed_query().
        top_k: Number of results to return. Defaults to 5.

    Returns:
        List of dicts with keys: id, text, score, section_number,
        section_title, part_name, source_doc, citation.

    Raises:
        ValueError: If the collection is empty (not yet indexed).
    """
    collection = get_collection()

    if collection.count() == 0:
        raise ValueError(
            "The ChromaDB collection is empty. "
            "Run scripts/index_cspa.py first to index the CSPA document."
        )

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, collection.count()),
        include=["documents", "metadatas", "distances"],
    )

    formatted = []
    for i in range(len(results["ids"][0])):
        metadata = results["metadatas"][0][i]
        distance = results["distances"][0][i]
        score = 1 - distance  # convert cosine distance to similarity

        formatted.append({
            "id": results["ids"][0][i],
            "text": results["documents"][0][i],
            "score": score,
            "distance": distance,  # <--- ADD THIS LINE TO SATISFY THE TEST
            "section_number": metadata.get("section_number", ""),
            "section_title": metadata.get("section_title", ""),
            "part_name": metadata.get("part_name", ""),
            "source_doc": metadata.get("source_doc", ""),
            "citation": metadata.get("citation", ""),
        })

    return formatted


def reset_collection() -> None:
    """
    Delete and recreate the ChromaDB collection.

    USE WITH CAUTION — deletes all indexed chunks and cannot be undone.
    Useful during development when re-indexing after chunking changes.
    """
    global _client, _collection

    collection_name = os.getenv("CHROMA_COLLECTION_NAME", "ontario_legislation")

    if _client is None:
        get_collection()

    _client.delete_collection(collection_name)
    _collection = None

    print(f"[vectorstore] Collection '{collection_name}' deleted and reset.")
