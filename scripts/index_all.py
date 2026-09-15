"""
scripts/index_all.py
---------------------
End-to-end indexing script: loads all documents (CSPA .docx, O. Reg .docx,
and LECA Guidelines .pdf), chunks them, embeds the chunks, and stores them in ChromaDB.

Run this once before starting the app to populate the vector store.
Re-running is safe — upsert() won't create duplicates.

Usage:
    python scripts/index_all.py

Branch: feature/chromadb-vectorstore
Issue:  #5 — ChromaDB vectorstore setup and indexing
"""

import sys
import os

# Add the repo root to the Python path so we can import from src/
# This is needed when running the script directly (not via pytest)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ingestion.pipeline import run_ingestion
from src.vectorstore.store import index_chunks, get_collection


def main() -> None:
    """Run the full ingestion pipeline: load → chunk → embed → index."""

    print("=" * 60)
    print("Ontario Oversight CRAG Indexing Pipeline (All Sources)")
    print("=" * 60)

    # Step 1 & 2: Load and Chunk all documents
    print("\n[Step 1-2/3] Loading and chunking all documents...")
    chunks = run_ingestion()

    # Step 3: Embed and index all chunks into ChromaDB
    print("\n[Step 3/3] Embedding and indexing chunks into ChromaDB...")
    index_chunks(chunks)

    # Verify the index
    collection = get_collection()
    print(f"\n{'=' * 60}")
    print(f"✓ Indexing complete!")
    print(f"  Collection: {collection.name}")
    print(f"  Documents:  {collection.count()}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
