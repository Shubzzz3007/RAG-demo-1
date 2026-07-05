# scripts/build_embeddings.py
# ============================================================
# Offline Pipeline — Build Embeddings
# ============================================================
# This script runs the OFFLINE pipeline:
#   1. Load all clinical documents
#   2. Chunk them using section-aware chunking
#   3. Generate embeddings using Azure OpenAI
#   4. Save chunks (as pickle) and embeddings (as numpy) to disk
#
# RUN THIS ONCE:
#   python scripts/build_embeddings.py
#
# OUTPUTS:
#   embeddings/chunks.pkl      — List of Chunk objects
#   embeddings/embeddings.npy  — numpy array (n_chunks, 1536)
#
# IMPORTANT:
#   This script calls the Azure OpenAI API, which costs money.
#   For ~1072 chunks, it makes ~11 API calls (100 per batch).
#   With text-embedding-3-small, this costs roughly $0.01.
# ============================================================

import pickle
import sys
from pathlib import Path

# Add project root to path so we can import src modules
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np

from src.config import EMBEDDINGS_DIR
from src.document_loader import load_documents
from src.chunker import chunk_documents
from src.embedding_service import EmbeddingService
from src.utils import get_logger, ensure_directory

logger = get_logger(__name__)


def main():
    """
    Run the complete offline embedding pipeline.

    Steps:
        1. Load documents from data/
        2. Chunk documents using section-aware chunking
        3. Generate embeddings via Azure OpenAI
        4. Save chunks and embeddings to embeddings/
    """
    logger.info("=" * 60)
    logger.info("OFFLINE PIPELINE: Building Embeddings")
    logger.info("=" * 60)

    # --- Step 1: Load documents ---
    logger.info("Step 1: Loading documents...")
    documents = load_documents()
    logger.info(f"Loaded {len(documents)} documents")

    # --- Step 2: Chunk documents ---
    logger.info("Step 2: Chunking documents...")
    chunks = chunk_documents(documents)
    logger.info(f"Created {len(chunks)} chunks")

    # --- Step 3: Generate embeddings ---
    logger.info("Step 3: Generating embeddings...")
    embedding_service = EmbeddingService()

    # Extract text from each chunk for embedding
    chunk_texts = [chunk.text for chunk in chunks]
    embeddings = embedding_service.embed_texts(chunk_texts)
    logger.info(f"Generated embeddings with shape: {embeddings.shape}")

    # --- Step 4: Save to disk ---
    logger.info("Step 4: Saving to disk...")
    ensure_directory(EMBEDDINGS_DIR)

    # Save chunks as pickle (preserves the full Chunk dataclass)
    chunks_path = EMBEDDINGS_DIR / "chunks.pkl"
    with open(chunks_path, "wb") as f:
        pickle.dump(chunks, f)
    logger.info(f"Saved {len(chunks)} chunks to {chunks_path}")

    # Save embeddings as numpy array
    embeddings_path = EMBEDDINGS_DIR / "embeddings.npy"
    np.save(embeddings_path, embeddings)
    logger.info(f"Saved embeddings ({embeddings.shape}) to {embeddings_path}")

    # --- Done ---
    logger.info("=" * 60)
    logger.info("OFFLINE PIPELINE COMPLETE")
    logger.info(f"  Chunks:     {len(chunks)}")
    logger.info(f"  Embeddings: {embeddings.shape}")
    logger.info(f"  Saved to:   {EMBEDDINGS_DIR}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
