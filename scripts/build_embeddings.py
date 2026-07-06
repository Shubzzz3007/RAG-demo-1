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
#   embeddings/chunks_baseline.pkl      — List of Chunk objects (Section-aware)
#   embeddings/embeddings_baseline.npy  — numpy array (n_chunks, 1536)
#   embeddings/chunks_recursive.pkl     — List of Chunk objects (Recursive)
#   embeddings/embeddings_recursive.npy — numpy array (n_chunks, 1536)
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
from src.chunker import chunk_documents, chunk_documents_recursive
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

    # --- Step 2: Chunk documents (Baseline) ---
    logger.info("Step 2a: Chunking documents (Baseline - Section-aware)...")
    chunks_baseline = chunk_documents(documents)
    logger.info(f"Created {len(chunks_baseline)} baseline chunks")

    logger.info("Step 2b: Chunking documents (Recursive - Sliding window)...")
    chunks_recursive = chunk_documents_recursive(documents)
    logger.info(f"Created {len(chunks_recursive)} recursive chunks")

    # --- Step 3: Generate embeddings ---
    logger.info("Step 3: Generating embeddings...")
    embedding_service = EmbeddingService()

    logger.info("Embedding baseline chunks...")
    chunk_texts_baseline = [chunk.text for chunk in chunks_baseline]
    embeddings_baseline = embedding_service.embed_texts(chunk_texts_baseline)
    
    logger.info("Embedding recursive chunks...")
    chunk_texts_recursive = [chunk.text for chunk in chunks_recursive]
    embeddings_recursive = embedding_service.embed_texts(chunk_texts_recursive)

    # --- Step 4: Save to disk ---
    logger.info("Step 4: Saving to disk...")
    ensure_directory(EMBEDDINGS_DIR)

    # Save Baseline
    path_chunks_baseline = EMBEDDINGS_DIR / "chunks_baseline.pkl"
    with open(path_chunks_baseline, "wb") as f:
        pickle.dump(chunks_baseline, f)
    np.save(EMBEDDINGS_DIR / "embeddings_baseline.npy", embeddings_baseline)

    # Save Recursive
    path_chunks_recursive = EMBEDDINGS_DIR / "chunks_recursive.pkl"
    with open(path_chunks_recursive, "wb") as f:
        pickle.dump(chunks_recursive, f)
    np.save(EMBEDDINGS_DIR / "embeddings_recursive.npy", embeddings_recursive)

    logger.info("=" * 60)
    logger.info("OFFLINE PIPELINE COMPLETE")
    logger.info(f"  Baseline Chunks:  {len(chunks_baseline)}")
    logger.info(f"  Recursive Chunks: {len(chunks_recursive)}")


if __name__ == "__main__":
    main()
