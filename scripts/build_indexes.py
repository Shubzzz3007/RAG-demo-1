# scripts/build_indexes.py
# ============================================================
# Offline Pipeline — Build FAISS Indexes
# ============================================================
# This script loads the saved embeddings and builds all 3 FAISS
# index types (Flat, IVF, HNSW), saving them to indexes/.
#
# PREREQUISITE:
#   Run 'python scripts/build_embeddings.py' first to generate
#   the embeddings.npy file.
#
# RUN THIS ONCE:
#   python scripts/build_indexes.py
#
# OUTPUTS:
#   indexes/flat.index  — Exact search baseline
#   indexes/ivf.index   — Approximate search with Voronoi cells
#   indexes/hnsw.index  — Graph-based ANN search
# ============================================================

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np

from src.config import EMBEDDINGS_DIR
from src.faiss_manager import build_all_indexes, save_index
from src.utils import get_logger

logger = get_logger(__name__)


def main():
    """
    Load embeddings from disk and build all FAISS indexes.
    """
    logger.info("=" * 60)
    logger.info("OFFLINE PIPELINE: Building FAISS Indexes")
    logger.info("=" * 60)

    # --- Step 1: Load embeddings ---
    embeddings_path = EMBEDDINGS_DIR / "embeddings.npy"

    if not embeddings_path.exists():
        logger.error(
            f"Embeddings file not found: {embeddings_path}. "
            f"Run 'python scripts/build_embeddings.py' first."
        )
        sys.exit(1)

    embeddings = np.load(embeddings_path)
    logger.info(f"Loaded embeddings: shape={embeddings.shape}")

    # --- Step 2: Build all indexes ---
    indexes = build_all_indexes(embeddings)

    # --- Step 3: Save all indexes ---
    for name, index in indexes.items():
        save_index(index, name)

    # --- Done ---
    logger.info("=" * 60)
    logger.info("FAISS INDEX BUILD COMPLETE")
    for name in indexes:
        logger.info(f"  {name}.index — {indexes[name].ntotal} vectors")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
