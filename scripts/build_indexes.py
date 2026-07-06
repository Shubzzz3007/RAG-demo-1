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
#   indexes/flat_baseline.index  — Exact search baseline
#   indexes/ivf_baseline.index   — Approximate search with Voronoi cells
#   indexes/hnsw_baseline.index  — Graph-based ANN search
#   (and the same with _recursive suffix)
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
    embeddings_baseline_path = EMBEDDINGS_DIR / "embeddings_baseline.npy"
    embeddings_recursive_path = EMBEDDINGS_DIR / "embeddings_recursive.npy"

    if not embeddings_baseline_path.exists() or not embeddings_recursive_path.exists():
        logger.error(
            f"Embeddings files not found. "
            f"Run 'python scripts/build_embeddings.py' first."
        )
        sys.exit(1)

    embeddings_baseline = np.load(embeddings_baseline_path)
    embeddings_recursive = np.load(embeddings_recursive_path)
    logger.info(f"Loaded baseline embeddings: shape={embeddings_baseline.shape}")
    logger.info(f"Loaded recursive embeddings: shape={embeddings_recursive.shape}")

    # --- Step 2: Build all indexes ---
    logger.info("Building baseline indexes...")
    indexes_baseline = build_all_indexes(embeddings_baseline)
    
    logger.info("Building recursive indexes...")
    indexes_recursive = build_all_indexes(embeddings_recursive)

    # --- Step 3: Save all indexes ---
    for name, index in indexes_baseline.items():
        save_index(index, f"{name}_baseline")
        
    for name, index in indexes_recursive.items():
        save_index(index, f"{name}_recursive")

    # --- Done ---
    logger.info("=" * 60)
    logger.info("FAISS INDEX BUILD COMPLETE")
    for name in indexes_baseline:
        logger.info(f"  {name}_baseline.index — {indexes_baseline[name].ntotal} vectors")
        logger.info(f"  {name}_recursive.index — {indexes_recursive[name].ntotal} vectors")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
