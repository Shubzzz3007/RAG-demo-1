# src/faiss_manager.py
# ============================================================
# FAISS Index Manager
# ============================================================
# This module builds and loads FAISS vector indexes.
#
# WHY 3 different index types?
#   Each index represents a different tradeoff:
#
#   Flat (IndexFlatIP)
#     - Exact search — compares query against ALL vectors
#     - Pros: 100% recall, no training needed
#     - Cons: Slow for large datasets (O(n) per query)
#     - Best for: Small datasets (<10K vectors), accuracy baseline
#
#   IVF (IndexIVFFlat)
#     - Approximate search using Voronoi partitioning
#     - Splits vectors into nlist clusters (cells)
#     - At query time, only searches nprobe closest clusters
#     - Pros: Much faster than Flat for large datasets
#     - Cons: May miss some relevant results
#     - Best for: Medium datasets (10K-1M vectors)
#
#   HNSW (IndexHNSWFlat)
#     - Graph-based approximate nearest neighbor search
#     - Builds a hierarchical navigable small world graph
#     - Pros: Fast, good recall, no training needed
#     - Cons: Uses more memory than IVF
#     - Best for: Production systems needing speed + accuracy
#
# ALL indexes use Inner Product on L2-normalized vectors,
# which is equivalent to cosine similarity.
#
# IMPORTANT:
#   - Indexes are built ONCE during the offline pipeline
#   - At runtime, we only LOAD the selected index
#   - Never rebuild indexes at runtime
# ============================================================

import faiss
import numpy as np
from pathlib import Path

from src.config import (
    INDEXES_DIR,
    EMBEDDING_DIMENSION,
    IVF_NLIST,
    IVF_NPROBE,
    HNSW_M,
    HNSW_EF_SEARCH,
    HNSW_EF_CONSTRUCTION,
)
from src.utils import get_logger, timer, ensure_directory

logger = get_logger(__name__)


# ============================================================
# INDEX BUILDING FUNCTIONS
# ============================================================

def build_flat_index(embeddings: np.ndarray) -> faiss.Index:
    """
    Build a Flat (exact search) FAISS index.

    This is the simplest index — it stores all vectors and
    performs exhaustive search (compares query to every vector).

    Inner Product (IP) is used because our embeddings are
    L2-normalized, making IP equivalent to cosine similarity.

    Args:
        embeddings: numpy array of shape (n_vectors, dimension).

    Returns:
        A trained FAISS IndexFlatIP.
    """
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)
    logger.info(
        f"Built Flat index: {index.ntotal} vectors, "
        f"dimension={dimension}"
    )
    return index


def build_ivf_index(embeddings: np.ndarray) -> faiss.Index:
    """
    Build an IVF (Inverted File) FAISS index.

    IVF partitions the vector space into nlist Voronoi cells.
    During search, only nprobe cells are checked, making it
    faster than Flat at the cost of some recall.

    The quantizer is a Flat index that maps queries to cells.
    The IVF index must be trained before adding vectors.

    Args:
        embeddings: numpy array of shape (n_vectors, dimension).

    Returns:
        A trained FAISS IndexIVFFlat with nprobe set.
    """
    dimension = embeddings.shape[1]

    # The quantizer determines which cell a vector belongs to
    quantizer = faiss.IndexFlatIP(dimension)

    # Create IVF index with inner product metric
    index = faiss.IndexIVFFlat(
        quantizer,
        dimension,
        IVF_NLIST,           # Number of clusters/cells
        faiss.METRIC_INNER_PRODUCT,
    )

    # IVF must be trained on the data to learn cluster centroids
    logger.info(f"Training IVF index with nlist={IVF_NLIST}...")
    index.train(embeddings)

    # Add vectors to the trained index
    index.add(embeddings)

    # Set search-time parameter: how many cells to probe
    index.nprobe = IVF_NPROBE

    logger.info(
        f"Built IVF index: {index.ntotal} vectors, "
        f"nlist={IVF_NLIST}, nprobe={IVF_NPROBE}"
    )
    return index


def build_hnsw_index(embeddings: np.ndarray) -> faiss.Index:
    """
    Build an HNSW (Hierarchical Navigable Small World) FAISS index.

    HNSW builds a multi-layered graph where each vector is connected
    to M neighbors. Search traverses the graph starting from an
    entry point, greedily moving to closer neighbors.

    HNSW does NOT need training — it builds the graph incrementally
    as vectors are added.

    Args:
        embeddings: numpy array of shape (n_vectors, dimension).

    Returns:
        A FAISS IndexHNSWFlat with search parameters set.
    """
    dimension = embeddings.shape[1]

    # Create HNSW index
    # M = number of neighbors per node in the graph
    index = faiss.IndexHNSWFlat(dimension, HNSW_M, faiss.METRIC_INNER_PRODUCT)

    # Set construction parameter (higher = better graph, slower build)
    index.hnsw.efConstruction = HNSW_EF_CONSTRUCTION

    # Set search parameter (higher = better recall, slower search)
    index.hnsw.efSearch = HNSW_EF_SEARCH

    # Add vectors (no training needed for HNSW)
    index.add(embeddings)

    logger.info(
        f"Built HNSW index: {index.ntotal} vectors, "
        f"M={HNSW_M}, efSearch={HNSW_EF_SEARCH}, "
        f"efConstruction={HNSW_EF_CONSTRUCTION}"
    )
    return index


# ============================================================
# SAVE AND LOAD FUNCTIONS
# ============================================================

def save_index(index: faiss.Index, name: str) -> Path:
    """
    Save a FAISS index to the indexes/ directory.

    Args:
        index: The FAISS index to save.
        name:  Index name (e.g., "flat", "ivf", "hnsw").

    Returns:
        Path to the saved index file.
    """
    ensure_directory(INDEXES_DIR)
    filepath = INDEXES_DIR / f"{name}.index"
    faiss.write_index(index, str(filepath))
    logger.info(f"Saved {name} index to {filepath}")
    return filepath


def load_index(name: str) -> faiss.Index:
    """
    Load a FAISS index from the indexes/ directory.

    This is used at RUNTIME to load whichever index the user
    selected in the Streamlit sidebar.

    Args:
        name: Index name (e.g., "flat", "ivf", "hnsw").

    Returns:
        The loaded FAISS index.

    Raises:
        FileNotFoundError: If the index file doesn't exist.
    """
    filepath = INDEXES_DIR / f"{name}.index"

    if not filepath.exists():
        raise FileNotFoundError(
            f"Index file not found: {filepath}. "
            f"Run 'python scripts/build_indexes.py' first."
        )

    index = faiss.read_index(str(filepath))

    # Restore IVF nprobe setting (FAISS doesn't persist this)
    if hasattr(index, "nprobe"):
        index.nprobe = IVF_NPROBE

    # Restore HNSW efSearch setting
    if hasattr(index, "hnsw"):
        index.hnsw.efSearch = HNSW_EF_SEARCH

    logger.info(f"Loaded {name} index: {index.ntotal} vectors")
    return index


# ============================================================
# BUILD ALL INDEXES
# ============================================================

@timer
def build_all_indexes(embeddings: np.ndarray) -> dict[str, faiss.Index]:
    """
    Build all 3 FAISS indexes from the provided embeddings.

    This is called by the offline build script.

    Args:
        embeddings: numpy array of shape (n_vectors, dimension).

    Returns:
        Dict mapping index name → FAISS index object.
    """
    logger.info(f"Building all indexes from {embeddings.shape[0]} vectors...")

    indexes = {
        "flat": build_flat_index(embeddings),
        "ivf": build_ivf_index(embeddings),
        "hnsw": build_hnsw_index(embeddings),
    }

    return indexes
