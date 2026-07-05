# src/retriever.py
# ============================================================
# Dense Retriever — Baseline Retrieval
# ============================================================
# This module performs dense embedding retrieval using FAISS.
#
# HOW IT WORKS:
#   1. Embed the user's query using Azure OpenAI
#   2. Search the selected FAISS index for nearest neighbors
#   3. Return the top-k chunks with similarity scores
#
# This is the BASELINE retrieval strategy. All enhancements
# (HyDE, MMR, cross-encoder) are compared against this baseline.
#
# WHY inner product for search?
#   Our embeddings are L2-normalized (done in EmbeddingService).
#   For normalized vectors: inner_product(a, b) = cosine_similarity(a, b).
#   This lets us use FAISS's fast inner product search while
#   effectively computing cosine similarity.
# ============================================================

import pickle
from dataclasses import dataclass
from pathlib import Path

import faiss
import numpy as np

from src.chunker import Chunk
from src.config import EMBEDDINGS_DIR, TOP_K
from src.embedding_service import EmbeddingService
from src.faiss_manager import load_index
from src.utils import get_logger, timer

logger = get_logger(__name__)


# ============================================================
# RETRIEVAL RESULT DATACLASS
# ============================================================

@dataclass
class RetrievalResult:
    """
    Represents a single retrieval result (chunk + score).

    Attributes:
        chunk:  The retrieved Chunk object (with full metadata)
        score:  Similarity score from FAISS (higher = more similar)
        rank:   Rank position (1-based, 1 = most similar)
    """
    chunk: Chunk
    score: float
    rank: int


# ============================================================
# DENSE RETRIEVER CLASS
# ============================================================

class DenseRetriever:
    """
    Performs dense embedding retrieval over the clinical corpus.

    The retriever loads:
      - All chunks from chunks.pkl (the text + metadata)
      - Embeddings from embeddings.npy (the vectors)
      - A FAISS index by name (flat, ivf, or hnsw)

    At query time, it embeds the query and searches the index.

    Usage:
        retriever = DenseRetriever(index_name="flat")
        results = retriever.search("What is metformin used for?", top_k=10)
    """

    def __init__(self, index_name: str = "flat"):
        """
        Initialize the retriever by loading chunks, embeddings, and index.

        Args:
            index_name: Which FAISS index to use ("flat", "ivf", "hnsw").
        """
        # --- Load chunks ---
        chunks_path = EMBEDDINGS_DIR / "chunks.pkl"
        if not chunks_path.exists():
            raise FileNotFoundError(
                f"Chunks file not found: {chunks_path}. "
                f"Run 'python scripts/build_embeddings.py' first."
            )

        with open(chunks_path, "rb") as f:
            self.chunks: list[Chunk] = pickle.load(f)
        logger.info(f"Loaded {len(self.chunks)} chunks")

        # --- Load embeddings ---
        embeddings_path = EMBEDDINGS_DIR / "embeddings.npy"
        if not embeddings_path.exists():
            raise FileNotFoundError(
                f"Embeddings file not found: {embeddings_path}. "
                f"Run 'python scripts/build_embeddings.py' first."
            )

        self.embeddings: np.ndarray = np.load(embeddings_path)
        logger.info(f"Loaded embeddings: {self.embeddings.shape}")

        # --- Load FAISS index ---
        self.index_name = index_name
        self.index: faiss.Index = load_index(index_name)

        # --- Initialize embedding service for query embedding ---
        self.embedding_service = EmbeddingService()

    def search(
        self,
        query: str,
        top_k: int = TOP_K,
        allowed_indices: list[int] | None = None,
    ) -> list[RetrievalResult]:
        """
        Search for chunks most similar to the query.

        Args:
            query:           The user's question.
            top_k:           Number of results to return.
            allowed_indices: If provided, only search over these chunk indices.
                             Used by metadata filtering to restrict search space.
                             If None, searches all chunks.

        Returns:
            List of RetrievalResult objects, sorted by score (highest first).
        """
        # --- Step 1: Embed the query ---
        query_embedding = self.embedding_service.embed_query(query)

        # --- Step 2: Search FAISS ---
        if allowed_indices is not None and len(allowed_indices) < len(self.chunks):
            # Metadata filtering: search only allowed chunks
            results = self._search_with_filter(
                query_embedding, allowed_indices, top_k
            )
        else:
            # No filter: search all chunks
            scores, indices = self.index.search(query_embedding, top_k)
            results = self._build_results(scores[0], indices[0])

        logger.info(
            f"Retrieved {len(results)} results for query: "
            f"'{query[:50]}...' using {self.index_name} index"
        )

        return results

    def search_by_embedding(
        self,
        query_embedding: np.ndarray,
        top_k: int = TOP_K,
        allowed_indices: list[int] | None = None,
    ) -> list[RetrievalResult]:
        """
        Search using a pre-computed embedding (used by HyDE).

        Instead of embedding the query text, this accepts an
        already-computed embedding vector. This is needed for HyDE,
        which embeds a hypothetical document instead of the raw query.

        Args:
            query_embedding: numpy array of shape (1, dimension).
            top_k:           Number of results to return.
            allowed_indices: Optional filter for allowed chunk indices.

        Returns:
            List of RetrievalResult objects, sorted by score.
        """
        if allowed_indices is not None and len(allowed_indices) < len(self.chunks):
            return self._search_with_filter(
                query_embedding, allowed_indices, top_k
            )

        scores, indices = self.index.search(query_embedding, top_k)
        return self._build_results(scores[0], indices[0])

    def _search_with_filter(
        self,
        query_embedding: np.ndarray,
        allowed_indices: list[int],
        top_k: int,
    ) -> list[RetrievalResult]:
        """
        Search only the subset of chunks that pass metadata filters.

        WHY not use FAISS's built-in IDSelector?
            FAISS's IDSelector works for some index types but not all.
            A simpler, universal approach: create a temporary Flat index
            containing only the allowed vectors, search it, then map
            results back to original indices.

        For our corpus size (~1072 chunks), this is fast enough.

        Args:
            query_embedding: The query vector.
            allowed_indices: Indices of chunks that pass the metadata filter.
            top_k:           Number of results to return.

        Returns:
            List of RetrievalResult objects.
        """
        # Extract embeddings for allowed indices only
        allowed_indices_array = np.array(allowed_indices, dtype=np.int64)
        filtered_embeddings = self.embeddings[allowed_indices_array]

        # Create a temporary flat index for filtered vectors
        temp_index = faiss.IndexFlatIP(filtered_embeddings.shape[1])
        temp_index.add(filtered_embeddings)

        # Search the temporary index
        actual_k = min(top_k, len(allowed_indices))
        scores, local_indices = temp_index.search(query_embedding, actual_k)

        # Map local indices back to original chunk indices
        results: list[RetrievalResult] = []
        for rank, (score, local_idx) in enumerate(
            zip(scores[0], local_indices[0]), start=1
        ):
            if local_idx == -1:  # FAISS returns -1 for missing results
                continue

            original_idx = allowed_indices[local_idx]
            results.append(RetrievalResult(
                chunk=self.chunks[original_idx],
                score=float(score),
                rank=rank,
            ))

        return results

    def _build_results(
        self,
        scores: np.ndarray,
        indices: np.ndarray,
    ) -> list[RetrievalResult]:
        """
        Convert raw FAISS output into RetrievalResult objects.

        Args:
            scores:  Array of similarity scores from FAISS.
            indices: Array of chunk indices from FAISS.

        Returns:
            List of RetrievalResult objects.
        """
        results: list[RetrievalResult] = []

        for rank, (score, idx) in enumerate(zip(scores, indices), start=1):
            if idx == -1:  # FAISS returns -1 for missing results
                continue

            results.append(RetrievalResult(
                chunk=self.chunks[idx],
                score=float(score),
                rank=rank,
            ))

        return results

    def get_all_chunks(self) -> list[Chunk]:
        """Return all loaded chunks (used by metadata filter)."""
        return self.chunks

    def get_embeddings(self) -> np.ndarray:
        """Return all embeddings (used by MMR for diversity calc)."""
        return self.embeddings
