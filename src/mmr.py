# src/mmr.py
# ============================================================
# MMR — Maximal Marginal Relevance
# ============================================================
# MMR is a re-ranking technique that balances RELEVANCE and DIVERSITY.
#
# THE PROBLEM:
#   Dense retrieval often returns very similar chunks.
#   For example, if you search for "metformin and CKD", the top 5
#   results might all be discharge summaries saying similar things.
#   You'd miss the guideline and PubMed evidence.
#
# HOW MMR SOLVES THIS:
#   Instead of just taking the top-k by similarity score, MMR
#   iteratively selects chunks that are:
#     - Relevant to the query (high similarity)
#     - Different from already-selected chunks (high diversity)
#
#   The balance is controlled by lambda:
#     lambda = 1.0 → pure relevance (same as top-k)
#     lambda = 0.0 → pure diversity (most different chunks)
#     lambda = 0.7 → good balance (our default)
#
# FORMULA:
#   MMR(d) = lambda * Sim(d, query) - (1 - lambda) * max(Sim(d, selected))
#
# FOR EXPLAINABILITY:
#   We return the MMR score alongside the original similarity score,
#   so the UI can show why chunks were re-ordered.
# ============================================================

from dataclasses import dataclass

import numpy as np

from src.retriever import RetrievalResult
from src.config import MMR_LAMBDA
from src.utils import get_logger

logger = get_logger(__name__)


# ============================================================
# MMR RESULT DATACLASS
# ============================================================

@dataclass
class MMRResult:
    """
    A retrieval result with both original and MMR scores.

    Attributes:
        chunk:        The retrieved chunk
        score:        Original similarity score from FAISS
        mmr_score:    MMR-adjusted score (relevance - redundancy)
        rank:         New rank after MMR reordering (1-based)
        original_rank: Original rank before MMR (1-based)
    """
    chunk: object  # Chunk type (avoid circular import)
    score: float
    mmr_score: float
    rank: int
    original_rank: int


# ============================================================
# MMR FUNCTION
# ============================================================

def apply_mmr(
    results: list[RetrievalResult],
    query_embedding: np.ndarray,
    chunk_embeddings: np.ndarray,
    chunks: list,
    top_k: int = 5,
    lambda_param: float = MMR_LAMBDA,
) -> list[MMRResult]:
    """
    Re-rank retrieval results using Maximal Marginal Relevance.

    Args:
        results:          List of RetrievalResult from dense retrieval.
        query_embedding:  The query vector, shape (1, dimension).
        chunk_embeddings: ALL chunk embeddings, shape (n_chunks, dimension).
        chunks:           ALL chunks list (to map indices).
        top_k:            Number of results to return after MMR.
        lambda_param:     Balance between relevance and diversity.
                          1.0 = pure relevance, 0.0 = pure diversity.

    Returns:
        List of MMRResult objects, re-ranked by MMR score.
    """
    if not results:
        return []

    # --- Step 1: Get the candidate chunk indices and their embeddings ---
    # Map chunk_id → index in the full chunks list
    chunk_id_to_index = {chunk.chunk_id: i for i, chunk in enumerate(chunks)}

    candidate_indices = []
    candidate_scores = {}  # chunk_id → original score
    original_ranks = {}    # chunk_id → original rank

    for result in results:
        idx = chunk_id_to_index.get(result.chunk.chunk_id)
        if idx is not None:
            candidate_indices.append(idx)
            candidate_scores[result.chunk.chunk_id] = result.score
            original_ranks[result.chunk.chunk_id] = result.rank

    if not candidate_indices:
        return []

    # Get embeddings for candidate chunks
    candidate_embeddings = chunk_embeddings[candidate_indices]

    # --- Step 2: Compute similarity to query ---
    # query_embedding shape: (1, dim), candidate shape: (n, dim)
    query_sim = np.dot(candidate_embeddings, query_embedding.T).flatten()

    # --- Step 3: Iteratively select chunks using MMR ---
    selected: list[int] = []          # Indices into candidate_indices
    remaining = list(range(len(candidate_indices)))

    mmr_results: list[MMRResult] = []

    for rank in range(1, min(top_k, len(candidate_indices)) + 1):
        best_idx = -1
        best_mmr_score = -float("inf")

        for idx in remaining:
            # Relevance: similarity to query
            relevance = query_sim[idx]

            # Diversity: maximum similarity to already-selected chunks
            if selected:
                selected_embeddings = candidate_embeddings[selected]
                chunk_embedding = candidate_embeddings[idx].reshape(1, -1)
                similarities = np.dot(selected_embeddings, chunk_embedding.T).flatten()
                max_sim_to_selected = float(np.max(similarities))
            else:
                max_sim_to_selected = 0.0

            # MMR formula
            mmr_score = (
                lambda_param * relevance
                - (1 - lambda_param) * max_sim_to_selected
            )

            if mmr_score > best_mmr_score:
                best_mmr_score = mmr_score
                best_idx = idx

        if best_idx == -1:
            break

        # Add the best chunk to selected
        selected.append(best_idx)
        remaining.remove(best_idx)

        # Build result
        original_chunk_idx = candidate_indices[best_idx]
        chunk = chunks[original_chunk_idx]

        mmr_results.append(MMRResult(
            chunk=chunk,
            score=candidate_scores.get(chunk.chunk_id, 0.0),
            mmr_score=best_mmr_score,
            rank=rank,
            original_rank=original_ranks.get(chunk.chunk_id, 0),
        ))

    logger.info(
        f"MMR re-ranked {len(results)} results → {len(mmr_results)} "
        f"(lambda={lambda_param})"
    )

    return mmr_results
