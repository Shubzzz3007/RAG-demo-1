# src/reranker.py
# ============================================================
# Cross-Encoder Reranker
# ============================================================
# This module re-scores retrieval results using a cross-encoder.
#
# WHY re-rank with a cross-encoder?
#   Bi-encoder (embedding) retrieval is fast but approximate:
#   - The query and document are embedded INDEPENDENTLY
#   - Similarity is computed by comparing two fixed-size vectors
#   - This misses fine-grained interactions between query and doc
#
#   A cross-encoder is more accurate but slower:
#   - It processes query AND document TOGETHER as one input
#   - The transformer attends to all token interactions
#   - This captures nuanced relevance (e.g., negation, context)
#   - But it's too slow to run on the entire corpus
#
# THE PIPELINE:
#   1. Bi-encoder retrieves top-k candidates (fast, approximate)
#   2. Cross-encoder re-scores just those k candidates (slow, accurate)
#   3. Re-rank by cross-encoder scores
#
# FOR EXPLAINABILITY:
#   We return BOTH the original bi-encoder rank and the new
#   cross-encoder rank, so the UI can show the re-ordering.
#
# MODEL: cross-encoder/ms-marco-MiniLM-L-6-v2
#   - Small (22M params), fast inference
#   - Trained on MS MARCO passage ranking dataset
#   - Good general-purpose relevance scoring
# ============================================================

from dataclasses import dataclass

from sentence_transformers import CrossEncoder

from src.config import CROSS_ENCODER_MODEL
from src.retriever import RetrievalResult
from src.utils import get_logger, timer

logger = get_logger(__name__)


# ============================================================
# RERANKED RESULT DATACLASS
# ============================================================

@dataclass
class RerankedResult:
    """
    A retrieval result with both original and cross-encoder scores.

    Attributes:
        chunk:           The retrieved chunk
        original_score:  Original bi-encoder similarity score
        reranker_score:  Cross-encoder relevance score
        rank:            New rank after reranking (1-based)
        original_rank:   Original rank before reranking (1-based)
    """
    chunk: object  # Chunk type
    original_score: float
    reranker_score: float
    rank: int
    original_rank: int


# ============================================================
# RERANKER CLASS
# ============================================================

class Reranker:
    """
    Cross-encoder reranker for improving retrieval quality.

    Usage:
        reranker = Reranker()
        reranked = reranker.rerank(query, results)
    """

    def __init__(self):
        """
        Load the cross-encoder model.

        The model is loaded once and reused for all queries.
        First load downloads the model (~80MB), subsequent loads
        use the cached version.
        """
        logger.info(f"Loading cross-encoder model: {CROSS_ENCODER_MODEL}")
        self.model = CrossEncoder(CROSS_ENCODER_MODEL)
        logger.info("Cross-encoder model loaded")

    @timer
    def rerank(
        self,
        query: str,
        results: list[RetrievalResult],
    ) -> list[RerankedResult]:
        """
        Re-rank retrieval results using the cross-encoder.

        Steps:
            1. Create (query, chunk_text) pairs
            2. Score all pairs with the cross-encoder
            3. Sort by cross-encoder score (highest first)
            4. Return results with both original and new scores

        Args:
            query:   The user's question.
            results: List of RetrievalResult from dense retrieval.

        Returns:
            List of RerankedResult, sorted by cross-encoder score.
        """
        if not results:
            return []

        # --- Step 1: Create pairs for scoring ---
        pairs = [(query, result.chunk.text) for result in results]

        # --- Step 2: Score with cross-encoder ---
        scores = self.model.predict(pairs)

        # --- Step 3: Combine with original results ---
        scored_results = []
        for result, ce_score in zip(results, scores):
            scored_results.append({
                "chunk": result.chunk,
                "original_score": result.score,
                "reranker_score": float(ce_score),
                "original_rank": result.rank,
            })

        # --- Step 4: Sort by cross-encoder score (descending) ---
        scored_results.sort(key=lambda x: x["reranker_score"], reverse=True)

        # --- Step 5: Assign new ranks ---
        reranked = [
            RerankedResult(
                chunk=item["chunk"],
                original_score=item["original_score"],
                reranker_score=item["reranker_score"],
                rank=new_rank,
                original_rank=item["original_rank"],
            )
            for new_rank, item in enumerate(scored_results, start=1)
        ]

        logger.info(
            f"Reranked {len(results)} results. "
            f"Top result changed: rank {reranked[0].original_rank} → 1"
        )

        return reranked
