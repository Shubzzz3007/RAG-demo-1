# src/metadata_filter.py
# ============================================================
# Metadata Filter — Pre-Retrieval Filtering
# ============================================================
# This module filters chunks by metadata BEFORE retrieval.
#
# WHY pre-retrieval filtering?
#   - Metadata filtering is NOT a retrieval algorithm
#   - It's a preprocessing step that narrows the search space
#   - It works with any retrieval strategy (dense, HyDE, etc.)
#   - Filtering before search is faster than filtering after
#
# HOW IT WORKS:
#   1. Takes the full list of chunks
#   2. Applies user-selected filters (doc_type, disease, specialty)
#   3. Returns the INDICES of chunks that pass all filters
#   4. These indices are passed to the retriever to restrict search
#
# FILTER LOGIC:
#   - Filters are AND-combined: a chunk must pass ALL active filters
#   - Within a multi-valued field (disease), the filter uses OR logic
#     (e.g., disease=["CKD", "diabetes"] matches chunks with either)
#   - "All" or empty selection means no filter (pass everything)
# ============================================================

from src.chunker import Chunk
from src.utils import get_logger

logger = get_logger(__name__)


# ============================================================
# METADATA FILTER CLASS
# ============================================================

class MetadataFilter:
    """
    Filters chunks by metadata fields before retrieval.

    Usage:
        mf = MetadataFilter(chunks)
        indices = mf.filter(
            doc_types=["guideline", "discharge_summary"],
            diseases=["heart failure"],
            specialties=["cardiology"]
        )
        # Pass 'indices' to retriever.search(allowed_indices=indices)
    """

    def __init__(self, chunks: list[Chunk]):
        """
        Initialize with the full list of chunks.

        Args:
            chunks: All chunks from the corpus.
        """
        self.chunks = chunks

        # Pre-compute available filter values for the UI dropdowns
        self._doc_types: set[str] = set()
        self._diseases: set[str] = set()
        self._specialties: set[str] = set()

        for chunk in chunks:
            if chunk.doc_type:
                self._doc_types.add(chunk.doc_type)
            if chunk.specialty:
                self._specialties.add(chunk.specialty)
            for disease in chunk.disease:
                self._diseases.add(disease)

        logger.info(
            f"MetadataFilter initialized: "
            f"{len(self._doc_types)} doc_types, "
            f"{len(self._diseases)} diseases, "
            f"{len(self._specialties)} specialties"
        )

    def filter(
        self,
        doc_types: list[str] | None = None,
        diseases: list[str] | None = None,
        specialties: list[str] | None = None,
    ) -> list[int]:
        """
        Return indices of chunks that match ALL specified filters.

        Filter logic:
          - None or empty list = no filter (everything passes)
          - doc_types: chunk.doc_type must be in the list
          - diseases: at least one of chunk.disease must be in the list
          - specialties: chunk.specialty must be in the list

        Args:
            doc_types:   List of allowed document types (or None for all).
            diseases:    List of allowed diseases (or None for all).
            specialties: List of allowed specialties (or None for all).

        Returns:
            List of chunk indices that pass all filters.
        """
        allowed_indices: list[int] = []

        for i, chunk in enumerate(self.chunks):
            # Check doc_type filter
            if doc_types and chunk.doc_type not in doc_types:
                continue

            # Check disease filter (OR logic: any match passes)
            if diseases:
                chunk_diseases = set(chunk.disease)
                filter_diseases = set(diseases)
                if not chunk_diseases.intersection(filter_diseases):
                    continue

            # Check specialty filter
            if specialties and chunk.specialty not in specialties:
                continue

            # Chunk passes all filters
            allowed_indices.append(i)

        logger.info(
            f"Metadata filter: {len(allowed_indices)}/{len(self.chunks)} "
            f"chunks passed (doc_types={doc_types}, "
            f"diseases={diseases}, specialties={specialties})"
        )

        return allowed_indices

    def get_available_doc_types(self) -> list[str]:
        """Return sorted list of unique document types in the corpus."""
        return sorted(self._doc_types)

    def get_available_diseases(self) -> list[str]:
        """Return sorted list of unique diseases in the corpus."""
        return sorted(self._diseases)

    def get_available_specialties(self) -> list[str]:
        """Return sorted list of unique specialties in the corpus."""
        return sorted(self._specialties)
