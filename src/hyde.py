# src/hyde.py
# ============================================================
# HyDE — Hypothetical Document Embeddings
# ============================================================
# HyDE is a retrieval enhancement technique.
#
# THE PROBLEM:
#   User queries are often short and use different vocabulary
#   than the documents they're looking for.
#   E.g., "Is metformin safe for kidney patients?" uses different
#   words than the guideline: "Metformin may be continued in
#   patients with eGFR above 30 mL/min/1.73m2..."
#
# HOW HYDE SOLVES THIS:
#   1. Ask the LLM to generate a HYPOTHETICAL answer to the query
#      (this answer may be inaccurate — that's fine)
#   2. Embed the hypothetical answer instead of the raw query
#   3. The hypothetical answer uses the same clinical vocabulary
#      as the actual documents, so embedding similarity is higher
#   4. Search FAISS with the hypothetical embedding
#
# FOR EXPLAINABILITY:
#   We return the hypothetical document so the UI can show
#   what HyDE generated and how it influenced retrieval.
#
# Reference: Gao et al., "Precise Zero-Shot Dense Retrieval
#            without Relevance Labels" (2022)
# ============================================================

from dataclasses import dataclass

import numpy as np

from src.embedding_service import EmbeddingService
from src.utils import get_logger

logger = get_logger(__name__)


# ============================================================
# HYDE RESULT DATACLASS
# ============================================================

@dataclass
class HyDEResult:
    """
    Result of HyDE processing.

    Attributes:
        hypothetical_document: The LLM-generated hypothetical answer
        hypothetical_embedding: Embedding of the hypothetical document
        original_query: The original user query (for reference)
    """
    hypothetical_document: str
    hypothetical_embedding: np.ndarray
    original_query: str


# ============================================================
# HYDE PROMPT
# ============================================================

HYDE_PROMPT = """You are a clinical knowledge assistant.
Given the following clinical question, write a short, factual paragraph
that would answer this question as if it appeared in a clinical document
(discharge summary, guideline, or medical abstract).

Do NOT say "I think" or "it is possible". Write as if you are stating
facts from a medical document. Keep the answer to 2-3 sentences.

Question: {query}

Hypothetical clinical document passage:"""


# ============================================================
# HYDE CLASS
# ============================================================

class HyDE:
    """
    Hypothetical Document Embeddings for improved retrieval.

    Usage:
        hyde = HyDE(llm_client, embedding_service)
        result = hyde.generate("Is metformin safe for CKD patients?")
        # Use result.hypothetical_embedding for FAISS search
    """

    def __init__(self, llm_client, embedding_service: EmbeddingService):
        """
        Initialize HyDE with an LLM client and embedding service.

        Args:
            llm_client:        The LLM client (src.llm.LLMClient)
            embedding_service: The embedding service for vectorizing
        """
        self.llm_client = llm_client
        self.embedding_service = embedding_service

    def generate(self, query: str) -> HyDEResult:
        """
        Generate a hypothetical document and embed it.

        Steps:
            1. Ask LLM to write a hypothetical clinical passage
            2. Embed the passage using Azure OpenAI embeddings
            3. Return both the passage and its embedding

        Args:
            query: The user's clinical question.

        Returns:
            HyDEResult with the hypothetical document and its embedding.
        """
        # --- Step 1: Generate hypothetical document ---
        prompt = HYDE_PROMPT.format(query=query)
        hypothetical_doc = self.llm_client.generate(
            user_message=prompt,
            system_message="You are a clinical document generator.",
        )

        logger.info(
            f"HyDE generated hypothetical document: "
            f"'{hypothetical_doc[:80]}...'"
        )

        # --- Step 2: Embed the hypothetical document ---
        hypothetical_embedding = self.embedding_service.embed_query(
            hypothetical_doc
        )

        return HyDEResult(
            hypothetical_document=hypothetical_doc,
            hypothetical_embedding=hypothetical_embedding,
            original_query=query,
        )
