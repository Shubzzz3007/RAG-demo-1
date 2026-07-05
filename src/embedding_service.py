# src/embedding_service.py
# ============================================================
# Embedding Service — Azure OpenAI
# ============================================================
# This module wraps the Azure OpenAI embeddings API to generate
# vector representations of text chunks.
#
# WHY a separate embedding service?
#   - Encapsulates all Azure OpenAI embedding logic in one place
#   - Handles batching (API has a limit on input size per call)
#   - Handles rate limiting with automatic retries
#   - Returns clean numpy arrays ready for FAISS
#
# IMPORTANT:
#   - Embeddings are generated ONCE during the offline pipeline
#     and saved to disk. They are never regenerated at runtime.
#   - At runtime, only the user's query is embedded (single call).
# ============================================================

import numpy as np
from openai import AzureOpenAI
from tqdm import tqdm

from src.config import (
    AZURE_OPENAI_EMBEDDING_ENDPOINT,
    AZURE_OPENAI_EMBEDDING_KEY,
    AZURE_OPENAI_EMBEDDING_API_VERSION,
    AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME,
    EMBEDDING_DIMENSION,
)
from src.utils import get_logger, timer

logger = get_logger(__name__)


# ============================================================
# EMBEDDING SERVICE CLASS
# ============================================================

class EmbeddingService:
    """
    Wrapper around Azure OpenAI Embeddings API.

    Usage:
        service = EmbeddingService()
        embeddings = service.embed_texts(["hello world", "clinical text"])
        query_embedding = service.embed_query("What is metformin?")
    """

    def __init__(self):
        """
        Initialize the Azure OpenAI embedding client.

        Reads credentials from config (which loads from .env).
        Validates that required settings are present.
        """
        # Validate that credentials are configured
        if not AZURE_OPENAI_EMBEDDING_ENDPOINT:
            raise ValueError(
                "AZURE_OPENAI_EMBEDDING_ENDPOINT not set in .env"
            )
        if not AZURE_OPENAI_EMBEDDING_KEY:
            raise ValueError(
                "AZURE_OPENAI_EMBEDDING_KEY not set in .env"
            )

        # Create the Azure OpenAI client for embeddings
        self.client = AzureOpenAI(
            azure_endpoint=AZURE_OPENAI_EMBEDDING_ENDPOINT,
            api_key=AZURE_OPENAI_EMBEDDING_KEY,
            api_version=AZURE_OPENAI_EMBEDDING_API_VERSION,
        )

        self.deployment_name = AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME
        self.dimension = EMBEDDING_DIMENSION

        logger.info(
            f"EmbeddingService initialized: "
            f"deployment={self.deployment_name}, "
            f"dimension={self.dimension}"
        )

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        """
        Embed a single batch of texts using the Azure OpenAI API.

        Args:
            texts: List of text strings to embed (max ~2048 recommended per batch).

        Returns:
            List of embedding vectors (each is a list of floats).

        Raises:
            Exception: If the API call fails after retries.
        """
        response = self.client.embeddings.create(
            input=texts,
            model=self.deployment_name,
        )

        # Extract embeddings in the same order as input
        embeddings = [item.embedding for item in response.data]
        return embeddings

    @timer
    def embed_texts(
        self,
        texts: list[str],
        batch_size: int = 100,
    ) -> np.ndarray:
        """
        Embed a list of texts in batches, returning a numpy array.

        WHY batching?
            Azure OpenAI has limits on the number of tokens per API call.
            Sending 100 texts at a time keeps us well under the limit
            while minimizing the number of API calls.

        Args:
            texts:      List of text strings to embed.
            batch_size: Number of texts per API call (default 100).

        Returns:
            numpy array of shape (n_texts, embedding_dimension).
            Vectors are L2-normalized for cosine similarity via inner product.
        """
        logger.info(f"Embedding {len(texts)} texts in batches of {batch_size}")

        all_embeddings: list[list[float]] = []

        # Process in batches with a progress bar
        for i in tqdm(range(0, len(texts), batch_size), desc="Embedding"):
            batch = texts[i : i + batch_size]
            batch_embeddings = self._embed_batch(batch)
            all_embeddings.extend(batch_embeddings)

        # Convert to numpy array
        embeddings_array = np.array(all_embeddings, dtype=np.float32)

        # L2-normalize so that inner product = cosine similarity
        # This is important for FAISS IndexFlatIP
        norms = np.linalg.norm(embeddings_array, axis=1, keepdims=True)
        # Avoid division by zero (shouldn't happen, but be safe)
        norms = np.maximum(norms, 1e-10)
        embeddings_array = embeddings_array / norms

        logger.info(
            f"Generated embeddings: shape={embeddings_array.shape}, "
            f"dtype={embeddings_array.dtype}"
        )

        return embeddings_array

    def embed_query(self, query: str) -> np.ndarray:
        """
        Embed a single query string.

        Used at runtime to embed the user's question before
        searching the FAISS index.

        Args:
            query: The user's question text.

        Returns:
            numpy array of shape (1, embedding_dimension), L2-normalized.
        """
        response = self.client.embeddings.create(
            input=[query],
            model=self.deployment_name,
        )

        embedding = np.array(
            [response.data[0].embedding], dtype=np.float32
        )

        # L2-normalize for cosine similarity
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        return embedding
