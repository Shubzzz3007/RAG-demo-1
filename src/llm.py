# src/llm.py
# ============================================================
# LLM Client — Azure OpenAI GPT-4o
# ============================================================
# This module provides a thin wrapper around Azure OpenAI's
# Chat Completions API.
#
# WHY a separate LLM module?
#   - Single place to manage the API client
#   - Consistent interface for all callers (RAG chain, HyDE, etc.)
#   - Easy to swap LLM providers later if needed
#   - Handles error handling and logging in one place
#
# USED BY:
#   - src/generation (RAG answer generation)
#   - src/hyde.py (hypothetical document generation)
# ============================================================

from openai import AzureOpenAI

from src.config import (
    AZURE_OPENAI_ENDPOINT,
    AZURE_OPENAI_API_KEY,
    AZURE_OPENAI_API_VERSION,
    AZURE_OPENAI_DEPLOYMENT_NAME,
    LLM_TEMPERATURE,
    LLM_MAX_TOKENS,
)
from src.utils import get_logger

logger = get_logger(__name__)


# ============================================================
# LLM CLIENT CLASS
# ============================================================

class LLMClient:
    """
    Wrapper around Azure OpenAI Chat Completions API.

    Usage:
        llm = LLMClient()
        response = llm.generate(
            user_message="What is metformin?",
            system_message="You are a clinical assistant."
        )
    """

    def __init__(self):
        """
        Initialize the Azure OpenAI client.

        Reads credentials from config (which loads from .env).
        """
        if not AZURE_OPENAI_ENDPOINT:
            raise ValueError("AZURE_OPENAI_ENDPOINT not set in .env")
        if not AZURE_OPENAI_API_KEY:
            raise ValueError("AZURE_OPENAI_API_KEY not set in .env")

        self.client = AzureOpenAI(
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
            api_key=AZURE_OPENAI_API_KEY,
            api_version=AZURE_OPENAI_API_VERSION,
        )

        self.deployment_name = AZURE_OPENAI_DEPLOYMENT_NAME

        logger.info(f"LLMClient initialized: deployment={self.deployment_name}")

    def generate(
        self,
        user_message: str,
        system_message: str = "You are a helpful assistant.",
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """
        Generate a response from the LLM.

        Args:
            user_message:   The user's prompt / question.
            system_message: The system prompt (sets behavior/role).
            temperature:    Override default temperature (0.0-2.0).
            max_tokens:     Override default max response length.

        Returns:
            The LLM's response text.

        Raises:
            Exception: If the API call fails.
        """
        # Use defaults from config if not overridden
        temp = temperature if temperature is not None else LLM_TEMPERATURE
        tokens = max_tokens if max_tokens is not None else LLM_MAX_TOKENS

        response = self.client.chat.completions.create(
            model=self.deployment_name,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message},
            ],
            temperature=temp,
            max_tokens=tokens,
        )

        # Extract the response text
        reply = response.choices[0].message.content.strip()

        logger.info(
            f"LLM response: {len(reply)} chars "
            f"(temp={temp}, max_tokens={tokens})"
        )

        return reply
