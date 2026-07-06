# src/evaluation.py
# ============================================================
# Core Evaluation Engine
# ============================================================
# This module contains both the code-based metrics (MRR, Recall)
# and the LLM-as-a-judge metrics (Faithfulness, Relevancy, Precision)
# using the dedicated GPT-4.1 Judge model.

import json
import os
from openai import AzureOpenAI
from src.config import (
    AZURE_OPENAI_ENDPOINT,
    AZURE_OPENAI_API_KEY,
    AZURE_OPENAI_API_VERSION,
)
from src.utils import get_logger

logger = get_logger(__name__)

# The dedicated judge deployment name from .env
JUDGE_DEPLOYMENT = os.getenv("AZURE_OPENAI_JUDGE_DEPLOYMENT_NAME", "gpt-4o")

def get_judge_client() -> AzureOpenAI:
    """Initialize the Azure OpenAI client for the Judge model."""
    if not AZURE_OPENAI_API_KEY or not AZURE_OPENAI_ENDPOINT:
        raise ValueError("Missing Azure OpenAI credentials in .env")

    return AzureOpenAI(
        api_key=AZURE_OPENAI_API_KEY,
        api_version=AZURE_OPENAI_API_VERSION,
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
    )

# ============================================================
# 1. LLM-AS-A-JUDGE METRICS (RAGAS INSPIRED)
# ============================================================

def evaluate_faithfulness(question: str, context: str, answer: str) -> tuple[float, str]:
    """
    Check if the answer is completely backed by the context.
    Returns: (score 0.0 to 1.0, reasoning)
    """
    client = get_judge_client()
    prompt = f"""
You are an expert clinical evaluator. Your task is to check if the generated answer is faithful to the provided context.
If the answer contains any claims not supported by the context, penalize the score.

Question: {question}
Context: {context}
Answer: {answer}

Respond ONLY with a valid JSON object containing exactly two keys:
- "score": a float between 0.0 and 1.0 (1.0 = fully faithful, 0.0 = completely hallucinated).
- "reasoning": a brief explanation of why this score was given.
"""
    try:
        response = client.chat.completions.create(
            model=JUDGE_DEPLOYMENT,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            response_format={"type": "json_object"}
        )
        result = json.loads(response.choices[0].message.content)
        return float(result.get("score", 0.0)), result.get("reasoning", "No reasoning provided.")
    except Exception as e:
        logger.error(f"Error evaluating faithfulness: {e}")
        return 0.0, f"Error: {e}"

def evaluate_relevancy(question: str, answer: str) -> tuple[float, str]:
    """
    Check if the answer directly addresses the user's question.
    Returns: (score 0.0 to 1.0, reasoning)
    """
    client = get_judge_client()
    prompt = f"""
You are an expert clinical evaluator. Your task is to check if the generated answer directly addresses the question asked.
Penalize answers that are off-topic, evasive, or overly verbose with irrelevant information.

Question: {question}
Answer: {answer}

Respond ONLY with a valid JSON object containing exactly two keys:
- "score": a float between 0.0 and 1.0 (1.0 = perfectly relevant, 0.0 = completely irrelevant).
- "reasoning": a brief explanation of why this score was given.
"""
    try:
        response = client.chat.completions.create(
            model=JUDGE_DEPLOYMENT,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            response_format={"type": "json_object"}
        )
        result = json.loads(response.choices[0].message.content)
        return float(result.get("score", 0.0)), result.get("reasoning", "No reasoning provided.")
    except Exception as e:
        logger.error(f"Error evaluating relevancy: {e}")
        return 0.0, f"Error: {e}"

def evaluate_context_precision(question: str, context: str) -> tuple[float, str]:
    """
    Check if the retrieved context contains the information needed to answer the question.
    Returns: (score 0.0 to 1.0, reasoning)
    """
    client = get_judge_client()
    prompt = f"""
You are an expert clinical evaluator. Your task is to evaluate the precision of the retrieved context.
Does the context actually contain the information necessary to answer the question?

Question: {question}
Context: {context}

Respond ONLY with a valid JSON object containing exactly two keys:
- "score": a float between 0.0 and 1.0 (1.0 = contains exact answer, 0.5 = partial/noisy, 0.0 = completely useless).
- "reasoning": a brief explanation of why this score was given.
"""
    try:
        response = client.chat.completions.create(
            model=JUDGE_DEPLOYMENT,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            response_format={"type": "json_object"}
        )
        result = json.loads(response.choices[0].message.content)
        return float(result.get("score", 0.0)), result.get("reasoning", "No reasoning provided.")
    except Exception as e:
        logger.error(f"Error evaluating context precision: {e}")
        return 0.0, f"Error: {e}"


# ============================================================
# 2. CODE-BASED METRICS (MATHEMATICAL)
# ============================================================

def calculate_mrr(retrieved_doc_ids: list[str], relevant_doc_ids: list[str]) -> float:
    """Calculate Mean Reciprocal Rank (1/rank of first relevant document)."""
    for rank, doc_id in enumerate(retrieved_doc_ids, start=1):
        if doc_id in relevant_doc_ids:
            return 1.0 / rank
    return 0.0

def calculate_precision_recall(retrieved_doc_ids: list[str], relevant_doc_ids: list[str], k: int) -> tuple[float, float]:
    """Calculate Precision@k and Recall@k."""
    if not relevant_doc_ids:
        return 0.0, 0.0
        
    top_k = retrieved_doc_ids[:k]
    relevant_retrieved = len([doc for doc in top_k if doc in relevant_doc_ids])
    
    precision = relevant_retrieved / k if k > 0 else 0.0
    recall = relevant_retrieved / len(relevant_doc_ids) if len(relevant_doc_ids) > 0 else 0.0
    
    return precision, recall

def calculate_trap_avoidance(retrieved_doc_ids: list[str], trap_doc_ids: list[str]) -> float:
    """Calculate Trap Avoidance Rate (1.0 = hit no traps, 0.0 = hit a trap)."""
    if not trap_doc_ids:
        return 1.0
    
    for doc in retrieved_doc_ids:
        if doc in trap_doc_ids:
            return 0.0 # Failed, hit a trap
            
    return 1.0 # Success, hit no traps
