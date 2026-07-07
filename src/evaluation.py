# src/evaluation.py
# ============================================================
# Core Evaluation Engine
# ============================================================
# This module contains both the code-based metrics (MRR, Recall)
# and the LLM-as-a-judge metrics (Faithfulness, Relevancy, Precision)
# using the dedicated GPT-4.1 Judge model.

import json
import os
import pandas as pd
from openai import AzureOpenAI
from src.config import (
    AZURE_OPENAI_ENDPOINT,
    AZURE_OPENAI_API_KEY,
    AZURE_OPENAI_API_VERSION,
    AZURE_OPENAI_EMBEDDING_ENDPOINT,
    AZURE_OPENAI_EMBEDDING_KEY,
    AZURE_OPENAI_EMBEDDING_API_VERSION,
    AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME,
)
from src.utils import get_logger

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings

logger = get_logger(__name__)

# The dedicated judge deployment name from .env
JUDGE_DEPLOYMENT = os.getenv("AZURE_OPENAI_JUDGE_DEPLOYMENT_NAME")

def get_ragas_dataset() -> list[dict]:
    """Load the pre-generated RAGAS ground-truth dataset if it exists."""
    path = "evaluation/ragas_dataset.json"
    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                data = json.load(f)
            return data.get("testset", [])
        except Exception as e:
            logger.error(f"Failed to load {path}: {e}")
    return []

# ============================================================
# 1. LLM-AS-A-JUDGE METRICS (OFFICIAL RAGAS PACKAGE)
# ============================================================

def evaluate_with_ragas(question: str, answer: str, contexts: list[str]) -> dict:
    """
    Evaluates a generated answer using the official RAGAS library.
    If the question matches one in the evaluation/ragas_dataset.json,
    it uses the ground truth to calculate Context Precision.
    Otherwise, it calculates Faithfulness and Answer Relevancy only.
    """
    if not AZURE_OPENAI_API_KEY or not AZURE_OPENAI_ENDPOINT:
        raise ValueError("Missing Azure OpenAI credentials in .env")

    # 1. Initialize LangChain Wrappers for Azure OpenAI
    judge_llm = AzureChatOpenAI(
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        openai_api_version=AZURE_OPENAI_API_VERSION,
        openai_api_key=AZURE_OPENAI_API_KEY,
        azure_deployment=JUDGE_DEPLOYMENT,
        temperature=0.0
    )
    
    embeddings = AzureOpenAIEmbeddings(
        azure_endpoint=AZURE_OPENAI_EMBEDDING_ENDPOINT,
        openai_api_version=AZURE_OPENAI_EMBEDDING_API_VERSION,
        openai_api_key=AZURE_OPENAI_EMBEDDING_KEY,
        azure_deployment=AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME,
    )

    # 2. Check if we have a ground truth for this question
    testset = get_ragas_dataset()
    ground_truth = ""
    for test in testset:
        if test.get("question", "").strip().lower() == question.strip().lower():
            ground_truth = test.get("ground_truth", "")
            break

    # 3. Format as HuggingFace Dataset
    # Ragas requires 'contexts' to be a list of lists of strings
    data_samples = {
        "question": [question],
        "answer": [answer],
        "contexts": [contexts],
    }
    
    # Decide which metrics to run based on whether we have ground truth
    metrics_to_run = [faithfulness, answer_relevancy]
    if ground_truth:
        data_samples["ground_truth"] = [ground_truth]
        metrics_to_run.append(context_precision)
        logger.info("Found ground truth for question. Running full RAGAS suite including context_precision.")
    else:
        logger.info("No ground truth found for question. Running RAGAS without context_precision.")

    dataset = Dataset.from_dict(data_samples)

    # 4. Run the official evaluation suite
    try:
        result = evaluate(
            dataset,
            metrics=metrics_to_run,
            llm=judge_llm,
            embeddings=embeddings,
            raise_exceptions=False
        )
        return dict(result)
    except Exception as e:
        logger.error(f"RAGAS Evaluation failed: {e}")
        return {}

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
