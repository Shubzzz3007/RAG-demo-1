# scripts/run_evals.py
# ============================================================
# Offline Evaluation Script
# ============================================================
# Runs the evaluation test cases across multiple configurations
# (e.g., chunking strategy, FAISS index type, enhancements).
# Exports results to evaluation/metrics_report.csv.

import os
import time
import pandas as pd
from tqdm import tqdm

from src.retriever import DenseRetriever
from src.mmr import apply_mmr
from src.reranker import Reranker
from src.llm import LLMClient
from src.prompts import SYSTEM_PROMPT, build_user_prompt
from src.config import TOP_K, TOP_K_FINAL
from evaluation.test_cases import TEST_CASES
from src.evaluation import (
    evaluate_faithfulness,
    evaluate_relevancy,
    evaluate_context_precision,
    calculate_mrr,
    calculate_precision_recall,
    calculate_trap_avoidance
)
from src.utils import get_logger

logger = get_logger(__name__)

# Configurations to test
CONFIGS = [
    {"chunking": "baseline", "index": "ivf", "mmr": False, "reranker": False},
    {"chunking": "recursive", "index": "ivf", "mmr": False, "reranker": False},
    {"chunking": "baseline", "index": "ivf", "mmr": True, "reranker": False},
    {"chunking": "baseline", "index": "ivf", "mmr": False, "reranker": True},
    {"chunking": "baseline", "index": "ivf", "mmr": True, "reranker": True},
]

def run_evaluations():
    print("🚀 Starting offline evaluation suite...")
    
    # Load shared resources
    llm = LLMClient()
    reranker = Reranker()
    
    results_list = []
    
    for config in CONFIGS:
        print(f"\n⚙️ Testing Config: {config}")
        
        chunking = config["chunking"]
        index_type = config["index"]
        use_mmr = config["mmr"]
        use_reranker = config["reranker"]
        
        # Load retriever for this config
        try:
            retriever = DenseRetriever(index_name=index_type, strategy=chunking)
        except Exception as e:
            print(f"Failed to load retriever for {chunking}/{index_type}: {e}")
            continue
            
        for idx, test_case in enumerate(tqdm(TEST_CASES, desc="Evaluating Queries")):
            query = test_case["query"]
            relevant_docs = test_case["relevant_doc_ids"]
            trap_docs = test_case["trap_doc_ids"]
            
            # 1. Retrieval
            start_time = time.time()
            raw_results = retriever.search(query=query, top_k=TOP_K)
            
            # 2. Enhancements
            enhanced_results = None
            if use_mmr:
                query_emb = retriever.embedding_service.embed_query(query)
                enhanced_results = apply_mmr(
                    results=raw_results,
                    query_embedding=query_emb,
                    chunk_embeddings=retriever.get_embeddings(),
                    chunks=retriever.get_all_chunks(),
                    top_k=TOP_K_FINAL
                )
            
            if use_reranker:
                results_to_rerank = enhanced_results if enhanced_results else raw_results[:TOP_K_FINAL]
                enhanced_results = reranker.rerank(query, results_to_rerank)
                
            # Final top chunks
            if use_reranker and enhanced_results:
                final_results = enhanced_results[:TOP_K_FINAL]
                final_chunks_for_llm = [
                    {"chunk": r.chunk, "score": r.reranker_score, "rank": r.rank} 
                    for r in final_results
                ]
            elif use_mmr and enhanced_results:
                final_results = enhanced_results[:TOP_K_FINAL]
                final_chunks_for_llm = [
                    {"chunk": r.chunk, "score": r.score, "rank": r.rank} 
                    for r in final_results
                ]
            else:
                final_results = raw_results[:TOP_K_FINAL]
                final_chunks_for_llm = [
                    {"chunk": r.chunk, "score": r.score, "rank": r.rank} 
                    for r in final_results
                ]
                
            retrieved_doc_ids = [r.chunk.doc_id for r in final_results]
            
            # 3. Code-based metrics
            mrr = calculate_mrr(retrieved_doc_ids, relevant_docs)
            precision, recall = calculate_precision_recall(retrieved_doc_ids, relevant_docs, TOP_K_FINAL)
            trap_avoidance = calculate_trap_avoidance(retrieved_doc_ids, trap_docs)
            
            # 4. Generate Answer
            user_prompt = build_user_prompt(query, final_chunks_for_llm)
            answer = llm.generate(user_prompt, SYSTEM_PROMPT)
            latency = time.time() - start_time
            
            # 5. LLM-as-a-judge metrics
            context_str = "\n\n".join([f"Document {r.chunk.doc_id}: {r.chunk.text}" for r in final_results])
            f_score, _ = evaluate_faithfulness(query, context_str, answer)
            r_score, _ = evaluate_relevancy(query, answer)
            cp_score, _ = evaluate_context_precision(query, context_str)
            
            # Append result
            results_list.append({
                "config_chunking": chunking,
                "config_index": index_type,
                "config_mmr": use_mmr,
                "config_reranker": use_reranker,
                "query": query,
                "mrr": mrr,
                "precision": precision,
                "recall": recall,
                "trap_avoidance": trap_avoidance,
                "faithfulness": f_score,
                "relevancy": r_score,
                "context_precision": cp_score,
                "latency": latency
            })
            
    # Save to CSV
    os.makedirs("evaluation", exist_ok=True)
    df = pd.DataFrame(results_list)
    out_path = "evaluation/metrics_report.csv"
    df.to_csv(out_path, index=False)
    print(f"\n✅ Evaluation complete. Results saved to {out_path}")

if __name__ == "__main__":
    run_evaluations()
