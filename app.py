# app.py
# ============================================================
# Explainable Clinical RAG — Streamlit Demo UI
# ============================================================
# This is the main entry point for the Streamlit application.
#
# ARCHITECTURE:
#   The app loads all components once at startup using
#   @st.cache_resource. This means:
#     - FAISS indexes are loaded once (not on every query)
#     - Chunks/embeddings are loaded once
#     - Cross-encoder model is loaded once
#     - Only the selected FAISS index is loaded
#
# UI LAYOUT:
#   Sidebar: Retrieval settings + metadata filters
#   Main:    Question → Settings → Results → Evidence → Answer
#
# PIPELINE FLOW:
#   1. User enters a question
#   2. Apply metadata filters (if any) → get allowed_indices
#   3. If HyDE is ON → generate hypothetical doc → embed it
#   4. Search FAISS with query (or HyDE) embedding
#   5. If MMR is ON → re-rank for diversity
#   6. If Cross-Encoder is ON → re-rank with cross-encoder
#   7. Send top chunks to LLM → generate answer
#   8. Display everything with full explainability
# ============================================================

import streamlit as st
import time

from src.retriever import DenseRetriever, RetrievalResult
from src.metadata_filter import MetadataFilter
from src.hyde import HyDE
from src.mmr import apply_mmr
from src.reranker import Reranker
from src.llm import LLMClient
from src.embedding_service import EmbeddingService
from src.prompts import SYSTEM_PROMPT, build_user_prompt
from src.config import TOP_K, TOP_K_FINAL
from src.utils import get_logger

logger = get_logger(__name__)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Explainable Clinical RAG",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CUSTOM STYLING
# ============================================================

st.markdown("""
<style>
    /* Main title styling */
    .main-title {
        font-size: 2rem;
        font-weight: 700;
        color: #1E88E5;
        margin-bottom: 0.5rem;
    }

    /* Confidence badges */
    .confidence-high {
        background-color: #43A047;
        color: white;
        padding: 4px 12px;
        border-radius: 12px;
        font-weight: 600;
        font-size: 0.9rem;
    }
    .confidence-medium {
        background-color: #FB8C00;
        color: white;
        padding: 4px 12px;
        border-radius: 12px;
        font-weight: 600;
        font-size: 0.9rem;
    }
    .confidence-low {
        background-color: #E53935;
        color: white;
        padding: 4px 12px;
        border-radius: 12px;
        font-weight: 600;
        font-size: 0.9rem;
    }

    /* Settings badge */
    .setting-badge {
        background-color: #E3F2FD;
        color: #1565C0;
        padding: 3px 10px;
        border-radius: 8px;
        font-size: 0.85rem;
        margin-right: 6px;
        display: inline-block;
        margin-bottom: 4px;
    }

    /* Document type badges */
    .doc-type-discharge {
        background-color: #E8F5E9;
        color: #2E7D32;
        padding: 2px 8px;
        border-radius: 6px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .doc-type-guideline {
        background-color: #FFF3E0;
        color: #E65100;
        padding: 2px 8px;
        border-radius: 6px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .doc-type-pubmed {
        background-color: #F3E5F5;
        color: #6A1B9A;
        padding: 2px 8px;
        border-radius: 6px;
        font-size: 0.8rem;
        font-weight: 600;
    }

    /* Chunk card */
    .chunk-card {
        border: 1px solid #E0E0E0;
        border-radius: 8px;
        padding: 12px;
        margin-bottom: 8px;
        background-color: #FAFAFA;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# CACHED RESOURCE LOADING
# ============================================================
# These functions load heavy resources once and cache them.
# They only re-run if the input parameters change.
# ============================================================

@st.cache_resource
def load_retriever(index_name: str, strategy: str) -> DenseRetriever:
    """Load the dense retriever with the specified FAISS index and strategy."""
    return DenseRetriever(index_name=index_name, strategy=strategy)


@st.cache_resource
def load_llm() -> LLMClient:
    """Load the LLM client."""
    return LLMClient()


@st.cache_resource
def load_embedding_service() -> EmbeddingService:
    """Load the embedding service."""
    return EmbeddingService()


@st.cache_resource
def load_reranker() -> Reranker:
    """Load the cross-encoder reranker model."""
    return Reranker()


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def get_doc_type_badge(doc_type: str) -> str:
    """Return an HTML badge for the document type."""
    badges = {
        "discharge_summary": '<span class="doc-type-discharge">📋 Discharge Summary</span>',
        "guideline": '<span class="doc-type-guideline">📖 Guideline</span>',
        "pubmed_abstract": '<span class="doc-type-pubmed">📄 PubMed Abstract</span>',
    }
    return badges.get(doc_type, doc_type)


def extract_confidence(answer: str) -> str:
    """Extract the confidence level from the LLM's answer."""
    answer_lower = answer.lower()
    if "**confidence: high**" in answer_lower or "confidence: high" in answer_lower:
        return "HIGH"
    elif "**confidence: medium**" in answer_lower or "confidence: medium" in answer_lower:
        return "MEDIUM"
    elif "**confidence: low**" in answer_lower or "confidence: low" in answer_lower:
        return "LOW"
    return "UNKNOWN"


def render_confidence_badge(confidence: str) -> str:
    """Return an HTML confidence badge."""
    badges = {
        "HIGH": '<span class="confidence-high">✅ HIGH</span>',
        "MEDIUM": '<span class="confidence-medium">⚠️ MEDIUM</span>',
        "LOW": '<span class="confidence-low">❌ LOW</span>',
        "UNKNOWN": '<span class="confidence-low">❓ UNKNOWN</span>',
    }
    return badges.get(confidence, confidence)


# ============================================================
# SIDEBAR — Retrieval Settings & Metadata Filters
# ============================================================

st.sidebar.markdown("## 🔧 Retrieval Settings")

# Chunking Strategy Selection
chunking_strategy = st.sidebar.selectbox(
    "Chunking Strategy",
    options=["baseline", "recursive"],
    format_func=lambda x: "Baseline (Section-Aware)" if x == "baseline" else "Recursive Overlap (400 chars)",
    index=0,
    help="Baseline = Section-aware splitting, Recursive = Fixed size sliding window",
)

# FAISS Index Selection
faiss_index = st.sidebar.selectbox(
    "FAISS Index",
    options=["flat", "ivf", "hnsw"],
    index=0,
    help="Flat = exact search, IVF = approximate (faster), HNSW = graph-based (fastest)",
)

st.sidebar.markdown("---")
st.sidebar.markdown("### ⚡ Retrieval Enhancements")

# HyDE Toggle
use_hyde = st.sidebar.checkbox(
    "HyDE (Hypothetical Document Embeddings)",
    value=False,
    help="Generate a hypothetical answer, embed it, and use that for retrieval",
)

# MMR Toggle
use_mmr = st.sidebar.checkbox(
    "MMR (Maximal Marginal Relevance)",
    value=False,
    help="Re-rank results for diversity — balance relevance vs. variety",
)

# Cross-Encoder Toggle
use_reranker = st.sidebar.checkbox(
    "Cross-Encoder Reranking",
    value=False,
    help="Re-score results with a cross-encoder for better relevance ranking",
)

# --- Metadata Filters ---
st.sidebar.markdown("---")
st.sidebar.markdown("### 🏷️ Metadata Filters")

# Load retriever to get available filter values
retriever = load_retriever(faiss_index, chunking_strategy)
metadata_filter = MetadataFilter(retriever.get_all_chunks())

# Document Type filter
available_doc_types = metadata_filter.get_available_doc_types()
doc_type_labels = {
    "discharge_summary": "Discharge Summary",
    "guideline": "Guideline",
    "pubmed_abstract": "PubMed Abstract",
}
selected_doc_types = st.sidebar.multiselect(
    "Document Type",
    options=available_doc_types,
    format_func=lambda x: doc_type_labels.get(x, x),
    default=[],
    help="Filter by document type (empty = all)",
)

# Disease filter
available_diseases = metadata_filter.get_available_diseases()
selected_diseases = st.sidebar.multiselect(
    "Disease",
    options=available_diseases,
    default=[],
    help="Filter by disease (empty = all)",
)

# Specialty filter
available_specialties = metadata_filter.get_available_specialties()
selected_specialties = st.sidebar.multiselect(
    "Specialty",
    options=available_specialties,
    default=[],
    help="Filter by medical specialty (empty = all)",
)


# ============================================================
# MAIN PANEL — Title & Question Input
# ============================================================

st.markdown('<div class="main-title">🏥 Explainable Clinical RAG</div>', unsafe_allow_html=True)
st.markdown(
    "Ask clinical questions over discharge summaries, guidelines, and PubMed abstracts. "
    "See exactly **why** each document was retrieved and **how confident** the answer is."
)

# Example questions
st.markdown("##### 💡 Example Questions")
example_cols = st.columns(2)
examples = [
    "What discharge instructions are recommended for a diabetic patient with heart failure?",
    "What are the contraindications for metformin in patients with renal impairment?",
    "Summarize the guidelines for beta-blocker use in heart failure.",
    "What is the recommended eGFR threshold for metformin use?",
]

# Create clickable example buttons
for i, example in enumerate(examples):
    col = example_cols[i % 2]
    if col.button(example, key=f"example_{i}", use_container_width=True):
        st.session_state["question"] = example

# Question input
question = st.text_area(
    "Enter your clinical question:",
    value=st.session_state.get("question", ""),
    height=80,
    placeholder="e.g., What are the contraindications for metformin in CKD patients?",
)

# ============================================================
# RUN PIPELINE
# ============================================================

if st.button("🔍 Search & Answer", type="primary", use_container_width=True):
    if not question.strip():
        st.warning("Please enter a question.")
        st.stop()

    # --- Track timing ---
    timings: dict[str, float] = {}

    # ==========================================================
    # Step 1: Applied Settings Display
    # ==========================================================
    st.markdown("---")
    st.markdown("### ⚙️ Applied Retrieval Settings")

    settings_html = f'<span class="setting-badge">📊 FAISS: {faiss_index.upper()}</span>'
    settings_html += f'<span class="setting-badge">🧩 Chunking: {chunking_strategy.title()}</span>'
    if use_hyde:
        settings_html += '<span class="setting-badge">🔮 HyDE: ON</span>'
    if use_mmr:
        settings_html += '<span class="setting-badge">🎯 MMR: ON</span>'
    if use_reranker:
        settings_html += '<span class="setting-badge">⚖️ Cross-Encoder: ON</span>'
    if selected_doc_types:
        labels = [doc_type_labels.get(dt, dt) for dt in selected_doc_types]
        settings_html += f'<span class="setting-badge">📁 Types: {", ".join(labels)}</span>'
    if selected_diseases:
        settings_html += f'<span class="setting-badge">🩺 Disease: {", ".join(selected_diseases)}</span>'
    if selected_specialties:
        settings_html += f'<span class="setting-badge">🏥 Specialty: {", ".join(selected_specialties)}</span>'

    st.markdown(settings_html, unsafe_allow_html=True)

    # ==========================================================
    # Step 2: Metadata Filtering
    # ==========================================================
    with st.spinner("Applying metadata filters..."):
        t0 = time.time()
        allowed_indices = None

        if selected_doc_types or selected_diseases or selected_specialties:
            allowed_indices = metadata_filter.filter(
                doc_types=selected_doc_types or None,
                diseases=selected_diseases or None,
                specialties=selected_specialties or None,
            )
            st.info(
                f"📂 Metadata filter: **{len(allowed_indices)}** / "
                f"{len(retriever.get_all_chunks())} chunks match filters"
            )

        timings["metadata_filter"] = time.time() - t0

    # ==========================================================
    # Step 3: HyDE (if enabled)
    # ==========================================================
    hyde_result = None
    search_embedding = None

    if use_hyde:
        with st.spinner("Generating hypothetical document (HyDE)..."):
            t0 = time.time()
            llm = load_llm()
            embedding_service = load_embedding_service()
            hyde_module = HyDE(llm, embedding_service)
            hyde_result = hyde_module.generate(question)
            search_embedding = hyde_result.hypothetical_embedding
            timings["hyde"] = time.time() - t0

    # ==========================================================
    # Step 4: Dense Retrieval
    # ==========================================================
    with st.spinner(f"Searching {faiss_index.upper()} index..."):
        t0 = time.time()

        if search_embedding is not None:
            # HyDE: use hypothetical embedding
            results = retriever.search_by_embedding(
                query_embedding=search_embedding,
                top_k=TOP_K,
                allowed_indices=allowed_indices,
            )
        else:
            # Standard: embed the raw query
            results = retriever.search(
                query=question,
                top_k=TOP_K,
                allowed_indices=allowed_indices,
            )

        timings["retrieval"] = time.time() - t0

    # Save pre-enhancement results for comparison
    pre_enhancement_results = list(results)

    # ==========================================================
    # Step 5: MMR (if enabled)
    # ==========================================================
    mmr_results = None
    if use_mmr and results:
        with st.spinner("Applying MMR for diversity..."):
            t0 = time.time()
            query_emb = (
                search_embedding
                if search_embedding is not None
                else retriever.embedding_service.embed_query(question)
            )
            mmr_results = apply_mmr(
                results=results,
                query_embedding=query_emb,
                chunk_embeddings=retriever.get_embeddings(),
                chunks=retriever.get_all_chunks(),
                top_k=TOP_K_FINAL,
            )
            timings["mmr"] = time.time() - t0

    # ==========================================================
    # Step 6: Cross-Encoder Reranking (if enabled)
    # ==========================================================
    reranked_results = None
    if use_reranker and results:
        with st.spinner("Reranking with cross-encoder..."):
            t0 = time.time()
            reranker = load_reranker()

            # Rerank the MMR results if MMR was applied,
            # otherwise rerank the raw retrieval results
            if mmr_results:
                # Convert MMR results back to RetrievalResult for reranker
                results_to_rerank = [
                    RetrievalResult(
                        chunk=r.chunk,
                        score=r.score,
                        rank=r.rank,
                    )
                    for r in mmr_results
                ]
            else:
                results_to_rerank = results[:TOP_K_FINAL]

            reranked_results = reranker.rerank(question, results_to_rerank)
            timings["reranking"] = time.time() - t0

    # ==========================================================
    # Step 7: Prepare final chunks for LLM
    # ==========================================================
    # Determine which results to use for answer generation
    # Priority: reranked > MMR > raw results
    final_chunks_for_llm: list[dict] = []

    if reranked_results:
        for r in reranked_results[:TOP_K_FINAL]:
            final_chunks_for_llm.append({
                "chunk": r.chunk,
                "score": r.reranker_score,
                "rank": r.rank,
            })
    elif mmr_results:
        for r in mmr_results[:TOP_K_FINAL]:
            final_chunks_for_llm.append({
                "chunk": r.chunk,
                "score": r.score,
                "rank": r.rank,
            })
    else:
        for r in results[:TOP_K_FINAL]:
            final_chunks_for_llm.append({
                "chunk": r.chunk,
                "score": r.score,
                "rank": r.rank,
            })

    # ==========================================================
    # Step 8: LLM Answer Generation
    # ==========================================================
    with st.spinner("Generating answer..."):
        t0 = time.time()
        llm = load_llm()
        user_prompt = build_user_prompt(question, final_chunks_for_llm)
        answer = llm.generate(
            user_message=user_prompt,
            system_message=SYSTEM_PROMPT,
        )
        timings["llm"] = time.time() - t0

    # Store results in session state so they persist when 'Evaluate' is clicked
    st.session_state["search_result"] = {
        "question": question,
        "answer": answer,
        "final_chunks_for_llm": final_chunks_for_llm,
        "hyde_result": hyde_result,
        "pre_enhancement_results": pre_enhancement_results,
        "reranked_results": reranked_results,
        "mmr_results": mmr_results,
        "timings": timings,
    }


# ============================================================
# DISPLAY RESULTS
# ============================================================
if "search_result" in st.session_state:
    res = st.session_state["search_result"]
    question = res["question"]
    answer = res["answer"]
    final_chunks_for_llm = res["final_chunks_for_llm"]
    hyde_result = res["hyde_result"]
    pre_enhancement_results = res["pre_enhancement_results"]
    reranked_results = res["reranked_results"]
    mmr_results = res["mmr_results"]
    timings = res["timings"]

    st.markdown("---")

    # --- Generated Answer ---
    st.markdown("### 💡 Generated Answer")

    # Extract and display confidence
    confidence = extract_confidence(answer)
    st.markdown(
        f"**Confidence:** {render_confidence_badge(confidence)}",
        unsafe_allow_html=True,
    )

    # Display the answer
    st.markdown(answer)

    # --- Evaluate this Response ---
    st.markdown("---")
    st.markdown("### 📊 Evaluate this Response")
    if st.button("Evaluate Answer (GPT-4.1 Judge)", type="secondary"):
        with st.spinner("Running Evaluations..."):
            from src.evaluation import (
                evaluate_faithfulness, 
                evaluate_relevancy, 
                evaluate_context_precision,
                calculate_mrr,
                calculate_precision_recall,
                calculate_trap_avoidance
            )
            from evaluation.test_cases import TEST_CASES
            
            # Prepare context string
            context_str = "\n\n".join([f"Document {item['chunk'].doc_id}: {item['chunk'].text}" for item in final_chunks_for_llm])
            
            # Run RAGAS metrics (Reference-free, can run on any query)
            f_score, f_reason = evaluate_faithfulness(question, context_str, answer)
            r_score, r_reason = evaluate_relevancy(question, answer)
            cp_score, cp_reason = evaluate_context_precision(question, context_str)
            
            # Try to match the query to our ground truth test cases to calculate code-based metrics
            matched_test_case = next((tc for tc in TEST_CASES if tc["query"].strip().lower() == question.strip().lower()), None)
            
            st.markdown("#### LLM-as-a-Judge Metrics (Reference-free)")
            col1, col2, col3 = st.columns(3)
            col1.metric("Faithfulness", f"{f_score:.2f} / 1.0")
            col2.metric("Answer Relevancy", f"{r_score:.2f} / 1.0")
            col3.metric("Context Precision", f"{cp_score:.2f} / 1.0")
            
            with st.expander("Show Judge Reasoning"):
                st.markdown(f"**Faithfulness:** {f_reason}")
                st.markdown(f"**Relevancy:** {r_reason}")
                st.markdown(f"**Context Precision:** {cp_reason}")
                
            st.markdown("#### Code-Based Metrics (Requires Ground Truth)")
            if matched_test_case:
                retrieved_doc_ids = [item["chunk"].doc_id for item in final_chunks_for_llm]
                relevant_docs = matched_test_case["relevant_doc_ids"]
                trap_docs = matched_test_case["trap_doc_ids"]
                
                mrr = calculate_mrr(retrieved_doc_ids, relevant_docs)
                precision, recall = calculate_precision_recall(retrieved_doc_ids, relevant_docs, len(final_chunks_for_llm))
                trap_avoidance = calculate_trap_avoidance(retrieved_doc_ids, trap_docs)
                
                col4, col5, col6, col7 = st.columns(4)
                col4.metric("MRR", f"{mrr:.3f}")
                col5.metric(f"Precision@{len(final_chunks_for_llm)}", f"{precision:.3f}")
                col6.metric(f"Recall@{len(final_chunks_for_llm)}", f"{recall:.3f}")
                col7.metric("Trap Avoidance", "✅ Passed" if trap_avoidance == 1.0 else "❌ Failed")
                
                st.caption(f"Evaluated against known test case scenario: `{matched_test_case['scenario']}`")
            else:
                st.info("💡 **MRR, Precision, Recall, and Trap Avoidance** could not be calculated because your query does not match any of the predefined ground truth test cases in `evaluation/test_cases.py`.\n\nCode-based metrics require a known list of 'correct' documents to compare against, whereas the LLM-as-a-judge metrics above evaluate the text directly.")

    # --- Citations ---
    st.markdown("---")
    st.markdown("### 📎 Citations")

    cited_docs: set[str] = set()
    for item in final_chunks_for_llm:
        doc_id = item["chunk"].doc_id
        if f"[{doc_id}]" in answer:
            cited_docs.add(doc_id)

    if cited_docs:
        for item in final_chunks_for_llm:
            chunk = item["chunk"]
            if chunk.doc_id in cited_docs:
                st.markdown(
                    f"- **[{chunk.doc_id}]** — "
                    f"{doc_type_labels.get(chunk.doc_type, chunk.doc_type)} | "
                    f"Specialty: {chunk.specialty} | "
                    f"Disease: {', '.join(chunk.disease)}"
                )
    else:
        st.caption("No explicit citations found in the answer.")

    # --- HyDE Section (if enabled) ---
    st.markdown("---")
    if hyde_result:
        with st.expander("🔮 HyDE — Hypothetical Document", expanded=False):
            st.markdown("**Generated Hypothetical Passage:**")
            st.info(hyde_result.hypothetical_document)
            st.caption(
                "This hypothetical passage was embedded and used for "
                "retrieval instead of the raw query."
            )

    # --- Retrieved Documents ---
    st.markdown("### 📄 Retrieved Documents")

    # Show the documents used for answer generation
    for item in final_chunks_for_llm:
        chunk = item["chunk"]
        score = item["score"]
        rank = item["rank"]

        with st.expander(
            f"Rank #{rank} — {chunk.doc_id} | Score: {score:.4f} | "
            f"{chunk.doc_type.replace('_', ' ').title()}",
            expanded=(rank <= 3),
        ):
            # Metadata row
            col1, col2, col3, col4 = st.columns(4)
            col1.markdown(
                f"**Document:** {chunk.doc_id}",
            )
            col2.markdown(
                get_doc_type_badge(chunk.doc_type),
                unsafe_allow_html=True,
            )
            col3.markdown(f"**Specialty:** {chunk.specialty}")
            col4.markdown(f"**Disease:** {', '.join(chunk.disease)}")

            # Chunk text
            st.markdown("**Content:**")
            st.text(chunk.text)

            # Section info
            st.caption(
                f"Section: {chunk.section_type} | "
                f"Chunk ID: {chunk.chunk_id} | "
                f"Source Priority: {chunk.source_priority}"
            )

    # --- Evidence Viewer: Before/After Comparison ---
    if reranked_results or mmr_results:
        with st.expander("🔍 Evidence Viewer — Ranking Comparison", expanded=False):
            col_before, col_after = st.columns(2)

            with col_before:
                st.markdown("**Before Enhancement (Dense Retrieval)**")
                for r in pre_enhancement_results[:TOP_K_FINAL]:
                    st.markdown(
                        f"#{r.rank} — **{r.chunk.doc_id}** "
                        f"(score: {r.score:.4f}) "
                        f"*{r.chunk.doc_type.replace('_', ' ')}*"
                    )

            with col_after:
                label = "After "
                if use_mmr:
                    label += "MMR"
                if use_reranker:
                    label += (" + " if use_mmr else "") + "Cross-Encoder"
                st.markdown(f"**{label}**")

                display_results = reranked_results or mmr_results
                for r in display_results[:TOP_K_FINAL]:
                    if hasattr(r, "reranker_score"):
                        score_str = f"CE: {r.reranker_score:.4f}"
                        orig_str = f"was #{r.original_rank}"
                    elif hasattr(r, "mmr_score"):
                        score_str = f"MMR: {r.mmr_score:.4f}"
                        orig_str = f"was #{r.original_rank}"
                    else:
                        score_str = f"score: {r.score:.4f}"
                        orig_str = ""

                    st.markdown(
                        f"#{r.rank} — **{r.chunk.doc_id}** "
                        f"({score_str}) {orig_str} "
                        f"*{r.chunk.doc_type.replace('_', ' ')}*"
                    )

    # --- Timing Breakdown ---
    with st.expander("⏱️ Latency Breakdown", expanded=False):
        total = sum(timings.values())
        for step, elapsed in timings.items():
            label = step.replace("_", " ").title()
            st.markdown(f"- **{label}**: {elapsed:.2f}s")
        st.markdown(f"- **Total**: {total:.2f}s")

