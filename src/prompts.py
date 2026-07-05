# src/prompts.py
# ============================================================
# Prompt Templates — Clinical RAG
# ============================================================
# This module contains all prompt templates used in the project.
#
# WHY centralized prompts?
#   - Easy to review and iterate on prompt engineering
#   - No prompts scattered across multiple files
#   - Clear separation of prompt logic from application logic
#
# PROMPT DESIGN PRINCIPLES:
#   1. The LLM must answer ONLY from retrieved evidence
#   2. If evidence is insufficient → refuse to answer
#   3. Every claim must cite its source document
#   4. The LLM must assess and report confidence
#   5. Distinguish patient-specific data from guideline data
# ============================================================


# ============================================================
# SYSTEM PROMPT
# ============================================================
# This sets the LLM's role and behavior rules.
# It is used for EVERY answer generation call.
# ============================================================

SYSTEM_PROMPT = """You are a clinical evidence assistant. Your role is to answer
clinical questions using ONLY the evidence provided in the retrieved documents below.

STRICT RULES:
1. Answer ONLY from the provided evidence. Do NOT use your general knowledge.
2. If the evidence is insufficient to answer the question, respond with:
   "I don't have enough evidence to answer this question."
3. Cite every claim using the document ID in square brackets, e.g., [DS-001], [GL-015].
4. Clearly distinguish between:
   - Patient-specific facts (from discharge summaries)
   - Clinical guidelines (from guideline documents)
   - Research evidence (from PubMed abstracts)
5. If evidence from different sources conflicts, explicitly note the conflict.
6. Be concise and clinically precise. Avoid speculation.

CONFIDENCE ASSESSMENT:
At the end of your response, assess the confidence level:
- HIGH: Multiple sources corroborate the answer, evidence is directly relevant
- MEDIUM: Single source, or evidence is partially relevant
- LOW: Evidence is tangential or sparse

Format your confidence as:
**Confidence: [HIGH/MEDIUM/LOW]**
Followed by a brief justification for the confidence level."""


# ============================================================
# USER PROMPT TEMPLATE
# ============================================================
# This template injects the retrieved evidence and the user's
# question into a structured prompt for the LLM.
# ============================================================

USER_PROMPT_TEMPLATE = """## Retrieved Evidence

{evidence_section}

## Question

{question}

## Instructions

Answer the question using ONLY the evidence above. Cite sources using document IDs
in square brackets (e.g., [DS-001]). If the evidence is insufficient, say so.
End with a confidence assessment (HIGH / MEDIUM / LOW)."""


# ============================================================
# EVIDENCE FORMATTING
# ============================================================

def format_evidence_section(chunks_with_scores: list[dict]) -> str:
    """
    Format retrieved chunks into a structured evidence section
    for the LLM prompt.

    Each chunk is labeled with its metadata so the LLM can
    distinguish between discharge summaries, guidelines, and
    PubMed abstracts, and can cite them properly.

    Args:
        chunks_with_scores: List of dicts with keys:
            - "chunk": Chunk object
            - "score": float (similarity or reranker score)
            - "rank": int (position in results)

    Returns:
        Formatted evidence string ready for the prompt.
    """
    evidence_parts: list[str] = []

    for item in chunks_with_scores:
        chunk = item["chunk"]
        score = item["score"]
        rank = item["rank"]

        # Format the document type label for clarity
        type_labels = {
            "discharge_summary": "Discharge Summary",
            "guideline": "Clinical Guideline",
            "pubmed_abstract": "PubMed Abstract",
        }
        doc_type_label = type_labels.get(chunk.doc_type, chunk.doc_type)

        # Build the evidence block
        evidence_block = (
            f"### Document [{chunk.doc_id}] — {doc_type_label}\n"
            f"- **Specialty**: {chunk.specialty}\n"
            f"- **Disease**: {', '.join(chunk.disease)}\n"
            f"- **Relevance Score**: {score:.4f}\n\n"
            f"{chunk.text}"
        )

        evidence_parts.append(evidence_block)

    return "\n\n---\n\n".join(evidence_parts)


def build_user_prompt(question: str, chunks_with_scores: list[dict]) -> str:
    """
    Build the complete user prompt with evidence and question.

    Args:
        question:           The user's clinical question.
        chunks_with_scores: Retrieved chunks with scores.

    Returns:
        Complete formatted prompt string.
    """
    evidence_section = format_evidence_section(chunks_with_scores)

    return USER_PROMPT_TEMPLATE.format(
        evidence_section=evidence_section,
        question=question,
    )
