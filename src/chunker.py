# src/chunker.py
# ============================================================
# Section-Aware Chunker
# ============================================================
# This module splits clinical documents into semantically
# meaningful chunks while preserving metadata.
#
# WHY section-aware chunking (and not fixed-size)?
#   Discharge summaries have a natural structure:
#     - Line 1: Patient demographics + diagnosis
#     - Line 2: Medications at discharge
#     - Line 3: Lab values (creatinine, eGFR)
#     - Line 4: Follow-up instructions
#     - Lines 5+: Additional notes (padding, education, etc.)
#
#   Splitting by fixed character count would break these
#   semantic boundaries. Section-aware chunking keeps related
#   clinical facts together, which improves retrieval quality.
#
# CHUNKING RULES:
#   1. Discharge summaries → split by line (each line = a section)
#      Then group small related lines into one chunk.
#   2. Guidelines → kept whole (they're short, ~100-150 chars)
#   3. PubMed abstracts → kept whole (short) or split by sentence
#      if they exceed MAX_CHUNK_SIZE.
#
# IMPORTANT: Every chunk inherits its parent document's metadata.
# This is critical for metadata filtering at retrieval time.
# ============================================================

import re
from dataclasses import dataclass, field
from typing import Optional

from src.config import MAX_CHUNK_SIZE, CHUNK_OVERLAP
from src.document_loader import Document
from src.utils import get_logger, timer

logger = get_logger(__name__)


# ============================================================
# CHUNK DATACLASS
# ============================================================

@dataclass
class Chunk:
    """
    A single chunk of text with its metadata.

    Every chunk carries the full metadata of its parent document.
    This allows metadata filtering at retrieval time without
    needing to look up the parent document.

    Attributes:
        chunk_id:        Unique ID (e.g., "DS-001_chunk_0")
        doc_id:          Parent document ID (e.g., "DS-001")
        text:            The chunk text content
        doc_type:        Inherited from parent document
        specialty:       Inherited from parent document
        disease:         Inherited from parent document
        patient_id:      Inherited from parent document
        date:            Inherited from parent document
        source_priority: Inherited from parent document
        section_type:    What kind of section this chunk represents
                         (e.g., "patient_info", "medications", "labs",
                          "follow_up", "additional_notes", "full_text")
    """
    chunk_id: str
    doc_id: str
    text: str
    doc_type: str = ""
    specialty: str = ""
    disease: list[str] = field(default_factory=list)
    patient_id: Optional[str] = None
    date: str = ""
    source_priority: str = ""
    section_type: str = "full_text"


# ============================================================
# SECTION DETECTION FOR DISCHARGE SUMMARIES
# ============================================================
# These patterns identify what type of clinical information
# each line in a discharge summary contains.
# ============================================================

# Maps regex patterns to section type labels
SECTION_PATTERNS: list[tuple[str, str]] = [
    (r"^Patient:", "patient_info"),
    (r"^Discharged on", "medications"),
    (r"^Creatinine|^eGFR|^Lab|^HbA1c|^BNP", "labs"),
    (r"^Advised follow-up|^Follow-up|^Advised", "follow_up"),
]


def _classify_line(line: str) -> str:
    """
    Classify a single line of a discharge summary into a section type.

    Uses regex pattern matching against known discharge summary patterns.
    If no pattern matches, the line is classified as "additional_notes".

    Args:
        line: A single line from a discharge summary.

    Returns:
        Section type string (e.g., "patient_info", "medications").
    """
    for pattern, section_type in SECTION_PATTERNS:
        if re.match(pattern, line, re.IGNORECASE):
            return section_type
    return "additional_notes"


# ============================================================
# CHUNKING FUNCTIONS BY DOCUMENT TYPE
# ============================================================

def _chunk_discharge_summary(doc: Document) -> list[Chunk]:
    """
    Chunk a discharge summary using section-aware splitting.

    Strategy:
        1. Split the document into lines
        2. Classify each line by section type
        3. Group consecutive lines of the same type into one chunk
        4. If a group exceeds MAX_CHUNK_SIZE, split it further

    This keeps related clinical information together. For example,
    "Patient: 68M with T2DM, HFrEF, CKD stage 3" stays as one chunk
    rather than being split in the middle.

    Args:
        doc: A Document with doc_type == "discharge_summary"

    Returns:
        List of Chunk objects.
    """
    lines = [line.strip() for line in doc.content.split("\n") if line.strip()]

    if not lines:
        return []

    # --- Step 1: Group lines by section type ---
    # We group consecutive lines with the same section type together.
    groups: list[tuple[str, str]] = []  # (section_type, text)
    current_type = _classify_line(lines[0])
    current_lines = [lines[0]]

    for line in lines[1:]:
        line_type = _classify_line(line)
        if line_type == current_type:
            # Same section → accumulate
            current_lines.append(line)
        else:
            # New section → save current group and start new one
            groups.append((current_type, "\n".join(current_lines)))
            current_type = line_type
            current_lines = [line]

    # Don't forget the last group
    groups.append((current_type, "\n".join(current_lines)))

    # --- Step 2: Create chunks from groups ---
    chunks: list[Chunk] = []

    for i, (section_type, text) in enumerate(groups):
        # If the text is too long, split it further with overlap
        if len(text) > MAX_CHUNK_SIZE:
            sub_chunks = _split_with_overlap(text, MAX_CHUNK_SIZE, CHUNK_OVERLAP)
            for j, sub_text in enumerate(sub_chunks):
                chunk = _create_chunk(
                    doc=doc,
                    text=sub_text,
                    chunk_index=len(chunks),
                    section_type=section_type,
                )
                chunks.append(chunk)
        else:
            chunk = _create_chunk(
                doc=doc,
                text=text,
                chunk_index=len(chunks),
                section_type=section_type,
            )
            chunks.append(chunk)

    return chunks


def _chunk_guideline(doc: Document) -> list[Chunk]:
    """
    Chunk a guideline document.

    Guidelines in this corpus are short (typically 80-155 characters),
    so we keep them as a single chunk. If a guideline somehow exceeds
    MAX_CHUNK_SIZE, we split it by sentence.

    Args:
        doc: A Document with doc_type == "guideline"

    Returns:
        List of Chunk objects (usually just one).
    """
    text = doc.content.strip()

    if len(text) <= MAX_CHUNK_SIZE:
        # Keep as a single chunk — most guidelines fit here
        return [_create_chunk(doc, text, chunk_index=0, section_type="guideline")]

    # Rare case: split long guidelines by sentence
    sentences = _split_into_sentences(text)
    return _group_sentences_into_chunks(doc, sentences, section_type="guideline")


def _chunk_pubmed_abstract(doc: Document) -> list[Chunk]:
    """
    Chunk a PubMed abstract.

    Most PubMed abstracts are short (140-450 characters), so they're
    kept whole. Longer abstracts (PM-171 to PM-200 are ~442 chars)
    are kept whole if under MAX_CHUNK_SIZE, otherwise split by sentence.

    Args:
        doc: A Document with doc_type == "pubmed_abstract"

    Returns:
        List of Chunk objects (usually just one).
    """
    text = doc.content.strip()

    if len(text) <= MAX_CHUNK_SIZE:
        return [_create_chunk(doc, text, chunk_index=0, section_type="abstract")]

    # Split longer abstracts by sentence
    sentences = _split_into_sentences(text)
    return _group_sentences_into_chunks(doc, sentences, section_type="abstract")


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def _create_chunk(
    doc: Document,
    text: str,
    chunk_index: int,
    section_type: str,
) -> Chunk:
    """
    Create a Chunk object that inherits metadata from its parent Document.

    Args:
        doc:          Parent document
        text:         Chunk text content
        chunk_index:  Index of this chunk within the document
        section_type: Type of section (e.g., "patient_info", "guideline")

    Returns:
        A Chunk object with all metadata fields populated.
    """
    return Chunk(
        chunk_id=f"{doc.doc_id}_chunk_{chunk_index}",
        doc_id=doc.doc_id,
        text=text,
        doc_type=doc.doc_type,
        specialty=doc.specialty,
        disease=list(doc.disease),  # copy to avoid shared references
        patient_id=doc.patient_id,
        date=doc.date,
        source_priority=doc.source_priority,
        section_type=section_type,
    )


def _split_into_sentences(text: str) -> list[str]:
    """
    Split text into sentences using a simple regex.

    Handles common abbreviations (e.g., "mg.", "mL.", "eGFR.") by
    splitting only on period-space-capital-letter patterns.

    Args:
        text: Input text to split.

    Returns:
        List of sentence strings.
    """
    # Split on period followed by space and uppercase letter
    # This avoids splitting "500 mg." or "eGFR: 38 mL/min."
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
    return [s.strip() for s in sentences if s.strip()]


def _split_with_overlap(text: str, max_size: int, overlap: int) -> list[str]:
    """
    Split text into fixed-size chunks with overlap.

    This is a fallback for sections that are too long. It splits
    by character count with an overlap window to preserve context.

    Args:
        text:     Text to split
        max_size: Maximum characters per chunk
        overlap:  Number of overlapping characters between chunks

    Returns:
        List of text chunks.
    """
    chunks = []
    start = 0

    while start < len(text):
        end = start + max_size
        chunk = text[start:end]
        chunks.append(chunk.strip())
        # Move start forward by (max_size - overlap) for the next chunk
        start += max_size - overlap

    return [c for c in chunks if c]  # Remove any empty chunks


def _group_sentences_into_chunks(
    doc: Document,
    sentences: list[str],
    section_type: str,
) -> list[Chunk]:
    """
    Group sentences into chunks that fit within MAX_CHUNK_SIZE.

    Accumulates sentences until adding the next one would exceed
    the limit, then starts a new chunk.

    Args:
        doc:          Parent document
        sentences:    List of sentences to group
        section_type: Section type label for all created chunks

    Returns:
        List of Chunk objects.
    """
    chunks: list[Chunk] = []
    current_text = ""

    for sentence in sentences:
        # Check if adding this sentence would exceed the limit
        candidate = f"{current_text} {sentence}".strip() if current_text else sentence

        if len(candidate) <= MAX_CHUNK_SIZE:
            current_text = candidate
        else:
            # Save current chunk and start a new one
            if current_text:
                chunks.append(_create_chunk(
                    doc, current_text, chunk_index=len(chunks),
                    section_type=section_type,
                ))
            current_text = sentence

    # Save the last accumulated chunk
    if current_text:
        chunks.append(_create_chunk(
            doc, current_text, chunk_index=len(chunks),
            section_type=section_type,
        ))

    return chunks


# ============================================================
# MAIN CHUNKING FUNCTION
# ============================================================

@timer
def chunk_documents(documents: list[Document]) -> list[Chunk]:
    """
    Chunk all documents using section-aware strategy.

    Routes each document to the appropriate chunking function
    based on its doc_type.

    Args:
        documents: List of Document objects from the loader.

    Returns:
        List of all Chunk objects across all documents.
    """
    all_chunks: list[Chunk] = []

    # Dispatch table: maps doc_type → chunking function
    chunkers = {
        "discharge_summary": _chunk_discharge_summary,
        "guideline": _chunk_guideline,
        "pubmed_abstract": _chunk_pubmed_abstract,
    }

    for doc in documents:
        chunker_fn = chunkers.get(doc.doc_type)

        if chunker_fn is None:
            # Unknown doc type → treat as a single chunk
            logger.warning(f"Unknown doc_type '{doc.doc_type}' for {doc.doc_id}")
            all_chunks.append(
                _create_chunk(doc, doc.content, chunk_index=0, section_type="unknown")
            )
            continue

        doc_chunks = chunker_fn(doc)
        all_chunks.extend(doc_chunks)

    # --- Summary ---
    type_counts: dict[str, int] = {}
    for chunk in all_chunks:
        type_counts[chunk.doc_type] = type_counts.get(chunk.doc_type, 0) + 1

    logger.info(
        f"Created {len(all_chunks)} chunks from {len(documents)} documents: "
        f"{type_counts}"
    )

    return all_chunks


# ============================================================
# QUICK TEST
# ============================================================
# Run this file directly to verify chunking works:
#   python -m src.chunker
# ============================================================

if __name__ == "__main__":
    from src.document_loader import load_documents

    docs = load_documents()
    chunks = chunk_documents(docs)

    print(f"\n{'='*60}")
    print(f"Total chunks: {len(chunks)}")
    print(f"{'='*60}\n")

    # Show section type distribution
    section_counts: dict[str, int] = {}
    for c in chunks:
        section_counts[c.section_type] = section_counts.get(c.section_type, 0) + 1
    print(f"Section types: {section_counts}\n")

    # Show a few example chunks
    for chunk in chunks[:5]:
        print(f"  {chunk.chunk_id} [{chunk.section_type}]")
        print(f"    doc_type:  {chunk.doc_type}")
        print(f"    disease:   {chunk.disease}")
        print(f"    text:      {chunk.text[:100]}...")
        print()
