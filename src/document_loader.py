# src/document_loader.py
# ============================================================
# Document Loader
# ============================================================
# This module loads the clinical corpus from disk.
#
# RESPONSIBILITIES:
#   1. Read all .txt files from the data/ directory
#   2. Read metadata.json and join metadata to each document
#   3. Return a list of structured Document objects
#
# WHY a Document dataclass?
#   - Every downstream module (chunker, embedder, retriever)
#     needs both the text AND the metadata.
#   - A dataclass is simpler than a dict — it has typed fields,
#     IDE autocomplete, and is immutable by default.
#
# IMPORTANT:
#   - The doc_id is derived from the filename (e.g., "DS-001.txt" → "DS-001")
#   - Documents without matching metadata are logged as warnings
#     but still loaded (with empty metadata fields).
# ============================================================

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from src.config import DATA_DIR, METADATA_FILE
from src.utils import get_logger, timer

logger = get_logger(__name__)


# ============================================================
# DOCUMENT DATACLASS
# ============================================================

@dataclass
class Document:
    """
    Represents a single clinical document with its metadata.

    Attributes:
        doc_id:          Unique identifier (e.g., "DS-001", "GL-015", "PM-042")
        content:         Full text content of the document
        doc_type:        One of: "discharge_summary", "guideline", "pubmed_abstract"
        specialty:       Medical specialty (e.g., "cardiology", "endocrinology")
        disease:         List of diseases mentioned (e.g., ["heart failure", "CKD"])
        patient_id:      Patient identifier (only for discharge summaries)
        date:            Document date as string (e.g., "2026-05-12")
        source_priority: Priority level ("patient_record", "guideline", "literature")
    """
    doc_id: str
    content: str
    doc_type: str = ""
    specialty: str = ""
    disease: list[str] = field(default_factory=list)
    patient_id: Optional[str] = None
    date: str = ""
    source_priority: str = ""


# ============================================================
# METADATA LOADER
# ============================================================

def _load_metadata(metadata_path: Path) -> dict[str, dict]:
    """
    Load metadata.json and return a dict keyed by doc_id.

    The metadata file is a JSON array of objects, each with a "doc_id" field.
    We convert it to a dict for O(1) lookup when joining with documents.

    Args:
        metadata_path: Path to metadata.json

    Returns:
        Dict mapping doc_id → metadata dict
    """
    if not metadata_path.exists():
        logger.warning(f"Metadata file not found: {metadata_path}")
        return {}

    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata_list = json.load(f)

    # Convert list to dict keyed by doc_id for fast lookup
    metadata_dict = {item["doc_id"]: item for item in metadata_list}
    logger.info(f"Loaded metadata for {len(metadata_dict)} documents")

    return metadata_dict


# ============================================================
# DOCUMENT LOADER
# ============================================================

@timer
def load_documents() -> list[Document]:
    """
    Load all clinical documents from the data/ directory and
    join them with metadata from metadata.json.

    Steps:
        1. Read metadata.json into a lookup dict
        2. Scan data/ for all .txt files
        3. For each file, extract doc_id from filename
        4. Join with metadata (if available)
        5. Create a Document object

    Returns:
        List of Document objects, sorted by doc_id.

    Raises:
        FileNotFoundError: If the data/ directory doesn't exist.
    """
    # --- Step 1: Load metadata ---
    metadata_dict = _load_metadata(METADATA_FILE)

    # --- Step 2: Scan for .txt files ---
    if not DATA_DIR.exists():
        raise FileNotFoundError(f"Data directory not found: {DATA_DIR}")

    txt_files = sorted(DATA_DIR.glob("*.txt"))
    logger.info(f"Found {len(txt_files)} text files in {DATA_DIR}")

    # --- Step 3-5: Load each document ---
    documents: list[Document] = []

    for filepath in txt_files:
        # Extract doc_id from filename: "DS-001.txt" → "DS-001"
        doc_id = filepath.stem

        # Read file content
        content = filepath.read_text(encoding="utf-8").strip()

        # Skip empty files
        if not content:
            logger.warning(f"Skipping empty file: {filepath.name}")
            continue

        # Look up metadata for this document
        meta = metadata_dict.get(doc_id, {})

        if not meta:
            logger.warning(f"No metadata found for {doc_id}")

        # Create Document with metadata fields
        doc = Document(
            doc_id=doc_id,
            content=content,
            doc_type=meta.get("doc_type", ""),
            specialty=meta.get("specialty", ""),
            disease=meta.get("disease", []),
            patient_id=meta.get("patient_id"),
            date=meta.get("date", ""),
            source_priority=meta.get("source_priority", ""),
        )

        documents.append(doc)

    # --- Summary ---
    # Count by document type for logging
    type_counts: dict[str, int] = {}
    for doc in documents:
        type_counts[doc.doc_type] = type_counts.get(doc.doc_type, 0) + 1

    logger.info(f"Loaded {len(documents)} documents: {type_counts}")

    return documents


# ============================================================
# QUICK TEST
# ============================================================
# Run this file directly to verify document loading works:
#   python -m src.document_loader
# ============================================================

if __name__ == "__main__":
    docs = load_documents()

    # Print a few examples
    print(f"\n{'='*60}")
    print(f"Total documents loaded: {len(docs)}")
    print(f"{'='*60}\n")

    # Show one of each type
    shown_types: set[str] = set()
    for doc in docs:
        if doc.doc_type not in shown_types:
            shown_types.add(doc.doc_type)
            print(f"--- {doc.doc_type.upper()} ---")
            print(f"  doc_id:     {doc.doc_id}")
            print(f"  specialty:  {doc.specialty}")
            print(f"  disease:    {doc.disease}")
            print(f"  patient_id: {doc.patient_id}")
            print(f"  date:       {doc.date}")
            print(f"  content:    {doc.content[:120]}...")
            print()

        if len(shown_types) == 3:
            break
