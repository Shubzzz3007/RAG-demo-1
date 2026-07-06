# src/config.py
# ============================================================
# Central Configuration
# ============================================================
# This module loads environment variables from .env and provides
# typed constants used across the entire project.
#
# WHY a central config?
#   - Single source of truth for all settings
#   - No module ever reads .env directly
#   - Easy to change defaults in one place
# ============================================================

import os

# Fix OpenMP conflict between FAISS and PyTorch (cross-encoder)
# Both libraries link their own copy of libomp on macOS, which
# causes a crash. This env var must be set BEFORE importing either.
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from pathlib import Path
from dotenv import load_dotenv

# ----------------------------------------------------------
# Load .env file from project root
# ----------------------------------------------------------
# find_dotenv() searches upward, but we explicitly point to
# the project root to be predictable.
PROJECT_ROOT = Path(__file__).parent.parent
load_dotenv(PROJECT_ROOT / ".env")


# ============================================================
# PATH CONFIGURATION
# ============================================================
# All paths are relative to the project root so the project
# works regardless of where it's run from.
# ============================================================

DATA_DIR = PROJECT_ROOT / "data"
INDEXES_DIR = PROJECT_ROOT / "indexes"
EMBEDDINGS_DIR = PROJECT_ROOT / "embeddings"
METADATA_FILE = PROJECT_ROOT / "metadata.json"

# Create output directories if they don't exist
INDEXES_DIR.mkdir(exist_ok=True)
EMBEDDINGS_DIR.mkdir(exist_ok=True)


# ============================================================
# AZURE OPENAI — LLM (GPT-4o)
# ============================================================
# Used for: answer generation, HyDE hypothetical documents,
#           confidence assessment
# ============================================================

AZURE_OPENAI_ENDPOINT: str = os.getenv("AZURE_OPENAI_ENDPOINT", "")
AZURE_OPENAI_API_KEY: str = os.getenv("AZURE_OPENAI_API_KEY", "")
AZURE_OPENAI_API_VERSION: str = os.getenv("AZURE_OPENAI_API_VERSION", "")
AZURE_OPENAI_DEPLOYMENT_NAME: str = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "")
AZURE_OPENAI_MODEL_NAME: str = os.getenv("AZURE_OPENAI_MODEL_NAME", "")


# ============================================================
# AZURE OPENAI — EMBEDDINGS (text-embedding-3-small)
# ============================================================
# Used for: chunk embeddings (offline) and query embeddings (runtime)
# Embedding dimension for text-embedding-3-small = 1536
# ============================================================

AZURE_OPENAI_EMBEDDING_ENDPOINT: str = os.getenv("AZURE_OPENAI_EMBEDDING_ENDPOINT", "")
AZURE_OPENAI_EMBEDDING_KEY: str = os.getenv("AZURE_OPENAI_EMBEDDING_KEY", "")
AZURE_OPENAI_EMBEDDING_API_VERSION: str = os.getenv("AZURE_OPENAI_EMBEDDING_API_VERSION", "")
AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME: str = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME", "")
AZURE_OPENAI_EMBEDDING_MODEL_NAME: str = os.getenv("AZURE_OPENAI_EMBEDDING_MODEL_NAME", "")

# Embedding dimension for text-embedding-3-small
EMBEDDING_DIMENSION: int = 1536


# ============================================================
# CHUNKING CONFIGURATION
# ============================================================
# Section-aware chunking is the ONLY strategy.
# These defaults control how long documents are split.
# ============================================================

# Maximum characters per chunk (for fallback splitting of long sections)
MAX_CHUNK_SIZE: int = 500

# Overlap between fallback splits (characters)
CHUNK_OVERLAP: int = 50

# Recursive chunking configuration
RECURSIVE_CHUNK_SIZE: int = 400
RECURSIVE_CHUNK_OVERLAP: int = 100


# ============================================================
# RETRIEVAL CONFIGURATION
# ============================================================

# Number of chunks to retrieve from FAISS
TOP_K: int = 10

# Number of chunks to pass to LLM after all filtering/reranking
TOP_K_FINAL: int = 5


# ============================================================
# FAISS INDEX CONFIGURATION
# ============================================================
# We build 3 index types. All use inner product on L2-normalized
# vectors, which is equivalent to cosine similarity.
# ============================================================

# IVF: number of Voronoi cells (clusters)
# Rule of thumb: sqrt(n_vectors) for small datasets
IVF_NLIST: int = 20

# IVF: number of cells to probe during search
# Higher = more accurate but slower
IVF_NPROBE: int = 5

# HNSW: number of neighbors per node in the graph
# Higher = better recall but more memory
HNSW_M: int = 32

# HNSW: search depth (ef_search)
# Higher = better recall but slower
HNSW_EF_SEARCH: int = 64

# HNSW: construction depth (ef_construction)
HNSW_EF_CONSTRUCTION: int = 200


# ============================================================
# MMR CONFIGURATION
# ============================================================

# Lambda parameter: 1.0 = pure relevance, 0.0 = pure diversity
MMR_LAMBDA: float = 0.7


# ============================================================
# CROSS-ENCODER CONFIGURATION
# ============================================================

# Model name from HuggingFace sentence-transformers
CROSS_ENCODER_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"


# ============================================================
# LLM CONFIGURATION
# ============================================================

# Temperature for answer generation (low = more deterministic)
LLM_TEMPERATURE: float = 0.1

# Max tokens for LLM response
LLM_MAX_TOKENS: int = 1024
