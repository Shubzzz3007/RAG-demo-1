# Explainable Clinical RAG — Deep Dive

## 1. The Big Picture

```
460 Documents (text files)          metadata.json
       ↓                                ↓
   Chunking (section-aware)    ←   Join by doc_id
       ↓
  1,072 Chunks (each with metadata)
       ↓
  Azure OpenAI Embeddings (text-embedding-3-small)
       ↓
  1,072 Vectors (each 1536 dimensions)
       ↓
  Build 3 FAISS Indexes ──→ indexes/flat.index
                           ──→ indexes/ivf.index
                           ──→ indexes/hnsw.index
```

Everything above happens **ONCE** (offline). At runtime, we only **load** and **search**.

---

## 2. How Documents Become Chunks (Metadata Mapping)

### Step 1: A document starts as a text file + metadata entry

**File:** `data/DS-001.txt`
```
Patient: 36F with type 1 diabetes mellitus.
Discharged on lisinopril 10 mg daily, empagliflozin 10 mg daily, furosemide 40 mg daily.
Creatinine at discharge: 2.0 mg/dL. eGFR: 84 mL/min.
Advised follow-up with cardiology in 4 weeks and internal medicine in 2 weeks.
```

**Entry in** `metadata.json`:
```json
{
  "doc_id": "DS-001",
  "doc_type": "discharge_summary",
  "specialty": "endocrinology",
  "disease": ["type 1 diabetes mellitus"],
  "patient_id": "P-001",
  "date": "2026-01-03",
  "source_priority": "patient_record"
}
```

### Step 2: The chunker splits by clinical section

Each line is classified by pattern matching:

| Line starts with | Section Type | What it contains |
|---|---|---|
| `Patient:` | `patient_info` | Demographics + diagnosis |
| `Discharged on` | `medications` | Discharge medications |
| `Creatinine` / `eGFR` | `labs` | Lab values at discharge |
| `Advised follow-up` | `follow_up` | Follow-up instructions |
| Anything else | `additional_notes` | Education, diet, social work |

DS-001 becomes **4 chunks**:

```
┌─────────────────────────────────────────────────────────┐
│ Chunk: DS-001_chunk_0                                   │
│ Text: "Patient: 36F with type 1 diabetes mellitus."     │
│ Section: patient_info                                   │
│ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ │
│ METADATA (copied from parent DS-001):                   │
│   doc_type:        discharge_summary                    │
│   specialty:       endocrinology                        │
│   disease:         [type 1 diabetes mellitus]           │
│   patient_id:      P-001                                │
│   date:            2026-01-03                           │
│   source_priority: patient_record                       │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│ Chunk: DS-001_chunk_1                                   │
│ Text: "Discharged on lisinopril 10 mg daily..."         │
│ Section: medications                                    │
│ METADATA: same as above (copied from DS-001)            │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│ Chunk: DS-001_chunk_2                                   │
│ Text: "Creatinine at discharge: 2.0 mg/dL..."          │
│ Section: labs                                           │
│ METADATA: same as above (copied from DS-001)            │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│ Chunk: DS-001_chunk_3                                   │
│ Text: "Advised follow-up with cardiology..."            │
│ Section: follow_up                                      │
│ METADATA: same as above (copied from DS-001)            │
└─────────────────────────────────────────────────────────┘
```

> [!IMPORTANT]
> **Every chunk carries a FULL COPY of its parent document's metadata.** This is critical — when FAISS returns chunk #742, we immediately know its doc_type, disease, specialty, etc. without needing to look up the parent document.

### Step 3: How different doc types are chunked

| Doc Type | Count | Chunking Rule | Chunks Per Doc |
|---|---|---|---|
| Discharge Summary | 200 docs | Split by line/section | 3-10 chunks each → **812 total** |
| Guideline | 60 docs | Kept whole (short, ~100-150 chars) | 1 chunk each → **60 total** |
| PubMed Abstract | 200 docs | Kept whole (short, ~150-450 chars) | 1 chunk each → **200 total** |
| **Total** | **460 docs** | | **1,072 chunks** |

---

## 3. What's in the `embeddings/` Folder

```
embeddings/
├── chunks.pkl       (pickle file — list of 1,072 Chunk objects with all metadata)
└── embeddings.npy   (numpy file — matrix of shape 1072 × 1536, float32, ~6.4 MB)
```

**The critical alignment:** Chunk at index `i` in `chunks.pkl` corresponds to vector at row `i` in `embeddings.npy`. This is how we go from a FAISS result (index number) back to the actual text and metadata.

```
chunks.pkl:      [Chunk_0, Chunk_1, Chunk_2, ..., Chunk_1071]
                     ↕        ↕        ↕              ↕
embeddings.npy:  [Vector_0, Vector_1, Vector_2, ..., Vector_1071]
                     ↕        ↕        ↕              ↕
FAISS index:     [  idx 0,   idx 1,   idx 2,  ...,   idx 1071 ]
```

When FAISS returns `index=742, score=0.72`, we look up `chunks[742]` to get the text and metadata.

---

## 4. What's in the `indexes/` Folder

```
indexes/
├── flat.index    (6,432 KB — exact search, stores raw vectors)
├── ivf.index     (6,561 KB — clustered search, stores vectors + cluster centroids)
└── hnsw.index    (6,717 KB — graph search, stores vectors + graph edges)
```

All three indexes contain the **same 1,072 vectors**. They differ in **how they organize and search** those vectors. Only ONE index is loaded at runtime (whichever the user selects).

### How Each FAISS Index Works Internally

### 🔵 Flat Index (IndexFlatIP) — Exact Search

```
Query Vector ──→ Compare with ALL 1,072 vectors ──→ Return top-k

Vector 0:   similarity = 0.42
Vector 1:   similarity = 0.38
Vector 2:   similarity = 0.71  ← rank #2
Vector 3:   similarity = 0.15
...
Vector 741: similarity = 0.73  ← rank #1
...
Vector 1071: similarity = 0.29

Sort all scores → return top-k highest
```

| Property | Value |
|---|---|
| **Search method** | Brute-force (compares against every single vector) |
| **Accuracy** | 100% — guaranteed to find the true nearest neighbors |
| **Speed** | O(n) per query — slow for millions of vectors, fine for 1,072 |
| **Memory** | Stores raw vectors (1,072 × 1,536 × 4 bytes = 6.4 MB) |
| **Training needed?** | No |
| **When to use** | Small datasets, or when you need an accuracy baseline |

---

### 🟢 IVF Index (IndexIVFFlat) — Clustered Search

**Build time** (training):
```
Step 1: Run K-means to partition 1,072 vectors into 20 clusters (nlist=20)

        Cluster 0: [v3, v17, v42, v89, ...]     ~54 vectors
        Cluster 1: [v1, v8, v55, v102, ...]      ~54 vectors
        Cluster 2: [v0, v12, v33, v77, ...]      ~54 vectors
        ...
        Cluster 19: [v5, v29, v61, v200, ...]    ~54 vectors

Step 2: Store each cluster's centroid (center point)
```

**Search time:**
```
Query Vector
    ↓
Step 1: Find the 5 nearest cluster centroids (nprobe=5)
    ↓
    Closest clusters: [Cluster 7, Cluster 3, Cluster 15, Cluster 0, Cluster 12]
    ↓
Step 2: Search ONLY vectors inside those 5 clusters
    ↓
    Instead of comparing 1,072 vectors, compare ~270 vectors (5 × 54)
    ↓
Return top-k from those ~270 vectors
```

| Property | Value |
|---|---|
| **Search method** | Only search the closest clusters (skip ~75% of vectors) |
| **Accuracy** | ~95-99% — might miss a relevant vector if it's in a cluster we didn't probe |
| **Speed** | O(n/nlist × nprobe) — much faster for large datasets |
| **Parameters** | `nlist=20` (number of clusters), `nprobe=5` (clusters to search) |
| **Training needed?** | Yes — must run K-means on the data |
| **When to use** | Medium-large datasets (10K-1M vectors) |

> [!NOTE]
> For our 1,072 vectors, IVF doesn't give much speed benefit — the dataset is too small. But it demonstrates the concept for production-scale datasets.

---

### 🟠 HNSW Index (IndexHNSWFlat) — Graph Search

**Build time:**
```
Build a multi-layer graph where each vector is a node connected to its neighbors:

Layer 2 (sparse):    v42 ─── v317 ─── v889
                      |         |
Layer 1 (medium):    v42 ── v17 ── v317 ── v555 ── v889
                      |      |       |       |       |
Layer 0 (dense):     v42─v17─v3─v317─v99─v555─v201─v889─v741─...
                      |   |   |   |   |    |    |    |    |
                     (every vector is here, connected to M=32 neighbors)
```

**Search time:**
```
Query Vector
    ↓
Step 1: Enter graph at the top layer (sparse)
    ↓    Start at entry point v42
    ↓    Check its neighbors → v317 is closer to query → move there
    ↓
Step 2: Drop down to Layer 1
    ↓    From v317, check neighbors → v555 is closer → move there
    ↓
Step 3: Drop down to Layer 0 (dense)
    ↓    From v555, greedily explore neighbors
    ↓    v201 is closer → move there
    ↓    v741 is closer → move there
    ↓    No neighbor is closer → STOP → v741 is the nearest neighbor
    ↓
Return top-k neighbors found during traversal
```

| Property | Value |
|---|---|
| **Search method** | Greedy graph traversal (like navigating a highway system) |
| **Accuracy** | ~98-99.9% — very high recall with proper parameters |
| **Speed** | O(log n) — fast even for millions of vectors |
| **Parameters** | `M=32` (neighbors per node), `efSearch=64` (search depth), `efConstruction=200` (build depth) |
| **Training needed?** | No — graph is built incrementally |
| **Memory** | Higher than Flat (stores vectors + graph edges) |
| **When to use** | Production systems needing both speed and accuracy |

---

### Index Comparison at a Glance

| | Flat | IVF | HNSW |
|---|---|---|---|
| **Approach** | Check everything | Check nearby clusters | Walk a graph |
| **Analogy** | Reading every book in a library | Go to the right shelf, then search | Follow a trail of increasingly relevant clues |
| **Recall** | 100% | ~95-99% | ~98-99.9% |
| **Speed** | Slowest | Medium | Fastest |
| **File Size** | 6,432 KB | 6,561 KB | 6,717 KB |
| **Needs Training** | No | Yes (K-means) | No |

---

## 5. What Happens When You Submit a Query

Let's trace a real query through every step:

> **Query:** "What are the contraindications for metformin in patients with renal impairment?"

### Configuration: Flat Index + HyDE ON + MMR ON + Cross-Encoder ON + Filter: Guidelines Only

```
┌─────────────────────────────────────────────────────────────────┐
│ STEP 1: METADATA FILTERING (pre-retrieval)                      │
│                                                                 │
│ User selected: doc_type = "guideline"                           │
│                                                                 │
│ Loop through all 1,072 chunks:                                  │
│   Chunk 0 (DS-001_chunk_0): doc_type=discharge_summary → SKIP   │
│   Chunk 1 (DS-001_chunk_1): doc_type=discharge_summary → SKIP   │
│   ...                                                           │
│   Chunk 812 (GL-001_chunk_0): doc_type=guideline → ✅ KEEP      │
│   Chunk 813 (GL-002_chunk_0): doc_type=guideline → ✅ KEEP      │
│   ...                                                           │
│                                                                 │
│ Result: allowed_indices = [812, 813, 814, ..., 871]  (60 chunks)│
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ STEP 2: HyDE (Hypothetical Document Embeddings)                 │
│                                                                 │
│ Send to GPT-4o:                                                 │
│   "Given this question, write a hypothetical clinical passage"  │
│                                                                 │
│ GPT-4o generates:                                               │
│   "Metformin is generally considered safe for patients with     │
│    CKD when eGFR is above 30 mL/min. Below eGFR 30, metformin  │
│    should be discontinued due to risk of lactic acidosis..."    │
│                                                                 │
│ Embed this hypothetical passage → get a 1536-dim vector         │
│                                                                 │
│ WHY? The hypothetical answer uses clinical vocabulary            │
│ ("eGFR", "lactic acidosis", "CKD") that matches the actual     │
│ documents better than the raw question does.                    │
│                                                                 │
│ Result: hypothetical_embedding (1 × 1536)                       │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ STEP 3: DENSE RETRIEVAL (FAISS search)                          │
│                                                                 │
│ Because metadata filter is active (only 60 chunks):             │
│   → Create a TEMPORARY flat index with only those 60 vectors    │
│   → Search using the HyDE embedding (not the raw query)        │
│                                                                 │
│ FAISS computes: inner_product(hyde_embedding, each_vector)      │
│ Since vectors are L2-normalized: inner_product = cosine_sim     │
│                                                                 │
│   GL-049: score = 0.7202  ← rank #1                            │
│   GL-050: score = 0.7201  ← rank #2                            │
│   GL-007: score = 0.6303  ← rank #3                            │
│   GL-006: score = 0.6303  ← rank #4                            │
│   GL-005: score = 0.6303  ← rank #5                            │
│   GL-033: score = 0.5891  ← rank #6                            │
│   ...                                                           │
│   GL-041: score = 0.4102  ← rank #10                           │
│                                                                 │
│ Result: top 10 chunks with scores                               │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ STEP 4: MMR RE-RANKING (diversity)                              │
│                                                                 │
│ Problem: GL-049 and GL-050 say almost the same thing!           │
│ MMR iteratively picks chunks that are:                          │
│   - Similar to query (high relevance)                           │
│   - Different from already-picked chunks (high diversity)       │
│                                                                 │
│ Iteration 1: Pick GL-049 (highest relevance, nothing to compare)│
│ Iteration 2: GL-050 is relevant but TOO SIMILAR to GL-049      │
│              → Pick GL-007 instead (different topic = diverse)  │
│ Iteration 3: Pick GL-033 (different from both GL-049 and GL-007)│
│ Iteration 4: Now pick GL-050 (it's relevant enough despite sim) │
│ Iteration 5: Pick GL-041 (adds another perspective)            │
│                                                                 │
│ Before MMR: [GL-049, GL-050, GL-007, GL-006, GL-005]            │
│ After MMR:  [GL-049, GL-007, GL-033, GL-050, GL-041]            │
│                                                                 │
│ Result: top 5 diverse chunks                                    │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ STEP 5: CROSS-ENCODER RE-RANKING                                │
│                                                                 │
│ The bi-encoder (embedding search) is fast but approximate.      │
│ The cross-encoder is slow but more accurate.                    │
│                                                                 │
│ Bi-encoder:   embed(query) + embed(chunk) → compare vectors     │
│ Cross-encoder: model(query + chunk together) → relevance score  │
│                                                                 │
│ For each of the 5 chunks, create a pair:                        │
│   ("What are contraindications for metformin...", "GL-049 text") │
│   ("What are contraindications for metformin...", "GL-007 text") │
│   ...                                                           │
│                                                                 │
│ Cross-encoder scores each pair:                                 │
│   GL-049: CE_score = 4.82  (was #1 → stays #1)                 │
│   GL-033: CE_score = 3.15  (was #3 → moves to #2)              │
│   GL-007: CE_score = 2.91  (was #2 → moves to #3)              │
│   GL-050: CE_score = 2.88  (was #4 → stays #4)                 │
│   GL-041: CE_score = 0.42  (was #5 → stays #5)                 │
│                                                                 │
│ Result: final top 5 in refined order                            │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ STEP 6: LLM ANSWER GENERATION                                  │
│                                                                 │
│ Build prompt with the top 5 chunks:                             │
│                                                                 │
│ SYSTEM: "You are a clinical evidence assistant. Answer ONLY     │
│          from provided evidence. Cite sources. Assess           │
│          confidence."                                           │
│                                                                 │
│ USER:   "## Retrieved Evidence                                  │
│          ### Document [GL-049] — Clinical Guideline             │
│          - Specialty: primary care                              │
│          - Disease: type 2 diabetes mellitus, CKD              │
│          Metformin is contraindicated in patients with eGFR     │
│          below 60 mL/min/1.73m2 due to risk of lactic acidosis.│
│          ...                                                    │
│          ## Question                                            │
│          What are the contraindications for metformin...?"      │
│                                                                 │
│ GPT-4o generates answer with [GL-049] citations + confidence   │
│                                                                 │
│ Result: answer + citations + confidence tier                    │
└─────────────────────────────────────────────────────────────────┘
```

---

## 6. All Possible Pipeline Combinations

### FAISS Index Options (pick one)

| Option | Internal Mechanism |
|---|---|
| **Flat** | Compare query against all vectors — exact |
| **IVF** | Find nearest cluster centroids, search only those clusters — approximate |
| **HNSW** | Walk a hierarchical graph toward nearest neighbors — approximate |

### Enhancement Options (each independently ON/OFF)

| Option | What Changes in the Pipeline |
|---|---|
| **HyDE OFF** | Embed the raw query text → search FAISS |
| **HyDE ON** | Ask LLM to write a hypothetical answer → embed *that* → search FAISS |
| **MMR OFF** | Take top-k results as-is |
| **MMR ON** | Re-order top-k to maximize diversity |
| **Cross-Encoder OFF** | Keep ranking from FAISS (or MMR) |
| **Cross-Encoder ON** | Re-score each chunk with a cross-encoder model → re-order |

### Metadata Filters (additive, all optional)

| Filter | Effect |
|---|---|
| **Doc Type** | Only search discharge summaries / guidelines / PubMed |
| **Disease** | Only search chunks tagged with specific diseases |
| **Specialty** | Only search chunks tagged with specific specialties |

### Total Possible Combinations

```
3 FAISS indexes × 2 HyDE × 2 MMR × 2 Cross-Encoder = 24 pipeline configurations
```

Each of these 24 can additionally have any combination of metadata filters applied.

### Combination Matrix

| # | FAISS | HyDE | MMR | Cross-Encoder | Description |
|---|---|---|---|---|---|
| 1 | Flat | ❌ | ❌ | ❌ | **Baseline** — pure dense retrieval |
| 2 | Flat | ✅ | ❌ | ❌ | HyDE improves query vocabulary matching |
| 3 | Flat | ❌ | ✅ | ❌ | MMR adds source diversity |
| 4 | Flat | ❌ | ❌ | ✅ | Cross-encoder refines ranking |
| 5 | Flat | ✅ | ✅ | ❌ | Better query matching + diversity |
| 6 | Flat | ✅ | ❌ | ✅ | Better query matching + refined ranking |
| 7 | Flat | ❌ | ✅ | ✅ | Diversity + refined ranking |
| 8 | Flat | ✅ | ✅ | ✅ | **All enhancements** with exact search |
| 9-16 | IVF | ... | ... | ... | Same 8 combos with IVF search |
| 17-24 | HNSW | ... | ... | ... | Same 8 combos with HNSW search |

---

## 7. How the Index File is Used at Runtime

```python
# At startup — user selects "hnsw" in sidebar
index = faiss.read_index("indexes/hnsw.index")   # Load ~6.7 MB file into memory
chunks = pickle.load("embeddings/chunks.pkl")      # Load 1,072 chunk objects
embeddings = np.load("embeddings/embeddings.npy")  # Load 1,072 vectors

# At query time
query_vector = embed("What about metformin?")      # → shape (1, 1536)
scores, indices = index.search(query_vector, k=10)  # FAISS returns indices

# indices = [741, 742, 48, 49, 103]
# scores  = [0.72, 0.71, 0.63, 0.63, 0.61]

# Map FAISS indices back to chunks
for idx, score in zip(indices[0], scores[0]):
    chunk = chunks[idx]          # Get chunk object with text + metadata
    print(chunk.doc_id)          # "GL-049"
    print(chunk.text)            # "Metformin is contraindicated..."
    print(chunk.disease)         # ["type 2 diabetes mellitus", "CKD"]
    print(chunk.doc_type)        # "guideline"
```

The index file is a **serialized FAISS data structure**. It contains the vectors organized for fast search, but NOT the chunk text or metadata — those live in `chunks.pkl`.
