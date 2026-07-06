import asyncio
from src.retriever import DenseRetriever

retriever = DenseRetriever(index_name="ivf", strategy="baseline")
results = retriever.search("What is the recommended eGFR threshold for metformin use?", top_k=5)

print("Retrieved doc IDs:")
for r in results:
    print(r.chunk.doc_id, r.score, r.chunk.text[:50])
