import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer

model = SentenceTransformer(
    "all-MiniLM-L6-v2",
    local_files_only=True
)

index = faiss.read_index(
    "vectorstore/faiss_index.bin"
)

with open(
    "vectorstore/chunks.pkl",
    "rb"
) as f:
    chunks = pickle.load(f)

query = input("Ask Question: ")

query_embedding = model.encode([query])

query_embedding = np.array(
    query_embedding,
    dtype=np.float32
)

faiss.normalize_L2(query_embedding)

distances, indices = index.search(
    query_embedding,
    3
)

for idx in indices[0]:
    print("\n")
    print(chunks[idx])