from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss

# ---------------~----------
# Load PDF
# -------------------------
reader = PdfReader("documents/ai_notes.pdf")

text = ""

for page in reader.pages:
    page_text = page.extract_text()

    if page_text:
        text += page_text

# -------------------------
# Chunking
# -------------------------
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

chunks = splitter.split_text(text)

print(f"\nTotal Chunks: {len(chunks)}")

# -------------------------
# Embedding Model
# -------------------------
print("\nLoading Model...")

model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

print("Model Loaded!")

# -------------------------
# Create Embeddings
# -------------------------
embeddings = model.encode(chunks)

embeddings = np.array(
    embeddings,
    dtype=np.float32
)

print("Embeddings Shape:", embeddings.shape)

# -------------------------
# Build FAISS Index
# -------------------------
dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)

index.add(embeddings)

print("FAISS Index Created")
print("Vectors Stored:", index.ntotal)

# -------------------------
# Query
# -------------------------
query = input("\nAsk Question: ")

query_embedding = model.encode([query])

query_embedding = np.array(
    query_embedding,
    dtype=np.float32
)

# -------------------------
# Search Top 3
# -------------------------
k = 3

distances, indices = index.search(
    query_embedding,
    k
)

print("\nTop Results\n")

for rank, idx in enumerate(indices[0], start=1):

    print("=" * 60)
    print(f"RESULT {rank}")
    print("=" * 60)

    print(chunks[idx])

    print("\nDistance:", distances[0][rank - 1])
    print()
