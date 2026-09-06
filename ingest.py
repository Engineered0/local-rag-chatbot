import os, chromadb
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

BASE = os.path.dirname(os.path.abspath(__file__))

model = SentenceTransformer("all-MiniLM-L6-v2")
col = chromadb.PersistentClient(f"{BASE}/chroma_db").get_or_create_collection("docs")

def load(path):
    if path.endswith(".pdf"):
        r = PdfReader(path)
        return "\n\n".join(p.extract_text() or "" for p in r.pages)
    return open(path, encoding="utf-8").read()

def chunk(text, size=500, overlap=50):
    w = text.split()
    return [" ".join(w[i:i+size]) for i in range(0, len(w), size - overlap)]

for name in os.listdir(f"{BASE}/docs"):
    if name.startswith("."):
        continue
    text = load(f"{BASE}/docs/{name}")
    if not text.strip():
        print(f"{name}: no text found, skipping")
        continue
    pieces = chunk(text)
    col.add(
        ids=[f"{name}-{i}" for i in range(len(pieces))],
        documents=pieces,
        embeddings=model.encode(pieces).tolist(),
        metadatas=[{"source": name} for _ in pieces],
    )
    print(f"{name}: {len(pieces)} chunks")

print(f"total: {col.count()}")