import os, chromadb, ollama
from sentence_transformers import SentenceTransformer

BASE = os.path.dirname(os.path.abspath(__file__))

model = SentenceTransformer("all-MiniLM-L6-v2")
col = chromadb.PersistentClient(f"{BASE}/chroma_db").get_or_create_collection("docs")

def ask(question):
    res = col.query(
        query_embeddings=model.encode([question]).tolist(),
        n_results=3,
    )
    chunks = res["documents"][0]
    sources = {m["source"] for m in res["metadatas"][0]}

    print("\n--- retrieved ---")
    for c in chunks:
        print(c[:200], "\n")

    prompt = (
        "Answer using only the context below.\n"
        "If the answer is not there, say 'Not in the documents.'\n\n"
        f"Context:\n{chr(10).join(chunks)}\n\n"
        f"Question: {question}\nAnswer:"
    )

    reply = ollama.chat(
        model="llama3.2:3b",
        messages=[{"role": "user", "content": prompt}],
    )["message"]["content"]

    print("--- answer ---")
    print(reply)
    print(f"\nsources: {', '.join(sources)}")

while True:
    q = input("\nask: ")
    if q.lower() in ("quit", "exit", ""):
        break
    ask(q)