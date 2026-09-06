import os, chromadb, ollama
from sentence_transformers import SentenceTransformer

BASE = os.path.dirname(os.path.abspath(__file__))

MODEL = "llama3.2:3b"
N_RESULTS = 5

model = SentenceTransformer("all-MiniLM-L6-v2")
col = chromadb.PersistentClient(f"{BASE}/chroma_db").get_or_create_collection("docs")


def ask(question, show_chunks=True):
    res = col.query(
        query_embeddings=model.encode([question]).tolist(),
        n_results=N_RESULTS,
    )
    chunks = res["documents"][0]
    metas = res["metadatas"][0]
    dists = res["distances"][0]

    if not chunks:
        print("nothing in the database — run ingest.py first")
        return

    if show_chunks:
        print("\n--- retrieved ---")
        for c, m, d in zip(chunks, metas, dists):
            print(f"[{m['source']} p{m['page']} {m['kind']}] dist={d:.3f}")
            print(c[:250].replace("\n", " "), "\n")

    if dists[0] > 1.0:
        print("weak match — the answer may not be in your documents\n")

    if any(m["kind"] == "image_only" for m in metas):
        print("a diagram page matched — check it manually\n")

    prompt = (
        "Answer using only the context below.\n"
        "If the answer is not in the context, say exactly: "
        "'Not in the documents.'\n"
        "Cite the page number shown in square brackets.\n\n"
        f"Context:\n{chr(10).join(chunks)}\n\n"
        f"Question: {question}\nAnswer:"
    )

    reply = ollama.chat(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
    )["message"]["content"]

    print("--- answer ---")
    print(reply)
    print("\nsources: " + ", ".join(
        sorted({f"{m['source']} p{m['page']}" for m in metas})))


if __name__ == "__main__":
    print(f"{col.count()} chunks loaded. Type 'quit' to exit.")
    while True:
        try:
            q = input("\nask: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if q.lower() in ("quit", "exit", ""):
            break
        ask(q)