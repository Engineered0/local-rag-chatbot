import os, chromadb, ollama
from config import CHAT_MODEL, N_RESULTS
from embedder import embed

BASE = os.path.dirname(os.path.abspath(__file__))
col = chromadb.PersistentClient(f"{BASE}/chroma_db").get_or_create_collection("docs")


def ask(question, source=None):
    kwargs = {"query_embeddings": embed([question]), "n_results": N_RESULTS}
    if source:
        kwargs["where"] = {"source": source}

    res = col.query(**kwargs)
    chunks = res["documents"][0]
    metas = res["metadatas"][0]
    dists = res["distances"][0]

    if not chunks:
        print("nothing found — run ingest.py, or check the @filename")
        return

    print("\n--- retrieved ---")
    for c, m, d in zip(chunks, metas, dists):
        loc = f"p{m['page']}" if m["page"] else ""
        print(f"[{m['source']} {loc} {m['kind']}] dist={d:.3f}")
        print(c[:250].replace("\n", " "), "\n")

    if dists[0] > 1.0:
        print("weak match — the answer may not be in your documents\n")
    if any(m["kind"] == "vision" for m in metas):
        print("part of this came from a figure description — verify in the PDF\n")
    if any(m["kind"] == "image_only" for m in metas):
        print("a diagram page matched — open it manually\n")

    prompt = (
        "Answer using only the context below.\n"
        "Each chunk is labelled with its source file and page.\n"
        "If chunks from different documents disagree, give the answer "
        "per document rather than merging them.\n"
        "If the answer is not in the context, say exactly: "
        "'Not in the documents.'\n"
        "Cite the page number shown in square brackets.\n\n"
        f"Context:\n{chr(10).join(chunks)}\n\n"
        f"Question: {question}\nAnswer:"
    )

    reply = ollama.chat(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": prompt}],
    )["message"]["content"]

    print("--- answer ---")
    print(reply)
    print("\nsources: " + ", ".join(
        sorted({f"{m['source']} p{m['page']}" for m in metas})))


def list_sources():
    everything = col.get(include=["metadatas"])
    names = sorted({m["source"] for m in everything["metadatas"]})
    for n in names:
        print(f"  {n}")


if __name__ == "__main__":
    print(f"{col.count()} chunks loaded.")
    print("commands:  @filename <question>   restrict to one file")
    print("           /files                 list documents")
    print("           quit\n")

    while True:
        try:
            q = input("ask: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if q.lower() in ("quit", "exit", ""):
            break
        if q == "/files":
            list_sources()
            continue

        src = None
        if q.startswith("@"):
            parts = q[1:].split(" ", 1)
            if len(parts) == 2:
                src, q = parts
            else:
                print("usage: @filename your question")
                continue

        ask(q, src)