import os, chromadb, pdfplumber, pytesseract
from pdf2image import convert_from_path
from sentence_transformers import SentenceTransformer

BASE = os.path.dirname(os.path.abspath(__file__))
DOCS = f"{BASE}/docs"

model = SentenceTransformer("all-MiniLM-L6-v2")
col = chromadb.PersistentClient(f"{BASE}/chroma_db").get_or_create_collection("docs")


def is_junk(text):
    """Table-of-contents lines and stubs match everything weakly."""
    if len(text.split()) < 20:
        return True
    return text.count(".") > len(text) * 0.15


def chunk_text(text, size=400):
    """Split on paragraph boundaries so code blocks stay intact."""
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks, cur = [], []
    for p in paras:
        cur.append(p)
        if sum(len(x.split()) for x in cur) >= size:
            chunks.append("\n\n".join(cur))
            cur = cur[-1:]                # carry last para as overlap
    if cur:
        chunks.append("\n\n".join(cur))
    return chunks


def flatten_table(table, name, page):
    """One self-contained line per table row."""
    out = []
    header = [str(c or "").strip() for c in table[0]]
    for row in table[1:]:
        cells = [str(c or "").strip() for c in row]
        pairs = ", ".join(f"{h}: {v}" for h, v in zip(header, cells) if h and v)
        if pairs:
            out.append(f"[{name} p{page}] {pairs}")
    return out


def ocr(path, page):
    img = convert_from_path(path, first_page=page, last_page=page, dpi=300)[0]
    return pytesseract.image_to_string(img)


def process_pdf(path, name):
    records = []
    with pdfplumber.open(path) as pdf:
        for pno, page in enumerate(pdf.pages, 1):

            for table in page.extract_tables():
                for line in flatten_table(table, name, pno):
                    records.append((line, pno, "table"))

            text = (page.extract_text() or "").strip()
            kind = "text"

            if not text:
                text = ocr(path, pno).strip()
                kind = "ocr"
                if not text:
                    records.append((
                        f"[{name} p{pno}] Image-only page, likely a diagram. "
                        f"Open the PDF at this page.", pno, "image_only"))
                    continue

            for c in chunk_text(text):
                if not is_junk(c):
                    records.append((f"[{name} p{pno}] {c}", pno, kind))
    return records


for name in sorted(os.listdir(DOCS)):
    if name.startswith("."):
        continue
    path = f"{DOCS}/{name}"

    if name.lower().endswith(".pdf"):
        records = process_pdf(path, name)
    else:
        text = open(path, encoding="utf-8").read()
        records = [(c, 0, "text") for c in chunk_text(text) if not is_junk(c)]

    if not records:
        print(f"{name}: nothing extracted")
        continue

    texts = [r[0] for r in records]
    col.upsert(
        ids=[f"{name}-p{r[1]}-{r[2]}-{i}" for i, r in enumerate(records)],
        documents=texts,
        embeddings=model.encode(texts, show_progress_bar=True).tolist(),
        metadatas=[{"source": name, "page": r[1], "kind": r[2]} for r in records],
    )

    counts = {}
    for _, _, k in records:
        counts[k] = counts.get(k, 0) + 1
    print(f"{name}: {len(records)} chunks {counts}")

print(f"\ntotal: {col.count()}")