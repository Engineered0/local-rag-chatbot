import ollama
from config import EMBED_MODEL


def embed(texts):
    """list of strings -> list of vectors"""
    r = ollama.embed(model=EMBED_MODEL, input=texts)
    return r["embeddings"]