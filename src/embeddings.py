"""Week 1 / Step 1.2 - Embedding generation (all-MiniLM-L6-v2, d = 384, L2-normalised).

Usage:  python -m src.embeddings
"""
from __future__ import annotations

import json
import time
from functools import lru_cache

import numpy as np

from .config import CHUNKS_JSON, CORPUS_EMB, EMBED_MODEL, QUERIES_JSON, QUERY_EMB


@lru_cache(maxsize=1)
def get_model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(EMBED_MODEL, device="cpu")


def embed(texts, batch_size: int = 64, progress: bool = False) -> np.ndarray:
    if isinstance(texts, str):
        texts = [texts]
    emb = get_model().encode(list(texts), batch_size=batch_size, show_progress_bar=progress,
                             normalize_embeddings=True, convert_to_numpy=True)
    return emb.astype(np.float32)


def load_chunks() -> list[dict]:
    return json.loads(CHUNKS_JSON.read_text(encoding="utf-8"))


def load_queries() -> list[dict]:
    return json.loads(QUERIES_JSON.read_text(encoding="utf-8"))


def main() -> None:
    chunks = load_chunks()
    t0 = time.perf_counter()
    X = embed([c["text"] for c in chunks], progress=True)
    print(f"Corpus embeddings {X.shape} in {time.perf_counter() - t0:.1f}s")
    np.save(CORPUS_EMB, X)

    queries = load_queries()
    Q = embed([q["query_text"] for q in queries])
    print(f"Query embeddings {Q.shape}")
    np.save(QUERY_EMB, Q)


if __name__ == "__main__":
    main()
