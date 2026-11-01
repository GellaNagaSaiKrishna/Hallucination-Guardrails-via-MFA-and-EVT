"""Synthetic in-distribution *questions* for density fitting and EVT calibration.

Motivation (query-passage asymmetry): the guardrail scores user *questions*, but the corpus
consists of *passages*.  Question embeddings occupy a shifted region of the embedding space,
so a density/threshold fitted only on passages is mis-calibrated for questions.  We generate
questions from every chunk with doc2query (T5 trained on MS MARCO) and fit / calibrate on
those instead.  Questions from training chunks fit the MFA; questions from validation
chunks calibrate the EVT tail, so the hand-curated test queries are never touched.

Usage:  python -m src.synthetic_queries [--per-chunk 3]
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch

from .config import DATA, SEED
from .embeddings import embed, load_chunks

D2Q_MODEL = "doc2query/msmarco-t5-base-v1"
SYN_JSON = DATA / "synthetic_queries.json"
SYN_EMB = DATA / "synthetic_query_embeddings.npy"


def generate(texts: list[str], per_chunk: int = 3, batch_size: int = 32) -> list[list[str]]:
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    torch.manual_seed(SEED)
    tok = AutoTokenizer.from_pretrained(D2Q_MODEL)
    model = AutoModelForSeq2SeqLM.from_pretrained(D2Q_MODEL).eval()
    out: list[list[str]] = []
    t0 = time.perf_counter()
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        enc = tok(batch, max_length=256, truncation=True, padding=True, return_tensors="pt")
        with torch.inference_mode():
            gen = model.generate(**enc, max_length=48, do_sample=True, top_p=0.95,
                                 num_return_sequences=per_chunk)
        dec = tok.batch_decode(gen, skip_special_tokens=True)
        for j in range(len(batch)):
            qs = dict.fromkeys(q.strip() for q in dec[j * per_chunk:(j + 1) * per_chunk])
            out.append([q for q in qs if q])
        done = i + len(batch)
        if (i // batch_size) % 20 == 0:
            el = time.perf_counter() - t0
            print(f"  {done:>5}/{len(texts)} chunks  {el:6.0f}s  (eta {el / done * (len(texts) - done):.0f}s)",
                  flush=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-chunk", type=int, default=3)
    args = ap.parse_args()
    torch.set_num_threads(max(1, torch.get_num_threads()))
    chunks = load_chunks()
    qs = generate([c["text"] for c in chunks], args.per_chunk)
    rows = [{"chunk_index": i, "chunk_id": c["chunk_id"], "query_text": q}
            for i, (c, ql) in enumerate(zip(chunks, qs)) for q in ql]
    SYN_JSON.write_text(json.dumps(rows, indent=1, ensure_ascii=False), encoding="utf-8")
    E = embed([r["query_text"] for r in rows], progress=True)
    np.save(SYN_EMB, E)
    print(f"{len(rows):,} synthetic questions from {len(chunks):,} chunks -> {SYN_JSON}")


if __name__ == "__main__":
    main()
