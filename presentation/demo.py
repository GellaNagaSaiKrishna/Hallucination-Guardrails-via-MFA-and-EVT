"""Live demo of the MFA + EVT guardrail.

Run from the project root:
    python presentation/demo.py                 # score the four demo questions at alpha = 0.01 and 0.1
    python presentation/demo.py -i              # then type your own questions (empty line to quit)
    python presentation/demo.py --rag "..."     # full guarded RAG answer with the LLM (slow on CPU: ~30-60 s)
    python presentation/demo.py --offline       # backup: print the saved results, no models needed
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
BACKUP = HERE / "demo_backup.json"
try:  # Windows consoles default to cp1252, which cannot print α / ²
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

QUESTIONS = [
    ("What is the FY2018 capital expenditure amount (in USD millions) for 3M?", "ID"),
    ("What drove operating margin change as of FY22 for AMD?", "Near-OOD"),
    ("What was Microsoft's revenue from its lunar data centers in 2019?", "Adversarial"),
    ("Who painted the Mona Lisa?", "Far-OOD"),
]
ALPHAS = (0.01, 0.1)


def show(rows):
    print(f"\n{'question':58s} {'type':12s} {'D²':>7s} {'S(q)':>8s}  {'α=0.01':8s} {'α=0.1':8s} {'ms':>6s}")
    print("-" * 115)
    for r in rows:
        dec = ["REFUSE" if r["S"] < a else "pass" for a in ALPHAS]
        print(f"{r['q'][:57]:58s} {r['type']:12s} {r['d2']:7.1f} {r['S']:8.4f}  {dec[0]:8s} {dec[1]:8s} {r['ms']:6.2f}")
    print("\nD² = distance to nearest MFA cluster;  S(q) = EVT p-value;  refuse if S(q) < α")


def score(guard, q, kind):
    from src.embeddings import embed
    e = embed(q)[0]
    t0 = time.perf_counter()
    d = guard.check_embedding(e)
    ms = (time.perf_counter() - t0) * 1e3
    return {"q": q, "type": kind, "d2": d.d_sq, "S": d.p_value, "cluster": d.cluster, "ms": ms}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--interactive", action="store_true")
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--rag", metavar="QUESTION")
    a = ap.parse_args()

    if a.offline:
        show(json.loads(BACKUP.read_text(encoding="utf-8")))
        return

    print("Loading MiniLM, the trained MFA (K=16, r=64) and the EVT tail ...", flush=True)
    from src.guardrail import MFAEVTGuardrail
    guard = MFAEVTGuardrail.load(alpha=0.1)
    rows = [score(guard, q, k) for q, k in QUESTIONS]
    show(rows)
    BACKUP.write_text(json.dumps(rows, indent=1), encoding="utf-8")

    if a.rag:
        import functools
        import torch
        import src.pipeline as P
        from src.pipeline import RAGPipeline

        @functools.lru_cache(maxsize=1)
        def _load_llm_bf16():
            # demo only: bfloat16 halves memory (~3 GB) so the LLM loads on a busy laptop
            from transformers import AutoModelForCausalLM, AutoTokenizer
            tok = AutoTokenizer.from_pretrained(P.LLM_MODEL)
            model = AutoModelForCausalLM.from_pretrained(P.LLM_MODEL, dtype=torch.bfloat16,
                                                         low_cpu_mem_usage=True).eval()
            return tok, model
        if not torch.cuda.is_available():
            P._load_llm = _load_llm_bf16
        print(f"\nGuarded RAG (α = 0.1) for: {a.rag}\n(loading the LLM, ~30-60 s on CPU) ...", flush=True)
        r = RAGPipeline(guarded=True)(a.rag)
        print(f"blocked = {r.blocked}   S(q) = {r.p_value:.4f}\nanswer: {r.answer}")
        for c in r.contexts:
            print("  source:", c[:140].replace("\n", " "), "...")

    if a.interactive:
        while True:
            q = input("\nYour question (empty to quit): ").strip()
            if not q:
                break
            show([score(guard, q, "custom")])


if __name__ == "__main__":
    main()
