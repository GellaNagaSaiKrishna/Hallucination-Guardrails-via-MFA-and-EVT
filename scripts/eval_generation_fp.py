"""End-to-end before/after on the hand-built false-premise pairs, with a real LLM.

For every FP-pairs question: company-filtered top-5 retrieval -> Qwen2.5-1.5B-Instruct (greedy,
the original study's generator and its strict "answer only from the context" prompt), run on CPU.
Then:
  * before = MFA+EVT gate only (the original system): the LLM answer is returned
  * after  = gate + NLI false-premise check: flagged questions get the fixed refusal

An answer is "substantive" if it does not match the original study's abstention regex
(src/hallucination_eval.is_abstention). For false-premise questions a substantive answer may
still correct the premise, so every substantive false-premise answer is also written to
results/false_premise/generations.jsonl for manual labelling (accepts / corrects premise).

Usage:  python scripts/eval_generation_fp.py [--model Qwen/Qwen2.5-1.5B-Instruct]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from service.generators import HFGenerator  # noqa: E402
from src.hallucination_eval import is_abstention  # noqa: E402

OUT = ROOT / "results" / "false_premise"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--method", default=None, help="checker column in scores.csv (default: service default)")
    args = ap.parse_args()
    cal = json.loads((OUT / "calibration.json").read_text())
    method = args.method or cal["service_default"]["method"]
    thr = cal["thresholds"][cal["service_default"]["model"]] if args.method is None else \
        json.loads((OUT / "summary.json").read_text())["methods"][method]["threshold"]

    from service.companies import detect_companies
    from service.store import MemoryStore
    from src.embeddings import embed
    store = MemoryStore.from_disk()
    scores = pd.read_csv(OUT / "scores.csv")
    df = scores[scores.set == "FP-pairs"].reset_index(drop=True)
    gen = HFGenerator(args.model, max_new_tokens=96)

    cache_path = OUT / "generations.jsonl"
    cache = {}
    if cache_path.exists():
        for line in cache_path.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            if r.get("model") == args.model:
                cache[r["qid"]] = r
    Q = embed(df.question.tolist())
    t0 = time.perf_counter()
    with cache_path.open("a", encoding="utf-8") as fh:
        for i, row in df.iterrows():
            if row.qid in cache:
                continue
            hits = store.search(Q[i], 5, detect_companies(row.question) or None)
            ans = gen.generate(row.question, [h.text for h in hits])
            r = {"qid": row.qid, "model": args.model, "question": row.question, "kind": row.kind,
                 "false_premise": bool(row.false_premise), "answer": ans, "abstained": is_abstention(ans),
                 "manual_label": None}
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
            fh.flush()
            cache[row.qid] = r
            print(f"{len(cache)}/{len(df)}  {time.perf_counter() - t0:.0f}s  {row.qid}  abstain={r['abstained']}",
                  flush=True)

    df["abstained"] = [cache[q]["abstained"] for q in df.qid]
    df["flagged"] = df[method] < thr
    df["refused_after"] = df.flagged | df.gate_refused | df.abstained
    df["refused_before"] = df.gate_refused | df.abstained
    res = {"model": args.model, "checker": method, "threshold": thr, "groups": {}}
    for name, mask in {"answerable": ~df.false_premise, "contradicted": df.kind.eq("contradicted"),
                       "absent": df.kind.eq("absent"), "all false-premise": df.false_premise}.items():
        sub = df[mask]
        res["groups"][name] = {
            "n": int(len(sub)),
            "substantive_answer_before": float((~sub.refused_before).mean()),
            "substantive_answer_after": float((~sub.refused_after).mean()),
        }
    (OUT / "generation_summary.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
