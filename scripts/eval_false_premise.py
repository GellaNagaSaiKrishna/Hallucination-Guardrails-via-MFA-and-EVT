"""Evaluate post-retrieval false-premise checkers, with honest calibration/test separation.

Question sets
  * FP-pairs   data/false_premise/eval_set.json: 30 hand-written matched pairs built from corpus
               chunks (30 answerable, 20 contradicted-premise, 10 absent-premise). TEST ONLY.
  * FB-ID      the 68 FinanceBench in-distribution questions (answerable, evidence verified)
  * ADV        the 100 counterfactual "Near-OOD adversarial" questions (absent premises about
               indexed companies) from the original project
  FB-ID and ADV are split by index parity: even -> calibration, odd -> test.

Protocol
  1. every question: MiniLM embed -> MFA+EVT gate (alpha=0.01) -> top-k cosine retrieval (exact,
     identical ranking to pgvector without HNSW approximation) -> checker score
  2. method selection: AUROC on the calibration split only
  3. threshold: the q-quantile of calibration *answerable* scores (q = target false-refusal rate),
     a split-conformal rule; no false-premise labels are used to set it
  4. report on the test split: false refusals on answerable questions, false-premise catch rate,
     and the end-to-end share of false-premise questions that still reach the generator

Usage:  python scripts/eval_false_premise.py [--k 5] [--target-frr 0.05] [--methods qa-minilm,nli-xsmall,...]
Writes results/false_premise/{scores.csv, summary.json, calibration.json}
(or results/false_premise_nofilter/ with --no-company-filter)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from service.premise import NLIPresuppositionChecker, QAAnswerabilityChecker  # noqa: E402
from service.companies import detect_companies  # noqa: E402
from service.store import MemoryStore  # noqa: E402
from src.embeddings import embed  # noqa: E402
from src.guardrail import MFAEVTGuardrail  # noqa: E402

OUT = ROOT / "results" / "false_premise"            # company-filtered retrieval (service default)
OUT_NOFILTER = ROOT / "results" / "false_premise_nofilter"
METHODS = {
    "qa-minilm": ("qa", "deepset/minilm-uncased-squad2"),
    "qa-tinyroberta": ("qa", "deepset/tinyroberta-squad2"),
    "nli-xsmall": ("nli", "cross-encoder/nli-deberta-v3-xsmall"),
    "nli-minilm2": ("nli", "cross-encoder/nli-MiniLM2-L6-H768"),
    "nli-deberta-base": ("nli", "cross-encoder/nli-deberta-v3-base"),
    "ce-relevance": ("ce", "cross-encoder/ms-marco-MiniLM-L-6-v2"),
}


class CERelevance:
    """Baseline: max query-passage relevance from an MS MARCO cross-encoder (a reranker cutoff)."""

    def __init__(self, model_name: str):
        from sentence_transformers import CrossEncoder
        self.model = CrossEncoder(model_name, device="cpu")

    def score(self, q: str, ctx: list[str]) -> float:
        return float(np.max(self.model.predict([(q, c) for c in ctx])))


def load_questions() -> pd.DataFrame:
    rows = []
    for x in json.loads((ROOT / "data" / "false_premise" / "eval_set.json").read_text(encoding="utf-8")):
        rows.append({"qid": x["id"], "set": "FP-pairs", "question": x["question"], "company": x["company"],
                     "false_premise": x["label"] == "false_premise",
                     "kind": x["fp_type"] or "answerable", "split": "test",
                     "evidence_chunk_id": x["evidence_chunk_id"]})
    tq = json.loads((ROOT / "data" / "test_queries.json").read_text(encoding="utf-8"))
    for label, name, fp, kind in (("IN_DISTRIBUTION", "FB-ID", False, "answerable"),
                                  ("NEAR_OOD_ADV", "ADV", True, "absent")):
        sub = [q for q in tq if q["label"] == label]
        for i, q in enumerate(sub):
            rows.append({"qid": q["query_id"], "set": name, "question": q["query_text"],
                         "company": q.get("company"), "false_premise": fp, "kind": kind,
                         "split": "calib" if i % 2 == 0 else "test", "evidence_chunk_id": None})
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--alpha", type=float, default=0.01)
    ap.add_argument("--target-frr", type=float, default=0.05)
    ap.add_argument("--methods", default=",".join(METHODS))
    ap.add_argument("--latency-budget-ms", type=float, default=500.0)
    ap.add_argument("--no-company-filter", action="store_true",
                    help="plain top-k over all 50 filings (writes results/false_premise_nofilter)")
    args = ap.parse_args()
    out_dir = OUT_NOFILTER if args.no_company_filter else OUT
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_questions()
    store = MemoryStore.from_disk()
    guard = MFAEVTGuardrail.load(alpha=args.alpha)
    t0 = time.perf_counter()
    Q = embed(df.question.tolist())
    print(f"embedded {len(df)} questions in {time.perf_counter() - t0:.1f}s")
    dec = [guard.check_embedding(q) for q in Q]
    df["gate_S"] = [d.p_value for d in dec]
    df["gate_refused"] = [d.is_ood for d in dec]
    comps = [[] if args.no_company_filter else detect_companies(t) for t in df.question]
    df["filter_companies"] = [";".join(c) for c in comps]
    hits = [store.search(q, args.k, c or None) for q, c in zip(Q, comps)]
    ctxs = [[h.text for h in hs] for hs in hits]
    df["retrieved_companies"] = [";".join(sorted({h.company for h in hs})) for hs in hits]
    df["evidence_retrieved"] = [ev in {h.chunk_id for h in hs} if ev else np.nan
                                for ev, hs in zip(df.evidence_chunk_id, hits)]

    # reuse scores of methods already computed with the same questions and k (results/.../scores.csv)
    cache, latency = None, {}
    if (out_dir / "scores.csv").exists() and (out_dir / "summary.json").exists():
        old = pd.read_csv(out_dir / "scores.csv")
        old_sum = json.loads((out_dir / "summary.json").read_text())
        if old_sum.get("k") == args.k and old.qid.tolist() == df.qid.tolist():
            cache, latency = old, old_sum.get("latency", {})
    for name in args.methods.split(","):
        if cache is not None and name in cache.columns and name in latency:
            df[name] = cache[name].to_numpy()
            print(f"[{name}] cached scores")
            continue
        kind, model = METHODS[name]
        t0 = time.perf_counter()
        if kind == "qa":
            chk = QAAnswerabilityChecker(model)
            fn = lambda q, c: chk.check(q, c).score  # noqa: E731
        elif kind == "nli":
            chk = NLIPresuppositionChecker(model)
            fn = lambda q, c: chk.check(q, c).score  # noqa: E731
        else:
            chk = CERelevance(model)
            fn = chk.score
        print(f"[{name}] loaded in {time.perf_counter() - t0:.1f}s")
        times, scores = [], []
        for q, c in zip(df.question, ctxs):
            t1 = time.perf_counter()
            scores.append(fn(q, c))
            times.append((time.perf_counter() - t1) * 1e3)
        df[name] = scores
        latency[name] = {"p50_ms": float(np.percentile(times, 50)), "p95_ms": float(np.percentile(times, 95))}
        print(f"[{name}] p50 {latency[name]['p50_ms']:.1f} ms  p95 {latency[name]['p95_ms']:.1f} ms (k={args.k})")

    df.to_csv(OUT / "scores.csv", index=False)
    summary = {"k": args.k, "alpha": args.alpha, "company_filter": not args.no_company_filter, "target_frr": args.target_frr, "latency": latency,
               "n": df.groupby(["set", "split", "kind"]).size().rename("n").reset_index().to_dict("records"),
               "evidence_retrieved_fp_pairs": float(df.loc[df.set == "FP-pairs", "evidence_retrieved"].mean()),
               "methods": {}}
    cal, test = df[df.split == "calib"], df[df.split == "test"]
    for name in args.methods.split(","):
        auroc_cal = roc_auc_score(cal.false_premise, -cal[name])
        thr = float(np.quantile(cal.loc[~cal.false_premise, name], args.target_frr))
        m = {"auroc_calib": auroc_cal, "threshold": thr}
        for set_name in ("FB-ID", "ADV", "FP-pairs"):
            s = test[test.set == set_name]
            m[f"auroc_test_{set_name}"] = (roc_auc_score(s.false_premise, -s[name])
                                           if s.false_premise.nunique() == 2 else None)
        # FP-pairs + odd FB-ID + odd ADV pooled
        m["auroc_test_all"] = roc_auc_score(test.false_premise, -test[name])
        flag = test[name] < thr
        refused = test.gate_refused | flag
        groups = {"FB-ID answerable": test.set.eq("FB-ID"),
                  "FP-pairs answerable": test.set.eq("FP-pairs") & ~test.false_premise,
                  "FP-pairs contradicted": test.kind.eq("contradicted"),
                  "FP-pairs absent": test.set.eq("FP-pairs") & test.kind.eq("absent"),
                  "ADV absent": test.set.eq("ADV")}
        m["test"] = {}
        for g, mask in groups.items():
            m["test"][g] = {"n": int(mask.sum()),
                            "gate_only_refused": float(test.gate_refused[mask].mean()),
                            "check_flagged": float(flag[mask].mean()),
                            "gate_plus_check_refused": float(refused[mask].mean())}
        summary["methods"][name] = m
    best = max(summary["methods"], key=lambda n: summary["methods"][n]["auroc_calib"])
    # service default: best calibration AUROC among checkers within a CPU latency budget
    # (free CPU hosting). NOTE: this budget was added after the first run had been inspected.
    fast = [n for n in summary["methods"] if latency[n]["p50_ms"] <= args.latency_budget_ms]
    default = max(fast, key=lambda n: summary["methods"][n]["auroc_calib"]) if fast else best
    summary["selected_by_calib_auroc"] = best
    summary["service_default"] = default
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    (out_dir / "calibration.json").write_text(json.dumps(
        {"service_default": {"method": default, "kind": METHODS[default][0], "model": METHODS[default][1],
                             "threshold": summary["methods"][default]["threshold"]},
         "thresholds": {METHODS[n][1]: summary["methods"][n]["threshold"] for n in summary["methods"]},
         "k": args.k, "company_filter": not args.no_company_filter, "target_frr": args.target_frr, "latency_budget_ms": args.latency_budget_ms,
         "selected_by_calib_auroc": best,
         "calibrated_on": "even-indexed FinanceBench ID questions (answerable scores only); "
                          "threshold = target_frr quantile"}, indent=2))

    print(f"\nselected on calibration AUROC: {best}; service default "
          f"(p50 <= {args.latency_budget_ms:.0f} ms): {default}")
    for name, m in summary["methods"].items():
        print(f"\n== {name}: calib AUROC {m['auroc_calib']:.3f}, test AUROC all {m['auroc_test_all']:.3f}, "
              f"FP-pairs {m['auroc_test_FP-pairs']:.3f}, threshold {m['threshold']:.3f}")
        for g, r in m["test"].items():
            print(f"   {g:24s} n={r['n']:3d}  gate {r['gate_only_refused']:.0%}  check {r['check_flagged']:.0%}"
                  f"  gate+check {r['gate_plus_check_refused']:.0%}")


if __name__ == "__main__":
    main()
