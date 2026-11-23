"""Per-stage latency of the LangGraph pipeline with the real models and vector store.

    DATABASE_URL=... python scripts/bench_service.py --out results/service_latency_local.json
    VECTOR_STORE=memory python scripts/bench_service.py              # exact numpy store

Questions: the 60 FP-pairs questions, 34 FinanceBench ID and 20 Far-OOD trivia questions, each
run once after a warm-up of 3 calls. Reports p50/p95 per node (embed, gate, retrieve,
premise_check, generate/refuse) and end to end, plus the routing outcome counts.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from service.graph import build_graph, load_components  # noqa: E402
from service.settings import get_settings  # noqa: E402


def questions() -> list[tuple[str, str]]:
    fp = json.loads((ROOT / "data" / "false_premise" / "eval_set.json").read_text(encoding="utf-8"))
    tq = json.loads((ROOT / "data" / "test_queries.json").read_text(encoding="utf-8"))
    out = [(("FP-pairs " + x["label"]), x["question"]) for x in fp]
    out += [("FB-ID", q["query_text"]) for q in tq if q["label"] == "IN_DISTRIBUTION"][1::2]
    out += [("Far-OOD", q["query_text"]) for q in tq if q["label"] == "FAR_OOD"][:20]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    s = get_settings()
    t0 = time.perf_counter()
    comps = load_components(s)
    load_s = time.perf_counter() - t0
    graph = build_graph(comps)
    qs = questions()
    for _, q in qs[:3]:
        graph.invoke({"question": q})
    stage = defaultdict(list)
    e2e, outcome = [], Counter()
    for group, q in qs:
        t1 = time.perf_counter()
        out = graph.invoke({"question": q})
        e2e.append((time.perf_counter() - t1) * 1e3)
        for k, v in out["timings_ms"].items():
            stage[k].append(v)
        outcome[(group, out.get("refusal_reason") or "answered")] += 1
    pct = lambda v: {"n": len(v), "p50_ms": round(float(np.percentile(v, 50)), 2),  # noqa: E731
                     "p95_ms": round(float(np.percentile(v, 95)), 2)}
    res = {"vector_store": s.vector_store, "premise_check": s.premise_check, "premise_model": s.premise_model,
           "generator": s.generator, "top_k": s.top_k, "ef_search": s.ef_search,
           "model_load_s": round(load_s, 1),
           "stages": {k: pct(v) for k, v in stage.items()}, "end_to_end": pct(e2e),
           "outcomes": {f"{g} -> {o}": n for (g, o), n in sorted(outcome.items())}}
    print(json.dumps(res, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
