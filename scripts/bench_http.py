"""Latency of a running service over HTTP (local docker compose or a deployed Space).

    python scripts/bench_http.py --url http://localhost:7860 --out results/service_latency_docker.json

Same question set as scripts/bench_service.py. Per-stage times come from the `timings_ms` the
service returns; `client_ms` is the full HTTP round trip.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import httpx
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from bench_service import questions  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:7860")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    qs = questions()
    stage, client, server, outcome = defaultdict(list), [], [], Counter()
    with httpx.Client(base_url=args.url, timeout=120) as c:
        health = c.get("/healthz").json()
        for _, q in qs[:3]:
            c.post("/ask", json={"question": q})
        for group, q in qs:
            t0 = time.perf_counter()
            b = c.post("/ask", json={"question": q}).raise_for_status().json()
            client.append((time.perf_counter() - t0) * 1e3)
            server.append(b["total_ms"])
            for k, v in b["timings_ms"].items():
                stage[k].append(v)
            outcome[(group, b["refusal_reason"] or "answered")] += 1
    pct = lambda v: {"n": len(v), "p50_ms": round(float(np.percentile(v, 50)), 2),  # noqa: E731
                     "p95_ms": round(float(np.percentile(v, 95)), 2)}
    res = {"url": args.url, "health": health, "stages": {k: pct(v) for k, v in stage.items()},
           "server_total": pct(server), "client_round_trip": pct(client),
           "outcomes": {f"{g} -> {o}": n for (g, o), n in sorted(outcome.items())}}
    print(json.dumps(res, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
