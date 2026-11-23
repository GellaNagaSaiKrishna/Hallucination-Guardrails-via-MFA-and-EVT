r"""LangGraph orchestration of the guarded RAG pipeline.

    START -> embed -> gate --(S(q) < alpha)------------------------------> refuse -> END
                         \-> retrieve -> premise_check --(unsupported)--> refuse -> END
                                                       \-> generate --------------> END

* gate:          MFA + Woodbury distance, EVT p-value (pre-retrieval, ~1 ms)
* retrieve:      pgvector cosine top-k (or the in-memory store), restricted to the filings of a
                 company named in the question when one is detected
* premise_check: NLI of the question's presupposition against the retrieved chunks (post-retrieval)
* generate:      pluggable generator (stub | extractive | hf:<model>)

Every node records its wall time in state["timings_ms"].
"""
from __future__ import annotations

import operator
import time
from dataclasses import dataclass
from typing import Annotated, Any, Callable, TypedDict

import numpy as np
from langgraph.graph import END, START, StateGraph

from .companies import detect_companies

OOD_MESSAGE = "Query outside knowledge base: I don't have information about this."
PREMISE_MESSAGE = ("I can't answer this from the filings: the retrieved passages do not support "
                   "the question's premise.")


def _merge(a: dict, b: dict) -> dict:
    return {**(a or {}), **(b or {})}


class GraphState(TypedDict, total=False):
    question: str
    top_k: int
    embedding: Any
    gate: dict
    hits: list
    companies: list
    premise: dict
    answer: str
    refused: bool
    refusal_reason: str | None
    timings_ms: Annotated[dict, _merge]
    trace: Annotated[list, operator.add]


@dataclass
class Components:
    embed: Callable[[str], np.ndarray]     # question -> (d,) L2-normalised embedding
    guard: Any                             # .check_embedding(q) -> GuardrailDecision
    store: Any                             # .search(q, k) -> list[Hit]
    checker: Any                           # .check(question, contexts) -> PremiseResult
    generator: Any                         # .generate(question, contexts, evidence) -> str
    top_k: int = 5
    company_filter: bool = True


def build_graph(c: Components):
    def timed(name: str, fn):
        def node(state: GraphState) -> dict:
            t0 = time.perf_counter()
            out = fn(state)
            out["timings_ms"] = {name: (time.perf_counter() - t0) * 1e3}
            out["trace"] = [name]
            return out
        return node

    def embed(state: GraphState) -> dict:
        return {"embedding": np.asarray(c.embed(state["question"]), dtype=np.float32).ravel()}

    def gate(state: GraphState) -> dict:
        d = c.guard.check_embedding(state["embedding"])
        return {"gate": {"ood": bool(d.is_ood), "p_value": float(d.p_value), "d_sq": float(d.d_sq),
                         "cluster": int(d.cluster), "alpha": float(getattr(c.guard, "alpha", float("nan")))}}

    def retrieve(state: GraphState) -> dict:
        companies = detect_companies(state["question"]) if c.company_filter else []
        hits = c.store.search(state["embedding"], state.get("top_k") or c.top_k, companies or None)
        return {"hits": hits, "companies": companies}

    def premise_check(state: GraphState) -> dict:
        r = c.checker.check(state["question"], [h.text for h in state["hits"]])
        return {"premise": {"supported": bool(r.supported), "score": float(r.score),
                            "threshold": float(r.threshold), "method": r.method,
                            "best_chunk": r.best_chunk, "evidence": r.evidence}}

    def generate(state: GraphState) -> dict:
        ev = (state.get("premise") or {}).get("evidence")
        ans = c.generator.generate(state["question"], [h.text for h in state["hits"]], ev)
        return {"answer": ans, "refused": False, "refusal_reason": None}

    def refuse(state: GraphState) -> dict:
        if state.get("gate", {}).get("ood"):
            return {"answer": OOD_MESSAGE, "refused": True, "refusal_reason": "out_of_distribution"}
        return {"answer": PREMISE_MESSAGE, "refused": True, "refusal_reason": "unsupported_premise"}

    def after_gate(state: GraphState) -> str:
        return "refuse" if state["gate"]["ood"] else "retrieve"

    def after_premise(state: GraphState) -> str:
        return "generate" if state["premise"]["supported"] else "refuse"

    g = StateGraph(GraphState)
    for name, fn in [("embed", embed), ("gate", gate), ("retrieve", retrieve),
                     ("premise_check", premise_check), ("generate", generate), ("refuse", refuse)]:
        g.add_node(name, timed(name, fn))
    g.add_edge(START, "embed")
    g.add_edge("embed", "gate")
    g.add_conditional_edges("gate", after_gate, {"refuse": "refuse", "retrieve": "retrieve"})
    g.add_edge("retrieve", "premise_check")
    g.add_conditional_edges("premise_check", after_premise, {"generate": "generate", "refuse": "refuse"})
    g.add_edge("generate", END)
    g.add_edge("refuse", END)
    return g.compile()


def load_components(settings) -> Components:
    """Real models: MiniLM embedder, MFA+EVT gate, pgvector/memory store, premise checker, generator."""
    import json
    import math
    from pathlib import Path

    from src.embeddings import embed as _embed
    from src.guardrail import MFAEVTGuardrail

    from .generators import make_generator
    from .premise import make_checker
    from .store import make_store

    thr = settings.premise_threshold
    model = settings.premise_model
    if math.isnan(thr):
        cal = Path(__file__).resolve().parent.parent / "results" / "false_premise" / "calibration.json"
        thr = 0.0
        if cal.exists():
            thr = float(json.loads(cal.read_text()).get("thresholds", {}).get(model, 0.0))
    return Components(
        embed=lambda q: _embed(q)[0],
        guard=MFAEVTGuardrail.load(alpha=settings.alpha),
        store=make_store(settings),
        checker=make_checker(settings.premise_check, model, thr),
        generator=make_generator(settings.generator, settings.max_new_tokens),
        top_k=settings.top_k,
        company_filter=settings.company_filter,
    )
