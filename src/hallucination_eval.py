"""Week 4 / Step 4.2 - Downstream hallucination evaluation: unguarded vs guarded RAG.

For every test query we run FAISS top-k retrieval + local LLM generation (unguarded).  The
guarded conditions apply the MFA+EVT guardrail first; blocked queries receive the fixed refusal
and never reach the LLM, all others receive exactly the same (deterministic, greedy) answer.

Two prompt regimes are evaluated:
  * strict : the system prompt tells the LLM to answer only from the context and to reply
             "I don't have information about this." otherwise   (all 350 queries)
  * naive  : a plain "use the context to answer" prompt without an abstention instruction
             (the 200 OOD queries - this is the regime the guardrail is designed to protect)

Metrics
  * Faithfulness (Ragas-style, judged by NLI instead of an LLM): the answer is split into
    sentences ("claims"); a claim is supported if a retrieved chunk (or the joined context)
    entails it (cross-encoder/nli-deberta-v3-base, P(entail) > 0.5).  Faithfulness =
    supported / claims.  Abstentions and refusals contain no claims and are reported separately.
  * Hallucination rate (OOD): fraction of OOD queries that receive a substantive
    (non-abstaining) answer.  The knowledge base cannot support any answer to these queries,
    so every substantive answer is ungrounded by construction.

Usage:  python -m src.hallucination_eval [--generate-only]     (resumable; caches generations)
"""
from __future__ import annotations

import argparse
import json
import re
import time

import numpy as np
import pandas as pd

from .config import ALPHA, NLI_MODEL, QUERY_EMB, RESULTS
from .embeddings import load_queries
from .guardrail import REJECTION_MESSAGE, FaissRetriever
from .pipeline import NAIVE_SYSTEM_PROMPT, SYSTEM_PROMPT, generate_answer

ALPHA_QCAL = 0.1   # question-calibrated EVT: tail starts at the median -> 5% expected ID false-rejection
REGIMES = {
    "strict": {"prompt": SYSTEM_PROMPT, "cache": RESULTS / "rag_generations.jsonl", "labels": None},
    "naive": {"prompt": NAIVE_SYSTEM_PROMPT, "cache": RESULTS / "rag_generations_naive.jsonl",
              "labels": ("NEAR_OOD", "NEAR_OOD_ADV", "FAR_OOD")},
}
ABSTAIN_RE = re.compile(
    r"(don'?t|do not|does not|doesn'?t|did not|didn'?t) (specifically )?"
    r"(have|contain|provide|include|mention|specify|see)|"
    r"no (specific |significant )?(information|mention|data)|no details|"
    r"not (provided|mentioned|available|specified|included|found)|"
    r"\bno\b[^.]{0,60}\bmentioned in the (given|provided)|"
    r"(cannot|can'?t|unable to) (find|answer|determine|provide)|outside (the|my) knowledge", re.I)


def is_abstention(answer: str) -> bool:
    return bool(ABSTAIN_RE.search(answer)) or not answer.strip()


def split_claims(answer: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])", answer.strip())
    return [p.strip() for p in parts if len(re.findall(r"[A-Za-z]{2,}", p)) >= 3]


class NLIJudge:
    def __init__(self):
        from sentence_transformers import CrossEncoder
        self.model = CrossEncoder(NLI_MODEL, device="cpu")
        cfg = self.model.model.config.id2label
        self.entail_idx = next(i for i, l in cfg.items() if l.lower().startswith("entail"))

    def faithfulness(self, answer: str, contexts: list[str]) -> tuple[float, int]:
        claims = split_claims(answer)
        if not claims or not contexts:
            return float("nan"), 0
        premises = list(contexts) + ["\n".join(contexts)]      # single chunks + the joined context
        pairs = [(c, cl) for cl in claims for c in premises]
        logits = np.asarray(self.model.predict(pairs, batch_size=32))
        probs = np.exp(logits - logits.max(1, keepdims=True))
        probs /= probs.sum(1, keepdims=True)
        ent = probs[:, self.entail_idx].reshape(len(claims), len(premises)).max(1)
        return float(np.mean(ent > 0.5)), len(claims)


def run_generation(queries, Q, retriever: FaissRetriever, regime: str) -> dict:
    cfg = REGIMES[regime]
    cache_path = cfg["cache"]
    cache = {}
    if cache_path.exists():
        for line in cache_path.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            cache[r["query_id"]] = r
    todo = [(i, q) for i, q in enumerate(queries) if q["query_id"] not in cache
            and (cfg["labels"] is None or q["label"] in cfg["labels"])]
    print(f"[{regime}] {len(cache)} cached generations, {len(todo)} to go", flush=True)
    t0 = time.perf_counter()
    with cache_path.open("a", encoding="utf-8") as fh:
        for n, (i, q) in enumerate(todo, 1):
            docs = retriever.search_embedding(Q[i])
            ctx = [d.page_content for d in docs]
            ans = generate_answer(q["query_text"], ctx, system_prompt=cfg["prompt"])
            r = {"query_id": q["query_id"], "label": q["label"], "query_text": q["query_text"],
                 "contexts": ctx, "chunk_ids": [d.metadata["chunk_id"] for d in docs], "answer": ans}
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
            fh.flush()
            cache[q["query_id"]] = r
            if n % 10 == 0:
                el = time.perf_counter() - t0
                print(f"  {n}/{len(todo)}  {el:.0f}s  eta {el / n * (len(todo) - n):.0f}s", flush=True)
    return cache


def evaluate_regime(regime: str, gens: dict, queries, qs: pd.DataFrame, judge: NLIJudge):
    rows = []
    for q in queries:
        if q["query_id"] not in gens:
            continue
        g = gens[q["query_id"]]
        faith, n_claims = judge.faithfulness(g["answer"], g["contexts"])
        rows.append({
            "regime": regime, "query_id": q["query_id"], "label": q["label"], "query_text": q["query_text"],
            "d_sq": qs.loc[q["query_id"], "guard_D2"], "S": qs.loc[q["query_id"], "guard_S"],
            "S_question_cv": qs.loc[q["query_id"], "guard_S_question_cv"],
            "unguarded_answer": g["answer"], "unguarded_abstain": is_abstention(g["answer"]),
            "unguarded_faithfulness": faith, "n_claims": n_claims, "unguarded_blocked": False,
        })
    df = pd.DataFrame(rows)
    conditions = {
        "guarded": qs["guard_S"] < ALPHA,                           # proposal: passage EVT, alpha = 0.01
        "guarded_qcal": qs["guard_S_question_cv"] < ALPHA_QCAL,     # ablation: ID-question calibrated EVT
    }
    for cond, blocked in conditions.items():
        b = blocked.loc[df.query_id].to_numpy()
        df[f"{cond}_blocked"] = b
        df[f"{cond}_answer"] = np.where(b, REJECTION_MESSAGE, df.unguarded_answer)
        df[f"{cond}_abstain"] = b | df.unguarded_abstain
        df[f"{cond}_faithfulness"] = np.where(b, np.nan, df.unguarded_faithfulness)
    return df


def summarise(df: pd.DataFrame) -> dict:
    def block(cond: str, sub: pd.DataFrame) -> dict:
        ab = sub[f"{cond}_abstain"]
        f = sub[f"{cond}_faithfulness"]
        return {"n": len(sub), "answered": int((~ab).sum()), "answer_rate": float((~ab).mean()),
                "blocked_rate": float(sub[f"{cond}_blocked"].mean()),
                "faithfulness_answered": float(np.nanmean(f[~ab])) if (~ab).any() else float("nan"),
                "unsupported_answer_rate": float(((~ab) & (f.fillna(0) < 0.5)).mean())}

    out = {}
    for cond in ("unguarded", "guarded", "guarded_qcal"):
        out[cond] = {}
        for name, mask in [("ID", df.label == "IN_DISTRIBUTION"), ("Near-OOD", df.label == "NEAR_OOD"),
                           ("Near-OOD adv.", df.label == "NEAR_OOD_ADV"), ("Far-OOD", df.label == "FAR_OOD"),
                           ("All OOD", df.label != "IN_DISTRIBUTION")]:
            if mask.any():
                out[cond][name] = block(cond, df[mask])
    return out


COLS = {"unguarded": "Unguarded RAG", "guarded": f"Guarded (EVT, α={ALPHA})",
        "guarded_qcal": f"Guarded (question-calibrated EVT, α={ALPHA_QCAL})"}
METRICS = [
    ("Faithfulness, ID (answered queries)", "faithfulness_answered", "ID"),
    ("ID answer rate", "answer_rate", "ID"),
    ("ID queries wrongly blocked", "blocked_rate", "ID"),
    ("OOD queries blocked by guardrail", "blocked_rate", "All OOD"),
    ("Hallucination rate, all OOD", "answer_rate", "All OOD"),
    ("Hallucination rate, Near-OOD", "answer_rate", "Near-OOD"),
    ("Hallucination rate, Near-OOD adv.", "answer_rate", "Near-OOD adv."),
    ("Hallucination rate, Far-OOD", "answer_rate", "Far-OOD"),
    ("Unsupported answers (NLI), all OOD", "unsupported_answer_rate", "All OOD"),
]


def plot_hallucination(summary: dict) -> None:
    """Grouped bars: share of OOD questions that get a substantive (ungrounded) answer, per prompt
    regime and OOD type, without and with the guardrail."""
    from .plotting import SERIES, TEXT2, apply_style, plt
    apply_style()
    groups = ["Near-OOD", "Near-OOD adv.", "Far-OOD", "All OOD"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
    for ax, regime in zip(axes, ("naive", "strict")):
        x = np.arange(len(groups))
        for i, (cond, color) in enumerate(zip(COLS, SERIES)):
            vals = [summary[regime][cond].get(g, {}).get("answer_rate", np.nan) for g in groups]
            bars = ax.bar(x + (i - 1) * 0.27, vals, width=0.25, color=color, label=COLS[cond])
            for b, v in zip(bars, vals):
                ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.0%}", ha="center", fontsize=7, color=TEXT2)
        ax.set_xticks(x, groups)
        ax.set_title(f"{regime} prompt")
    axes[0].set_ylabel("OOD questions answered\n(= hallucination rate)")
    axes[0].set_ylim(0, 0.85)
    axes[1].legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(RESULTS / "hallucination_rates.png")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--generate-only", action="store_true",
                    help="only produce/cache the unguarded LLM answers (independent of the guardrail)")
    args = ap.parse_args()
    queries = load_queries()
    Q = np.load(QUERY_EMB)
    retriever = FaissRetriever.from_disk()
    gens = {r: run_generation(queries, Q, retriever, r) for r in REGIMES}
    if args.generate_only:
        return

    qs = pd.read_csv(RESULTS / "query_scores.csv").set_index("query_id")
    judge = NLIJudge()
    frames, summary, table = [], {"alpha": ALPHA, "alpha_question_calibrated": ALPHA_QCAL}, []
    for regime in REGIMES:
        df = evaluate_regime(regime, gens[regime], queries, qs, judge)
        frames.append(df)
        summary[regime] = summarise(df)
        for metric, key, sub in METRICS:
            if sub in summary[regime]["unguarded"]:
                table.append({"Prompt": regime, "Metric": metric,
                              **{COLS[c]: summary[regime][c][sub][key] for c in COLS}})
    pd.concat(frames).to_csv(RESULTS / "rag_eval_per_query.csv", index=False)
    (RESULTS / "rag_eval_summary.json").write_text(json.dumps(summary, indent=2))
    tdf = pd.DataFrame(table)
    tdf.to_csv(RESULTS / "rag_comparison.csv", index=False)
    plot_hallucination(summary)
    pd.set_option("display.width", 250)
    print(tdf.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
