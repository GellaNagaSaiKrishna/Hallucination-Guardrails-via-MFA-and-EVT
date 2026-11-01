"""Weeks 2-4 - Full benchmark: BIC model selection, EVT calibration, baselines,
OOD metrics, operating points, plots and latency profiling.

Two density-fitting sources are compared for every method:
  * passage : embeddings of the training corpus chunks (as in the proposal)
  * query   : embeddings of doc2query questions generated from the training chunks
EVT / thresholds are always calibrated on the matching *validation* split (held-out chunks
or questions generated from held-out chunks).  The hand-curated test queries are used only
for evaluation.

Usage:  python -m src.run_experiments [--fast]
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd
from scipy import stats

from . import baselines as B
from .config import (ALPHA, CORPUS_EMB, DATA, K_GRID, MODELS, QUERY_EMB, R_GRID, RESULTS, SEED,
                     SPLIT_NPZ, VAL_DIST, VAL_FRACTION)
from .embeddings import load_queries
from .evt import WeibullTail
from .metrics import roc_points, split_metrics
from .mfa import MFA
from .plotting import SPLIT_COLORS, SPLIT_NAMES, TEXT2, color_for, plt
from .synthetic_queries import SYN_EMB, SYN_JSON
from .woodbury import naive_mahalanobis_sq

MAIN_SOURCE = "passage"   # query-fit was evaluated as an ablation and found worse (see report)
SOURCE_NAMES = {"passage": "Passage-fit", "query": "Query-fit"}
EVT_NAME = "MFA + EVT (proposed)"
ALPHAS = (0.5, 0.2, 0.1, 0.05, 0.01, 0.001)


# ----------------------------------------------------------------------------- data
def load_data():
    X = np.load(CORPUS_EMB).astype(np.float64)
    Q = np.load(QUERY_EMB).astype(np.float64)
    queries = load_queries()
    labels = np.array([q["label"] for q in queries])
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(X))
    n_val = int(VAL_FRACTION * len(X))
    val_idx, train_idx = np.sort(perm[:n_val]), np.sort(perm[n_val:])
    np.savez(SPLIT_NPZ, train_idx=train_idx, val_idx=val_idx)

    sources = {"passage": (X[train_idx], X[val_idx])}
    if SYN_EMB.exists():
        S = np.load(SYN_EMB).astype(np.float64)
        chunk_of = np.array([r["chunk_index"] for r in json.loads(SYN_JSON.read_text(encoding="utf-8"))])
        is_val = np.isin(chunk_of, val_idx)
        sources["query"] = (S[~is_val], S[is_val])
    return sources, Q, labels, queries


# ----------------------------------------------------------------------------- model selection
def bic_grid(Xtr: np.ndarray, Ks, Rs, source: str):
    rows, models = [], {}
    for K in Ks:
        for r in Rs:
            m = MFA(n_components=K, rank=r, random_state=SEED).fit(Xtr)
            ll = m.log_likelihood(Xtr)
            rows.append({"source": source, "K": K, "r": r, "log_likelihood": ll,
                         "n_params": m.n_parameters(), "BIC": m.bic(Xtr), "iterations": m.n_iter_,
                         "converged": m.converged_, "fit_seconds": m.fit_time_})
            models[(K, r)] = m
            print(f"  [{source}] K={K:2d} r={r:2d}  logL={ll:14,.1f}  BIC={rows[-1]['BIC']:14,.1f}"
                  f"  iters={m.n_iter_:2d}  {m.fit_time_:5.1f}s", flush=True)
    df = pd.DataFrame(rows)
    best = df.loc[df.BIC.idxmin()]
    return df, models, (int(best.K), int(best.r))


# ----------------------------------------------------------------------------- scoring
def score_all(source: str, Xtr, Xva, Q, K: int, mfa: MFA, evt: WeibullTail):
    """Return {method: (test_scores, val_scores)}; higher = more OOD."""
    out = {
        "MFA min-Mahalanobis D²": (mfa.min_mahalanobis(Q), mfa.min_mahalanobis(Xva)),
        "MFA neg. log-likelihood": (-mfa.score_samples(Q), -mfa.score_samples(Xva)),
    }
    # S(q) is a monotone (non-increasing) transform of D^2 that saturates at 1 below tau_0, so
    # ranking by 1 - S only adds ties.  For threshold-free metrics we rank by D^2 itself; EVT
    # matters for the *operating point* (see operating_points).
    out[EVT_NAME] = out["MFA min-Mahalanobis D²"]
    for b in [B.CosineFAISS(), B.KMeansEuclidean(K, SEED), B.DiagGMM(K, SEED),
              B.FullGMM(K, SEED), B.LedoitWolfMahalanobis()]:
        t0 = time.perf_counter()
        b.fit(Xtr)
        out[b.name] = (b.score(Q), b.score(Xva))
        print(f"  [{source}] baseline {b.name:30s} fitted in {time.perf_counter() - t0:5.1f}s", flush=True)
    return out


def operating_points(source, scores, labels, evt: WeibullTail, mfa_d_test):
    rows = []

    def rates(reject):
        return {f"{SPLIT_NAMES[l]} {'pass' if l == 'IN_DISTRIBUTION' else 'reject'} rate":
                float(np.mean(~reject[labels == l]) if l == "IN_DISTRIBUTION" else np.mean(reject[labels == l]))
                for l in SPLIT_COLORS}

    # every scorer at an empirical 99th-percentile validation threshold (1% false-alarm budget)
    for name, (s_test, s_val) in scores.items():
        if name == EVT_NAME:
            continue
        thr = np.percentile(s_val, 99)
        rows.append({"source": source, "method": name, "rule": "val 99th pct", **rates(s_test > thr)})
    # the proposed EVT rule at several alphas
    S = evt.survival(mfa_d_test)
    for a in ALPHAS:
        rows.append({"source": source, "method": EVT_NAME, "rule": f"S(q) < {a}", **rates(S < a)})
    # ablation: EVT calibrated on held-out ID *questions* (3-fold cross-fitting, no test leakage)
    S_cf = cross_fitted_question_evt(mfa_d_test, labels)
    for a in ALPHAS:
        rows.append({"source": source, "method": EVT_NAME + " · ID-question calibrated",
                     "rule": f"S(q) < {a} (3-fold CV)", **rates(S_cf < a)})
    return rows


def cross_fitted_question_evt(d_test: np.ndarray, labels: np.ndarray, n_folds: int = 3,
                              tail_percentile: float = 50.0) -> np.ndarray:
    """Fit the Weibull tail on the D^2 of ID test questions from the *other* folds and score each
    query with the model that never saw it.  OOD queries are assigned to folds round-robin.
    With only ~100 calibration questions the tail starts at the median (50 exceedances)."""
    rng = np.random.default_rng(SEED)
    fold = np.empty(len(labels), dtype=int)
    id_idx = np.flatnonzero(labels == "IN_DISTRIBUTION")
    fold[id_idx] = rng.permutation(np.arange(len(id_idx)) % n_folds)
    ood_idx = np.flatnonzero(labels != "IN_DISTRIBUTION")
    fold[ood_idx] = np.arange(len(ood_idx)) % n_folds
    S = np.empty(len(labels))
    for f in range(n_folds):
        calib = id_idx[fold[id_idx] != f]
        e = WeibullTail.fit(d_test[calib], tail_percentile=tail_percentile)
        S[fold == f] = e.survival(d_test[fold == f])
    return S


# ----------------------------------------------------------------------------- latency
def latency_profile(mfa: MFA, Q: np.ndarray, n: int = 1000, n_naive: int = 50):
    from .embeddings import embed
    from .guardrail import MFAEVTGuardrail
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(Q), n)
    g = MFAEVTGuardrail(mfa, WeibullTail(0, 1, 1))
    g.check_embedding(Q[0])  # warm-up
    t = []
    for i in idx:
        t0 = time.perf_counter()
        g.check_embedding(Q[i])
        t.append((time.perf_counter() - t0) * 1e3)
    rows = [("Woodbury MFA scoring (guardrail)", np.array(t))]

    # dense precision pre-computed once: O(K d^2) per query
    precs = [np.linalg.inv(f.W @ f.W.T + np.diag(f.psi)) for f in mfa._factors]
    t = []
    for i in idx:
        t0 = time.perf_counter()
        min((Q[i] - f.mu) @ P @ (Q[i] - f.mu) for f, P in zip(mfa._factors, precs))
        t.append((time.perf_counter() - t0) * 1e3)
    rows.append(("Dense precomputed precision", np.array(t)))

    # naive: build and invert Sigma_k for every query: O(K d^3)
    t = []
    for i in idx[:n_naive]:
        t0 = time.perf_counter()
        min(naive_mahalanobis_sq(Q[i], f.mu, f.W, f.psi) for f in mfa._factors)
        t.append((time.perf_counter() - t0) * 1e3)
    rows.append(("Naive full inversion per query", np.array(t)))

    # query embedding (MiniLM, CPU) for context
    texts = [q["query_text"] for q in load_queries()]
    embed(texts[0])
    t = []
    for s in texts[:200]:
        t0 = time.perf_counter()
        embed(s)
        t.append((time.perf_counter() - t0) * 1e3)
    rows.append(("Query embedding (MiniLM, CPU)", np.array(t)))

    return pd.DataFrame([{"component": name, "n": len(v), "mean_ms": v.mean(), "p50_ms": np.percentile(v, 50),
                          "p95_ms": np.percentile(v, 95), "p99_ms": np.percentile(v, 99)} for name, v in rows])


# ----------------------------------------------------------------------------- plots
def plot_bic(bic_df: pd.DataFrame):
    srcs = list(bic_df.source.unique())
    fig, axes = plt.subplots(1, len(srcs), figsize=(5.2 * len(srcs), 3.8), squeeze=False)
    for ax, s in zip(axes[0], srcs):
        d = bic_df[bic_df.source == s].pivot(index="K", columns="r", values="BIC")
        im = ax.imshow(d.values / 1e6, cmap="Blues_r", aspect="auto")
        ax.set_xticks(range(len(d.columns)), d.columns)
        ax.set_yticks(range(len(d.index)), d.index)
        ax.set_xlabel("factor rank r")
        ax.set_ylabel("components K")
        ax.grid(False)
        best = np.unravel_index(np.nanargmin(d.values), d.shape)
        for i in range(d.shape[0]):
            for j in range(d.shape[1]):
                ax.text(j, i, f"{d.values[i, j] / 1e6:.3f}" + ("\n(best)" if (i, j) == best else ""),
                        ha="center", va="center", fontsize=8,
                        color="white" if (i, j) == best else "#0b0b0b", fontweight="bold" if (i, j) == best else None)
        ax.set_title(f"BIC (×10⁶), {SOURCE_NAMES[s]}  — lower is better")
        fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(RESULTS / "bic_heatmap.png")
    plt.close(fig)


def plot_roc_pr(scores, labels, source):
    from sklearn.metrics import precision_recall_curve
    is_id = labels == "IN_DISTRIBUTION"
    methods = [m for m in scores if m != EVT_NAME]
    panels = [("ID vs all OOD", np.ones_like(is_id)), ("ID vs Near-OOD", is_id | (labels == "NEAR_OOD")),
              ("ID vs Far-OOD", is_id | (labels == "FAR_OOD"))]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    for ax, (title, mask) in zip(axes, panels):
        y = (~is_id[mask]).astype(int)
        for m in methods:
            fpr, tpr = roc_points(y, scores[m][0][mask])
            ax.plot(fpr, tpr, color=color_for(m), lw=1.8, label=m)
        ax.plot([0, 1], [0, 1], ls=":", color=TEXT2, lw=1)
        ax.set_xlabel("False positive rate (ID rejected)")
        ax.set_ylabel("True positive rate (OOD rejected)")
        ax.set_title(f"{title} — {SOURCE_NAMES[source]}")
    axes[0].legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(RESULTS / f"roc_curve{'' if source == MAIN_SOURCE else '_' + source}.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4.6))
    y = (~is_id).astype(int)
    for m in methods:
        p, r, _ = precision_recall_curve(y, scores[m][0])
        ax.plot(r, p, color=color_for(m), lw=1.8, label=m)
    ax.set_xlabel("Recall (OOD)")
    ax.set_ylabel("Precision (OOD)")
    ax.set_title(f"Precision–Recall, ID vs all OOD — {SOURCE_NAMES[source]}")
    ax.legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    fig.savefig(RESULTS / f"pr_curve{'' if source == MAIN_SOURCE else '_' + source}.png")
    plt.close(fig)


def plot_score_dist(d_by_source, labels, evts):
    srcs = list(d_by_source)
    fig, axes = plt.subplots(1, len(srcs), figsize=(6.2 * len(srcs), 4), squeeze=False)
    for ax, s in zip(axes[0], srcs):
        d_test, d_val = d_by_source[s]
        bins = np.linspace(0, max(np.percentile(np.concatenate([d_test, d_val]), 99.5), evts[s].threshold(ALPHA) * 1.05), 60)
        ax.hist(d_val, bins=bins, density=True, color="#c3c2b7", alpha=0.6, label="validation (calibration)")
        for l in SPLIT_COLORS:
            ax.hist(d_test[labels == l], bins=bins, density=True, histtype="step", lw=2,
                    color=SPLIT_COLORS[l], label=f"test {SPLIT_NAMES[l]}")
        e = evts[s]
        ax.axvline(e.tau0, color=TEXT2, ls="--", lw=1)
        ax.axvline(e.threshold(ALPHA), color="#e34948", ls="-", lw=1.2)
        ymax = ax.get_ylim()[1]
        ax.text(e.tau0, ymax * 0.55, "τ₀ (90th pct) ", fontsize=8, color=TEXT2, rotation=90, ha="right", va="center")
        ax.text(e.threshold(ALPHA), ymax * 0.55, f"S(q) = {ALPHA} ", fontsize=8, color=TEXT2, rotation=90,
                ha="right", va="center")
        ax.set_xlabel("min-component Mahalanobis distance D²_M(q)")
        ax.set_ylabel("density")
        ax.set_title(f"Score distributions — {SOURCE_NAMES[s]}")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(RESULTS / "score_dist.png")
    plt.close(fig)


def plot_ll(models_sel):
    fig, ax = plt.subplots(figsize=(6, 3.8))
    for s, m in models_sel.items():
        h = np.array(m.log_likelihood_history_)
        ax.plot(np.arange(1, len(h) + 1), h / m.n_train_, marker="o", ms=4, color=color_for(SOURCE_NAMES[s]),
                label=f"{SOURCE_NAMES[s]} (K={m.n_components}, r={m.rank})")
    ax.set_xlabel("EM iteration")
    ax.set_ylabel("mean log-likelihood per sample")
    ax.set_title("EM convergence (monotone increase)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(RESULTS / "em_loglik.png")
    plt.close(fig)


def plot_qq(val_d_by_source, evts):
    srcs = list(val_d_by_source)
    fig, axes = plt.subplots(1, len(srcs), figsize=(4.6 * len(srcs), 4.2), squeeze=False)
    for ax, s in zip(axes[0], srcs):
        e = evts[s]
        E = np.sort(e.exceedances(val_d_by_source[s])[val_d_by_source[s] > e.tau0])
        p = (np.arange(1, len(E) + 1) - 0.5) / len(E)
        th = stats.weibull_min.ppf(p, e.kappa, 0, e.lam)
        ax.scatter(th, E, s=12, color=color_for(SOURCE_NAMES[s]), edgecolor="white", lw=0.4)
        lim = [0, max(th.max(), E.max()) * 1.05]
        ax.plot(lim, lim, ls=":", color=TEXT2, lw=1)
        ax.set_xlabel("Weibull theoretical quantile")
        ax.set_ylabel("empirical exceedance")
        ax.set_title(f"EVT Q-Q — {SOURCE_NAMES[s]}\nκ={e.kappa:.2f}, λ={e.lam:.1f}, KS p={e.ks_pvalue:.2f}")
    fig.tight_layout()
    fig.savefig(RESULTS / "evt_qq.png")
    plt.close(fig)


def plot_auroc_bars(metrics_df):
    d = metrics_df[metrics_df.source == MAIN_SOURCE].set_index("method")
    methods = list(d.index)
    fig, ax = plt.subplots(figsize=(8, 4.2))
    y = np.arange(len(methods))
    for i, (col, name, off) in enumerate([("Near AUROC", "Near-OOD", -0.2), ("Far AUROC", "Far-OOD", 0.2)]):
        ax.barh(y + off, d[col], height=0.38, color=SPLIT_COLORS["NEAR_OOD" if i == 0 else "FAR_OOD"],
                label=name, edgecolor="white", lw=2)
        for yy, v in zip(y + off, d[col]):
            ax.text(v + 0.005, yy, f"{v:.3f}", va="center", fontsize=8, color=TEXT2)
    ax.set_yticks(y, methods)
    ax.invert_yaxis()
    ax.set_xlim(0.5, 1.06)
    ax.set_xlabel("AUROC")
    ax.set_title(f"AUROC by OOD type — {SOURCE_NAMES[MAIN_SOURCE]}")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2, fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(RESULTS / "auroc_bars.png")
    plt.close(fig)


# ----------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true", help="small grid for smoke testing")
    args = ap.parse_args()
    Ks, Rs = ((4, 8), (8, 16)) if args.fast else (K_GRID, R_GRID)

    sources, Q, labels, queries = load_data()
    print({s: (a.shape, b.shape) for s, (a, b) in sources.items()})

    bic_frames, metric_rows, op_rows, per_query = [], [], [], {}
    models_sel, evts, d_by_source, val_d = {}, {}, {}, {}
    summary = {"alpha": ALPHA, "sources": {}}
    for s, (Xtr, Xva) in sources.items():
        print(f"\n=== source: {s}  train={len(Xtr):,}  val={len(Xva):,}")
        df, models, (K, r) = bic_grid(Xtr, Ks, Rs, s)
        bic_frames.append(df)
        mfa = models[(K, r)]
        dv = mfa.min_mahalanobis(Xva)
        evt = WeibullTail.fit(dv)
        print(f"  selected K={K} r={r};  EVT: {evt}")
        mfa.save(MODELS / f"mfa_{s}.npz")
        evt.save(MODELS / f"evt_{s}.json")
        np.save(DATA / f"validation_distances_{s}.npy", dv)
        models_sel[s], evts[s] = mfa, evt

        scores = score_all(s, Xtr, Xva, Q, K, mfa, evt)
        for name, (st, _) in scores.items():
            metric_rows.append({"source": s, "method": name, **split_metrics(labels, st)})
            per_query[f"{s}|{name}"] = st
        d_test = scores["MFA min-Mahalanobis D²"][0]
        d_by_source[s], val_d[s] = (d_test, dv), dv
        op_rows += operating_points(s, scores, labels, evt, d_test)
        plot_roc_pr(scores, labels, s)
        summary["sources"][s] = {
            "K": K, "r": r, "n_train": len(Xtr), "n_val": len(Xva),
            "mfa_params": mfa.n_parameters(),
            "full_gmm_params": int(K * (Xtr.shape[1] * (Xtr.shape[1] + 1) // 2 + Xtr.shape[1]) + K - 1),
            "evt": {"tau0": evt.tau0, "kappa": evt.kappa, "lambda": evt.lam, "ks_pvalue": evt.ks_pvalue,
                    "threshold_at_alpha": evt.threshold(ALPHA)},
            "em_iterations": mfa.n_iter_, "em_converged": mfa.converged_,
        }

    main = MAIN_SOURCE if MAIN_SOURCE in sources else "passage"
    summary["main_source"] = main
    models_sel[main].save(MODELS / "mfa_selected.npz")
    evts[main].save(MODELS / "evt_selected.json")
    evts[main].save(RESULTS / "evt_params.json")
    np.save(VAL_DIST, val_d[main])

    bic_df = pd.concat(bic_frames)
    bic_df.to_csv(RESULTS / "bic_grid.csv", index=False)
    metrics_df = pd.DataFrame(metric_rows)
    metrics_df.to_csv(RESULTS / "ood_metrics.csv", index=False)
    op_df = pd.DataFrame(op_rows)
    op_df.to_csv(RESULTS / "operating_points.csv", index=False)
    qdf = pd.DataFrame({"query_id": [q["query_id"] for q in queries], "label": labels,
                        "query_text": [q["query_text"] for q in queries]})
    for k, v in per_query.items():
        qdf[k] = v
    d_main = d_by_source[main][0]
    qdf["guard_D2"] = d_main
    qdf["guard_S"] = evts[main].survival(d_main)
    qdf["guard_S_question_cv"] = cross_fitted_question_evt(d_main, labels)
    qdf.to_csv(RESULTS / "query_scores.csv", index=False)

    plot_bic(bic_df)
    plot_score_dist(d_by_source, labels, evts)
    plot_ll(models_sel)
    plot_qq(val_d, evts)
    plot_auroc_bars(metrics_df)

    print("\nLatency profiling ...")
    lat = latency_profile(models_sel[main], Q)
    lat.to_csv(RESULTS / "latency_report.csv", index=False)
    summary["latency"] = lat.to_dict(orient="records")
    (RESULTS / "summary.json").write_text(json.dumps(summary, indent=2, default=float))

    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 30)
    print("\n", bic_df.round(1).to_string(index=False))
    print("\n", metrics_df[["source", "method", "All AUROC", "Near AUROC", "Far AUROC", "All FPR@95TPR",
                            "Near FPR@95TPR"]].round(3).to_string(index=False))
    print("\n", op_df.round(3).to_string(index=False))
    print("\n", lat.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
