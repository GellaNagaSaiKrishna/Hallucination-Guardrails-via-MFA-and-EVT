"""Generate (and optionally execute) the project notebooks.

    python scripts/build_notebooks.py            # write + execute
    python scripts/build_notebooks.py --no-exec  # write only
"""
from __future__ import annotations

import argparse
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parent.parent
NB_DIR = ROOT / "notebooks"

SETUP = """import sys, json, os
from pathlib import Path
ROOT = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()
os.chdir(ROOT); sys.path.insert(0, str(ROOT))
import numpy as np, pandas as pd
from IPython.display import Image, display
pd.set_option('display.width', 200); pd.set_option('display.max_colwidth', 120)"""


def md(s):
    return nbf.v4.new_markdown_cell(s)


def code(s):
    return nbf.v4.new_code_cell(s)


NOTEBOOKS = {
    "01_data_pipeline.ipynb": [
        md("# 01 · Data pipeline\nSEC 10-K filings (EDGAR) → clean text → `RecursiveCharacterTextSplitter(500, 50)` → "
           "MiniLM embeddings (d = 384, L2-normalised) → FAISS index.  Rebuild with "
           "`python scripts/fetch_financebench_filings.py`, `python scripts/build_financebench_dataset.py` and "
           "`python -m src.embeddings`."),
        code(SETUP),
        code("from src.embeddings import load_chunks, load_queries\n"
             "chunks = pd.DataFrame(load_chunks()); queries = pd.DataFrame(load_queries())\n"
             "print(len(chunks), 'chunks from', chunks.doc_name.nunique(), 'filings')\n"
             "chunks.groupby(['source','company']).agg(filings=('doc_name','nunique'), chunks=('text','size'))"),
        code("chunks.char_count.describe()"),
        code("for t in chunks.sample(3, random_state=1).text: print(t, '\\n---')"),
        md("## Evaluation question sets\n"
           "| Label | Source |\n|---|---|\n"
           "| ID | FinanceBench questions on knowledge-base companies' 10-Ks (evidence verified in our text) |\n"
           "| Near-OOD | FinanceBench questions on held-out companies (no filing indexed) |\n"
           "| Near-OOD adv. | counterfactual questions built by entity perturbation (stress test) |\n"
           "| Far-OOD | Open Trivia Database |"),
        code("display(queries.groupby(['label','source']).size().rename('questions').to_frame())\n"
             "queries.groupby('label').head(3)[['query_id','label','query_text']]"),
        code("rep = json.load(open('data/financebench/dataset_report.json', encoding='utf-8'))\n"
             "print('held-out companies:', rep['held_out_companies'])\n"
             "pd.Series([e['reason'].split(' (')[0] for e in rep['excluded_questions']]).value_counts().rename('excluded FinanceBench questions')"),
        md("## Embedding geometry"),
        code("import matplotlib.pyplot as plt\nfrom sklearn.decomposition import PCA\nfrom src.plotting import SPLIT_COLORS, SPLIT_NAMES\n"
             "X = np.load('data/corpus_embeddings.npy'); Q = np.load('data/query_embeddings.npy')\n"
             "pca = PCA(2, random_state=0).fit(X)\n"
             "fig, ax = plt.subplots(figsize=(7,5))\n"
             "ax.scatter(*pca.transform(X[::5]).T, s=3, c='#c3c2b7', label='corpus chunks (every 5th)')\n"
             "labels = queries.label.values\n"
             "for l in SPLIT_COLORS:\n"
             "    ax.scatter(*pca.transform(Q[labels==l]).T, s=16, c=SPLIT_COLORS[l], edgecolor='white', lw=.4, label=SPLIT_NAMES[l])\n"
             "ax.legend(fontsize=8); ax.set_title('PCA of MiniLM embeddings'); plt.show()"),
    ],
    "02_mfa_training.ipynb": [
        md("# 02 · Mixture of Factor Analyzers — EM and BIC selection\n"
           "Custom NumPy implementation in `src/mfa.py`; low-rank algebra in `src/woodbury.py`."),
        code(SETUP),
        md("## Woodbury identity check"),
        code("from src.woodbury import WoodburyFactor\nrng = np.random.default_rng(0); d, r = 384, 16\n"
             "f = WoodburyFactor.build(rng.normal(size=d), rng.normal(size=(d, r))*.05, rng.uniform(1e-3, 3e-3, d))\n"
             "S = f.W @ f.W.T + np.diag(f.psi)\n"
             "print('max |Woodbury - inv| =', np.abs(f.precision() - np.linalg.inv(S)).max())\n"
             "print('logdet error =', abs(f.log_det - np.linalg.slogdet(S)[1]))"),
        md("## Fit one MFA live (K = 8, r = 16) on training passages"),
        code("from src.mfa import MFA\nsplit = np.load('data/train_val_split.npz')\n"
             "X = np.load('data/corpus_embeddings.npy').astype(float)\nXtr = X[split['train_idx']]\n"
             "m = MFA(8, 16, verbose=True).fit(Xtr)\n"
             "print(f'{m.n_iter_} iterations, converged={m.converged_}, {m.fit_time_:.1f}s, params={m.n_parameters():,}')\n"
             "print('full-cov GMM params would be', 8*(384*385//2 + 384) + 7)"),
        md("## BIC grid (K × r)"),
        code("bic = pd.read_csv('results/bic_grid.csv'); bic.round(1)"),
        code("display(Image('results/bic_heatmap.png'))\ndisplay(Image('results/em_loglik.png'))"),
        code("json.load(open('results/summary.json'))['sources']"),
    ],
    "03_evt_calibration.ipynb": [
        md("# 03 · Extreme Value Theory calibration\nPeaks-over-threshold: τ₀ = 90th percentile of held-out "
           "D²_M, Weibull fitted to the exceedances, S(q) = exp(−((D²−τ₀)/λ)^κ)."),
        code(SETUP),
        code("from src.evt import WeibullTail\nfrom src.mfa import MFA\n"
             "for s in ['passage']:\n"
             "    e = WeibullTail.load(f'results/models/evt_{s}.json')\n"
             "    print(s, e, ' threshold D* at alpha=0.01:', round(e.threshold(0.01), 1))"),
        code("display(Image('results/evt_qq.png'))\ndisplay(Image('results/score_dist.png'))"),
        md("## Sensitivity to α"),
        code("op = pd.read_csv('results/operating_points.csv')\n"
             "op[op.method.str.contains('EVT')].round(3)"),
    ],
    "04_benchmarking.ipynb": [
        md("# 04 · Benchmarking\nAUROC, AUPR, FPR@95%TPR for MFA(+EVT) and all baselines; operating points; "
           "latency; downstream hallucination."),
        code(SETUP),
        code("m = pd.read_csv('results/ood_metrics.csv')\n"
             "m[['method','All AUROC','Near AUROC','Adv AUROC','Far AUROC','All FPR@95TPR','Near FPR@95TPR','All AUPR_OOD']].round(3)"),
        code("display(Image('results/roc_curve.png'))\ndisplay(Image('results/auroc_bars.png'))\ndisplay(Image('results/pr_curve.png'))"),
        md("## Operating points (1 % validation false-alarm budget vs. EVT α)"),
        code("pd.read_csv('results/operating_points.csv').round(3)"),
        md("## Latency"),
        code("pd.read_csv('results/latency_report.csv').round(3)"),
        md("## Downstream hallucination: unguarded vs guarded RAG"),
        code("pd.read_csv('results/rag_comparison.csv').round(3)"),
        md("Naive prompt (no abstention instruction): OOD queries that received a substantive answer "
           "without the guardrail, and whether each guarded variant blocks them."),
        code("r = pd.read_csv('results/rag_eval_per_query.csv')\n"
             "pd.set_option('display.max_colwidth', 120)\n"
             "n = r[(r.regime=='naive') & ~r.unguarded_abstain]\n"
             "n[['label','query_text','unguarded_answer','guarded_blocked','guarded_qcal_blocked']].head(12)"),
    ],
    "demo.ipynb": [
        md("# Demo · type any query, see the guardrail decision\nEdit `query` and re-run.  The 2-D plot shows the "
           "MFA component means (PCA) and where the query lands."),
        code(SETUP),
        code("from src.guardrail import MFAEVTGuardrail, FaissRetriever, GuardrailRetriever\nfrom src.embeddings import embed\n"
             "guard = MFAEVTGuardrail.load()\nretriever = GuardrailRetriever(guardrail=guard, base_retriever=FaissRetriever.from_disk())\n"
             "def check(query):\n"
             "    docs = retriever.invoke(query); d = retriever.last_decision\n"
             "    print(f'{query}\\n  D2 = {d.d_sq:.1f}   S(q) = {d.p_value:.4f}   nearest component = {d.cluster}   '\n"
             "          f'-> {d.label}  ({d.latency_ms:.2f} ms)')\n"
             "    for doc in docs: print('   •', doc.metadata['chunk_id'], doc.page_content[:110].replace('\\n',' '), '...')\n"
             "    return d"),
        code("for q in ['What is the FY2018 capital expenditure amount (in USD millions) for 3M?',\n"
             "          'What are the main risk factors related to Tesla vehicle production?',\n"
             "          'What was Microsoft revenue from its lunar data centers in 2019?',\n"
             "          'Who painted the Mona Lisa?']:\n"
             "    check(q); print()"),
        code("import matplotlib.pyplot as plt\nfrom sklearn.decomposition import PCA\n"
             "X = np.load('data/corpus_embeddings.npy')\npca = PCA(2, random_state=0).fit(X)\n"
             "query = 'Who won the 2010 FIFA World Cup?'\nd = check(query); q2 = pca.transform(embed(query))[0]\n"
             "fig, ax = plt.subplots(figsize=(6.5,5))\nax.scatter(*pca.transform(X[::10]).T, s=3, c='#c3c2b7', label='corpus chunks')\n"
             "mu = pca.transform(guard.mfa.means_)\nax.scatter(*mu.T, s=60, c='#2a78d6', edgecolor='white', label='MFA component means')\n"
             "ax.scatter(*q2, s=140, marker='*', c='#e34948' if d.is_ood else '#1baf7a', edgecolor='white', label=f'query → {d.label}')\n"
             "ax.legend(fontsize=8); ax.set_title('Guardrail view (PCA)'); plt.show()"),
    ],
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-exec", action="store_true")
    args = ap.parse_args()
    NB_DIR.mkdir(exist_ok=True)
    for name, cells in NOTEBOOKS.items():
        nb = nbf.v4.new_notebook()
        nb.cells = cells
        nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
        if not args.no_exec:
            from nbconvert.preprocessors import ExecutePreprocessor
            ExecutePreprocessor(timeout=1200, kernel_name="python3").preprocess(nb, {"metadata": {"path": str(NB_DIR)}})
        nbf.write(nb, NB_DIR / name)
        print("wrote", name)


if __name__ == "__main__":
    main()
