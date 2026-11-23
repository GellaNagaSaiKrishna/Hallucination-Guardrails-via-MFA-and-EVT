# Hallucination Guardrails via MFA and EVT

**A pre-retrieval statistical guardrail for Retrieval-Augmented Generation (RAG).** Before a question reaches the retriever and the LLM, the guardrail checks whether it lies inside the knowledge base's embedding distribution. It uses a **Mixture of Factor Analyzers (MFA)** density with fast **Woodbury** low-rank algebra, and calibrates the decision threshold with **Extreme Value Theory (EVT)**. Out-of-scope questions are refused, so the LLM never gets the chance to hallucinate an answer to them.

📄 **[Final report](report/updated_report.md)** ([PDF](report/updated_report.pdf)) · 📝 [Proposal](docs/proposal.md) · 📓 [Notebooks](notebooks/)

---

## Cloud service (branch `cloud-service`)

This branch turns the research code into a deployable service. It adds pgvector (Postgres)
retrieval, a post-retrieval **false-premise check**, LangGraph orchestration, a FastAPI app, a
Docker image and a free deployment path (Hugging Face Spaces + Neon). Everything below was
measured on a laptop CPU (Intel Core Ultra 5 225H, no GPU) unless stated otherwise.

### Why a false-premise check

The report names this as the remaining gap (§3 and §4 of [the report](report/updated_report.md)):

> "**False premises look in-distribution**, so the MFA is near chance on the adversarial set; cosine does better because invented entities have no close chunk."

> "**False-premise questions about indexed companies cannot be detected from the question's embedding.** They need a post-retrieval answerability or NLI check, so the guardrail is best used as a cheap first layer rather than a complete solution."

> Future work: "... add a post-retrieval check ..."

At α = 0.01 the original gate refuses 0% of the adversarial set (adversarial AUROC 0.559).

### Architecture

```
            +------------- LangGraph (service/graph.py) ---------------------------------+
question -> | embed -> gate ---S(q) < alpha-------------------------------------> refuse  | -> JSON
            | MiniLM   MFA + Woodbury + EVT                                              |
            |           | S(q) >= alpha                                                  |
            |           v                                                                |
            |        retrieve -> premise_check ---unsupported-------------------> refuse |
            |        pgvector    NLI(chunk, presupposition)                              |
            |        top-5       | supported                                             |
            |                    v                                                       |
            |                 generate  (extractive | hf:<model> | stub)                 |
            +----------------------------------------------------------------------------+
```

| Node | What it does | Code |
|---|---|---|
| `embed` | all-MiniLM-L6-v2, L2-normalised (unchanged) | `src/embeddings.py` |
| `gate` | MFA (K=16, r=64) Woodbury distance, EVT p-value; refuse if S(q) < α = 0.01 (unchanged model) | `src/guardrail.py` |
| `retrieve` | pgvector cosine top-5 (HNSW m=32, ef_construction=200, ef_search=200). If the question names an indexed company, exact search inside that company's filings (`WHERE company = ANY(...)`) | `service/store.py`, `service/companies.py` |
| `premise_check` | turns the question into the statement it presupposes ("Why did X decline?" → "X declined.") and scores it with an NLI cross-encoder against each retrieved chunk; score = max P(entail) − max P(contradict); refuse if below the calibrated threshold | `service/premise.py` |
| `generate` | `GENERATOR=extractive` (default: no LLM, returns the best supporting sentence), `hf:<model id>` (local HF chat model), `stub` (tests) | `service/generators.py` |

API (`service/api.py`): `POST /ask {"question": ..., "top_k": 5}` returns the answer or the
refusal with its reason (`out_of_distribution` | `unsupported_premise`), the gate p-value, the
premise score and threshold, the sources and per-node timings. `GET /healthz` reports whether
the models are loaded and the database is reachable.

### False-premise evaluation (measured)

**Data.** A new set, [`data/false_premise/eval_set.json`](data/false_premise/eval_set.json), has
30 matched pairs, each built from one corpus chunk ([`scripts/false_premise_pairs.py`](scripts/false_premise_pairs.py)).
Each pair has an answerable question and a false-premise twin: 20 twins have a **contradicted**
premise (the filing says sales rose; the question asks why they fell) and 10 have an **absent**
premise (an acquisition or place that appears in none of that company's chunks, which the build
script checks). The original 68 FinanceBench ID questions and 100 adversarial questions are also
used. Both are split by index parity into calibration (even) and test (odd) halves.

**Protocol** ([`scripts/eval_false_premise.py`](scripts/eval_false_premise.py)):
the checker is chosen by AUROC on the calibration half only, and the threshold is the 5% quantile
of the checker's scores on calibration *answerable* questions. No false-premise label is used to
set it. All numbers below are on the test half plus the 60 hand-built questions, which were never
used for tuning. Two caveats. The presupposition rules were written after the hand-built questions,
so they may suit that phrasing better than unseen questions. And the company filter was added
after a first run showed retrieval returning other companies' chunks; both configurations are
reported, and the filtered one is used because its calibration AUROC is higher.

Six checkers were compared (top-5 retrieval). The selected one is **NLI DeBERTa-v3-base + company filter**:

| Checker (company-filtered retrieval) | p50 ms / question | calib AUROC | test AUROC, hand-built pairs | FB-ID answerable refused | hand-built answerable refused | contradicted caught | absent caught | adversarial caught |
|---|---|---|---|---|---|---|---|---|
| **NLI DeBERTa-v3-base (selected)** | 975 | **0.888** | **0.912** | 9% (3/34) | 3% (1/30) | 60% (12/20) | 70% (7/10) | 42% (21/50) |
| NLI MiniLM2-L6-H768 (fast option) | 365 | 0.686 | 0.908 | 21% | 10% | 75% | 40% | 36% |
| NLI DeBERTa-v3-xsmall | 304 | 0.625 | 0.856 | 18% | 10% | 60% | 50% | 48% |
| SQuAD 2.0 QA (MiniLM) | 197 | 0.629 | 0.859 | 6% | 0% | 0% | 0% | 2% |
| SQuAD 2.0 QA (TinyRoBERTa) | 371 | 0.614 | 0.814 | 0% | 0% | 10% | 10% | 4% |
| Reranker relevance cutoff (ms-marco MiniLM) | 101 | 0.523 | 0.787 | 9% | 0% | 0% | 0% | 12% |
| *Gate only (original system)* | ~1 | n/a | n/a | 0% | 0% | 0% | 0% | 0% |

The MFA + EVT gate passes every question in these sets (it refuses none of them at α = 0.01), so
the "refused" and "caught" columns come entirely from the new check.

Without the company filter (plain top-5 over all filings, [`results/false_premise_nofilter/`](results/false_premise_nofilter/summary.json)),
the same DeBERTa checker has a lower calibration AUROC (0.872) and refuses 18% of held-out
FinanceBench answerable questions. The right evidence chunk is in the top 5 for only 47%
(filtered) / 43% (unfiltered) of the hand-built questions, so many decisions rest on neighbouring
chunks of the same filing.

**Reading the numbers honestly.** The check catches about 60–70% of false premises in the
hand-built set and 42% of the adversarial set, and refuses 3–9% of answerable questions against a
5% target. The threshold comes from only 34 calibration questions, so the realised rate is noisy:
MiniLM2's threshold generalised badly, refusing 21% of held-out questions. All sets are small,
so each percentage carries roughly ±15–20 points of uncertainty at n = 20–34.

**With a real LLM** ([`scripts/eval_generation_fp.py`](scripts/eval_generation_fp.py)):
Qwen2.5-1.5B-Instruct (greedy, CPU, the original study's generator and its strict
"answer only from the context" prompt) answered all 60 hand-built questions from the same
company-filtered top-5 chunks. An answer counts as substantive if it does not match the original
study's abstention regex. I then labelled each substantive answer to a false-premise question as
*accepts* or *corrects* the premise ([`results/false_premise/generations.jsonl`](results/false_premise/generations.jsonl), field `manual_label`;
these 15 labels were made by Claude, the AI assistant that wrote this branch, and are worth a human spot-check).

| Hand-built questions (n = 30 each) | Gate only (before) | Gate + premise check (after) |
|---|---|---|
| False-premise questions answered substantively | 15 / 30 (50%) | 9 / 30 (30%) |
| ... of which the answer **accepts the false premise** | 14 / 30 (47%) | 9 / 30 (30%) |
| Answerable questions answered substantively | 29 / 30 (97%) | 28 / 30 (93%) |

The LLM itself rejected 15 of 30 false premises (it abstained) and corrected one (Meta's Reality Labs
loss). In the other 14 answers it explained the false premise, e.g. *"Why did Costco's net sales decline
in 2021?"* got *"The net sales declined by $2,047 million ... due to a decrease in comparable sales of
16%"*, while the filing says sales rose 18% on comparable sales of +16%. The check removes 5 of those 14
and costs one answerable question (P15_A, about Microsoft's Nuance acquisition).

### Latency (measured, per stage)

Measured with [`scripts/bench_http.py`](scripts/bench_http.py) against the Docker container
(`OMP_NUM_THREADS=2`, the same thread count as the 2-vCPU free Space) on 114 questions: the 60
hand-built questions, 34 FinanceBench ID questions and 20 trivia questions. Each was run once after
3 warm-up calls. Stage times are measured inside the service (LangGraph node wall time).

| Stage | Docker + local pgvector, p50 / p95 ms | Docker + Neon (pooled, ~250 ms network RTT from the laptop), p50 / p95 ms |
|---|---|---|
| embed (MiniLM) | 11.0 / 43.8 | 9.7 / 18.0 |
| gate (MFA + EVT) | 0.98 / 1.64 | 0.92 / 1.48 |
| retrieve (pgvector) | 5.2 / 77.2 | 249.4 / 256.1 |
| premise_check (NLI DeBERTa-v3-base, 5 chunks) | 1191 / 2142 | 1213 / 1688 |
| generate (extractive) | 0.04 / 0.08 | 0.04 / 0.07 |
| **end to end (server)** | **1138 / 2145** | **1339 / 1949** |
| out-of-distribution refusals (gate only, 18 of 114) | about 12 ms in total | about 12 ms in total |

* The premise check dominates. With MiniLM2 instead of DeBERTa it is 365 ms p50 (eval script, default torch threads; DeBERTa there: 975 ms), at the
  accuracy cost shown above. Out-of-distribution questions never reach it: they are refused in about 12 ms.
* Retrieval over the network is one round trip per query. Each search is sent as a single
  `SET LOCAL hnsw.ef_search; SELECT ...` string; before that change, BEGIN/SET/SELECT/COMMIT cost 4 round trips
  (980 ms p50 against Neon). Query execution on Neon is about 1 ms (`EXPLAIN ANALYZE`). On Hugging Face, put the Space and the
  Neon project in nearby regions to keep the round trip small.
* Routing was identical in all three setups (in-memory exact store, local pgvector, Neon):
  hand-built false premises 19/30 refused, hand-built answerable 1/30 refused, FinanceBench ID 3/34 refused,
  trivia 18/20 refused by the gate (1 more by the premise check, 1 answered).
* Container: image 3.75 GB (1.23 GB compressed), memory 640–790 MiB under load (`docker stats`), cold start
  (container start to models loaded, image already local) 5.5 s.
* Original gate, unchanged: 0.77 ms p50 in the report; here 0.9–1.0 ms inside the service.

### pgvector

* `scripts/ingest_pgvector.py` loads the 52,943 chunks + 384-d embeddings with binary `COPY`,
  builds an HNSW cosine index (m=32, ef_construction=200) and a btree on `company`, and prints sizes.
  It works with any Postgres that has the `vector` extension (docker `pgvector/pgvector:pg16`,
  Neon, Supabase, Cloud SQL, Azure Flexible Server).
* Measured size: 224 MiB for the table with both indexes (float32, 52,943 rows), about 232 MiB for the
  whole database, on both local Postgres 16 and Neon (PostgreSQL 18, pgvector 0.8.7). That fits
  the 0.5 GB Neon free tier. Storing `halfvec` (float16) instead gives 133 MiB.
  Load time on the Neon free tier: COPY 139 s, HNSW build 45 s (local container: 3.5 s and 40 s).
* HNSW recall@5 against exact search on the 311 original test questions: 0.80 with pgvector's
  default-like settings (m=16, ef_construction=64, ef_search=40); **0.985** with m=32,
  ef_construction=200, ef_search=200 (the shipped settings). Company-filtered queries are exact:
  they scan the 500–4,000 rows of one company's filings.
* The store is PgBouncer-safe, which Neon's pooled endpoint requires: no server-side prepared
  statements, and only transaction-scoped `SET LOCAL`.

### Run locally

```bash
python -m venv .venv && .venv/Scripts/activate        # Windows; use .venv/bin/activate elsewhere
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements-service.txt pytest httpx pandas faiss-cpu
docker compose up -d db                                  # pgvector on localhost:5433
python scripts/ingest_pgvector.py --database-url postgresql://guard:guard@localhost:5433/guardrail
docker compose up --build api                            # API on http://localhost:7860
curl -s localhost:7860/ask -H 'content-type: application/json' \
     -d '{"question": "Why did Costco'"'"'s net sales decline in 2021?"}'
```

Without Docker: `DATABASE_URL=postgresql://... uvicorn service.api:app --port 7860`, or
`VECTOR_STORE=memory` to use the numpy store over `data/corpus_embeddings.npy` with no database.

Tests: `python -m pytest -q`. Set `TEST_DATABASE_URL` to also run the pgvector-vs-exact test.
The graph and API tests use fakes for every model; two integration tests load the small real
models if they are in the local Hugging Face cache.

Reproduce the evaluation: `python scripts/build_false_premise_set.py`,
`python scripts/eval_false_premise.py` (and `--no-company-filter`),
`python scripts/eval_generation_fp.py`, `python scripts/bench_service.py`,
`python scripts/bench_http.py --url http://localhost:7860`.

### Deploy for free (Hugging Face Spaces + Neon)

Step-by-step instructions are in [`deploy/huggingface/README.md`](deploy/huggingface/README.md). In short:

```bash
# 1. Neon: create a project, run CREATE EXTENSION vector; then load the data (direct, non-pooler host)
DATABASE_URL="$(sed 's/-pooler\././' ../neon_url.txt)" python scripts/ingest_pgvector.py
# 2. Space: stage, create, set the DATABASE_URL secret, upload
python deploy/huggingface/deploy_space.py --space NagaSaiKrishna/guarded-rag-10k \
       --token-file ../hf_token.txt --database-url-file ../neon_url.txt
```

| Resource | Tier | Cost |
|---|---|---|
| Hugging Face Space (Docker, CPU basic: 2 vCPU, 16 GB RAM) | free, sleeps when idle | $0 |
| Neon Postgres + pgvector (0.5 GB storage) | free; the data uses about 232 MiB | $0 |

### Files added on this branch

```
service/                 settings, store (pgvector + memory), companies, premise, generators, graph, api
scripts/                 ingest_pgvector, false_premise_pairs, build_false_premise_set, eval_false_premise,
                         eval_generation_fp, bench_service, bench_http
data/false_premise/      eval_set.json (60 questions)
results/false_premise*/  scores.csv, summary.json, calibration.json, generations, latency
deploy/huggingface/      README.md, deploy_space.py, space_README.md (Space front matter)
Dockerfile, .dockerignore, docker-compose.yml, requirements-service.txt
tests/                   test_graph, test_api, test_premise, test_store, conftest (fakes)
```

---

*The rest of this README describes the original research project.*

## The problem

A RAG system always retrieves the *closest* chunks, even when the knowledge base cannot answer the question. Given irrelevant context, the LLM often answers anyway, which produces **hallucinations**: fluent answers that the knowledge base does not support.

| Question | Kind | What should happen |
|---|---|---|
| *"What is the FY2018 capital expenditure amount for 3M?"* | **In-distribution (ID)**: answerable from the indexed filings | answer it |
| *"What drove operating margin change as of FY22 for AMD?"* | **Near-OOD**: same domain, but AMD's filings are not indexed | refuse |
| *"What was Apple's net revenue from the Apple Car project in 2024?"* | **Near-OOD (adversarial)**: indexed company, but the fact does not exist | refuse |
| *"Which Mario game first introduced Wario?"* | **Far-OOD**: unrelated topic | refuse |

**Goal:** reject out-of-distribution questions *before* retrieval, cheaply (target < 10 ms), with a statistically calibrated false-rejection rate.

## Approach

The architecture from the [proposal](docs/proposal.md):

<p align="center"><img src="docs/figures/proposal_workflow.png" width="620" alt="Proposal workflow: offline MFA and EVT fitting, online Woodbury distance and p-value, reject or pass to the retriever"></p>

As implemented:

<p align="center"><img src="docs/figures/flow_pipeline.png" width="900" alt="Implemented pipeline: offline MFA and EVT fitting, online Woodbury distance, p-value and decision"></p>

- **MFA:** each topic cluster is modelled as a Gaussian with covariance `WWᵀ + Ψ` (r main directions plus per-dimension noise), trained from scratch with EM. → [src/mfa.py](src/mfa.py), [notebook 02](notebooks/02_mfa_training.ipynb)
- **Woodbury:** inverts `WWᵀ + Ψ` through an r × r matrix, so scoring a question costs O(d·r) instead of O(d³). → [src/woodbury.py](src/woodbury.py)
- **EVT:** a Weibull model of the extreme tail of validation distances turns a distance into a p-value S(q). → [src/evt.py](src/evt.py), [notebook 03](notebooks/03_evt_calibration.ipynb)

## Data

| Set | Size | Source |
|---|---|---|
| Knowledge base | 52,943 chunks from 50 SEC 10-K filings (43 FinanceBench filings + 7 recent big-tech 10-Ks) | [SEC EDGAR](https://www.sec.gov/edgar/search/) |
| ID questions | 68 analyst-written questions on indexed companies (answer evidence verified in our text) | [FinanceBench](https://huggingface.co/datasets/PatronusAI/financebench) ([paper](https://arxiv.org/abs/2311.11944)) |
| Near-OOD | 43 FinanceBench questions on 9 held-out companies | FinanceBench |
| Near-OOD (adversarial) | 100 counterfactual questions (entity perturbation), a stress test | constructed for this project |
| Far-OOD | 100 general-knowledge trivia questions | [Open Trivia DB](https://opentdb.com/) |

The full data description, including how the split was made and the checks performed, is in [report §1.1](report/updated_report.md#11-data-and-benchmark).

## Key results

Selected model: **MFA with K = 16 clusters, rank r = 64** (373k parameters, 69% fewer than a full-covariance GMM). EVT tail: τ₀ = 483, κ = 1.05, λ = 50.9 (bootstrap KS p = 0.16).

**Detection (AUROC, higher is better):**

| Method | Near-OOD (real) | Near-OOD adversarial | Far-OOD |
|---|---|---|---|
| **MFA + EVT (proposed)** | **0.736** | 0.559 | 0.988 |
| MFA log-likelihood | **0.756** | 0.608 | 0.991 |
| Cosine similarity (FAISS) | 0.662 | 0.762 | 0.996 |
| K-Means / diagonal GMM | 0.481 / 0.507 | 0.669 / 0.656 | 0.996 / 0.994 |
| Full-covariance GMM | 0.772 | 0.619 | 0.986 |

<p align="center"><img src="results/roc_curve.png" width="900" alt="ROC curves per OOD type"></p>

**End-to-end hallucination rate on OOD questions** (naive RAG prompt):

| | Unguarded | Guarded α = 0.01 | Guarded α = 0.1 (question-calibrated) |
|---|---|---|---|
| All OOD | 44.4% | **20.2%** | **14.8%** |
| Far-OOD | 67% | 8% | 2% |
| Near-OOD | 72% | 72% | 58% |
| ID questions blocked | 0% | **0%** | 7.4% |

<p align="center"><img src="results/hallucination_rates.png" width="900" alt="Hallucination rates with and without the guardrail"></p>

**Takeaways**
- **Far-OOD is solved:** AUROC ≥ 0.99, and the guardrail removes almost all trivia hallucinations.
- **On real Near-OOD questions** (companies outside the knowledge base), covariance-aware densities (MFA, full GMM) beat cosine similarity, K-Means and the diagonal GMM.
- **False-premise questions about indexed companies** (the adversarial set) look in-distribution, so density gates cannot catch them and a post-retrieval check is needed.
- **Fast:** 0.77 ms per question (p50), 1.45 ms at p95, about 2,000× faster than naive covariance inversion.

## Repository structure

```
├── README.md
├── report/            final report (updated_report.md, updated_report.pdf)
├── docs/              proposal and its figure
├── src/               library code
│   ├── data_pipeline.py      HTML cleaning and chunking (500 chars, 50 overlap)
│   ├── embeddings.py         MiniLM embeddings
│   ├── woodbury.py           low-rank-plus-diagonal Gaussian algebra
│   ├── mfa.py                Mixture of Factor Analyzers: EM, BIC
│   ├── evt.py                Weibull peaks-over-threshold calibration
│   ├── guardrail.py          MFAEVTGuardrail + LangChain retrievers
│   ├── baselines.py          cosine, K-Means, diagonal/full GMM, Ledoit-Wolf
│   ├── metrics.py            AUROC, AUPR, FPR@95%TPR
│   ├── run_experiments.py    benchmark: BIC grid, EVT, baselines, plots, latency
│   ├── pipeline.py           end-to-end RAG (Qwen2.5-1.5B-Instruct)
│   └── hallucination_eval.py unguarded vs guarded RAG, NLI faithfulness
├── scripts/           dataset building and notebook generation
├── notebooks/         01 data · 02 MFA · 03 EVT · 04 benchmark · demo
├── data/              chunks, embeddings, test questions, FinanceBench metadata
├── results/           metrics (CSV/JSON), figures, trained models
└── tests/             unit tests (Woodbury vs dense, EM monotonicity, BIC, EVT)
```

## Reproduce

```bash
pip install -r requirements.txt
python scripts/fetch_financebench_filings.py   # download the FinanceBench 10-Ks from EDGAR
python scripts/build_financebench_dataset.py   # knowledge base + test questions
python -m src.embeddings                       # MiniLM embeddings
python -m src.run_experiments                  # MFA, EVT, baselines, figures (~1.5 h on CPU)
python -m src.hallucination_eval               # RAG answers + NLI (GPU recommended)
python scripts/build_notebooks.py              # execute the notebooks
python -m pytest -q
```

Try the guardrail:

```python
from src.guardrail import MFAEVTGuardrail
g = MFAEVTGuardrail.load()
print(g.check("Who painted the Mona Lisa?"))       # → OOD, refused before retrieval
```

Live demo (four example questions, optional full RAG answer, interactive mode): see [presentation/DEMO.md](presentation/DEMO.md).

```bash
python presentation/demo.py              # guardrail decisions at α = 0.01 and 0.1
python presentation/demo.py --rag "..."  # full guarded RAG answer (loads the LLM)
python presentation/demo.py -i           # type your own questions
python presentation/demo.py --offline    # saved results, no models needed
```

The earlier pilot version (7 big-tech filings, generated ID questions) is kept on the [`old` branch](https://github.com/GellaNagaSaiKrishna/Hallucination-Guardrails-via-MFA-and-EVT/tree/old).
