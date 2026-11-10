# Hallucination Guardrails via MFA and EVT

MFA: Mixture of Factor Analyzers

EVT: Extreme Value Theory

GMM: Gaussian Mixture Model


**Project Title:** Latent Manifold Density Estimation via Mixture of Factor Analyzers for Out-of-Distribution Hallucination Guardrails in RAG  

---

## 1.Problem Statement

Retrieval-Augmented Generation (RAG) systems fail silently when presented with unanswerable or Out-of-Distribution (OOD) queries. Because vector databases (e.g., FAISS, ChromaDB) perform nearest-neighbor searches deterministically, they force-retrieve the closest context chunks even when a query lies entirely outside the indexed knowledge base. Conditioned on these irrelevant chunks, the Large Language Model (LLM) hallucinates plausible yet ungrounded facts.

Standard heuristics (such as raw cosine similarity cutoffs or K-Means clustering) fail in high-dimensional embedding spaces because embeddings lie along complex, anisotropic submanifolds. Furthermore, full-covariance Gaussian Mixture Models (GMMs) cannot be computed due to singularity and the curse of dimensionality (O(K*d^2) parameters).

**Objective:** To develop, derive, and benchmark a statistical pre-retrieval guardrail that uses a Mixture of Factor Analyzers (MFA) with Woodbury low-rank matrix inversion and Extreme Value Theory (EVT) to detect and reject OOD queries in embedding space before retrieval and LLM inference occur.

---

## 2. Theoretical Formulation & Architecture

```
                        Indexed Corpus Chunks (e.g., SEC 10-Ks)
                                          │
                                          ▼
                      Embedding Extraction (BGE / MiniLM: ℝ^d)
                                          │
                                          ▼
         ┌─────────────────────────────────────────────────────────────────┐
         │ 1. Offline Density Estimation (EM Algorithm)                   │
         │    - Fit Mixture of Factor Analyzers (K clusters, rank r ≪ d)   │
         │    - Fit EVT Weibull parameters (λ, κ) on validation tail       │
         └─────────────────────────────────────────────────────────────────┘
                                          │
──────────────────────────────────────────┼──────────────────────────────────────────
                                          │ Online Query Execution
                                          ▼
                             User Query q ∈ ℝ^d
                                          │
                                          ▼
         ┌─────────────────────────────────────────────────────────────────┐
         │ 2. Real-Time Pre-Retrieval Statistical Guardrail               │
         │    - Woodbury low-rank Mahalanobis inversion: D_M²(q)           │
         │    - Weibull p-value calculation: S(q)                          │
         └─────────────────────────────────────────────────────────────────┘
                                          │
                         ┌────────────────┴────────────────┐
                         ▼                                 ▼
             S(q) < α (Out-of-Distribution)    S(q) ≥ α (In-Distribution)
                         │                                 │
                         ▼                                 ▼
            [Bypass LLM Generation]               [Pass to Vector Index]
          Return Standardized Message:           Retrieve Top-K Chunks &
          "Query outside knowledge base"         Feed to LLM Generator
```


---

## 3. Dataset & Benchmark Setup

**In-Distribution (ID) Corpus Options & Sources:**
- **SEC 10-K Filings (FY 2022–2023):** Sourced from the official SEC EDGAR database (~8,000–10,000 text chunks):
  - [SEC EDGAR Company Search](https://www.sec.gov/edgar/searchedgar/companysearch)
  - [Apple Inc. (AAPL) 10-K](https://www.sec.gov/edgar/browse/?CIK=0000320193)
  - [Microsoft Corp. (MSFT) 10-K](https://www.sec.gov/edgar/browse/?CIK=0000789019)
  - [Alphabet Inc. (GOOGL) 10-K](https://www.sec.gov/edgar/browse/?CIK=0001652044)
- **Alternative Technical Corpus:** [Python 3.12 Official Documentation](https://docs.python.org/3.12/)

**Test Evaluation Sets & Sources:**

| Set                      | Count       | Description / Data Source |
|--------------------------|-------------|---------------------------|
| **ID Test Set**          | 150 queries | Direct factual questions answerable strictly from the corpus. |
| **Near-OOD / Semantic Drift** | 100 queries | Queries using identical vocabulary and entities but requesting unmentioned events or unreleased features. |
| **Far-OOD**              | 100 queries | General-knowledge QA benchmarks: [HotpotQA](https://hotpotqa.github.io/) / [HuggingFace HotpotQA](https://huggingface.co/datasets/hotpot_qa) and [TriviaQA](https://huggingface.co/datasets/mandar90/trivia_qa). |

**Comparative Baselines:**
1. Cosine similarity threshold to nearest neighbor (standard FAISS baseline)
2. Euclidean distance to nearest K-Means centroid
3. Diagonal-covariance GMM log-likelihood cutoff

---



## 4. Four-Week Milestone Plan

| Week   | Focus                        | Key Deliverable                                                              |
|--------|------------------------------|------------------------------------------------------------------------------|
| Week 1 | Data Pipeline & Baselines    | Corpus chunked, embedded, FAISS indexed, and query splits curated            |
| Week 2 | MFA & EM Implementation      | Custom NumPy MFA with Woodbury inversion fully implemented                   |
| Week 3 | EVT Calibration & Integration| Weibull tail fitted, LangChain guardrail retriever operational                |
| Week 4 | Benchmarking & Report        | AUROC, FPR@95, Ragas faithfulness scores, latency profiles, final report     |

---

