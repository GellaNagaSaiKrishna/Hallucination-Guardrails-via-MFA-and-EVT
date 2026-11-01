### Key Mathematical Components

**1. Mixture of Factor Analyzers (MFA):**
Each cluster covariance is decomposed into a low-rank loading matrix W_k (rank r << d) and diagonal uniqueness Psi_k:

    p(x) = sum_k pi_k * N(x | mu_k, W_k * W_k^T + Psi_k)

**2. Fast Mahalanobis Distance via Woodbury Identity:**
Reduces matrix inversion cost from O(d^3) to O(r^3 + r^2*d), achieving sub-10ms inference:

    Sigma_k^{-1} = Psi_k^{-1} - Psi_k^{-1} W_k (I_r + W_k^T Psi_k^{-1} W_k)^{-1} W_k^T Psi_k^{-1}

**3. Extreme Value Theory (EVT) Hypothesis Testing:**
A Weibull distribution is fitted on validation tail distances to compute the survival probability S(q). Queries with S(q) < alpha (alpha = 0.01) are rejected prior to LLM generation.

---
## 4. Evaluation Metrics

| Metric              | Purpose                                                                        |
|---------------------|--------------------------------------------------------------------------------|
| AUROC               | Separability between ID and OOD queries across all thresholds                  |
| FPR@95% TPR         | False positive rate when maintaining 95% true positive rate on ID queries      |
| Faithfulness (Ragas)| Groundedness of generated answers with vs. without the guardrail               |
| P95 Latency         | Additional inference overhead; target < 10ms per query                         |