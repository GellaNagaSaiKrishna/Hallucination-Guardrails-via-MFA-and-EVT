"""Week 2 - Mixture of Factor Analyzers fitted with a custom EM algorithm (NumPy/SciPy).

    p(x) = sum_k pi_k N(x | mu_k, W_k W_k^T + Psi_k),   W_k in R^{d x r},  Psi_k diagonal

E-step (per component k, using the current parameters):
    gamma_ik    = pi_k N(x_i | k) / sum_j pi_j N(x_i | j)          (log-sum-exp)
    beta_k      = (I_r + W_k^T Psi_k^{-1} W_k)^{-1} W_k^T Psi_k^{-1}
    E[y | x,k]  = beta_k (x - mu_k)
    E[yy^T|x,k] = I_r - beta_k W_k + E[y] E[y]^T

M-step (AECM ordering: pi, mu first, then the factor-analysis update about the new mean):
    pi_k  = N_k / N,             mu_k = sum_i gamma_ik x_i / N_k
    W_k   = [sum_i gamma_ik (x_i - mu_k) E[y_i]^T] [sum_i gamma_ik E[y_i y_i^T]]^{-1}
    Psi_k = diag( S_k - W_k beta_k S_k ),   S_k = weighted scatter about mu_k
    Psi_k >= eps * Var(X)  (numerical floor, eps = 1e-4)

Every operation is O(N d r); no d x d matrix is ever formed.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.special import logsumexp
from sklearn.cluster import KMeans

from .woodbury import WoodburyFactor


@dataclass
class MFA:
    n_components: int = 8
    rank: int = 16
    max_iter: int = 50
    tol: float = 1e-5
    psi_floor_eps: float = 1e-4
    random_state: int = 0
    verbose: bool = False

    # fitted state
    weights_: np.ndarray | None = None
    means_: np.ndarray | None = None
    loadings_: np.ndarray | None = None     # (K, d, r)
    psis_: np.ndarray | None = None         # (K, d)
    log_likelihood_history_: list = field(default_factory=list)
    n_iter_: int = 0
    converged_: bool = False
    fit_time_: float = 0.0
    n_train_: int = 0
    _factors: list | None = None

    # ------------------------------------------------------------------ init
    def _init_params(self, X: np.ndarray, floor: np.ndarray) -> None:
        N, d = X.shape
        K, r = self.n_components, self.rank
        km = KMeans(n_clusters=K, n_init=10, random_state=self.random_state).fit(X)
        self.means_ = km.cluster_centers_.astype(np.float64)
        self.loadings_ = np.zeros((K, d, r))
        self.psis_ = np.zeros((K, d))
        for k in range(K):
            Xk = X[km.labels_ == k]
            if len(Xk) < 2:
                Xk = X
            R = Xk - self.means_[k]
            # PPCA closed form from the top-r right singular vectors of the residuals
            _, s, Vt = np.linalg.svd(R, full_matrices=False)
            eig = s ** 2 / len(Xk)
            eig_full = np.concatenate([eig, np.zeros(max(0, d - len(eig)))])
            sigma2 = eig_full[r:].mean() if d > r else 0.0
            lam = np.maximum(eig_full[:r] - sigma2, 1e-8)
            Vr = np.zeros((d, r))
            m = min(r, Vt.shape[0])
            Vr[:, :m] = Vt[:m].T
            self.loadings_[k] = Vr * np.sqrt(lam)
            diag_S = np.mean(R ** 2, axis=0)
            self.psis_[k] = np.maximum(diag_S - np.sum(self.loadings_[k] ** 2, axis=1), floor)
        self.weights_ = np.full(K, 1.0 / K)

    def _refresh(self) -> None:
        self._factors = [WoodburyFactor.build(self.means_[k], self.loadings_[k], self.psis_[k])
                         for k in range(self.n_components)]

    # ------------------------------------------------------------------ EM
    def _log_resp(self, X: np.ndarray) -> np.ndarray:
        """log pi_k + log N(x | k), shape (N, K)."""
        return np.column_stack([np.log(self.weights_[k]) + f.log_pdf(X)
                                for k, f in enumerate(self._factors)])

    def e_step(self, X: np.ndarray) -> tuple[np.ndarray, float]:
        lr = self._log_resp(X)
        norm = logsumexp(lr, axis=1, keepdims=True)
        return np.exp(lr - norm), float(norm.sum())

    def m_step(self, X: np.ndarray, gamma: np.ndarray, floor: np.ndarray) -> None:
        N, d = X.shape
        r = self.rank
        Nk = gamma.sum(axis=0) + 1e-10
        self.weights_ = np.maximum(Nk / N, 1e-12)
        self.weights_ /= self.weights_.sum()
        eye = np.eye(r)
        for k in range(self.n_components):
            g = gamma[:, k]
            beta = self._factors[k].posterior_beta()                 # (r, d) from old W, Psi
            W_old = self.loadings_[k]
            mu = g @ X / Nk[k]
            D = X - mu                                              # (N, d)
            Ey = D @ beta.T                                         # (N, r)  E[y | x, k]
            gEy = g[:, None] * Ey
            SB = D.T @ gEy / Nk[k]                                  # (d, r) = S beta^T
            Eyy = eye - beta @ W_old + Ey.T @ gEy / Nk[k]           # (r, r) mean E[yy^T]
            W_new = np.linalg.solve(Eyy.T, SB.T).T                  # SB @ Eyy^{-1}
            diag_S = g @ (D * D) / Nk[k]
            psi = diag_S - np.sum(W_new * SB, axis=1)
            self.means_[k] = mu
            self.loadings_[k] = W_new
            self.psis_[k] = np.maximum(psi, floor)
        self._refresh()

    def fit(self, X: np.ndarray) -> "MFA":
        t0 = time.perf_counter()
        X = np.asarray(X, dtype=np.float64)
        self.n_train_ = X.shape[0]
        floor = self.psi_floor_eps * X.var(axis=0) + 1e-12
        self._init_params(X, floor)
        self._refresh()
        self.log_likelihood_history_ = []
        prev = -np.inf
        for it in range(1, self.max_iter + 1):
            gamma, ll = self.e_step(X)
            self.log_likelihood_history_.append(ll)
            if self.verbose:
                print(f"  iter {it:3d}  logL = {ll:,.2f}")
            if np.isfinite(prev) and abs(ll - prev) <= self.tol * abs(prev):
                self.converged_ = True
                break
            prev = ll
            self.m_step(X, gamma, floor)
        self.n_iter_ = it
        # final log-likelihood under the final parameters
        _, ll = self.e_step(X)
        if self.log_likelihood_history_[-1] != ll:
            self.log_likelihood_history_.append(ll)
        self.fit_time_ = time.perf_counter() - t0
        return self

    # ------------------------------------------------------------------ scoring
    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """log p(x) under the mixture."""
        return logsumexp(self._log_resp(np.asarray(X, dtype=np.float64)), axis=1)

    def mahalanobis_all(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        return np.column_stack([f.mahalanobis_sq(X) for f in self._factors])

    def min_mahalanobis(self, X: np.ndarray) -> np.ndarray:
        """D_M^2(q) = min_k (q - mu_k)^T Sigma_k^{-1} (q - mu_k)."""
        return self.mahalanobis_all(X).min(axis=1)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._log_resp(np.asarray(X, dtype=np.float64)).argmax(axis=1)

    def log_likelihood(self, X: np.ndarray) -> float:
        return float(self.score_samples(X).sum())

    # ------------------------------------------------------------------ model selection
    def n_parameters(self) -> int:
        """Free parameters: per component W (d r - r(r-1)/2 after rotation), Psi (d), mu (d); plus K-1 weights."""
        d, r, K = self.means_.shape[1], self.rank, self.n_components
        return int(K * (d * r - r * (r - 1) // 2 + d + d) + (K - 1))

    def bic(self, X: np.ndarray) -> float:
        return -2.0 * self.log_likelihood(X) + self.n_parameters() * np.log(X.shape[0])

    # ------------------------------------------------------------------ persistence
    def save(self, path: str | Path) -> None:
        np.savez_compressed(path, weights=self.weights_, means=self.means_, loadings=self.loadings_,
                            psis=self.psis_, ll_hist=np.array(self.log_likelihood_history_),
                            meta=np.array([self.n_components, self.rank, self.n_iter_, self.n_train_]))

    @classmethod
    def load(cls, path: str | Path) -> "MFA":
        z = np.load(path)
        K, r, n_iter, n_train = (int(v) for v in z["meta"])
        m = cls(n_components=K, rank=r)
        m.weights_, m.means_, m.loadings_, m.psis_ = z["weights"], z["means"], z["loadings"], z["psis"]
        m.log_likelihood_history_ = list(z["ll_hist"])
        m.n_iter_, m.n_train_ = n_iter, n_train
        m._refresh()
        return m
