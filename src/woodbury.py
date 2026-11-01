"""Low-rank-plus-diagonal Gaussian utilities via the Woodbury identity.

For Sigma = W W^T + Psi  (W: d x r, Psi diagonal):

    Sigma^{-1} = Psi^{-1} - Psi^{-1} W (I_r + W^T Psi^{-1} W)^{-1} W^T Psi^{-1}
    log|Sigma| = sum_j log psi_j + 2 * sum_i log L_ii,   L = chol(I_r + W^T Psi^{-1} W)

so the quadratic form costs O(r d + r^2) per point after an O(d r^2 + r^3) setup,
instead of O(d^3) for a dense inverse.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import cho_solve, solve_triangular

LOG_2PI = np.log(2.0 * np.pi)


@dataclass
class WoodburyFactor:
    """Pre-computed quantities for one component (mu, W, psi)."""
    mu: np.ndarray          # (d,)
    W: np.ndarray           # (d, r)
    psi: np.ndarray         # (d,)
    inv_psi: np.ndarray     # (d,)
    W_ipsi: np.ndarray      # (d, r)  = Psi^{-1} W
    L: np.ndarray           # (r, r)  lower Cholesky factor of M = I + W^T Psi^{-1} W
    log_det: float          # log|Sigma|

    @classmethod
    def build(cls, mu: np.ndarray, W: np.ndarray, psi: np.ndarray) -> "WoodburyFactor":
        inv_psi = 1.0 / psi
        W_ipsi = W * inv_psi[:, None]
        M = np.eye(W.shape[1]) + W.T @ W_ipsi
        L = np.linalg.cholesky(M)
        log_det = float(np.sum(np.log(psi)) + 2.0 * np.sum(np.log(np.diag(L))))
        return cls(mu, W, psi, inv_psi, W_ipsi, L, log_det)

    @property
    def dim(self) -> int:
        return self.mu.shape[0]

    def mahalanobis_sq(self, X: np.ndarray) -> np.ndarray:
        """(x - mu)^T Sigma^{-1} (x - mu) for every row of X, shape (N,)."""
        delta = np.atleast_2d(X) - self.mu
        term1 = np.einsum("nd,nd,d->n", delta, delta, self.inv_psi)
        v = delta @ self.W_ipsi                                    # (N, r)
        sol = solve_triangular(self.L, v.T, lower=True)            # (r, N)
        term2 = np.sum(sol * sol, axis=0)
        return np.maximum(term1 - term2, 0.0)

    def log_pdf(self, X: np.ndarray) -> np.ndarray:
        return -0.5 * (self.dim * LOG_2PI + self.log_det + self.mahalanobis_sq(X))

    def posterior_beta(self) -> np.ndarray:
        """beta = (I + W^T Psi^{-1} W)^{-1} W^T Psi^{-1},  shape (r, d)  ( = W^T Sigma^{-1} )."""
        return cho_solve((self.L, True), self.W_ipsi.T)

    def precision(self) -> np.ndarray:
        """Dense Sigma^{-1} assembled via Woodbury (only for testing / small d)."""
        beta = self.posterior_beta()
        return np.diag(self.inv_psi) - self.W_ipsi @ beta


def naive_mahalanobis_sq(x: np.ndarray, mu: np.ndarray, W: np.ndarray, psi: np.ndarray) -> float:
    """Reference O(d^3) path: build the dense covariance and invert it every call."""
    sigma = W @ W.T + np.diag(psi)
    delta = x - mu
    return float(delta @ np.linalg.inv(sigma) @ delta)
