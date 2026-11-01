"""Week 3 / Step 3.5 - Baseline OOD scorers.  Convention: higher score = more OOD.

1. Cosine to nearest neighbour (FAISS IndexFlatIP on L2-normalised vectors):  1 - max_i cos(q, x_i)
2. Euclidean distance to the nearest K-Means centroid:                         min_k ||q - c_k||
3. Diagonal-covariance GMM negative log-likelihood:                            -log p_diag(q)

Extra reference scorers used in the ablation:
  * Full-covariance GMM NLL (sklearn, reg_covar) - shows the cost / conditioning of d x d covariances.
  * Single Gaussian Mahalanobis with Ledoit-Wolf shrinkage (a common "Mahalanobis OOD" baseline).
"""
from __future__ import annotations

import faiss
import numpy as np
from sklearn.cluster import KMeans
from sklearn.covariance import LedoitWolf
from sklearn.mixture import GaussianMixture


class CosineFAISS:
    name = "Cosine NN (FAISS)"

    def fit(self, X: np.ndarray) -> "CosineFAISS":
        X = np.ascontiguousarray(X, dtype=np.float32)
        self.index = faiss.IndexFlatIP(X.shape[1])
        self.index.add(X)
        return self

    def score(self, Q: np.ndarray) -> np.ndarray:
        sims, _ = self.index.search(np.ascontiguousarray(Q, dtype=np.float32), 1)
        return 1.0 - sims[:, 0].astype(np.float64)


class KMeansEuclidean:
    name = "K-Means Euclidean"

    def __init__(self, n_clusters: int = 8, random_state: int = 0):
        self.km = KMeans(n_clusters=n_clusters, n_init=10, random_state=random_state)

    def fit(self, X: np.ndarray) -> "KMeansEuclidean":
        self.km.fit(X)
        return self

    def score(self, Q: np.ndarray) -> np.ndarray:
        return self.km.transform(Q).min(axis=1)


class DiagGMM:
    name = "Diagonal GMM"

    def __init__(self, n_components: int = 8, random_state: int = 0, covariance_type: str = "diag"):
        self.gmm = GaussianMixture(n_components=n_components, covariance_type=covariance_type,
                                   random_state=random_state, reg_covar=1e-6, max_iter=200)

    def fit(self, X: np.ndarray) -> "DiagGMM":
        self.gmm.fit(X)
        return self

    def score(self, Q: np.ndarray) -> np.ndarray:
        return -self.gmm.score_samples(Q)


class FullGMM(DiagGMM):
    name = "Full-cov GMM"

    def __init__(self, n_components: int = 8, random_state: int = 0):
        super().__init__(n_components, random_state, covariance_type="full")


class LedoitWolfMahalanobis:
    name = "Single Gaussian (Ledoit-Wolf)"

    def fit(self, X: np.ndarray) -> "LedoitWolfMahalanobis":
        self.lw = LedoitWolf().fit(X)
        return self

    def score(self, Q: np.ndarray) -> np.ndarray:
        return self.lw.mahalanobis(Q)
