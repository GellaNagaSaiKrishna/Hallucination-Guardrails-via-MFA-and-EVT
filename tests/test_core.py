"""Unit tests for the mathematical core:  python -m pytest -q"""
import numpy as np
import pytest
from scipy.stats import multivariate_normal

from src.evt import WeibullTail, compute_ood_score
from src.metrics import fpr_at_95_tpr
from src.mfa import MFA
from src.woodbury import WoodburyFactor, naive_mahalanobis_sq


@pytest.fixture
def factor():
    rng = np.random.default_rng(0)
    d, r = 40, 4
    return WoodburyFactor.build(rng.normal(size=d), rng.normal(size=(d, r)), rng.uniform(0.1, 1, d)), rng


def test_woodbury_matches_dense(factor):
    f, rng = factor
    X = rng.normal(size=(9, f.dim))
    S = f.W @ f.W.T + np.diag(f.psi)
    np.testing.assert_allclose(f.precision(), np.linalg.inv(S), atol=1e-9)
    np.testing.assert_allclose(f.log_pdf(X), multivariate_normal(f.mu, S).logpdf(X), rtol=1e-9)
    ref = [naive_mahalanobis_sq(x, f.mu, f.W, f.psi) for x in X]
    np.testing.assert_allclose(f.mahalanobis_sq(X), ref, rtol=1e-9)


def _synthetic_mixture(rng, n=600, d=20, r=2):
    X = []
    for k in range(3):
        mu = rng.normal(scale=5, size=d)
        W = rng.normal(size=(d, r))
        X.append(mu + rng.normal(size=(n, r)) @ W.T + rng.normal(scale=0.3, size=(n, d)))
    return np.vstack(X)


def test_em_monotone_and_recovers_structure():
    rng = np.random.default_rng(1)
    X = _synthetic_mixture(rng)
    m = MFA(n_components=3, rank=2, max_iter=100).fit(X)
    ll = np.array(m.log_likelihood_history_)
    assert np.all(np.diff(ll) > -1e-6 * abs(ll[-1])), "EM log-likelihood must not decrease"
    assert np.allclose(np.sort(m.weights_), 1 / 3, atol=0.02)
    # far-away point gets a much larger distance than training points
    assert m.min_mahalanobis(X.max(0, keepdims=True) * 3)[0] > np.percentile(m.min_mahalanobis(X), 99)


def test_bic_prefers_true_k():
    rng = np.random.default_rng(2)
    X = _synthetic_mixture(rng)
    bics = {K: MFA(n_components=K, rank=2).fit(X).bic(X) for K in (1, 3, 6)}
    assert min(bics, key=bics.get) == 3


def test_mfa_save_load_roundtrip(tmp_path):
    rng = np.random.default_rng(3)
    X = _synthetic_mixture(rng, n=200)
    m = MFA(n_components=3, rank=2).fit(X)
    m.save(tmp_path / "m.npz")
    m2 = MFA.load(tmp_path / "m.npz")
    np.testing.assert_allclose(m.min_mahalanobis(X), m2.min_mahalanobis(X))


def test_evt_survival_and_threshold():
    rng = np.random.default_rng(4)
    d = rng.chisquare(50, size=5000)
    e = WeibullTail.fit(d)
    assert e.survival(np.array([e.tau0 - 1]))[0] == 1.0
    thr = e.threshold(0.01)
    np.testing.assert_allclose(e.survival(np.array([thr]))[0], 0.01, rtol=1e-6)
    assert compute_ood_score(thr * 1.5, e)[0] == "OOD"
    assert compute_ood_score(np.median(d), e) == ("IN_DISTRIBUTION", 1.0)


def test_fpr_at_95():
    y = np.array([0] * 100 + [1] * 100)
    s = np.concatenate([np.arange(100), np.arange(100) + 1000])
    assert fpr_at_95_tpr(y, s) == 0.0
