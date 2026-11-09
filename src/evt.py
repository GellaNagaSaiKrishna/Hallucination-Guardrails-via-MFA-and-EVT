"""Week 3 - Extreme Value Theory calibration of the Mahalanobis tail.

Peaks-over-threshold with a Weibull model on exceedances:
    tau_0 = 90th percentile of validation D_M^2
    E     = { D - tau_0 : D > tau_0 }
    F(e)  = 1 - exp(-(e / lambda)^kappa)        (scipy.stats.weibull_min, floc = 0)

Score of a query (p-value under H0: "query is in-distribution"):
    S(q) = exp( -( max(0, D_M^2(q) - tau_0) / lambda )^kappa )
Reject as OOD when S(q) < alpha.

S(q) is the tail survival *conditional on exceeding tau_0* (as in the proposal).  The
unconditional p-value is (1 - p_tail) * S(q) with p_tail = 0.90; both are exposed.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from scipy import stats


@dataclass
class WeibullTail:
    tau0: float
    kappa: float
    lam: float
    tail_percentile: float = 90.0
    n_exceed: int = 0
    ks_stat: float = float("nan")
    ks_pvalue: float = float("nan")

    @classmethod
    def fit(cls, distances: np.ndarray, tail_percentile: float = 90.0) -> "WeibullTail":
        distances = np.asarray(distances, dtype=np.float64)
        tau0 = float(np.percentile(distances, tail_percentile))
        E = distances[distances > tau0] - tau0
        kappa, _, lam = stats.weibull_min.fit(E, floc=0)
        ks = stats.kstest(E, "weibull_min", args=(kappa, 0, lam))
        return cls(tau0, float(kappa), float(lam), tail_percentile, int(E.size),
                   float(ks.statistic), float(ks.pvalue))

    def exceedances(self, distances: np.ndarray) -> np.ndarray:
        return np.maximum(0.0, np.asarray(distances, dtype=np.float64) - self.tau0)

    def survival(self, distances: np.ndarray) -> np.ndarray:
        """Conditional tail p-value S(q) in [0, 1]; equals 1 for D <= tau_0."""
        return np.exp(-(self.exceedances(distances) / self.lam) ** self.kappa)

    def unconditional_pvalue(self, distances: np.ndarray) -> np.ndarray:
        d = np.asarray(distances, dtype=np.float64)
        p_tail = 1.0 - self.tail_percentile / 100.0
        return np.where(d > self.tau0, p_tail * self.survival(d), np.nan)

    def threshold(self, alpha: float) -> float:
        """Distance D* at which S(D*) = alpha."""
        return self.tau0 + self.lam * (-np.log(alpha)) ** (1.0 / self.kappa)

    def bootstrap_ks_pvalue(self, distances: np.ndarray, n_boot: int = 2000, seed: int = 0) -> float:
        """Parametric-bootstrap (Lilliefors-type) KS p-value.  The plain ``ks_pvalue`` is optimistic
        because kappa and lambda are estimated from the same exceedances; here the fit is repeated on
        every simulated sample so the null distribution of the KS statistic accounts for it."""
        d = np.asarray(distances, dtype=np.float64)
        E = d[d > self.tau0] - self.tau0
        ks0 = stats.kstest(E, "weibull_min", args=(self.kappa, 0, self.lam)).statistic
        rng = np.random.default_rng(seed)
        hits = 0
        for _ in range(n_boot):
            sim = stats.weibull_min.rvs(self.kappa, 0, self.lam, size=E.size, random_state=rng)
            k, _, lam = stats.weibull_min.fit(sim, floc=0)
            hits += stats.kstest(sim, "weibull_min", args=(k, 0, lam)).statistic >= ks0
        return (hits + 1) / (n_boot + 1)

    def save(self, path: str | Path) -> None:
        d = asdict(self)
        d["lambda"] = d.pop("lam")
        Path(path).write_text(json.dumps(d, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "WeibullTail":
        d = json.loads(Path(path).read_text())
        d["lam"] = d.pop("lambda")
        return cls(**d)


def compute_ood_score(d_sq: float, evt: WeibullTail, alpha: float = 0.01) -> tuple[str, float]:
    """Decision rule from the blueprint (Step 3.3)."""
    p_val = float(evt.survival(np.array([d_sq]))[0])
    return ("OOD" if p_val < alpha else "IN_DISTRIBUTION"), p_val
