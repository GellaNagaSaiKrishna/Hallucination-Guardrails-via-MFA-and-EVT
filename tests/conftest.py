"""Shared fakes: the graph and API tests never load MiniLM, the MFA, Postgres or a QA model."""
from __future__ import annotations

import numpy as np
import pytest

from service.premise import PremiseResult
from service.store import Hit


class FakeDecision:
    def __init__(self, ood: bool):
        self.is_ood, self.p_value, self.d_sq, self.cluster = ood, (0.001 if ood else 0.7), 900.0 if ood else 400.0, 3


class FakeGuard:
    """'OOD' iff the question mentions trivia keywords (embedding[0] == 1)."""
    alpha = 0.01

    def check_embedding(self, q):
        return FakeDecision(bool(q[0] > 0.5))


def fake_embed(question: str) -> np.ndarray:
    v = np.zeros(384, dtype=np.float32)
    v[0] = 1.0 if any(w in question.lower() for w in ("mario", "painted", "trivia")) else 0.0
    v[1] = 1.0
    return v


class FakeStore:
    def __init__(self):
        self.calls = 0

    def search(self, q, k, companies=None):
        self.calls += 1
        self.last_companies = companies
        return [Hit(f"C_{i}", "Costco", "2021", f"Net sales increased 18% to $192,052 (chunk {i}).", 0.9 - i / 10)
                for i in range(k)]

    def ping(self):
        return True


class FakeChecker:
    """Premise 'unsupported' iff the question contains 'decline'."""
    method = "fake"
    threshold = 0.0

    def check(self, question, contexts):
        bad = "decline" in question.lower()
        return PremiseResult(not bad, -3.0 if bad else 4.0, 0.0, 0, None if bad else "18%", 1.0, self.method)


class FakeGenerator:
    name = "fake"

    def generate(self, question, contexts, evidence=None):
        return f"ANSWER({evidence})"


@pytest.fixture
def components():
    from service.graph import Components
    return Components(embed=fake_embed, guard=FakeGuard(), store=FakeStore(), checker=FakeChecker(),
                      generator=FakeGenerator(), top_k=3)
