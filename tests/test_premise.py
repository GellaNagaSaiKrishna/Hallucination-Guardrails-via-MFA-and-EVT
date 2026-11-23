import os

import numpy as np
import pytest

from service.generators import ExtractiveGenerator, StubGenerator, make_generator
from service.premise import NLIPresuppositionChecker, NoopChecker, QAAnswerabilityChecker, presupposition


@pytest.mark.parametrize("q, expected", [
    ("Why did Costco's net sales decline in 2021?", "Costco's net sales declined in 2021."),
    ("What drove the increase in Costco's net sales in 2021?", "There was an increase in Costco's net sales in 2021."),
    ("What caused Ulta Beauty's net sales to fall in fiscal 2022?", "Ulta Beauty's net sales fell in fiscal 2022."),
    ("When did Amcor complete its acquisition of Bemis?", "Amcor completed its acquisition of Bemis."),
    ("How much goodwill impairment did Kraft Heinz recognize in 2019?",
     "Kraft Heinz recognized goodwill impairment in 2019."),
    ("Is this a question with no rule", "Is this a question with no rule."),
])
def test_presupposition_rules(q, expected):
    assert presupposition(q) == expected


def test_noop_checker():
    r = NoopChecker().check("q", ["a"])
    assert r.supported and r.method == "off"


class _FakeQA(QAAnswerabilityChecker):
    """Threshold logic without a model: scores are injected."""

    def __init__(self, margins, threshold):
        self.threshold, self._m = threshold, np.asarray(margins, dtype=float)

    def scores(self, question, contexts):
        return self._m, [f"span{i}" for i in range(len(self._m))]


def test_qa_checker_uses_max_margin_over_chunks():
    r = _FakeQA([-5.0, 1.5, -2.0], threshold=0.0).check("q", ["a", "b", "c"])
    assert r.supported and r.best_chunk == 1 and r.evidence == "span1" and r.score == 1.5
    r = _FakeQA([-5.0, -1.5], threshold=0.0).check("q", ["a", "b"])
    assert not r.supported and r.score == -1.5


def test_qa_checker_refuses_without_contexts():
    r = _FakeQA([], threshold=0.0).check("q", [])
    assert not r.supported and r.best_chunk is None


def test_generators():
    ctx = ["Net sales increased 18% to $192,052. Membership fees rose 9%."]
    assert ExtractiveGenerator().generate("q", ctx, evidence="18%") == "Net sales increased 18% to $192,052."
    assert ExtractiveGenerator().generate("membership fees?", ctx).startswith("Membership fees")
    assert "stub" in StubGenerator().generate("q", ctx)
    assert make_generator("stub").name == "stub"
    with pytest.raises(ValueError):
        make_generator("nope")


def _model_cached(name: str) -> bool:
    from huggingface_hub import try_to_load_from_cache
    return isinstance(try_to_load_from_cache(name, "config.json"), str)


@pytest.mark.skipif(not _model_cached("deepset/minilm-uncased-squad2") or os.environ.get("SKIP_MODEL_TESTS"),
                    reason="SQuAD2 model not in the local HF cache")
def test_real_squad2_model_separates_a_matched_pair():
    """Integration test with the real model on one hand-built pair (P01, Costco 2021)."""
    chk = QAAnswerabilityChecker("deepset/minilm-uncased-squad2", threshold=0.0)
    ctx = ["Net sales increased $28,832 or 18% during 2021. The improvement was attributable to an increase "
           "in comparable sales of 16%, and sales at new warehouses opened in 2020 and 2021."]
    ok = chk.check("What drove the increase in Costco's net sales in 2021?", ctx)
    bad = chk.check("Why did Costco's net sales decline in 2021?", ctx)
    assert ok.score > bad.score


COSTCO = ["Net sales increased $28,832 or 18% during 2021. The improvement was attributable to an increase "
          "in comparable sales of 16%, and sales at new warehouses opened in 2020 and 2021."]


@pytest.mark.skipif(not _model_cached("cross-encoder/nli-MiniLM2-L6-H768") or os.environ.get("SKIP_MODEL_TESTS"),
                    reason="NLI model not in the local HF cache")
def test_real_nli_model_flags_contradicted_premise():
    """Service default checker with its calibrated threshold on the Costco pair (P01)."""
    import json
    from pathlib import Path
    cal = json.loads((Path(__file__).resolve().parent.parent / "results" / "false_premise" /
                      "calibration.json").read_text())
    thr = cal["thresholds"]["cross-encoder/nli-MiniLM2-L6-H768"]
    chk = NLIPresuppositionChecker("cross-encoder/nli-MiniLM2-L6-H768", threshold=thr)
    assert chk.check("What drove the increase in Costco's net sales in 2021?", COSTCO).supported
    assert not chk.check("Why did Costco's net sales decline in 2021?", COSTCO).supported


def test_detect_companies():
    from service.companies import detect_companies
    assert detect_companies("Why did Costco's net sales decline?") == ["Costco"]
    assert detect_companies("What was Microsoft's revenue?") == ["Microsoft", "Microsoft Corp."]
    assert detect_companies("Are JnJ's FY2022 financials those of a growth company?") == ["Johnson & Johnson"]
    assert detect_companies("What drove AMD's operating margin change?") == []
    assert detect_companies("the aes of a meta-analysis") == []        # AES is case-sensitive
