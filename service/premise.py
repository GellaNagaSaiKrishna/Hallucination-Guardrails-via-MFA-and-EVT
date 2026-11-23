"""Post-retrieval false-premise check.

The MFA + EVT gate looks only at the question's embedding. A question such as
"Why did Costco's net sales decline in 2021?" sits squarely inside the 10-K distribution even
though the filing says sales *rose* 18%, so the gate passes it (the original report: adversarial
AUROC 0.56, "false premises look in-distribution"). This module looks at the question *together
with the retrieved chunks*.

Two kinds of checker are implemented and compared in scripts/eval_false_premise.py:

* ``NLIPresuppositionChecker`` (service default, cross-encoder/nli-MiniLM2-L6-H768): rule-based
  conversion of the question into the statement it presupposes ("Why did X decline?" ->
  "X declined") and an NLI cross-encoder over (retrieved chunk, presupposition). The score is
  max_chunk P(entailment) - max_chunk P(contradiction); a low score means the retrieved
  passages contradict the premise or support nothing in it.
* ``QAAnswerabilityChecker``: an extractive QA model fine-tuned on SQuAD 2.0, whose unanswerable
  questions were written adversarially (negation, antonyms, entity swaps; Rajpurkar et al., 2018).
  The score is max_chunk(best span logit - null logit). It was evaluated but not selected.

The threshold is calibrated on answerable questions only (the 5% quantile of their scores, a
split-conformal style rule), so no false-premise labels are used to set it.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass
from functools import lru_cache

import numpy as np


@dataclass
class PremiseResult:
    supported: bool
    score: float
    threshold: float
    best_chunk: int | None
    evidence: str | None
    latency_ms: float
    method: str


# --------------------------------------------------------------------------------------------
# QA answerability (SQuAD 2.0)
# --------------------------------------------------------------------------------------------
class QAAnswerabilityChecker:
    method = "qa"

    def __init__(self, model_name: str = "deepset/minilm-uncased-squad2", threshold: float = 0.0,
                 max_len: int = 256, max_answer_tokens: int = 30):
        import torch
        from transformers import AutoModelForQuestionAnswering, AutoTokenizer

        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForQuestionAnswering.from_pretrained(model_name).eval()
        self.threshold = threshold
        self.max_len = max_len
        self.max_answer_tokens = max_answer_tokens
        self.model_name = model_name

    def scores(self, question: str, contexts: list[str]) -> tuple[np.ndarray, list[str]]:
        """Per-chunk answerability margin (best span logit - null logit) and the best span text."""
        if not contexts:
            return np.array([]), []
        enc = self.tok([question] * len(contexts), contexts, truncation="only_second",
                       max_length=self.max_len, padding=True, return_tensors="pt",
                       return_offsets_mapping=True)
        offsets = enc.pop("offset_mapping")
        seq_ids = [enc.sequence_ids(i) for i in range(len(contexts))]
        with self.torch.inference_mode():
            out = self.model(**enc)
        S = out.start_logits.numpy()
        E = out.end_logits.numpy()
        margins, spans = [], []
        L = self.max_answer_tokens
        for i, ctx in enumerate(contexts):
            ctx_mask = np.array([s == 1 for s in seq_ids[i]])
            null = S[i, 0] + E[i, 0]
            s = np.where(ctx_mask, S[i], -1e9)
            e = np.where(ctx_mask, E[i], -1e9)
            # best (start <= end < start + L) pair
            pair = s[:, None] + e[None, :]
            n = len(s)
            band = np.triu(np.ones((n, n), dtype=bool)) & ~np.triu(np.ones((n, n), dtype=bool), k=L)
            pair = np.where(band, pair, -1e9)
            a, b = np.unravel_index(int(pair.argmax()), pair.shape)
            margins.append(float(pair[a, b] - null))
            o = offsets[i]
            spans.append(ctx[int(o[a][0]):int(o[b][1])])
        return np.array(margins), spans

    def check(self, question: str, contexts: list[str]) -> PremiseResult:
        t0 = time.perf_counter()
        m, spans = self.scores(question, contexts)
        if m.size == 0:
            return PremiseResult(False, float("-inf"), self.threshold, None, None,
                                 (time.perf_counter() - t0) * 1e3, self.method)
        j = int(m.argmax())
        return PremiseResult(bool(m[j] >= self.threshold), float(m[j]), self.threshold, j, spans[j],
                             (time.perf_counter() - t0) * 1e3, self.method)


# --------------------------------------------------------------------------------------------
# NLI over an extracted presupposition
# --------------------------------------------------------------------------------------------
_AUX = r"(?:did|does|do|has|have|had|is|are|was|were|will|would|could|can)"
_IRREGULAR = {"grow": "grew", "fall": "fell", "shrink": "shrank", "pay": "paid", "sell": "sold",
              "make": "made", "rise": "rose", "drive": "drove", "lead": "led", "buy": "bought",
              "take": "took", "spend": "spent", "begin": "began", "become": "became", "give": "gave",
              "bring": "brought", "hold": "held", "lose": "lost", "win": "won", "say": "said"}
_FILLER = re.compile(r"^(according to [^,]+, |based on [^,]+, |in (fy|fiscal )?\d{4}, )", re.I)


def _past(verb: str) -> str:
    v = verb.lower()
    if v in _IRREGULAR:
        return _IRREGULAR[v]
    if v.endswith("e"):
        return verb + "d"
    if re.search(r"[^aeiou]y$", v):
        return verb[:-1] + "ied"
    return verb + "ed"


def presupposition(question: str) -> str:
    """Very small rule set that turns a wh-question into the declarative statement it presupposes.
    Falls back to the question itself (minus '?') when no rule applies."""
    q = _FILLER.sub("", question.strip().rstrip("?").strip())
    q = re.sub(r"(?i)^by how much", "How much", q)
    m = re.match(r"(?i)^(?:what|which) (?:was|were|is|are) (.+)$", q)
    if m:                                  # "What was the value of X" -> "There was the value of X"
        return f"There was {m.group(1)}."
    # What drove / caused / explains the X  ->  There was X
    m = re.match(r"(?i)^what (?:drove|caused|explains|explained|led to|contributed to) (?:the )?(.+?) "
                 r"(to|in|at|of|for)\b(.*)$", q)
    m2 = re.match(r"(?i)^what (?:drove|caused|explains|explained|led to|contributed to) (.+?) to (\w+)(.*)$", q)
    if m2 and not re.match(r"(?i)^(the|a|an) ", m2.group(1)):
        # "What caused X to fall in 2022" -> "X fell in 2022"
        return f"{m2.group(1)} {_past(m2.group(2))}{m2.group(3)}."
    if m:
        return f"There was {('an ' if m.group(1)[0].lower() in 'aeiou' else 'a ')}{m.group(1)} {m.group(2)}{m.group(3)}."
    # Why / How / When / Where + aux + clause   ->  clause (past tense if 'did')
    m = re.match(rf"(?i)^(?:why|how|when|where) ({_AUX}) (.+)$", q)
    if m:
        return _declarative(m.group(1), m.group(2))
    # How much / How many / What / Which <NP> + aux + clause  ->  clause
    m = re.match(rf"(?i)^(?:how much|how many|what|which|who) (.+?) ({_AUX}) (.+)$", q)
    if m:
        return _declarative(m.group(2), m.group(3), obj=m.group(1))
    return q + "."


_VERBS = {"increase", "decrease", "decline", "fall", "grow", "rise", "drop", "expand", "contract",
          "collapse", "shrink", "change", "improve", "deteriorate", "affect", "impact", "drive", "cause",
          "generate", "perform", "pay", "record", "recognize", "report", "sell", "buy", "acquire",
          "complete", "relocate", "move", "own", "lease", "reduce", "commit", "decide", "shut", "close",
          "open", "launch", "release", "spend", "invest", "earn", "lose", "contribute", "exit", "enter",
          "issue", "repurchase", "announce", "disclose", "state", "describe", "plan", "expect", "use"}
_PREP = {"in", "by", "to", "from", "at", "for", "during", "of", "on", "with", "compared"}


def _declarative(aux: str, clause: str, obj: str | None = None) -> str:
    aux = aux.lower()
    words = clause.split()
    if aux in ("did", "does", "do"):
        # main verb: first word from a generic finance-verb lexicon; otherwise the word before the
        # first preposition (skipping adverbs).
        idx = next((i for i in range(1, len(words)) if words[i].lower() in _VERBS), None)
        if idx is None:
            j = next((i for i in range(1, len(words)) if words[i].lower() in _PREP), len(words))
            idx = j - 1
            while idx > 1 and words[idx].lower().endswith("ly"):
                idx -= 1
        if idx >= 1:
            w = words[idx]
            verb = (w if w.lower() == "shut" else _past(w)) if aux == "did" else w
            rest = ([obj] if obj else []) + words[idx + 1:]
            return " ".join(words[:idx] + [verb] + rest) + "."
        return clause + "."
    # is / are / was ...: insert the auxiliary after the subject (first lowercase non-determiner)
    for i in range(1, len(words)):
        if words[i].islower() and words[i] not in {"the", "a", "an", "of", "and"}:
            return " ".join(words[:i] + [aux] + words[i:]) + "."
    return clause + "."


class NLIPresuppositionChecker:
    method = "nli"

    def __init__(self, model_name: str = "cross-encoder/nli-deberta-v3-xsmall", threshold: float = 0.0):
        from sentence_transformers import CrossEncoder
        self.model = CrossEncoder(model_name, device="cpu")
        id2label = {int(k): v.lower() for k, v in self.model.model.config.id2label.items()}
        self.ent = next(i for i, l in id2label.items() if l.startswith("entail"))
        self.con = next(i for i, l in id2label.items() if l.startswith("contra"))
        self.threshold = threshold
        self.model_name = model_name

    def probs(self, question: str, contexts: list[str]) -> np.ndarray:
        h = presupposition(question)
        logits = np.asarray(self.model.predict([(c, h) for c in contexts], batch_size=16))
        p = np.exp(logits - logits.max(1, keepdims=True))
        return p / p.sum(1, keepdims=True)

    def check(self, question: str, contexts: list[str]) -> PremiseResult:
        t0 = time.perf_counter()
        if not contexts:
            return PremiseResult(False, float("-inf"), self.threshold, None, None, 0.0, self.method)
        p = self.probs(question, contexts)
        score = float(p[:, self.ent].max() - p[:, self.con].max())
        j = int(p[:, self.ent].argmax())
        return PremiseResult(score >= self.threshold, score, self.threshold, j, contexts[j],
                             (time.perf_counter() - t0) * 1e3, self.method)


class NoopChecker:
    method = "off"
    threshold = float("-inf")

    def check(self, question: str, contexts: list[str]) -> PremiseResult:
        return PremiseResult(True, float("inf"), self.threshold, 0 if contexts else None, None, 0.0, self.method)


@lru_cache(maxsize=4)
def make_checker(kind: str, model_name: str, threshold: float):
    if kind == "off":
        return NoopChecker()
    if kind == "qa":
        return QAAnswerabilityChecker(model_name, threshold)
    if kind == "nli":
        return NLIPresuppositionChecker(model_name, threshold)
    raise ValueError(f"unknown PREMISE_CHECK {kind!r}")
