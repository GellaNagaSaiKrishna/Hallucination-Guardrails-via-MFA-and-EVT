"""Pluggable answer generators, selected with the GENERATOR env var.

  stub          deterministic echo, for tests
  extractive    (default) no LLM: the answer span the SQuAD 2.0 model found plus the sentence that
                contains it, with a citation. Runs on the free CPU tier in milliseconds.
  hf:<model>    a local Hugging Face chat model, e.g. hf:Qwen/Qwen2.5-0.5B-Instruct (CPU-feasible)
                or hf:Qwen/Qwen2.5-1.5B-Instruct (the generator of the original study)
"""
from __future__ import annotations

import re
from functools import lru_cache
from typing import Protocol

SYSTEM_PROMPT = (
    "You are a financial analyst assistant answering questions about company 10-K filings. "
    "Answer the question using ONLY the provided context. Be concise (at most 3 sentences). "
    "If the context does not contain the answer, reply exactly: \"I don't have information about this.\""
)


class Generator(Protocol):
    name: str

    def generate(self, question: str, contexts: list[str], evidence: str | None = None) -> str: ...


class StubGenerator:
    name = "stub"

    def generate(self, question: str, contexts: list[str], evidence: str | None = None) -> str:
        return f"[stub answer to: {question}] (based on {len(contexts)} chunks)"


_SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9•(])")


class ExtractiveGenerator:
    """Returns the sentence of the best chunk containing the QA evidence span (or, without a span,
    the sentence with the highest word overlap with the question)."""
    name = "extractive"

    def generate(self, question: str, contexts: list[str], evidence: str | None = None) -> str:
        if not contexts:
            return "I don't have information about this."
        best_ctx, span = contexts[0], (evidence or "").strip()
        if span:
            for c in contexts:
                if span in c:
                    best_ctx = c
                    break
        sents = [s.strip() for s in _SENT.split(best_ctx) if s.strip()]
        if span:
            hit = next((s for s in sents if span in s), None)
            if hit:
                return hit
        qw = set(re.findall(r"[a-z0-9]+", question.lower()))
        return max(sents, key=lambda s: len(qw & set(re.findall(r"[a-z0-9]+", s.lower())))) if sents else best_ctx


class HFGenerator:
    def __init__(self, model_name: str, max_new_tokens: int = 96):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.name = f"hf:{model_name}"
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float32).eval()
        self.max_new_tokens = max_new_tokens

    def generate(self, question: str, contexts: list[str], evidence: str | None = None) -> str:
        ctx = "\n\n".join(f"[{i + 1}] {c}" for i, c in enumerate(contexts))
        messages = [{"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"Context:\n{ctx}\n\nQuestion: {question}"}]
        prompt = self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        enc = self.tok(prompt, return_tensors="pt")
        with self.torch.inference_mode():
            out = self.model.generate(**enc, max_new_tokens=self.max_new_tokens, do_sample=False,
                                      pad_token_id=self.tok.eos_token_id)
        return self.tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()


@lru_cache(maxsize=2)
def make_generator(spec: str, max_new_tokens: int = 96) -> Generator:
    if spec == "stub":
        return StubGenerator()
    if spec == "extractive":
        return ExtractiveGenerator()
    if spec.startswith("hf:"):
        return HFGenerator(spec[3:], max_new_tokens)
    raise ValueError(f"unknown GENERATOR {spec!r} (stub | extractive | hf:<model id>)")
