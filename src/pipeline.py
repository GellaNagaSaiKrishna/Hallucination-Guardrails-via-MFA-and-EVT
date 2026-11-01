"""End-to-end RAG pipeline, with and without the MFA + EVT guardrail.

    Unguarded:  query -> FAISS top-k -> LLM -> answer
    Guarded:    query -> MFA+EVT guardrail -> (OOD: fixed refusal, LLM bypassed)
                                           -> (ID : FAISS top-k -> LLM -> answer)

The generator is a local instruction-tuned LLM (Qwen2.5-1.5B-Instruct, greedy decoding) so
the experiment is reproducible offline without an API key.

Usage (interactive):  python -m src.pipeline "What was Apple's revenue from services?"
"""
from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from functools import lru_cache

import torch

from .config import LLM_MODEL, TOP_K
from .guardrail import REJECTION_MESSAGE, FaissRetriever, GuardrailRetriever, MFAEVTGuardrail

SYSTEM_PROMPT = (
    "You are a financial analyst assistant answering questions about company 10-K filings. "
    "Answer the question using ONLY the provided context. Be concise (at most 3 sentences). "
    "If the context does not contain the answer, reply exactly: "
    "\"I don't have information about this.\""
)
# Plain RAG prompt with no abstention instruction (the common "stuff the context" template).
NAIVE_SYSTEM_PROMPT = (
    "You are a financial analyst assistant. Use the provided context to answer the question. "
    "Be concise (at most 3 sentences)."
)


@lru_cache(maxsize=1)
def _load_llm():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(LLM_MODEL)
    model = AutoModelForCausalLM.from_pretrained(LLM_MODEL, dtype=torch.float32).eval()
    return tok, model


def generate_answer(question: str, contexts: list[str], max_new_tokens: int = 96,
                    system_prompt: str = SYSTEM_PROMPT) -> str:
    tok, model = _load_llm()
    ctx = "\n\n".join(f"[{i + 1}] {c}" for i, c in enumerate(contexts))
    messages = [{"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Context:\n{ctx}\n\nQuestion: {question}"}]
    prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    enc = tok(prompt, return_tensors="pt")
    with torch.inference_mode():
        out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                             pad_token_id=tok.eos_token_id)
    return tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()


@dataclass
class RAGResult:
    question: str
    answer: str
    contexts: list = field(default_factory=list)
    blocked: bool = False
    p_value: float | None = None
    d_sq: float | None = None
    seconds: float = 0.0


class RAGPipeline:
    def __init__(self, guarded: bool = True, k: int = TOP_K):
        self.base = FaissRetriever.from_disk(k=k)
        self.guarded = guarded
        self.retriever = (GuardrailRetriever(guardrail=MFAEVTGuardrail.load(), base_retriever=self.base)
                          if guarded else self.base)

    def __call__(self, question: str) -> RAGResult:
        t0 = time.perf_counter()
        docs = self.retriever.invoke(question)
        dec = getattr(self.retriever, "last_decision", None)
        if self.guarded and dec is not None and dec.is_ood:
            return RAGResult(question, REJECTION_MESSAGE, [], True, dec.p_value, dec.d_sq,
                             time.perf_counter() - t0)
        ctx = [d.page_content for d in docs]
        ans = generate_answer(question, ctx)
        return RAGResult(question, ans, ctx, False, dec.p_value if dec else None,
                         dec.d_sq if dec else None, time.perf_counter() - t0)


if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) or "What are NVIDIA's two reportable segments?"
    r = RAGPipeline(guarded=True)(q)
    print(f"Q: {q}\nblocked={r.blocked}  S(q)={r.p_value}  D2={r.d_sq}\nA: {r.answer}")
    for c in r.contexts:
        print("  -", c[:160].replace("\n", " "), "...")
