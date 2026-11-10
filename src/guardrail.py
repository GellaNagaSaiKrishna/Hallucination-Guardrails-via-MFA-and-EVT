"""Week 3 / Steps 3.3-3.4 - Pre-retrieval statistical guardrail and LangChain retrievers.

    query --embed--> q --MFA (Woodbury)--> D_M^2(q) --EVT--> S(q)
        S(q) <  alpha  -> OOD: return no documents (LLM is bypassed)
        S(q) >= alpha  -> pass to the FAISS retriever
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import faiss
import numpy as np
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict

from .config import ALPHA, CHUNKS_JSON, CORPUS_EMB, MODELS, TOP_K
from .embeddings import embed
from .evt import WeibullTail
from .mfa import MFA

REJECTION_MESSAGE = "Query outside knowledge base: I don't have information about this."


@dataclass
class GuardrailDecision:
    is_ood: bool
    p_value: float
    d_sq: float
    cluster: int
    latency_ms: float

    @property
    def label(self) -> str:
        return "OOD" if self.is_ood else "IN_DISTRIBUTION"


class MFAEVTGuardrail:
    """MFA density + Weibull tail, operating directly on (normalised) embeddings."""

    def __init__(self, mfa: MFA, evt: WeibullTail, alpha: float = ALPHA):
        self.mfa, self.evt, self.alpha = mfa, evt, alpha

    @classmethod
    def load(cls, model_dir: Path = MODELS, alpha: float = ALPHA) -> "MFAEVTGuardrail":
        return cls(MFA.load(model_dir / "mfa_selected.npz"),
                   WeibullTail.load(model_dir / "evt_selected.json"), alpha)

    def score(self, Q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Returns (D_M^2, S) for a batch of embeddings."""
        d = self.mfa.min_mahalanobis(Q)
        return d, self.evt.survival(d)

    def check_embedding(self, q: np.ndarray) -> GuardrailDecision:
        t0 = time.perf_counter()
        maha = self.mfa.mahalanobis_all(q.reshape(1, -1))[0]
        k = int(maha.argmin())
        p = float(self.evt.survival(maha[k:k + 1])[0])
        return GuardrailDecision(p < self.alpha, p, float(maha[k]), k,
                                 (time.perf_counter() - t0) * 1e3)

    def check(self, query: str) -> GuardrailDecision:
        return self.check_embedding(embed(query)[0])


class FaissRetriever(BaseRetriever):
    """Plain cosine top-k retriever (IndexFlatIP on L2-normalised MiniLM embeddings)."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    index: Any
    chunks: list
    k: int = TOP_K

    @classmethod
    def from_disk(cls, k: int = TOP_K) -> "FaissRetriever":
        X = np.ascontiguousarray(np.load(CORPUS_EMB), dtype=np.float32)
        index = faiss.IndexFlatIP(X.shape[1])
        index.add(X)
        chunks = json.loads(Path(CHUNKS_JSON).read_text(encoding="utf-8"))
        return cls(index=index, chunks=chunks, k=k)

    def search_embedding(self, q: np.ndarray) -> list[Document]:
        sims, idx = self.index.search(np.ascontiguousarray(q.reshape(1, -1), dtype=np.float32), self.k)
        docs = []
        for s, i in zip(sims[0], idx[0]):
            c = self.chunks[int(i)]
            docs.append(Document(page_content=c["text"],
                                 metadata={"chunk_id": c["chunk_id"], "company": c.get("company", c.get("ticker")),
                                           "score": float(s)}))
        return docs

    def _get_relevant_documents(self, query: str, *,
                                run_manager: CallbackManagerForRetrieverRun) -> list[Document]:
        return self.search_embedding(embed(query)[0])


class GuardrailRetriever(BaseRetriever):
    """Wraps any retriever; returns [] (guardrail triggered) for OOD queries."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    guardrail: Any
    base_retriever: Any
    last_decision: Any = None

    def _get_relevant_documents(self, query: str, *,
                                run_manager: CallbackManagerForRetrieverRun) -> list[Document]:
        q = embed(query)[0]
        decision = self.guardrail.check_embedding(q)
        self.last_decision = decision
        if decision.is_ood:
            return []
        if hasattr(self.base_retriever, "search_embedding"):
            return self.base_retriever.search_embedding(q)
        return self.base_retriever.invoke(query)
