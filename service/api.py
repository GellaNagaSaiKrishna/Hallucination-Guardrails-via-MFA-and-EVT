"""FastAPI service.

    GET  /healthz   liveness + readiness (models loaded, vector store reachable)
    POST /ask       {"question": "...", "top_k": 5} -> answer or refusal with the full trace

Run:  uvicorn service.api:app --host 0.0.0.0 --port ${PORT:-7860}
"""
from __future__ import annotations

import os
import threading
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .graph import Components, build_graph, load_components
from .settings import get_settings


class AskRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=1000)
    top_k: int | None = Field(None, ge=1, le=20)


class Source(BaseModel):
    chunk_id: str
    company: str
    year: str
    score: float
    text: str


class AskResponse(BaseModel):
    question: str
    answer: str
    refused: bool
    refusal_reason: str | None
    gate: dict | None
    premise: dict | None
    companies: list[str]
    sources: list[Source]
    trace: list[str]
    timings_ms: dict
    total_ms: float


class State:
    components: Components | None = None
    graph = None
    error: str | None = None
    lock = threading.Lock()


def create_app(components: Components | None = None) -> FastAPI:
    state = State()

    def ensure_loaded():
        if state.graph is not None:
            return
        with state.lock:
            if state.graph is None:
                comps = components or load_components(get_settings())
                state.components, state.graph = comps, build_graph(comps)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if os.environ.get("EAGER_LOAD", "1") == "1":
            try:
                ensure_loaded()
            except Exception as e:  # keep serving /healthz so the failure is visible
                state.error = f"{type(e).__name__}: {e}"
        yield

    app = FastAPI(title="MFA-EVT guarded RAG", version="1.0", lifespan=lifespan)

    @app.get("/healthz")
    def healthz():
        ready = state.graph is not None
        store_ok = bool(ready and state.components.store.ping())
        body = {"status": "ok" if ready and store_ok else "degraded", "models_loaded": ready,
                "vector_store_ok": store_ok, "error": state.error}
        if ready:
            body["generator"] = getattr(state.components.generator, "name", "?")
            body["premise_check"] = getattr(state.components.checker, "method", "?")
        return body

    @app.post("/ask", response_model=AskResponse)
    def ask(req: AskRequest):
        try:
            ensure_loaded()
        except Exception as e:
            raise HTTPException(503, f"service not ready: {type(e).__name__}: {e}")
        t0 = time.perf_counter()
        out = state.graph.invoke({"question": req.question, "top_k": req.top_k})
        hits = out.get("hits") or []
        return AskResponse(
            question=req.question, answer=out["answer"], refused=out["refused"],
            refusal_reason=out.get("refusal_reason"), gate=out.get("gate"), premise=out.get("premise"),
            companies=out.get("companies") or [],
            sources=[Source(chunk_id=h.chunk_id, company=h.company, year=str(h.year), score=h.score,
                            text=h.text) for h in hits],
            trace=out.get("trace", []), timings_ms=out.get("timings_ms", {}),
            total_ms=(time.perf_counter() - t0) * 1e3)

    @app.get("/")
    def root():
        return {"service": "MFA-EVT guarded RAG over 50 SEC 10-K filings",
                "endpoints": {"POST /ask": {"question": "str", "top_k": "int (optional)"}, "GET /healthz": {}}}

    return app


app = create_app()
