"""Runtime configuration, read from environment variables (12-factor; works the same locally,
in docker compose and on Hugging Face Spaces)."""
from __future__ import annotations

import os
from dataclasses import dataclass, field


def _f(name: str, default: float) -> float:
    return float(os.environ.get(name, default))


def _i(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


@dataclass(frozen=True)
class Settings:
    # postgres://user:pass@host:5432/db?sslmode=require   (Neon, docker compose, any Postgres+pgvector)
    database_url: str = field(default_factory=lambda: os.environ.get("DATABASE_URL", ""))
    # "pgvector" (default when DATABASE_URL is set) or "memory" (numpy brute force over data/*.npy;
    # used by tests and as an offline fallback)
    vector_store: str = field(default_factory=lambda: os.environ.get(
        "VECTOR_STORE", "pgvector" if os.environ.get("DATABASE_URL") else "memory"))
    table: str = field(default_factory=lambda: os.environ.get("PGVECTOR_TABLE", "chunks"))
    top_k: int = field(default_factory=lambda: _i("TOP_K", 5))
    # HNSW search breadth; 200 gives recall@5 = 0.985 vs exact search on the 311 test questions
    # (index m=32, ef_construction=200; see README)
    ef_search: int = field(default_factory=lambda: _i("HNSW_EF_SEARCH", 200))
    # MFA + EVT pre-retrieval gate: refuse if S(q) < alpha
    alpha: float = field(default_factory=lambda: _f("GUARD_ALPHA", 0.01))
    # post-retrieval false-premise check
    premise_check: str = field(default_factory=lambda: os.environ.get("PREMISE_CHECK", "qa"))  # qa|off
    premise_model: str = field(default_factory=lambda: os.environ.get(
        "PREMISE_MODEL", "deepset/minilm-uncased-squad2"))
    premise_threshold: float = field(default_factory=lambda: _f("PREMISE_THRESHOLD", float("nan")))
    # generator: "extractive" (default, CPU, no LLM), "stub" (tests), "hf:<model id>" (local HF LLM)
    generator: str = field(default_factory=lambda: os.environ.get("GENERATOR", "extractive"))
    max_new_tokens: int = field(default_factory=lambda: _i("MAX_NEW_TOKENS", 96))


def get_settings() -> Settings:
    return Settings()
