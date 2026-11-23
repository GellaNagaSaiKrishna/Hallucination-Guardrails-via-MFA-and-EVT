"""Vector stores: pgvector (Postgres) for the service, an in-memory numpy store for tests/offline.

Both return the same `Hit` records and use cosine similarity on L2-normalised MiniLM embeddings,
so FAISS IndexFlatIP, the numpy store and pgvector (`<=>` cosine distance) rank identically up to
HNSW approximation.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

ROOT = Path(__file__).resolve().parent.parent


@dataclass
class Hit:
    chunk_id: str
    company: str
    year: str
    text: str
    score: float           # cosine similarity


class VectorStore(Protocol):
    def search(self, q: np.ndarray, k: int, companies: list[str] | None = None) -> list[Hit]: ...
    def ping(self) -> bool: ...


class MemoryStore:
    """Exact cosine search over data/corpus_embeddings.npy (same ranking as the original FAISS index)."""

    def __init__(self, X: np.ndarray, chunks: list[dict]):
        self.X = np.ascontiguousarray(X, dtype=np.float32)
        self.chunks = chunks
        self.company = np.array([c["company"] for c in chunks])

    @classmethod
    def from_disk(cls) -> "MemoryStore":
        X = np.load(ROOT / "data" / "corpus_embeddings.npy")
        chunks = json.loads((ROOT / "data" / "chunk_texts.json").read_text(encoding="utf-8"))
        return cls(X, chunks)

    def search(self, q: np.ndarray, k: int, companies: list[str] | None = None) -> list[Hit]:
        sims = self.X @ q.astype(np.float32).ravel()
        if companies:
            sims = np.where(np.isin(self.company, companies), sims, -np.inf)
            k = min(k, int(np.isfinite(sims).sum())) or k
        k = min(k, len(sims))
        idx = np.argpartition(-sims, k - 1)[:k]
        idx = idx[np.argsort(-sims[idx])]
        return [Hit(self.chunks[i]["chunk_id"], self.chunks[i]["company"], str(self.chunks[i]["year"]),
                    self.chunks[i]["text"], float(sims[i])) for i in idx]

    def ping(self) -> bool:
        return True


class PgVectorStore:
    """pgvector-backed store. Works with any Postgres with the `vector` extension
    (docker `pgvector/pgvector`, Neon, Supabase, Cloud SQL, Azure Flexible Server)."""

    def __init__(self, database_url: str, table: str = "chunks", ef_search: int = 200):
        """Every search is a single network round trip (it matters on a remote database: about
        250 ms per round trip from the development laptop to Neon, versus 1 ms of query execution)."""
        from pgvector.psycopg import register_vector
        from psycopg_pool import ConnectionPool

        self.table = table
        self.ef_search = int(ef_search)

        def _configure(conn):
            register_vector(conn)

        # prepare_threshold=None: no server-side prepared statements, so the store also works through
        # PgBouncer in transaction mode (Neon "-pooler" endpoints, Supabase pooler).
        self.pool = ConnectionPool(database_url, min_size=1, max_size=4, configure=_configure,
                                   kwargs={"autocommit": True, "prepare_threshold": None}, open=True)

    def search(self, q: np.ndarray, k: int, companies: list[str] | None = None) -> list[Hit]:
        v = q.astype(np.float32).ravel()
        cols = f"SELECT chunk_id, company, year, text, 1 - (embedding <=> %s) AS score FROM {self.table}"
        with self.pool.connection() as conn:
            if companies:
                # Exact search inside the named companies' filings (<= ~4k rows): btree on company
                # + sort. Ordering by "distance + 0" keeps the planner off the HNSW index, whose
                # post-filtered approximate scan could return fewer than k rows.
                rows = conn.execute(f"{cols} WHERE company = ANY(%s) ORDER BY (embedding <=> %s) + 0 LIMIT %s",
                                    (v, list(companies), v, k)).fetchall()
            else:
                # HNSW with a per-query ef_search. "SET LOCAL ...; SELECT ..." is sent as ONE
                # simple-protocol query string, which Postgres runs as a single implicit
                # transaction: SET LOCAL applies to the SELECT, nothing leaks into the pooled
                # session (PgBouncer-safe), and it costs one round trip instead of four
                # (BEGIN/SET/SELECT/COMMIT). The simple protocol has no bind parameters, so the
                # vector is inlined as a literal built only from formatted floats.
                lit = "'[" + ",".join(f"{x:.8g}" for x in v.tolist()) + "]'"   # typed by the column (vector/halfvec)
                sql = (f"SET LOCAL hnsw.ef_search = {self.ef_search}; "
                       f"SELECT chunk_id, company, year, text, 1 - (embedding <=> {lit}) AS score "
                       f"FROM {self.table} ORDER BY embedding <=> {lit} LIMIT {int(k)}")
                cur = conn.execute(sql)
                cur.nextset()
                rows = cur.fetchall()
        return [Hit(r[0], r[1], r[2], r[3], float(r[4])) for r in rows]

    def ping(self) -> bool:
        try:
            with self.pool.connection() as conn:
                conn.execute("SELECT 1").fetchone()
            return True
        except Exception:
            return False

    def close(self) -> None:
        self.pool.close()


def make_store(settings) -> VectorStore:
    if settings.vector_store == "pgvector":
        if not settings.database_url:
            raise RuntimeError("VECTOR_STORE=pgvector but DATABASE_URL is not set")
        return PgVectorStore(settings.database_url, settings.table, settings.ef_search)
    if settings.vector_store == "memory":
        return MemoryStore.from_disk()
    raise ValueError(f"unknown VECTOR_STORE {settings.vector_store!r}")
