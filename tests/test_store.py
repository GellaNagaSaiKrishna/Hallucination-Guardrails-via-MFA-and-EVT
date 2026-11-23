"""Vector store tests. The pgvector test runs only when TEST_DATABASE_URL points at a database
that was loaded with scripts/ingest_pgvector.py (e.g. the docker compose `db` service)."""
import os

import numpy as np
import pytest

from service.store import MemoryStore


def _store():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(50, 8)).astype(np.float32)
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    chunks = [{"chunk_id": f"c{i}", "company": "X", "year": 2020, "text": f"t{i}"} for i in range(50)]
    return X, MemoryStore(X, chunks)


def test_memory_store_exact_topk():
    X, s = _store()
    hits = s.search(X[7], 5)
    assert hits[0].chunk_id == "c7" and abs(hits[0].score - 1.0) < 1e-5
    assert [h.score for h in hits] == sorted([h.score for h in hits], reverse=True)
    assert len(s.search(X[0], 500)) == 50


def test_memory_store_company_filter():
    X, s = _store()
    s.company[:10] = "A"
    hits = s.search(X[30], 5, companies=["A"])
    assert len(hits) == 5 and all(h.chunk_id in {f"c{i}" for i in range(10)} for h in hits)


@pytest.mark.skipif(not os.environ.get("TEST_DATABASE_URL"), reason="TEST_DATABASE_URL not set")
def test_pgvector_matches_exact_search():
    from service.store import PgVectorStore
    mem = MemoryStore.from_disk()
    pg = PgVectorStore(os.environ["TEST_DATABASE_URL"])
    try:
        assert pg.ping()
        rng = np.random.default_rng(1)
        recall = []
        for i in rng.choice(len(mem.chunks), 20, replace=False):
            q = mem.X[i]
            exact = {h.chunk_id for h in mem.search(q, 5)}
            got = pg.search(q, 5)
            assert got[0].score > 0.999          # the chunk itself (or an identical duplicate chunk)
            recall.append(len(exact & {h.chunk_id for h in got}) / 5)
            comp = mem.chunks[i]["company"]          # filtered search is exact
            # same scores (ids may differ only between identical duplicate chunks, e.g. Nike 2018/2019)
            np.testing.assert_allclose([h.score for h in pg.search(q, 5, [comp])],
                                       [h.score for h in mem.search(q, 5, [comp])], atol=1e-5)
        assert np.mean(recall) >= 0.9          # HNSW is approximate
    finally:
        pg.close()
