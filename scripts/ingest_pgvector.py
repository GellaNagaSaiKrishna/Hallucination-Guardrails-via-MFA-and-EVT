"""Load the 52,943 chunks + MiniLM embeddings into Postgres/pgvector.

Works against any Postgres with the `vector` extension: local docker compose, Neon (free tier),
Supabase, Cloud SQL, Azure Flexible Server.

    python scripts/ingest_pgvector.py --database-url "$DATABASE_URL"            # full load + HNSW
    python scripts/ingest_pgvector.py --database-url ... --halfvec              # 2-byte floats (half size)
    python scripts/ingest_pgvector.py --database-url ... --size-only            # just print table sizes

Idempotent: the table is dropped and recreated unless --append is given. Uses binary COPY, so a
full load takes well under a minute on a local container.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import psycopg
from pgvector import HalfVector
from pgvector.psycopg import register_vector

ROOT = Path(__file__).resolve().parent.parent


def ddl(table: str, dim: int, halfvec: bool) -> list[str]:
    vtype = f"halfvec({dim})" if halfvec else f"vector({dim})"
    return [
        "CREATE EXTENSION IF NOT EXISTS vector",
        f"DROP TABLE IF EXISTS {table}",
        f"""CREATE TABLE {table} (
                id        integer PRIMARY KEY,
                chunk_id  text NOT NULL UNIQUE,
                doc_name  text NOT NULL,
                company   text NOT NULL,
                year      text NOT NULL,
                text      text NOT NULL,
                embedding {vtype} NOT NULL)""",
    ]


def index_sql(table: str, halfvec: bool, m: int = 32, ef_construction: int = 200) -> str:
    ops = "halfvec_cosine_ops" if halfvec else "vector_cosine_ops"
    return (f"CREATE INDEX IF NOT EXISTS {table}_embedding_hnsw ON {table} "
            f"USING hnsw (embedding {ops}) WITH (m = {int(m)}, ef_construction = {int(ef_construction)})")


def report_sizes(conn, table: str) -> dict:
    row = conn.execute(f"""SELECT pg_total_relation_size('{table}'), pg_relation_size('{table}'),
                                  pg_indexes_size('{table}'), pg_database_size(current_database()),
                                  (SELECT count(*) FROM {table})""").fetchone()
    names = ("total_bytes", "heap_bytes", "indexes_bytes", "database_bytes", "rows")
    sizes = dict(zip(names, map(int, row)))
    for k, v in sizes.items():
        print(f"  {k:15s} {v:>14,d}" + (f"  ({v / 2**20:.1f} MiB)" if k != "rows" else ""))
    return sizes


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--database-url", default=os.environ.get("DATABASE_URL"))
    ap.add_argument("--table", default=os.environ.get("PGVECTOR_TABLE", "chunks"))
    ap.add_argument("--halfvec", action="store_true", help="store float16 halfvec (pgvector >= 0.7)")
    ap.add_argument("--no-index", action="store_true")
    ap.add_argument("--hnsw-m", type=int, default=32)
    ap.add_argument("--hnsw-ef-construction", type=int, default=200)
    ap.add_argument("--maintenance-work-mem", default="256MB",
                    help="memory for the HNSW build (Neon free tier: keep <= 256MB)")
    ap.add_argument("--size-only", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="load only the first N chunks (smoke tests)")
    args = ap.parse_args()
    if not args.database_url:
        raise SystemExit("set --database-url or DATABASE_URL")

    with psycopg.connect(args.database_url, autocommit=True) as conn:
        if args.size_only:
            report_sizes(conn, args.table)
            return
        chunks = json.loads((ROOT / "data" / "chunk_texts.json").read_text(encoding="utf-8"))
        X = np.load(ROOT / "data" / "corpus_embeddings.npy").astype(np.float32)
        assert len(chunks) == X.shape[0], (len(chunks), X.shape)
        if args.limit:
            chunks, X = chunks[:args.limit], X[:args.limit]
        for stmt in ddl(args.table, X.shape[1], args.halfvec):
            conn.execute(stmt)
        register_vector(conn)

        t0 = time.perf_counter()
        vtype = "halfvec" if args.halfvec else "vector"
        with conn.cursor() as cur:
            with cur.copy(f"COPY {args.table} (id, chunk_id, doc_name, company, year, text, embedding) "
                          f"FROM STDIN WITH (FORMAT BINARY)") as copy:
                copy.set_types(["int4", "text", "text", "text", "text", "text", vtype])
                for i, (c, v) in enumerate(zip(chunks, X)):
                    emb = HalfVector(v) if args.halfvec else v
                    # NUL bytes are not allowed in Postgres text
                    copy.write_row([i, c["chunk_id"], c["doc_name"], c["company"], str(c["year"]),
                                    c["text"].replace("\x00", ""), emb])
        print(f"copied {len(chunks):,} rows in {time.perf_counter() - t0:.1f}s")

        if not args.no_index:
            t0 = time.perf_counter()
            conn.execute(f"SET maintenance_work_mem = '{args.maintenance_work_mem}'")
            conn.execute(index_sql(args.table, args.halfvec, args.hnsw_m, args.hnsw_ef_construction))
            print(f"built HNSW index in {time.perf_counter() - t0:.1f}s")
        conn.execute(f"ANALYZE {args.table}")
        report_sizes(conn, args.table)


if __name__ == "__main__":
    main()
