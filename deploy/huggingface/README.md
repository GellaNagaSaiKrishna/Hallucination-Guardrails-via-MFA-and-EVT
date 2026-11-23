# Free deployment: Hugging Face Spaces (Docker) + Neon Postgres (pgvector)

| Piece | Service | Free-tier limits | What we use |
|---|---|---|---|
| API container | Hugging Face Spaces, Docker SDK, CPU basic | 2 vCPU, 16 GB RAM, 50 GB ephemeral disk; sleeps after 48 h without traffic | 640–790 MiB under load with the DeBERTa-v3-base NLI checker (measured with `docker stats`); image 3.75 GB |
| Vector DB | Neon Postgres (pgvector 0.8.x) | 0.5 GB storage per project; compute scales to zero when idle | 224 MiB table incl. HNSW + company indexes (about 232 MiB database), measured on Neon after ingestion |

Nothing here costs money. Neither service needs a credit card for these tiers.

## 1. Neon: create the database and load the chunks

1. Create a project at <https://console.neon.tech> (any region close to the Space, e.g. AWS us-east-1).
2. In the SQL editor run:
   ```sql
   CREATE EXTENSION IF NOT EXISTS vector;
   ```
3. Copy the connection string (*Connection details* -> *Pooled connection*). Keep it out of git:
   save it in a file outside the repository, e.g. `../neon_url.txt`.
4. Load the 52,943 chunks + embeddings and build the indexes. Use the **direct** host (no
   `-pooler` in the hostname) for the bulk load, because `COPY` and `CREATE INDEX` are long
   session-level operations:
   ```bash
   python -m venv .venv && .venv/bin/pip install numpy "psycopg[binary]" pgvector
   DATABASE_URL="$(sed 's/-pooler\././' ../neon_url.txt)" .venv/bin/python scripts/ingest_pgvector.py
   ```
   Measured on the free tier: COPY 138.6 s, HNSW build 45.4 s, 224 MiB total. If the project
   ever runs short of space, `--halfvec` stores float16 vectors (132.8 MiB measured locally).
5. Check: `python scripts/ingest_pgvector.py --size-only --database-url "$(cat ../neon_url.txt)"`.

The service itself uses the **pooled** URL. The store is PgBouncer-safe: no server-side
prepared statements (`prepare_threshold=None`) and only transaction-scoped `SET LOCAL`.

## 2. Hugging Face Space

The script stages exactly the files the Dockerfile needs into `deploy/huggingface/_space/`
(code, the fitted MFA/EVT model, the premise-check calibration, and a README with the Space
front matter `sdk: docker`, `app_port: 7860`). It then creates the Space, stores `DATABASE_URL` as a
Space **secret**, sets the non-secret variables and uploads the folder. Tokens and URLs are read
from files or env vars and are never printed or written into the staged folder.

```bash
pip install huggingface_hub
# dry run: stage only, no network
python deploy/huggingface/deploy_space.py --dry-run
# publish (suggested Space name: NagaSaiKrishna/guarded-rag-10k)
python deploy/huggingface/deploy_space.py \
    --space NagaSaiKrishna/guarded-rag-10k \
    --token-file ../hf_token.txt \
    --database-url-file ../neon_url.txt
```

You can use `HF_TOKEN` / `DATABASE_URL` environment variables instead of the two files. The first
build on Spaces takes several minutes (it installs CPU torch and bakes MiniLM + the NLI model into
the image, so a cold start downloads nothing).

Manual alternative: create a Space (SDK **Docker**, hardware **CPU basic**), add the secret
`DATABASE_URL` under *Settings -> Variables and secrets*, then `git push` the contents of
`deploy/huggingface/_space/` to the Space repository.

## 3. Use it

```bash
SPACE=https://nagasaikrishna-guarded-rag-10k.hf.space
curl $SPACE/healthz
curl -s $SPACE/ask -H 'content-type: application/json' \
     -d '{"question": "Why did Costco'"'"'s net sales decline in 2021?"}'
```

## Configuration (Space variables)

| Variable | Default | Meaning |
|---|---|---|
| `DATABASE_URL` (secret) | none | Postgres URL with pgvector; required |
| `GUARD_ALPHA` | 0.01 | EVT p-value below which a question is refused before retrieval |
| `PREMISE_CHECK` | `nli` | `nli`, `qa` or `off` |
| `PREMISE_MODEL` | `cross-encoder/nli-deberta-v3-base` | must be baked into the image (Docker build arg `PREMISE_MODEL`); `cross-encoder/nli-MiniLM2-L6-H768` is the faster, less accurate option |
| `PREMISE_THRESHOLD` | from `results/false_premise/calibration.json` | override the calibrated threshold |
| `COMPANY_FILTER` | 1 | restrict retrieval to a company named in the question |
| `GENERATOR` | `extractive` | `extractive`, `stub` or `hf:<model id>` (an LLM on 2 vCPU is slow) |
| `TOP_K` | 5 | retrieval depth |
| `HNSW_EF_SEARCH` | 200 | HNSW search breadth per query (sent with the query, one round trip) |

## Stopping / tearing down

* Pause the Space: *Settings -> Pause Space* (or `HfApi().pause_space("NagaSaiKrishna/guarded-rag-10k")`).
* Delete it: *Settings -> Delete this Space*.
* Neon: compute already scales to zero when idle. Delete the project in the console to remove the data.
