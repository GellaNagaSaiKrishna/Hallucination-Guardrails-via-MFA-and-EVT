"""Stage and (optionally) publish the service as a Hugging Face Docker Space.

    # 1. dry run: only builds the staging folder deploy/huggingface/_space (no network)
    python deploy/huggingface/deploy_space.py --dry-run

    # 2. publish (creates the Space if needed, sets the DATABASE_URL secret, uploads files)
    python deploy/huggingface/deploy_space.py --space NagaSaiKrishna/guarded-rag-10k \
        --token-file ../hf_token.txt --database-url-file ../neon_url.txt

Secrets are read from files or env vars (HF_TOKEN, DATABASE_URL), never printed, and never
written into the staged folder. DATABASE_URL is stored as a Space *secret* (runtime env var).
"""
from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
STAGE = HERE / "_space"

FILES = [
    "Dockerfile", "requirements-service.txt",
    "src/__init__.py", "src/config.py", "src/embeddings.py", "src/evt.py", "src/guardrail.py",
    "src/mfa.py", "src/woodbury.py",
    "results/models/mfa_selected.npz", "results/models/evt_selected.json",
    "results/false_premise/calibration.json",
]
DIRS = ["service"]


def _secret(value: str | None, file: str | None, env: str) -> str | None:
    if value:
        return value.strip()
    if file:
        return Path(file).read_text(encoding="utf-8").strip()
    return os.environ.get(env, "").strip() or None


def stage() -> Path:
    if STAGE.exists():
        shutil.rmtree(STAGE)
    STAGE.mkdir(parents=True)
    for f in FILES:
        dst = STAGE / f
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / f, dst)
    for d in DIRS:
        shutil.copytree(ROOT / d, STAGE / d, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    shutil.copy2(HERE / "space_README.md", STAGE / "README.md")       # Space front matter
    size = sum(p.stat().st_size for p in STAGE.rglob("*") if p.is_file())
    print(f"staged {sum(1 for p in STAGE.rglob('*') if p.is_file())} files ({size / 2**20:.1f} MiB) in {STAGE}")
    return STAGE


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--space", default="NagaSaiKrishna/guarded-rag-10k")
    ap.add_argument("--token-file", default=None, help="file containing a HF write token (or set HF_TOKEN)")
    ap.add_argument("--database-url-file", default=None, help="file containing DATABASE_URL (or set DATABASE_URL)")
    ap.add_argument("--private", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    folder = stage()
    if args.dry_run:
        print("dry run: nothing uploaded")
        return

    from huggingface_hub import HfApi
    token = _secret(None, args.token_file, "HF_TOKEN")
    db_url = _secret(None, args.database_url_file, "DATABASE_URL")
    if not token:
        raise SystemExit("no HF token: pass --token-file or set HF_TOKEN")
    if not db_url:
        raise SystemExit("no DATABASE_URL: pass --database-url-file or set DATABASE_URL")

    api = HfApi(token=token)
    api.create_repo(args.space, repo_type="space", space_sdk="docker", private=args.private, exist_ok=True)
    api.add_space_secret(args.space, "DATABASE_URL", db_url)
    for k, v in {"GENERATOR": "extractive", "PREMISE_CHECK": "nli", "GUARD_ALPHA": "0.01",
                 "TOP_K": "5", "HNSW_EF_SEARCH": "200"}.items():
        api.add_space_variable(args.space, k, v)
    api.upload_folder(folder_path=str(folder), repo_id=args.space, repo_type="space",
                      commit_message="Deploy guarded RAG service")
    print(f"uploaded. Space: https://huggingface.co/spaces/{args.space}")
    print(f"API:   https://{args.space.replace('/', '-').lower()}.hf.space/healthz")


if __name__ == "__main__":
    main()
