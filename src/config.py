"""Central paths and default hyperparameters."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RAW = DATA / "raw"
RESULTS = ROOT / "results"
MODELS = ROOT / "results" / "models"

CHUNKS_JSON = DATA / "chunk_texts.json"
CORPUS_EMB = DATA / "corpus_embeddings.npy"
QUERIES_JSON = DATA / "test_queries.json"
QUERY_EMB = DATA / "query_embeddings.npy"
SPLIT_NPZ = DATA / "train_val_split.npz"
VAL_DIST = DATA / "validation_distances.npy"

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
NLI_MODEL = "cross-encoder/nli-deberta-v3-base"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
SEED = 0
VAL_FRACTION = 0.2
K_GRID = (2, 4, 8, 16)
R_GRID = (8, 16, 32, 64)   # proposal grid {8,16,32} extended with 64 (BIC still decreasing at 32)
EVT_TAIL_PERCENTILE = 90.0
ALPHA = 0.01
TOP_K = 3

# Labels used throughout: ID = 0 (negative), OOD = 1 (positive)
LABELS = ("IN_DISTRIBUTION", "NEAR_OOD", "FAR_OOD")

for _p in (RESULTS, MODELS):
    _p.mkdir(parents=True, exist_ok=True)
