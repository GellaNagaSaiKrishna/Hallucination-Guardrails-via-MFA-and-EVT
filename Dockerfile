# Guarded RAG service: FastAPI + LangGraph + pgvector, CPU only.
# Built for Hugging Face Spaces (Docker SDK, free CPU tier); also runs anywhere that sets $PORT.
FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    TOKENIZERS_PARALLELISM=false \
    OMP_NUM_THREADS=2 \
    PORT=7860

# HF Spaces runs the container as uid 1000
RUN useradd -m -u 1000 user

COPY requirements-service.txt /tmp/requirements-service.txt
RUN pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu \
 && pip install -r /tmp/requirements-service.txt

USER user
ENV HOME=/home/user \
    HF_HOME=/home/user/.cache/huggingface
WORKDIR /home/user/app

# Bake the two small models into the image so a cold start does not download anything:
# MiniLM embedder (~90 MB) + the false-premise NLI cross-encoder (DeBERTa-v3-base)
ARG PREMISE_MODEL=cross-encoder/nli-deberta-v3-base
ENV PREMISE_MODEL=${PREMISE_MODEL}
RUN python -c "from huggingface_hub import snapshot_download as s; \
s('sentence-transformers/all-MiniLM-L6-v2', allow_patterns=['*.json','*.txt','*.safetensors','1_Pooling/*']); \
s('${PREMISE_MODEL}', allow_patterns=['*.json','*.txt','*.safetensors','*.model'])"
ENV HF_HUB_OFFLINE=1

# code + fitted guardrail (MFA K=16 r=64, EVT tail) + calibrated premise threshold.
# The 52,943 chunks live in Postgres/pgvector, not in the image.
COPY --chown=user src/__init__.py src/config.py src/embeddings.py src/evt.py src/guardrail.py \
     src/mfa.py src/woodbury.py ./src/
COPY --chown=user service ./service
COPY --chown=user results/models/mfa_selected.npz results/models/evt_selected.json ./results/models/
COPY --chown=user results/false_premise/calibration.json ./results/false_premise/

EXPOSE 7860
CMD ["sh", "-c", "exec uvicorn service.api:app --host 0.0.0.0 --port ${PORT:-7860} --workers 1"]
