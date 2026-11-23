---
title: Guarded RAG over SEC 10-K filings
colorFrom: indigo
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
short_description: MFA+EVT pre-retrieval gate + NLI false-premise check
---

# Guarded RAG (MFA + EVT gate, false-premise check, pgvector)

A FastAPI + LangGraph service over 52,943 chunks from 50 SEC 10-K filings.

    embed -> MFA/EVT gate -> pgvector top-k -> false-premise NLI check -> answer | refuse

* `GET /healthz`
* `POST /ask` with `{"question": "What drove the increase in Costco's net sales in 2021?"}`

Source code, evaluation and measurements: https://github.com/GellaNagaSaiKrishna/Hallucination-Guardrails-via-MFA-and-EVT (branch `cloud-service`).
