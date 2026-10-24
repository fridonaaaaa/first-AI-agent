# ARCHITECTURE.md

_Updated as the system is built. See DECISIONS.md for why each choice was made._

## Overview

Multi-hop RAG system that answers questions requiring evidence from multiple documents,
cites its sources, flags contradictions, and refuses to answer when evidence is missing.

## Component diagram

```
Browser (Georgian UI, HTML/JS)
   │  HTTPS
FastAPI (Docker, free host) ──── GET  /health
   │                        ──── POST /ask          (open, rate-limited per IP + daily cap)
   │                        ──── GET  /documents
   │                        ──── POST /documents    (admin token required)
   │                        ──── PUT  /documents/{id}
   │                        ──── DELETE /documents/{id}
   │                        ──── GET  /chunks/{id}
   │
   ├─ app/rag/            Multi-step pipeline
   │    ├─ query_understanding.py   language detect → translate → decompose
   │    ├─ tool_loop.py             bounded agent loop (search / get / compare)
   │    ├─ conflict.py              candidate pairs → structured LLM verdict
   │    ├─ sufficiency.py           abstention gate
   │    ├─ generator.py             evidence-only answer + [chunk_id] citations
   │    ├─ citation_verifier.py     post-generation citation check
   │    └─ llm/                     LLMClient interface + providers
   │
   ├─ app/retrieval/
   │    ├─ embeddings.py            interface + chosen provider
   │    ├─ vector_store.py          pgvector cosine search
   │    ├─ bm25.py                  hand-written BM25 over postings table
   │    ├─ fusion.py                hand-written Reciprocal Rank Fusion
   │    ├─ rerank.py                coverage-aware MMR-style reranker
   │    └─ hybrid.py                wires all modes
   │
   ├─ app/ingestion/
   │    ├─ parsers.py               md/txt, HTML, PDF, JSONL → ParsedDocument
   │    ├─ chunkers.py              fixed-token-overlap, sentence-window
   │    ├─ dedupe.py                exact hash + near-dup simhash → dup_group links
   │    └─ registry.py             add/update/delete/list documents in DB
   │
   ├─ app/core/
   │    ├─ settings.py             all config from env vars
   │    ├─ errors.py               typed exception hierarchy
   │    └─ logging.py              structlog
   │
   └─ External Postgres + pgvector (Neon or Supabase free tier, TBD after Phase 0 storage check)
```

## Layer rules

`api` → `rag` / `ingestion` → `retrieval` → `db`

- No SQL in API routes.
- No LLM calls outside `app/rag/llm/`.
- All config from `app/core/settings.py`, never hard-coded.
