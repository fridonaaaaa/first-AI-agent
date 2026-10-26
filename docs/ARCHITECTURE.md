# ARCHITECTURE.md

_See DECISIONS.md for the why behind each choice._

## Overview

Multi-hop RAG system that answers questions requiring evidence from multiple
documents, cites its sources, flags contradictions, and refuses to answer when
evidence is missing.

## Layer diagram

```
Browser (Georgian UI — app/ui/)
   │  HTTP/HTTPS
FastAPI  (app/main.py)
   ├── GET  /health
   ├── POST /ask                    ← rate-limited (IP + daily cap)
   ├── GET  /documents
   ├── POST/PUT/DELETE /documents   ← admin token (hmac.compare_digest)
   │
   ├── app/api/          API layer — no SQL, no LLM calls
   │    ├── auth.py
   │    ├── middleware.py
   │    └── routes/ask.py, documents.py, health.py
   │
   ├── app/rag/          RAG pipeline — all LLM calls live here
   │    ├── pipeline.py            run_pipeline() orchestrates 8 steps
   │    ├── query_understanding.py lang detect → translate → rewrite → decompose
   │    ├── citation.py            extract + verify [chunk_id] markers
   │    ├── conflict.py            dup-group/numeric pairs → LLM verdict
   │    ├── abstention.py          score gate + LLM sufficiency check
   │    └── llm/
   │         ├── base.py           LLMClient ABC + complete_json()
   │         ├── gemini.py         Gemini REST, retry/backoff, disk cache
   │         ├── mock.py           scriptable mock (unit tests)
   │         └── factory.py        get_llm_client() from LLM_PROVIDER env var
   │
   ├── app/retrieval/    Search & ranking — no LLM calls
   │    ├── embeddings.py          lazy SentenceTransformer singleton
   │    ├── vector_store.py        pgvector cosine (<=>)
   │    ├── bm25.py                Okapi BM25            [OWNER-DEFENDED]
   │    ├── fusion.py              Reciprocal Rank Fusion [OWNER-DEFENDED]
   │    ├── rerank.py              Coverage-aware reranker[OWNER-DEFENDED]
   │    └── hybrid.py              wires dense+BM25+RRF+reranker
   │
   ├── app/ingestion/    Ingest — no LLM calls
   │    ├── parsers.py             md/txt, html, pdf, jsonl, hotpotqa
   │    ├── chunkers.py            fixed_token + sentence_window
   │    ├── dedupe.py              SHA-256 + SimHash 64-bit [OWNER-DEFENDED]
   │    ├── tokeniser.py           shared BM25 tokeniser
   │    └── registry.py           add/update/delete + BM25 stats consistency
   │
   └── app/core/
        ├── settings.py   typed Pydantic settings from env vars
        ├── db.py         get_db() context manager
        ├── errors.py     typed exception hierarchy
        └── logging.py    structlog

External: PostgreSQL + pgvector on Supabase (free tier)
```

## Database schema (5 tables)

```
documents    doc_id, title, "source", format, language, content_hash
chunks       chunk_id, doc_id, text, embedding VECTOR(384), dup_group UUID,
             strategy, position, content_hash, metadata JSONB{token_count}
bm25_postings  term, chunk_id, tf   (tf = count / doc_length, normalised)
bm25_stats     term, df, idf
bm25_corpus    total_docs, avg_doc_len, embedding_model  (singleton)
```

## RAG pipeline — 8 steps

```
1. understand_query   detect lang → translate Georgian → rewrite → ≤4 sub-queries
2. hybrid_search      embed → dense top-40 + BM25 top-40 → RRF → top-10
3. check_abstention   gate 1: top RRF score < threshold → abstain immediately
                      gate 2: LLM sufficiency → abstain with what was found
4. detect_conflicts   same dup_group / numeric values → LLM structured verdict
5. generate_answer    LLM reads chunks + conflict notice → [chunk_id] citations
6. verify_citations   strip any [id] not in retrieved set (anti-hallucination)
7. return result      answer + sources + conflicts + sub_queries + steps_used
```

## Owner-defended modules (interview targets)

| Module | Algorithm | Lines | Explain doc |
|--------|-----------|-------|-------------|
| `dedupe.py` | SimHash 64-bit (char-4-grams, Hamming ≤5) | 148 | dedupe.md |
| `bm25.py` | Okapi BM25 (normalised TF stored, IDF from stats) | 147 | bm25.md |
| `fusion.py` | RRF: score = Σ 1/(k + rank_i) | 47 | fusion.md |
| `rerank.py` | Coverage MMR: λ·rel + (1-λ)·gain − dup_penalty | 136 | rerank.py.md |

## Security

- Admin writes: `hmac.compare_digest` — timing-attack safe
- Rate limiting: per-IP (20/min) + global daily cap (1000)
- Retrieved text: treated as untrusted DATA, never as instructions
- UI: all server-side values escaped with `escHtml()` — no XSS
- Secrets: env vars only, never in image or repo (`.dockerignore` excludes `.env`)
