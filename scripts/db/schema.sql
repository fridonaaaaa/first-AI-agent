-- Run once to set up the schema.
-- Requires the pgvector extension to be available in the Postgres instance.

CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- ────────────────────────────────────────────────────────────
-- documents
-- One row per source document (or synthetic document).
-- ────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS documents (
    id              UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    "source"        TEXT        NOT NULL,           -- URL or dataset ID (e.g. hotpotqa/abc123)
    title           TEXT        NOT NULL,
    format          TEXT        NOT NULL,           -- md | txt | html | pdf | jsonl
    version         INTEGER     NOT NULL DEFAULT 1,
    content_hash    TEXT        NOT NULL,           -- SHA-256 of normalised text; exact-dup key
    language        TEXT        NOT NULL DEFAULT 'en',
    synthetic       BOOLEAN     NOT NULL DEFAULT FALSE,
    conflict_with   UUID        REFERENCES documents(id),
    metadata        JSONB       NOT NULL DEFAULT '{}',
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE UNIQUE INDEX IF NOT EXISTS documents_content_hash_idx ON documents(content_hash);

-- ────────────────────────────────────────────────────────────
-- chunks
-- One row per chunk produced by an ingestion chunker.
-- The embedding column dimension is set at migration time from
-- the configured EMBEDDING_DIM.  Both chunking strategies share
-- this table, distinguished by the `strategy` column.
-- ────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS chunks (
    id              UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    doc_id          UUID        NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    text            TEXT        NOT NULL,
    embedding       VECTOR(384),                    -- dimension must match EMBEDDING_DIM
    strategy        TEXT        NOT NULL,           -- fixed_token | sentence_window
    dup_group       UUID,                           -- links near-duplicate chunks; NULL = unique
    position        INTEGER     NOT NULL,           -- chunk index within its document (0-based)
    content_hash    TEXT        NOT NULL,           -- SHA-256 of chunk text; exact-dup key
    metadata        JSONB       NOT NULL DEFAULT '{}',
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS chunks_doc_id_idx       ON chunks(doc_id);
CREATE INDEX IF NOT EXISTS chunks_dup_group_idx    ON chunks(dup_group) WHERE dup_group IS NOT NULL;
CREATE INDEX IF NOT EXISTS chunks_strategy_idx     ON chunks(strategy);

-- ANN index; built after bulk embedding is complete (Phase 2)
-- CREATE INDEX chunks_embedding_idx ON chunks USING hnsw (embedding vector_cosine_ops);

-- ────────────────────────────────────────────────────────────
-- bm25_postings
-- One row per (term, chunk) pair; tf is normalised term frequency.
-- Scored in Python — no ts_rank, no tsvector.
-- ────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS bm25_postings (
    term        TEXT    NOT NULL,
    chunk_id    UUID    NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
    tf          REAL    NOT NULL,   -- raw term count ÷ chunk token count
    PRIMARY KEY (term, chunk_id)
);

CREATE INDEX IF NOT EXISTS bm25_postings_term_idx ON bm25_postings(term);

-- ────────────────────────────────────────────────────────────
-- bm25_stats
-- Per-term corpus statistics; updated on every add/update/delete.
-- ────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS bm25_stats (
    term        TEXT    PRIMARY KEY,
    df          INTEGER NOT NULL,   -- number of chunks containing this term
    idf         REAL    NOT NULL    -- precomputed IDF = log((N - df + 0.5) / (df + 0.5) + 1)
);

-- ────────────────────────────────────────────────────────────
-- bm25_corpus
-- Single-row table for corpus-level BM25 parameters.
-- ────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS bm25_corpus (
    id          INTEGER PRIMARY KEY DEFAULT 1 CHECK (id = 1),
    total_docs  INTEGER NOT NULL DEFAULT 0,
    avg_doc_len REAL    NOT NULL DEFAULT 0.0,
    embedding_model TEXT NOT NULL DEFAULT '',   -- must match EMBEDDING_MODEL env var
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

INSERT INTO bm25_corpus (id) VALUES (1) ON CONFLICT DO NOTHING;
