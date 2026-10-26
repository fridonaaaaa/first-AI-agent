# DEPLOYMENT.md

## Target architecture

| Component | Service | Cost |
|-----------|---------|------|
| App server | Render (free Web Service) or Hugging Face Spaces Docker | $0 |
| Database | Supabase free tier — PostgreSQL + pgvector | $0 |
| LLM | Google Gemini API free tier (gemini-2.0-flash) | $0 |
| Total | | **$0** |

## Prerequisites

- Docker installed locally
- Supabase project with pgvector extension enabled
- Gemini API key (free tier)

## Step 1 — Clone and configure

```bash
git clone <repo-url>
cd "first AI agent"
cp .env.example .env
# Edit .env — fill in DATABASE_URL, ADMIN_TOKEN, GEMINI_API_KEY
```

`.env` must never be committed. It is excluded by `.gitignore` and `.dockerignore`.

## Step 2 — Create database schema

```bash
psql "$DATABASE_URL" -f scripts/db/schema.sql
```

The SQL file is idempotent (`CREATE TABLE IF NOT EXISTS`, `CREATE INDEX IF NOT EXISTS`).
Run it once; re-running is safe.

## Step 3 — Load and embed corpus

```bash
# Requires: venv activated + PYTHONPATH set, or pip install -e .
python scripts/load_hotpotqa.py   # ~50 000 HotpotQA paragraphs → documents + chunks
python scripts/embed_corpus.py    # fills embedding column in chunks (batch 64, resume-safe)
```

Embedding time: ~30 min on CPU for 50 000 chunks. The script is resume-safe — re-running
skips already-embedded chunks.

## Step 4 — Run locally (no Docker)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
uvicorn app.main:app --reload
# Open http://localhost:8000
```

## Step 5 — Run with Docker Compose

```bash
docker compose up --build
# App: http://localhost:8000
# Postgres (local dev only): localhost:5432
```

The compose file starts a local pgvector container for development.
For production, point `DATABASE_URL` at Supabase and remove the `db` service.

## Step 6 — Deploy to Render (free tier)

1. Push repo to GitHub (no `.env`, no data dumps — `.gitignore` enforces this).
2. Create a **Web Service** on Render, connect the GitHub repo.
3. Set runtime:
   - **Environment:** Docker
   - **Dockerfile path:** `Dockerfile`
4. Add environment variables in Render's dashboard (do NOT bake them into the image):
   - `DATABASE_URL` — Supabase session-pooler URL
   - `ADMIN_TOKEN` — strong random string
   - `LLM_PROVIDER` — `gemini`
   - `GEMINI_API_KEY`
5. Deploy. First deploy pulls ~1.4 GB (sentence-transformers model download).
   Subsequent deploys reuse the Docker layer cache.

### Free-tier constraints

| Limit | Render free | Impact |
|-------|------------|--------|
| RAM | 512 MB | sentence-transformers paraphrase-multilingual-MiniLM-L12-v2 uses ~420 MB — tight |
| CPU | shared | Embedding encoding is slow but functional |
| Sleep | Spins down after 15 min inactivity | First request after sleep: ~30 s cold start |
| Hours/month | 750 h | Enough for 1 service running all month |

If RAM is tight, set `SENTENCE_TRANSFORMERS_HOME=/tmp` so the model downloads to disk rather than /root.

## Environment variables

See `.env.example` for the full list. Required at runtime:

| Variable | Required | Default | Purpose |
|----------|----------|---------|---------|
| `DATABASE_URL` | Yes | — | Postgres connection string (Supabase session-pooler) |
| `ADMIN_TOKEN` | Yes | — | Protects POST/PUT/DELETE /documents |
| `LLM_PROVIDER` | No | `gemini` | `gemini` or `mock` |
| `GEMINI_API_KEY` | If gemini | — | Google AI Studio API key |
| `EMBEDDING_MODEL` | No | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | Must match what was used at embed time |
| `RATE_LIMIT_PER_IP_PER_MINUTE` | No | `20` | Per-IP /ask rate limit |
| `DAILY_CAP_ASK` | No | `1000` | Global daily /ask cap |
| `LLM_CACHE_DIR` | No | `.cache/llm` | Disk cache for LLM responses |

## Verifying a deployment

```bash
# Health
curl https://<your-app>.onrender.com/health

# Ask a question
curl -X POST https://<your-app>.onrender.com/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "Who invented the telephone?"}'

# List documents (should return 50 000+ after loading corpus)
curl https://<your-app>.onrender.com/documents
```

## Restart persistence

After a Render restart (or sleep/wake cycle):
- BM25 stats and embeddings persist in Supabase — no rebuild needed.
- The LLM disk cache (`.cache/llm/`) is ephemeral on Render free tier — responses will be re-fetched on the next cold start, but correctness is unaffected.
- The embedding model singleton reloads on first request.

## Rate limits

- Per-IP: `RATE_LIMIT_PER_IP_PER_MINUTE` (default 20 req/min, sliding 60 s window)
- Global daily: `DAILY_CAP_ASK` (default 1 000 requests/day)
- Both limits reset without a restart (in-memory counters, keyed by date/IP).

## Secrets

- Never in the Docker image or the repo.
- Injected as env vars from Render's dashboard (or host secret store).
- `.env` is excluded by both `.gitignore` and `.dockerignore`.
