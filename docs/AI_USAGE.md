# AI_USAGE.md

Log of what Claude Code generated each session. Owner fills in "Changed by owner:" lines.

---

## Session 2026-10-24

**Generated:**
- Full repo skeleton: folder structure, `__init__.py` files, `.gitignore`, `.env.example`
- `pyproject.toml` with initial dependencies
- `Dockerfile` (slim Python 3.12, non-root user, healthcheck)
- `docker-compose.yml` (app + pgvector/pgvector:pg16 DB)
- `app/core/settings.py` — typed Pydantic settings reading all env vars
- `app/core/errors.py` — typed exception hierarchy
- `app/core/logging.py` — structlog configuration (dev/prod renderer)
- `app/main.py` — FastAPI app with lifespan hook
- `app/api/routes/health.py` — GET /health endpoint
- `tests/test_health.py` — health check test
- `scripts/db/schema.sql` — full Postgres schema (documents, chunks, bm25_postings, bm25_stats, bm25_corpus)
- `scripts/build_subset.py` — seeded HotpotQA subset builder writing data/manifest.json
- `docs/STATUS.md`, `docs/DECISIONS.md`, `docs/AI_USAGE.md`, `docs/ARCHITECTURE.md`, `docs/DEPLOYMENT.md`, `docs/EVAL_REPORT.md` — doc skeletons
- Spike script stubs: `scripts/spikes/spike_embeddings.py`, `scripts/spikes/spike_llm.py`

**Changed by owner:**
