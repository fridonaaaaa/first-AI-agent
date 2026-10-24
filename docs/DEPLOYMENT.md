# DEPLOYMENT.md

_Filled in after Phase 7 verification._

## Target architecture

- **App:** Docker container on a free host (Hugging Face Spaces Docker or Render — TBD after Phase 0 free-tier verification)
- **Database:** Postgres + pgvector on Neon or Supabase free tier (TBD after Phase 0 storage measurement)
- **Cost:** $0

## Environment variables

See `.env.example` for the full list. Required at runtime:
- `DATABASE_URL`
- `ADMIN_TOKEN`
- `LLM_PROVIDER` + the matching API key

## Storage numbers

_UNVERIFIED — to be filled after Phase 0 spike C._

## RAM numbers

_UNVERIFIED — to be filled after Phase 0 spike A (embedding model RAM measurement)._

## Cold-start / sleep behaviour

_To be documented after deployment in Phase 7._

## Restart persistence test

_To be documented after deployment in Phase 7._

## Rate limits

- Per-IP: `RATE_LIMIT_PER_IP_PER_MINUTE` (default 20 req/min)
- Global daily cap on /ask: `DAILY_CAP_ASK` (default 1000)
