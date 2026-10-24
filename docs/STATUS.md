# STATUS.md

## Current phase
Phase 0 — Foundation and spikes

## Last finished item
Spike C complete: 50,003 unique paragraphs measured (seed=42, validation split only).
Storage math done: 384-dim recommended (~287 MB), fits free DB tier.
LLM provider decided: Gemini.

## In progress
Phase 1 — Ingestion pipeline

## Next
1. **Owner action required:** Verify Neon or Supabase free storage cap and pick the DB host.
2. **Owner action required:** Get Gemini API key; verify current free model name and rate limits at ai.google.dev.
3. Phase 1: parsers (md/txt, HTML, PDF, JSONL)
4. Phase 1: two chunkers (fixed-token-overlap, sentence-window)
5. Phase 1: dedupe.py (owner-defended — stop for review after)
6. Phase 1: document registry (add/update/delete/list)
7. Phase 1: bulk HotpotQA loader

## Blockers
- Free-tier storage cap for Neon/Supabase: UNVERIFIED (owner must check)
- Gemini free model name + rate limits: UNVERIFIED (owner must check ai.google.dev)
- Embedding model for Georgian: decided to use 384-dim multilingual model (paraphrase-multilingual-MiniLM-L12-v2); Georgian support UNVERIFIED on model card

## Test status
- tests/test_health.py: PASSING (1/1)
