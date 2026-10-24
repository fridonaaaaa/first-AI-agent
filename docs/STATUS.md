# STATUS.md

## Current phase
Phase 0 — Foundation and spikes

## Last finished item
Repo skeleton: folder structure, FastAPI /health, settings, errors, logging, DB schema SQL, build_subset.py

## In progress
- Phase 0 spikes: embedding model selection, LLM provider selection, storage math

## Next
1. Install dependencies and run the health test
2. Run scripts/build_subset.py and report unique paragraph count
3. Spike A: test 2–3 multilingual embedding models for Georgian support
4. Spike B: test Gemini/Groq free tiers for Georgian quality + tool-calling
5. Spike C: storage math from real unique paragraph count

## Blockers
- Unique paragraph count: UNVERIFIED (need to run build_subset.py)
- Free-tier limits for Neon/Supabase/Render/HF Spaces: UNVERIFIED
- Free-tier limits for Gemini/Groq: UNVERIFIED
- Which embedding model supports Georgian best: UNVERIFIED

## Test status
- tests/test_health.py: PENDING (dependencies not installed yet)
