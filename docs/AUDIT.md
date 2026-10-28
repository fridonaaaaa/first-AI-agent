# Phase 0 Audit — AI Investigation Engine
_Date: 2026-10-28. Read-only pass. No code changes._

---

## 1. Repo state

**`git status`** (uncommitted):
- `modified: Dockerfile` — added libgomp1
- `modified: app/retrieval/embeddings.py` — Gemini backend, retry logic
- `modified: scripts/embed_corpus.py` — --sleep flag, retry on failure
- `untracked: gorda.py` — scratch file, not part of the project

**Pytest:** `129 passed` (run via `.venv/bin/pytest -q`)

---

## 2. Embedding progress

| Metric | Count |
|--------|-------|
| Total chunks | 78,051 |
| Embedded | 25,220 |
| NULL (pending) | 52,831 |

Embed job appears **running** — progress is non-zero and was advancing. Job is
`python3 scripts/embed_corpus.py --batch 1024` running on the Mac.

---

## 3. DB size

| Table | Size |
|-------|------|
| bm25_postings | 299 MB |
| chunks | 192 MB |
| documents | 15 MB |
| bm25_stats | 8.4 MB |
| bm25_corpus | 64 kB |
| **Total DB** | **526 MB** |

**⚠️ CRITICAL: DB is 526 MB, exceeding the Supabase free-tier 500 MB limit.**
Headroom: −26 MB (already over). Phase 3 corpus loading must be skipped.

---

## 4. Claim verification

### a. File formats — PARTIALLY VERIFIED

Backend `app/ingestion/parsers.py` supports: **txt, md, html, pdf, jsonl** (5 formats).

UI `app/ui/index.html:594` accept attribute: `.pdf,.docx,.txt,.md,.html,.htm`

**Issues:**
- `.docx` is in the UI accept list but **no docx parser exists** in parsers.py → upload will fail silently or error.
- `.jsonl` is supported by the backend but **not in the UI accept list** → inaccessible from UI.

### b. Two chunking strategies — VERIFIED

`app/ingestion/chunkers.py:36` — `chunk_fixed_token()`
`app/ingestion/chunkers.py:82` — `chunk_sentence_window()`
Both dispatched from `chunk_document()` at line 129.

### c. Document update endpoint — VERIFIED

`app/api/routes/documents.py:225` — `PUT /documents/{doc_id}` exists and calls
`update_document()` from `app/ingestion/registry.py:189`.

**UI:** No update/replace action in the file sidebar. The UI has delete (✕) only.

### d. Tool calling — NOT TRUE

The pipeline docstring at `pipeline.py:13–16` claims a "tool loop" with
`search_documents`, `get_document`, and `compare_claims`. **The actual code does
not implement this.** `run_pipeline()` calls `hybrid_search()` directly twice and
has no LLM function-calling dispatch, no tool schema definitions, and no tool loop.
`max_tool_steps` is used only to limit the second `hybrid_search` call. The claim
in the status report is false.

### e. Abstention gate — PARTIALLY DEAD CODE

Threshold: `settings.abstention_threshold = 0.010` (`app/core/settings.py:48`).

Minimum possible RRF score when results exist: `1/(k + top_k) = 1/(60+10) = 0.01428`.

Since `0.01428 > 0.010`, **the score-based pre-LLM gate can never fire** when
`hybrid_search` returns any results. The `no_results` branch (empty list) still
works correctly. The `low_score` branch is dead code for any non-empty result set.

### f. Embedding fallback safety — RISK

If fastembed import fails and Gemini is used at query time, `_encode_gemini()`
returns 384-dim vectors (outputDimensionality=384 is set). Dimension matches.
**However:** there is no dimension guard anywhere. If `EMBEDDING_BACKEND` is
changed, or a different model is used, mismatched vectors would be stored or
queried silently with no error. Should add a guard.

### g. Public upload — VERIFIED (visitors CANNOT upload)

`app/api/auth.py:17` — `require_admin` checks `x_admin_token` header via
`hmac.compare_digest`. All POST, PUT, DELETE document endpoints have
`dependencies=[Depends(require_admin)]`. A visitor without the token cannot
upload, update, or delete anything. Read and search are open.

This is correct security-wise but contradicts Phase 2 item 6 (session-scoped
uploads should be open). That is a feature gap, not a bug.

### h. Live Render

Cannot check directly from local environment. Deployment was confirmed working
earlier in the session. Health endpoint was responding. Cold start on Render free
tier is ~30–60 s after inactivity.

### i. Documentation — EXISTS, INCOMPLETE

| File | Lines | Notes |
|------|-------|-------|
| README.md | 148 | Exists; needs live URL and Render section |
| docs/ARCHITECTURE.md | 102 | Exists; pre-session-id, needs update |
| docs/DEPLOYMENT.md | 142 | Exists; pre-libgomp/fastembed, needs update |
| docs/DECISIONS.md | 204 | Exists; missing today's decisions |
| docs/EVAL_REPORT.md | 72 | **Placeholder only — all numbers are `—`** |
| docs/AI_USAGE.md | 163 | Exists; today's session not yet logged |

### j. Eval scaffolding — EXISTS, SCHEMA MISMATCH

Scripts:
- `scripts/eval_retrieval.py` — exists, `--help` works
- `scripts/eval_rag.py` — exists, `--help` works
- `eval/configs/` — 4 JSON config files exist

Questions:
- `eval/questions/dev.jsonl` — **15 lines** (need ≥40)
- `eval/questions/test.jsonl` — **25 lines** (need ≥20 — count OK but schema wrong)

**Schema mismatch:** existing questions use `type`, `answer`, `supporting_titles`.
Required schema: `id`, `split`, `category`, `question`, `gold_answer`,
`gold_doc_ids`, `expected_behavior`. The eval scripts may not handle the current
schema correctly — needs verification before running.

---

## 5. Corpus

| Metric | Value |
|--------|-------|
| Documents | 29,034 |
| Chunks | 78,051 |
| DB size | 526 MB |

**Phase 3 must be skipped.** Adding more documents would push the DB further over
the 500 MB Supabase free-tier limit. 78,051 chunks already satisfies the "≥50,000
records" requirement (interpret "records" as chunks).

Estimated cost of loading to 50,000 *documents* (from ~29k now):
~21,000 more docs × ~2.7 chunks/doc × ~2.5 KB/chunk ≈ +140 MB — would bring DB to
~666 MB, well above the limit. **Skip.**

---

## 6. Risk list

| # | Risk | Severity |
|---|------|----------|
| 1 | DB already at 526 MB, over Supabase 500 MB free limit | HIGH |
| 2 | Tool calling claimed in docs but not implemented in code | HIGH |
| 3 | Eval questions: only 15 dev / wrong schema / no gold_doc_ids | HIGH |
| 4 | Abstention score gate is dead code (threshold too low for RRF) | MEDIUM |
| 5 | UI accepts .docx but no docx parser → silent upload failure | MEDIUM |
| 6 | .jsonl format in backend not exposed in UI accept attribute | LOW |
| 7 | No embedding dimension guard — model switch would corrupt vectors silently | MEDIUM |
| 8 | EVAL_REPORT.md contains only placeholder dashes — no real numbers | HIGH |
| 9 | Today's code changes (Dockerfile, embeddings, UI, session_id) not committed | MEDIUM |
| 10 | `gorda.py` scratch file in repo root — should not be committed | LOW |
