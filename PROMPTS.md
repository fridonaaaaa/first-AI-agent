# PROMPTS.md — copy-paste prompts for Claude Code

How to use:
1. Create the repo folder. Put `CLAUDE.md`, `PLAN.md`, `PROMPTS.md` in its root. Claude Code loads `CLAUDE.md` from the project root at session start; the START prompt below also forces it to read the other files.
2. Start every session with the START prompt. End every session with the END prompt.
3. Run one phase prompt at a time. Review the diff and run the tests yourself before moving on.
4. If Claude Code says "UNVERIFIED" or asks a question, answer it — do not tell it to guess.

---

## START (every session)

```
Follow the session start routine in CLAUDE.md. Read CLAUDE.md, PLAN.md, docs/STATUS.md, the tail of docs/DECISIONS.md and docs/AI_USAGE.md. Run git status and the tests. Then tell me in 5 lines: current phase, last finished item, what you will do now, and the 3 rules from CLAUDE.md most relevant to it. Do not write code until I say go.
```

## END (every session)

```
Follow the session end routine in CLAUDE.md: make sure tests pass or list failures honestly, update docs/STATUS.md, append to docs/DECISIONS.md and docs/AI_USAGE.md (leave "Changed by owner:" empty), and commit. Then list in 5 lines what I must review before the next session.
```

---

## PHASE 0 — Foundation and spikes (Sat–Sun)

```
Phase 0 of PLAN.md. Goal: repo skeleton plus the measurements that decide the architecture. Do NOT build features yet.

Build:
1. Repo skeleton exactly as in PLAN.md section 3: package layout, pyproject/requirements, pytest config, Dockerfile (slim, non-root, healthcheck), docker-compose for local dev, .env.example (placeholders only), .gitignore (data/, .env, caches), docs/ files with headings (STATUS, DECISIONS, AI_USAGE, ARCHITECTURE, DEPLOYMENT, EVAL_REPORT), docs/explain/.
2. Typed settings module (env vars only), structured logging, typed error classes.
3. FastAPI app with only GET /health. Test for it.
4. DB schema + migration script for Postgres/pgvector: documents, chunks (with embedding vector, strategy, dup_group, version, metadata jsonb), plus tables for BM25 postings and stats. Do not run against any real DB yet; show me the SQL.
5. scripts/build_subset.py per PLAN.md section 5.1: seeded, writes data/manifest.json, prints the number of UNIQUE paragraphs. Report the real number to me.

Spikes (write small scripts in scripts/spikes/, run them, report real numbers, never estimate):
A. Embedding model: shortlist 2–3 multilingual candidates that support Georgian — verify each model's language support from its official model card/docs and tell me which claim you could not verify. Measure on 30 Georgian-English question/paragraph pairs: retrieval hit-rate, embedding speed on CPU, RAM, vector dimension.
B. LLM providers (free ones I can get keys for): test the same 20 Georgian questions for answer quality and tool-calling support. Check current free-tier limits from the provider's official docs/console and quote the source URL. If you cannot verify, say UNVERIFIED.
C. Storage math: from the unique paragraph count and the chosen dimension, compute expected DB size (vectors + text + postings) and tell me which free DB tier fits — after you verify the current storage caps from official pages.

Stop after the report. Do not pick the providers for me; give a recommendation with evidence and let me decide. Log everything in docs/DECISIONS.md.
```

## PHASE 1 — Ingestion (Sun)

```
Phase 1. Goal: ingestion pipeline satisfying every ingestion row of PLAN.md section 4.

Build, test-first:
1. Parsers behind one interface `parse(path_or_bytes) -> ParsedDocument(title, text, metadata, source, format)`: markdown/text, HTML (strip boilerplate, keep title/headings), PDF (text + page numbers), JSONL (HotpotQA paragraph records). Metadata must include source URL/id, format, language guess, created/fetched date when available, content hash.
2. Two chunkers behind one interface: (a) fixed-token window with overlap, (b) sentence-window grouping that respects sentence boundaries and keeps title in each chunk's metadata. Parameters from config.
3. app/ingestion/dedupe.py (owner-defended module — follow the CLAUDE.md rules for those): exact duplicate collapse by normalized-text hash (keep sources as aliases) AND near-duplicate detection with a hand-written simhash (or shingle-Jaccard) that LINKS docs through dup_group instead of deleting. Explain doc + tests including: exact dup, near-dup, and "same paragraph with one changed number" must be linked, not removed.
4. Document registry operations: add, update (new version replaces old chunks in ONE transaction), delete (chunks, postings, stats all consistent), list. Tests proving the keyword-index stats stay consistent after update/delete.
5. Bulk loader for the HotpotQA subset that goes through the SAME pipeline, resumable, with progress logging and a final count of unique records.
6. scripts/fetch_postmortems.py: parse README entries, polite fetching (rate limit, user agent, disk cache), save HTML/PDF under data/ (gitignored), skip and log failures. Do not commit article text.

Show me the test results and the count of records ingested. Stop for my review of dedupe.py before continuing.
```

## PHASE 2 — Retrieval (Mon)

```
Phase 2. Goal: hybrid search with my own algorithm components.

Follow the "Owner-defended modules" rules in CLAUDE.md strictly. Tests first.
1. app/retrieval/embeddings.py: interface + the provider I chose in Phase 0; batch embedding with retry; stores model name and dimension; refuses to query if the stored model differs from the configured one.
2. scripts/embed_corpus.py: resumable offline embedding job writing to Postgres, with progress and throughput logs. Start it and report time per 1,000 records.
3. app/retrieval/vector_store.py: pgvector search (top-k, optional filters).
4. app/retrieval/bm25.py: hand-written BM25 (k1, b configurable) scored in Python over postings. Unit tests with hand-computed expected scores on a tiny corpus. No ts_rank, no tsvector, no search library.
5. app/retrieval/fusion.py: hand-written Reciprocal Rank Fusion with configurable k. Tests with hand-computed examples, ties, and missing-in-one-list cases.
6. app/retrieval/rerank.py: coverage-aware reranker (relevance + coverage of uncovered sub-queries/entities + near-duplicate penalty, MMR-style greedy). Tests with a multi-hop toy example where plain RRF misses the second hop and the reranker fixes it.
7. app/retrieval/hybrid.py wiring it all, with config-selectable modes: dense_only, bm25_only, hybrid, hybrid_rerank.
8. docs/explain/{bm25,fusion,rerank}.md as specified.
9. A quick retrieval-only eval script on 20 questions printing Recall@5/10 and MRR for the four modes. (Full eval is Phase 5.)

Stop after bm25.py, fusion.py and rerank.py are written so I can read them. Show the numbers honestly even if hybrid does not beat dense.
```

## PHASE 3 — RAG pipeline (Mon–Tue)

```
Phase 3. Goal: the multi-step, evidence-grounded, citation-verified RAG pipeline.

Test-first, with a MockLLM for all unit tests.
1. app/rag/llm/: LLMClient interface (generate, generate_json with schema validation, tool-calling), providers for the free default I chose and an optional Claude provider, plus MockLLM. Timeouts, retry with exponential backoff, specific handling for rate-limit (429) and malformed JSON (one repair retry, then typed error). Disk cache keyed by hash of (provider, model, prompt, params).
2. Query understanding: language detection, Georgian->English search-query rewrite, decomposition into <=4 sub-queries (validated JSON).
3. Tools with JSON schemas: search_documents(query, filters), get_document(doc_id), compare_claims(chunk_ids). A bounded agent loop (max steps, max tokens) that stops cleanly and returns what it has when the budget is exhausted.
4. Conflict detection: candidate pairs (same dup_group / same title / numeric-date mismatch) -> structured LLM verdict {conflict, chunk_ids, explanation}. Never auto-resolve.
5. Sufficiency gate and abstention: retrieval-confidence threshold + LLM sufficiency judgement; abstain message lists what WAS found. Thresholds from config, tuned on DEV only (later).
6. Answer generation: only from evidence, [chunk_id] citations, answer language = question language, injection-safe prompt (retrieved text is DATA, delimited, never instructions).
7. Citation verifier: every cited id must be in the retrieved set; unsupported citations are removed/flagged and reported in the response.
8. Response schema: answer, status (answered | abstained | conflicting), sources[], conflicts[], trace[] (steps, sub-queries, scores, token counts, latency).
9. Failure tests: LLM timeout, 429, garbage JSON, empty retrieval, budget exhausted, a retrieved doc containing "ignore previous instructions".

Do not call a real LLM in tests. Report token usage per question from one real manual run using the free provider.
```

## PHASE 4 — API, UI, auth (Tue)

```
Phase 4. Goal: functional (not pretty) backend API and Georgian UI.

1. Endpoints: POST /ask; GET /documents; POST /documents (upload); PUT /documents/{id} (update); DELETE /documents/{id}; GET /documents/{id}; GET /chunks/{id}; GET /health. Pydantic schemas, consistent error format.
2. Auth: ADMIN_TOKEN from env, hmac.compare_digest, required for POST/PUT/DELETE on /documents. Tests for missing/wrong/right token.
3. Protection: per-IP rate limit and a global daily cap on /ask (config), upload size/type/count validation.
4. Single static page (vanilla HTML/JS, no build step) with all text from one i18n file in Georgian: ask box, answer with clickable citations, sources panel (title, source URL, format, version, chunk text, score), conflict display, abstention display, trace/debug toggle; document list with upload/update/delete (asks for the admin token, kept only in memory).
5. Mount the static UI in FastAPI so it is one container.
6. API integration tests with a MockLLM and a test DB.

I will deploy the container myself. Give me the exact env var list and run command.
```

## PHASE 5 — Evaluation (Tue night–Wed)

```
Phase 5. Goal: the evaluation system required by the PDF, honest and reproducible.

I will write/review the question set with you. First propose a schema and generate a DRAFT of eval/questions/*.jsonl following PLAN.md section 8 (~60 questions, DEV/TEST split, types: simple, bridge multi-hop, comparison, unanswerable, conflicting, Georgian, post-mortem). Gold data comes from HotpotQA supporting facts or from my manual labels — never invent gold answers. For post-mortem questions, leave gold as TODO for me. For unanswerable questions use eval/holdout.json (gold paragraphs removed from the index). For conflicting questions use make_synthetic.py output.

Then build:
1. eval/metrics.py (tested): Recall@5/10, MRR, answer correctness (EM/F1 + LLM judge, judge model recorded), citation precision/recall, abstention accuracy, conflict precision/recall.
2. eval/run_eval.py: takes a config file, runs the pipeline (LLM calls cached), saves results/<config>_<timestamp>.json with full per-question outputs and the config + git commit.
3. Configs: C1 dense_only, C2 hybrid, C3 hybrid_rerank, C4 full pipeline; plus chunking-strategy comparison on the slice.
4. scripts/make_report_tables.py: generate markdown tables from result files (no hand-typed numbers).
5. Run retrieval-only metrics for all configs first (no LLM cost). Run LLM-dependent metrics next, respecting the free quota, resuming from cache.
6. Error analysis helper: list the worst questions per config with retrieved vs gold, so I can categorize failures. Do NOT write conclusions the data does not support.

TEST split only for final numbers. Report honestly when something does not work.
```

## PHASE 6 — Docs and bonus (Wed–Thu)

```
Phase 6. Goal: deliverables.
1. README.md (Georgian): what it is, architecture diagram, run instructions (local + Docker), env vars, tech choices with justification (from DECISIONS.md), limitations, free-tier limits, cost per query (from real token counts), link to live demo.
2. docs/EVAL_REPORT.md (Georgian): methodology, dataset/subset justification, metrics, tables (generated), per-type results, error analysis with real failing examples, why each failure happens, concrete improvement ideas. Everything must trace to files in eval/results/.
3. docs/AI_USAGE.md final form: tools used, what each helped with, what I changed myself. Use the per-session log entries; leave my "Changed by owner" lines for me to complete.
4. Bonus, only if all core items are green, in order: LLM/embedding comparison using the existing provider interface; prompt-injection test set with results; cache hit-rate and latency metrics; streaming answers. Ask me before starting each one.
Do not claim anything in docs that is not backed by code or results in the repo.
```

## PHASE 7 — Deploy verification and interview prep (Thu)

```
Phase 7. Goal: prove the system works publicly and that I can defend it.
1. Checklist run against the live URL (I will give it): upload/update/delete with token, wrong token rejected, ask in Georgian and English, abstention case, conflict case, restart persistence test (I restart; you verify data survived), secret scan of repo history.
2. docs/DEPLOYMENT.md: architecture, env vars, storage/RAM numbers, sleep/cold-start behaviour, rate limits, $0 cost statement.
3. docs/INTERVIEW.md: step-by-step walkthrough of the RAG pipeline with file and function names; for each owner-defended module a one-page recap; 25 likely interview questions with short answers; 10 "find the bug" drills (insert a realistic bug in a COPY of a module under drills/ with a hidden answer key) and 5 "add a feature live" drills (e.g. change RRF weights, add a metadata filter, add a new file format).
Then quiz me one question at a time without showing answers first.
```

---

## Handy one-liners

Ask for verification instead of guessing:
```
Before you continue: list every fact in your plan that you have not verified from an official source or a real run, and mark each UNVERIFIED. Then ask me which ones to check.
```

Check requirements coverage at any point:
```
Open PLAN.md section 4. For every row, tell me DONE / PARTIAL / NOT STARTED with the file or test that proves it. Do not mark DONE without a passing test or a real run.
```

Rule-violation audit:
```
Audit the repo against the DO NOT list in CLAUDE.md. Grep for secrets, ts_rank/tsvector, bare except, hard-coded eval answers, committed data files, and any dependency not logged in DECISIONS.md. Report findings; fix nothing yet.
```
