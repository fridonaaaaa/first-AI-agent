# PLAN.md — AI Investigation Engine

Deadline: Thursday 2026-10-29 (owner stated; the PDF itself gives no date — confirm end-of-day).
Owner: DevOps engineer. Deployment/infra by the owner; application code by Claude Code with owner review.

## 1. Verified facts (checked 2026-10-24)

- HotpotQA (HF `hotpotqa/hotpot_qa`): license CC BY-SA 4.0. Config `distractor`: train 90,447 rows, validation 7,405 rows. Each row has `question`, `answer`, `type` (bridge/comparison), `level`, `supporting_facts` (titles + sentence ids) and `context` (10 paragraphs: `title` + list of `sentences`).
- `danluu/post-mortems` is NOT a set of documents. It is one README of ~300 entries: company name, link to an EXTERNAL article (many via web.archive.org), and a 1–3 sentence summary. The full texts must be fetched from those links. Some links are PDFs, most are HTML. Categories in the README: Config Errors, Hardware/Power Failures, Conflicts, Time, Database, Uncategorized.
- UNVERIFIED (measure in Phase 0): number of UNIQUE paragraphs in the HotpotQA subset; free-tier storage/RAM/sleep limits of Neon, Supabase, Hugging Face Spaces, Render; free LLM limits for Gemini/Groq (they change; check each provider's console/doc page); Georgian quality of candidate embedding models and LLMs.

## 2. Decisions

| # | Decision | Status |
|---|---|---|
| D1 | Corpus = HotpotQA paragraphs (each paragraph = one record) + post-mortem articles + synthetic docs | decided |
| D2 | Postgres + pgvector on a free cloud tier (Neon or Supabase; pick after storage check) | decided, provider TBD |
| D3 | One Docker container: FastAPI backend + static HTML/JS UI (Georgian) | decided |
| D4 | Admin token for upload/update/delete; asking is open + rate limited | decided |
| D5 | LLM behind a provider interface; free default (Gemini or Groq, tested on Georgian); Claude optional | decided, default TBD |
| D6 | Georgian UI/README/reports; questions in Georgian or English; documents English | decided |
| D7 | Own components: BM25, RRF, coverage-aware reranker, near-dup detection (Draft-and-Defend: Claude Code drafts + tests + explain doc, owner reviews/rewrites) | owner to confirm |
| D8 | "50,000 records" counted as unique paragraphs after dedupe | owner reading; optionally confirm with Upgaming |

### Critical technical constraints
- **Embedding model must be identical for documents and queries.** Options: (A) small multilingual model run locally inside the container (check RAM), or (B) one hosted embedding API for both (check free rate limits for 50k docs). Phase 0 spike decides. 50k vectors cost 50,000 × dims × 4 bytes: 384 dims ≈ 77 MB, 1024 dims ≈ 205 MB, before text and index overhead.
- **BM25 index storage.** Two options, decided by a Phase 0 measurement: (1) postings table in Postgres (term, chunk_id, tf) + stats table, scored in Python; (2) in-memory index built at startup from stored chunks. Check against free DB storage cap and free container RAM.
- **Georgian vs BM25.** Documents are English, so Georgian queries are translated/rewritten to English by the query-understanding step before BM25. Semantic search uses the multilingual embedding directly as a second path.
- **Near-duplicates are features, not bugs.** Synthetic conflicting documents are near-duplicates of originals by design. Dedupe must link them (`dup_group`), not delete them.
- **Chunking comparison on a smaller slice.** Storing two chunking strategies for all 50k records doubles storage. Compare strategies on a slice (post-mortems + ~5k paragraphs), then run the full corpus with the winner.
- **Retrieval metrics need no LLM.** Recall@K and MRR run for free on every config. Only answer correctness, citation accuracy and abstention need LLM calls — cache them and spread over days if the free quota is tight.

## 3. Architecture

```
Browser (Georgian UI, HTML/JS)
   │  HTTPS
FastAPI (Docker, free host) ──── /ask (open, rate-limited)
   │                        ──── /documents (admin token): upload, update, delete, list
   │
   ├─ rag/        query rewrite+translate → decompose → tool loop → conflicts → answer → citation check
   │    └─ llm/   LLMClient interface (gemini | groq | claude | mock), retries, timeouts, disk cache
   ├─ retrieval/  dense (pgvector) + BM25 (own) → RRF (own) → coverage reranker (own)
   ├─ ingestion/  parsers (md/txt, html, pdf, jsonl) → metadata → chunkers (2 strategies) → dedupe → store
   └─ db          Postgres + pgvector (external, persistent): documents, chunks, postings/stats
```

Repo layout:
```
app/{api,core,ingestion,retrieval,rag,ui}   eval/{questions,configs,results}
scripts/   tests/   docs/{STATUS,DECISIONS,AI_USAGE,ARCHITECTURE,DEPLOYMENT,EVAL_REPORT}.md  docs/explain/
Dockerfile  docker-compose.yml  .env.example  CLAUDE.md  PLAN.md  PROMPTS.md  README.md
```

## 4. Requirement traceability (every line of the PDF)

| PDF requirement | How we meet it | Phase | Done when |
|---|---|---|---|
| ≥3 file formats | `.md/.txt`, `.html`, `.pdf`, `.jsonl` parsers | 1 | tests ingest one file of each type |
| Text + metadata extraction | title, source URL, format, dates, page/section, content hash | 1 | metadata visible in UI source view |
| Chunking | sentence-window and fixed-token-with-overlap | 1 | both implemented, unit-tested |
| ≥2 chunking strategies compared | eval on slice, Recall@K/MRR per strategy | 5 | table in Eval Report |
| Duplicate handling | exact hash collapse + near-dup link (own simhash) | 1 | tests for exact, near, conflicting-doc cases |
| Add / update / delete docs | versioned replace in a transaction; BM25 stats kept consistent | 1, 4 | API + UI + tests incl. index consistency |
| Justify architecture | `docs/DECISIONS.md`, README | 6 | every D-row has rationale |
| Vector DB | Postgres + pgvector | 0, 2 | vectors persisted, ANN/exact search works |
| Embeddings + semantic search | multilingual embedding model | 2 | dense Recall@K measured |
| Hybrid search | dense + own BM25 fused by own RRF | 2 | beats each single path in eval (or honestly reported if not) |
| ≥1 own search/ranking component | BM25, RRF, coverage reranker, simhash — all hand-written | 2 | owner can explain each line |
| Not plain SQL text search | BM25 scored in Python from postings | 2 | no `ts_rank` anywhere (grep check) |
| ≥1 LLM | provider interface; free default | 3 | live answers on deployed app |
| RAG + multi-doc merge | retrieve per sub-query, merge, generate from evidence only | 3 | multi-hop questions answered with ≥2 cited docs |
| Multi-step handling | rewrite → decompose → tool loop with step budget | 3 | trace visible in UI/debug endpoint |
| Evidence-based answers | answer only from provided chunks | 3 | prompt + post-check tests |
| Citations | `[chunk_id]` markers, verified against retrieved set | 3 | unverified citations flagged in tests |
| Contradiction detection | candidate pairs by dup_group/entity + LLM structured check | 3 | conflict eval set P/R measured |
| Abstention | score gate + LLM sufficiency check | 3 | unanswerable eval set measured |
| LLM error handling | timeout, retry/backoff, typed errors, 429 handling, fallback message | 3 | tests with mock failures |
| Tool calling | `search_documents`, `get_document`, `compare_claims` | 3 | schemas + loop tested |
| ≥40 eval questions | ~60: simple, multi-hop, comparison, unanswerable, conflicting, Georgian | 5 | JSONL committed, dev/test split |
| Quantitative eval | Recall@5/10, MRR, answer correctness, citation accuracy, abstention accuracy, conflict P/R | 5 | results JSON + generated tables |
| ≥2 configs compared | dense-only vs hybrid vs hybrid+rerank(+decomposition) | 5 | ablation table |
| Error analysis | where/why it fails, how to improve | 5, 6 | section in Eval Report with real failing examples |
| Python backend + UI | FastAPI + static page | 4 | upload, manage, ask, view sources all work |
| ≥50,000 records | unique paragraphs after dedupe | 0, 1 | count reported by script |
| Multi-doc linked info | HotpotQA bridge/comparison + post-mortem cross-references | 5 | eval covers it |
| Source identity preserved | doc_id/source/title/version on every chunk | 1 | UI shows source per citation |
| Incomplete + contradictory data | unanswerable cases (gold docs removed) + synthetic conflicting docs | 5 | labelled in eval set |
| Reproducible eval + subset | seed, manifest of IDs, cached LLM calls | 0, 5 | `make eval` reproduces numbers |
| Public URL, free hosting | Docker on free host + external free DB | 0 (skeleton), 7 | URL works from another network |
| Persistence after restart | all state in external Postgres | 7 | restart test documented |
| Secrets protected | env/secret store, `.env.example`, no secrets in image/repo | 0, 7 | secret scan clean |
| Deployment architecture + limits documented | `docs/DEPLOYMENT.md` | 7 | sleep/RAM/rate limits written down |
| Modular code, error handling, tests | per CLAUDE.md | all | CI-style `pytest` green |
| Perf + LLM cost awareness | latency log, tokens per query, cache hit rate | 6 | numbers in README |
| README, Eval Report, AI Usage Report | docs | 6 | all three exist, Georgian |
| Interview readiness | `docs/explain/*`, pipeline walkthrough, break-fix drills | 7 | owner passes drills without AI |

Bonus (only after all above is green, in this order): LLM/embedding comparison (cheap, we have the interface) → prompt-injection test set → caching metrics → streaming answers → monitoring dashboard → load test.

## 5. Data plan

1. `scripts/build_subset.py`: from HotpotQA `distractor` validation (7,405 questions, 10 paragraphs each), take all context paragraphs of N questions chosen with a fixed seed. Deduplicate by normalized text hash. Count unique records. If < 50,000, add train questions (same seed) until ≥ 50,000 unique. Write `data/manifest.json` (question IDs + counts) — commit the manifest, not the data.
2. `scripts/fetch_postmortems.py`: parse the README entries (company, URL, summary), fetch each article politely (rate limit, user-agent, cache to disk), save HTML or PDF. Failures are logged and skipped, never invented. Commit URL list only.
3. `scripts/make_synthetic.py`: (a) conflict docs: copy a real paragraph and change one key fact (date/number/name), mark `synthetic=true`, `conflict_with=<doc_id>`; (b) unanswerable questions: remove the gold paragraphs of chosen questions from the index and list them in `eval/holdout.json`; (c) a few prompt-injection docs for the bonus test. Synthetic docs are visibly labelled in metadata.
4. Eval questions are drawn from questions whose paragraphs are in the corpus.

## 6. Retrieval design

- Dense: cosine similarity over pgvector, top-k each.
- Sparse: own BM25 (k1, b configurable), tokenization = lowercase, alnum split, light stopword list; English only.
- Fusion: own RRF, `score(d) = Σ 1/(k + rank_i(d))`, k configurable (start 60).
- Reranker (own, "coverage-aware"): after fusion, greedily pick results that maximize relevance + new coverage of the sub-queries (and entities) not yet covered, with a penalty for near-duplicates of already-picked chunks (MMR-style). This directly serves multi-hop questions, where the second hop is often low-scoring for the original query.
- All weights/params live in config files, so eval configs are just config files.

## 7. RAG design

1. Detect language; translate/rewrite to a clean English search query (Georgian users).
2. Decompose into ≤4 sub-queries (LLM, JSON output, validated).
3. Tool loop (bounded steps/tokens): LLM may call `search_documents`, `get_document`, `compare_claims`.
4. Merge → rerank → conflict check on candidate pairs (same `dup_group`, same title, or numeric/date mismatch) → structured LLM verdict.
5. Sufficiency gate: low retrieval confidence OR LLM says insufficient → abstain with what was found.
6. Generate answer in the user's language, only from evidence, with `[chunk_id]` citations.
7. Citation verifier: every cited ID must exist in retrieved set and pass a minimal support check; otherwise flag/remove.
8. Return answer + sources + trace (steps, queries used, scores) for the UI.

## 8. Eval plan

- ~60 questions in `eval/questions/*.jsonl`: ~12 simple, ~14 multi-hop bridge, ~8 comparison, ~8 unanswerable, ~8 conflicting, ~10 Georgian (cross-lingual), ~6 post-mortem questions (hand-labelled gold; e.g. "which outages in the list were caused by config changes and what services failed?").
- Split: DEV (tune) / TEST (report). Never tune on TEST.
- Metrics: Recall@5, Recall@10, MRR (gold supporting paragraphs); answer correctness (exact/F1 for HotpotQA-style, LLM judge for free-form, judge ≠ generator model when possible, judge bias noted); citation accuracy (precision and recall vs gold supporting facts); abstention accuracy (should-abstain vs should-answer); conflict-detection precision/recall.
- Configs: C1 dense-only; C2 hybrid RRF; C3 hybrid + coverage reranker; C4 = C3 + decomposition/tool loop. Chunking compared on slice.
- Report: tables generated by script, per-question-type breakdown, error taxonomy (retrieval miss / second-hop miss / wrong abstain / missed conflict / citation error / Georgian translation error), real failing examples, concrete improvements.

## 9. Deployment plan (owner does infra)

- Dockerfile: slim Python base, non-root user, healthcheck, no secrets baked in, models/data not in image unless required (decide after RAM check).
- Host: Docker-capable free tier (Hugging Face Spaces Docker or Render are candidates; verify current free limits on their pricing/docs pages first). DB: Neon or Supabase free tier (verify storage cap against the measured DB size).
- Deploy an EMPTY skeleton with `/health` on Sunday to surface hosting problems early.
- Document in `docs/DEPLOYMENT.md`: architecture, env vars, cold-start/sleep behaviour, rate limits, storage numbers, cost = $0, restart-persistence test.

## 10. Schedule (adjust to progress)

| Day | Goal |
|---|---|
| Sat 10-24 | Repo + CLAUDE.md + docs skeleton; Phase 0 spikes started (embedding model, LLM, DB size) |
| Sun 10-25 | Finish Phase 0; Phase 1 ingestion; start bulk embedding job; deploy skeleton |
| Mon 10-26 | Phase 2 retrieval (BM25/RRF/rerank + explain docs); start Phase 3 |
| Tue 10-27 | Phase 3 RAG pipeline; Phase 4 API+UI+auth; write eval questions (owner) in parallel; run retrieval-only evals |
| Wed 10-28 | Phase 5 full eval (cached LLM calls), error analysis; deploy final |
| Thu 10-29 | Phase 6–7: README, Eval Report, AI Usage Report, interview drills, buffer |

## 11. Top risks

1. Free LLM quota too small for eval → cache, small judge prompts, retrieval metrics first, spread over days.
2. Embedding model needs more RAM than the free host → quantized/smaller model or hosted embeddings for both sides.
3. DB storage cap smaller than corpus+index → reduce to just over 50k unique records, smaller dims, postings pruning.
4. Georgian quality of free LLM/embedding model → test in Phase 0 with 20 Georgian questions before committing.
5. Time → bonus items are cut first; core requirements and honest evaluation come first.
6. Interview → every owner-defended module has an explain doc and drills.
