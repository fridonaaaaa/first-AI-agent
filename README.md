# AI საგამოძიებო სისტემა

მრავალსაფეხუროვანი RAG სისტემა, რომელიც პასუხობს კომპლექსურ კითხვებს 50 000+ ტექსტური ჩანაწერის მონაცემთა ბაზაზე. სისტემა ადასტურებს პასუხებს კონკრეტული წყაროებით, აღმოაჩენს წინააღმდეგობებს და უარს ამბობს, თუ მტკიცებულება არ არის საკმარისი.

**Upgaming Talent Growth Program — 2026 წ.**

---

## რა შეუძლია სისტემას

- კითხვებს პასუხობს ქართულ და ინგლისურ ენებზე
- პასუხებს ადასტურებს კონკრეტული წყაროებით (`[chunk_id]` ციტატები)
- აღმოაჩენს წინააღმდეგობებს სხვადასხვა წყაროს შორის
- უარს ამბობს პასუხზე, თუ მტკიცებულება არ არის საკმარისი
- ამუშავებს მრავალსაფეხუროვან კითხვებს (multi-hop)

---

## არქიტექტურა

```
ბრაუზერი (ქართული UI)
      │
FastAPI (Docker)
      │
  ┌───┴──────────────────────────────────────────┐
  │ RAG pipeline                                 │
  │  1. კითხვის გაგება  → ქართ→ინგლ, დეკომპოზ.  │
  │  2. Dense ძიება     → pgvector cosine        │
  │  3. BM25 ძიება      → საკუთარი Okapi BM25    │
  │  4. RRF შერწყმა     → საკუთარი RRF           │
  │  5. გადალაგება      → coverage-aware reranker│
  │  6. კონფლიქტი       → LLM სტრუქტ. შეფასება  │
  │  7. გენერაცია       → მხოლოდ მოძიებ. ტექსტი │
  │  8. ციტ. შემოწმება  → ჰალუცინ. ID-ების წაშლა│
  └──────────────────────────────────────────────┘
      │
PostgreSQL + pgvector (Supabase)
```

---

## გაშვება

**1. კლონირება**
```bash
git clone <repo-url>
cd "first AI agent"
cp .env.example .env
# შეავსეთ .env: DATABASE_URL, ADMIN_TOKEN, GEMINI_API_KEY
```

**2. სქემა (Supabase SQL Editor-ში ან psql-ით)**
```bash
psql $DATABASE_URL -f scripts/db/schema.sql
```

**3. ინსტალაცია**
```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

**4. მონაცემების ჩატვირთვა**
```bash
python scripts/load_hotpotqa.py
python scripts/embed_corpus.py
```

**5. სერვერი**
```bash
uvicorn app.main:app --reload
# http://localhost:8000
```

**Docker-ით:**
```bash
docker compose up --build
```

---

## ტესტები

```bash
pytest                          # 112 unit test — ყველა გადის
pytest tests/test_registry.py   # ინტეგრ. (DB საჭიროა)
```

---

## შეფასება

```bash
python scripts/eval_retrieval.py --split dev --config eval/configs/c2_hybrid_rrf.json
python scripts/eval_rag.py       --split dev --config eval/configs/c4_full_pipeline.json
python scripts/make_report.py    --split dev
```

| კონფ. | სახელი | აღწერა |
|-------|--------|---------|
| C1 | dense_only | მხოლოდ dense ძიება |
| C2 | hybrid_rrf | Dense + BM25 + RRF |
| C3 | hybrid_rerank | C2 + coverage reranker |
| C4 | full_pipeline | C3 + კითხვის დეკომპოზიცია |

---

## API

| Endpoint | მეთოდი | წვდომა |
|----------|--------|--------|
| `/health` | GET | ღია |
| `/ask` | POST | ღია, rate-limited |
| `/documents` | GET | ღია |
| `/documents` | POST/PUT/DELETE | Admin token |

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "ვინ გამოიგონა ტელეფონი?"}'
```

---

## გარემოს ცვლადები

იხ. `.env.example`. მთავარი: `DATABASE_URL`, `ADMIN_TOKEN`, `LLM_PROVIDER`, `GEMINI_API_KEY`.

---

## AI-ის გამოყენება

`docs/AI_USAGE.md` — კოდის რომელი ნაწილები გენერირდა Claude Code-ის მიერ.

## ლიცენზია

HotpotQA — CC BY-SA 4.0. სასწავლო პროექტი.
