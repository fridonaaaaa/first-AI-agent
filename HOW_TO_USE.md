# How to Use — AI Investigation Engine

**Live site:** https://first-ai-agent-rq2b.onrender.com/

---

## Asking Questions

1. Open the live site in your browser.
2. Type your question in the chat box (Georgian or English both work).
3. Press **Send**.
4. The system searches 50,000+ documents, cites its sources, and flags contradictions if it finds them.
5. If there is not enough evidence to answer, it will say so — it does not guess.

---

## Uploading Documents (Admin)

To upload new documents to the knowledge base you need the admin password.

**Password: `1234`**

### Steps

1. Click the **Upload** button on the site.
2. Enter the password `1234` when prompted.
3. Select a `.txt` or `.pdf` file (max 10 MB).
4. Click **Upload**. The document is processed and becomes searchable within seconds.

### Deleting Documents

1. Find the document in the document list on the site.
2. Click **Delete**.
3. Enter the password `1234` when prompted.

---

## Tips

- Ask multi-part questions — the engine breaks them into sub-queries automatically.
- If the answer cites `[chunk_id]` references, those point to the exact passages used.
- If two sources disagree, the answer will show both sides instead of picking one.
- The site may take 30–60 seconds to wake up if it has been idle (Render free tier sleeps).

---

## API (for developers)

```bash
# Ask a question
curl -X POST https://first-ai-agent-rq2b.onrender.com/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "Who invented the telephone?"}'

# Upload a document
curl -X POST https://first-ai-agent-rq2b.onrender.com/documents \
  -H "X-Admin-Token: 1234" \
  -F "file=@yourfile.txt"

# List documents
curl https://first-ai-agent-rq2b.onrender.com/documents

# Health check
curl https://first-ai-agent-rq2b.onrender.com/health
```

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend | FastAPI (Python) |
| Database | PostgreSQL + pgvector (Supabase) |
| Embeddings | sentence-transformers (multilingual) |
| Search | Hybrid BM25 + dense vector + RRF |
| LLM | Gemini (free tier) |
| Hosting | Render (free tier) |
