FROM python:3.12-slim

RUN useradd --create-home appuser
WORKDIR /home/appuser/app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq-dev gcc curl \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml .
# Install only production deps (no pytest/httpx) to keep image lean
RUN pip install --no-cache-dir "."

# Pre-download the embedding model into the image so the first request is fast.
# Without this, Render would download ~470 MB at cold-start, causing a timeout.
ENV SENTENCE_TRANSFORMERS_HOME=/home/appuser/.cache/st
RUN python -c "\
from sentence_transformers import SentenceTransformer; \
SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')"
RUN chown -R appuser:appuser /home/appuser/.cache

COPY --chown=appuser:appuser . .

USER appuser

EXPOSE 8000

# Render's health check hits /health; allow 60 s for model warm-up on cold start.
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
