FROM python:3.12-slim

RUN useradd --create-home appuser
WORKDIR /home/appuser/app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq-dev gcc curl libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml .
# Install only production deps (no pytest/httpx) to keep image lean
RUN pip install --no-cache-dir "."

COPY --chown=appuser:appuser . .

USER appuser

EXPOSE 7860

# HF Spaces hits /health to verify the container is ready.
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
