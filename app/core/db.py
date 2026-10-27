"""Database connection management."""

import time
from contextlib import contextmanager
from typing import Generator

import psycopg2
import psycopg2.extras

from app.core.settings import get_settings


def _connect(retries: int = 4, delay: float = 1.5):
    """Connect with retry — macOS DNS resolution occasionally fails transiently."""
    last_exc = None
    for attempt in range(retries):
        try:
            conn = psycopg2.connect(get_settings().database_url)
            psycopg2.extras.register_uuid(conn)
            return conn
        except psycopg2.OperationalError as exc:
            last_exc = exc
            if attempt < retries - 1:
                time.sleep(delay)
    raise last_exc


@contextmanager
def get_db() -> Generator:
    """
    Open a psycopg2 connection, commit on success, rollback on error, always close.
    Use for single-request operations.
    """
    conn = _connect()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def open_connection():
    """
    Return a raw connection for long-running jobs (bulk ingestion).
    Caller is responsible for commit/rollback/close.
    """
    conn = _connect()
    # Disable statement timeout for bulk-load jobs — Supabase free tier
    # sets a short default that fires on large BM25 stats upserts.
    conn.cursor().execute("SET statement_timeout = 0")
    conn.commit()
    return conn
