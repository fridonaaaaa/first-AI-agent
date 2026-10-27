"""Database connection management."""

import time
from contextlib import contextmanager
from typing import Generator

import psycopg2
import psycopg2.extras
import psycopg2.pool

from app.core.settings import get_settings

_pool: psycopg2.pool.ThreadedConnectionPool | None = None


def _new_conn():
    url = get_settings().database_url
    conn = psycopg2.connect(url)
    psycopg2.extras.register_uuid(conn)
    return conn


def _get_pool() -> psycopg2.pool.ThreadedConnectionPool:
    global _pool
    if _pool is None or _pool.closed:
        url = get_settings().database_url
        _pool = psycopg2.pool.ThreadedConnectionPool(1, 5, url)
    return _pool


def _connect(retries: int = 4, delay: float = 1.5):
    """Connect with retry — used only for long-running bulk jobs."""
    last_exc = None
    for attempt in range(retries):
        try:
            return _new_conn()
        except psycopg2.OperationalError as exc:
            last_exc = exc
            if attempt < retries - 1:
                time.sleep(delay)
    raise last_exc


@contextmanager
def get_db() -> Generator:
    """
    Borrow a connection from the pool, commit on success, rollback on error,
    return to pool when done. Replaces a broken connection transparently.
    """
    pool = _get_pool()
    conn = pool.getconn()

    # Verify the connection is alive; replace it if the server dropped it.
    try:
        conn.cursor().execute("SELECT 1")
    except Exception:
        try:
            pool.putconn(conn, close=True)
        except Exception:
            pass
        conn = _new_conn()

    # Register UUID type adapter (idempotent, fast).
    psycopg2.extras.register_uuid(conn)

    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        try:
            pool.putconn(conn)
        except Exception:
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
