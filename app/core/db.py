"""Database connection management."""

from contextlib import contextmanager
from typing import Generator

import psycopg2
import psycopg2.extras

from app.core.settings import get_settings


@contextmanager
def get_db() -> Generator:
    """
    Open a psycopg2 connection, commit on success, rollback on error, always close.
    Use for single-request operations.
    """
    conn = psycopg2.connect(get_settings().database_url)
    psycopg2.extras.register_uuid(conn)
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
    conn = psycopg2.connect(get_settings().database_url)
    psycopg2.extras.register_uuid(conn)
    return conn
