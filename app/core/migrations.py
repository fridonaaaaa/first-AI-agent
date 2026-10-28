"""
Database migrations — idempotent schema updates applied at startup.

Each migration uses IF NOT EXISTS / IF EXISTS guards so it is safe to run
on every startup. The connection is committed by the caller (lifespan).
"""

from __future__ import annotations

import psycopg2

from app.core.logging import get_logger

log = get_logger("migrations")


def apply_migrations(conn: "psycopg2.extensions.connection") -> None:
    """Run all pending schema migrations against the live DB."""
    with conn.cursor() as cur:
        # 001: per-chat session isolation on documents
        cur.execute(
            "ALTER TABLE documents ADD COLUMN IF NOT EXISTS session_id text"
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_documents_session_id ON documents(session_id)"
        )
    log.info("migrations_applied")
