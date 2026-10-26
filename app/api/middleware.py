"""
Rate limiting middleware — per-IP per-minute + global daily cap.

Uses an in-memory store (dict of counters). This is sufficient for a single-
process Docker deployment. The counts reset on restart, which is acceptable
for the free-tier use case.

Per-IP limit: RATE_LIMIT_PER_IP_PER_MINUTE (env var, default 20).
Daily cap:    DAILY_CAP_ASK (env var, default 1000 total across all IPs).

Only /ask is rate-limited. Admin endpoints use token auth instead.
"""

from __future__ import annotations

import time
from collections import defaultdict

from fastapi import Request
from fastapi.responses import JSONResponse

from app.core.settings import get_settings

# {ip: (window_start_epoch, count)}
_ip_windows: dict[str, tuple[float, int]] = {}
_daily_count: dict[str, int] = defaultdict(int)  # {"YYYY-MM-DD": count}


def _today() -> str:
    import datetime
    return datetime.date.today().isoformat()


def check_rate_limit(request: Request) -> JSONResponse | None:
    """
    Return a 429 JSONResponse if any limit is exceeded, else None.
    Call this at the start of rate-limited endpoints.
    """
    settings = get_settings()
    ip = request.client.host if request.client else "unknown"
    now = time.monotonic()

    # Per-IP sliding window (1 minute)
    window_start, count = _ip_windows.get(ip, (now, 0))
    if now - window_start > 60:
        window_start, count = now, 0
    count += 1
    _ip_windows[ip] = (window_start, count)

    if count > settings.rate_limit_per_ip_per_minute:
        return JSONResponse(
            status_code=429,
            content={"error": "Too many requests. Please wait a minute."},
        )

    # Global daily cap
    today = _today()
    _daily_count[today] += 1
    if _daily_count[today] > settings.daily_cap_ask:
        return JSONResponse(
            status_code=429,
            content={"error": "Daily question limit reached. Try again tomorrow."},
        )

    return None
