"""
Admin token authentication.

The ADMIN_TOKEN env var protects upload, update, and delete endpoints.
Comparison uses hmac.compare_digest to prevent timing attacks.
"""

from __future__ import annotations

import hmac

from fastapi import Header, HTTPException, status

from app.core.settings import get_settings


def require_admin(x_admin_token: str = Header(...)) -> None:
    """
    FastAPI dependency — raises 401 if the token is wrong or missing.

    Usage:
        @router.post("/documents", dependencies=[Depends(require_admin)])
    """
    settings = get_settings()
    if not hmac.compare_digest(x_admin_token, settings.admin_token):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing admin token.",
        )
