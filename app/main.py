import asyncio
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.api.routes.ask import router as ask_router
from app.api.routes.documents import router as documents_router
from app.api.routes.health import router as health_router
from app.core.db import get_db
from app.core.logging import configure_logging, get_logger
from app.core.migrations import apply_migrations
from app.core.settings import get_settings

_log = get_logger("startup")


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    configure_logging(settings.log_level, settings.environment)

    # Run migrations with retries — Render containers sometimes need a moment
    # to establish network connectivity before the DB becomes reachable.
    last_exc: Exception | None = None
    for attempt in range(5):
        try:
            with get_db() as conn:
                apply_migrations(conn)
            break
        except Exception as exc:
            last_exc = exc
            _log.warning("migration_retry", attempt=attempt + 1, error=str(exc))
            await asyncio.sleep(2.0 * (attempt + 1))
    else:
        _log.error("migration_failed", error=str(last_exc))
        raise RuntimeError(f"DB migration failed after retries: {last_exc}") from last_exc

    # Warmup skipped — embedding backend may be remote (HF API) or local.
    # First query pays the cold-start cost either way.
    yield


app = FastAPI(
    title="AI Investigation Engine",
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(health_router)
app.include_router(ask_router)
app.include_router(documents_router)

# Serve the Georgian UI from app/ui/
_ui_path = Path(__file__).parent / "ui"
if _ui_path.exists():
    app.mount("/", StaticFiles(directory=str(_ui_path), html=True), name="ui")
