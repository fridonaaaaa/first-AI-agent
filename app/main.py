from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.api.routes.ask import router as ask_router
from app.api.routes.documents import router as documents_router
from app.api.routes.health import router as health_router
from app.core.db import get_db
from app.core.logging import configure_logging
from app.core.migrations import apply_migrations
from app.core.settings import get_settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    configure_logging(settings.log_level, settings.environment)
    with get_db() as conn:
        apply_migrations(conn)
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
