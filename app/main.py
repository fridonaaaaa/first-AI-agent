from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.routes.health import router as health_router
from app.core.logging import configure_logging
from app.core.settings import get_settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    configure_logging(settings.log_level, settings.environment)
    yield


app = FastAPI(
    title="AI Investigation Engine",
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(health_router)
