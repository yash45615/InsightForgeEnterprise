from __future__ import annotations

from fastapi import FastAPI

from insightforge.api.agent_routes import router as agent_router
from insightforge.api.qa_routes import router as qa_router
from insightforge.api.routes import router as api_router
from insightforge.config import settings

app = FastAPI(
    title=settings.app_name,
)


@app.get("/")
def root() -> dict[str, str]:
    """Return basic API information."""
    return {
        "application": "InsightForge Enterprise",
        "status": "running",
    }


@app.get("/health")
def health() -> dict[str, str]:
    """Return the application health status."""
    return {"status": "healthy"}


app.include_router(api_router)
app.include_router(agent_router)
app.include_router(qa_router)
