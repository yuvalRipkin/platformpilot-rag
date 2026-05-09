import logging

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/ready")
async def ready(
    request: Request, db: AsyncSession = Depends(get_db)
) -> JSONResponse:
    if getattr(request.app.state, "embedder", None) is None:
        return JSONResponse(
            status_code=503,
            content={"status": "not_ready", "reason": "embedder not loaded"},
        )
    try:
        await db.execute(text("SELECT 1"))
    except Exception:
        logger.exception("Readiness check failed")
        return JSONResponse(
            status_code=503,
            content={"status": "not_ready", "reason": "database unavailable"},
        )
    return JSONResponse(status_code=200, content={"status": "ready"})
