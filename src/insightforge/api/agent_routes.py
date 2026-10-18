from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from insightforge.agents.service import InvestigationService
from insightforge.database.session import get_db

router = APIRouter(prefix="/agent", tags=["Agent"])
service = InvestigationService()


class InvestigationRequest(BaseModel):
    topic: str = Field(min_length=1, max_length=50)


@router.post("/investigate")
def investigate(
    request: InvestigationRequest,
    db: Session = Depends(get_db),  # noqa: B008
) -> dict[str, object]:
    try:
        return service.investigate(request.topic, db)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
