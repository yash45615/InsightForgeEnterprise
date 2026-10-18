from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from insightforge.analytics.service import (
    get_campaign_performance,
    get_channel_performance,
    get_overview,
    get_sentiment_distribution,
)
from insightforge.database.session import get_db

router = APIRouter(prefix="/analytics", tags=["Analytics"])


@router.get("/overview")
def analytics_overview(
    db: Session = Depends(get_db),  # noqa: B008
) -> dict[str, int | float]:
    return get_overview(db)


@router.get("/channels")
def channel_performance(
    db: Session = Depends(get_db),  # noqa: B008
) -> list[dict[str, object]]:
    return get_channel_performance(db)


@router.get("/sentiment")
def sentiment_distribution(
    db: Session = Depends(get_db),  # noqa: B008
) -> list[dict[str, object]]:
    return get_sentiment_distribution(db)


@router.get("/campaigns")
def campaign_performance(
    db: Session = Depends(get_db),  # noqa: B008
) -> list[dict[str, object]]:
    return get_campaign_performance(db)
