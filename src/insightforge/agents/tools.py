from __future__ import annotations

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from insightforge.database.models import (
    Campaign,
    Customer,
    Feedback,
    Transaction,
)


def get_business_overview(db: Session) -> dict[str, int | float]:
    """Return a small, allow-listed set of business metrics."""
    customer_count = db.scalar(select(func.count()).select_from(Customer)) or 0
    transaction_count = db.scalar(select(func.count()).select_from(Transaction)) or 0
    feedback_count = db.scalar(select(func.count()).select_from(Feedback)) or 0

    completed_revenue = (
        db.scalar(
            select(func.coalesce(func.sum(Transaction.amount), 0)).where(
                Transaction.status == "completed"
            )
        )
        or 0
    )

    return {
        "customers": int(customer_count),
        "transactions": int(transaction_count),
        "feedback_records": int(feedback_count),
        "completed_revenue": round(float(completed_revenue), 2),
    }


def get_campaign_metrics(db: Session) -> list[dict[str, object]]:
    """Return campaign funnel metrics without accepting arbitrary SQL."""
    campaigns = db.scalars(select(Campaign).order_by(Campaign.campaign_id)).all()

    results: list[dict[str, object]] = []

    for campaign in campaigns:
        impressions = int(campaign.impressions)
        clicks = int(campaign.clicks)
        conversions = int(campaign.conversions)
        spend = float(campaign.spend)

        results.append(
            {
                "campaign_id": int(campaign.campaign_id),
                "campaign_name": campaign.campaign_name,
                "channel": campaign.channel,
                "impressions": impressions,
                "clicks": clicks,
                "conversions": conversions,
                "spend": round(spend, 2),
                "ctr": round(clicks / impressions, 4) if impressions > 0 else None,
                "conversion_rate": round(conversions / clicks, 4)
                if clicks > 0
                else None,
                "cost_per_conversion": round(spend / conversions, 2)
                if conversions > 0
                else None,
            }
        )

    return results


def get_feedback_summary(db: Session) -> list[dict[str, object]]:
    """Summarize feedback using database aggregates."""
    statement = (
        select(
            Feedback.sentiment_label,
            func.count(Feedback.feedback_id).label("record_count"),
            func.avg(Feedback.rating).label("average_rating"),
        )
        .group_by(Feedback.sentiment_label)
        .order_by(Feedback.sentiment_label)
    )

    rows = db.execute(statement).all()

    return [
        {
            "sentiment": row.sentiment_label,
            "count": int(row.record_count),
            "average_rating": (
                round(float(row.average_rating), 2)
                if row.average_rating is not None
                else None
            ),
        }
        for row in rows
    ]
