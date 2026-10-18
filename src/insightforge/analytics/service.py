from sqlalchemy import case, func, select
from sqlalchemy.orm import Session

from insightforge.database.models import (
    Campaign,
    Customer,
    Feedback,
    Transaction,
)


def get_overview(db: Session) -> dict[str, int | float]:
    customer_count = db.scalar(select(func.count()).select_from(Customer)) or 0

    active_customers = (
        db.scalar(
            select(func.count())
            .select_from(Customer)
            .where(Customer.is_active.is_(True))
        )
        or 0
    )

    completed = db.execute(
        select(
            func.count(Transaction.transaction_id),
            func.coalesce(func.sum(Transaction.amount), 0.0),
            func.avg(Transaction.amount),
        ).where(Transaction.status == "completed")
    ).one()

    feedback_count = db.scalar(select(func.count()).select_from(Feedback)) or 0

    return {
        "total_customers": int(customer_count),
        "active_customers": int(active_customers),
        "completed_orders": int(completed[0] or 0),
        "completed_revenue": round(float(completed[1] or 0), 2),
        "average_order_value": round(float(completed[2] or 0), 2),
        "feedback_records": int(feedback_count),
    }


def get_channel_performance(db: Session) -> list[dict[str, object]]:
    statement = (
        select(
            Customer.acquisition_channel.label("channel"),
            func.count(func.distinct(Customer.customer_id)).label("customers"),
            func.count(Transaction.transaction_id).label("orders"),
            func.coalesce(
                func.sum(
                    case(
                        (
                            Transaction.status == "completed",
                            Transaction.amount,
                        ),
                        else_=0.0,
                    )
                ),
                0.0,
            ).label("revenue"),
        )
        .select_from(Customer)
        .outerjoin(
            Transaction,
            Transaction.customer_id == Customer.customer_id,
        )
        .group_by(Customer.acquisition_channel)
        .order_by(func.sum(Transaction.amount).desc())
    )

    rows = db.execute(statement).mappings().all()

    return [
        {
            "channel": row["channel"],
            "customers": int(row["customers"]),
            "orders": int(row["orders"]),
            "revenue": round(float(row["revenue"] or 0), 2),
        }
        for row in rows
    ]


def get_sentiment_distribution(
    db: Session,
) -> list[dict[str, object]]:
    statement = (
        select(
            Feedback.sentiment_label.label("sentiment"),
            func.count(Feedback.feedback_id).label("count"),
            func.avg(Feedback.rating).label("average_rating"),
        )
        .group_by(Feedback.sentiment_label)
        .order_by(Feedback.sentiment_label)
    )

    rows = db.execute(statement).mappings().all()

    return [
        {
            "sentiment": row["sentiment"],
            "count": int(row["count"]),
            "average_rating": round(float(row["average_rating"]), 2),
        }
        for row in rows
    ]


def get_campaign_performance(
    db: Session,
) -> list[dict[str, object]]:
    statement = select(
        Campaign.campaign_id,
        Campaign.campaign_name,
        Campaign.channel,
        Campaign.spend,
        Campaign.impressions,
        Campaign.clicks,
        Campaign.conversions,
    ).order_by(Campaign.campaign_id)

    rows = db.execute(statement).mappings().all()
    results = []

    for row in rows:
        clicks = int(row["clicks"])
        impressions = int(row["impressions"])
        conversions = int(row["conversions"])
        spend = float(row["spend"])

        results.append(
            {
                "campaign_id": row["campaign_id"],
                "campaign_name": row["campaign_name"],
                "channel": row["channel"],
                "spend": round(spend, 2),
                "impressions": impressions,
                "clicks": clicks,
                "conversions": conversions,
                "click_through_rate": round(
                    clicks / impressions if impressions else 0, 4
                ),
                "conversion_rate": round(conversions / clicks if clicks else 0, 4),
                "cost_per_conversion": round(
                    spend / conversions if conversions else 0, 2
                ),
            }
        )

    return results
