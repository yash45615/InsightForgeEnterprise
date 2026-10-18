from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd
from sqlalchemy import delete

from insightforge.database.models import (
    Base,
    Campaign,
    Customer,
    Feedback,
    Transaction,
)
from insightforge.database.session import SessionLocal, engine

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"


def load_csv(filename: str) -> pd.DataFrame:
    path = RAW_DIR / filename

    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run scripts/generate_data.py first.")

    return pd.read_csv(path)


def validate_data(
    customers: pd.DataFrame,
    campaigns: pd.DataFrame,
    transactions: pd.DataFrame,
    feedback: pd.DataFrame,
) -> None:
    datasets = {
        "customers": (customers, "customer_id"),
        "campaigns": (campaigns, "campaign_id"),
        "transactions": (transactions, "transaction_id"),
        "feedback": (feedback, "feedback_id"),
    }

    for name, (df, primary_key) in datasets.items():
        if df.empty:
            raise ValueError(f"{name} dataset is empty.")

        if df[primary_key].isna().any():
            raise ValueError(f"{name} has missing primary keys.")

        if df[primary_key].duplicated().any():
            raise ValueError(f"{name} has duplicate primary keys.")

        allowed_missing = {"campaign_id"} if name == "transactions" else set()

        required_columns = [
            column for column in df.columns if column not in allowed_missing
        ]

        allowed_missing = {"campaign_id"} if name == "transactions" else set()

        required_columns = [
            column for column in df.columns if column not in allowed_missing
        ]

        if df[required_columns].isna().any().any():
            allowed_missing = {"campaign_id"} if name == "transactions" else set()

            required_columns = [
                column for column in df.columns if column not in allowed_missing
            ]

            if df[required_columns].isna().any().any():
                raise ValueError(f"{name} contains missing required values.")

    if customers["email"].duplicated().any():
        raise ValueError("Customer emails must be unique.")

    customer_ids = set(customers["customer_id"])
    campaign_ids = set(campaigns["campaign_id"])

    if not transactions["customer_id"].isin(customer_ids).all():
        raise ValueError("Transactions contain unknown customer IDs.")

    if not feedback["customer_id"].isin(customer_ids).all():
        raise ValueError("Feedback contains unknown customer IDs.")

    linked_campaigns = transactions["campaign_id"].dropna()

    if not linked_campaigns.isin(campaign_ids).all():
        raise ValueError("Transactions contain unknown campaign IDs.")

    if not (
        (campaigns["conversions"] <= campaigns["clicks"])
        & (campaigns["clicks"] <= campaigns["impressions"])
        & (campaigns["spend"] >= 0)
    ).all():
        raise ValueError("Campaign funnel or spend validation failed.")

    if not transactions["status"].isin(["completed", "refunded", "cancelled"]).all():
        raise ValueError("Invalid transaction status.")

    if not feedback["rating"].between(1, 5).all():
        raise ValueError("Feedback ratings must be between 1 and 5.")

    print("Data validation passed.")


def parse_datetime(value: object) -> datetime:
    return datetime.fromisoformat(str(value))


def main() -> None:
    customers_df = load_csv("customers.csv")
    campaigns_df = load_csv("campaigns.csv")
    transactions_df = load_csv("transactions.csv")
    feedback_df = load_csv("feedback.csv")

    validate_data(
        customers_df,
        campaigns_df,
        transactions_df,
        feedback_df,
    )

    Base.metadata.create_all(bind=engine)

    with SessionLocal() as session:
        try:
            # Delete in child-to-parent order for repeatable ingestion.
            session.execute(delete(Feedback))
            session.execute(delete(Transaction))
            session.execute(delete(Customer))
            session.execute(delete(Campaign))
            session.flush()

            campaigns = [
                Campaign(
                    campaign_id=int(r.campaign_id),
                    campaign_name=str(r.campaign_name),
                    channel=str(r.channel),
                    start_date=parse_datetime(r.start_date),
                    spend=float(r.spend),
                    impressions=int(r.impressions),
                    clicks=int(r.clicks),
                    conversions=int(r.conversions),
                )
                for r in campaigns_df.itertuples(index=False)
            ]

            customers = [
                Customer(
                    customer_id=int(r.customer_id),
                    customer_name=str(r.customer_name),
                    email=str(r.email),
                    signup_date=parse_datetime(r.signup_date),
                    acquisition_channel=str(r.acquisition_channel),
                    segment=str(r.segment),
                    region=str(r.region),
                    monthly_fee=float(r.monthly_fee),
                    is_active=str(r.is_active).lower() == "true",
                )
                for r in customers_df.itertuples(index=False)
            ]

            transactions = [
                Transaction(
                    transaction_id=int(r.transaction_id),
                    customer_id=int(r.customer_id),
                    campaign_id=(
                        int(r.campaign_id) if pd.notna(r.campaign_id) else None
                    ),
                    transaction_date=parse_datetime(r.transaction_date),
                    product=str(r.product),
                    amount=float(r.amount),
                    status=str(r.status),
                )
                for r in transactions_df.itertuples(index=False)
            ]

            feedback = [
                Feedback(
                    feedback_id=int(r.feedback_id),
                    customer_id=int(r.customer_id),
                    feedback_date=parse_datetime(r.feedback_date),
                    category=str(r.category),
                    sentiment_label=str(r.sentiment_label),
                    comment=str(r.comment),
                    rating=int(r.rating),
                )
                for r in feedback_df.itertuples(index=False)
            ]

            session.add_all(campaigns)
            session.add_all(customers)
            session.flush()

            session.add_all(transactions)
            session.add_all(feedback)
            session.commit()

            print("Database ingestion completed successfully.")
            print(f"Customers: {len(customers)}")
            print(f"Campaigns: {len(campaigns)}")
            print(f"Transactions: {len(transactions)}")
            print(f"Feedback records: {len(feedback)}")

        except Exception:
            session.rollback()
            raise


if __name__ == "__main__":
    main()
