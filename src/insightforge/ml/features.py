from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
RAW_DIR = ROOT / "data" / "raw"


def build_customer_features() -> pd.DataFrame:
    customers = pd.read_csv(RAW_DIR / "customers.csv")
    transactions = pd.read_csv(RAW_DIR / "transactions.csv")

    customers["signup_date"] = pd.to_datetime(customers["signup_date"], errors="raise")
    transactions["transaction_date"] = pd.to_datetime(
        transactions["transaction_date"], errors="raise"
    )

    transactions["amount"] = pd.to_numeric(transactions["amount"], errors="raise")
    transactions["status"] = transactions["status"].astype(str).str.strip().str.lower()

    completed = transactions.loc[transactions["status"] == "completed"].copy()

    all_transaction_features = (
        transactions.groupby("customer_id")
        .agg(
            transaction_count=("transaction_id", "count"),
            average_transaction_amount=("amount", "mean"),
        )
        .reset_index()
    )

    completed_features = (
        completed.groupby("customer_id")
        .agg(
            completed_order_count=("transaction_id", "count"),
            completed_revenue=("amount", "sum"),
        )
        .reset_index()
    )

    if not transactions.empty:
        reference_date = transactions["transaction_date"].max()
        latest_transaction = (
            transactions.groupby("customer_id")["transaction_date"]
            .max()
            .rename("last_transaction_date")
            .reset_index()
        )
    else:
        reference_date = customers["signup_date"].max()
        latest_transaction = pd.DataFrame(
            columns=["customer_id", "last_transaction_date"]
        )

    features = customers.merge(all_transaction_features, on="customer_id", how="left")
    features = features.merge(completed_features, on="customer_id", how="left")
    features = features.merge(latest_transaction, on="customer_id", how="left")

    count_columns = ["transaction_count", "completed_order_count"]
    revenue_columns = [
        "average_transaction_amount",
        "completed_revenue",
    ]

    features[count_columns] = features[count_columns].fillna(0)
    features[revenue_columns] = features[revenue_columns].fillna(0)

    features["days_since_last_transaction"] = (
        reference_date - features["last_transaction_date"]
    ).dt.days
    features["days_since_last_transaction"] = features[
        "days_since_last_transaction"
    ].fillna((reference_date - features["signup_date"]).dt.days)

    features["is_active_numeric"] = (
        features["is_active"]
        .astype(str)
        .str.lower()
        .map({"true": 1, "false": 0, "1": 1, "0": 0})
    )

    if features["is_active_numeric"].isna().any():
        raise ValueError("Unexpected values found in is_active.")

    features["monthly_fee"] = pd.to_numeric(features["monthly_fee"], errors="raise")

    return features.sort_values("customer_id").reset_index(drop=True)
