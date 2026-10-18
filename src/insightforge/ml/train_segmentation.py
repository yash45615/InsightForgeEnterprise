from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from insightforge.ml.features import build_customer_features

ROOT = Path(__file__).resolve().parents[3]
ARTIFACT_DIR = ROOT / "artifacts" / "ml"
REPORT_DIR = ROOT / "reports" / "ml"

NUMERIC_FEATURES = [
    "monthly_fee",
    "transaction_count",
    "completed_order_count",
    "completed_revenue",
    "average_transaction_amount",
    "days_since_last_transaction",
    "is_active_numeric",
]


def main() -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    customers = build_customer_features()

    channel_features = pd.get_dummies(
        customers["acquisition_channel"].fillna("unknown"),
        prefix="channel",
        dtype=float,
    )

    numeric = customers[NUMERIC_FEATURES].astype(float)
    X = pd.concat([numeric, channel_features], axis=1)
    X = X.replace([float("inf"), float("-inf")], 0).fillna(0)

    if len(X) < 3:
        raise ValueError("At least three customers are required.")

    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "kmeans",
                KMeans(n_clusters=3, random_state=42, n_init=10),
            ),
        ]
    )

    labels = model.fit_predict(X)

    output = customers[["customer_id", "customer_name", "acquisition_channel"]].copy()
    output["cluster"] = labels

    profiles = (
        customers.assign(cluster=labels)
        .groupby("cluster")
        .agg(
            customer_count=("customer_id", "count"),
            avg_monthly_fee=("monthly_fee", "mean"),
            avg_completed_revenue=("completed_revenue", "mean"),
            avg_completed_orders=("completed_order_count", "mean"),
            avg_days_since_transaction=("days_since_last_transaction", "mean"),
        )
        .reset_index()
    )

    joblib.dump(model, ARTIFACT_DIR / "customer_segmentation.joblib")
    (ARTIFACT_DIR / "segmentation_features.json").write_text(
        json.dumps(list(X.columns), indent=2),
        encoding="utf-8",
    )
    output.to_csv(REPORT_DIR / "customer_segments.csv", index=False)
    profiles.to_csv(REPORT_DIR / "segment_profiles.csv", index=False)

    print(f"Customers segmented: {len(output)}")
    print(f"Clusters created: {output['cluster'].nunique()}")
    print("\nSegment profiles:")
    print(profiles.round(2).to_string(index=False))
    print(f"\nReports saved to: {REPORT_DIR}")


if __name__ == "__main__":
    main()
