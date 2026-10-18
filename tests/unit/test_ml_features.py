import numpy as np

from insightforge.ml.features import build_customer_features


def test_customer_features_have_expected_rows():
    features = build_customer_features()
    assert len(features) == 500
    assert features["customer_id"].is_unique


def test_customer_features_have_required_columns():
    features = build_customer_features()

    required = {
        "customer_id",
        "transaction_count",
        "completed_order_count",
        "completed_revenue",
        "average_transaction_amount",
        "days_since_last_transaction",
        "is_active_numeric",
    }

    assert required.issubset(features.columns)


def test_customer_numeric_features_are_finite():
    features = build_customer_features()

    numeric = features[
        [
            "monthly_fee",
            "transaction_count",
            "completed_order_count",
            "completed_revenue",
            "average_transaction_amount",
            "days_since_last_transaction",
            "is_active_numeric",
        ]
    ]

    assert np.isfinite(numeric.to_numpy(dtype=float)).all()


def test_transaction_counts_and_revenue_are_nonnegative():
    features = build_customer_features()

    assert (features["transaction_count"] >= 0).all()
    assert (features["completed_order_count"] >= 0).all()
    assert (features["completed_revenue"] >= 0).all()
