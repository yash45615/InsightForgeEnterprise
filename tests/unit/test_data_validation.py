import pandas as pd
import pytest
from scripts.ingest_data import validate_data


def make_datasets():
    customers = pd.DataFrame(
        [
            {
                "customer_id": 1,
                "customer_name": "Customer One",
                "email": "one@example.com",
                "signup_date": "2025-01-01",
                "acquisition_channel": "Organic",
                "segment": "Standard",
                "region": "North",
                "monthly_fee": 50.0,
                "is_active": True,
            }
        ]
    )

    campaigns = pd.DataFrame(
        [
            {
                "campaign_id": 1,
                "campaign_name": "Campaign One",
                "channel": "Email",
                "start_date": "2025-01-01",
                "spend": 100.0,
                "impressions": 1000,
                "clicks": 100,
                "conversions": 10,
            }
        ]
    )

    transactions = pd.DataFrame(
        [
            {
                "transaction_id": 1,
                "customer_id": 1,
                "campaign_id": 1,
                "transaction_date": "2025-02-01",
                "product": "Analytics",
                "amount": 100.0,
                "status": "completed",
            }
        ]
    )

    feedback = pd.DataFrame(
        [
            {
                "feedback_id": 1,
                "customer_id": 1,
                "feedback_date": "2025-02-01",
                "category": "product",
                "sentiment_label": "positive",
                "comment": "Very useful product.",
                "rating": 5,
            }
        ]
    )

    return customers, campaigns, transactions, feedback


def test_valid_datasets_pass_validation():
    validate_data(*make_datasets())


def test_unknown_customer_is_rejected():
    customers, campaigns, transactions, feedback = make_datasets()
    transactions.loc[0, "customer_id"] = 999

    with pytest.raises(ValueError, match="unknown customer"):
        validate_data(customers, campaigns, transactions, feedback)


def test_invalid_rating_is_rejected():
    customers, campaigns, transactions, feedback = make_datasets()
    feedback.loc[0, "rating"] = 8

    with pytest.raises(ValueError, match="ratings"):
        validate_data(customers, campaigns, transactions, feedback)


def test_duplicate_customer_id_is_rejected():
    customers, campaigns, transactions, feedback = make_datasets()
    customers = pd.concat([customers, customers], ignore_index=True)

    with pytest.raises(ValueError, match="duplicate primary keys"):
        validate_data(customers, campaigns, transactions, feedback)
