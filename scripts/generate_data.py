from __future__ import annotations

import random
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
from faker import Faker

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"

SEED = 42
CUSTOMER_COUNT = 500
TRANSACTION_COUNT = 3000
FEEDBACK_COUNT = 1000

fake = Faker("en_US")
Faker.seed(SEED)
random.seed(SEED)

CHANNELS = ["Organic", "Paid Search", "Social", "Referral", "Email"]
SEGMENTS = ["Standard", "Premium", "Enterprise"]
PRODUCTS = ["Analytics", "Storage", "Automation", "Security", "Support"]

POSITIVE_COMMENTS = [
    "The product is easy to use and saved our team time.",
    "Support responded quickly and solved my problem.",
    "The onboarding experience was smooth.",
    "The latest update improved our workflow.",
    "Good value for the price and reliable service.",
]

NEGATIVE_COMMENTS = [
    "The application is slow during busy periods.",
    "I had trouble completing the onboarding process.",
    "The billing information was confusing.",
    "Customer support took too long to respond.",
    "The latest update introduced frustrating errors.",
]

NEUTRAL_COMMENTS = [
    "I use the product several times each week.",
    "The dashboard contains the information I need.",
    "I would like more reporting options.",
    "The product has the features described in the plan.",
    "I am evaluating this product for my team.",
]


def timestamp_between(start: datetime, end: datetime) -> datetime:
    seconds = int((end - start).total_seconds())
    return start + timedelta(seconds=random.randint(0, seconds))


def generate_customers() -> pd.DataFrame:
    now = datetime(2026, 10, 1)
    start = datetime(2024, 1, 1)
    records = []

    for customer_id in range(1, CUSTOMER_COUNT + 1):
        signup = timestamp_between(start, now)

        records.append(
            {
                "customer_id": customer_id,
                "customer_name": fake.name(),
                "email": f"customer{customer_id}@example.com",
                "signup_date": signup,
                "acquisition_channel": random.choice(CHANNELS),
                "segment": random.choices(SEGMENTS, weights=[0.65, 0.28, 0.07])[0],
                "region": random.choice(["North", "South", "East", "West", "Central"]),
                "monthly_fee": round(random.uniform(20, 250), 2),
                "is_active": random.random() > 0.15,
            }
        )

    return pd.DataFrame(records)


def generate_campaigns() -> pd.DataFrame:
    records = []
    start = datetime(2025, 1, 1)

    for campaign_id in range(1, 13):
        campaign_start = start + timedelta(days=(campaign_id - 1) * 35)
        spend = round(random.uniform(5000, 30000), 2)
        impressions = random.randint(20000, 150000)
        clicks = random.randint(
            max(1, int(impressions * 0.01)),
            max(2, int(impressions * 0.12)),
        )
        conversions = random.randint(10, max(11, int(clicks * 0.20)))

        records.append(
            {
                "campaign_id": campaign_id,
                "campaign_name": f"Campaign {campaign_id:02d}",
                "channel": random.choice(CHANNELS),
                "start_date": campaign_start,
                "spend": spend,
                "impressions": impressions,
                "clicks": clicks,
                "conversions": conversions,
            }
        )

    return pd.DataFrame(records)


def generate_transactions(customers: pd.DataFrame) -> pd.DataFrame:
    records = []
    start = datetime(2025, 1, 1)
    end = datetime(2026, 9, 30)

    for transaction_id in range(1, TRANSACTION_COUNT + 1):
        customer = customers.sample(n=1, random_state=SEED + transaction_id).iloc[0]

        records.append(
            {
                "transaction_id": transaction_id,
                "customer_id": int(customer["customer_id"]),
                "campaign_id": random.choice([None, None, None, *range(1, 13)]),
                "transaction_date": timestamp_between(start, end),
                "product": random.choice(PRODUCTS),
                "amount": round(random.uniform(15, 1200), 2),
                "status": random.choices(
                    ["completed", "refunded", "cancelled"],
                    weights=[0.90, 0.06, 0.04],
                )[0],
            }
        )

    return pd.DataFrame(records)


def generate_feedback(customers: pd.DataFrame) -> pd.DataFrame:
    records = []
    categories = ["product", "billing", "support", "onboarding"]

    for feedback_id in range(1, FEEDBACK_COUNT + 1):
        customer = customers.sample(n=1, random_state=SEED * 10 + feedback_id).iloc[0]

        sentiment = random.choices(
            ["positive", "neutral", "negative"],
            weights=[0.45, 0.25, 0.30],
        )[0]

        comment_pool = {
            "positive": POSITIVE_COMMENTS,
            "neutral": NEUTRAL_COMMENTS,
            "negative": NEGATIVE_COMMENTS,
        }

        records.append(
            {
                "feedback_id": feedback_id,
                "customer_id": int(customer["customer_id"]),
                "feedback_date": timestamp_between(
                    datetime(2025, 1, 1),
                    datetime(2026, 9, 30),
                ),
                "category": random.choice(categories),
                "sentiment_label": sentiment,
                "comment": random.choice(comment_pool[sentiment]),
                "rating": {
                    "positive": random.randint(4, 5),
                    "neutral": 3,
                    "negative": random.randint(1, 2),
                }[sentiment],
            }
        )

    return pd.DataFrame(records)


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    customers = generate_customers()
    campaigns = generate_campaigns()
    transactions = generate_transactions(customers)
    feedback = generate_feedback(customers)

    datasets = {
        "customers.csv": customers,
        "campaigns.csv": campaigns,
        "transactions.csv": transactions,
        "feedback.csv": feedback,
    }

    for filename, dataframe in datasets.items():
        path = RAW_DIR / filename
        dataframe.to_csv(path, index=False)
        print(f"Created {path.relative_to(ROOT)}: {len(dataframe)} rows")

    print("\nSynthetic dataset generation complete.")


if __name__ == "__main__":
    main()
