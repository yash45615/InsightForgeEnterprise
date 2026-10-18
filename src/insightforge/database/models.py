from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class Customer(Base):
    __tablename__ = "customers"

    customer_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    customer_name: Mapped[str] = mapped_column(String(150), nullable=False)
    email: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)
    signup_date: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    acquisition_channel: Mapped[str] = mapped_column(String(50), nullable=False)
    segment: Mapped[str] = mapped_column(String(30), nullable=False)
    region: Mapped[str] = mapped_column(String(30), nullable=False)
    monthly_fee: Mapped[float] = mapped_column(Float, nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False)

    transactions: Mapped[list[Transaction]] = relationship(back_populates="customer")
    feedback: Mapped[list[Feedback]] = relationship(back_populates="customer")

    __table_args__ = (
        CheckConstraint("monthly_fee >= 0", name="ck_customer_fee_nonnegative"),
        Index("ix_customers_channel", "acquisition_channel"),
        Index("ix_customers_segment", "segment"),
    )


class Campaign(Base):
    __tablename__ = "campaigns"

    campaign_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    campaign_name: Mapped[str] = mapped_column(String(100), nullable=False)
    channel: Mapped[str] = mapped_column(String(50), nullable=False)
    start_date: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    spend: Mapped[float] = mapped_column(Float, nullable=False)
    impressions: Mapped[int] = mapped_column(Integer, nullable=False)
    clicks: Mapped[int] = mapped_column(Integer, nullable=False)
    conversions: Mapped[int] = mapped_column(Integer, nullable=False)

    transactions: Mapped[list[Transaction]] = relationship(back_populates="campaign")

    __table_args__ = (
        CheckConstraint("spend >= 0", name="ck_campaign_spend_nonnegative"),
        CheckConstraint(
            "impressions >= clicks AND clicks >= conversions AND conversions >= 0",
            name="ck_campaign_funnel",
        ),
    )


class Transaction(Base):
    __tablename__ = "transactions"

    transaction_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    customer_id: Mapped[int] = mapped_column(
        ForeignKey("customers.customer_id"), nullable=False
    )
    campaign_id: Mapped[int | None] = mapped_column(
        ForeignKey("campaigns.campaign_id"), nullable=True
    )
    transaction_date: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    product: Mapped[str] = mapped_column(String(80), nullable=False)
    amount: Mapped[float] = mapped_column(Float, nullable=False)
    status: Mapped[str] = mapped_column(String(20), nullable=False)

    customer: Mapped[Customer] = relationship(back_populates="transactions")
    campaign: Mapped[Campaign | None] = relationship(back_populates="transactions")

    __table_args__ = (
        CheckConstraint("amount >= 0", name="ck_transaction_amount_nonnegative"),
        CheckConstraint(
            "status IN ('completed', 'refunded', 'cancelled')",
            name="ck_transaction_status",
        ),
        Index("ix_transactions_customer", "customer_id"),
        Index("ix_transactions_date", "transaction_date"),
    )


class Feedback(Base):
    __tablename__ = "feedback"

    feedback_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    customer_id: Mapped[int] = mapped_column(
        ForeignKey("customers.customer_id"), nullable=False
    )
    feedback_date: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    category: Mapped[str] = mapped_column(String(50), nullable=False)
    sentiment_label: Mapped[str] = mapped_column(String(20), nullable=False)
    comment: Mapped[str] = mapped_column(Text, nullable=False)
    rating: Mapped[int] = mapped_column(Integer, nullable=False)

    customer: Mapped[Customer] = relationship(back_populates="feedback")

    __table_args__ = (
        CheckConstraint("rating BETWEEN 1 AND 5", name="ck_feedback_rating"),
        CheckConstraint(
            "sentiment_label IN ('positive', 'neutral', 'negative')",
            name="ck_feedback_sentiment",
        ),
        Index("ix_feedback_customer", "customer_id"),
        Index("ix_feedback_sentiment", "sentiment_label"),
    )
