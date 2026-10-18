from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[3]
RAW_DIR = ROOT / "data" / "raw"
ARTIFACT_DIR = ROOT / "artifacts" / "ml"
REPORT_DIR = ROOT / "reports" / "ml"


def main() -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    feedback = pd.read_csv(RAW_DIR / "feedback.csv")
    feedback = feedback.dropna(subset=["comment", "sentiment_label"]).copy()
    feedback["comment"] = feedback["comment"].astype(str).str.strip()
    feedback["sentiment_label"] = (
        feedback["sentiment_label"].astype(str).str.strip().str.lower()
    )
    feedback = feedback.loc[feedback["comment"] != ""]

    if feedback["sentiment_label"].nunique() < 2:
        raise ValueError("At least two sentiment classes are required.")

    X_train, X_test, y_train, y_test = train_test_split(
        feedback["comment"],
        feedback["sentiment_label"],
        test_size=0.2,
        random_state=42,
        stratify=feedback["sentiment_label"],
    )

    model = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    strip_accents="unicode",
                    ngram_range=(1, 2),
                    max_features=5000,
                    min_df=1,
                    sublinear_tf=True,
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    macro_f1 = f1_score(y_test, predictions, average="macro", zero_division=0)
    report = classification_report(
        y_test, predictions, output_dict=True, zero_division=0
    )
    labels = sorted(feedback["sentiment_label"].unique())

    matrix = confusion_matrix(y_test, predictions, labels=labels)
    matrix_df = pd.DataFrame(
        matrix,
        index=[f"actual_{label}" for label in labels],
        columns=[f"predicted_{label}" for label in labels],
    )

    joblib.dump(model, ARTIFACT_DIR / "sentiment_classifier.joblib")
    matrix_df.to_csv(REPORT_DIR / "sentiment_confusion_matrix.csv")

    metrics = {
        "model": "TF-IDF + Logistic Regression",
        "dataset": "synthetic feedback",
        "total_records": int(len(feedback)),
        "training_records": int(len(X_train)),
        "test_records": int(len(X_test)),
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "classification_report": report,
        "random_state": 42,
    }

    (REPORT_DIR / "sentiment_metrics.json").write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )

    print(f"Training records: {len(X_train)}")
    print(f"Test records: {len(X_test)}")
    print(f"Accuracy: {accuracy:.3f}")
    print(f"Macro F1: {macro_f1:.3f}")
    print("\nClassification report:")
    print(classification_report(y_test, predictions, zero_division=0))
    print(f"Reports saved to: {REPORT_DIR}")


if __name__ == "__main__":
    main()
