from pathlib import Path
import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)


BASE_DIR = Path(__file__).resolve().parents[1]

PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"
MODEL_DIR = BASE_DIR / "models"
REPORT_DIR = BASE_DIR / "reports"

X_TEST_PATH = PROCESSED_DATA_DIR / "X_test.pkl"
Y_TEST_PATH = PROCESSED_DATA_DIR / "y_test.pkl"

BEST_MODEL_PATH = MODEL_DIR / "best_model.pkl"
BEST_THRESHOLD_PATH = MODEL_DIR / "best_threshold.pkl"

THRESHOLD_REPORT_PATH = REPORT_DIR / "threshold_tuning_report.csv"


def load_data_and_model():
    """
    Load test data and the trained best model.
    """
    X_test = joblib.load(X_TEST_PATH)
    y_test = joblib.load(Y_TEST_PATH)
    model = joblib.load(BEST_MODEL_PATH)

    return X_test, y_test, model


def evaluate_thresholds(y_true, y_proba):
    """
    Evaluate model performance across different probability thresholds.
    """
    thresholds = np.arange(0.10, 0.91, 0.01)
    results = []

    for threshold in thresholds:
        y_pred = (y_proba >= threshold).astype(int)

        results.append({
            "threshold": round(threshold, 2),
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred),
            "f1_score": f1_score(y_true, y_pred),
            "roc_auc": roc_auc_score(y_true, y_proba),
            "confusion_matrix": confusion_matrix(y_true, y_pred).tolist()
        })

    return pd.DataFrame(results)


def select_best_threshold(results_df: pd.DataFrame):
    """
    Select the best threshold based on business-oriented logic.

    Priority:
    1. Keep recall >= 0.70
    2. Maximize F1-score
    """

    business_candidates = results_df[results_df["recall"] >= 0.70]

    if not business_candidates.empty:
        best_row = business_candidates.sort_values(
            by=["f1_score", "precision"],
            ascending=False
        ).iloc[0]
    else:
        best_row = results_df.sort_values(
            by="f1_score",
            ascending=False
        ).iloc[0]

    return best_row


def main():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading test data and model...")
    X_test, y_test, model = load_data_and_model()

    print("Predicting churn probabilities...")
    y_proba = model.predict_proba(X_test)[:, 1]

    print("Evaluating thresholds...")
    results_df = evaluate_thresholds(y_test, y_proba)

    best_row = select_best_threshold(results_df)
    best_threshold = float(best_row["threshold"])

    results_df.to_csv(THRESHOLD_REPORT_PATH, index=False)
    joblib.dump(best_threshold, BEST_THRESHOLD_PATH)

    print("\nThreshold tuning completed successfully!")
    print(f"Best threshold: {best_threshold}")
    print("\nBest threshold metrics:")
    print(f"Accuracy : {best_row['accuracy']:.4f}")
    print(f"Precision: {best_row['precision']:.4f}")
    print(f"Recall   : {best_row['recall']:.4f}")
    print(f"F1-score : {best_row['f1_score']:.4f}")
    print(f"ROC-AUC  : {best_row['roc_auc']:.4f}")
    print(f"Confusion Matrix: {best_row['confusion_matrix']}")

    print(f"\nThreshold report saved to: {THRESHOLD_REPORT_PATH}")
    print(f"Best threshold saved to: {BEST_THRESHOLD_PATH}")


if __name__ == "__main__":
    main()