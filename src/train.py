from pathlib import Path
import joblib
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)


BASE_DIR = Path(__file__).resolve().parents[1]

PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"
MODEL_DIR = BASE_DIR / "models"
REPORT_DIR = BASE_DIR / "reports"

X_TRAIN_PATH = PROCESSED_DATA_DIR / "X_train.pkl"
X_TEST_PATH = PROCESSED_DATA_DIR / "X_test.pkl"
Y_TRAIN_PATH = PROCESSED_DATA_DIR / "y_train.pkl"
Y_TEST_PATH = PROCESSED_DATA_DIR / "y_test.pkl"

BEST_MODEL_PATH = MODEL_DIR / "best_model.pkl"
MODEL_REPORT_PATH = REPORT_DIR / "model_report.md"


def load_processed_data():
    """
    Load processed training and testing datasets.
    """
    X_train = joblib.load(X_TRAIN_PATH)
    X_test = joblib.load(X_TEST_PATH)
    y_train = joblib.load(Y_TRAIN_PATH)
    y_test = joblib.load(Y_TEST_PATH)

    return X_train, X_test, y_train, y_test


def get_models():
    """
    Define machine learning models for comparison.
    """

    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42
        ),

        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        ),

        "XGBoost": XGBClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=42
        )
    }

    return models


def evaluate_model(model, X_test, y_test):
    """
    Evaluate a trained model using common classification metrics.
    """

    y_pred = model.predict(X_test)

    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
    else:
        y_proba = y_pred

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "classification_report": classification_report(y_test, y_pred)
    }

    return metrics


def train_and_evaluate_models():
    """
    Train multiple models, evaluate them, and save the best model.
    """

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading processed data...")
    X_train, X_test, y_train, y_test = load_processed_data()

    print("Preparing models...")
    models = get_models()

    results = []
    trained_models = {}

    for model_name, model in models.items():
        print(f"\nTraining {model_name}...")

        model.fit(X_train, y_train)
        trained_models[model_name] = model

        print(f"Evaluating {model_name}...")
        metrics = evaluate_model(model, X_test, y_test)

        results.append({
            "model": model_name,
            "accuracy": metrics["accuracy"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1_score": metrics["f1_score"],
            "roc_auc": metrics["roc_auc"]
        })

        print(f"{model_name} Results:")
        print(f"Accuracy : {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall   : {metrics['recall']:.4f}")
        print(f"F1-score : {metrics['f1_score']:.4f}")
        print(f"ROC-AUC  : {metrics['roc_auc']:.4f}")
        print("Confusion Matrix:")
        print(metrics["confusion_matrix"])

    results_df = pd.DataFrame(results)

    print("\nModel Comparison:")
    print(results_df)

    # For churn prediction, ROC-AUC is a strong metric for selecting the best model.
    best_model_name = results_df.sort_values(by="roc_auc", ascending=False).iloc[0]["model"]
    best_model = trained_models[best_model_name]

    print(f"\nBest model selected: {best_model_name}")

    joblib.dump(best_model, BEST_MODEL_PATH)
    print(f"Best model saved to: {BEST_MODEL_PATH}")

    save_model_report(results_df, best_model_name)

    print("\nTraining process completed successfully!")


def save_model_report(results_df: pd.DataFrame, best_model_name: str):
    """
    Save model comparison results to a Markdown report.
    """

    report_content = "# Model Training Report\n\n"

    report_content += "## Model Comparison\n\n"
    report_content += results_df.to_markdown(index=False)
    report_content += "\n\n"

    report_content += "## Best Model\n\n"
    report_content += f"The best model selected based on ROC-AUC is **{best_model_name}**.\n\n"

    report_content += "## Notes\n\n"
    report_content += (
        "For customer churn prediction, recall and ROC-AUC are important metrics. "
        "Recall helps identify more customers who are likely to churn, while ROC-AUC "
        "measures the model's ability to distinguish between churn and non-churn customers."
    )

    with open(MODEL_REPORT_PATH, "w", encoding="utf-8") as file:
        file.write(report_content)

    print(f"Model report saved to: {MODEL_REPORT_PATH}")


if __name__ == "__main__":
    train_and_evaluate_models()