from pathlib import Path
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import shap


BASE_DIR = Path(__file__).resolve().parents[1]

PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"
MODEL_DIR = BASE_DIR / "models"
REPORT_DIR = BASE_DIR / "reports"
FIGURE_DIR = REPORT_DIR / "figures"

X_TEST_PATH = PROCESSED_DATA_DIR / "X_test.pkl"
BEST_MODEL_PATH = MODEL_DIR / "best_model.pkl"
PREPROCESSOR_PATH = MODEL_DIR / "preprocessor.pkl"

SHAP_SUMMARY_PLOT_PATH = FIGURE_DIR / "shap_summary_plot.png"
SHAP_BAR_PLOT_PATH = FIGURE_DIR / "shap_feature_importance.png"
SHAP_IMPORTANCE_CSV_PATH = REPORT_DIR / "shap_feature_importance.csv"


def load_artifacts():
    """
    Load processed test data, trained model, and preprocessor.
    """
    X_test = joblib.load(X_TEST_PATH)
    model = joblib.load(BEST_MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)

    return X_test, model, preprocessor


def get_feature_names(preprocessor):
    """
    Get feature names after preprocessing.
    """
    try:
        feature_names = preprocessor.get_feature_names_out()
    except Exception:
        feature_names = [f"feature_{i}" for i in range(preprocessor.transformers_.shape[0])]

    return feature_names


def convert_to_dataframe(X, feature_names):
    """
    Convert processed feature matrix into a pandas DataFrame.
    """
    if hasattr(X, "toarray"):
        X = X.toarray()

    return pd.DataFrame(X, columns=feature_names)


def generate_shap_explanations():
    """
    Generate SHAP summary plot and feature importance report.
    """

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading artifacts...")
    X_test, model, preprocessor = load_artifacts()

    print("Getting feature names...")
    feature_names = get_feature_names(preprocessor)

    print("Preparing sample data for SHAP...")
    X_test_df = convert_to_dataframe(X_test, feature_names)

    # Use a sample to make SHAP faster
    X_sample = X_test_df.sample(
        n=min(1000, len(X_test_df)),
        random_state=42
    )

    print("Creating SHAP explainer...")
    explainer = shap.TreeExplainer(model)

    print("Calculating SHAP values...")
    shap_values = explainer.shap_values(X_sample)

    print("Saving SHAP summary plot...")
    plt.figure()
    shap.summary_plot(
        shap_values,
        X_sample,
        show=False
    )
    plt.tight_layout()
    plt.savefig(SHAP_SUMMARY_PLOT_PATH, dpi=300, bbox_inches="tight")
    plt.close()

    print("Saving SHAP feature importance bar plot...")
    plt.figure()
    shap.summary_plot(
        shap_values,
        X_sample,
        plot_type="bar",
        show=False
    )
    plt.tight_layout()
    plt.savefig(SHAP_BAR_PLOT_PATH, dpi=300, bbox_inches="tight")
    plt.close()

    print("Saving SHAP feature importance CSV...")
    shap_importance = pd.DataFrame({
        "feature": X_sample.columns,
        "mean_abs_shap_value": abs(shap_values).mean(axis=0)
    }).sort_values(by="mean_abs_shap_value", ascending=False)

    shap_importance.to_csv(SHAP_IMPORTANCE_CSV_PATH, index=False)

    print("\nSHAP explanation completed successfully!")
    print(f"SHAP summary plot saved to: {SHAP_SUMMARY_PLOT_PATH}")
    print(f"SHAP bar plot saved to: {SHAP_BAR_PLOT_PATH}")
    print(f"SHAP importance CSV saved to: {SHAP_IMPORTANCE_CSV_PATH}")

    print("\nTop 10 important features:")
    print(shap_importance.head(10))


if __name__ == "__main__":
    generate_shap_explanations()