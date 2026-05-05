from pathlib import Path
from typing import Dict, Any
import joblib
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]

MODEL_DIR = BASE_DIR / "models"

PREPROCESSOR_PATH = MODEL_DIR / "preprocessor.pkl"
BEST_MODEL_PATH = MODEL_DIR / "best_model.pkl"
BEST_THRESHOLD_PATH = MODEL_DIR / "best_threshold.pkl"


def load_prediction_artifacts():
    """
    Load trained model, preprocessor, and best threshold.
    """
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    model = joblib.load(BEST_MODEL_PATH)
    threshold = joblib.load(BEST_THRESHOLD_PATH)

    return preprocessor, model, threshold


def get_risk_level(churn_probability: float) -> str:
    """
    Convert churn probability into a risk level.
    """
    if churn_probability >= 0.7:
        return "High"
    elif churn_probability >= 0.4:
        return "Medium"
    else:
        return "Low"


def get_recommended_action(risk_level: str) -> str:
    """
    Recommend a retention action based on risk level.
    """
    actions = {
        "High": "Offer a personalized discount, contact the customer proactively, and suggest a contract upgrade.",
        "Medium": "Send a satisfaction survey, provide a small incentive, and monitor customer engagement.",
        "Low": "Maintain regular communication and continue standard customer care."
    }

    return actions.get(risk_level, "No recommendation available.")


def predict_churn(customer_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Predict churn probability and risk level for a single customer.
    """

    preprocessor, model, threshold = load_prediction_artifacts()

    customer_df = pd.DataFrame([customer_data])

    customer_processed = preprocessor.transform(customer_df)

    churn_probability = float(model.predict_proba(customer_processed)[:, 1][0])
    churn_prediction = int(churn_probability >= threshold)

    risk_level = get_risk_level(churn_probability)
    recommended_action = get_recommended_action(risk_level)

    result = {
        "churn_probability": round(churn_probability, 4),
        "threshold": threshold,
        "churn_prediction": churn_prediction,
        "prediction_label": "Churn" if churn_prediction == 1 else "Not Churn",
        "risk_level": risk_level,
        "recommended_action": recommended_action
    }

    return result


if __name__ == "__main__":
    sample_customer = {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 5,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "Yes",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 90.5,
        "TotalCharges": 452.5
    }

    prediction = predict_churn(sample_customer)

    print("Prediction result:")
    print(prediction)