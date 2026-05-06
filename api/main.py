from typing import Dict, Any
from fastapi import FastAPI
from pydantic import BaseModel, Field

from src.predict import predict_churn


app = FastAPI(
    title="Customer Churn Prediction API",
    description="An API for predicting customer churn probability and risk level.",
    version="1.0.0"
)


class CustomerData(BaseModel):
    gender: str = Field(..., example="Female")
    SeniorCitizen: int = Field(..., example=0)
    Partner: str = Field(..., example="Yes")
    Dependents: str = Field(..., example="No")
    tenure: int = Field(..., example=5)
    PhoneService: str = Field(..., example="Yes")
    MultipleLines: str = Field(..., example="No")
    InternetService: str = Field(..., example="Fiber optic")
    OnlineSecurity: str = Field(..., example="No")
    OnlineBackup: str = Field(..., example="No")
    DeviceProtection: str = Field(..., example="No")
    TechSupport: str = Field(..., example="No")
    StreamingTV: str = Field(..., example="Yes")
    StreamingMovies: str = Field(..., example="Yes")
    Contract: str = Field(..., example="Month-to-month")
    PaperlessBilling: str = Field(..., example="Yes")
    PaymentMethod: str = Field(..., example="Electronic check")
    MonthlyCharges: float = Field(..., example=90.5)
    TotalCharges: float = Field(..., example=452.5)


@app.get("/")
def read_root() -> Dict[str, str]:
    """
    Root endpoint.
    """
    return {
        "message": "Customer Churn Prediction API is running."
    }


@app.get("/health")
def health_check() -> Dict[str, str]:
    """
    Health check endpoint.
    """
    return {
        "status": "healthy"
    }


@app.post("/predict")
def predict(customer: CustomerData) -> Dict[str, Any]:
    """
    Predict churn probability for a customer.
    """
    customer_dict = customer.model_dump()
    prediction = predict_churn(customer_dict)

    return prediction