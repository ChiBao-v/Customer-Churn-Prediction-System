from pathlib import Path
import sys

import streamlit as st
import pandas as pd


# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.predict import predict_churn


# =========================
# Page Configuration
# =========================
st.set_page_config(
    page_title="Customer Churn Prediction System",
    page_icon="📉",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================
# Custom CSS
# =========================
st.markdown(
    """
    <style>
    .main {
        background-color: #f7f9fc;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    .hero-card {
        background: linear-gradient(135deg, #1f4e79 0%, #2563eb 100%);
        padding: 2rem;
        border-radius: 20px;
        color: white;
        margin-bottom: 1.5rem;
        box-shadow: 0 8px 25px rgba(0,0,0,0.12);
    }

    .hero-title {
        font-size: 2.4rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
    }

    .hero-subtitle {
        font-size: 1.05rem;
        opacity: 0.95;
        line-height: 1.6;
    }

    .section-card {
        background-color: white;
        padding: 1.5rem;
        border-radius: 18px;
        box-shadow: 0 4px 18px rgba(0,0,0,0.06);
        margin-bottom: 1.2rem;
        border: 1px solid #eef2f7;
    }

    .metric-card {
        background-color: white;
        padding: 1.3rem;
        border-radius: 18px;
        box-shadow: 0 4px 18px rgba(0,0,0,0.06);
        text-align: center;
        border: 1px solid #eef2f7;
    }

    .metric-label {
        font-size: 0.9rem;
        color: #64748b;
        font-weight: 600;
        margin-bottom: 0.4rem;
    }

    .metric-value {
        font-size: 2rem;
        font-weight: 800;
        color: #0f172a;
    }

    .risk-high {
        background-color: #fee2e2;
        color: #991b1b;
        padding: 1rem;
        border-radius: 14px;
        border-left: 6px solid #ef4444;
        font-weight: 600;
    }

    .risk-medium {
        background-color: #fef3c7;
        color: #92400e;
        padding: 1rem;
        border-radius: 14px;
        border-left: 6px solid #f59e0b;
        font-weight: 600;
    }

    .risk-low {
        background-color: #dcfce7;
        color: #166534;
        padding: 1rem;
        border-radius: 14px;
        border-left: 6px solid #22c55e;
        font-weight: 600;
    }

    .insight-box {
        background-color: #eff6ff;
        color: #1e3a8a;
        padding: 1rem;
        border-radius: 14px;
        border-left: 6px solid #3b82f6;
        margin-bottom: 1rem;
    }

    .small-note {
        font-size: 0.88rem;
        color: #64748b;
        line-height: 1.5;
    }

    div[data-testid="stMetricValue"] {
        font-size: 1.8rem;
        font-weight: 800;
    }

    .stButton > button {
        width: 100%;
        border-radius: 12px;
        height: 3rem;
        font-weight: 700;
        background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%);
        color: white;
        border: none;
    }

    .stButton > button:hover {
        background: linear-gradient(135deg, #1d4ed8 0%, #1e40af 100%);
        color: white;
        border: none;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# =========================
# Helper Functions
# =========================
def get_risk_css_class(risk_level: str) -> str:
    if risk_level == "High":
        return "risk-high"
    elif risk_level == "Medium":
        return "risk-medium"
    return "risk-low"


def get_risk_emoji(risk_level: str) -> str:
    if risk_level == "High":
        return "🔴"
    elif risk_level == "Medium":
        return "🟡"
    return "🟢"


def render_metric_card(label: str, value: str):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


# =========================
# Sidebar
# =========================
with st.sidebar:
    st.markdown("## 📌 Project Info")

    st.markdown(
        """
        **Customer Churn Prediction System**

        This app predicts whether a customer is likely to churn based on customer profile, services, contract type, and billing information.
        """
    )

    st.markdown("---")

    st.markdown("### 🧠 Model")
    st.markdown(
        """
        - Best model: **XGBoost**
        - Optimized threshold: **0.30**
        - Focus metric: **Recall**
        - Explainability: **SHAP**
        """
    )

    st.markdown("---")

    st.markdown("### 🎯 Risk Levels")
    st.markdown(
        """
        🟢 **Low Risk**: Customer is likely to stay  
        🟡 **Medium Risk**: Customer needs monitoring  
        🔴 **High Risk**: Customer needs proactive retention  
        """
    )

    st.markdown("---")

    st.markdown(
        """
        <p class="small-note">
        This dashboard is designed for portfolio demonstration and business-oriented machine learning deployment.
        </p>
        """,
        unsafe_allow_html=True
    )


# =========================
# Hero Section
# =========================
st.markdown(
    """
    <div class="hero-card">
        <div class="hero-title">📉 Customer Churn Prediction System</div>
        <div class="hero-subtitle">
            An end-to-end machine learning application that predicts customer churn probability, 
            classifies customer risk level, and recommends retention actions for business users.
        </div>
    </div>
    """,
    unsafe_allow_html=True
)


# =========================
# Top Summary Cards
# =========================
summary_col1, summary_col2, summary_col3, summary_col4 = st.columns(4)

with summary_col1:
    render_metric_card("Best Model", "XGBoost")

with summary_col2:
    render_metric_card("Threshold", "0.30")

with summary_col3:
    render_metric_card("Recall", "76.47%")

with summary_col4:
    render_metric_card("ROC-AUC", "84.22%")


st.markdown("<br>", unsafe_allow_html=True)


# =========================
# Tabs
# =========================
tab_predict, tab_about, tab_guide = st.tabs(
    ["🔮 Churn Prediction", "📊 Model Overview", "💡 Business Guide"]
)


# =========================
# Prediction Tab
# =========================
with tab_predict:
    st.markdown(
        """
        <div class="section-card">
            <h3>🔮 Customer Information</h3>
            <p class="small-note">
            Enter customer information below and click <b>Predict Churn</b> to estimate churn probability and risk level.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    with st.form("prediction_form"):
        st.markdown("### 👤 Customer Profile")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            gender = st.selectbox("Gender", ["Female", "Male"])
        with col2:
            senior_citizen = st.selectbox("Senior Citizen", [0, 1])
        with col3:
            partner = st.selectbox("Partner", ["Yes", "No"])
        with col4:
            dependents = st.selectbox("Dependents", ["Yes", "No"])

        st.markdown("### 📞 Services Information")

        col5, col6, col7 = st.columns(3)

        with col5:
            phone_service = st.selectbox("Phone Service", ["Yes", "No"])
            multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
            internet_service = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])

        with col6:
            online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
            online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
            device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])

        with col7:
            tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
            streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
            streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])

        st.markdown("### 📄 Contract & Billing Information")

        col8, col9, col10 = st.columns(3)

        with col8:
            contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
            paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])

        with col9:
            payment_method = st.selectbox(
                "Payment Method",
                [
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer (automatic)",
                    "Credit card (automatic)"
                ]
            )
            tenure = st.number_input(
                "Tenure (months)",
                min_value=0,
                max_value=100,
                value=5
            )

        with col10:
            monthly_charges = st.number_input(
                "Monthly Charges",
                min_value=0.0,
                max_value=200.0,
                value=90.5,
                step=0.1
            )

            total_charges = st.number_input(
                "Total Charges",
                min_value=0.0,
                max_value=10000.0,
                value=452.5,
                step=0.1
            )

        submitted = st.form_submit_button("🚀 Predict Churn")

    customer_data = {
        "gender": gender,
        "SeniorCitizen": senior_citizen,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges
    }

    if submitted:
        prediction = predict_churn(customer_data)

        churn_probability = prediction["churn_probability"]
        prediction_label = prediction["prediction_label"]
        risk_level = prediction["risk_level"]
        recommended_action = prediction["recommended_action"]
        threshold = prediction["threshold"]

        risk_class = get_risk_css_class(risk_level)
        risk_emoji = get_risk_emoji(risk_level)

        st.markdown("## 📌 Prediction Result")

        result_col1, result_col2, result_col3 = st.columns(3)

        with result_col1:
            st.metric(
                label="Churn Probability",
                value=f"{churn_probability * 100:.2f}%"
            )

        with result_col2:
            st.metric(
                label="Prediction",
                value=prediction_label
            )

        with result_col3:
            st.metric(
                label="Risk Level",
                value=f"{risk_emoji} {risk_level}"
            )

        st.markdown("### Probability Indicator")
        st.progress(min(churn_probability, 1.0))

        st.markdown(
            f"""
            <div class="{risk_class}">
                <h4>{risk_emoji} {risk_level} Risk Customer</h4>
                <p><b>Recommended Action:</b> {recommended_action}</p>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown("<br>", unsafe_allow_html=True)

        col_result_a, col_result_b = st.columns([1, 1])

        with col_result_a:
            st.markdown("### ⚙️ Decision Logic")
            st.info(
                f"""
                The model predicts churn based on probability.

                Current optimized threshold: **{threshold}**

                If churn probability is greater than or equal to this threshold, the customer is classified as **Churn**.
                """
            )

        with col_result_b:
            st.markdown("### 📦 Input Data")
            st.json(customer_data)

        st.markdown("### 🧾 Prediction Summary")

        summary_df = pd.DataFrame(
            {
                "Item": [
                    "Churn Probability",
                    "Decision Threshold",
                    "Prediction Label",
                    "Risk Level",
                    "Recommended Action"
                ],
                "Value": [
                    f"{churn_probability * 100:.2f}%",
                    threshold,
                    prediction_label,
                    risk_level,
                    recommended_action
                ]
            }
        )

        st.dataframe(summary_df, use_container_width=True)


# =========================
# Model Overview Tab
# =========================
with tab_about:
    st.markdown("## 📊 Model Overview")

    st.markdown(
        """
        <div class="section-card">
            <h3>Machine Learning Pipeline</h3>
            <p>
            The system follows a complete machine learning workflow: data preprocessing, 
            model training, evaluation, threshold tuning, prediction service, and dashboard deployment.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    model_results = pd.DataFrame(
        {
            "Model": ["Logistic Regression", "Random Forest", "XGBoost"],
            "Accuracy": [0.7381, 0.7821, 0.8041],
            "Precision": [0.5043, 0.6175, 0.6644],
            "Recall": [0.7834, 0.4706, 0.5294],
            "F1-score": [0.6136, 0.5341, 0.5893],
            "ROC-AUC": [0.8413, 0.8217, 0.8422]
        }
    )

    st.markdown("### 🏆 Model Comparison")
    st.dataframe(model_results, use_container_width=True)

    st.markdown("### 🎚️ Threshold Tuning Result")

    threshold_results = pd.DataFrame(
        {
            "Metric": ["Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC"],
            "After Threshold Tuning": [0.7615, 0.5356, 0.7647, 0.6300, 0.8422]
        }
    )

    st.dataframe(threshold_results, use_container_width=True)

    st.markdown(
        """
        <div class="insight-box">
            <b>Key Insight:</b> The optimized threshold improves recall from 52.94% to 76.47%.
            This is useful for churn prediction because the business wants to identify more customers who are likely to leave.
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("### 🧠 SHAP Explainability")

    st.markdown(
        """
        SHAP is used to explain how different features contribute to churn predictions. 
        This helps business users understand why a customer is classified as high-risk or low-risk.
        """
    )

    shap_col1, shap_col2 = st.columns(2)

    shap_feature_path = PROJECT_ROOT / "reports" / "figures" / "shap_feature_importance.png"
    shap_summary_path = PROJECT_ROOT / "reports" / "figures" / "shap_summary_plot.png"

    with shap_col1:
        st.markdown("#### SHAP Feature Importance")
        if shap_feature_path.exists():
            st.image(str(shap_feature_path), use_container_width=True)
        else:
            st.warning("SHAP feature importance image not found.")

    with shap_col2:
        st.markdown("#### SHAP Summary Plot")
        if shap_summary_path.exists():
            st.image(str(shap_summary_path), use_container_width=True)
        else:
            st.warning("SHAP summary plot image not found.")


# =========================
# Business Guide Tab
# =========================
with tab_guide:
    st.markdown("## 💡 Business Guide")

    st.markdown(
        """
        <div class="section-card">
            <h3>How to Use the Prediction Result</h3>
            <p>
            The prediction result should be used to support customer retention decisions. 
            High-risk customers should receive more proactive and personalized retention actions.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    guide_col1, guide_col2, guide_col3 = st.columns(3)

    with guide_col1:
        st.markdown(
            """
            <div class="risk-low">
                <h4>🟢 Low Risk</h4>
                <p>Maintain regular communication and continue standard customer care.</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with guide_col2:
        st.markdown(
            """
            <div class="risk-medium">
                <h4>🟡 Medium Risk</h4>
                <p>Send a satisfaction survey, provide a small incentive, and monitor customer engagement.</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with guide_col3:
        st.markdown(
            """
            <div class="risk-high">
                <h4>🔴 High Risk</h4>
                <p>Offer a personalized discount, contact the customer proactively, and suggest a contract upgrade.</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("### 📌 Recommended Retention Strategy")

    strategy_df = pd.DataFrame(
        {
            "Risk Level": ["Low", "Medium", "High"],
            "Business Priority": ["Low", "Medium", "High"],
            "Recommended Action": [
                "Maintain regular communication and standard care.",
                "Send satisfaction survey and provide a small incentive.",
                "Contact proactively and offer personalized retention incentives."
            ]
        }
    )

    st.dataframe(strategy_df, use_container_width=True)

    st.markdown(
        """
        <div class="insight-box">
            <b>Business Value:</b> Instead of offering promotions to all customers, 
            the company can focus resources on customers who are more likely to churn.
            This helps reduce marketing waste and improve customer retention efficiency.
        </div>
        """,
        unsafe_allow_html=True
    )