from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib

from src.utils import load_raw_data


BASE_DIR = Path(__file__).resolve().parents[1]

PROCESSED_DATA_DIR = BASE_DIR / 'data' / 'processed'
MODEL_DIR = BASE_DIR / 'models'

X_TRAIN_PATH = PROCESSED_DATA_DIR / 'X_train.pkl'
X_TEST_PATH = PROCESSED_DATA_DIR / 'X_test.pkl'
Y_TRAIN_PATH = PROCESSED_DATA_DIR / 'y_train.pkl'
Y_TEST_PATH = PROCESSED_DATA_DIR / 'y_test.pkl'
PREPROCESSOR_PATH = MODEL_DIR / 'preprocessor.pkl'

# Clean the raw Telco Customer Churn dataset
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if 'customerID' in df.columns:
        df = df.drop(columns=['customerID'])

    # Convert TotalCharges to numeric
    df['TotalCharges']  = pd.to_numeric(df['TotalCharges'], errors= 'coerce')

    # Fill missing TotalCharges with median value
    df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())

    # Convert target variable: Yes -> 1, No -> 0
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
    return df


# Split dataframe into features X and target y.
def split_features_target(df: pd.DataFrame):
    X = df.drop(columns=['Churn'])
    y = df['Churn']

    return X, y

# Build preprocessing pipeline for numerical and categorical features.
def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numerical_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X.select_dtypes(include=["object", "string"]).columns.tolist()

    numerical_transformer = Pipeline(
        steps= [('scaler', StandardScaler())]
    )

    categorical_transformer = Pipeline(
        steps=[('encoder', OneHotEncoder(handle_unknown='ignore'))]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )
    return preprocessor

# Main function to clean, split, transform, and save processed data
def preprocess_data():

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading raw data...")
    df = load_raw_data()

    print("Cleaning data...")
    df_cleaned = clean_data(df)

    print("Splitting features and target...")
    X, y = split_features_target(df_cleaned)

    print("Splitting train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print("Building preprocessor...")
    preprocessor = build_preprocessor(X_train)

    print("Fitting and transforming training data...")
    X_train_processed = preprocessor.fit_transform(X_train)

    print("Transforming test data...")
    X_test_processed = preprocessor.transform(X_test)

    print("Saving processed data...")
    joblib.dump(X_train_processed, X_TRAIN_PATH)
    joblib.dump(X_test_processed, X_TEST_PATH)
    joblib.dump(y_train, Y_TRAIN_PATH)
    joblib.dump(y_test, Y_TEST_PATH)

    print("Saving preprocessor...")
    joblib.dump(preprocessor, PREPROCESSOR_PATH)

    print("Data preprocessing completed successfully!")
    print(f"X_train shape: {X_train_processed.shape}")
    print(f"X_test shape: {X_test_processed.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"y_test shape: {y_test.shape}")


if __name__ == "__main__":
    preprocess_data()


