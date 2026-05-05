from pathlib import Path
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]
RAW_DATA_PATH = BASE_DIR / "data" / "raw" / "telco_churn.csv"

# Load data 
def load_raw_data() -> pd.DataFrame:
    """
    Load the raw Telco Customer Churn dataset.
    """
    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at {RAW_DATA_PATH}. "
            "Please place telco_churn.csv inside data/raw/."
        )

    df = pd.read_csv(RAW_DATA_PATH)
    return df


if __name__ == "__main__":
    data = load_raw_data()
    print("Dataset loaded successfully!")
    print(f"Shape: {data.shape}")
    print(data.head())