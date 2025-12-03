import pandas as pd
from typing import List
from pathlib import Path

def read_customer_csv(path_or_file) -> pd.DataFrame:
    """
    Expects a CSV with at least:
      customer_id, feature_1, feature_2, ..., churned (0/1)
    For demo: if churned absent, we won't be able to train.
    """
    df = pd.read_csv(path_or_file)
    return df

def ensure_features_present(df, features: List[str]):
    missing = [f for f in features if f not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")
    return True
