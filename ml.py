"""
Simple churn model utilities.
Train: basic logistic regression on synthetic / uploaded features
Predict: returns probability for churn
Save / load model with joblib
"""
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib
from pathlib import Path

MODEL_PATH = Path("models/churn_model.joblib")
SCHEMA_PATH = Path("models/feature_columns.txt")

def train_from_dataframe(df: pd.DataFrame, label_col="churned"):
    """
    df: must contain features and a boolean/int label column named label_col
    Returns: trained pipeline
    """
    # simple feature selection: use numeric columns except label
    numeric = df.select_dtypes(include="number").columns.tolist()
    if label_col not in numeric:
        raise ValueError(f"{label_col} must be numeric in dataframe")
    features = [c for c in numeric if c != label_col]
    X = df[features]
    y = df[label_col].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000))
    ])
    pipeline.fit(X_train, y_train)
    acc = pipeline.score(X_test, y_test)
    # persist
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    with open(SCHEMA_PATH, "w") as f:
        f.write("\n".join(features))
    return pipeline, acc, features

def load_model():
    if MODEL_PATH.exists():
        return joblib.load(MODEL_PATH)
    return None

def predict_probabilities(df: pd.DataFrame):
    """
    Input df must contain the same feature columns used at training.
    Returns array of probabilities for class '1' (churn)
    """
    model = load_model()
    if model is None:
        raise RuntimeError("Model not trained yet.")
    return model.predict_proba(df)[:, 1]
