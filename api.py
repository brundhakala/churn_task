from fastapi import FastAPI, File, UploadFile, HTTPException
import pandas as pd
from io import BytesIO
from database import init_db, get_session
from models import Customer, ChurnScore, Feedback, Campaign, CampaignEvent
from ml import train_from_dataframe, load_model, predict_probabilities, MODEL_PATH
from sentiment import analyze_sentiment
from utils import read_customer_csv
from sqlmodel import select
import os
from typing import List

app = FastAPI(title="ChurnGuard POC")

@app.on_event("startup")
def on_startup():
    init_db()

@app.post("/upload/customers-csv")
async def upload_customers_csv(file: UploadFile = File(...)):
    contents = await file.read()
    df = pd.read_csv(BytesIO(contents))
    # Expect at least customer_id column and numeric features
    if "customer_id" not in df.columns:
        raise HTTPException(status_code=422, detail="CSV must contain customer_id column")
    # Save customers into DB if needed (demo)
    from database import get_session
    sess = get_session()
    inserted = 0
    for _, row in df.iterrows():
        c = Customer(org_id=1, external_id=str(row["customer_id"]), name=row.get("name"))
        sess.add(c)
        inserted += 1
    sess.commit()
    return {"rows": len(df), "inserted_customers": inserted}

@app.post("/train")
async def train(file: UploadFile = File(...), label_col: str = "churned"):
    """
    Train model from an uploaded CSV that contains numeric feature columns and a label column.
    """
    contents = await file.read()
    df = pd.read_csv(BytesIO(contents))
    if label_col not in df.columns:
        raise HTTPException(status_code=422, detail=f"Label column '{label_col}' not found in CSV")
    model, acc, features = train_from_dataframe(df, label_col)
    return {"accuracy": float(acc), "features": features, "model_path": str(MODEL_PATH)}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Predict churn probabilities for an uploaded CSV that must contain the same numeric feature columns used at training.
    """
    contents = await file.read()
    df = pd.read_csv(BytesIO(contents))
    # load feature schema
    schema_file = "models/feature_columns.txt"
    if not os.path.exists(schema_file):
        raise HTTPException(status_code=400, detail="No trained model/schema found. Train first.")
    with open(schema_file) as f:
        features = [l.strip() for l in f if l.strip()]
    missing = [f for f in features if f not in df.columns]
    if missing:
        raise HTTPException(status_code=422, detail=f"Missing features in uploaded CSV: {missing}")
    X = df[features]
    probs = predict_probabilities(X)
    df_result = df.copy()
    df_result["churn_probability"] = probs
    # persist churn scores to DB for demo
    sess = get_session()
    for idx, row in df_result.iterrows():
        # In real system map customer_id to internal id
        cs = ChurnScore(customer_id=int(row.get("customer_id", idx)), probability=float(row["churn_probability"]))
        sess.add(cs)
    sess.commit()
    return {"predictions": df_result[["customer_id", "churn_probability"]].to_dict(orient="records")}

@app.post("/feedback")
async def feedback(customer_id: int, text: str):
    s = analyze_sentiment(text)
    sess = get_session()
    fb = Feedback(customer_id=customer_id, text=text, sentiment=s)
    sess.add(fb)
    sess.commit()
    return {"customer_id": customer_id, "sentiment": s}

@app.post("/campaigns/{campaign_id}/trigger")
async def trigger_campaign(campaign_id: int, top_n: int = 50):
    """
    Example: pick top N customers by churn probability and enqueue campaign events.
    """
    sess = get_session()
    stmt = select(ChurnScore).order_by(ChurnScore.probability.desc()).limit(top_n)
    results = sess.exec(stmt).all()
    events = []
    for cs in results:
        ev = CampaignEvent(campaign_id=campaign_id, customer_id=cs.customer_id, status="queued", details=f"prob={cs.probability:.3f}")
        sess.add(ev)
        events.append({"customer_id": cs.customer_id, "probability": cs.probability})
    sess.commit()
    return {"enqueued": len(events), "customers": events}

@app.get("/health")
def health():
    return {"status": "ok"}
