import uvicorn
import pandas as pd
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from xgboost import XGBClassifier
import json

# -------------------------------------------------
# MODEL WRAPPER
# -------------------------------------------------
with open("./artifacts/feature_names.json") as f: 
    feature_names = json.load(f)
    
class SentinelModel:
    def __init__(self, model_path: str, metadata_path: str):
        # Load model
        self.model = XGBClassifier()
        self.model.load_model(model_path)

        # Load metadata
        with open(metadata_path) as f:
            meta = json.load(f)

        self.threshold = float(meta["sentinel_model"]["threshold"])
        self.feature_names = feature_names

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        prob = self.predict_proba(X)
        return (prob >= self.threshold).astype(int)

# -------------------------------------------------
# CONFIG
# -------------------------------------------------

MODEL_PATH = "./artifacts/sentinel.xgb"
META_PATH = "./artifacts/metadata.json"

model = SentinelModel(MODEL_PATH, META_PATH)

app = FastAPI(title="WaterOps Sentinel API", version="1.0")

# -------------------------------------------------
# REQUEST SCHEMA
# -------------------------------------------------

class PredictRequest(BaseModel):
    incident_id: str
    features: List[float]  # full feature vector

# -------------------------------------------------
# ENDPOINT
# -------------------------------------------------

@app.post("/predict")
def predict(req: PredictRequest):
    # Convert features into a DataFrame with correct dtype
    X = pd.DataFrame([req.features], columns=model.feature_names).astype("float32")

    risk_score = float(model.predict(X)[0])
    alert = risk_score >= model.threshold

    return {
        "incident_id": req.incident_id,
        "risk_score": risk_score,
        "alert": alert,
        "threshold": model.threshold
    }

# -------------------------------------------------
# ENTRYPOINT
# -------------------------------------------------

def main():
    uvicorn.run(
        "waterops.serve:app",
        host="localhost",
        port=8000,
        reload=False
    )

if __name__ == "__main__":
    main()



    