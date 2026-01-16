from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
from inference.model_loader import load_model
from inference.drift_checker import DriftChecker

app = FastAPI(title="Self-Healing ML Inference Service")

model = load_model()
drift_checker = DriftChecker(threshold=0.05)

class PredictionRequest(BaseModel):
    features: list[list[float]]

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(request: PredictionRequest):
    X = np.array(request.features, dtype=np.float32)
    preds = model.predict(X)

    drift_detected = drift_checker.check(X)

    return {
        "predictions": preds.tolist(),
        "drift_detected": drift_detected
    }
