from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
from inference.model_loader import load_model
from inference.drift_checker import DriftChecker
import time
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from monitoring.metrics import (
    REQUEST_COUNT,
    REQUEST_LATENCY,
    PREDICTION_COUNT,
    DRIFT_COUNT
)
from fastapi.responses import Response


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
    start_time = time.time()

    X = np.array(request.features, dtype=np.float32)
    preds = model.predict(X)

    PREDICTION_COUNT.inc()

    drift_detected = drift_checker.check(X)
    if drift_detected:
        DRIFT_COUNT.inc()

    latency = time.time() - start_time
    REQUEST_LATENCY.labels(endpoint="/predict").observe(latency)
    REQUEST_COUNT.labels(
        endpoint="/predict",
        method="POST",
        status="200"
    ).inc()

    return {
        "predictions": preds.tolist(),
        "drift_detected": drift_detected
    }



@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
