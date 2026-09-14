from datetime import datetime, timezone

from fastapi import FastAPI
from pydantic import BaseModel

from app.model import MODEL_VERSION, predict
from app.drift import calculate_psi, drift_detected


app = FastAPI(title="Dockerized ML API")


class PredictionRequest(BaseModel):
    value: float


class DriftRequest(BaseModel):
    training_data: list[float]
    production_data: list[float]


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/predict")
def prediction(request: PredictionRequest):
    result = predict(request.value)

    prediction_log = {
        "input": request.value,
        "prediction": result,
        "model_version": MODEL_VERSION,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }

    return prediction_log


@app.post("/drift")
def detect_drift(request: DriftRequest):
    psi = calculate_psi(
        request.training_data,
        request.production_data
    )

    return {
        "psi": psi,
        "drift_detected": drift_detected(psi)
    }