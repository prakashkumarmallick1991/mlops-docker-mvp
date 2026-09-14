from fastapi import FastAPI
from pydantic import BaseModel

from app.model import predict

app = FastAPI(title="Dockerized ML API")


class PredictionRequest(BaseModel):
    value: float


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/predict")
def prediction(request: PredictionRequest):
    result = predict(request.value)

    return {
        "input": request.value,
        "prediction": result
    }
