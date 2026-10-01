from pathlib import Path
from time import perf_counter

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

MODEL_PATH = Path(__file__).resolve().parent.parent / "model" / "wine_model.joblib"
ARTIFACT = joblib.load(MODEL_PATH)
MODEL = ARTIFACT["model"]
FEATURE_NAMES = ARTIFACT["feature_names"]
TARGET_NAMES = ARTIFACT["target_names"]

app = FastAPI(
    title="Wine Classifier API",
    description="Cloud Run model-serving API for Lab Assignment 5",
    version="1.0.0",
)


class WineRequest(BaseModel):
    features: list[float] = Field(
        ..., min_length=13, max_length=13, description="13 wine features in model order"
    )


class WineResponse(BaseModel):
    predicted_class: int
    predicted_label: str
    inference_latency_ms: float


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/features")
def features() -> dict[str, list[str]]:
    return {"feature_names": FEATURE_NAMES}


@app.post("/predict", response_model=WineResponse)
def predict(request: WineRequest) -> WineResponse:
    if len(request.features) != len(FEATURE_NAMES):
        raise HTTPException(status_code=400, detail="Exactly 13 features are required")

    started = perf_counter()
    row = pd.DataFrame([request.features], columns=FEATURE_NAMES)
    predicted_class = int(MODEL.predict(row)[0])
    elapsed_ms = (perf_counter() - started) * 1000

    return WineResponse(
        predicted_class=predicted_class,
        predicted_label=TARGET_NAMES[predicted_class],
        inference_latency_ms=round(elapsed_ms, 3),
    )
