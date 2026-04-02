"""
StockSense AI — FastAPI Serving Layer

Endpoints:
  GET  /health                — liveness + model metadata
  POST /predict               — single prediction from OHLCV JSON
  POST /predict/csv           — bulk prediction from uploaded CSV
  POST /explain               — SHAP explanation for one prediction
  GET  /tickers               — supported tickers info

Run:
  uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
"""

from __future__ import annotations

import json
import time
import traceback
from io import StringIO
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator

#  Paths 
BASE_DIR      = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"

sys_path_hack = str(BASE_DIR)
import sys
if sys_path_hack not in sys.path:
    sys.path.insert(0, sys_path_hack)

from model.features import engineer_features, extract_feature_row, FEATURE_NAMES

#  Load Artifacts 
def _load():
    try:
        model     = joblib.load(ARTIFACTS_DIR / "model.pkl")
        scaler    = joblib.load(ARTIFACTS_DIR / "scaler.pkl")
        explainer = joblib.load(ARTIFACTS_DIR / "shap_explainer.pkl")
        with open(ARTIFACTS_DIR / "metadata.json") as f:
            meta = json.load(f)
        return model, scaler, explainer, meta
    except FileNotFoundError as e:
        raise RuntimeError(f"Artifact not found: {e}. Run 'python -m model.train' first.") from e

model, scaler, explainer, METADATA = _load()
THRESHOLD       = METADATA["threshold"]
FEATURE_DISPLAY = METADATA["feature_display"]

#  App 
app = FastAPI(
    title="StockSense AI",
    description=(
        "AI-powered stock direction prediction API. "
        "Predicts UP/DOWN signal for the next 5 trading days with SHAP explanations."
    ),
    version=METADATA.get("model_version", "1.0.0"),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

#  Schemas 
class OHLCVRow(BaseModel):
    """Single OHLCV candle."""
    date:   str
    open:   float = Field(..., gt=0)
    high:   float = Field(..., gt=0)
    low:    float = Field(..., gt=0)
    close:  float = Field(..., gt=0)
    volume: float = Field(..., gt=0)

    @validator("high")
    def high_gte_low(cls, v, values):
        if "low" in values and v < values["low"]:
            raise ValueError("high must be >= low")
        return v


class PredictRequest(BaseModel):
    ticker: str = Field(..., example="AAPL")
    candles: List[OHLCVRow] = Field(
        ...,
        min_items=60,
        description="At least 60 OHLCV candles (most recent last).",
    )


class FactorDetail(BaseModel):
    feature:      str
    display_name: str
    value:        float
    shap_value:   float
    direction:    str   # "bullish" | "bearish"


class PredictResponse(BaseModel):
    ticker:          str
    signal:          str            # "BUY" | "HOLD/SELL"
    confidence:      float          # probability of UP move
    forward_days:    int
    top_factors:     List[FactorDetail]
    raw_features:    Dict[str, float]
    prediction_time_ms: float


class HealthResponse(BaseModel):
    status:          str
    model_version:   str
    test_auc:        float
    test_accuracy:   float
    forward_days:    int
    trained_at:      str
    features_count:  int


#  Helpers 
def candles_to_df(candles: List[OHLCVRow]) -> pd.DataFrame:
    data = [c.dict() for c in candles]
    df   = pd.DataFrame(data)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").set_index("date")
    return df


def run_prediction(df_ohlcv: pd.DataFrame):
    """Run full feature engineering → scale → predict → shap pipeline."""
    feat_row = extract_feature_row(df_ohlcv)

    if feat_row.isnull().any().any():
        null_cols = feat_row.columns[feat_row.isnull().any()].tolist()
        raise HTTPException(
            status_code=422,
            detail=f"Not enough data to compute features: {null_cols}. "
                   "Supply at least 60 candles.",
        )

    X_scaled = pd.DataFrame(
        scaler.transform(feat_row),
        columns=feat_row.columns,
    )

    prob   = float(model.predict_proba(X_scaled)[0, 1])
    signal = "BUY" if prob >= THRESHOLD else "HOLD/SELL"

    # SHAP
    sv     = explainer(X_scaled)
    shap_v = sv.values[0]

    factors = []
    for fname, sval, fval in zip(FEATURE_NAMES, shap_v, feat_row.values[0]):
        factors.append(FactorDetail(
            feature      = fname,
            display_name = FEATURE_DISPLAY.get(fname, fname),
            value        = round(float(fval), 6),
            shap_value   = round(float(sval), 6),
            direction    = "bullish" if sval > 0 else "bearish",
        ))

    factors.sort(key=lambda x: abs(x.shap_value), reverse=True)
    raw = {f: round(float(v), 6) for f, v in zip(FEATURE_NAMES, feat_row.values[0])}

    return prob, signal, factors, raw


#  Endpoints 
@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status         = "ok",
        model_version  = METADATA.get("model_version", "1.0.0"),
        test_auc       = METADATA.get("test_auc", 0.0),
        test_accuracy  = METADATA.get("test_accuracy", 0.0),
        forward_days   = METADATA.get("forward_days", 5),
        trained_at     = METADATA.get("trained_at", ""),
        features_count = len(FEATURE_NAMES),
    )


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    t0 = time.perf_counter()
    try:
        df_ohlcv          = candles_to_df(req.candles)
        prob, signal, factors, raw = run_prediction(df_ohlcv)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")

    elapsed_ms = (time.perf_counter() - t0) * 1000

    return PredictResponse(
        ticker           = req.ticker.upper(),
        signal           = signal,
        confidence       = round(prob, 4),
        forward_days     = METADATA.get("forward_days", 5),
        top_factors      = factors[:10],
        raw_features     = raw,
        prediction_time_ms = round(elapsed_ms, 2),
    )


@app.post("/predict/csv")
async def predict_csv(file: UploadFile = File(...)):
    """
    Upload a CSV with columns: date, open, high, low, close, volume
    Returns predictions for every valid window in the file.
    """
    content = await file.read()
    try:
        df = pd.read_csv(StringIO(content.decode("utf-8")))
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"CSV parse error: {e}")

    required = {"date", "open", "high", "low", "close", "volume"}
    if not required.issubset(df.columns):
        raise HTTPException(
            status_code=422,
            detail=f"CSV must contain columns: {required}. Got: {list(df.columns)}",
        )

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").set_index("date")

    try:
        prob, signal, factors, raw = run_prediction(df)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {
        "signal":     signal,
        "confidence": round(prob, 4),
        "top_factors": [f.dict() for f in factors[:8]],
    }


@app.get("/tickers")
def tickers():
    return {
        "tickers_trained_on": METADATA.get("tickers_trained", []),
        "note": (
            "The model was trained on synthetic data mimicking these tickers. "
            "Swap train.py's generate_ohlcv() with a real data source (yfinance, Alpha Vantage, etc.) "
            "to make live predictions."
        ),
    }
