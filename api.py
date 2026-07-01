"""
api.py – FastAPI REST API for the AQI Platform
Exposes prediction and live data endpoints.
"""

import os
import joblib
import numpy as np
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field
from live_data import fetch_live_aqi
from ai_advisor import generate_advisory

app = FastAPI(
    title="AQI Prediction & Decision Intelligence API",
    description="REST API to predict AQI values, retrieve live conditions, and get AI-powered health recommendations.",
    version="1.0.0",
)

# ── Load Model and Scaler ─────────────────────────────────────────────────────
MODEL_PATH = "aqi_model.pkl"
SCALER_PATH = "scaler.pkl"

if not os.path.exists(MODEL_PATH):
    raise RuntimeError(f"Model file '{MODEL_PATH}' not found. Please run 'python train_model.py' first.")

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH) if os.path.exists(SCALER_PATH) else None

# ── AQI Classification Helper ──────────────────────────────────────────────────
AQI_SCALE = [
    (50,  "Good",                  "#56d364"),
    (100, "Moderate",              "#e3b341"),
    (150, "Unhealthy for Sensitive","#f0883e"),
    (200, "Unhealthy",             "#f85149"),
    (300, "Very Unhealthy",        "#bc8cff"),
    (500, "Hazardous",             "#da3633"),
]

def classify_aqi(val: float):
    for ceiling, label, color in AQI_SCALE:
        if val <= ceiling:
            return label, color
    return "Hazardous", "#da3633"

# ── Pydantic Request/Response Models ──────────────────────────────────────────
class PredictionRequest(BaseModel):
    co: float = Field(..., description="Carbon Monoxide (CO) AQI sub-index value (typical range: 0 to 50)", ge=0.0)
    ozone: float = Field(..., description="Ozone AQI sub-index value (typical range: 0 to 100)", ge=0.0)
    no2: float = Field(..., description="Nitrogen Dioxide (NO₂) AQI sub-index value (typical range: 0 to 100)", ge=0.0)
    pm25: float = Field(..., description="Fine Particulate Matter (PM 2.5) AQI sub-index value (typical range: 0 to 500)", ge=0.0)
    user_profile: str = Field(
        default="General public",
        description="User profile for advisory personalization. Options: 'General public', 'Asthma / respiratory condition', 'Parent of young children', 'Elderly', 'Outdoor athlete / runner'"
    )

class PredictionResponse(BaseModel):
    predicted_aqi: float
    category: str
    color: str
    advisory: str

class LiveAQIResponse(BaseModel):
    city: str
    aqi: float
    category: str
    color: str
    co: float
    ozone: float
    no2: float
    pm25: float
    dominant_pollutant: str
    time: str
    advisory: str

# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/")
def read_root():
    return {
        "message": "Welcome to the AQI Analytics Platform API",
        "docs_url": "/docs",
        "endpoints": {
            "predict": "POST /predict",
            "live_aqi": "GET /live/{city}"
        }
    }

@app.post("/predict", response_model=PredictionResponse)
def predict_aqi_endpoint(payload: PredictionRequest):
    try:
        # Scale inputs if scaler exists
        input_arr = np.array([[payload.co, payload.ozone, payload.no2, payload.pm25]])
        if scaler:
            input_arr = scaler.transform(input_arr)
        
        predicted_val = float(model.predict(input_arr)[0])
        category, color = classify_aqi(predicted_val)
        
        # Generate advisory
        advisory_text = generate_advisory(
            predicted_val, category, payload.co, payload.ozone, payload.no2, payload.pm25,
            user_profile=payload.user_profile
        )
        
        return PredictionResponse(
            predicted_aqi=round(predicted_val, 2),
            category=category,
            color=color,
            advisory=advisory_text
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.get("/live/{city}", response_model=LiveAQIResponse)
def get_live_aqi_endpoint(city: str, user_profile: str = Query("General public", description="User profile for advisory personalization")):
    live = fetch_live_aqi(city)
    
    if live is None:
        raise HTTPException(status_code=404, detail=f"Could not retrieve data for city '{city}'")
    
    if "error" in live:
        if live["error"] == "no_token":
            raise HTTPException(
                status_code=401,
                detail="AQICN token is not configured. Please set the AQICN_TOKEN secret or environment variable."
            )
        else:
            raise HTTPException(status_code=400, detail=f"AQICN error: {live['error']}")
            
    try:
        aqi_val = float(live["aqi"])
        category, color = classify_aqi(aqi_val)
        
        # Generate advisory
        advisory_text = generate_advisory(
            aqi_val, category, live["co"], live["ozone"], live["no2"], live["pm25"],
            city=live["city"], user_profile=user_profile
        )
        
        return LiveAQIResponse(
            city=live["city"],
            aqi=aqi_val,
            category=category,
            color=color,
            co=live["co"],
            ozone=live["ozone"],
            no2=live["no2"],
            pm25=live["pm25"],
            dominant_pollutant=live["dominant_pollutant"],
            time=live["time"],
            advisory=advisory_text
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating live report: {str(e)}")
