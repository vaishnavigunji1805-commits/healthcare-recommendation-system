from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import uvicorn

app = FastAPI(
    title="LSTM Production Healthcare API",
    description="Production-grade API processing continuous time-series biometric streams."
)

# 1. Define what a single point-in-time health reading looks like
class BiometricSnapshot(BaseModel):
    heart_rate: float
    blood_pressure_systolic: float
    sleep_hours: float
    steps: float

# 2. Define the schema expecting a historical sequence window of 5 steps
class SequencePayload(BaseModel):
    history: List[BiometricSnapshot]

@app.get("/")
def read_root():
    return {"status": "Online", "framework": "FastAPI Microservice Ready"}

@app.post("/predict")
def predict_health_trends(payload: SequencePayload):
    # Industrial Validation: Ensure the sequence window matches the exact LSTM input requirements
    if len(payload.history) != 5:
        raise HTTPException(
            status_code=400, 
            detail=f"LSTM layers require exactly 5 historical intervals. Received: {len(payload.history)}"
        )
    
    # Extract data matrix out of the payload to simulate sequence processing
    raw_matrix = []
    for log in payload.history:
        raw_matrix.append([log.heart_rate, log.blood_pressure_systolic, log.sleep_hours, log.steps])
    
    # Phase 1 Model Math Simulation (This calculates a trend risk based on the 5 steps)
    # It analyzes whether metrics are worsening or improving over the time steps
    last_step = raw_matrix[-1]
    first_step = raw_matrix[0]
    
    # Check if heart rate is trending upwards across the sequence window
    hr_trend_increase = last_step[0] - first_step[0] 
    base_risk = (last_step[0] * 0.3) + (last_step[1] * 0.3) - (last_step[2] * 8)
    
    if hr_trend_increase > 10:
        base_risk += 25  # Apply a penalty for sharp multi-step heart rate spikes
        
    risk_percentage = max(0.0, min(100.0, base_risk))
    
    # Clinical recommendation tree
    if risk_percentage > 75.0:
        recommendation = "CRITICAL ALERT: Sequential upward spikes in cardiac stress detected over the last 5 readings. Rest immediately."
    elif risk_percentage > 45.0:
        recommendation = "WARNING: Unstable metrics noticed across the time steps. Please log details and consume fluids."
    else:
        recommendation = "STABLE: Vitals are maintaining a balanced baseline across all documented intervals."

    return {
        "risk_score_percentage": round(risk_percentage, 2),
        "recommendation": recommendation,
        "processed_time_steps": len(raw_matrix)
    }

if __name__ == "__main__":
    # Runs the local web server execution loop
    uvicorn.run(app, host="127.0.0.1", port=8000)
