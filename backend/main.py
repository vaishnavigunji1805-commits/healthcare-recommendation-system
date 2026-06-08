from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from model_handler import HealthcareModelHandler
import uvicorn

app = FastAPI(title="LSTM Production Healthcare API")

# Initialize the global engine handler
engine = HealthcareModelHandler()

class BiometricSnapshot(BaseModel):
    heart_rate: float
    blood_pressure_systolic: float
    sleep_hours: float
    steps: float

class SequencePayload(BaseModel):
    history: List[BiometricSnapshot]

@app.post("/predict")
def predict_health_trends(payload: SequencePayload):
    if len(payload.history) != 5:
        raise HTTPException(status_code=400, detail="LSTM input layers expect exactly 5 sequence blocks.")
    
    # Parse incoming Pydantic matrix rows into standard numerical lists
    formatted_matrix = []
    for snapshot in payload.history:
        formatted_matrix.append([
            snapshot.heart_rate,
            snapshot.blood_pressure_systolic,
            snapshot.sleep_hours,
            snapshot.steps
        ])
        
    # Run the sequence through our deep learning pipeline worker
    risk_probability = engine.run_prediction(formatted_matrix)
    risk_percentage = round(risk_probability * 100, 2)
    
    # Decision boundaries
    if risk_percentage > 70.0:
        recommendation = "CRITICAL: Sequential multi-step anomalies mapped by LSTM layers. Rest recommended immediately."
    elif risk_percentage > 40.0:
        recommendation = "WARNING: Variance trends detected over recent time frames. Monitor activity and maintain hydration."
    else:
        recommendation = "OPTIMAL: Time-series biometric patterns are running within stable control limits."
        
    return {
        "risk_score_percentage": risk_percentage,
        "recommendation": recommendation
    }

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
