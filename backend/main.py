@app.post("/predict")
def predict_health_trends(payload: SequencePayload):
    if len(payload.history) != 5:
        raise HTTPException(status_code=400, detail="Sequence windows require exactly 5 logs.")
    
    formatted_matrix = []
    for snapshot in payload.history:
        formatted_matrix.append([
            snapshot.heart_rate,
            snapshot.blood_pressure_systolic,
            snapshot.sleep_hours,
            snapshot.steps
        ])
        
    # Execute our upgraded XAI deep prediction pipeline
    pipeline_result = engine.run_prediction_with_xai(formatted_matrix)
    risk_percentage = round(pipeline_result["risk_score"] * 100, 2)
    
    if risk_percentage > 70.0:
        recommendation = "CRITICAL: Sequential anomalies mapped via LSTM cells. Rest suggested immediately."
    elif risk_percentage > 40.0:
        recommendation = "WARNING: Variance trends detected over recent time steps. Monitor activity carefully."
    else:
        recommendation = "OPTIMAL: Time-series biometric frames are within stable, healthy parameters."
        
    return {
        "risk_score_percentage": risk_percentage,
        "recommendation": recommendation,
        "shap_attributions": pipeline_result["attributions"] # Passes XAI coordinates down the wire
    }
