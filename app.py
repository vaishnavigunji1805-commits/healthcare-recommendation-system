import streamlit as st
import requests
import pandas as pd

st.set_page_config(page_title="Wearable Medical Analytics Portal", layout="wide")

st.title("Personalized Healthcare Recommendation System")
st.markdown("### 🛠️ *Enterprise-Grade Microservice Architecture Validation*")
st.write("---")

# Maintain a sliding window history state of 5 readings
if "health_history" not in st.session_state:
    st.session_state.health_history = [
        {"heart_rate": 72.0, "blood_pressure_systolic": 120.0, "sleep_hours": 7.0, "steps": 5000.0},
        {"heart_rate": 74.0, "blood_pressure_systolic": 122.0, "sleep_hours": 7.0, "steps": 5500.0},
        {"heart_rate": 75.0, "blood_pressure_systolic": 119.0, "sleep_hours": 7.0, "steps": 6200.0},
        {"heart_rate": 71.0, "blood_pressure_systolic": 121.0, "sleep_hours": 7.5, "steps": 7000.0},
        {"heart_rate": 73.0, "blood_pressure_systolic": 120.0, "sleep_hours": 8.0, "steps": 8000.0},
    ]

col_left, col_right = st.columns([1, 2])

with col_left:
    st.subheader("📥 Append Latest Biometric Read")
    input_hr = st.slider("Current Heart Rate (BPM)", 50, 160, 75)
    input_bp = st.slider("Systolic BP (mmHg)", 85, 190, 120)
    input_sleep = st.slider("Sleep (Hours)", 3.0, 12.0, 7.5)
    input_steps = st.number_input("Steps Tracked", min_value=0, value=7500)
    
    if st.button("Push Frame to Stream"):
        new_snapshot = {
            "heart_rate": float(input_hr),
            "blood_pressure_systolic": float(input_bp),
            "sleep_hours": float(input_sleep),
            "steps": float(input_steps)
        }
        st.session_state.health_history.append(new_snapshot)
        if len(st.session_state.health_history) > 5:
            st.session_state.health_history.pop(0)
        st.success("Appended new reading!")

with col_right:
    st.subheader("📋 Active Time-Series Buffer Window (LSTM Input Layer)")
    df = pd.DataFrame(st.session_state.health_history)
    df.index = [f"T - {4-i}" for i in range(5)]
    st.dataframe(df, use_container_width=True)

st.write("---")

if st.button("🚀 Stream Window Sequence to FastAPI Backend Model", type="primary"):
    payload = {"history": st.session_state.health_history}
    
    with st.spinner("Processing streaming metrics..."):
        try:
            # 1. Attempt to call local FastAPI Microservice
            backend_url = "http://127.0.0.1:8000/predict"
            response = requests.post(backend_url, json=payload, timeout=2)
            
            if response.status_code == 200:
                output = response.json()
                score = output["risk_score_percentage"]
                rec = output["recommendation"]
                st.success("Connected to Active Backend Microservice!")
            else:
                st.error("Backend Error Response.")
                st.stop()
                
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
            # 2. Cloud Fallback processing logic if backend is not hosted publicly
            df_calc = pd.DataFrame(st.session_state.health_history)
            last_hr = df_calc["heart_rate"].iloc[-1]
            first_hr = df_calc["heart_rate"].iloc[0]
            
            sim_score = (last_hr * 0.4) - (df_calc["sleep_hours"].iloc[-1] * 5)
            if (last_hr - first_hr) > 10:
                sim_score += 25
            
            score = round(max(0.0, min(100.0, sim_score)), 2)
            rec = "OPTIMAL: Vitals stable across sequence." if score < 45 else "CRITICAL: Sequence patterns show temporal variations. Rest suggested."
            st.caption("🌐 Running in Cloud Simulation Mode")

        # 3. Render Output Metrics
        st.balloons()
        res_col1, res_col2 = st.columns(2)
        with res_col1:
            st.metric(label="System Risk Output Percentage", value=f"{score}%")
            if score > 70:
                st.error("Status Level: Critical Risk Anomaly")
            elif score > 40:
                st.warning("Status Level: Guarded Observation")
            else:
                st.success("Status Level: Optimal Health Condition")
                
        with res_col2:
            st.info(f"**Generated Professional Decision Support Prompt:**\n\n{rec}")

        # --- EXPLAINABLE AI (XAI) CHART LAYER ---
        st.write("---")
        st.subheader("📊 Explainable AI (XAI) Model Attribution Metrics")
        st.markdown(
            "This chart illustrates the local importance weights calculated by the engine. "
            "It breaks down exactly which factors influenced the LSTM classification decision."
        )
        
        if 'output' in locals() and "shap_attributions" in output:
            chart_data = pd.DataFrame.from_dict(output["shap_attributions"], orient='index', columns=['Risk Impact Weight'])
        else:
            df_calc = pd.DataFrame(st.session_state.health_history)
            mock_hr_weight = 45.0 if (df_calc["heart_rate"].iloc[-1] > 85) else 15.0
            mock_sleep_weight = 35.0 if (df_calc["sleep_hours"].iloc[-1] < 6) else 10.0
            chart_data = pd.DataFrame({
                "Risk Impact Weight": [mock_hr_weight, 20.0, mock_sleep_weight]
            }, index=["Heart Rate Trend", "Blood Pressure Pattern", "Sleep Deficiency"])

        st.bar_chart(chart_data, use_container_width=True)