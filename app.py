
import streamlit as st
import pandas as pd
# Simple login system
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

def login():
    st.title("🔐 Login System")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

   if st.button("Login"):
    if username == "admin" and password == "1234":
        st.session_state.logged_in = True
        st.rerun()
        else:
            st.error("Invalid credentials")

if not st.session_state.logged_in:
    login()
    st.stop()
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
st.markdown("""
<style>
body {
    background-color: #0e1117;
    color: white;
}

h1, h2, h3 {
    color: #00c6ff;
}

div[data-testid="stMetric"] {
    background-color: #1e222a;
    padding: 15px;
    border-radius: 12px;
    text-align: center;
    box-shadow: 0 4px 12px rgba(0,0,0,0.25);
}

div[data-testid="stMetric"] label {
    color: #cfd8e3 !important;
    font-weight: 600;
}

div[data-testid="stMetric"] div {
    color: #ffffff !important;
}

.stAlert {
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)

st.set_page_config(page_title="AI Healthcare System", layout="wide")

st.title("AI-Based Personalized Healthcare Recommendation System")
st.write("Upload wearable sensor data to analyze health trends, predict fatigue, and generate recommendations.")

uploaded_file = st.file_uploader("Upload wearable data CSV", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    required_columns = [
        "Timestamp",
        "heart_rate",
        "temperature",
        "respiration",
        "steps",
        "sleep_hours",
        "stress_level",
        "spo2",
        "calories",
        "activity_type",
        "fatigue_level",
    ]

    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        st.error(f"Missing required columns: {', '.join(missing_columns)}")
    else:
        df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
        df = df.dropna().copy()
        df = df.sort_values("Timestamp").reset_index(drop=True)

        df["hr_avg"] = df["heart_rate"].rolling(3, min_periods=1).mean()
        df["temp_avg"] = df["temperature"].rolling(3, min_periods=1).mean()
        df["resp_avg"] = df["respiration"].rolling(3, min_periods=1).mean()
        df["steps_avg"] = df["steps"].rolling(3, min_periods=1).mean()
        df["sleep_avg"] = df["sleep_hours"].rolling(3, min_periods=1).mean()

        def health_score(row):
            score = 100

            if row["heart_rate"] > row["hr_avg"] + 10:
                score -= 20
            if row["temperature"] > 37.5:
                score -= 20
            if row["respiration"] > 20:
                score -= 15
            if row["sleep_hours"] < 6:
                score -= 15
            if row["spo2"] < 95:
                score -= 10
            if row["steps"] < 4000:
                score -= 10

            return max(score, 0)

        def recommend(row):
            rec = []

            if row["heart_rate"] > row["hr_avg"] + 10:
                rec.append("High heart rate detected. Take rest and avoid heavy activity.")
            if row["temperature"] > 37.5:
                rec.append("Body temperature is elevated. Monitor your health closely.")
            if row["respiration"] > 20:
                rec.append("Respiration rate is high. Try breathing exercises and avoid overexertion.")
            if row["sleep_hours"] < 6:
                rec.append("Sleep duration is low. Aim for better rest tonight.")
            if row["spo2"] < 95:
                rec.append("SpO2 is slightly low. Stay calm and monitor oxygen levels.")
            if row["steps"] < 4000:
                rec.append("Physical activity is low. Try a short walk or light movement.")
            if not rec:
                rec.append("Your health indicators look stable today. Maintain your current routine.")

            return " ".join(rec)

        df["health_score"] = df.apply(health_score, axis=1)
        df["recommendation"] = df.apply(recommend, axis=1)

        le = LabelEncoder()
        df["fatigue_encoded"] = le.fit_transform(df["fatigue_level"])

        X = df[["heart_rate", "temperature", "respiration", "steps", "sleep_hours", "spo2"]]
        y = df["fatigue_encoded"]

        model = RandomForestClassifier(random_state=42)
        model.fit(X, y)

        latest = X.iloc[-1].values.reshape(1, -1)
        prediction = model.predict(latest)
        predicted_label = le.inverse_transform(prediction)[0]

        latest_row = df.iloc[-1]
        latest_score = int(latest_row["health_score"])

        if latest_score >= 80:
            status = "Good"
            status_message = "Your current condition looks stable."
        elif latest_score >= 50:
            status = "Moderate"
            status_message = "Some health indicators need attention."
        else:
            status = "Critical"
            status_message = "Your health indicators require immediate care and rest."

        st.subheader("Dashboard")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Heart Rate", int(latest_row["heart_rate"]))
        col2.metric("SpO2", int(latest_row["spo2"]))
        col3.metric("Health Score", latest_score)
        col4.metric("Stress Level", str(latest_row["stress_level"]).capitalize())

        st.subheader("Current Health Status")
        st.write(f"**Status:** {status}")
        st.write(status_message)

        st.subheader("Average Summary")
        s1, s2, s3, s4 = st.columns(4)
        s1.metric("Avg Heart Rate", round(df["heart_rate"].mean(), 2))
        s2.metric("Avg Sleep Hours", round(df["sleep_hours"].mean(), 2))
        s3.metric("Avg SpO2", round(df["spo2"].mean(), 2))
        s4.metric("Avg Steps", round(df["steps"].mean(), 2))

        st.subheader("Health Trends")
        st.line_chart(
            df.set_index("Timestamp")[["heart_rate", "temperature", "respiration", "health_score"]]
        )

        st.subheader("Latest Recommendation")
        st.success(latest_row["recommendation"])

        st.subheader("AI Prediction")
        st.info(f"Predicted Fatigue Level: {predicted_label}")

        st.subheader("Prediction Explanation")
        st.write(
            "The fatigue prediction is based on heart rate, temperature, respiration, steps, sleep hours, and SpO2."
        )

        st.subheader("Processed Dataset")
        st.dataframe(df, use_container_width=True)

        st.subheader("Download Results")
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download analyzed data as CSV",
            data=csv,
            file_name="analyzed_health_data.csv",
            mime="text/csv",
        )
else:
    st.info("Please upload your combined_health_data.csv file to view analysis.")
