import sqlite3
from datetime import datetime

import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder


st.set_page_config(page_title="AI Healthcare System", layout="wide")

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


# ---------------- DATABASE ---------------- #

def get_connection():
    return sqlite3.connect("healthcare_app.db", check_same_thread=False)


conn = get_connection()
cursor = conn.cursor()


def create_tables():
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS health_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL,
            timestamp_saved TEXT NOT NULL,
            heart_rate REAL,
            spo2 REAL,
            health_score REAL,
            stress_level TEXT,
            fatigue_prediction TEXT,
            recommendation TEXT
        )
    """)
    conn.commit()


create_tables()


def signup_user(username, password):
    try:
        cursor.execute(
            "INSERT INTO users (username, password) VALUES (?, ?)",
            (username, password)
        )
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False


def login_user(username, password):
    cursor.execute(
        "SELECT * FROM users WHERE username = ? AND password = ?",
        (username, password)
    )
    return cursor.fetchone()


def save_health_record(username, latest_row, predicted_label):
    cursor.execute("""
        INSERT INTO health_history (
            username, timestamp_saved, heart_rate, spo2, health_score,
            stress_level, fatigue_prediction, recommendation
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        username,
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        float(latest_row["heart_rate"]),
        float(latest_row["spo2"]),
        float(latest_row["health_score"]),
        str(latest_row["stress_level"]),
        str(predicted_label),
        str(latest_row["recommendation"])
    ))
    conn.commit()


def get_user_history(username):
    cursor.execute("""
        SELECT timestamp_saved, heart_rate, spo2, health_score,
               stress_level, fatigue_prediction, recommendation
        FROM health_history
        WHERE username = ?
        ORDER BY id DESC
    """, (username,))
    rows = cursor.fetchall()
    columns = [
        "saved_at", "heart_rate", "spo2", "health_score",
        "stress_level", "fatigue_prediction", "recommendation"
    ]
    return pd.DataFrame(rows, columns=columns)


# ---------------- SESSION ---------------- #

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "username" not in st.session_state:
    st.session_state.username = ""

if "auth_mode" not in st.session_state:
    st.session_state.auth_mode = "login"


# ---------------- AUTH UI ---------------- #

def auth_page():
    st.title("AI-Based Personalized Healthcare Recommendation System")
    st.subheader("User Authentication")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Login"):
            st.session_state.auth_mode = "login"
    with col2:
        if st.button("Sign Up"):
            st.session_state.auth_mode = "signup"

    st.write(f"Current mode: **{st.session_state.auth_mode.capitalize()}**")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.session_state.auth_mode == "signup":
        if st.button("Create Account"):
            if not username or not password:
                st.error("Please enter both username and password.")
            else:
                created = signup_user(username, password)
                if created:
                    st.success("Account created successfully. Now login.")
                    st.session_state.auth_mode = "login"
                else:
                    st.error("Username already exists.")
    else:
        if st.button("Login Now"):
            user = login_user(username, password)
            if user:
                st.session_state.logged_in = True
                st.session_state.username = username
                st.rerun()
            else:
                st.error("Invalid credentials.")


if not st.session_state.logged_in:
    auth_page()
    st.stop()


# ---------------- APP HEADER ---------------- #

top1, top2 = st.columns([5, 1])
with top1:
    st.title("Healthcare Dashboard")
    st.write(f"Welcome, **{st.session_state.username}**")
with top2:
    if st.button("Logout"):
        st.session_state.logged_in = False
        st.session_state.username = ""
        st.rerun()


# ---------------- APP BODY ---------------- #

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

    if row['sleep_hours'] < 6:
        rec.append("Sleep more tonight.")

    if row['steps'] < row['steps_avg']:
        rec.append("Increase your physical activity.")

    if row['resting_heart_rate'] > row['rhr_avg'] + 3:
        rec.append("Take rest, your recovery seems low.")
        
    if row["stress_level"] == "high":
        rec.append("High stress detected. Try relaxation techniques.")

    if not rec:
        rec.append("You are doing great. Keep it up!")
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
        #  Alert System
if latest_score < 50:
    st.error(" Alert: Health condition is critical! Immediate care needed.")
elif latest_score < 70:
    st.warning(" Warning: Monitor your health closely.")
else:
    st.success(" Your health is stable.")
    st.subheader("Average Summary")
    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Avg Heart Rate", round(df["heart_rate"].mean(), 2))
    s2.metric("Avg Sleep Hours", round(df["sleep_hours"].mean(), 2))
    s3.metric("Avg SpO2", round(df["spo2"].mean(), 2))
    s4.metric("Avg Steps", round(df["steps"].mean(), 2))
    st.subheader("Health Trends")
#  Weekly Summary
st.subheader(" Weekly Summary")

weekly_avg = df.resample('D', on='Timestamp').mean(numeric_only=True)

st.line_chart(weekly_avg[['heart_rate', 'health_score']])
        st.line_chart(
    df.set_index("Timestamp")[["heart_rate", "temperature", "respiration", "health_score"]],
    height=300
)
        st.subheader("Latest Recommendation")
        st.success(latest_row["recommendation"])

        st.subheader("AI Prediction")
        st.info(f"Predicted Fatigue Level: {predicted_label}")

        st.subheader("Prediction Explanation")
        st.write(
            "The fatigue prediction is based on heart rate, temperature, respiration, steps, sleep hours, and SpO2."
        )

        if st.button("Save This Analysis"):
            save_health_record(st.session_state.username, latest_row, predicted_label)
            st.success("Analysis saved to database successfully.")

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

# ================= USER HISTORY =================

st.subheader("User History")
history_df = get_user_history(st.session_state.username)
st.subheader("User History")
history_df = get_user_history(st.session_state.username)

#  CLEAR HISTORY BUTTON WITH CONFIRMATION
if st.button("Clear My History "):
    confirm = st.checkbox("Are you sure?")
    if confirm:
        cursor.execute(
            "DELETE FROM health_history WHERE username = ?",
            (st.session_state.username,)
        )
        conn.commit()
        st.success("History cleared successfully.")
        st.rerun()

#  KEEP THIS PART SAME
if history_df.empty:
    st.info("No saved history yet.")
else:
    st.dataframe(history_df, use_container_width=True)

# Existing logic
if history_df.empty:
    st.info("No saved history yet.")
else:
    st.dataframe(history_df, use_container_width=True)

if history_df.empty:
    st.info("No saved history yet.")
else:
    st.dataframe(history_df, use_container_width=True)

    # 📈 History Analytics
    st.subheader("Health Score Trend (History)")
    history_df["saved_at"] = pd.to_datetime(history_df["saved_at"])

    st.line_chart(
        history_df.set_index("saved_at")["health_score"],
        height=300
    )

    # 📊 Stress Distribution
    st.subheader("Stress Level Distribution")
    stress_counts = history_df["stress_level"].value_counts()
    st.bar_chart(stress_counts)

    # 📉 Performance Insight
    st.subheader("Performance Insight")

    avg_score = history_df["health_score"].mean()
    latest_score = history_df["health_score"].iloc[0]
    oldest_score = history_df["health_score"].iloc[-1]

    if avg_score >= 80:
        st.success("Overall health trend is GOOD")
    elif avg_score >= 50:
        st.warning("Health trend is MODERATE")
    else:
        st.error("Health trend is CRITICAL")

    if latest_score > oldest_score:
        st.success("Your health is improving")
    elif latest_score < oldest_score:
        st.error("Your health is declining")
    else:
        st.info("No significant change in health")
