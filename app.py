```python
import sqlite3
import hashlib
import random
from datetime import datetime

import pandas as pd
import streamlit as st

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

# ---------------- PAGE CONFIG ---------------- #

st.set_page_config(
    page_title="AI Healthcare System",
    layout="wide"
)

# ---------------- LIGHT MODE CSS ---------------- #

st.markdown("""
<style>

/* Main Background */
.stApp {
    background-color: #f4f6f9;
    color: #000000;
}

/* Headings */
h1, h2, h3 {
    color: #0d6efd;
    font-weight: bold;
}

/* Metric Cards */
div[data-testid="stMetric"] {
    background: white;
    border: 1px solid #dce3ea;
    border-radius: 15px;
    padding: 20px;
    box-shadow: 0px 2px 8px rgba(0,0,0,0.08);
}

/* Buttons */
.stButton > button {
    background-color: #0d6efd;
    color: white;
    border-radius: 10px;
    border: none;
    padding: 10px 18px;
    font-weight: bold;
}

.stButton > button:hover {
    background-color: #0b5ed7;
    color: white;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: white;
}

</style>
""", unsafe_allow_html=True)

# ---------------- DATABASE CONNECTION ---------------- #

def get_connection():
    return sqlite3.connect(
        "healthcare_app.db",
        check_same_thread=False
    )

conn = get_connection()
cursor = conn.cursor()

# ---------------- CREATE TABLES ---------------- #

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

# ---------------- PASSWORD HASHING ---------------- #

def hash_password(password):

    return hashlib.sha256(
        password.encode()
    ).hexdigest()

# ---------------- SIGNUP ---------------- #

def signup_user(username, password):

    hashed_password = hash_password(password)

    try:

        cursor.execute(
            "INSERT INTO users (username, password) VALUES (?, ?)",
            (username, hashed_password)
        )

        conn.commit()

        return True

    except sqlite3.IntegrityError:

        return False

# ---------------- LOGIN ---------------- #

def login_user(username, password):

    hashed_password = hash_password(password)

    cursor.execute(
        "SELECT * FROM users WHERE username=? AND password=?",
        (username, hashed_password)
    )

    return cursor.fetchone()

# ---------------- SAVE HEALTH RECORD ---------------- #

def save_health_record(
    username,
    latest_row,
    predicted_label
):

    cursor.execute("""
    INSERT INTO health_history (
        username,
        timestamp_saved,
        heart_rate,
        spo2,
        health_score,
        stress_level,
        fatigue_prediction,
        recommendation
    )
    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """,
    (
        username,
        datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        ),
        float(latest_row["heart_rate"]),
        float(latest_row["spo2"]),
        float(latest_row["health_score"]),
        str(latest_row["stress_level"]),
        str(predicted_label),
        str(latest_row["recommendation"])
    ))

    conn.commit()

# ---------------- GET USER HISTORY ---------------- #

def get_user_history(username):

    cursor.execute("""
    SELECT
        timestamp_saved,
        heart_rate,
        spo2,
        health_score,
        stress_level,
        fatigue_prediction,
        recommendation
    FROM health_history
    WHERE username=?
    ORDER BY id DESC
    """, (username,))

    rows = cursor.fetchall()

    columns = [
        "Saved At",
        "Heart Rate",
        "SpO2",
        "Health Score",
        "Stress Level",
        "Fatigue Prediction",
        "Recommendation"
    ]

    return pd.DataFrame(
        rows,
        columns=columns
    )

# ---------------- SESSION STATE ---------------- #

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "username" not in st.session_state:
    st.session_state.username = ""

if "auth_mode" not in st.session_state:
    st.session_state.auth_mode = "login"

# ---------------- AUTH PAGE ---------------- #

def auth_page():

    st.title(
        "AI-Based Personalized Healthcare Recommendation System"
    )

    st.subheader("User Authentication")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Login"):
            st.session_state.auth_mode = "login"

    with col2:
        if st.button("Sign Up"):
            st.session_state.auth_mode = "signup"

    username = st.text_input("Username")

    password = st.text_input(
        "Password",
        type="password"
    )

    # ---------------- SIGNUP ---------------- #

    if st.session_state.auth_mode == "signup":

        if st.button("Create Account"):

            if not username or not password:

                st.error(
                    "Please enter username and password"
                )

            else:

                created = signup_user(
                    username,
                    password
                )

                if created:

                    st.success(
                        "Account created successfully"
                    )

                    st.session_state.auth_mode = "login"

                else:

                    st.error(
                        "Username already exists"
                    )

    # ---------------- LOGIN ---------------- #

    else:

        if st.button("Login Now"):

            user = login_user(
                username,
                password
            )

            if user:

                st.session_state.logged_in = True

                st.session_state.username = username

                st.rerun()

            else:

                st.error(
                    "Invalid credentials"
                )

# ---------------- LOGIN CHECK ---------------- #

if not st.session_state.logged_in:

    auth_page()

    st.stop()

# ---------------- SIDEBAR MENU ---------------- #

menu = [
    "Healthcare Analysis",
    "Live Monitoring Dashboard",
    "User History"
]

choice = st.sidebar.selectbox(
    "Navigation",
    menu
)

# ---------------- HEADER ---------------- #

top1, top2 = st.columns([5, 1])

with top1:

    st.title("Healthcare Dashboard")

    st.write(
        f"Welcome, {st.session_state.username}"
    )

with top2:

    if st.button("Logout"):

        st.session_state.logged_in = False

        st.session_state.username = ""

        st.rerun()

# =========================================================
# LIVE MONITORING DASHBOARD
# =========================================================

if choice == "Live Monitoring Dashboard":

    st.title("Live Health Monitoring Dashboard")

    st.write(
        "Hybrid Wearable Sensor Data Analytics"
    )

    # ---------------- SIMULATED SENSOR VALUES ---------------- #

    heart_rate = random.randint(60, 120)
    spo2 = random.randint(92, 100)
    steps = random.randint(1000, 15000)
    sleep_hours = round(random.uniform(4, 9), 1)
    calories = random.randint(1500, 3500)
    temperature = round(random.uniform(36.0, 38.5), 1)

    # ---------------- METRICS ---------------- #

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Heart Rate",
            f"{heart_rate} BPM"
        )

    with col2:
        st.metric(
            "SpO2 Level",
            f"{spo2}%"
        )

    with col3:
        st.metric(
            "Daily Steps",
            steps
        )

    col4, col5, col6 = st.columns(3)

    with col4:
        st.metric(
            "Sleep Hours",
            f"{sleep_hours} hrs"
        )

    with col5:
        st.metric(
            "Calories Burned",
            calories
        )

    with col6:
        st.metric(
            "Temperature",
            f"{temperature} °C"
        )

    # ---------------- ALERTS ---------------- #

    st.subheader("Health Alerts")

    if heart_rate > 100:
        st.warning(
            "⚠ High Heart Rate Detected"
        )

    if spo2 < 95:
        st.error(
            "⚠ Low Oxygen Level Detected"
        )

    if sleep_hours < 6:
        st.warning(
            "⚠ Poor Sleep Quality"
        )

    if steps < 4000:
        st.info(
            "🚶 Increase Physical Activity"
        )

    if (
        heart_rate <= 100
        and spo2 >= 95
        and sleep_hours >= 6
        and steps >= 4000
    ):
        st.success(
            "✅ Health Status Normal"
        )

    # ---------------- HEALTH SCORE ---------------- #

    health_score = 100

    if heart_rate > 100:
        health_score -= 20

    if spo2 < 95:
        health_score -= 20

    if sleep_hours < 6:
        health_score -= 15

    if steps < 4000:
        health_score -= 10

    st.subheader("Overall Health Score")

    st.metric(
        "Health Score",
        health_score
    )

    # ---------------- CHARTS ---------------- #

    st.subheader("Live Health Trends")

    trend_data = pd.DataFrame({
        "Time": pd.date_range(
            start=datetime.now(),
            periods=10,
            freq="H"
        ),
        "Heart Rate": [
            random.randint(60, 120)
            for _ in range(10)
        ],
        "SpO2": [
            random.randint(92, 100)
            for _ in range(10)
        ],
        "Steps": [
            random.randint(1000, 15000)
            for _ in range(10)
        ]
    })

    st.line_chart(
        trend_data.set_index("Time")
    )

    # ---------------- BMI CALCULATOR ---------------- #

    st.subheader("BMI Calculator")

    c1, c2 = st.columns(2)

    with c1:
        height = st.number_input(
            "Height (meters)",
            min_value=0.0,
            format="%.2f"
        )

    with c2:
        weight = st.number_input(
            "Weight (kg)",
            min_value=0.0,
            format="%.2f"
        )

    if height > 0:

        bmi = weight / (height ** 2)

        st.metric(
            "BMI",
            round(bmi, 2)
        )

        if bmi < 18.5:
            st.info("Underweight")

        elif bmi < 25:
            st.success("Normal Weight")

        elif bmi < 30:
            st.warning("Overweight")

        else:
            st.error("Obesity")

# =========================================================
# HEALTHCARE ANALYSIS
# =========================================================

elif choice == "Healthcare Analysis":

    uploaded_file = st.file_uploader(
        "Upload wearable data CSV",
        type=["csv"]
    )

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
            "fatigue_level"
        ]

        missing_columns = [
            col for col in required_columns
            if col not in df.columns
        ]

        if missing_columns:

            st.error(
                f"Missing columns: {', '.join(missing_columns)}"
            )

        else:

            # ---------------- PREPROCESSING ---------------- #

            df["Timestamp"] = pd.to_datetime(
                df["Timestamp"],
                errors="coerce"
            )

            df = df.dropna().copy()

            df = df.sort_values(
                "Timestamp"
            ).reset_index(drop=True)

            df["hr_avg"] = df[
                "heart_rate"
            ].rolling(
                3,
                min_periods=1
            ).mean()

            # ---------------- HEALTH SCORE ---------------- #

            def calculate_health_score(row):

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

            df["health_score"] = df.apply(
                calculate_health_score,
                axis=1
            )

            # ---------------- RECOMMENDATION ---------------- #

            def recommend(row):

                rec = []

                if row["heart_rate"] > row["hr_avg"] + 10:
                    rec.append(
                        "High heart rate detected."
                    )

                if row["temperature"] > 37.5:
                    rec.append(
                        "Body temperature elevated."
                    )

                if row["respiration"] > 20:
                    rec.append(
                        "Respiration rate is high."
                    )

                if row["sleep_hours"] < 6:
                    rec.append(
                        "Sleep duration is low."
                    )

                if row["spo2"] < 95:
                    rec.append(
                        "Low oxygen level detected."
                    )

                if row["stress_level"] == "high":
                    rec.append(
                        "Stress level is high."
                    )

                if not rec:
                    rec.append(
                        "Health indicators stable."
                    )

                return " ".join(rec)

            df["recommendation"] = df.apply(
                recommend,
                axis=1
            )

            # ---------------- MACHINE LEARNING ---------------- #

            le = LabelEncoder()

            df["fatigue_encoded"] = le.fit_transform(
                df["fatigue_level"]
            )

            X = df[
                [
                    "heart_rate",
                    "temperature",
                    "respiration",
                    "steps",
                    "sleep_hours",
                    "spo2"
                ]
            ]

            y = df["fatigue_encoded"]

            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=0.2,
                random_state=42
            )

            model = RandomForestClassifier(
                random_state=42
            )

            model.fit(
                X_train,
                y_train
            )

            y_pred = model.predict(X_test)

            accuracy = accuracy_score(
                y_test,
                y_pred
            )

            latest = X.iloc[-1].values.reshape(1, -1)

            prediction = model.predict(latest)

            predicted_label = le.inverse_transform(
                prediction
            )[0]

            latest_row = df.iloc[-1]

            # ---------------- DASHBOARD ---------------- #

            st.subheader("Health Dashboard")

            d1, d2, d3, d4 = st.columns(4)

            d1.metric(
                "Heart Rate",
                int(latest_row["heart_rate"])
            )

            d2.metric(
                "SpO2",
                int(latest_row["spo2"])
            )

            d3.metric(
                "Health Score",
                int(latest_row["health_score"])
            )

            d4.metric(
                "Stress Level",
                latest_row["stress_level"]
            )

            # ---------------- AI PREDICTION ---------------- #

            st.subheader("AI Prediction")

            st.success(
                f"Predicted Fatigue Level: {predicted_label}"
            )

            # ---------------- RECOMMENDATION ---------------- #

            st.subheader("Latest Recommendation")

            st.info(
                latest_row["recommendation"]
            )

            # ---------------- MODEL ACCURACY ---------------- #

            st.subheader("Model Performance")

            st.metric(
                "Accuracy",
                f"{accuracy * 100:.2f}%"
            )

            # ---------------- CONFUSION MATRIX ---------------- #

            cm = confusion_matrix(
                y_test,
                y_pred
            )

            st.subheader("Confusion Matrix")

            st.dataframe(
                pd.DataFrame(cm),
                use_container_width=True
            )

            # ---------------- CLASSIFICATION REPORT ---------------- #

            report = classification_report(
                y_test,
                y_pred,
                output_dict=True
            )

            report_df = pd.DataFrame(
                report
            ).transpose()

            st.subheader(
                "Classification Report"
            )

            st.dataframe(
                report_df,
                use_container_width=True
            )

            # ---------------- FEATURE IMPORTANCE ---------------- #

            importance_df = pd.DataFrame({
                "Feature": X.columns,
                "Importance": model.feature_importances_
            })

            st.subheader(
                "Feature Importance"
            )

            st.bar_chart(
                importance_df.set_index(
                    "Feature"
                )
            )

            # ---------------- HEALTH TRENDS ---------------- #

            st.subheader("Health Trends")

            st.line_chart(
                df.set_index("Timestamp")[
                    [
                        "heart_rate",
                        "health_score"
                    ]
                ]
            )

            # ---------------- SAVE ANALYSIS ---------------- #

            if st.button("Save This Analysis"):

                save_health_record(
                    st.session_state.username,
                    latest_row,
                    predicted_label
                )

                st.success(
                    "Analysis Saved Successfully"
                )

            # ---------------- DOWNLOAD CSV ---------------- #

            csv = df.to_csv(
                index=False
            ).encode("utf-8")

            st.download_button(
                label="Download Analyzed Data",
                data=csv,
                file_name="health_analysis.csv",
                mime="text/csv"
            )

# =========================================================
# USER HISTORY
# =========================================================

elif choice == "User History":

    st.subheader("User History")

    history_df = get_user_history(
        st.session_state.username
    )

    if history_df.empty:

        st.info("No history found")

    else:

        st.dataframe(
            history_df,
            use_container_width=True
        )

# ---------------- FOOTER ---------------- #

st.subheader("Project Conclusion")

st.write(
    "This AI-powered healthcare system analyzes "
    "wearable sensor data, predicts fatigue levels "
    "using machine learning, provides personalized "
    "healthcare recommendations, and includes a "
    "live wearable monitoring dashboard using "
    "hybrid wearable sensor simulation."
)
```

