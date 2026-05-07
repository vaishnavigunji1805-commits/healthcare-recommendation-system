import sqlite3
import hashlib
from datetime import datetime

import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

# ---------------- PAGE CONFIG ---------------- #

st.set_page_config(
    page_title="AI Healthcare System",
    layout="wide"
)

# ---------------- CUSTOM CSS ---------------- #

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
}

.stAlert {
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)

# ---------------- DATABASE ---------------- #

def get_connection():
    return sqlite3.connect(
        "healthcare_app.db",
        check_same_thread=False
    )

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

# ---------------- PASSWORD HASH ---------------- #

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

# ---------------- USER FUNCTIONS ---------------- #

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

def login_user(username, password):

    hashed_password = hash_password(password)

    cursor.execute(
        "SELECT * FROM users WHERE username=? AND password=?",
        (username, hashed_password)
    )

    return cursor.fetchone()

def save_health_record(username, latest_row, predicted_label):

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
        "saved_at",
        "heart_rate",
        "spo2",
        "health_score",
        "stress_level",
        "fatigue_prediction",
        "recommendation"
    ]

    return pd.DataFrame(rows, columns=columns)

# ---------------- SESSION ---------------- #

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "username" not in st.session_state:
    st.session_state.username = ""

if "auth_mode" not in st.session_state:
    st.session_state.auth_mode = "login"

# ---------------- AUTH PAGE ---------------- #

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

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    # ---------- SIGNUP ---------- #

    if st.session_state.auth_mode == "signup":

        if st.button("Create Account"):

            if not username or not password:
                st.error("Please enter username and password")

            else:

                created = signup_user(username, password)

                if created:
                    st.success("Account created successfully")

                    st.session_state.auth_mode = "login"

                else:
                    st.error("Username already exists")

    # ---------- LOGIN ---------- #

    else:

        if st.button("Login Now"):

            user = login_user(username, password)

            if user:

                st.session_state.logged_in = True
                st.session_state.username = username

                st.rerun()

            else:
                st.error("Invalid credentials")

# ---------------- STOP IF NOT LOGGED IN ---------------- #

if not st.session_state.logged_in:
    auth_page()
    st.stop()

# ---------------- HEADER ---------------- #

top1, top2 = st.columns([5, 1])

with top1:
    st.title("Healthcare Dashboard")
    st.write(f"Welcome, {st.session_state.username}")

with top2:
    if st.button("Logout"):

        st.session_state.logged_in = False
        st.session_state.username = ""

        st.rerun()

# ---------------- FILE UPLOAD ---------------- #

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

        df["hr_avg"] = df["heart_rate"].rolling(
            3,
            min_periods=1
        ).mean()

        # ---------------- HEALTH SCORE ---------------- #

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

        df["health_score"] = df.apply(
            health_score,
            axis=1
        )

        # ---------------- RECOMMENDATION ---------------- #

        def recommend(row):

            rec = []

            if row["heart_rate"] > row["hr_avg"] + 10:
                rec.append("High heart rate detected.")

            if row["temperature"] > 37.5:
                rec.append("Body temperature elevated.")

            if row["respiration"] > 20:
                rec.append("Respiration rate is high.")

            if row["sleep_hours"] < 6:
                rec.append("Sleep duration is low.")

            if row["spo2"] < 95:
                rec.append("Low oxygen level detected.")

            if row["stress_level"] == "high":
                rec.append("Stress level is high.")

            if not rec:
                rec.append("Health indicators stable.")

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

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        accuracy = accuracy_score(
            y_test,
            y_pred
        )

        cm = confusion_matrix(
            y_test,
            y_pred
        )

        # ---------------- PREDICTION ---------------- #

        latest = X.iloc[-1].values.reshape(1, -1)

        prediction = model.predict(latest)

        predicted_label = le.inverse_transform(
            prediction
        )[0]

        latest_row = df.iloc[-1]

        # ---------------- MODEL PERFORMANCE ---------------- #

        st.subheader("Model Performance")

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Model Accuracy",
                f"{accuracy * 100:.2f}%"
            )

        with col2:
            st.metric(
                "Training Samples",
                len(X_train)
            )

        st.write("Confusion Matrix")
        st.dataframe(cm)
        # ---------------- FEATURE IMPORTANCE ---------------- #

st.subheader("Feature Importance")

importance_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
})

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)

st.dataframe(importance_df)

st.bar_chart(
    importance_df.set_index("Feature")
)

        # ---------------- DASHBOARD ---------------- #

        st.subheader("Dashboard")

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

        st.info(
            f"Predicted Fatigue Level: {predicted_label}"
        )

        # ---------------- RECOMMENDATION ---------------- #

        st.subheader("Latest Recommendation")

        st.success(
            latest_row["recommendation"]
        )

        # ---------------- HEALTH TRENDS ---------------- #

        st.subheader("Health Trends")

        st.line_chart(
            df.set_index("Timestamp")[
                ["heart_rate", "health_score"]
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
                "Analysis saved successfully"
            )

        # ---------------- DOWNLOAD CSV ---------------- #

        csv = df.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            label="Download analyzed data",
            data=csv,
            file_name="health_analysis.csv",
            mime="text/csv"
        )

# ---------------- USER HISTORY ---------------- #

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

# ---------------- CONCLUSION ---------------- #

st.subheader("Conclusion")

st.write(
    "This system analyzes wearable sensor data, "
    "predicts fatigue level using machine learning, "
    "stores history in a database, and provides "
    "personalized healthcare recommendations."
)

