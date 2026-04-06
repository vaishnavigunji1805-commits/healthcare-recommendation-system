import streamlit as st
import pandas as pd

st.set_page_config(page_title="AI Healthcare System", layout="wide")

st.title("Recommendation System")

uploaded_file = st.file_uploader("Upload wearable data CSV", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df = df.dropna()

    df['hr_avg'] = df['heart_rate'].rolling(3, min_periods=1).mean()
    df['temp_avg'] = df['temperature'].rolling(3, min_periods=1).mean()
    df['resp_avg'] = df['respiration'].rolling(3, min_periods=1).mean()

    def health_score(row):
        score = 100

        if row['heart_rate'] > row['hr_avg'] + 10:
            score -= 20
        if row['temperature'] > 37.5:
            score -= 20
        if row['respiration'] > 20:
            score -= 15
        if row['sleep_hours'] < 6:
            score -= 15
        if row['spo2'] < 95:
            score -= 10

        return max(score, 0)

    def recommend(row):
        rec = []

        if row['heart_rate'] > row['hr_avg'] + 10:
            rec.append("High heart rate detected. Take some rest.")
        if row['temperature'] > 37.5:
            rec.append("Body temperature is high. Monitor your condition.")
        if row['respiration'] > 20:
            rec.append("Respiration rate is elevated. Avoid overexertion.")
        if row['sleep_hours'] < 6:
            rec.append("Sleep duration is low. Try to rest more tonight.")
        if row['spo2'] < 95:
            rec.append("Oxygen level is slightly low. Observe carefully.")
        if row['steps'] < 4000:
            rec.append("Physical activity is low. Try a short walk.")
        if not rec:
            rec.append("Your health indicators look stable today.")

        return " ".join(rec)

    df['health_score'] = df.apply(health_score, axis=1)
    df['recommendation'] = df.apply(recommend, axis=1)

    col1, col2, col3 = st.columns(3)
    col1.metric("Heart Rate", int(df.iloc[-1]['heart_rate']))
    col2.metric("SpO2", int(df.iloc[-1]['spo2']))
    col3.metric("Health Score", int(df.iloc[-1]['health_score']))

    st.subheader("Health Trends")
    st.line_chart(
        df.set_index('Timestamp')[['heart_rate', 'temperature', 'respiration', 'health_score']]
    )

    st.subheader("Latest Recommendation")
    st.success(df.iloc[-1]['recommendation'])

    st.subheader("Full Dataset")
    st.dataframe(df)
    from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

# Encode target
le = LabelEncoder()
df['fatigue_encoded'] = le.fit_transform(df['fatigue_level'])

# Features
X = df[['heart_rate', 'temperature', 'respiration', 'steps', 'sleep_hours', 'spo2']]
y = df['fatigue_encoded']

# Train model
model = RandomForestClassifier()
model.fit(X, y)

# Predict latest row
latest = X.iloc[-1].values.reshape(1, -1)
prediction = model.predict(latest)
predicted_label = le.inverse_transform(prediction)[0]
st.subheader("🤖 AI Prediction")
st.info(f"Predicted Fatigue Level: {predicted_label}")
