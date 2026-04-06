import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.title("Personalized Healthcare Recommendation System")

uploaded_file = st.file_uploader("Upload wearable data CSV", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    df['date'] = pd.to_datetime(df['date'])

    df = df.dropna()

    # Baselines
    df['steps_avg'] = df['steps'].rolling(7, min_periods=1).mean()
    df['sleep_avg'] = df['sleep_hours'].rolling(7, min_periods=1).mean()
    df['rhr_avg'] = df['resting_heart_rate'].rolling(7, min_periods=1).mean()

    def recommend(row):
        rec = []
        if row['sleep_hours'] < 6:
            rec.append("Sleep more tonight.")
        if row['steps'] < row['steps_avg']:
            rec.append("Increase your physical activity.")
        if row['resting_heart_rate'] > row['rhr_avg'] + 3:
            rec.append("Take rest, your recovery seems low.")
        if not rec:
            rec.append("You are doing great. Keep it up!")
        return " ".join(rec)

    df['recommendation'] = df.apply(recommend, axis=1)

    st.subheader("Data")
    st.dataframe(df)

    st.subheader("Steps Trend")
    st.line_chart(df.set_index('date')['steps'])

    st.subheader("Sleep Trend")
    st.line_chart(df.set_index('date')['sleep_hours'])

    st.subheader("Today's Recommendation")
    st.success(df.iloc[-1]['recommendation'])
