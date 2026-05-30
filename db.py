import sqlite3

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
