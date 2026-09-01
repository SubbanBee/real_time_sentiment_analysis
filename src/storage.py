import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = PROJECT_ROOT / "data" / "sentiment_records.db"


def get_connection():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(DB_PATH, check_same_thread=False)


def initialize_database():
    with get_connection() as connection:
        connection.execute("""
            CREATE TABLE IF NOT EXISTS analyzed_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                source TEXT NOT NULL,
                text TEXT NOT NULL,
                sentiment TEXT NOT NULL,
                confidence REAL,
                topic TEXT
            )
        """)
        connection.commit()


def save_prediction(source, text, sentiment, confidence, topic=""):
    initialize_database()

    with get_connection() as connection:
        connection.execute("""
            INSERT INTO analyzed_records
            (timestamp, source, text, sentiment, confidence, topic)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            datetime.now(timezone.utc).isoformat(),
            source,
            text,
            sentiment,
            float(confidence) if confidence is not None else 0.0,
            topic
        ))
        connection.commit()


def get_records():
    initialize_database()

    with get_connection() as connection:
        records = pd.read_sql_query("""
            SELECT id, timestamp, source, text, sentiment, confidence, topic
            FROM analyzed_records
            ORDER BY timestamp DESC
        """, connection)

    if not records.empty:
        records["timestamp"] = pd.to_datetime(
            records["timestamp"], errors="coerce", utc=True
        )

    return records


def clear_records():
    initialize_database()

    with get_connection() as connection:
        connection.execute("DELETE FROM analyzed_records")
        connection.commit()
