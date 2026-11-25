"""
SQLite Persistence Layer for Eye Image Analysis & Diabetes Risk Screening.

Stores prediction history cleanly in `backend/eye_predictions.db` with
separate tables for eye image analyses and tabular diabetes risk screenings.
"""

import json
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone

DB_PATH = os.path.join(os.path.dirname(__file__), "eye_predictions.db")


@contextmanager
def get_connection(db_path: str = DB_PATH):
    """Yield a SQLite connection with Row factory and commit/close handling."""
    conn = sqlite3.connect(db_path, timeout=10.0)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db(db_path: str = DB_PATH) -> None:
    """Create database tables if they do not yet exist."""
    with get_connection(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS eye_predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                filename TEXT NOT NULL,
                file_size_bytes INTEGER NOT NULL,
                image_width INTEGER,
                image_height INTEGER,
                analysis_mode TEXT NOT NULL,
                detected_modality TEXT NOT NULL,
                quality_status TEXT NOT NULL,
                quality_score REAL,
                sharpness_score REAL,
                brightness_score REAL,
                prediction_label TEXT NOT NULL,
                confidence_pct REAL,
                class_probabilities_json TEXT,
                extracted_features_json TEXT,
                model_name TEXT,
                disclaimer TEXT NOT NULL,
                user_role TEXT
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS tabular_diabetes_predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                pregnancies REAL,
                glucose REAL,
                blood_pressure REAL,
                skin_thickness REAL,
                insulin REAL,
                bmi REAL,
                dpf REAL,
                age REAL,
                prediction_label TEXT NOT NULL,
                probability_pct REAL NOT NULL,
                risk_tier TEXT NOT NULL,
                model_name TEXT
            )
            """
        )


def save_eye_prediction(record: dict, db_path: str = DB_PATH) -> int:
    """Persist an eye analysis result row and return its inserted ID."""
    init_db(db_path)
    created_at = record.get("created_at") or datetime.now(timezone.utc).isoformat()
    with get_connection(db_path) as conn:
        cur = conn.execute(
            """
            INSERT INTO eye_predictions (
                created_at, filename, file_size_bytes, image_width, image_height,
                analysis_mode, detected_modality, quality_status, quality_score,
                sharpness_score, brightness_score, prediction_label, confidence_pct,
                class_probabilities_json, extracted_features_json, model_name,
                disclaimer, user_role
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                created_at,
                record.get("filename", "upload"),
                int(record.get("file_size_bytes", 0)),
                record.get("image_width"),
                record.get("image_height"),
                record.get("analysis_mode", "auto"),
                record.get("detected_modality", "unknown"),
                record.get("quality_status", "unknown"),
                record.get("quality_score"),
                record.get("sharpness_score"),
                record.get("brightness_score"),
                record.get("prediction_label", "Unclassified"),
                record.get("confidence_pct"),
                json.dumps(record.get("class_probabilities") or {}),
                json.dumps(record.get("extracted_features") or {}),
                record.get("model_name", ""),
                record.get("disclaimer", ""),
                record.get("user_role", "Clinician"),
            ),
        )
        return int(cur.lastrowid)


def get_eye_history(limit: int = 30, db_path: str = DB_PATH) -> list[dict]:
    """Retrieve most recent eye image analysis history entries."""
    init_db(db_path)
    with get_connection(db_path) as conn:
        rows = conn.execute(
            """
            SELECT * FROM eye_predictions
            ORDER BY id DESC
            LIMIT ?
            """,
            (int(limit),),
        ).fetchall()
    results = []
    for row in rows:
        item = dict(row)
        item["class_probabilities"] = json.loads(item.pop("class_probabilities_json") or "{}")
        item["extracted_features"] = json.loads(item.pop("extracted_features_json") or "{}")
        results.append(item)
    return results


def delete_eye_prediction(record_id: int, db_path: str = DB_PATH) -> bool:
    """Delete a specific prediction history entry by ID."""
    init_db(db_path)
    with get_connection(db_path) as conn:
        cur = conn.execute("DELETE FROM eye_predictions WHERE id = ?", (int(record_id),))
        return cur.rowcount > 0


def clear_eye_history(db_path: str = DB_PATH) -> int:
    """Clear all saved eye prediction history entries."""
    init_db(db_path)
    with get_connection(db_path) as conn:
        cur = conn.execute("DELETE FROM eye_predictions")
        return int(cur.rowcount)


def save_diabetes_prediction(record: dict, db_path: str = DB_PATH) -> int:
    """Persist a tabular Pima diabetes prediction row and return its ID."""
    init_db(db_path)
    created_at = record.get("created_at") or datetime.now(timezone.utc).isoformat()
    with get_connection(db_path) as conn:
        cur = conn.execute(
            """
            INSERT INTO tabular_diabetes_predictions (
                created_at, pregnancies, glucose, blood_pressure, skin_thickness,
                insulin, bmi, dpf, age, prediction_label, probability_pct,
                risk_tier, model_name
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                created_at,
                record.get("pregnancies"),
                record.get("glucose"),
                record.get("blood_pressure"),
                record.get("skin_thickness"),
                record.get("insulin"),
                record.get("bmi"),
                record.get("dpf"),
                record.get("age"),
                record.get("prediction_label", "Unknown"),
                float(record.get("probability_pct", 0.0)),
                record.get("risk_tier", "Low"),
                record.get("model_name", ""),
            ),
        )
        return int(cur.lastrowid)


def get_diabetes_history(limit: int = 20, db_path: str = DB_PATH) -> list[dict]:
    """Retrieve most recent tabular diabetes screening predictions."""
    init_db(db_path)
    with get_connection(db_path) as conn:
        rows = conn.execute(
            """
            SELECT * FROM tabular_diabetes_predictions
            ORDER BY id DESC
            LIMIT ?
            """,
            (int(limit),),
        ).fetchall()
    return [dict(row) for row in rows]
