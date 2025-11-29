"""Database initialization and connection"""

import sqlite3
import json
from datetime import datetime
from typing import Dict, List, Optional
import os

DB_PATH = "ghia.db"


def get_connection():
    """Get database connection"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


# def init_db():
#     """Initialize database tables"""
#     conn = get_connection()
#     cursor = conn.cursor()

#     # Main intake records table
#     cursor.execute("""
#     CREATE TABLE IF NOT EXISTS intake_records (
#         id INTEGER PRIMARY KEY AUTOINCREMENT,
#         created_at TEXT NOT NULL,

#         -- Raw data
#         raw_transcript TEXT,
#         language_detected TEXT DEFAULT 'hi',

#         -- Structured intake
#         chief_complaint TEXT,
#         symptoms_json TEXT,
#         duration TEXT,
#         severity TEXT,
#         associated_symptoms_json TEXT,
#         medical_history TEXT,
#         age_group TEXT,
#         gender TEXT,

#         -- Risk & Triage
#         risk_level TEXT NOT NULL,

#         -- Agent decisions trace
#         agent_decisions_json TEXT,

#         -- Summaries
#         summary_english TEXT,
#         summary_hindi TEXT,

#         -- Follow-up
#         follow_up_questions_json TEXT,
#         recommended_action TEXT
#     )
#     """)

#     # Conversation history for multi-turn
#     cursor.execute("""
#     CREATE TABLE IF NOT EXISTS conversation_history (
#         id INTEGER PRIMARY KEY AUTOINCREMENT,
#         session_id TEXT NOT NULL,
#         turn_number INTEGER NOT NULL,
#         role TEXT NOT NULL,
#         content TEXT NOT NULL,
#         timestamp TEXT NOT NULL
#     )
#     """)

#     conn.commit()
#     conn.close()


def init_db():
    """Initialize the SQLite database with required tables"""
    conn = get_connection()
    cursor = conn.cursor()

    # Create intake_records table with all required columns
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS intake_records (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id TEXT,
        patient_id TEXT,
        phc_id TEXT,
        raw_transcript TEXT,
        normalized_text TEXT,
        chief_complaint TEXT,
        symptoms_json TEXT,
        duration TEXT,
        severity TEXT,
        risk_level TEXT,
        red_flags_json TEXT,
        summary_english TEXT,
        summary_hindi TEXT,
        recommended_action TEXT,
        icd11_codes_json TEXT,
        treatment_json TEXT,
        referral_json TEXT,
        language_detected TEXT,
        age_group TEXT,
        gender TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)

    # Check if phc_id column exists, add if missing (for existing databases)
    cursor.execute("PRAGMA table_info(intake_records)")
    columns = [col[1] for col in cursor.fetchall()]

    if "phc_id" not in columns:
        cursor.execute("ALTER TABLE intake_records ADD COLUMN phc_id TEXT")
        print("✅ Added phc_id column to intake_records")

    if "age_group" not in columns:
        cursor.execute("ALTER TABLE intake_records ADD COLUMN age_group TEXT")

    if "gender" not in columns:
        cursor.execute("ALTER TABLE intake_records ADD COLUMN gender TEXT")

    conn.commit()
    conn.close()
    print("✅ Database initialized")

def get_db():
    """
    Dependency injection for database connection.
    Returns a simple database wrapper for use with FastAPI Depends().
    """
    conn = get_connection()
    try:
        yield DatabaseWrapper(conn)
    finally:
        conn.close()


class DatabaseWrapper:
    """Simple wrapper to provide fetch_all and fetch_one methods"""
    
    def __init__(self, conn):
        self.conn = conn
        self.cursor = conn.cursor()
    
    def fetch_all(self, query: str, params: tuple = ()) -> List[Dict]:
        """Execute query and return all results as list of dicts"""
        self.cursor.execute(query, params)
        rows = self.cursor.fetchall()
        return [dict(row) for row in rows]
    
    def fetch_one(self, query: str, params: tuple = ()) -> Optional[Dict]:
        """Execute query and return single result as dict"""
        self.cursor.execute(query, params)
        row = self.cursor.fetchone()
        return dict(row) if row else None
    
    def execute(self, query: str, params: tuple = ()):
        """Execute a query without returning results"""
        self.cursor.execute(query, params)
        self.conn.commit()
        return self.cursor.lastrowid

def save_intake_record(
    raw_transcript: str,
    language_detected: str,
    intake_data: Dict,
    risk_level: str,
    agent_decisions: List[Dict],
    summary_english: str,
    summary_hindi: str,
    follow_up_questions: List[str],
    recommended_action: str,
) -> int:
    """Save a complete intake record and return its ID"""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
    INSERT INTO intake_records (
        created_at, raw_transcript, language_detected,
        chief_complaint, symptoms_json, duration, severity,
        associated_symptoms_json, medical_history, age_group, gender,
        risk_level, agent_decisions_json,
        summary_english, summary_hindi,
        follow_up_questions_json, recommended_action
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """,
        (
            datetime.utcnow().isoformat(),
            raw_transcript,
            language_detected,
            intake_data.get("chief_complaint"),
            json.dumps(intake_data.get("symptoms", []), ensure_ascii=False),
            intake_data.get("duration"),
            intake_data.get("severity"),
            json.dumps(intake_data.get("associated_symptoms", []), ensure_ascii=False),
            intake_data.get("medical_history"),
            intake_data.get("age_group"),
            intake_data.get("gender"),
            risk_level,
            json.dumps(agent_decisions, ensure_ascii=False),
            summary_english,
            summary_hindi,
            json.dumps(follow_up_questions, ensure_ascii=False),
            recommended_action,
        ),
    )

    record_id = cursor.lastrowid
    conn.commit()
    conn.close()

    return record_id


def get_all_intakes() -> List[Dict]:
    """Get all intake records for dashboard"""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
    SELECT id, created_at, chief_complaint, risk_level,
           summary_english, summary_hindi, symptoms_json,
           associated_symptoms_json, duration, severity,
           recommended_action
    FROM intake_records
    ORDER BY created_at DESC
    """)

    rows = cursor.fetchall()
    conn.close()

    results = []
    for row in rows:
        results.append(
            {
                "id": row["id"],
                "created_at": row["created_at"],
                "chief_complaint": row["chief_complaint"],
                "risk_level": row["risk_level"],
                "summary_english": row["summary_english"],
                "summary_hindi": row["summary_hindi"],
                "symptoms": json.loads(row["symptoms_json"])
                if row["symptoms_json"]
                else [],
                "associated_symptoms": json.loads(row["associated_symptoms_json"])
                if row["associated_symptoms_json"]
                else [],
                "duration": row["duration"],
                "severity": row["severity"],
                "recommended_action": row["recommended_action"],
            }
        )

    return results


def get_intake_by_id(record_id: int) -> Optional[Dict]:
    """Get a single intake record by ID"""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM intake_records WHERE id = ?", (record_id,))
    row = cursor.fetchone()
    conn.close()

    if not row:
        return None

    return {
        "id": row["id"],
        "created_at": row["created_at"],
        "raw_transcript": row["raw_transcript"],
        "language_detected": row["language_detected"],
        "chief_complaint": row["chief_complaint"],
        "symptoms": json.loads(row["symptoms_json"]) if row["symptoms_json"] else [],
        "duration": row["duration"],
        "severity": row["severity"],
        "associated_symptoms": json.loads(row["associated_symptoms_json"])
        if row["associated_symptoms_json"]
        else [],
        "risk_level": row["risk_level"],
        "agent_decisions": json.loads(row["agent_decisions_json"])
        if row["agent_decisions_json"]
        else [],
        "summary_english": row["summary_english"],
        "summary_hindi": row["summary_hindi"],
        "follow_up_questions": json.loads(row["follow_up_questions_json"])
        if row["follow_up_questions_json"]
        else [],
        "recommended_action": row["recommended_action"],
    }
