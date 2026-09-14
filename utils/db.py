# ============================================================
#  DATABASE HELPER FUNCTIONS
# ============================================================
# All SQLite operations live here so the different pages don't
# each repeat their own copy of the SQL.
# ============================================================

import sqlite3
import os
import pandas as pd

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "database", "database.db")


def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Creates the reports table if it doesn't already exist."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS reports (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            location TEXT,
            description TEXT,
            latitude REAL,
            longitude REAL,
            accuracy REAL,
            image TEXT,
            garbage_type TEXT,
            classification_mode TEXT,
            priority TEXT,
            status TEXT DEFAULT 'Unresolved',
            duplicate_count INTEGER DEFAULT 1,
            date TEXT
        )
    """)
    conn.commit()
    conn.close()


def get_all_reports():
    """Returns every report as a pandas DataFrame."""
    conn = get_connection()
    df = pd.read_sql_query("SELECT * FROM reports", conn)
    conn.close()
    return df


def get_reports_as_dicts():
    """Returns every report as a list of plain dicts (used for duplicate checking)."""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT id, latitude, longitude, date FROM reports")
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return rows


def insert_report(data):
    """Inserts a brand new (non-duplicate) report."""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO reports
            (location, description, latitude, longitude, accuracy, image,
             garbage_type, classification_mode, priority, status, duplicate_count, date)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
    """, (
        data["location"], data["description"], data["latitude"], data["longitude"],
        data.get("accuracy"), data["image"], data["garbage_type"],
        data["classification_mode"], data["priority"], data.get("status", "Unresolved"),
        data.get("duplicate_count", 1), data["date"],
    ))
    conn.commit()
    conn.close()


def increment_duplicate(report_id, new_priority=None):
    """Called when a NEW report turns out to be a duplicate of an existing spot."""
    conn = get_connection()
    cur = conn.cursor()
    if new_priority:
        cur.execute(
            "UPDATE reports SET duplicate_count = duplicate_count + 1, priority = ? WHERE id = ?",
            (new_priority, report_id),
        )
    else:
        cur.execute("UPDATE reports SET duplicate_count = duplicate_count + 1 WHERE id = ?", (report_id,))
    conn.commit()
    conn.close()


def mark_resolved(report_id):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("UPDATE reports SET status = 'Resolved' WHERE id = ?", (report_id,))
    conn.commit()
    conn.close()


def mark_unresolved(report_id):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("UPDATE reports SET status = 'Unresolved' WHERE id = ?", (report_id,))
    conn.commit()
    conn.close()
