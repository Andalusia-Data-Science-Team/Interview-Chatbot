import sqlite3
from datetime import datetime
import config


def get_db():
    conn = sqlite3.connect(config.DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("""CREATE TABLE IF NOT EXISTS interviews(
        token TEXT PRIMARY KEY, name TEXT, email TEXT,
        status TEXT DEFAULT 'pending',
        created_at TEXT, expires_at TEXT,
        profile TEXT, questions TEXT, answers TEXT,
        result_path TEXT, score REAL, recommendation TEXT
    )""")
    return conn


def create_interview(token, name, email, expires_at):
    conn = get_db()
    conn.execute(
        "INSERT INTO interviews (token,name,email,created_at,expires_at) VALUES (?,?,?,?,?)",
        (token, name, email, datetime.now().isoformat(), expires_at),
    )
    conn.commit()
    conn.close()


def get_interview(token):
    conn = get_db()
    row = conn.execute("SELECT * FROM interviews WHERE token=?", (token,)).fetchone()
    conn.close()
    return row


def update_interview(token, **fields):
    conn = get_db()
    sets = ", ".join(f"{k}=?" for k in fields)
    conn.execute(f"UPDATE interviews SET {sets} WHERE token=?", (*fields.values(), token))
    conn.commit()
    conn.close()


def all_interviews():
    conn = get_db()
    rows = conn.execute("SELECT * FROM interviews ORDER BY created_at DESC").fetchall()
    conn.close()
    return rows
