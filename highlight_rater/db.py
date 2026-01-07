"""
Database module for highlight ratings.

SQLite-based storage for raters, sessions, and ratings.
"""

import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Optional

# Database file location
DB_PATH = Path(__file__).parent / "ratings.db"


def get_connection() -> sqlite3.Connection:
    """Get a database connection with row factory."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


@contextmanager
def get_db():
    """Context manager for database connections."""
    conn = get_connection()
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    """Initialize the database schema."""
    with get_db() as conn:
        conn.executescript("""
            -- Raters (human evaluators)
            CREATE TABLE IF NOT EXISTS raters (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL UNIQUE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            -- Rating sessions (batches of ratings)
            CREATE TABLE IF NOT EXISTS sessions (
                id INTEGER PRIMARY KEY,
                rater_id INTEGER NOT NULL REFERENCES raters(id),
                chapters_file TEXT NOT NULL,
                video_id TEXT,
                started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                completed_at TIMESTAMP
            );

            -- Individual event ratings
            CREATE TABLE IF NOT EXISTS ratings (
                id INTEGER PRIMARY KEY,
                session_id INTEGER NOT NULL REFERENCES sessions(id),
                event_id INTEGER NOT NULL,
                game_id INTEGER NOT NULL,
                rating INTEGER NOT NULL CHECK (rating >= 0 AND rating <= 3),
                response_time_ms INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(session_id, event_id)
            );

            -- Algorithm scores for comparison
            CREATE TABLE IF NOT EXISTS algorithm_scores (
                id INTEGER PRIMARY KEY,
                event_id INTEGER NOT NULL,
                game_id INTEGER NOT NULL,
                chapters_file TEXT NOT NULL,
                delta REAL NOT NULL,
                abs_delta REAL NOT NULL,
                event_type TEXT,
                positions TEXT,
                event_time REAL,
                UNIQUE(event_id, chapters_file)
            );

            -- Create indexes for common queries
            CREATE INDEX IF NOT EXISTS idx_ratings_session ON ratings(session_id);
            CREATE INDEX IF NOT EXISTS idx_ratings_event ON ratings(event_id);
            CREATE INDEX IF NOT EXISTS idx_sessions_rater ON sessions(rater_id);
            CREATE INDEX IF NOT EXISTS idx_algo_scores_game ON algorithm_scores(game_id);
        """)


# ============== Rater CRUD ==============

def create_rater(name: str) -> int:
    """Create a new rater, return their ID."""
    with get_db() as conn:
        cursor = conn.execute(
            "INSERT INTO raters (name) VALUES (?)",
            (name,)
        )
        return cursor.lastrowid


def get_rater(rater_id: int) -> Optional[dict]:
    """Get a rater by ID."""
    with get_db() as conn:
        row = conn.execute(
            "SELECT * FROM raters WHERE id = ?",
            (rater_id,)
        ).fetchone()
        return dict(row) if row else None


def get_rater_by_name(name: str) -> Optional[dict]:
    """Get a rater by name."""
    with get_db() as conn:
        row = conn.execute(
            "SELECT * FROM raters WHERE name = ?",
            (name,)
        ).fetchone()
        return dict(row) if row else None


def get_or_create_rater(name: str) -> int:
    """Get existing rater by name or create new one."""
    rater = get_rater_by_name(name)
    if rater:
        return rater['id']
    return create_rater(name)


def list_raters() -> list[dict]:
    """List all raters."""
    with get_db() as conn:
        rows = conn.execute("SELECT * FROM raters ORDER BY name").fetchall()
        return [dict(row) for row in rows]


# ============== Session CRUD ==============

def create_session(rater_id: int, chapters_file: str, video_id: str = None) -> int:
    """Create a new rating session."""
    with get_db() as conn:
        cursor = conn.execute(
            "INSERT INTO sessions (rater_id, chapters_file, video_id) VALUES (?, ?, ?)",
            (rater_id, chapters_file, video_id)
        )
        return cursor.lastrowid


def get_session(session_id: int) -> Optional[dict]:
    """Get a session by ID."""
    with get_db() as conn:
        row = conn.execute(
            "SELECT * FROM sessions WHERE id = ?",
            (session_id,)
        ).fetchone()
        return dict(row) if row else None


def complete_session(session_id: int):
    """Mark a session as completed."""
    with get_db() as conn:
        conn.execute(
            "UPDATE sessions SET completed_at = ? WHERE id = ?",
            (datetime.now().isoformat(), session_id)
        )


def list_sessions(rater_id: int = None) -> list[dict]:
    """List sessions, optionally filtered by rater."""
    with get_db() as conn:
        if rater_id:
            rows = conn.execute(
                """SELECT s.*, r.name as rater_name
                   FROM sessions s JOIN raters r ON s.rater_id = r.id
                   WHERE rater_id = ? ORDER BY started_at DESC""",
                (rater_id,)
            ).fetchall()
        else:
            rows = conn.execute(
                """SELECT s.*, r.name as rater_name
                   FROM sessions s JOIN raters r ON s.rater_id = r.id
                   ORDER BY started_at DESC"""
            ).fetchall()
        return [dict(row) for row in rows]


def get_incomplete_session(rater_id: int, chapters_file: str) -> Optional[dict]:
    """Find an incomplete session for a rater and chapters file."""
    with get_db() as conn:
        row = conn.execute(
            """SELECT * FROM sessions
               WHERE rater_id = ? AND chapters_file = ? AND completed_at IS NULL
               ORDER BY started_at DESC LIMIT 1""",
            (rater_id, chapters_file)
        ).fetchone()
        return dict(row) if row else None


# ============== Rating CRUD ==============

def create_rating(session_id: int, event_id: int, game_id: int,
                  rating: int, response_time_ms: int = None) -> int:
    """Create a new rating."""
    with get_db() as conn:
        cursor = conn.execute(
            """INSERT INTO ratings (session_id, event_id, game_id, rating, response_time_ms)
               VALUES (?, ?, ?, ?, ?)""",
            (session_id, event_id, game_id, rating, response_time_ms)
        )
        return cursor.lastrowid


def get_ratings_for_session(session_id: int) -> list[dict]:
    """Get all ratings for a session."""
    with get_db() as conn:
        rows = conn.execute(
            "SELECT * FROM ratings WHERE session_id = ? ORDER BY created_at",
            (session_id,)
        ).fetchall()
        return [dict(row) for row in rows]


def get_rated_event_ids(session_id: int) -> set[int]:
    """Get set of event IDs already rated in a session."""
    with get_db() as conn:
        rows = conn.execute(
            "SELECT event_id FROM ratings WHERE session_id = ?",
            (session_id,)
        ).fetchall()
        return {row['event_id'] for row in rows}


def get_rating_count(session_id: int) -> int:
    """Get count of ratings in a session."""
    with get_db() as conn:
        row = conn.execute(
            "SELECT COUNT(*) as count FROM ratings WHERE session_id = ?",
            (session_id,)
        ).fetchone()
        return row['count']


# ============== Algorithm Scores ==============

def upsert_algorithm_score(event_id: int, game_id: int, chapters_file: str,
                           delta: float, event_type: str, positions: list,
                           event_time: float):
    """Insert or update an algorithm score."""
    with get_db() as conn:
        conn.execute(
            """INSERT INTO algorithm_scores
               (event_id, game_id, chapters_file, delta, abs_delta, event_type, positions, event_time)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(event_id, chapters_file) DO UPDATE SET
               delta = excluded.delta, abs_delta = excluded.abs_delta,
               event_type = excluded.event_type, positions = excluded.positions,
               event_time = excluded.event_time""",
            (event_id, game_id, chapters_file, delta, abs(delta),
             event_type, ','.join(map(str, positions)) if positions else '', event_time)
        )


def get_algorithm_scores(chapters_file: str, game_id: int = None) -> list[dict]:
    """Get algorithm scores, optionally filtered by game."""
    with get_db() as conn:
        if game_id:
            rows = conn.execute(
                """SELECT * FROM algorithm_scores
                   WHERE chapters_file = ? AND game_id = ?
                   ORDER BY abs_delta DESC""",
                (chapters_file, game_id)
            ).fetchall()
        else:
            rows = conn.execute(
                """SELECT * FROM algorithm_scores
                   WHERE chapters_file = ?
                   ORDER BY abs_delta DESC""",
                (chapters_file,)
            ).fetchall()
        return [dict(row) for row in rows]


def load_events_from_chapters(chapters_file: str, chapters_data: dict):
    """Load all player events from chapters data into algorithm_scores table."""
    for chapter in chapters_data.get('chapters', []):
        game_id = chapter.get('game_id')
        for event in chapter.get('player_events', []):
            upsert_algorithm_score(
                event_id=event['id'],
                game_id=game_id,
                chapters_file=chapters_file,
                delta=event['delta'],
                event_type=event.get('type', 'unknown'),
                positions=event.get('positions', []),
                event_time=event.get('time', 0)
            )


# Initialize database on import
init_db()
