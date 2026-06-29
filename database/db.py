"""Base de données SQLite AquaReco - utilisateurs, préférences, favoris."""
from __future__ import annotations

import sqlite3
from pathlib import Path

import bcrypt

DB_PATH = Path(__file__).parent / "aquareco.db"

_DDL = """
CREATE TABLE IF NOT EXISTS users (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  email TEXT UNIQUE NOT NULL,
  password_hash TEXT NOT NULL,
  profil TEXT NOT NULL,
  date_creation TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS preferences (
  user_id INTEGER PRIMARY KEY,
  distance_max INTEGER DEFAULT 50,
  type_eau_pref TEXT DEFAULT 'tous',
  score_min TEXT DEFAULT 'tous',
  FOREIGN KEY (user_id) REFERENCES users(id)
);

CREATE TABLE IF NOT EXISTS favoris (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  user_id INTEGER NOT NULL,
  code_site TEXT NOT NULL,
  date_ajout TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  UNIQUE(user_id, code_site),
  FOREIGN KEY (user_id) REFERENCES users(id)
);
"""


def _conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """Crée les tables si elles n'existent pas encore."""
    with _conn() as conn:
        conn.executescript(_DDL)
        existing_cols = {row["name"] for row in conn.execute("PRAGMA table_info(preferences)")}
        for col, coltype in [("ville_pref", "TEXT"), ("lat_pref", "REAL"), ("lon_pref", "REAL")]:
            if col not in existing_cols:
                conn.execute(f"ALTER TABLE preferences ADD COLUMN {col} {coltype}")


# ── Utilisateurs ──────────────────────────────────────────────────────────────

def create_user(email: str, password: str, profil: str) -> int | None:
    """Crée un compte. Retourne l'id généré ou None si l'email est déjà utilisé."""
    pw_hash = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
    try:
        with _conn() as conn:
            cur = conn.execute(
                "INSERT INTO users (email, password_hash, profil) VALUES (?, ?, ?)",
                (email.lower().strip(), pw_hash, profil),
            )
            user_id = cur.lastrowid
            conn.execute("INSERT INTO preferences (user_id) VALUES (?)", (user_id,))
            return user_id
    except sqlite3.IntegrityError:
        return None


def authenticate_user(email: str, password: str) -> dict | None:
    """Vérifie les identifiants. Retourne le dict utilisateur ou None."""
    with _conn() as conn:
        row = conn.execute(
            "SELECT * FROM users WHERE email = ?",
            (email.lower().strip(),),
        ).fetchone()
    if row is None:
        return None
    if bcrypt.checkpw(password.encode(), row["password_hash"].encode()):
        return dict(row)
    return None


# ── Préférences ───────────────────────────────────────────────────────────────

def get_preferences(user_id: int) -> dict:
    """Retourne les préférences de l'utilisateur (avec valeurs par défaut)."""
    with _conn() as conn:
        row = conn.execute(
            "SELECT * FROM preferences WHERE user_id = ?", (user_id,)
        ).fetchone()
    if row is None:
        return {
            "user_id": user_id, "distance_max": 50, "type_eau_pref": "tous", "score_min": "tous",
            "ville_pref": None, "lat_pref": None, "lon_pref": None,
        }
    return dict(row)


def save_location_pref(user_id: int, ville: str, lat: float, lon: float) -> None:
    """Sauvegarde la dernière ville recherchée comme ville préférée (localisation par défaut)."""
    with _conn() as conn:
        conn.execute(
            """INSERT INTO preferences (user_id, ville_pref, lat_pref, lon_pref)
               VALUES (?, ?, ?, ?)
               ON CONFLICT(user_id) DO UPDATE SET
                 ville_pref = excluded.ville_pref,
                 lat_pref = excluded.lat_pref,
                 lon_pref = excluded.lon_pref""",
            (user_id, ville, lat, lon),
        )


def save_preferences(user_id: int, distance_max: int, type_eau_pref: str, score_min: str) -> None:
    """Crée ou met à jour les préférences."""
    with _conn() as conn:
        conn.execute(
            """INSERT INTO preferences (user_id, distance_max, type_eau_pref, score_min)
               VALUES (?, ?, ?, ?)
               ON CONFLICT(user_id) DO UPDATE SET
                 distance_max = excluded.distance_max,
                 type_eau_pref = excluded.type_eau_pref,
                 score_min = excluded.score_min""",
            (user_id, distance_max, type_eau_pref, score_min),
        )


# ── Favoris ───────────────────────────────────────────────────────────────────

def get_favoris(user_id: int) -> list[str]:
    """Retourne la liste ordonnée de code_site favoris de l'utilisateur."""
    with _conn() as conn:
        rows = conn.execute(
            "SELECT code_site FROM favoris WHERE user_id = ? ORDER BY date_ajout",
            (user_id,),
        ).fetchall()
    return [r["code_site"] for r in rows]


def add_favori(user_id: int, code_site: str) -> None:
    """Ajoute un favori (ignoré si déjà présent)."""
    with _conn() as conn:
        conn.execute(
            "INSERT OR IGNORE INTO favoris (user_id, code_site) VALUES (?, ?)",
            (user_id, code_site),
        )


def remove_favori(user_id: int, code_site: str) -> None:
    """Supprime un favori."""
    with _conn() as conn:
        conn.execute(
            "DELETE FROM favoris WHERE user_id = ? AND code_site = ?",
            (user_id, code_site),
        )
