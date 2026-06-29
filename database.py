"""
Stockage PostgreSQL/PostGIS - AquaReco
Crée les tables, les index spatiaux et insère les données ETL + scores.

Configuration via variables d'environnement (fichier .env ou shell) :
  DB_HOST      hôte PostgreSQL            (défaut : localhost)
  DB_PORT      port                       (défaut : 5432)
  DB_NAME      nom de la base             (défaut : aquareco)
  DB_USER      utilisateur                (défaut : postgres)
  DB_PASSWORD  mot de passe               (défaut : postgres)

Prérequis :
  pip install psycopg2-binary python-dotenv
  Extension PostGIS activée dans la base (voir instructions en bas de fichier).
"""

import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")


# ── Chargement de la configuration ────────────────────────────────────────────

def _load_env() -> None:
    """Charge le fichier .env s'il existe (via python-dotenv ou manuellement)."""
    env_file = Path(__file__).parent / ".env"
    if not env_file.exists():
        return
    try:
        from dotenv import load_dotenv
        load_dotenv(env_file)
    except ImportError:
        # Fallback manuel si python-dotenv n'est pas installé
        with open(env_file) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, _, val = line.partition("=")
                    os.environ.setdefault(key.strip(), val.strip().strip('"').strip("'"))


def get_db_config() -> dict:
    """Retourne les paramètres de connexion depuis les variables d'environnement."""
    _load_env()
    return {
        "host":     os.environ.get("DB_HOST",     "localhost"),
        "port":     int(os.environ.get("DB_PORT", "5432")),
        "dbname":   os.environ.get("DB_NAME",     "aquareco"),
        "user":     os.environ.get("DB_USER",     "postgres"),
        "password": os.environ.get("DB_PASSWORD", "postgres"),
    }


def get_connection():
    """Ouvre et retourne une connexion psycopg2."""
    try:
        import psycopg2
    except ImportError as e:
        raise ImportError(
            "psycopg2 non installé. Lancez : pip install psycopg2-binary"
        ) from e
    cfg = get_db_config()
    conn = psycopg2.connect(**cfg)
    logger.info(
        "Connexion PostgreSQL : %s@%s:%s/%s",
        cfg["user"], cfg["host"], cfg["port"], cfg["dbname"]
    )
    return conn


# ── DDL - Création des tables ──────────────────────────────────────────────────

DDL_SITES = """
CREATE TABLE IF NOT EXISTS sites (
    code_site       VARCHAR(30) PRIMARY KEY,
    nom_site        TEXT,
    commune         TEXT,
    region          TEXT,
    departement     VARCHAR(10),
    type_eau        VARCHAR(50),
    origine_eau     VARCHAR(50),
    longitude       DOUBLE PRECISION,
    latitude        DOUBLE PRECISION,
    geom            GEOMETRY(POINT, 4326),     -- PostGIS, EPSG:4326 (WGS 84)
    date_import     TIMESTAMP DEFAULT NOW()
);
"""

DDL_SITES_INDEX = """
CREATE INDEX IF NOT EXISTS idx_sites_geom
    ON sites USING GIST (geom);
"""

DDL_ANALYSES = """
CREATE TABLE IF NOT EXISTS analyses (
    id                  BIGSERIAL PRIMARY KEY,
    code_site           VARCHAR(30) REFERENCES sites(code_site) ON DELETE CASCADE,
    saison              SMALLINT,
    date_prelevement    DATE,
    enterococci         REAL,
    ecoli               REAL,
    statut_prelevement  TEXT,
    UNIQUE (code_site, date_prelevement, statut_prelevement)
);
CREATE INDEX IF NOT EXISTS idx_analyses_site_saison
    ON analyses (code_site, saison);
"""

DDL_SCORES = """
CREATE TABLE IF NOT EXISTS scores (
    code_site               VARCHAR(30),
    saison                  SMALLINT,
    classement_officiel     SMALLINT,
    score_bacterio          REAL,
    score_tendance          REAL,
    score_meteo             REAL,
    score_ouverture         REAL,
    score_final_expert      REAL,
    score_final_appris      REAL,
    precip_7j_median        REAL,
    precip_7j_max           REAL,
    n_prelevements          INTEGER,
    nb_interdictions        INTEGER,
    jours_fermeture         INTEGER,
    date_calcul             TIMESTAMP DEFAULT NOW(),
    PRIMARY KEY (code_site, saison),
    FOREIGN KEY (code_site) REFERENCES sites(code_site) ON DELETE CASCADE
);
"""

DDL_EQUIPEMENTS = """
CREATE TABLE IF NOT EXISTS equipements (
    code_site       VARCHAR(30) PRIMARY KEY
                        REFERENCES sites(code_site) ON DELETE CASCADE,
    parking         BOOLEAN,
    sanitaires      BOOLEAN,
    pmr             BOOLEAN,
    douche          BOOLEAN,
    poste_secours   BOOLEAN,
    date_import     TIMESTAMP DEFAULT NOW()
);
"""

# Requête de vérification de l'extension PostGIS
CHECK_POSTGIS = "SELECT PostGIS_Version();"


# ── Helpers ────────────────────────────────────────────────────────────────────

def _check_postgis(cur) -> bool:
    """Vérifie que PostGIS est activé. Lève une erreur explicite sinon."""
    try:
        cur.execute(CHECK_POSTGIS)
        version = cur.fetchone()[0]
        logger.info("PostGIS version : %s", version)
        return True
    except Exception:
        raise RuntimeError(
            "Extension PostGIS non disponible.\n"
            "Activez-la avec : CREATE EXTENSION IF NOT EXISTS postgis;"
        )


def _safe_float(val) -> float | None:
    """Convertit une valeur numérique nullable en float Python ou None."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def _safe_int(val) -> int | None:
    try:
        if pd.isna(val):
            return None
    except TypeError:
        pass
    try:
        return int(val)
    except (TypeError, ValueError):
        return None


# ── Création du schéma ─────────────────────────────────────────────────────────

def create_schema(conn) -> None:
    """Crée toutes les tables et index si inexistants."""
    with conn.cursor() as cur:
        _check_postgis(cur)
        for ddl in [DDL_SITES, DDL_SITES_INDEX, DDL_ANALYSES, DDL_SCORES, DDL_EQUIPEMENTS]:
            for stmt in ddl.strip().split(";"):
                stmt = stmt.strip()
                if stmt:
                    cur.execute(stmt)
    conn.commit()
    logger.info("Schema PostgreSQL cree / verifie.")


# ── Insertion des données ──────────────────────────────────────────────────────

def upsert_sites(conn, df_site_year: pd.DataFrame) -> int:
    """
    Insère ou met à jour la table `sites` (un enregistrement par code_site unique).
    La géométrie PostGIS est construite à partir de longitude/latitude.
    """
    # On garde la dernière saison disponible pour chaque site
    df = (
        df_site_year
        .sort_values("saison", ascending=False)
        .drop_duplicates(subset=["code_site"])
        [["code_site", "nom_site", "commune", "region", "departement",
          "type_eau", "origine_eau", "longitude", "latitude"]]
    )

    sql = """
        INSERT INTO sites (
            code_site, nom_site, commune, region, departement,
            type_eau, origine_eau, longitude, latitude, geom
        ) VALUES (
            %s, %s, %s, %s, %s, %s, %s, %s, %s,
            ST_SetSRID(ST_MakePoint(%s, %s), 4326)
        )
        ON CONFLICT (code_site) DO UPDATE SET
            nom_site    = EXCLUDED.nom_site,
            commune     = EXCLUDED.commune,
            region      = EXCLUDED.region,
            longitude   = EXCLUDED.longitude,
            latitude    = EXCLUDED.latitude,
            geom        = EXCLUDED.geom,
            date_import = NOW();
    """

    rows = []
    for _, r in df.iterrows():
        lon = _safe_float(r.get("longitude"))
        lat = _safe_float(r.get("latitude"))
        rows.append((
            r["code_site"],
            r.get("nom_site"),     r.get("commune"),
            r.get("region"),       r.get("departement"),
            r.get("type_eau"),     r.get("origine_eau"),
            lon, lat,
            lon, lat,              # ST_MakePoint(lon, lat)
        ))

    with conn.cursor() as cur:
        cur.executemany(sql, rows)
    conn.commit()
    logger.info("Table sites : %d lignes inseres/mises a jour.", len(rows))
    return len(rows)


def upsert_analyses(conn, df_analyses: pd.DataFrame) -> int:
    """Insère les prélèvements (dédoublonnage par code_site + date + statut)."""
    df = df_analyses[["code_site", "saison", "date_prelevement",
                       "enterococci", "ecoli", "statut_prelevement"]].copy()

    sql = """
        INSERT INTO analyses (
            code_site, saison, date_prelevement,
            enterococci, ecoli, statut_prelevement
        ) VALUES (%s, %s, %s, %s, %s, %s)
        ON CONFLICT (code_site, date_prelevement, statut_prelevement) DO NOTHING;
    """

    rows = []
    for _, r in df.iterrows():
        date = r["date_prelevement"]
        date = date.date() if hasattr(date, "date") and pd.notna(date) else None
        if date is None:
            continue
        rows.append((
            r["code_site"],
            _safe_int(r.get("saison")),
            date,
            _safe_float(r.get("enterococci")),
            _safe_float(r.get("ecoli")),
            r.get("statut_prelevement"),
        ))

    # Insertion par lots de 5000 pour la performance
    BATCH = 5000
    with conn.cursor() as cur:
        for i in range(0, len(rows), BATCH):
            cur.executemany(sql, rows[i: i + BATCH])
            conn.commit()

    logger.info("Table analyses : %d lignes inseres.", len(rows))
    return len(rows)


def upsert_scores(conn, df_scored: pd.DataFrame) -> int:
    """Insère ou met à jour les scores calculés."""
    sql = """
        INSERT INTO scores (
            code_site, saison, classement_officiel,
            score_bacterio, score_tendance, score_meteo, score_ouverture,
            score_final_expert, score_final_appris,
            precip_7j_median, precip_7j_max,
            n_prelevements, nb_interdictions, jours_fermeture
        ) VALUES (
            %s, %s, %s,
            %s, %s, %s, %s,
            %s, %s,
            %s, %s,
            %s, %s, %s
        )
        ON CONFLICT (code_site, saison) DO UPDATE SET
            classement_officiel  = EXCLUDED.classement_officiel,
            score_bacterio       = EXCLUDED.score_bacterio,
            score_tendance       = EXCLUDED.score_tendance,
            score_meteo          = EXCLUDED.score_meteo,
            score_ouverture      = EXCLUDED.score_ouverture,
            score_final_expert   = EXCLUDED.score_final_expert,
            score_final_appris   = EXCLUDED.score_final_appris,
            precip_7j_median     = EXCLUDED.precip_7j_median,
            precip_7j_max        = EXCLUDED.precip_7j_max,
            date_calcul          = NOW();
    """

    rows = []
    for _, r in df_scored.iterrows():
        rows.append((
            r["code_site"],
            _safe_int(r.get("saison")),
            _safe_int(r.get("classement")),
            _safe_float(r.get("score_bacterio")),
            _safe_float(r.get("score_tendance")),
            _safe_float(r.get("score_meteo")),
            _safe_float(r.get("score_ouverture")),
            _safe_float(r.get("score_expert")),
            _safe_float(r.get("score_appris")),
            _safe_float(r.get("precip_7j_median")),
            _safe_float(r.get("precip_7j_max")),
            _safe_int(r.get("n_prelevements")),
            _safe_int(r.get("nb_interdictions")),
            _safe_int(r.get("jours_fermeture")),
        ))

    BATCH = 5000
    with conn.cursor() as cur:
        for i in range(0, len(rows), BATCH):
            cur.executemany(sql, rows[i: i + BATCH])
            conn.commit()

    logger.info("Table scores : %d lignes inseres/mises a jour.", len(rows))
    return len(rows)


def upsert_equipements(conn, df_equip: pd.DataFrame) -> int:
    """Insère ou met à jour les équipements OSM pour chaque site."""
    sql = """
        INSERT INTO equipements (
            code_site, parking, sanitaires, pmr, douche, poste_secours
        ) VALUES (%s, %s, %s, %s, %s, %s)
        ON CONFLICT (code_site) DO UPDATE SET
            parking       = EXCLUDED.parking,
            sanitaires    = EXCLUDED.sanitaires,
            pmr           = EXCLUDED.pmr,
            douche        = EXCLUDED.douche,
            poste_secours = EXCLUDED.poste_secours,
            date_import   = NOW();
    """
    rows = [
        (
            r["code_site"],
            bool(r["parking"]),
            bool(r["sanitaires"]),
            bool(r["pmr"]),
            bool(r["douche"]),
            bool(r["poste_secours"]),
        )
        for _, r in df_equip.iterrows()
    ]
    with conn.cursor() as cur:
        cur.executemany(sql, rows)
    conn.commit()
    logger.info("Table equipements : %d lignes inseres/mises a jour.", len(rows))
    return len(rows)


# ── Requêtes de démonstration ──────────────────────────────────────────────────

def query_sites_within_radius(conn, lat: float, lon: float,
                               radius_km: float = 10.0, limit: int = 20) -> pd.DataFrame:
    """
    Retourne les sites dans un rayon de radius_km autour du point (lat, lon).
    Utilise l'index spatial PostGIS pour la performance.
    """
    sql = """
        SELECT
            s.code_site,
            s.nom_site,
            s.commune,
            s.type_eau,
            ROUND(ST_Distance(
                s.geom::geography,
                ST_SetSRID(ST_MakePoint(%s, %s), 4326)::geography
            ) / 1000.0, 2) AS distance_km,
            sc.score_final_expert,
            sc.classement_officiel,
            sc.saison
        FROM sites s
        LEFT JOIN scores sc
            ON s.code_site = sc.code_site
            AND sc.saison = (SELECT MAX(saison) FROM scores WHERE code_site = s.code_site)
        WHERE ST_DWithin(
            s.geom::geography,
            ST_SetSRID(ST_MakePoint(%s, %s), 4326)::geography,
            %s * 1000
        )
        ORDER BY distance_km, sc.score_final_expert DESC NULLS LAST
        LIMIT %s;
    """
    with conn.cursor() as cur:
        cur.execute(sql, (lon, lat, lon, lat, radius_km, limit))
        cols = [d.name for d in cur.description]
        return pd.DataFrame(cur.fetchall(), columns=cols)


def query_top_sites_by_region(conn, region: str,
                               saison: int = 2024, limit: int = 10) -> pd.DataFrame:
    """Retourne les meilleurs sites d'une région pour une saison donnée."""
    sql = """
        SELECT
            s.code_site,
            s.nom_site,
            s.commune,
            s.type_eau,
            sc.score_final_expert,
            sc.classement_officiel,
            sc.score_bacterio,
            sc.score_meteo
        FROM sites s
        JOIN scores sc ON s.code_site = sc.code_site
        WHERE s.region ILIKE %s
          AND sc.saison = %s
          AND sc.score_final_expert IS NOT NULL
        ORDER BY sc.score_final_expert DESC
        LIMIT %s;
    """
    with conn.cursor() as cur:
        cur.execute(sql, (f"%{region}%", saison, limit))
        cols = [d.name for d in cur.description]
        return pd.DataFrame(cur.fetchall(), columns=cols)


# ── Pipeline complet ───────────────────────────────────────────────────────────

def load_to_db(df_site_year: pd.DataFrame,
               df_analyses: pd.DataFrame,
               df_scored: pd.DataFrame,
               df_equip: pd.DataFrame | None = None) -> None:
    """
    Pipeline d'insertion complet :
      1. Connexion et vérification PostGIS
      2. Création du schéma
      3. Insertion sites → analyses → scores → equipements (si fourni)
    """
    conn = get_connection()
    try:
        create_schema(conn)
        upsert_sites(conn, df_site_year)
        upsert_analyses(conn, df_analyses)
        upsert_scores(conn, df_scored)
        if df_equip is not None and not df_equip.empty:
            upsert_equipements(conn, df_equip)
        logger.info("Chargement PostgreSQL termine.")
    finally:
        conn.close()


# ── Instructions SQL de mise en place ─────────────────────────────────────────
SETUP_SQL = """
-- Créer la base et activer PostGIS (à exécuter en tant que superuser)
-- Lancer depuis psql :

CREATE DATABASE aquareco
    ENCODING 'UTF8'
    LC_COLLATE 'fr_FR.UTF-8'
    LC_CTYPE 'fr_FR.UTF-8'
    TEMPLATE template0;

\\c aquareco

CREATE EXTENSION IF NOT EXISTS postgis;
CREATE EXTENSION IF NOT EXISTS postgis_topology;

-- Vérification
SELECT PostGIS_Full_Version();

-- Créer un utilisateur dédié (optionnel)
-- CREATE USER aquareco_user WITH PASSWORD 'mot_de_passe_fort';
-- GRANT ALL PRIVILEGES ON DATABASE aquareco TO aquareco_user;
"""


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    print("Instructions de configuration PostgreSQL/PostGIS :")
    print(SETUP_SQL)

    print("\nTest de connexion (configure .env ou variables d'env) :")
    try:
        conn = get_connection()
        with conn.cursor() as cur:
            cur.execute("SELECT version();")
            print("  PostgreSQL :", cur.fetchone()[0])
            _check_postgis(cur)
        conn.close()
        print("  Connexion et PostGIS OK.")
    except Exception as e:
        print(f"  [WARN] {e}")
        print("  La base n'est pas disponible — module importe sans erreur.")
