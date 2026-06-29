"""
ETL pipeline - AquaReco
Charge, nettoie et fusionne les 4 types de fichiers CSV data.gouv.fr (2020-2024).
"""

import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).parent


# ── Utilitaires ────────────────────────────────────────────────────────────────

def normalize_col(name: str) -> str:
    """Convertit un nom de colonne en ASCII snake_case sans accents."""
    name = unicodedata.normalize("NFD", str(name))
    name = name.encode("ascii", "ignore").decode("ascii")
    name = re.sub(r"[^a-z0-9]+", "_", name.lower())
    name = re.sub(r"_+", "_", name)
    return name.strip("_")


def _keyword_rename(columns: list, mapping: list) -> dict:
    """
    Construit un dict de renommage par correspondance de sous-chaîne.
    mapping: liste ordonnée de (keyword, nouveau_nom). Chaque colonne
    ne peut être mappée qu'une fois ; l'ordre détermine la priorité.
    """
    used_cols = set()
    result = {}
    for keyword, new_name in mapping:
        for col in columns:
            if col not in used_cols and keyword in col:
                result[col] = new_name
                used_cols.add(col)
                break
    return result


def _read_csv(path: Path) -> pd.DataFrame:
    """Lit un CSV séparé par ';' en latin-1, normalise les noms de colonnes."""
    df = pd.read_csv(path, sep=";", encoding="latin-1", dtype=str, low_memory=False)
    df = df.dropna(axis=1, how="all")                     # supprime les colonnes vides (trailing ';')
    df.columns = [normalize_col(c) for c in df.columns]
    return df


def _find_files(data_dir: Path, fragment: str) -> list:
    return sorted(data_dir.glob(f"*{fragment}*.csv"))


# ── Mappings de renommage (ordonnés par priorité) ──────────────────────────────

SITES_MAPPING = [
    ("code_unique_d_identification", "code_site"),   # avant precedent_code_unique
    ("precedent_code_unique", "ancien_code"),
    ("saison_balneaire", "saison"),
    ("region", "region"),
    ("departement", "departement"),
    ("evolution", "evolution"),
    ("nom_du_site", "nom_site"),
    ("code_insee", "code_insee"),
    ("nom_de_la_commune", "commune"),
    ("date_declaration_ue", "date_declaration_ue"),
    ("type_d_eau", "type_eau"),
    ("longitude", "longitude"),
    ("latitude", "latitude"),
]

CARACT_MAPPING = [
    ("code_unique_d_identification", "code_site"),
    ("saison_balneaire", "saison"),
    ("region", "region"),
    ("departement", "departement"),
    ("origine_eau", "origine_eau"),
    ("classement", "classement"),
    ("mise_en", "statut_controle"),        # "mise_en_oeuvre_du_controle_sanitaire"
    ("calendrier", "statut_calendrier"),
    ("contrainte_geo", "contrainte_geo"),
    ("lien", "lien_profil"),
]

ANALYSES_MAPPING = [
    ("code_unique_d_identification", "code_site"),
    ("saison_balneaire", "saison"),
    ("region", "region"),
    ("departement", "departement"),
    ("date_du_prelevement", "date_prelevement"),   # 2020 : "Date du prélèvement"
    ("date_de_prelevement", "date_prelevement"),   # 2021-2024 : "Date de prélèvement"
    ("enterocoques", "enterococci"),
    ("escherichia", "ecoli"),
    ("statut_du_prelevement", "statut_prelevement"),
]

SAISON_MAPPING = [
    ("code_unique_d_identification", "code_site"),
    ("saison_balneaire", "saison"),
    ("region", "region"),
    ("departement", "departement"),
    ("type_d_evenement", "type_evenement"),
    ("date_de_debut", "date_debut"),
    ("date_de_fin", "date_fin"),
    ("mesures", "mesures_gestion"),
]


# ── Chargement brut ────────────────────────────────────────────────────────────

def load_sites(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    frames = []
    for f in _find_files(data_dir, "liste-des-sites"):
        df = _read_csv(f)
        df = df.rename(columns=_keyword_rename(list(df.columns), SITES_MAPPING))
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def load_caracteristiques(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    frames = []
    for f in _find_files(data_dir, "caracteristiques"):
        df = _read_csv(f)
        df = df.rename(columns=_keyword_rename(list(df.columns), CARACT_MAPPING))
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def load_analyses(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    frames = []
    for f in _find_files(data_dir, "resultats-danalyses"):
        df = _read_csv(f)
        df = df.rename(columns=_keyword_rename(list(df.columns), ANALYSES_MAPPING))
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def load_saison_info(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    frames = []
    for f in _find_files(data_dir, "informations-sur-la-saison"):
        df = _read_csv(f)
        df = df.rename(columns=_keyword_rename(list(df.columns), SAISON_MAPPING))
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


# ── Nettoyage ──────────────────────────────────────────────────────────────────

def _parse_coord(series: pd.Series) -> pd.Series:
    """Coordonnée en notation française (virgule décimale) → float."""
    return pd.to_numeric(series.str.replace(",", ".", regex=False), errors="coerce")


def _parse_micro(series: pd.Series) -> pd.Series:
    """
    Valeur microbiologique → float.
    Gère les cas '<N' (censure à gauche) et '>N' (censure à droite).
    '<10' → 5  (moitié du seuil, valeur conventionnelle)
    '>10000' → 15000  (x1.5, indique un dépassement majeur)
    """
    s = series.astype(str).str.strip()
    is_lt = s.str.startswith("<")
    is_gt = s.str.startswith(">")
    nums = pd.to_numeric(s.str.lstrip("<>"), errors="coerce")
    result = nums.copy()
    result[is_lt] = nums[is_lt] * 0.5
    result[is_gt] = nums[is_gt] * 1.5
    return result


def clean_sites(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["saison"] = pd.to_numeric(df["saison"], errors="coerce").astype("Int64")
    df["longitude"] = _parse_coord(df["longitude"])
    df["latitude"] = _parse_coord(df["latitude"])
    # Normalise la casse des valeurs catégorielles (incohérente entre 2020-2023 et 2024)
    for col in ("type_eau", "region", "commune"):
        if col in df.columns:
            df[col] = df[col].str.strip().str.title()
    df = df.drop_duplicates(subset=["code_site", "saison"])
    keep = ["saison", "region", "departement", "code_site", "nom_site",
            "commune", "type_eau", "longitude", "latitude", "evolution"]
    return df[[c for c in keep if c in df.columns]]


# Les fichiers 2020-2023 encodent le classement en texte français ;
# 2024 utilise les entiers 1-4 de la Directive 2006/7/CE.
_CLASSEMENT_TEXT_MAP = {
    "excellent": 1, "bon": 2, "suffisant": 3, "insuffisant": 4,
    "non classe": None, "non classee": None, "baignade supprimee": None, "0": None,
    "1": 1, "2": 2, "3": 3, "4": 4,
}


def _parse_classement(series: pd.Series) -> pd.Series:
    """Convertit les valeurs textuelles ou numériques du classement en entier 1-4."""
    import unicodedata, re

    def _norm(s):
        s = unicodedata.normalize("NFD", str(s))
        s = s.encode("ascii", "ignore").decode("ascii").lower()
        return re.sub(r"[^a-z0-9]+", "", s)

    return series.map(lambda x: _CLASSEMENT_TEXT_MAP.get(_norm(x), None) if pd.notna(x) else None
                      ).astype("Int64")


def clean_caracteristiques(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["saison"] = pd.to_numeric(df["saison"], errors="coerce").astype("Int64")
    df["classement"] = _parse_classement(df["classement"])
    if "origine_eau" in df.columns:
        df["origine_eau"] = df["origine_eau"].str.strip().str.title()
    df = df.drop_duplicates(subset=["code_site", "saison"])
    keep = ["saison", "code_site", "origine_eau", "classement",
            "statut_controle", "statut_calendrier", "contrainte_geo"]
    return df[[c for c in keep if c in df.columns]]


def clean_analyses(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["saison"] = pd.to_numeric(df["saison"], errors="coerce").astype("Int64")
    df["date_prelevement"] = pd.to_datetime(
        df["date_prelevement"], dayfirst=True, errors="coerce", format="mixed"
    )
    df["enterococci"] = _parse_micro(df["enterococci"])
    df["ecoli"] = _parse_micro(df["ecoli"])
    # Supprime les lignes sans aucune valeur microbiologique
    df = df.dropna(subset=["enterococci", "ecoli"], how="all")
    keep = ["saison", "code_site", "date_prelevement",
            "enterococci", "ecoli", "statut_prelevement"]
    return df[[c for c in keep if c in df.columns]]


def clean_saison_info(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["saison"] = pd.to_numeric(df["saison"], errors="coerce").astype("Int64")
    df["date_debut"] = pd.to_datetime(df["date_debut"], dayfirst=True, errors="coerce")
    df["date_fin"] = pd.to_datetime(df["date_fin"], dayfirst=True, errors="coerce")
    keep = ["saison", "code_site", "type_evenement",
            "date_debut", "date_fin", "mesures_gestion"]
    return df[[c for c in keep if c in df.columns]]


# ── Agrégation des événements de saison ────────────────────────────────────────

def aggregate_events(df_saison: pd.DataFrame) -> pd.DataFrame:
    """
    Agrège les événements (interdictions, alertes) par site et par saison.
    Retourne nb_interdictions et jours_fermeture par (code_site, saison).
    """
    df = df_saison.copy()
    df["duree_jours"] = (df["date_fin"] - df["date_debut"]).dt.days.clip(lower=0)

    mask_interdit = df["type_evenement"].str.contains(
        "interdiction|interdit|fermeture", case=False, na=False
    )
    interdictions = df[mask_interdit]

    agg = (
        interdictions
        .groupby(["code_site", "saison"])
        .agg(
            nb_interdictions=("type_evenement", "count"),
            jours_fermeture=("duree_jours", "sum"),
        )
        .reset_index()
    )
    return agg


# ── Pipeline principal ─────────────────────────────────────────────────────────

def build_consolidated(data_dir: Path = DATA_DIR) -> tuple:
    """
    Pipeline ETL complet.

    Retourne
    --------
    df_site_year : DataFrame consolidé au niveau site × saison
    df_analyses  : DataFrame granulaire des prélèvements (nécessaire pour le score bactério)
    """
    print("Chargement des fichiers CSV…")
    sites  = clean_sites(load_sites(data_dir))
    caract = clean_caracteristiques(load_caracteristiques(data_dir))
    analy  = clean_analyses(load_analyses(data_dir))
    saison = clean_saison_info(load_saison_info(data_dir))

    print(f"  Sites        : {len(sites):,} lignes  ({sites['saison'].nunique()} saisons, {sites['code_site'].nunique()} sites)")
    print(f"  Caractérist. : {len(caract):,} lignes")
    print(f"  Analyses     : {len(analy):,} prélèvements")
    print(f"  Évén. saison : {len(saison):,} lignes")

    events = aggregate_events(saison)
    key = ["code_site", "saison"]

    df = sites.merge(caract, on=key, how="left", suffixes=("", "_dup"))
    # Supprime les colonnes dupliquées issues du merge (region, departement…)
    df = df[[c for c in df.columns if not c.endswith("_dup")]]
    df = df.merge(events, on=key, how="left")
    df["nb_interdictions"] = df["nb_interdictions"].fillna(0).astype(int)
    df["jours_fermeture"]  = df["jours_fermeture"].fillna(0).astype(int)

    print(f"\nDataFrame consolidé : {len(df):,} lignes × {df.shape[1]} colonnes")
    print(f"Valeurs manquantes (%) :\n{(df.isnull().mean() * 100).round(1).to_string()}")

    return df, analy


# ── Rapport qualité données ────────────────────────────────────────────────────

def missing_report(df: pd.DataFrame) -> pd.DataFrame:
    """Retourne un DataFrame avec le taux de valeurs manquantes par colonne."""
    report = pd.DataFrame({
        "n_missing": df.isnull().sum(),
        "pct_missing": (df.isnull().mean() * 100).round(2),
        "dtype": df.dtypes.astype(str),
    })
    return report.sort_values("pct_missing", ascending=False)


if __name__ == "__main__":
    df_site_year, df_analyses = build_consolidated()
    print("\nAperçu du DataFrame consolidé :")
    print(df_site_year.head())
