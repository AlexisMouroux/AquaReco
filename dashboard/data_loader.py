"""Chargement et cache des données AquaReco."""
from __future__ import annotations

import json
import pathlib
import re
import unicodedata
import urllib.parse
import urllib.request

import joblib
import numpy as np
import pandas as pd
import streamlit as st

ROOT = pathlib.Path(__file__).parent.parent
PARQUET = ROOT / "outputs" / "features_temporal.parquet"
MODEL_PATH = ROOT / "outputs" / "model_qualite.pkl"
FEATURE_COLS_PATH = ROOT / "outputs" / "feature_columns.pkl"

SITES_CSV = {
    yr: ROOT / f"saison-balneaire-{yr}-liste-des-sites-de-baignade.csv"
    for yr in range(2020, 2025)
}
CARACT_CSV = {
    yr: ROOT / f"saison-balneaire-{yr}-caracteristiques-des-sites-de-baignade.csv"
    for yr in range(2020, 2025)
}

# classement_predit (sortie de model_qualite.pkl) : 0=Excellente, 1=Bonne, 3=Non conforme
# (l'indice interne du modele 0/1/2 est remappe vers 0/1/3 pour ce dict - voir load_sites)
CLASSEMENT_LABELS = {0: "Excellente", 1: "Bonne", 3: "Non conforme"}
CLASSEMENT_COLORS = {0: "#2ECC71", 1: "#F39C12", 3: "#E74C3C"}
CLASSEMENT_ORDER = {0: 0, 1: 1, 3: 2}

# Remappage indice modele (0=Excellent,1=Bon,2=Non conf.) -> labels legacy (0,1,3)
_MODEL_IDX_TO_LABEL = {0: 0, 1: 1, 2: 3}

# Classement officiel EU (caracteristiques CSV) : 1=Excellent, 2=Bon, 3=Suffisant, 4=Insuffisant
HIST_LABELS = {1: "Excellente", 2: "Bonne", 3: "Suffisante", 4: "Non conforme"}
HIST_COLORS = {1: "#2ECC71", 2: "#F39C12", 3: "#E67E22", 4: "#E74C3C"}


def _parse_classement_officiel(val) -> int | None:
    """Normalise un classement officiel (texte ou chiffre) → entier 1-4 ou None."""
    s = unicodedata.normalize("NFD", str(val).strip()).encode("ascii", "ignore").decode().lower()
    s = re.sub(r"[^a-z0-9]", "", s)
    mapping = {
        "excellent": 1, "excellente": 1,
        "bon": 2, "bonne": 2,
        "suffisant": 3, "suffisante": 3,
        "insuffisant": 4, "nonconforme": 4,
        "1": 1, "2": 2, "3": 3, "4": 4,
    }
    return mapping.get(s)


def _load_sites_meta() -> pd.DataFrame:
    """Charge les métadonnées des sites (nom, commune, lat, lon) depuis les CSV listes."""
    frames = []
    for yr, path in SITES_CSV.items():
        if not path.exists():
            continue
        df = pd.read_csv(path, sep=";", encoding="latin-1")
        df.columns = [c.strip() for c in df.columns]

        rename: dict[str, str] = {}
        for col in df.columns:
            cl = col.lower()
            if "code unique" in cl and "pr" not in cl:
                rename[col] = "code_site"
            elif "nom du site" in cl:
                rename[col] = "nom_site"
            elif "commune" in cl and "code" not in cl and "insee" not in cl:
                rename[col] = "commune"
            elif "longitude" in cl:
                rename[col] = "longitude"
            elif "latitude" in cl:
                rename[col] = "latitude"
        df = df.rename(columns=rename)

        keep = [c for c in ["code_site", "nom_site", "commune", "longitude", "latitude"] if c in df.columns]
        frames.append(df[keep].assign(saison=yr))

    meta = pd.concat(frames, ignore_index=True)
    for col in ["longitude", "latitude"]:
        meta[col] = (
            meta[col].astype(str)
            .str.replace(",", ".", regex=False)
            .pipe(pd.to_numeric, errors="coerce")
        )
    meta = meta.sort_values("saison", ascending=False).drop_duplicates("code_site")
    return meta[["code_site", "nom_site", "commune", "longitude", "latitude"]]


@st.cache_resource(show_spinner=False)
def _load_model():
    """Charge le modèle de classification persisté (HistGradientBoosting + SMOTE,
    entraîné sur classement_eu_correct - voir smote.py) et l'ordre de ses features."""
    model = joblib.load(MODEL_PATH)
    feature_cols = joblib.load(FEATURE_COLS_PATH)
    return model, feature_cols


@st.cache_data(show_spinner="Chargement des données…")
def load_sites() -> pd.DataFrame:
    """DataFrame par site : classement prédit par le modèle, équipements, tendance, coordonnées."""
    feat = pd.read_parquet(PARQUET)
    model, feature_cols = _load_model()

    # Ligne la plus récente disponible par site (toutes saisons confondues)
    feat_latest = (
        feat.sort_values(["code_site", "date_prelevement"])
        .groupby("code_site", as_index=False)
        .tail(1)
        .copy()
    )

    X = feat_latest[feature_cols].values.astype(float)
    pred  = model.predict(X)
    proba = model.predict_proba(X)

    feat_latest["classement_predit"]  = [_MODEL_IDX_TO_LABEL[p] for p in pred]
    feat_latest["proba_excellente"]   = proba[:, 0]
    feat_latest["proba_bonne"]        = proba[:, 1]
    feat_latest["proba_non_conforme"] = proba[:, 2]

    agg_cols = ["parking", "sanitaires", "pmr", "douche", "poste_secours"]
    site_agg = feat_latest[
        ["code_site", "type_eau", "tendance", "classement_predit",
         "proba_excellente", "proba_bonne", "proba_non_conforme"]
        + [c for c in agg_cols if c in feat_latest.columns]
    ].copy()

    type_map = {
        "rivi\xe8re": "Rivière",
        "lac": "Lac",
        "eau c\xf4ti\xe8re": "Eau côtière",
        "eau de mer": "Eau côtière",
        "eau de transition": "Eau de transition",
    }

    def _normalize(s: str) -> str:
        return type_map.get(str(s).strip().lower(), str(s).strip().title())

    site_agg["type_eau"] = site_agg["type_eau"].apply(_normalize)

    meta = _load_sites_meta()
    sites = site_agg.merge(meta, on="code_site", how="left")
    return sites.dropna(subset=["latitude", "longitude"]).reset_index(drop=True)


@st.cache_data(show_spinner=False)
def load_historique_officiel() -> pd.DataFrame:
    """Classements officiels EU 2020-2024 depuis les CSV caractéristiques.

    Retourne : code_site, saison, classement_num (1=Excellent … 4=Insuffisant).
    """
    frames = []
    for yr, path in CARACT_CSV.items():
        if not path.exists():
            continue
        df = pd.read_csv(path, sep=";", encoding="latin-1")
        df.columns = [c.strip() for c in df.columns]

        code_col = next(
            (c for c in df.columns if "code unique" in c.lower() and "pr" not in c.lower()),
            None,
        )
        cl_col = next((c for c in df.columns if "classement" in c.lower()), None)
        if code_col is None or cl_col is None:
            continue

        sub = df[[code_col, cl_col]].rename(columns={code_col: "code_site", cl_col: "classement_raw"})
        sub["saison"] = yr
        sub["classement_num"] = sub["classement_raw"].apply(_parse_classement_officiel)
        frames.append(sub[["code_site", "saison", "classement_num"]].dropna(subset=["classement_num"]))

    if not frames:
        return pd.DataFrame(columns=["code_site", "saison", "classement_num"])
    return pd.concat(frames, ignore_index=True)


@st.cache_data(show_spinner=False)
def geocode_ville(query: str) -> tuple[float, float] | None:
    """Géocode une ville via Nominatim. Retourne (lat, lon) ou None."""
    params = urllib.parse.urlencode({"q": query, "format": "json", "limit": 1, "countrycodes": "fr"})
    url = f"https://nominatim.openstreetmap.org/search?{params}"
    req = urllib.request.Request(url, headers={"User-Agent": "AquaReco-TER/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=6) as r:
            data = json.loads(r.read())
        if data:
            return float(data[0]["lat"]), float(data[0]["lon"])
    except Exception:
        pass
    return None


def haversine_km(lat1: float, lon1: float, lats: pd.Series, lons: pd.Series) -> pd.Series:
    """Distance de Haversine en km entre un point de référence et une série de points."""
    R = 6371.0
    lat1_r = np.radians(lat1)
    lat2_r = np.radians(lats)
    dlat = lat2_r - lat1_r
    dlon = np.radians(lons) - np.radians(lon1)
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1_r) * np.cos(lat2_r) * np.sin(dlon / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))
