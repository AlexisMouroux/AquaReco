"""
Météo - AquaReco
Interroge l'API OpenMeteo (historique, gratuite, sans clé) pour récupérer :
  - Précipitations journalières (mm)
  - Température moyenne de l'air à 2 m (°C)
  - Vitesse du vent moyenne à 10 m (km/h)
  - Direction du vent dominante (°), encodée en composantes sin/cos

Les variables sont calculées sur une fenêtre glissante de 7 jours
précédant chaque prélèvement, puis agrégées au niveau site × saison
pour le scoring expert, ou retournées au niveau prélèvement pour les
features ML (preprocessing.py).

Encodage circulaire de la direction du vent :
  vent_sin = sin(direction × π / 180)
  vent_cos = cos(direction × π / 180)
  Ce codage est nécessaire car 0° et 358° (nord) sont proches - sans lui,
  le modèle percevrait une distance de 358 entre deux directions identiques.
"""

import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

# ── Paramètres ─────────────────────────────────────────────────────────────────

OPENMETEO_URL = "https://archive-api.open-meteo.com/v1/archive"

# Arrondi des coordonnées pour le clustering (0.5° ≈ 55 km en latitude)
COORD_ROUND   = 0.25

# Fenêtre glissante des précipitations (jours)
WINDOW_DAYS   = 7

# Précipitations de référence : au-delà de ce seuil, score = 0
# (50 mm sur 7 jours est un événement pluvieux fort)
PRECIP_REF_MM = 50.0

# Délai entre deux appels API pour respecter le fair use OpenMeteo.
# 0.5s suffit pour éviter les 429 sur des sessions de ~100 requêtes.
REQUEST_DELAY = 1.5

FETCH_START   = "2019-12-24"    # 7 jours avant le 1er jan 2020
FETCH_END     = "2024-12-31"

OUTPUT_DIR    = Path(__file__).parent / "outputs"
# Cache v2 inclut précipitations + température + vent (renommé pour forcer rebuild)
CACHE_FILE         = OUTPUT_DIR / "meteo_daily_cache.parquet"
# Cache partiel : sauvegarde incrémentale pendant la construction (reprise après interruption)
CACHE_FILE_PARTIAL = OUTPUT_DIR / "meteo_daily_cache_partial.parquet"
# Ancien cache précipitations seules - conservé pour compatibilité score.py
CACHE_FILE_V1      = OUTPUT_DIR / "precip_cache.parquet"
SCORES_FILE        = OUTPUT_DIR / "meteo_scores.csv"

# Fréquence de sauvegarde incrémentale (toutes les N cellules récupérées)
SAVE_EVERY_N_CELLS = 10


# ── Appel API OpenMeteo ────────────────────────────────────────────────────────

def fetch_daily_meteo(lat: float, lon: float,
                      start: str = FETCH_START,
                      end: str   = FETCH_END,
                      retries: int = 5) -> pd.DataFrame | None:
    """
    Récupère précipitations, température et vent journaliers via OpenMeteo.

    Retourne un DataFrame avec colonnes :
      date, precip_mm, temp_mean, wind_speed, wind_dir_deg
    ou None si échec.
    """
    params = {
        "latitude":   round(lat, 4),
        "longitude":  round(lon, 4),
        "start_date": start,
        "end_date":   end,
        "daily":      ",".join([
            "precipitation_sum",
            "temperature_2m_mean",
            "wind_speed_10m_mean",
            "wind_direction_10m_dominant",
        ]),
        "timezone":   "UTC",
    }
    for attempt in range(retries):
        try:
            resp = requests.get(OPENMETEO_URL, params=params, timeout=30)

            # Gestion spécifique du rate-limiting : backoff long exponentiel
            if resp.status_code == 429:
                wait = 20 * (attempt + 1)
                logger.warning(
                    "429 Too Many Requests (lat=%.2f, lon=%.2f) — attente %ds (tentative %d/%d)",
                    lat, lon, wait, attempt + 1, retries,
                )
                time.sleep(wait)
                continue

            resp.raise_for_status()
            data = resp.json()
            daily = data["daily"]
            df = pd.DataFrame({
                "date":         pd.to_datetime(daily["time"]),
                "precip_mm":    daily["precipitation_sum"],
                "temp_mean":    daily["temperature_2m_mean"],
                "wind_speed":   daily["wind_speed_10m_mean"],
                "wind_dir_deg": daily["wind_direction_10m_dominant"],
            })
            for col in ["precip_mm", "temp_mean", "wind_speed", "wind_dir_deg"]:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            df["precip_mm"]    = df["precip_mm"].fillna(0.0)
            df["wind_dir_deg"] = df["wind_dir_deg"].fillna(0.0)
            return df
        except requests.RequestException as e:
            if attempt < retries - 1:
                wait = 5 * (attempt + 1)
                logger.warning(
                    "Erreur reseau (lat=%.2f, lon=%.2f) — attente %ds : %s",
                    lat, lon, wait, e,
                )
                time.sleep(wait)
            else:
                logger.warning("Echec API (lat=%.2f, lon=%.2f) : %s", lat, lon, e)
                return None


def fetch_precipitation(lat: float, lon: float,
                        start: str = FETCH_START,
                        end: str   = FETCH_END,
                        retries: int = 3) -> pd.DataFrame | None:
    """Alias de compatibilité - appelle fetch_daily_meteo et ne retourne que precip."""
    df = fetch_daily_meteo(lat, lon, start, end, retries)
    if df is None:
        return None
    return df[["date", "precip_mm"]].copy()


# ── Clustering géographique ────────────────────────────────────────────────────

def _cluster_coords(df_sites: pd.DataFrame) -> pd.DataFrame:
    """
    Arrondit lat/lon à COORD_ROUND pour regrouper les sites proches
    en une même « cellule météo ».

    Retourne un DataFrame avec colonnes ['code_site', 'lat_r', 'lon_r'].
    """
    sites = df_sites[["code_site", "latitude", "longitude"]].drop_duplicates("code_site")
    sites = sites.dropna(subset=["latitude", "longitude"])
    prec = COORD_ROUND
    sites = sites.copy()
    sites["lat_r"] = (sites["latitude"]  / prec).round() * prec
    sites["lon_r"] = (sites["longitude"] / prec).round() * prec
    return sites[["code_site", "lat_r", "lon_r"]]


# ── Construction du cache de précipitations ────────────────────────────────────

def build_precip_cache(df_sites: pd.DataFrame,
                       use_cache: bool = True) -> pd.DataFrame:
    """
    Récupère (ou charge depuis le cache) les données météo journalières pour
    toutes les cellules géographiques couvrant les sites.

    Variables récupérées : précipitations, température, vitesse et direction du vent.
    Rolling 7 jours calculé pour chaque variable.

    Retourne un DataFrame avec colonnes :
      lat_r, lon_r, date,
      precip_mm, temp_mean, wind_speed, wind_dir_deg,
      precip_7j, temp_7j, wind_speed_7j, wind_sin_7j, wind_cos_7j
    """
    OUTPUT_DIR.mkdir(exist_ok=True)

    if use_cache and CACHE_FILE.exists():
        logger.info("Cache meteo charge : %s", CACHE_FILE)
        return pd.read_parquet(CACHE_FILE)

    # Fallback vers l'ancien cache précipitations seules si disponible
    if use_cache and CACHE_FILE_V1.exists():
        logger.warning(
            "Cache v1 (precip seules) detecte. Reconstruction du cache v2 "
            "pour inclure temperature et vent. Ce cache sera utilise desormais."
        )

    cluster_df = _cluster_coords(df_sites)
    cells      = cluster_df[["lat_r", "lon_r"]].drop_duplicates()
    n_cells    = len(cells)
    logger.info("%d cellules meteorologiques a interroger.", n_cells)

    # Reprise : charge le cache partiel existant pour éviter de re-fetcher les cellules déjà traitées
    already_fetched: set[tuple] = set()
    records: list[pd.DataFrame] = []
    if CACHE_FILE_PARTIAL.exists():
        partial = pd.read_parquet(CACHE_FILE_PARTIAL)
        already_fetched = set(zip(partial["lat_r"], partial["lon_r"]))
        records.append(partial)
        logger.info(
            "Cache partiel charge : %d cellules deja recuperees, %d restantes.",
            len(already_fetched), n_cells - len(already_fetched),
        )

    new_records: list[pd.DataFrame] = []
    skipped = 0
    for i, (_, row) in enumerate(cells.iterrows(), 1):
        key = (row.lat_r, row.lon_r)
        if key in already_fetched:
            skipped += 1
            continue

        logger.info("  [%d/%d] lat=%.1f lon=%.1f", i, n_cells, row.lat_r, row.lon_r)
        df_day = fetch_daily_meteo(row.lat_r, row.lon_r)
        if df_day is None:
            continue
        df_day["lat_r"] = row.lat_r
        df_day["lon_r"] = row.lon_r
        new_records.append(df_day)
        already_fetched.add(key)
        time.sleep(REQUEST_DELAY)

        # Sauvegarde incrémentale toutes les SAVE_EVERY_N_CELLS cellules nouvelles
        if len(new_records) % SAVE_EVERY_N_CELLS == 0:
            partial_so_far = pd.concat(records + new_records, ignore_index=True)
            partial_so_far.to_parquet(CACHE_FILE_PARTIAL, index=False)
            logger.info("  Sauvegarde partielle : %d cellules", len(already_fetched))

    if skipped:
        logger.info("  %d cellules ignorees (deja en cache partiel).", skipped)

    all_records = records + new_records
    if not all_records:
        raise RuntimeError("Aucune donnee meteorologique recuperee.")

    cache = pd.concat(all_records, ignore_index=True)
    cache = cache.sort_values(["lat_r", "lon_r", "date"])

    grp = cache.groupby(["lat_r", "lon_r"])

    # Précipitations : somme glissante 7 jours
    cache["precip_7j"] = grp["precip_mm"].transform(
        lambda s: s.rolling(WINDOW_DAYS, min_periods=1).sum()
    )
    # Température : moyenne glissante 7 jours
    cache["temp_7j"] = grp["temp_mean"].transform(
        lambda s: s.rolling(WINDOW_DAYS, min_periods=1).mean()
    )
    # Vent : vitesse - moyenne glissante 7 jours
    cache["wind_speed_7j"] = grp["wind_speed"].transform(
        lambda s: s.rolling(WINDOW_DAYS, min_periods=1).mean()
    )
    # Vent : direction - encodage circulaire avant moyennage (évite l'artefact 0/360)
    cache["_wsin"] = np.sin(np.radians(cache["wind_dir_deg"].fillna(0)))
    cache["_wcos"] = np.cos(np.radians(cache["wind_dir_deg"].fillna(0)))
    cache["wind_sin_7j"] = grp["_wsin"].transform(
        lambda s: s.rolling(WINDOW_DAYS, min_periods=1).mean()
    )
    cache["wind_cos_7j"] = grp["_wcos"].transform(
        lambda s: s.rolling(WINDOW_DAYS, min_periods=1).mean()
    )
    cache = cache.drop(columns=["_wsin", "_wcos"])

    cache.to_parquet(CACHE_FILE, index=False)
    logger.info("Cache meteo sauvegarde : %s  (%d lignes)", CACHE_FILE, len(cache))

    # Supprime le cache partiel maintenant que le cache final est complet
    if CACHE_FILE_PARTIAL.exists():
        CACHE_FILE_PARTIAL.unlink()
        logger.info("Cache partiel supprime.")

    return cache


# ── Score météo ────────────────────────────────────────────────────────────────

def meteo_score_from_precip(precip_7j: pd.Series) -> pd.Series:
    """
    Convertit une somme de précipitations sur 7 jours [mm] en score [0, 100].

    Formule linéaire :
      score = max(0,  100 × (1 − precip_7j / PRECIP_REF_MM))
      → 0 mm  : 100 (temps sec, pas de risque de lessivage)
      → 25 mm : 50
      → 50 mm : 0  (événement pluvieux fort)
    """
    return (100.0 * (1.0 - precip_7j.clip(upper=PRECIP_REF_MM) / PRECIP_REF_MM)
            ).clip(lower=0.0)


# ── Assignation aux prélèvements ───────────────────────────────────────────────

def assign_meteo_to_samples(df_analyses : pd.DataFrame,
                             df_sites    : pd.DataFrame,
                             precip_cache: pd.DataFrame) -> pd.DataFrame:
    """
    Joint le cache météo à chaque prélèvement selon la cellule et la date.

    Retourne df_analyses enrichi avec :
      precip_7j, temp_7j, wind_speed_7j, wind_sin_7j, wind_cos_7j,
      score_meteo_sample
    """
    cluster_df = _cluster_coords(df_sites)
    df = df_analyses.merge(cluster_df, on="code_site", how="left")

    # Colonnes disponibles dans le cache (v1 ou v2)
    meteo_cols = ["lat_r", "lon_r", "date", "precip_7j"]
    for col in ["temp_7j", "wind_speed_7j", "wind_sin_7j", "wind_cos_7j"]:
        if col in precip_cache.columns:
            meteo_cols.append(col)

    lookup = precip_cache[meteo_cols].rename(columns={"date": "date_prelevement"})
    df = df.merge(lookup, on=["lat_r", "lon_r", "date_prelevement"], how="left")

    df["precip_7j"] = df["precip_7j"].fillna(0.0)
    df["score_meteo_sample"] = meteo_score_from_precip(df["precip_7j"])

    return df


# ── Agrégation au niveau site × saison ────────────────────────────────────────

def aggregate_meteo_to_site_year(df_samples: pd.DataFrame) -> pd.DataFrame:
    """
    Agrège les features météo au niveau site × saison (pour score expert et ML).

    Retourne les colonnes :
      score_meteo, precip_7j_median, precip_7j_max,
      temp_7j_median, wind_speed_7j_median,
      wind_sin_7j_median, wind_cos_7j_median,
      n_prelev_meteo
    """
    mask = ~df_samples["statut_prelevement"].str.contains(
        "pre|post|hors", case=False, na=False
    )
    df = df_samples[mask]

    agg_dict = {
        "score_meteo":         ("score_meteo_sample", "median"),
        "precip_7j_median":    ("precip_7j",          "median"),
        "precip_7j_max":       ("precip_7j",          "max"),
        "n_prelev_meteo":      ("score_meteo_sample", "count"),
    }
    for col, agg_name in [
        ("temp_7j",        "temp_7j_median"),
        ("wind_speed_7j",  "wind_speed_7j_median"),
        ("wind_sin_7j",    "wind_sin_7j_median"),
        ("wind_cos_7j",    "wind_cos_7j_median"),
    ]:
        if col in df.columns:
            agg_dict[agg_name] = (col, "median")

    agg = df.groupby(["code_site", "saison"]).agg(**agg_dict).reset_index()
    agg["score_meteo"] = agg["score_meteo"].clip(0, 100).round(2)
    return agg


# ── Pipeline complet ───────────────────────────────────────────────────────────

def compute_meteo_scores(df_analyses: pd.DataFrame,
                         df_sites   : pd.DataFrame,
                         use_cache  : bool = True) -> pd.DataFrame:
    """
    Pipeline météo complet : fetch → rolling sum → score → agrégation saison.

    Paramètres
    ----------
    df_analyses : DataFrame granulaire des prélèvements (issu de etl.py)
    df_sites    : DataFrame consolidé site × saison (issu de etl.py)
    use_cache   : True = réutilise le cache Parquet si présent

    Retourne
    --------
    DataFrame site × saison avec colonnes score_meteo, precip_7j_median,
    precip_7j_max, n_prelev_meteo.
    """
    logger.info("Construction du cache de precipitations OpenMeteo...")
    cache = build_precip_cache(df_sites, use_cache=use_cache)

    logger.info("Assignation des precipitations aux prelevements...")
    df_samples = assign_meteo_to_samples(df_analyses, df_sites, cache)

    logger.info("Agregation au niveau site x saison...")
    df_meteo = aggregate_meteo_to_site_year(df_samples)

    df_meteo.to_csv(SCORES_FILE, index=False, encoding="utf-8-sig")
    logger.info("Scores meteo sauvegardes : %s  (%d lignes)", SCORES_FILE, len(df_meteo))

    return df_meteo


# ── Features météo par site×saison (pour preprocessing.py / ML) ───────────────

def get_meteo_features_for_ml(df_analyses: pd.DataFrame,
                               df_sites: pd.DataFrame,
                               use_cache: bool = True) -> pd.DataFrame:
    """
    Retourne un DataFrame site×saison avec les features météo agrégées
    prêtes pour l'entraînement ML :
      precip_7j_median, temp_7j_median,
      wind_speed_7j_median, wind_sin_7j_median, wind_cos_7j_median

    Si le cache v2 (avec vent) n'existe pas, le construit automatiquement.
    """
    cache = build_precip_cache(df_sites, use_cache=use_cache)

    # Filtre les prélèvements en-saison uniquement
    mask_saison = ~df_analyses["statut_prelevement"].str.contains(
        r"pr[eé]-?saison|pre.?saison", case=False, na=False, regex=True
    )
    df_an_saison = df_analyses[mask_saison]

    df_samples = assign_meteo_to_samples(df_an_saison, df_sites, cache)

    meteo_cols = ["precip_7j"]
    for col in ["temp_7j", "wind_speed_7j", "wind_sin_7j", "wind_cos_7j"]:
        if col in df_samples.columns:
            meteo_cols.append(col)

    agg_dict = {}
    rename_map = {
        "precip_7j":      "precip_7j_median",
        "temp_7j":        "temp_7j_median",
        "wind_speed_7j":  "wind_speed_7j_median",
        "wind_sin_7j":    "wind_sin_7j_median",
        "wind_cos_7j":    "wind_cos_7j_median",
    }
    for col in meteo_cols:
        agg_dict[rename_map[col]] = (col, "median")

    return (
        df_samples.groupby(["code_site", "saison"])
                  .agg(**agg_dict)
                  .reset_index()
    )


# ── Features météo par prélèvement individuel (pour ML granulaire) ────────────

def get_meteo_features_per_sample(df_analyses: pd.DataFrame,
                                   df_sites   : pd.DataFrame,
                                   use_cache  : bool = True) -> pd.DataFrame:
    """
    Retourne un DataFrame avec une ligne par prélèvement enrichie des
    features météo des 7 jours précédant la date du prélèvement.

    Colonnes retournées :
      code_site, date_prelevement, saison,
      precip_7j, temp_7j, wind_speed_7j, wind_sin_7j, wind_cos_7j

    Sauvegarde le résultat dans outputs/meteo_features_ml.csv.
    """
    cache = build_precip_cache(df_sites, use_cache=use_cache)

    cluster_df = _cluster_coords(df_sites)
    df = df_analyses[["code_site", "date_prelevement", "saison"]].copy()
    df = df.merge(cluster_df, on="code_site", how="left")

    # Colonnes météo disponibles dans le cache
    meteo_cols = ["lat_r", "lon_r", "date", "precip_7j"]
    for col in ["temp_7j", "wind_speed_7j", "wind_sin_7j", "wind_cos_7j"]:
        if col in cache.columns:
            meteo_cols.append(col)

    lookup = cache[meteo_cols].rename(columns={"date": "date_prelevement"})
    df = df.merge(lookup, on=["lat_r", "lon_r", "date_prelevement"], how="left")

    keep_cols = ["code_site", "date_prelevement", "saison"]
    for col in ["precip_7j", "temp_7j", "wind_speed_7j", "wind_sin_7j", "wind_cos_7j"]:
        if col in df.columns:
            keep_cols.append(col)

    result = df[keep_cols].reset_index(drop=True)

    out_path = OUTPUT_DIR / "meteo_features_ml.csv"
    result.to_csv(out_path, index=False, encoding="utf-8-sig")
    logger.info(
        "Features meteo par prelevement sauvegardees : %s  (%d lignes, %d colonnes)",
        out_path, len(result), len(result.columns),
    )

    # Aperçu sur 5 prélèvements avec données météo renseignées
    sample = result.dropna(subset=["precip_7j"]).head(5)
    if not sample.empty:
        print("\n  Apercu meteo_features_per_sample (5 premiers prelevements renseignes) :")
        print(sample.to_string(index=False))

    return result


# ── Mode démo (3 sites) ────────────────────────────────────────────────────────

def run_demo(df_analyses: pd.DataFrame, df_sites: pd.DataFrame) -> None:
    """
    Démo sur 3 sites représentatifs :
      - un site côtier breton
      - un lac alpin
      - une rivière en Occitanie
    Affiche précipitations hebdomadaires et scores par saison.
    """
    import unicodedata as _ud

    def _norm_str(s: str) -> str:
        """Supprime les accents et met en minuscules."""
        s = _ud.normalize("NFD", str(s)).encode("ascii", "ignore").decode("ascii")
        return s.lower()

    def _pick_site(keyword: str) -> str:
        """Retourne le code du site le plus prélevé dont le type_eau contient keyword."""
        mask = df_sites["type_eau"].map(
            lambda x: keyword in _norm_str(x) if pd.notna(x) else False
        )
        candidates = df_sites[mask]["code_site"].value_counts()
        if candidates.empty:
            return None
        for code in candidates.index:
            if code.startswith("FR") and len(code) > 5:
                return code
        return candidates.index[0]

    demo_sites = {
        "Cote (eau cotiere)": _pick_site("cotiere"),
        "Lac":                _pick_site("lac"),
        "Riviere":            _pick_site("riviere"),
    }
    demo_sites = {k: v for k, v in demo_sites.items() if v}

    print("\n" + "=" * 60)
    print("  DEMO meteo.py — 3 sites representatifs")
    print("=" * 60)

    for label, code in demo_sites.items():
        site_info = df_sites[df_sites["code_site"] == code].iloc[0]
        lat, lon  = site_info["latitude"], site_info["longitude"]

        print(f"\nSite : {code}  ({label})")
        print(f"  Nom    : {site_info.get('nom_site', '?')}")
        print(f"  Coords : lat={lat:.4f}, lon={lon:.4f}")

        # Récupère les précipitations sur une période courte pour la démo
        df_prec = fetch_precipitation(lat, lon, start="2024-05-01", end="2024-09-30")
        if df_prec is None:
            print("  [WARN] Echec recuperation meteo.")
            continue

        df_prec["precip_7j"] = df_prec["precip_mm"].rolling(WINDOW_DAYS, min_periods=1).sum()
        df_prec["score_meteo"] = meteo_score_from_precip(df_prec["precip_7j"])

        # Prélèvements de ce site en 2024
        samples = df_analyses[
            (df_analyses["code_site"] == code) & (df_analyses["saison"] == 2024)
        ][["date_prelevement", "enterococci", "ecoli"]].copy()

        if len(samples) == 0:
            print("  Aucun prelevement en 2024.")
            continue

        samples = samples.merge(
            df_prec[["date", "precip_7j", "score_meteo"]].rename(
                columns={"date": "date_prelevement"}),
            on="date_prelevement", how="left"
        )

        print(f"\n  Saison 2024 — {len(samples)} prelevements")
        print(f"  {'Date':<12}  {'E.coli':>8}  {'Entero':>8}  {'P7j(mm)':>9}  {'ScoreMet':>9}")
        print("  " + "-" * 54)
        for _, r in samples.sort_values("date_prelevement").iterrows():
            date_s = r["date_prelevement"].strftime("%Y-%m-%d") if pd.notna(r["date_prelevement"]) else "?"
            ec     = f"{r['ecoli']:.0f}" if pd.notna(r['ecoli']) else "?"
            ent    = f"{r['enterococci']:.0f}" if pd.notna(r['enterococci']) else "?"
            p7     = f"{r['precip_7j']:.1f}" if pd.notna(r.get('precip_7j')) else "?"
            sc     = f"{r['score_meteo']:.1f}" if pd.notna(r.get('score_meteo')) else "?"
            print(f"  {date_s:<12}  {ec:>8}  {ent:>8}  {p7:>9}  {sc:>9}")

        # Résumé
        scores_valid = samples["score_meteo"].dropna()
        if len(scores_valid) > 0:
            print(f"\n  Score meteo median : {scores_valid.median():.1f}/100")
            print(f"  P7j max (mm)       : {samples['precip_7j'].max():.1f}")
            corr = samples[["precip_7j", "ecoli"]].dropna().corr()
            if not corr.empty:
                r_val = corr.loc["precip_7j", "ecoli"]
                print(f"  Corr P7j vs E.coli : {r_val:.3f}")

        time.sleep(REQUEST_DELAY)

    print()


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    from etl import build_consolidated
    df_site_year, df_analyses = build_consolidated()
    build_precip_cache(df_site_year, use_cache=True)
    get_meteo_features_per_sample(df_analyses, df_site_year, use_cache=True)
