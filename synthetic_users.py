"""
Génération de profils utilisateurs synthétiques - AquaReco Semaine 4
Produit 1 200 utilisateurs (6 profils × 200) et leurs interactions positives.
France métropolitaine uniquement (hors DOM-TOM 971-976).
"""

import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from pathlib import Path

import numpy as np
import pandas as pd

OUTPUT_DIR = Path(__file__).parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

RNG = np.random.default_rng(42)

DOM_TOM = {"971", "972", "973", "974", "976"}

# ── Villes de départ ──────────────────────────────────────────────────────────

CITIES_FR = {
    "Paris":      (48.8566,  2.3522),
    "Lyon":       (45.7640,  4.8357),
    "Marseille":  (43.2965,  5.3698),
    "Bordeaux":   (44.8378, -0.5792),
    "Lille":      (50.6292,  3.0573),
    "Nantes":     (47.2184, -1.5536),
    "Toulouse":   (43.6047,  1.4442),
    "Strasbourg": (48.5734,  7.7521),
}

CITIES_EU = {
    "Londres":    (51.5074, -0.1278),
    "Berlin":     (52.5200, 13.4050),
    "Madrid":     (40.4168, -3.7038),
    "Amsterdam":  (52.3676,  4.9041),
    "Rome":       (41.9028, 12.4964),
    "Bruxelles":  (50.8503,  4.3517),
}

# ── Définition des 6 profils ──────────────────────────────────────────────────

PROFILES = {
    "famille": {
        "score_min_mu": 75, "score_min_sigma": 8,
        "dist_max_mu": 150, "dist_max_sigma": 80,
        "pmr_proba": 0.4,   "san_proba": 0.5,
        "type_pref": ["Lac"],
        "cities": CITIES_FR,
    },
    "sportif": {
        "score_min_mu": 70, "score_min_sigma": 10,
        "dist_max_mu": 150, "dist_max_sigma": 60,
        "pmr_proba": 0.1,   "san_proba": 0.2,
        "type_pref": [],
        "cities": CITIES_FR,
    },
    "senior": {
        "score_min_mu": 85, "score_min_sigma": 6,
        "dist_max_mu": 60,  "dist_max_sigma": 20,
        "pmr_proba": 0.8,   "san_proba": 0.6,
        "type_pref": ["Lac", "Mer"],
        "cities": CITIES_FR,
    },
    "aventurier": {
        "score_min_mu": 50, "score_min_sigma": 12,
        "dist_max_mu": 250, "dist_max_sigma": 80,
        "pmr_proba": 0.05,  "san_proba": 0.1,
        "type_pref": ["Riviere", "Mer"],
        "cities": CITIES_FR,
    },
    "vacancier_cotier": {
        "score_min_mu": 65, "score_min_sigma": 10,
        "dist_max_mu": 40,  "dist_max_sigma": 15,
        "pmr_proba": 0.2,   "san_proba": 0.3,
        "type_pref": ["Mer", "Cote"],
        "cities": CITIES_FR,
    },
    "touriste_etranger": {
        "score_min_mu": 70, "score_min_sigma": 8,
        "dist_max_mu": 9999, "dist_max_sigma": 1,   # pas de contrainte distance
        "pmr_proba": 0.2,   "san_proba": 0.3,
        "type_pref": ["Mer", "Cote", "Lac"],
        "cities": CITIES_EU,
    },
}

N_PER_PROFILE   = 200
TOP_K           = 20       # interactions positives par utilisateur


# ── Utilitaires ────────────────────────────────────────────────────────────────

def _haversine_vec(lat0: float, lon0: float,
                   lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
    R = 6_371.0
    phi0, phi = np.radians(lat0), np.radians(lats)
    dphi = np.radians(lats - lat0)
    dlam = np.radians(lons - lon0)
    a = np.sin(dphi/2)**2 + np.cos(phi0)*np.cos(phi)*np.sin(dlam/2)**2
    return 2 * R * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def _norm_type(s: str) -> str:
    """Normalise type_eau : 'Eau Côtière' → 'Cote', 'Eau de Mer' → 'Mer', etc."""
    import unicodedata, re
    s = unicodedata.normalize("NFD", str(s)).encode("ascii", "ignore").decode("ascii")
    s = s.lower()
    if "cotier" in s or "cotiere" in s: return "Cote"
    if "lac"    in s:                   return "Lac"
    if "riviere" in s or "fleuve" in s or "cours" in s: return "Riviere"
    if "mer"    in s:                   return "Mer"
    if "transit" in s:                  return "Transition"
    return "Autre"


def _clamp(val: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, val))


# ── Chargement des sites ───────────────────────────────────────────────────────

def load_sites() -> pd.DataFrame:
    df = pd.read_csv(OUTPUT_DIR / "sites_scores.csv")
    # Garde la dernière saison par site (score le plus récent)
    df = df.sort_values("saison", ascending=False).drop_duplicates("code_site")
    # Exclut DOM-TOM
    df = df[~df["departement"].astype(str).isin(DOM_TOM)]
    df = df.dropna(subset=["latitude", "longitude", "score_expert"]).copy()
    df["type_norm"] = df["type_eau"].map(_norm_type)

    # Fusionne les équipements OSM si disponibles
    eq_path = OUTPUT_DIR / "osm_equipements.csv"
    if eq_path.exists():
        eq = pd.read_csv(eq_path, sep=";")[
            ["code_site", "parking", "sanitaires", "pmr", "douche", "poste_secours"]
        ]
        df = df.merge(eq, on="code_site", how="left")
    else:
        for col in ["parking", "sanitaires", "pmr", "douche", "poste_secours"]:
            df[col] = False

    for col in ["parking", "sanitaires", "pmr", "douche", "poste_secours"]:
        df[col] = df[col].fillna(False).astype(bool)

    return df.reset_index(drop=True)


# ── Score de pertinence site × utilisateur ────────────────────────────────────

def relevance_score(sites: pd.DataFrame,
                    user_lat: float, user_lon: float,
                    score_min: float, dist_max_km: float,
                    type_pref: list, need_pmr: bool,
                    need_san: bool) -> np.ndarray:
    """
    Score ∈ [0, 1] combinant :
      - qualité du site (40%)
      - pénalité distance (30%)
      - bonus type d'eau préféré (20%)
      - bonus équipements (10%)
    """
    lats = sites["latitude"].values
    lons = sites["longitude"].values
    dists = _haversine_vec(user_lat, user_lon, lats, lons)

    # Qualité normalisée [0,1]
    q     = sites["score_expert"].values / 100.0
    q_pen = np.where(sites["score_expert"].values >= score_min, q,
                     q * 0.5)   # pénalité si sous le seuil min

    # Distance : score = exp(-d / dist_max) → 1 si proche, ~0.37 à dist_max
    if dist_max_km >= 5000:          # touriste sans contrainte
        d_score = np.ones(len(sites))
    else:
        d_score = np.exp(-dists / max(dist_max_km, 1.0))
        d_score = np.where(dists > dist_max_km * 2, d_score * 0.2, d_score)

    # Type d'eau
    if type_pref:
        type_score = np.where(sites["type_norm"].isin(type_pref), 1.0, 0.4)
    else:
        type_score = np.ones(len(sites))

    # Équipements
    eq_score = np.ones(len(sites))
    if need_pmr:
        eq_score += np.where(sites["pmr"].values, 0.3, -0.1)
    if need_san:
        eq_score += np.where(sites["sanitaires"].values, 0.2, 0.0)
    eq_score = np.clip(eq_score / eq_score.max(), 0, 1)

    return 0.40 * q_pen + 0.30 * d_score + 0.20 * type_score + 0.10 * eq_score


# ── Génération des utilisateurs ───────────────────────────────────────────────

def generate_users(sites: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    users_rows        = []
    interactions_rows = []
    user_id           = 0

    for profile_name, p in PROFILES.items():
        city_names = list(p["cities"].keys())
        city_coords = list(p["cities"].values())

        for _ in range(N_PER_PROFILE):
            user_id += 1

            # Ville de départ (aléatoire dans la liste du profil)
            city_idx  = RNG.integers(0, len(city_names))
            city_name = city_names[city_idx]
            home_lat, home_lon = city_coords[city_idx]

            # Paramètres avec bruit gaussien
            score_min = float(_clamp(
                RNG.normal(p["score_min_mu"], p["score_min_sigma"]), 0, 100))
            dist_max  = float(_clamp(
                RNG.normal(p["dist_max_mu"],  p["dist_max_sigma"]),  5, 9999))
            need_pmr  = bool(RNG.random() < p["pmr_proba"])
            need_san  = bool(RNG.random() < p["san_proba"])

            # Légère perturbation de la position (représente l'hôtel / gîte)
            jitter_km = 20
            lat = home_lat + RNG.normal(0, jitter_km / 111.0)
            lon = home_lon + RNG.normal(0, jitter_km / (111.0 * np.cos(np.radians(home_lat))))

            users_rows.append({
                "user_id":      user_id,
                "profil":       profile_name,
                "ville_depart": city_name,
                "latitude":     round(lat, 4),
                "longitude":    round(lon, 4),
                "score_min":    round(score_min, 1),
                "dist_max_km":  round(dist_max, 1),
                "need_pmr":     need_pmr,
                "need_sanitaires": need_san,
                "type_pref":    "|".join(p["type_pref"]) if p["type_pref"] else "indifferent",
            })

            # Calcul du score de pertinence pour chaque site
            scores = relevance_score(
                sites, lat, lon,
                score_min, dist_max,
                p["type_pref"], need_pmr, need_san,
            )

            # Top-K interactions positives
            top_idx = np.argsort(scores)[::-1][:TOP_K]
            for rank, idx in enumerate(top_idx, 1):
                interactions_rows.append({
                    "user_id":          user_id,
                    "code_site":        sites.iloc[idx]["code_site"],
                    "nom_site":         sites.iloc[idx]["nom_site"],
                    "rank":             rank,
                    "relevance_score":  round(float(scores[idx]), 4),
                    "distance_km":      round(float(
                        _haversine_vec(lat, lon,
                                       np.array([sites.iloc[idx]["latitude"]]),
                                       np.array([sites.iloc[idx]["longitude"]]))[0]
                    ), 1),
                })

    df_users = pd.DataFrame(users_rows)
    df_inter = pd.DataFrame(interactions_rows)
    return df_users, df_inter


# ── Statistiques de contrôle ──────────────────────────────────────────────────

def print_stats(df_users: pd.DataFrame, df_inter: pd.DataFrame,
                sites: pd.DataFrame) -> None:
    print(f"\n{'='*60}")
    print("  PROFILS SYNTHETIQUES — STATISTIQUES")
    print(f"{'='*60}")
    print(f"  Utilisateurs generes  : {len(df_users):,}")
    print(f"  Interactions positives: {len(df_inter):,}  ({TOP_K} par utilisateur)")
    print(f"  Sites couverts (metro): {len(sites):,}")
    print()
    print(f"  {'Profil':<22} {'n':>5}  {'score_min':>9}  {'dist_max':>9}  {'pmr%':>6}  {'san%':>6}")
    print("  " + "-"*65)
    for prof, grp in df_users.groupby("profil", sort=False):
        print(f"  {prof:<22} {len(grp):>5}  "
              f"{grp['score_min'].mean():>8.1f}  "
              f"{grp['dist_max_km'].mean():>8.1f}  "
              f"{grp['need_pmr'].mean()*100:>5.1f}%  "
              f"{grp['need_sanitaires'].mean()*100:>5.1f}%")

    print()
    print("  Top 5 sites les plus recommandes :")
    top_sites = (
        df_inter.groupby("code_site")
        .agg(n_recommandations=("user_id", "count"),
             relevance_moy=("relevance_score", "mean"))
        .nlargest(5, "n_recommandations")
        .merge(sites[["code_site", "nom_site", "type_norm", "score_expert"]],
               on="code_site", how="left")
    )
    for _, r in top_sites.iterrows():
        print(f"    {str(r['nom_site'])[:40]:<41} "
              f"({r['type_norm']:<12}) "
              f"recommande {int(r['n_recommandations']):>4}x  "
              f"Q={r['score_expert']:.0f}  "
              f"rel_moy={r['relevance_moy']:.3f}")

    print(f"\n  Fichiers :  outputs/synthetic_users.csv")
    print(f"              outputs/synthetic_interactions.csv")
    print("="*60)


# ── Point d'entrée ────────────────────────────────────────────────────────────

def run_synthetic() -> tuple[pd.DataFrame, pd.DataFrame]:
    print("Chargement des sites (France metropolitaine)...")
    sites = load_sites()
    print(f"  {len(sites):,} sites disponibles apres filtrage DOM-TOM.")

    print("Generation des utilisateurs synthetiques...")
    df_users, df_inter = generate_users(sites)

    df_users.to_csv(OUTPUT_DIR / "synthetic_users.csv",
                    index=False, encoding="utf-8-sig")
    df_inter.to_csv(OUTPUT_DIR / "synthetic_interactions.csv",
                    index=False, encoding="utf-8-sig")

    print_stats(df_users, df_inter, sites)
    return df_users, df_inter


if __name__ == "__main__":
    run_synthetic()
