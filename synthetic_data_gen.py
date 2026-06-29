"""
synthetic_data_gen.py - Génère utilisateurs et interactions synthétiques.
6 profils × 500 utilisateurs = 3 000 utilisateurs.
Score de pertinence continu avec décroissance exponentielle (cohérent avec recommender.py).
"""

import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from pathlib import Path
import unicodedata

import numpy as np
import pandas as pd

OUTPUT_DIR = Path(__file__).parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

EARTH_RADIUS_KM = 6371
N_PER_PROFILE      = 500   # utilisateurs par profil
N_PER_USER         = 50    # interactions cibles par utilisateur
RELEVANCE_WEIGHTS  = (0.5, 0.1, 0.1, 0.3)  # (w_qualité, w_type, w_équip, w_distance)

# Compatibilité type_eau_pref (user) → type_norm (site)
TYPE_EXACT = {
    "lac":     "Lac",
    "mer":     "Mer",
    "riviere": "Riviere",
}
TYPE_PARTIAL = {          # compatible mais pas exact → score 0.5
    "mer":     "Cote",
    "riviere": "Transition",
}

EQUIP_SITE_COLS = ["parking", "sanitaires", "pmr", "douche", "poste_secours"]
EQUIP_USER_COLS = ["equip_parking", "equip_sanitaires", "equip_pmr",
                   "equip_douche", "equip_poste_secours"]

# ── Définition des 6 profils ──────────────────────────────────────────────────
# score_min : (mu, sigma) - tiré d'une gaussienne clampée à [0,1]
# dist_max  : (mu, sigma) - tiré d'une gaussienne clampée à [5, 500]

PROFILE_DEFS = {
    "Famille": {
        "score_min":  (0.72, 0.08),
        "dist_max":   (110, 50),
        "type_pref":  ["all", "all", "lac"],    # pondération par répétition
        "p_parking":  0.85, "p_sanitaires": 0.90,
        "p_pmr":      0.30, "p_douche": 0.55, "p_secours": 0.75,
    },
    "Sportif": {
        "score_min":  (0.45, 0.10),
        "dist_max":   (130, 55),
        "type_pref":  ["all", "riviere", "mer", "lac"],
        "p_parking":  0.50, "p_sanitaires": 0.40,
        "p_pmr":      0.10, "p_douche": 0.30, "p_secours": 0.25,
    },
    "Senior": {
        "score_min":  (0.82, 0.07),
        "dist_max":   (35, 20),
        "type_pref":  ["lac", "mer", "lac"],
        "p_parking":  0.90, "p_sanitaires": 0.88,
        "p_pmr":      0.72, "p_douche": 0.62, "p_secours": 0.68,
    },
    "Aventurier": {
        "score_min":  (0.28, 0.12),
        "dist_max":   (165, 65),
        "type_pref":  ["riviere", "riviere", "all"],
        "p_parking":  0.30, "p_sanitaires": 0.20,
        "p_pmr":      0.05, "p_douche": 0.15, "p_secours": 0.20,
    },
    "Vacancier": {
        "score_min":  (0.62, 0.09),
        "dist_max":   (30, 18),
        "type_pref":  ["mer", "mer", "all"],
        "p_parking":  0.72, "p_sanitaires": 0.78,
        "p_pmr":      0.20, "p_douche": 0.52, "p_secours": 0.62,
    },
    "Confort": {
        "score_min":  (0.76, 0.08),
        "dist_max":   (95, 45),
        "type_pref":  ["all", "lac", "mer"],
        "p_parking":  0.92, "p_sanitaires": 0.96,
        "p_pmr":      0.32, "p_douche": 0.78, "p_secours": 0.62,
    },
}


# ── Utilitaires ───────────────────────────────────────────────────────────────

def _haversine_vec(lat0: float, lon0: float,
                   lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
    phi0 = np.radians(lat0)
    phi  = np.radians(lats)
    dphi = np.radians(lats - lat0)
    dlam = np.radians(lons - lon0)
    a = np.sin(dphi/2)**2 + np.cos(phi0) * np.cos(phi) * np.sin(dlam/2)**2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def _norm_type(s: str) -> str:
    s = unicodedata.normalize("NFD", str(s)).encode("ascii", "ignore").decode("ascii").lower()
    if "cotier" in s or "cotiere" in s: return "Cote"
    if "lac"    in s:                   return "Lac"
    if "riviere" in s or "fleuve" in s or "cours" in s: return "Riviere"
    if "mer"    in s:                   return "Mer"
    if "transit" in s:                  return "Transition"
    return "Autre"


# ── Chargement des sites enrichis ─────────────────────────────────────────────

def load_enriched_sites() -> pd.DataFrame:
    """Charge sites_scores.csv (lat/lon, score_expert) + équipements OSM."""
    ss_path = OUTPUT_DIR / "sites_scores.csv"
    if not ss_path.exists():
        raise FileNotFoundError(f"{ss_path} introuvable — lancez main.py d'abord.")

    df = pd.read_csv(ss_path)
    df = (df.sort_values("saison", ascending=False)
            .drop_duplicates("code_site")
            [["code_site", "nom_site", "type_eau", "latitude", "longitude", "score_expert"]]
            .dropna(subset=["latitude", "longitude", "score_expert"])
            .reset_index(drop=True))
    df["type_norm"] = df["type_eau"].map(_norm_type)

    osm_path = OUTPUT_DIR / "osm_equipements.csv"
    if osm_path.exists():
        osm = pd.read_csv(osm_path, sep=";")[["code_site"] + EQUIP_SITE_COLS]
        df = df.merge(osm, on="code_site", how="left")
    for col in EQUIP_SITE_COLS:
        df[col] = df[col].fillna(False).astype(bool)

    return df


# ── Génération des utilisateurs (profils stratifiés) ─────────────────────────

def generate_synthetic_users() -> pd.DataFrame:
    """Génère N_PER_PROFILE × 6 profils = 3000 utilisateurs synthétiques."""
    rng = np.random.default_rng(42)
    rows = []
    user_id = 0

    for profile_name, p in PROFILE_DEFS.items():
        for _ in range(N_PER_PROFILE):
            score_min = float(np.clip(rng.normal(p["score_min"][0], p["score_min"][1]), 0, 1))
            dist_max  = float(np.clip(rng.normal(p["dist_max"][0],  p["dist_max"][1]),  5, 500))
            type_pref = rng.choice(p["type_pref"])

            rows.append({
                "user_id":             f"user_{user_id:05d}",
                "user_type":           profile_name,
                "latitude":            float(rng.uniform(42.0, 51.0)),
                "longitude":           float(rng.uniform(-5.5, 8.0)),
                "score_min":           round(score_min, 4),
                "distance_max":        round(dist_max,  1),
                "type_eau_pref":       type_pref,
                "equip_parking":       int(rng.random() < p["p_parking"]),
                "equip_sanitaires":    int(rng.random() < p["p_sanitaires"]),
                "equip_pmr":           int(rng.random() < p["p_pmr"]),
                "equip_douche":        int(rng.random() < p["p_douche"]),
                "equip_poste_secours": int(rng.random() < p["p_secours"]),
            })
            user_id += 1

    return pd.DataFrame(rows)


# ── Scoring sites × utilisateur (vectorisé) ───────────────────────────────────

def _score_sites(user: pd.Series, df_sites: pd.DataFrame,
                 site_lats: np.ndarray, site_lons: np.ndarray,
                 site_q: np.ndarray, site_types: np.ndarray,
                 site_equip: dict) -> tuple:
    """Score vectorisé pour tous les sites vs un utilisateur. Retourne (scores, dists)."""
    wq, wt, we, wd = RELEVANCE_WEIGHTS
    score_min = float(user["score_min"])
    dist_max  = float(user["distance_max"])
    pref      = str(user["type_eau_pref"])

    # Qualité
    q = np.where(site_q >= score_min, site_q, site_q * 0.5)

    # Type
    if pref == "all":
        type_sc = np.ones(len(df_sites))
    else:
        exact   = TYPE_EXACT.get(pref, "")
        partial = TYPE_PARTIAL.get(pref, "")
        type_sc = np.zeros(len(df_sites))
        type_sc = np.where(site_types == exact,   1.0, type_sc)
        type_sc = np.where(site_types == partial, 0.5, type_sc)

    # Équipements
    wanted = [sc for sc, uc in zip(EQUIP_SITE_COLS, EQUIP_USER_COLS) if user.get(uc, 0)]
    if not wanted:
        equip_sc = np.ones(len(df_sites))
    else:
        present = np.zeros(len(df_sites))
        for col in wanted:
            present += site_equip[col]
        equip_sc = present / len(wanted)

    # Distance - décroissance exponentielle (cohérent avec recommender.py)
    dists   = _haversine_vec(user["latitude"], user["longitude"], site_lats, site_lons)
    dist_sc = np.exp(-dists / max(dist_max, 1.0))

    score = wq * q + wt * type_sc + we * equip_sc + wd * dist_sc
    return score, dists


# ── Génération des interactions ───────────────────────────────────────────────

def generate_synthetic_interactions(df_users: pd.DataFrame,
                                    df_sites: pd.DataFrame) -> pd.DataFrame:
    """
    Top-N_PER_USER sites par score de pertinence pour chaque utilisateur.
    Pas de filtre dur : tous les sites sont évalués, les meilleurs retenus.
    """
    rng = np.random.default_rng(42)

    site_ids   = df_sites["code_site"].values
    site_lats  = df_sites["latitude"].values
    site_lons  = df_sites["longitude"].values
    site_q     = (df_sites["score_expert"].values / 100.0).astype(float)
    site_types = df_sites["type_norm"].values
    site_equip = {c: df_sites[c].values.astype(float) for c in EQUIP_SITE_COLS}

    rows = []
    for _, user in df_users.iterrows():
        score, dists = _score_sites(
            user, df_sites, site_lats, site_lons, site_q, site_types, site_equip
        )

        noisy  = score + rng.normal(0, 0.05, len(score))
        n_take = min(N_PER_USER, len(score))
        top    = np.argsort(-noisy)[:n_take]

        for i in top:
            rating = float(np.clip(3.5 + score[i] * 1.5 + rng.normal(0, 0.1), 3.5, 5.0))
            rows.append({
                "user_id":    user["user_id"],
                "code_site":  site_ids[i],
                "interaction": 1,
                "rating":     rating,
            })

    return pd.DataFrame(rows)


# ── Point d'entrée ────────────────────────────────────────────────────────────

def run() -> None:
    print("Génération des données synthétiques (6 profils × 500 users, distance-aware)...")

    df_sites = load_enriched_sites()
    print(f"  Sites enrichis chargés : {len(df_sites):,}")

    df_users = generate_synthetic_users()
    users_path = OUTPUT_DIR / "synthetic_users.csv"
    df_users.to_csv(users_path, index=False, encoding="utf-8-sig")
    n_users = len(df_users)
    print(f"  Utilisateurs générés   : {n_users:,} → {users_path}")

    # Distribution des profils
    print("\n  Distribution des profils :")
    for pname, cnt in df_users["user_type"].value_counts().items():
        print(f"    {pname:<14} : {cnt:>4}  "
              f"score_min_moy={df_users.loc[df_users['user_type']==pname,'score_min'].mean():.2f}  "
              f"dist_max_moy={df_users.loc[df_users['user_type']==pname,'distance_max'].mean():.0f} km")

    print("\n  Génération des interactions...")
    df_inter = generate_synthetic_interactions(df_users, df_sites)
    inter_path = OUTPUT_DIR / "synthetic_interactions.csv"
    df_inter.to_csv(inter_path, index=False, encoding="utf-8-sig")
    n_inter = len(df_inter)
    print(f"  Interactions générées  : {n_inter:,} ({n_inter/n_users:.0f}/user) → {inter_path}")

    # Vérification : distribution des scores et distance par profil
    print("\n  Vérification (score moyen, distance médiane des interactions) :")
    site_lats_v = df_sites["latitude"].values
    site_lons_v = df_sites["longitude"].values
    site_q_v    = (df_sites["score_expert"].values / 100.0).astype(float)
    site_types_v = df_sites["type_norm"].values
    site_equip_v = {c: df_sites[c].values.astype(float) for c in EQUIP_SITE_COLS}
    code_to_idx  = {c: i for i, c in enumerate(df_sites["code_site"].values)}

    for ut, grp_u in df_users.groupby("user_type"):
        scores_all, dists_all = [], []
        for _, u in grp_u.head(50).iterrows():  # échantillon 50 users/profil pour vitesse
            u_inter = df_inter[df_inter["user_id"] == u["user_id"]]
            if u_inter.empty:
                continue
            idxs = [code_to_idx[c] for c in u_inter["code_site"] if c in code_to_idx]
            if not idxs:
                continue
            sc, ds = _score_sites(u, df_sites, site_lats_v, site_lons_v,
                                  site_q_v, site_types_v, site_equip_v)
            scores_all.extend(sc[idxs].tolist())
            dists_all.extend(ds[idxs].tolist())

        if scores_all:
            print(f"    {ut:<14} score_moy={np.mean(scores_all):.3f}  "
                  f"dist_médiane={np.median(dists_all):6.1f} km  "
                  f"dist_max_moy={grp_u['distance_max'].mean():.0f} km")

    print("Done.")


if __name__ == "__main__":
    run()
