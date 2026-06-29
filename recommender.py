"""
recommender.py - AquaReco Semaine 5
Moteur de recommandation : 3 approches (Content-based, Collaborative, LambdaMART).
Split 70/10/20 (train/validation/test) par utilisateur.
Grid search sur la validation pour optimiser les poids du content-based.
"""

import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from pathlib import Path
import unicodedata

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
import lightgbm as lgb

OUTPUT_DIR = Path(__file__).parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

EARTH_RADIUS_KM = 6371

EQUIP_SITE_COLS = ["parking", "sanitaires", "pmr", "douche", "poste_secours"]
EQUIP_USER_COLS = ["equip_parking", "equip_sanitaires", "equip_pmr",
                   "equip_douche", "equip_poste_secours"]

TYPE_EXACT   = {"lac": "Lac", "mer": "Mer", "riviere": "Riviere"}
TYPE_PARTIAL = {"mer": "Cote", "riviere": "Transition"}   # compatible → 0.5

DEFAULT_WEIGHTS = (0.4, 0.2, 0.1, 0.3)   # (w_qualité, w_type, w_équip, w_distance)


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


# ── Chargement ────────────────────────────────────────────────────────────────

def load_data():
    ss_path = OUTPUT_DIR / "sites_scores.csv"
    if not ss_path.exists():
        raise FileNotFoundError("sites_scores.csv introuvable.")

    df_scores = pd.read_csv(ss_path)
    df_sites  = (df_scores
                 .sort_values("saison", ascending=False)
                 .drop_duplicates("code_site")
                 [["code_site", "nom_site", "type_eau",
                   "latitude", "longitude", "score_expert"]]
                 .dropna(subset=["latitude", "longitude", "score_expert"])
                 .reset_index(drop=True))
    df_sites["type_norm"] = df_sites["type_eau"].map(_norm_type)

    osm_path = OUTPUT_DIR / "osm_equipements.csv"
    if osm_path.exists():
        osm = pd.read_csv(osm_path, sep=";")[["code_site"] + EQUIP_SITE_COLS]
        df_sites = df_sites.merge(osm, on="code_site", how="left")
    for col in EQUIP_SITE_COLS:
        df_sites[col] = df_sites[col].fillna(False).astype(bool)

    df_users = pd.read_csv(OUTPUT_DIR / "synthetic_users.csv")
    df_inter = pd.read_csv(OUTPUT_DIR / "synthetic_interactions.csv")

    n_users = df_users["user_id"].nunique()
    print(f"  Sites  : {len(df_sites):,}")
    print(f"  Users  : {n_users:,}")
    print(f"  Inter. : {len(df_inter):,}  ({len(df_inter)/n_users:.0f}/user)")
    return df_sites, df_users, df_inter


# ── Split 70 / 10 / 20 ───────────────────────────────────────────────────────

def make_three_way_split(df_inter: pd.DataFrame,
                         val_ratio: float = 0.10,
                         test_ratio: float = 0.20,
                         seed: int = 42):
    """
    Split aléatoire par utilisateur.
    Retourne (train_inter, val_sets, test_sets).
    val_sets  = {user_id: set(code_site)}  - utilisé uniquement pour la grille de poids.
    test_sets = {user_id: set(code_site)}  - utilisé uniquement pour l'évaluation finale.
    """
    rng = np.random.default_rng(seed)
    train_rows = []
    val_sets   = {}
    test_sets  = {}

    for user_id, grp in df_inter.groupby("user_id"):
        idx = grp.index.tolist()
        rng.shuffle(idx)
        n        = len(idx)
        n_test   = max(1, int(n * test_ratio))
        n_val    = max(1, int(n * val_ratio))
        n_train  = n - n_test - n_val
        if n_train < 1:                # garantir ≥ 1 interaction en train
            n_val  = max(0, n_val - 1)
            n_train = n - n_test - n_val

        test_idx  = idx[:n_test]
        val_idx   = idx[n_test: n_test + n_val]
        train_idx = idx[n_test + n_val:]

        if train_idx:
            train_rows.append(grp.loc[train_idx])
        val_sets[user_id]  = set(grp.loc[val_idx,  "code_site"]) if val_idx  else set()
        test_sets[user_id] = set(grp.loc[test_idx, "code_site"])

    train_inter = pd.concat(train_rows).reset_index(drop=True)
    return train_inter, val_sets, test_sets


# ── Pré-calcul des composantes (pour la grille de recherche) ──────────────────

def precompute_cb_components(df_sites: pd.DataFrame, df_users: pd.DataFrame):
    """
    Pré-calcule pour chaque paire (user, site) les 4 composantes :
      q_arr      - qualité pénalisée
      type_arr   - correspondance type d'eau
      eq_arr     - proportion d'équipements souhaités présents
      dp_arr     - pénalité distance (× 0 si hors rayon)
    Retourne (q_arr, type_arr, eq_arr, dp_arr, site_ids) en float32.
    Mémoire : 4 × n_users × n_sites × 4 octets.
    """
    n_u = len(df_users)
    n_s = len(df_sites)

    site_ids   = df_sites["code_site"].values
    site_lats  = df_sites["latitude"].values
    site_lons  = df_sites["longitude"].values
    site_q_raw = (df_sites["score_expert"].values / 100.0).astype(np.float32)
    site_types = df_sites["type_norm"].values
    site_equip = {c: df_sites[c].values.astype(np.float32) for c in EQUIP_SITE_COLS}

    q_arr  = np.empty((n_u, n_s), dtype=np.float32)
    t_arr  = np.empty((n_u, n_s), dtype=np.float32)
    eq_arr = np.empty((n_u, n_s), dtype=np.float32)
    ds_arr = np.empty((n_u, n_s), dtype=np.float32)   # distance score (exp)

    for i, (_, user) in enumerate(df_users.iterrows()):
        score_min = float(user["score_min"])
        dist_max  = float(user["distance_max"])
        pref      = str(user["type_eau_pref"])

        q = np.where(site_q_raw >= score_min, site_q_raw, site_q_raw * 0.5)
        q_arr[i] = q

        if pref == "all":
            t_arr[i] = 1.0
        else:
            exact   = TYPE_EXACT.get(pref, "")
            partial = TYPE_PARTIAL.get(pref, "")
            ts      = np.zeros(n_s, dtype=np.float32)
            ts      = np.where(site_types == exact,   1.0, ts)
            ts      = np.where(site_types == partial, 0.5, ts)
            t_arr[i] = ts

        wanted = [sc for sc, uc in zip(EQUIP_SITE_COLS, EQUIP_USER_COLS) if user.get(uc, 0)]
        if not wanted:
            eq_arr[i] = 1.0
        else:
            present = np.zeros(n_s, dtype=np.float32)
            for col in wanted:
                present += site_equip[col]
            eq_arr[i] = present / len(wanted)

        dists      = _haversine_vec(user["latitude"], user["longitude"], site_lats, site_lons)
        ds_arr[i]  = np.exp(-dists / max(dist_max, 1.0)).astype(np.float32)

    return q_arr, t_arr, eq_arr, ds_arr, site_ids


# ── Grille de recherche des poids (sur le jeu de validation) ─────────────────

def grid_search_weights(q_arr, t_arr, eq_arr, ds_arr, site_ids,
                        df_users: pd.DataFrame,
                        train_inter: pd.DataFrame,
                        val_sets: dict,
                        k: int = 5) -> pd.DataFrame:
    """
    Teste toutes les combinaisons (wq, wt, we, wd) avec wq+wt+we+wd=1.
    La distance est la 4ème composante additive (décroissance exponentielle).
    Évalue Precision@k sur val_sets uniquement - jamais sur le test.
    """
    combos = [
        (wq, wt, we, wd)
        for wq in [0.3, 0.4, 0.5]
        for wt in [0.1, 0.2, 0.3]
        for we in [0.1, 0.2]
        for wd in [0.1, 0.2, 0.3]
        if abs(wq + wt + we + wd - 1.0) < 1e-9
    ]
    print(f"    {len(combos)} combinaisons valides à tester...")

    train_lookup = {uid: set(grp["code_site"])
                    for uid, grp in train_inter.groupby("user_id")}
    user_ids = df_users["user_id"].values

    results = []
    for wq, wt, we, wd in combos:
        precisions = []
        for i, uid in enumerate(user_ids):
            rel = val_sets.get(uid, set())
            if not rel:
                continue
            scores = wq * q_arr[i] + wt * t_arr[i] + we * eq_arr[i] + wd * ds_arr[i]
            train_s = train_lookup.get(uid, set())
            if train_s:
                mask   = np.isin(site_ids, list(train_s))
                scores = np.where(mask, -1.0, scores)
            top = [site_ids[j] for j in np.argsort(-scores)[:k] if scores[j] >= 0][:k]
            precisions.append(len(set(top) & rel) / k)

        results.append({
            "wq": wq, "wt": wt, "we": we, "wd": wd,
            "precision_val@5": round(np.mean(precisions), 4) if precisions else 0.0,
        })

    return pd.DataFrame(results).sort_values("precision_val@5", ascending=False).reset_index(drop=True)


# ── Approche 1 : Content-based (vectorisé) ────────────────────────────────────

def approach_content_based(df_sites: pd.DataFrame,
                            df_users: pd.DataFrame,
                            train_inter: pd.DataFrame,
                            k: int = 10,
                            weights: tuple = DEFAULT_WEIGHTS) -> dict:
    """
    Scoring vectorisé par utilisateur avec distance comme 4ème composante additive.
    score = wq·qualité + wt·type + we·équip + wd·exp(-d/dist_max)
    Pas de coupure brutale : la décroissance exponentielle pénalise graduellement
    les sites éloignés (0.37 à dist_max, 0.14 à 2×dist_max).
    Exclut les sites du jeu d'entraînement.
    """
    wq, wt, we, wd = weights
    recommendations = {}

    site_ids   = df_sites["code_site"].values
    site_lats  = df_sites["latitude"].values
    site_lons  = df_sites["longitude"].values
    site_q_raw = df_sites["score_expert"].values / 100.0
    site_types = df_sites["type_norm"].values
    site_equip = {c: df_sites[c].values.astype(float) for c in EQUIP_SITE_COLS}

    train_lookup = {uid: set(grp["code_site"])
                    for uid, grp in train_inter.groupby("user_id")}

    for _, user in df_users.iterrows():
        uid       = user["user_id"]
        score_min = float(user["score_min"])
        dist_max  = float(user["distance_max"])
        pref      = str(user["type_eau_pref"])

        q = np.where(site_q_raw >= score_min, site_q_raw, site_q_raw * 0.5)

        if pref == "all":
            type_sc = np.ones(len(df_sites))
        else:
            exact   = TYPE_EXACT.get(pref, "")
            partial = TYPE_PARTIAL.get(pref, "")
            type_sc = np.zeros(len(df_sites))
            type_sc = np.where(site_types == exact,   1.0, type_sc)
            type_sc = np.where(site_types == partial, 0.5, type_sc)

        wanted = [sc for sc, uc in zip(EQUIP_SITE_COLS, EQUIP_USER_COLS) if user.get(uc, 0)]
        if not wanted:
            equip_sc = np.ones(len(df_sites))
        else:
            present = np.zeros(len(df_sites))
            for col in wanted:
                present += site_equip[col]
            equip_sc = present / len(wanted)

        dists    = _haversine_vec(user["latitude"], user["longitude"], site_lats, site_lons)
        dist_sc  = np.exp(-dists / max(dist_max, 1.0))
        final    = wq * q + wt * type_sc + we * equip_sc + wd * dist_sc

        train_s = train_lookup.get(uid, set())
        if train_s:
            final = np.where(np.isin(site_ids, list(train_s)), -1.0, final)

        top_idx = np.argsort(-final)
        recommendations[uid] = [site_ids[i] for i in top_idx if final[i] >= 0][:k]

    return recommendations


# ── Approche 2 : Filtrage collaboratif ────────────────────────────────────────

def approach_collaborative(df_sites: pd.DataFrame,
                            df_users: pd.DataFrame,
                            train_inter: pd.DataFrame,
                            k: int = 10) -> dict:
    """
    User-based CF entraîné sur train_inter.
    Matrice sparse pour les grands jeux de données.
    Les sites test (absents du train) peuvent être recommandés.
    """
    inter_pivot = train_inter.pivot_table(
        index="user_id", columns="code_site", values="interaction", fill_value=0
    )
    pivot_index = list(inter_pivot.index)
    sparse_mat  = csr_matrix(inter_pivot.values, dtype=np.float32)
    sim_matrix  = cosine_similarity(sparse_mat)

    train_lookup = {uid: set(grp["code_site"])
                    for uid, grp in train_inter.groupby("user_id")}
    recommendations = {}

    for user_idx, user_id in enumerate(pivot_index):
        sim_row   = sim_matrix[user_idx]
        neighbors = [n for n in np.argsort(-sim_row) if n != user_idx][:k]

        train_s   = train_lookup.get(user_id, set())
        scores    = {}
        for n_idx in neighbors:
            n_id  = pivot_index[n_idx]
            n_sim = float(sim_row[n_idx])
            for site in train_inter[train_inter["user_id"] == n_id]["code_site"].unique():
                if site not in train_s:
                    scores[site] = scores.get(site, 0.0) + n_sim

        recommendations[user_id] = [s for s, _ in
                                     sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]]

    for _, user in df_users.iterrows():
        if user["user_id"] not in recommendations:
            recommendations[user["user_id"]] = []

    return recommendations


# ── Approche 3 : LambdaMART (regression sur rating) ──────────────────────────

def approach_lambdamart(df_sites: pd.DataFrame,
                        df_users: pd.DataFrame,
                        train_inter: pd.DataFrame,
                        k: int = 10) -> dict:
    """
    Ranking supervisé (pointwise) entraîné sur les ratings du train set.
    Prédiction vectorisée par utilisateur (un appel model.predict par user).
    """
    # ── Entraînement sur un sous-échantillon
    sample_users = df_users.sample(min(100, len(df_users)), random_state=42)
    sample_sites = df_sites.head(1000).reset_index(drop=True)

    train_avg = (train_inter
                 .groupby(["user_id", "code_site"])["rating"]
                 .mean()
                 .reset_index())
    train_dict = {(r.user_id, r.code_site): r.rating
                  for r in train_avg.itertuples(index=False)}

    s_lats = sample_sites["latitude"].values
    s_lons = sample_sites["longitude"].values
    s_q    = (sample_sites["score_expert"].values / 100.0)

    feat_rows, labels = [], []
    for _, user in sample_users.iterrows():
        dists = _haversine_vec(user["latitude"], user["longitude"], s_lats, s_lons)
        for j, (_, site) in enumerate(sample_sites.iterrows()):
            feat_rows.append([
                user["score_min"],
                user["distance_max"] / 150.0,
                dists[j] / 500.0,
                s_q[j],
                float(site.get("parking", False)),
                float(site.get("sanitaires", False)),
                float(site.get("pmr", False)),
            ])
            labels.append(train_dict.get((user["user_id"], site["code_site"]), 0.0))

    X_train = np.array(feat_rows, dtype=np.float32)
    y_train = np.array(labels,    dtype=np.float32)

    model = None
    try:
        ds = lgb.Dataset(X_train, label=y_train)
        params = {"objective": "regression", "metric": "rmse",
                  "num_leaves": 31, "learning_rate": 0.05, "verbose": -1}
        model = lgb.train(params, ds, num_boost_round=50)
    except Exception as e:
        print(f"  [WARN] LambdaMART entraînement échoué : {e}")

    # ── Prédiction vectorisée (un appel par utilisateur)
    site_ids  = df_sites["code_site"].values
    site_lats = df_sites["latitude"].values
    site_lons = df_sites["longitude"].values
    site_q    = (df_sites["score_expert"].values / 100.0).astype(np.float32)
    site_park = df_sites["parking"].values.astype(np.float32)
    site_san  = df_sites["sanitaires"].values.astype(np.float32)
    site_pmr  = df_sites["pmr"].values.astype(np.float32)
    n_s       = len(df_sites)

    train_lookup = {uid: set(grp["code_site"])
                    for uid, grp in train_inter.groupby("user_id")}
    recommendations = {}

    for _, user in df_users.iterrows():
        uid = user["user_id"]

        if model is None:
            train_s = train_lookup.get(uid, set())
            cands   = df_sites[~df_sites["code_site"].isin(train_s)]
            recommendations[uid] = list(
                cands["code_site"].sample(min(k, len(cands)), random_state=42).values
            )
            continue

        dists = _haversine_vec(user["latitude"], user["longitude"], site_lats, site_lons)
        X = np.column_stack([
            np.full(n_s, user["score_min"],         dtype=np.float32),
            np.full(n_s, user["distance_max"]/150.0, dtype=np.float32),
            (dists / 500.0).astype(np.float32),
            site_q, site_park, site_san, site_pmr,
        ])
        preds = model.predict(X)

        train_s = train_lookup.get(uid, set())
        if train_s:
            preds[np.isin(site_ids, list(train_s))] = -1.0

        top_idx = np.argsort(-preds)
        recommendations[uid] = [site_ids[i] for i in top_idx if preds[i] >= 0][:k]

    return recommendations


# ── Calcul des métriques ──────────────────────────────────────────────────────

def compute_metrics(recommendations: dict,
                    test_sets: dict,
                    k_values: list = [5, 10]) -> dict:
    """Évalue Precision@k, NDCG@k et MAP sur le jeu de test uniquement."""
    metrics = {}
    for k in k_values:
        prec, ndcg = [], []
        for uid, top_all in recommendations.items():
            rel = test_sets.get(uid, set())
            if not rel:
                continue
            top_k = top_all[:k]
            hits  = len(set(top_k) & rel)
            prec.append(hits / k)
            g = sum(1 / np.log2(i + 2) for i, s in enumerate(top_k) if s in rel)
            ideal = sum(1 / np.log2(i + 2) for i in range(min(len(rel), k)))
            ndcg.append(g / ideal if ideal > 0 else 0)
        metrics[f"precision@{k}"] = round(np.mean(prec), 4) if prec else 0.0
        metrics[f"ndcg@{k}"]      = round(np.mean(ndcg), 4) if ndcg else 0.0

    maps = []
    for uid, top_all in recommendations.items():
        rel = test_sets.get(uid, set())
        if not rel:
            continue
        ap, hits = 0.0, 0
        for i, s in enumerate(top_all):
            if s in rel:
                hits += 1
                ap   += hits / (i + 1)
        maps.append(ap / len(rel))
    metrics["map"] = round(np.mean(maps), 4) if maps else 0.0
    return metrics


# ── Exemples de recommandations ───────────────────────────────────────────────

def _show_example(user: pd.Series, profil: str,
                  rec_cb, rec_cf, rec_lm, test_sets, df_sites) -> None:
    uid      = user["user_id"]
    test_ids = test_sets.get(uid, set())
    print(f"\n  Profil : {profil.upper()}  (id={uid})")
    print(f"    score_min={user['score_min']:.2f}  dist_max={user['distance_max']:.0f} km  "
          f"type={user['type_eau_pref']}")
    print(f"    Vrais positifs (test) : {len(test_ids)}")
    for label, recs in [("Content-based", rec_cb), ("Collaboratif ", rec_cf),
                         ("LambdaMART   ", rec_lm)]:
        top5 = recs.get(uid, [])[:5]
        hits = len(set(top5) & test_ids)
        print(f"    {label} Top-5 : {top5}  hits={hits}")


def _show_distance_comparison(user: pd.Series, rec_cb: dict,
                               test_sets: dict, df_sites: pd.DataFrame,
                               best_weights: tuple) -> None:
    """
    Affiche pour les 5 sites recommandés : distance, score exponentiel (nouveau)
    et ancien score de pénalité linéaire - pour valider la décroissance douce.
    """
    uid      = user["user_id"]
    dist_max = float(user["distance_max"])
    test_ids = test_sets.get(uid, set())
    top5     = rec_cb.get(uid, [])[:5]

    site_map  = df_sites.set_index("code_site")
    user_lat, user_lon = user["latitude"], user["longitude"]

    wq, wt, we, wd = best_weights
    print(f"\n  Décroissance distance pour {user['user_type'].upper()} (dist_max={dist_max:.0f} km)")
    print(f"  {'Site':<14}  {'Dist km':>8}  {'exp(-d/dmax)':>13}  "
          f"{'Ancien (lin)':>13}  {'Contrib dist':>13}  Hit?")
    print("  " + "─" * 72)

    for sid in top5:
        if sid not in site_map.index:
            continue
        row  = site_map.loc[sid]
        d    = float(_haversine_vec(user_lat, user_lon,
                                    np.array([row["latitude"]]),
                                    np.array([row["longitude"]]))[0])
        new_ds = float(np.exp(-d / max(dist_max, 1.0)))
        old_dp = float(max(0.0, 1.0 - d / dist_max))
        hit    = "✓" if sid in test_ids else "✗"
        print(f"  {sid:<14}  {d:>8.1f}  {new_ds:>13.4f}  "
              f"{old_dp:>13.4f}  {wd * new_ds:>13.4f}  {hit}")


# ── Point d'entrée ────────────────────────────────────────────────────────────

def run_recommender() -> None:
    print("=" * 70)
    print("  MOTEUR DE RECOMMANDATION — AquaReco")
    print("=" * 70)

    print("\n  Chargement des données...")
    df_sites, df_users, df_inter = load_data()

    print("\n  Split 70/10/20 (train / validation / test) par utilisateur...")
    train_inter, val_sets, test_sets = make_three_way_split(df_inter)
    n_val  = sum(len(v) for v in val_sets.values())
    n_test = sum(len(v) for v in test_sets.values())
    print(f"    Train       : {len(train_inter):,} interactions")
    print(f"    Validation  : {n_val:,} vrais positifs  ({n_val/len(val_sets):.1f}/user)")
    print(f"    Test        : {n_test:,} vrais positifs  ({n_test/len(test_sets):.1f}/user)")

    # ── Content-based avec poids par défaut (4 composantes dont distance exp)
    dw = DEFAULT_WEIGHTS
    print(f"\n  Content-based avec poids défaut ({dw[0]}/{dw[1]}/{dw[2]}/{dw[3]})...")
    rec_cb_def     = approach_content_based(df_sites, df_users, train_inter,
                                             weights=DEFAULT_WEIGHTS)
    metrics_cb_def = compute_metrics(rec_cb_def, test_sets)

    # ── Grid search sur la validation (jamais sur le test)
    print("\n  Grid search des poids (sur jeu de validation uniquement)...")
    q_arr, t_arr, eq_arr, ds_arr, site_ids = precompute_cb_components(df_sites, df_users)
    df_grid = grid_search_weights(q_arr, t_arr, eq_arr, ds_arr, site_ids,
                                  df_users, train_inter, val_sets, k=5)

    print("\n  Top 5 combinaisons de poids (validation) :")
    print(f"  {'wq':>5}  {'wt':>5}  {'we':>5}  {'wd':>5}  {'Precision_val@5':>16}")
    for _, row in df_grid.head(5).iterrows():
        print(f"  {row['wq']:>5.1f}  {row['wt']:>5.1f}  {row['we']:>5.1f}  "
              f"{row['wd']:>5.1f}  {row['precision_val@5']:>16.4f}")

    best   = df_grid.iloc[0]
    best_w = (best["wq"], best["wt"], best["we"], best["wd"])
    print(f"\n  → Meilleurs poids retenus : "
          f"wq={best_w[0]}, wt={best_w[1]}, we={best_w[2]}, wd={best_w[3]}")

    # ── Content-based avec poids optimisés
    print("\n  Content-based avec poids optimisés...")
    rec_cb_opt     = approach_content_based(df_sites, df_users, train_inter,
                                             weights=best_w)
    metrics_cb_opt = compute_metrics(rec_cb_opt, test_sets)

    # ── Collaborative et LambdaMART
    print("  Collaborative filtering...")
    rec_cf = approach_collaborative(df_sites, df_users, train_inter)

    print("  LambdaMART...")
    rec_lm = approach_lambdamart(df_sites, df_users, train_inter)

    metrics_cf = compute_metrics(rec_cf, test_sets)
    metrics_lm = compute_metrics(rec_lm, test_sets)

    # ── Tableau comparatif avant / après
    def_label = f"Content-based défaut  ({dw[0]}/{dw[1]}/{dw[2]}/{dw[3]})"
    opt_label = f"Content-based optimisé ({best_w[0]}/{best_w[1]}/{best_w[2]}/{best_w[3]})"
    results = [
        {"approach": def_label, **metrics_cb_def},
        {"approach": opt_label, **metrics_cb_opt},
        {"approach": "Collaborative",  **metrics_cf},
        {"approach": "LambdaMART",     **metrics_lm},
    ]
    df_comp = pd.DataFrame(results)
    out_csv = OUTPUT_DIR / "recommender_comparison.csv"
    df_comp.to_csv(out_csv, index=False, encoding="utf-8-sig")

    print(f"\n  {'='*70}")
    print(f"  TABLEAU COMPARATIF — avant / après optimisation des poids")
    print(f"  {'='*70}")
    print(df_comp.to_string(index=False))
    print(f"\n  Fichier sauvegardé : {out_csv}")

    # ── Exemples Famille + Aventurier + décroissance distance
    print(f"\n  {'='*70}")
    print(f"  EXEMPLES DE RECOMMANDATIONS (poids optimisés)")
    print(f"  {'='*70}")
    famille_user = None
    for profil in ["Famille", "Aventurier"]:
        grp = df_users[df_users["user_type"] == profil]
        if grp.empty:
            continue
        u = grp.iloc[0]
        _show_example(u, profil, rec_cb_opt, rec_cf, rec_lm, test_sets, df_sites)
        if profil == "Famille":
            famille_user = u

    # ── Décroissance douce : comparaison ancienne / nouvelle formule (utilisateur Famille)
    if famille_user is not None:
        print(f"\n  {'='*70}")
        print(f"  VALIDATION DÉCROISSANCE EXPONENTIELLE — utilisateur Famille")
        print(f"  {'='*70}")
        _show_distance_comparison(famille_user, rec_cb_opt, test_sets, df_sites, best_w)


if __name__ == "__main__":
    run_recommender()
