"""
investigate_metrics.py - Diagnostic des métriques du moteur de recommandation AquaReco
Ne modifie aucun modèle, observe seulement.
"""

import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from pathlib import Path
import unicodedata

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

OUTPUT_DIR = Path(__file__).parent / "outputs"
EARTH_RADIUS_KM = 6371


# ── Utilitaires ───────────────────────────────────────────────────────────────

def haversine_vec(lat0, lon0, lats, lons):
    phi0 = np.radians(lat0)
    phi  = np.radians(lats)
    dphi = np.radians(lats - lat0)
    dlam = np.radians(lons - lon0)
    a = np.sin(dphi/2)**2 + np.cos(phi0)*np.cos(phi)*np.sin(dlam/2)**2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def norm_type(s):
    s = unicodedata.normalize("NFD", str(s)).encode("ascii", "ignore").decode("ascii").lower()
    if "cotier" in s or "cotiere" in s: return "Cote"
    if "lac" in s: return "Lac"
    if "riviere" in s or "fleuve" in s or "cours" in s: return "Riviere"
    if "mer" in s: return "Mer"
    if "transit" in s: return "Transition"
    return "Autre"


def sep(char="─", n=70):
    print(char * n)


# ── Chargement ────────────────────────────────────────────────────────────────

def load():
    df_features = pd.read_parquet(OUTPUT_DIR / "features_temporal.parquet")
    df_users    = pd.read_csv(OUTPUT_DIR / "synthetic_users.csv")
    df_inter    = pd.read_csv(OUTPUT_DIR / "synthetic_interactions.csv")

    # Sites enrichis : une ligne par site (saison la plus récente)
    df_scores = pd.read_csv(OUTPUT_DIR / "sites_scores.csv")
    df_sites = (df_scores
                .sort_values("saison", ascending=False)
                .drop_duplicates("code_site")
                [["code_site", "nom_site", "type_eau", "latitude", "longitude",
                  "score_expert", "departement"]]
                .dropna(subset=["latitude", "longitude", "score_expert"])
                .reset_index(drop=True))
    df_sites["type_norm"] = df_sites["type_eau"].map(norm_type)

    return df_features, df_sites, df_users, df_inter


# ── Recopie exacte de recommender.py:approach_content_based ───────────────────

def content_scores_buggues(user, df_sites):
    """Score tel quel dans recommender.py - pour montrer le bug."""
    scores = []
    for _, site in df_sites.iterrows():
        site_id       = site["code_site"]
        quality_score = 0.5                                  # BUG: hardcodé

        pref_type  = user.get("type_eau_pref", "all")       # colonne bien lue
        type_match = 1.0 if pref_type == "all" else 0.7     # BUG: pas de comparaison site/user

        equip_match  = 0.5                                   # BUG: hardcodé
        final_score  = quality_score * 0.5 + type_match * 0.3 + equip_match * 0.2
        scores.append((site_id, final_score))
    return scores


# ── Score correct (logique synthetic_users.py) ────────────────────────────────

def content_scores_corrects(df_sites, user_lat, user_lon,
                             score_min_norm, dist_max_km,
                             type_pref_str, need_pmr, need_san):
    """
    Reproduit la fonction relevance_score de synthetic_users.py.
    score_min_norm : valeur normalisée [0,1] (score_min du CSV des users).
    """
    lats  = df_sites["latitude"].values
    lons  = df_sites["longitude"].values
    dists = haversine_vec(user_lat, user_lon, lats, lons)

    # Qualité [0,1] avec pénalité si sous le seuil
    q      = df_sites["score_expert"].values / 100.0
    # score_min du CSV est déjà en [0,1] ; on compare à score_expert/100
    q_pen  = np.where(q >= score_min_norm, q, q * 0.5)

    # Distance
    if dist_max_km >= 5000:
        d_score = np.ones(len(df_sites))
    else:
        d_score = np.exp(-dists / max(dist_max_km, 1.0))
        d_score = np.where(dists > dist_max_km * 2, d_score * 0.2, d_score)

    # Type d'eau préféré
    type_pref_list = [t.strip() for t in type_pref_str.split("|")
                      if t.strip() and t.strip() != "indifferent"]
    if type_pref_list:
        type_score = np.where(df_sites["type_norm"].isin(type_pref_list), 1.0, 0.4)
    else:
        type_score = np.ones(len(df_sites))

    # Équipements (colonnes absentes dans df_sites → eq_score constant)
    eq_score = np.ones(len(df_sites))

    return 0.40 * q_pen + 0.30 * d_score + 0.20 * type_score + 0.10 * eq_score, dists


# ═══════════════════════════════════════════════════════════════════════════════
# PARTIE 1 - Profils utilisateurs × recommandations × vrais positifs
# ═══════════════════════════════════════════════════════════════════════════════

def part1_profiles(df_features, df_sites, df_users, df_inter):
    print()
    sep("═")
    print("PARTIE 1 — Profils × Recommandations × Vrais positifs")
    sep("═")

    # Profils disponibles dans les données réelles
    available = df_users["user_type"].unique().tolist()
    target = available[:3]   # les 3 premiers profils présents
    print(f"  Profils disponibles : {available}")
    print(f"  Profils analysés    : {target}")

    inter_pivot = df_inter.pivot_table(
        index="user_id", columns="code_site", values="interaction", fill_value=0
    )
    pivot_index = list(inter_pivot.index)
    sim_matrix  = cosine_similarity(inter_pivot.values)

    for profil in target:
        users_profil = df_users[df_users["user_type"] == profil]
        if users_profil.empty:
            continue

        user    = users_profil.iloc[0]
        user_id = user["user_id"]

        print(f"\n{'─'*70}")
        print(f"  Profil : {profil.upper()}  (user_id={user_id})")
        sep("─")

        # Préférences
        print("  Préférences :")
        print(f"    score_min      = {user['score_min']:.4f}  (plage [0,1])")
        print(f"    distance_max   = {user['distance_max']} km")
        print(f"    type_eau_pref  = {user['type_eau_pref']}")
        print(f"    equip_pmr      = {user['equip_pmr']}")
        print(f"    equip_sanitaires = {user['equip_sanitaires']}")

        # Vrais positifs
        true_df   = df_inter[df_inter["user_id"] == user_id].sort_values("rating", ascending=False)
        true_ids  = set(true_df["code_site"].tolist())
        print(f"\n  Vrais positifs (interactions) : {len(true_ids)} sites")
        top5_true = true_df.head(5)
        for _, row in top5_true.iterrows():
            site_info = df_sites[df_sites["code_site"] == row["code_site"]]
            nom = site_info["nom_site"].values[0][:35] if not site_info.empty else "?"
            print(f"    {row['code_site']:<14}  rating={row['rating']:.2f}  {nom}")

        # Recommandations content-based (bugguées)
        scores_bug = content_scores_buggues(user, df_sites)
        top5_bug   = sorted(scores_bug, key=lambda x: x[1], reverse=True)[:5]
        unique_sc  = set(s for _, s in scores_bug)
        print(f"\n  Content-based (bugué) — Top 5 :")
        for rank, (sid, sc) in enumerate(top5_bug, 1):
            hit = "✓ HIT" if sid in true_ids else "✗"
            print(f"    [{rank}] {sid:<14}  score={sc:.4f}  {hit}")
        print(f"    → Scores distincts : {len(unique_sc)}  (valeurs : {sorted(unique_sc)})")

        # Recommandations collaboratives
        if user_id in pivot_index:
            u_pos     = pivot_index.index(user_id)
            sim_row   = sim_matrix[u_pos]
            neighbors = [i for i in np.argsort(-sim_row) if i != u_pos][:10]

            rec_cf = {}
            for n_idx in neighbors:
                n_id     = pivot_index[n_idx]
                n_sites  = df_inter[df_inter["user_id"] == n_id]["code_site"].unique()
                for s in n_sites:
                    if s not in true_ids:          # exclusion explicite des vrais positifs
                        rec_cf[s] = rec_cf.get(s, 0) + 1

            top5_cf = sorted(rec_cf.items(), key=lambda x: x[1], reverse=True)[:5]
            print(f"\n  Collaboratif — Top 5 :")
            for rank, (sid, cnt) in enumerate(top5_cf, 1):
                hit = "✓ HIT" if sid in true_ids else "✗"
                print(f"    [{rank}] {sid:<14}  votes={cnt}  {hit}")
            print(f"    → Vrais positifs exclus explicitement → Precision = 0 garanti")
        else:
            print(f"\n  Collaboratif : user_id={user_id} absent de la matrice.")

        # Explication
        type_pref_val = user.get("type_eau_pref", "all")
        type_match_val = 1.0 if type_pref_val == "all" else 0.7
        expected_score = 0.5 * 0.5 + type_match_val * 0.3 + 0.5 * 0.2
        print(f"\n  Diagnostic :")
        print(f"    Content-based : quality=0.5 (hardcodé), equip=0.5 (hardcodé),")
        print(f"    type_eau_pref='{type_pref_val}' → type_match={type_match_val}")
        print(f"    → Tous les sites obtiennent score={expected_score:.4f}")
        print(f"       Le classement dépend de l'ordre des lignes dans df_sites, pas des préférences.")
        print(f"    Collaboratif : exclut user_visited = vrais positifs → 0 hit possible.")


# ═══════════════════════════════════════════════════════════════════════════════
# PARTIE 2 - Analyse du calcul de Precision@k et qualité du jeu de test
# ═══════════════════════════════════════════════════════════════════════════════

def part2_precision_analysis(df_features, df_sites, df_users, df_inter):
    print()
    sep("═")
    print("PARTIE 2 — Définition des vrais positifs et qualité du jeu de test")
    sep("═")

    print("""
  Définition d'un vrai positif dans compute_metrics (recommender.py:200) :
    relevant = set(df_inter[df_inter['user_id'] == user_id]['code_site'].unique())

  Problème : ce sont TOUTES les interactions, il n'y a pas de split train/test.
  Le collaboratif exclut ces mêmes interactions → Precision = 0 par construction.
""")

    inter_per_user = df_inter.groupby("user_id")["code_site"].nunique()
    print(f"  Interactions par utilisateur :")
    print(f"    Minimum  : {inter_per_user.min()}")
    print(f"    Maximum  : {inter_per_user.max()}")
    print(f"    Moyenne  : {inter_per_user.mean():.1f}")
    print(f"    Médiane  : {inter_per_user.median():.0f}")
    print(f"    → Chaque utilisateur a en moyenne {inter_per_user.mean():.0f} interactions")

    n_sites = df_sites["code_site"].nunique()
    mean_rel = inter_per_user.mean()
    p_hit_random = 1 - ((n_sites - mean_rel) / n_sites) ** 5
    print(f"\n  Probabilité de hit aléatoire :")
    print(f"    Sites uniques dans sites_scores : {n_sites:,}")
    print(f"    P(≥1 hit en Top-5 aléatoire)   ≈ {p_hit_random*100:.3f}%")
    print(f"    Precision@5 espérée aléatoire   ≈ {mean_rel / n_sites:.5f}")
    print(f"    → Content-based (0.004) ≈ 8× le hasard, mais très loin d'être utile.")
    print(f"    → Collaboratif (0.0) est en dessous du hasard à cause du bug d'exclusion.")

    print(f"\n  Interactions par profil :")
    merged = df_inter.merge(df_users[["user_id", "user_type"]], on="user_id", how="left")
    cnt_by_type = merged.groupby("user_type")["code_site"].count()
    n_by_type   = df_users.groupby("user_type").size()
    for ut in cnt_by_type.index:
        moy = cnt_by_type[ut] / n_by_type[ut]
        print(f"    {ut:<22} : {moy:.0f} interactions/utilisateur")

    no_inter = set(df_users["user_id"]) - set(df_inter["user_id"])
    print(f"\n  Utilisateurs sans interaction : {len(no_inter)}")


# ═══════════════════════════════════════════════════════════════════════════════
# PARTIE 3 - Scores content-based détaillés (utilisateur Famille)
# ═══════════════════════════════════════════════════════════════════════════════

def part3_content_detail(df_features, df_sites, df_users, df_inter):
    print()
    sep("═")
    print("PARTIE 3 — Scores content-based détaillés (utilisateur Famille)")
    sep("═")

    famille_users = df_users[df_users["user_type"] == "Famille"]
    if famille_users.empty:
        # Prendre le premier profil disponible
        famille_users = df_users.head(1)
        profil_nom = df_users.iloc[0]["user_type"]
        print(f"  (Profil 'Famille' absent — utilisation de '{profil_nom}')")
    else:
        profil_nom = "Famille"

    user    = famille_users.iloc[0]
    user_id = user["user_id"]
    user_lat  = user["latitude"]
    user_lon  = user["longitude"]
    score_min = user["score_min"]        # déjà en [0,1]
    dist_max  = user["distance_max"]
    type_pref = user["type_eau_pref"]
    need_pmr  = bool(user["equip_pmr"])
    need_san  = bool(user["equip_sanitaires"])

    print(f"\n  Utilisateur {profil_nom} (id={user_id}) :")
    print(f"    lat/lon      : ({user_lat:.4f}, {user_lon:.4f})")
    print(f"    score_min    : {score_min:.4f}  (= {score_min*100:.1f} sur 100)")
    print(f"    distance_max : {dist_max} km")
    print(f"    type_eau_pref: {type_pref}")
    print(f"    pmr/san      : {need_pmr}/{need_san}")

    true_df  = df_inter[df_inter["user_id"] == user_id].sort_values("rating", ascending=False)
    true_ids = set(true_df["code_site"].tolist())

    # ── A) Scores bugués
    scores_bug = content_scores_buggues(user, df_sites)
    top5_bug   = sorted(scores_bug, key=lambda x: x[1], reverse=True)[:5]
    unique_sc  = set(s for _, s in scores_bug)

    print(f"\n  A) Scores bugués (recommender.py actuel) — Top 5 :")
    print(f"  {'Site':<14}  {'Score':<8}  {'Hit?'}")
    for sid, sc in top5_bug:
        hit = "✓" if sid in true_ids else "✗"
        print(f"  {sid:<14}  {sc:<8.4f}  {hit}")

    type_pref_val  = user.get("type_eau_pref", "all")
    type_match_val = 1.0 if type_pref_val == "all" else 0.7
    final_sc = 0.5*0.5 + type_match_val*0.3 + 0.5*0.2
    print(f"\n  Décomposition du score bugué :")
    print(f"    quality_score = 0.5 (hardcodé, ignorant score_expert du site)")
    print(f"    type_eau_pref = '{type_pref_val}' → type_match = {type_match_val}")
    print(f"    equip_match   = 0.5 (hardcodé, ignorant parking/pmr/sanitaires)")
    print(f"    → score final = 0.5×0.5 + {type_match_val}×0.3 + 0.5×0.2 = {final_sc:.4f}")
    print(f"    → {len(unique_sc)} score(s) distinct(s) pour {len(scores_bug)} sites")

    # ── B) Scores corrects
    sc_corrects, dists = content_scores_corrects(
        df_sites, user_lat, user_lon, score_min, dist_max,
        type_pref, need_pmr, need_san
    )
    top5_ok_idx = np.argsort(sc_corrects)[::-1][:5]

    print(f"\n  B) Scores corrects (logique relevance_score) — Top 5 recommandés :")
    print(f"  {'Site':<14}  {'Score':<8}  {'Dist km':<9}  {'Q/100':<7}  {'Type':<12}  Hit?")
    for i in top5_ok_idx:
        row  = df_sites.iloc[i]
        sid  = row["code_site"]
        sc   = sc_corrects[i]
        d    = dists[i]
        q    = row["score_expert"]
        t    = row["type_norm"]
        hit  = "✓" if sid in true_ids else "✗"
        print(f"  {sid:<14}  {sc:<8.4f}  {d:<9.1f}  {q:<7.1f}  {t:<12}  {hit}")

    # ── C) Vrais positifs et leurs scores corrects
    print(f"\n  C) Vrais positifs (Top 5 par rating) — scores corrects associés :")
    print(f"  {'Site':<14}  {'Rating':<8}  {'Score_c':<9}  {'Dist km':<9}  {'Q/100':<7}  Type")
    for _, row_i in true_df.head(5).iterrows():
        sid = row_i["code_site"]
        mask = df_sites["code_site"] == sid
        if mask.any():
            loc  = df_sites.index[mask][0]
            pos  = df_sites.index.get_loc(loc)
            sc   = sc_corrects[pos]
            d    = dists[pos]
            q    = df_sites.loc[loc, "score_expert"]
            t    = df_sites.loc[loc, "type_norm"]
        else:
            sc = d = q = float("nan"); t = "?"
        print(f"  {sid:<14}  {row_i['rating']:<8.2f}  {sc:<9.4f}  {d:<9.1f}  {q:<7.1f}  {t}")

    # ── D) Distribution des distances des vrais positifs
    true_sites_df = df_sites[df_sites["code_site"].isin(true_ids)]
    if len(true_sites_df) > 0:
        d_true = haversine_vec(user_lat, user_lon,
                               true_sites_df["latitude"].values,
                               true_sites_df["longitude"].values)
        print(f"\n  D) Distribution des distances des vrais positifs :")
        print(f"     Médiane : {np.median(d_true):.1f} km")
        print(f"     Max     : {np.max(d_true):.1f} km")
        print(f"     distance_max user : {dist_max} km")
        pct_dans_rayon = (d_true <= dist_max).mean() * 100
        print(f"     % vrais positifs dans le rayon : {pct_dans_rayon:.1f}%")

    # ── Synthèse
    print()
    sep("─")
    print("  SYNTHÈSE DES CAUSES — Precision@5 ≈ 0")
    sep("─")
    print("""
  1. CONTENT-BASED — 2 bugs qui rendent le score constant :
     a) quality_score = 0.5  → ignore score_expert (colonne "score_expert" non lue)
     b) equip_match   = 0.5  → ignore parking, pmr, sanitaires (colonnes non lues)
     c) type_match    = 1.0 si type_pref=="all" sinon 0.7
        → La colonne "type_eau_pref" est bien lue, mais la comparaison ne vérifie
          pas si le type du site correspond au type préféré. Elle vaut toujours 0.7
          pour les users avec une préférence (personne n'a "all").
     Résultat : tous les sites ont exactement le même score final.
     Le Top-5 = les 5 premières lignes du DataFrame df_sites, sans rapport avec
     les préférences de l'utilisateur.

  2. COLLABORATIF — Exclusion garantissant Precision = 0 :
     user_visited = interactions de l'utilisateur = vrais positifs.
     La fonction ne recommande que des sites hors user_visited.
     → Un vrai positif ne peut JAMAIS être recommandé.

  3. ÉVALUATION — Pas de séparation train/test :
     Les vrais positifs = toutes les interactions (pas de jeu de test séparé).
     Il faudrait conserver ~20% des interactions par utilisateur en test set,
     entraîner sur les 80% restants, puis évaluer sur le test set.

  4. DENSITÉ :
     ~240 interactions/utilisateur sur ~4 300 sites uniques dans sites_scores.
     P(hit aléatoire en Top-5) ≈ 0.06% — métriques intrinsèquement basses
     sans scoring réellement personnalisé.
""")


# ═══════════════════════════════════════════════════════════════════════════════
# Lancement
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print()
    sep("═")
    print("  INVESTIGATION — MÉTRIQUES DU MOTEUR DE RECOMMANDATION")
    sep("═")

    print("\n  Chargement des données...")
    df_features, df_sites, df_users, df_inter = load()
    print(f"    Sites (sites_scores) : {len(df_sites):,}")
    print(f"    Features (temporal)  : {len(df_features):,}")
    print(f"    Utilisateurs         : {len(df_users):,}")
    print(f"    Interactions         : {len(df_inter):,}")

    part1_profiles(df_features, df_sites, df_users, df_inter)
    part2_precision_analysis(df_features, df_sites, df_users, df_inter)
    part3_content_detail(df_features, df_sites, df_users, df_inter)

    print()
    sep("═")
    print("  FIN DU DIAGNOSTIC")
    sep("═")


if __name__ == "__main__":
    main()
