"""
Score de qualité multicritère - AquaReco
Calcule Q ∈ [0, 100] pour chaque site × saison, selon deux approches de pondération :
  - Pondération experte  : poids fixés à partir de la Directive 2006/7/CE
  - Pondération apprise  : régression supervisée (Ridge) sur le classement officiel
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, r2_score

# ── Seuils Directive 2006/7/CE, Annexe I ──────────────────────────────────────
# (UFC/100 mL, percentiles P95 pour excellent/bon, P90 pour suffisant)

THRESHOLDS = {
    # eau douce (inland)
    "douce": {
        "ecoli":       {"excellent_p95": 500,  "bon_p95": 1000, "suffisant_p90": 900},
        "enterococci": {"excellent_p95": 200,  "bon_p95": 400,  "suffisant_p90": 330},
    },
    # eaux côtières et de transition (coastal / transitional)
    "cotiere": {
        "ecoli":       {"excellent_p95": 250, "bon_p95": 500, "suffisant_p90": 500},
        "enterococci": {"excellent_p95": 100, "bon_p95": 200, "suffisant_p90": 185},
    },
}

# Correspondance type d'eau → catégorie réglementaire
WATER_TYPE_MAP = {
    "eau douce":         "douce",
    "eau cotiere":       "cotiere",
    "eau de transition": "cotiere",   # même seuils que côtière
}

# Correspondance classement officiel → score de base [0-100]
CLASSEMENT_SCORE = {1: 100, 2: 75, 3: 50, 4: 25}


# ── Utilitaires ────────────────────────────────────────────────────────────────

def _water_category(origine_eau: pd.Series) -> pd.Series:
    """Mappe 'Eau douce' / 'Eau côtière' / 'Eau de transition' vers 'douce' ou 'cotiere'."""
    import unicodedata, re

    def norm(s):
        s = unicodedata.normalize("NFD", str(s))
        s = s.encode("ascii", "ignore").decode("ascii").lower()
        return re.sub(r"[^a-z]+", " ", s).strip()

    return origine_eau.map(lambda x: WATER_TYPE_MAP.get(norm(x), "douce"))


def _eu_class_from_percentiles(
    p_ec: float, p_ent: float, water: str, percentile: int = 95
) -> int:
    """
    Retourne la classe EU (1-4) à partir de percentiles mesurés.
    percentile: 95 (excellent/bon) ou 90 (suffisant est évalué au P90).
    """
    th = THRESHOLDS.get(water, THRESHOLDS["douce"])
    pkey = "excellent_p95" if percentile == 95 else "suffisant_p90"
    if percentile == 95:
        if p_ec <= th["ecoli"]["excellent_p95"] and p_ent <= th["enterococci"]["excellent_p95"]:
            return 1
        if p_ec <= th["ecoli"]["bon_p95"] and p_ent <= th["enterococci"]["bon_p95"]:
            return 2
    # Évaluation suffisant au P90 (appelée séparément - ici on passe déjà le bon percentile)
    if p_ec <= th["ecoli"]["suffisant_p90"] and p_ent <= th["enterococci"]["suffisant_p90"]:
        return 3
    return 4


# ── Sous-score bactériologique ─────────────────────────────────────────────────

def compute_bacterio_stats(df_analyses: pd.DataFrame) -> pd.DataFrame:
    """
    Calcule les statistiques percentiles des indicateurs microbiologiques
    par site et par saison, en utilisant une fenêtre glissante de 4 ans
    (conformément à la Directive 2006/7/CE).

    Retourne un DataFrame avec P90 et P95 de E. coli et entérocoques.
    """
    # Exclut les prélèvements hors saison (pré-saison, post-saison)
    mask_in_season = ~df_analyses["statut_prelevement"].str.contains(
        "pre|post|hors", case=False, na=False
    )
    df = df_analyses[mask_in_season].copy()

    def rolling_percentile(group, col, pct):
        return np.nanpercentile(group[col].dropna(), pct) if len(group[col].dropna()) > 0 else np.nan

    records = []
    sites = df["code_site"].unique()
    years = sorted(df["saison"].dropna().unique())

    for site in sites:
        site_df = df[df["code_site"] == site]
        for year in years:
            # Fenêtre glissante : année courante + 3 années précédentes
            window = site_df[site_df["saison"].between(year - 3, year)]
            if len(window) == 0:
                continue
            ec  = window["ecoli"].dropna()
            ent = window["enterococci"].dropna()
            if len(ec) == 0 and len(ent) == 0:
                continue
            records.append({
                "code_site":   site,
                "saison":      year,
                "n_prelevements": len(window),
                "ec_p90":      np.nanpercentile(ec, 90)  if len(ec)  > 0 else np.nan,
                "ec_p95":      np.nanpercentile(ec, 95)  if len(ec)  > 0 else np.nan,
                "ent_p90":     np.nanpercentile(ent, 90) if len(ent) > 0 else np.nan,
                "ent_p95":     np.nanpercentile(ent, 95) if len(ent) > 0 else np.nan,
                "ec_median":   ec.median()  if len(ec)  > 0 else np.nan,
                "ent_median":  ent.median() if len(ent) > 0 else np.nan,
            })

    return pd.DataFrame(records)


def bacterio_score(df_stats: pd.DataFrame, df_site_year: pd.DataFrame) -> pd.Series:
    """
    Calcule le sous-score bactériologique (0-100) pour chaque site × saison.
    Utilise les percentiles P95 pour excellent/bon et P90 pour suffisant.

    Retourne une Series indexée sur le même index que df_stats.
    """
    # Récupère le type d'eau
    water_cat = (
        df_site_year[["code_site", "saison", "origine_eau"]]
        .drop_duplicates(subset=["code_site", "saison"])
        .assign(water_cat=lambda d: _water_category(d["origine_eau"]))
    )
    merged = df_stats.merge(
        water_cat[["code_site", "saison", "water_cat"]],
        on=["code_site", "saison"], how="left"
    )
    merged["water_cat"] = merged["water_cat"].fillna("douce")

    scores = []
    for _, row in merged.iterrows():
        water = row["water_cat"]
        th = THRESHOLDS[water]
        ec_p95, ent_p95 = row["ec_p95"], row["ent_p95"]
        ec_p90, ent_p90 = row["ec_p90"], row["ent_p90"]

        if pd.isna(ec_p95) and pd.isna(ent_p95):
            scores.append(np.nan)
            continue

        ec_p95  = ec_p95  if not pd.isna(ec_p95)  else ec_p90
        ent_p95 = ent_p95 if not pd.isna(ent_p95) else ent_p90

        # Classe EU calculée sur P95
        if (ec_p95  <= th["ecoli"]["excellent_p95"] and
                ent_p95 <= th["enterococci"]["excellent_p95"]):
            base = 100
        elif (ec_p95  <= th["ecoli"]["bon_p95"] and
                ent_p95 <= th["enterococci"]["bon_p95"]):
            base = 75
        elif (row["ec_p90"]  <= th["ecoli"]["suffisant_p90"] and
                row["ent_p90"] <= th["enterococci"]["suffisant_p90"]):
            base = 50
        else:
            base = 25

        # Malus proportionnel à la distance au seuil excellent (adoucit la discontinuité)
        ec_ratio  = ec_p95  / th["ecoli"]["excellent_p95"]
        ent_ratio = ent_p95 / th["enterococci"]["excellent_p95"]
        penalty = min(20, 10 * max(0, max(ec_ratio, ent_ratio) - 1))
        scores.append(max(0.0, float(base) - penalty))

    return pd.Series(scores, index=merged.index, name="score_bacterio")


# ── Sous-score de tendance ─────────────────────────────────────────────────────

def tendance_score(df_site_year: pd.DataFrame) -> pd.Series:
    """
    Calcule le sous-score de tendance (0-100) à partir de l'évolution du classement
    officiel sur les 3-4 dernières saisons pour chaque site × saison.

    Un classement décroissant (amélioration) donne un bonus ; croissant (dégradation)
    donne un malus.
    """
    df = df_site_year[["code_site", "saison", "classement"]].dropna(subset=["classement"]).copy()
    df["classement"] = df["classement"].astype(float)

    scores_map = {}
    for site, grp in df.groupby("code_site"):
        grp = grp.sort_values("saison")
        for _, row in grp.iterrows():
            year = row["saison"]
            # Fenêtre des 4 dernières années disponibles pour ce site
            window = grp[grp["saison"].between(year - 3, year)]
            cl_recent = row["classement"]
            base = CLASSEMENT_SCORE.get(int(cl_recent), 50)

            if len(window) >= 2:
                # Régression linéaire simple OLS sur (saison, classement)
                x = window["saison"].values.astype(float)
                y = window["classement"].values.astype(float)
                x_centered = x - x.mean()
                slope = (x_centered * y).sum() / (x_centered ** 2).sum()
                # slope < 0 → classement diminue → qualité s'améliore → bonus
                trend_bonus = -slope * 12   # ±1 unité de classe/an ≈ ±12 pts
                trend_bonus = float(np.clip(trend_bonus, -20, 20))
            else:
                trend_bonus = 0.0

            score = float(np.clip(base + trend_bonus, 0, 100))
            scores_map[(site, year)] = score

    return df_site_year.apply(
        lambda r: scores_map.get((r["code_site"], r["saison"]), np.nan), axis=1
    ).rename("score_tendance")


# ── Score d'ouverture (fermetures / interdictions) ────────────────────────────

def ouverture_score(df_site_year: pd.DataFrame) -> pd.Series:
    """
    Sous-score basé sur le nombre de jours de fermeture dans la saison.
    0 jour → 100 ; 30+ jours → 0 (pénalité linéaire).
    """
    jours = df_site_year["jours_fermeture"].fillna(0)
    score = (100 - jours.clip(upper=30) * (100 / 30)).clip(lower=0)
    return score.rename("score_ouverture")


# ── Score expert ───────────────────────────────────────────────────────────────

EXPERT_WEIGHTS = {
    "score_bacterio":  0.55,   # indicateur réglementaire principal
    "score_tendance":  0.20,   # trajectoire pluriannuelle
    "score_meteo":     0.15,   # risque de lessivage dans les 7 jours
    "score_ouverture": 0.10,   # fermetures sanitaires
}


def expert_score(df: pd.DataFrame) -> pd.Series:
    """
    Combine les sous-scores avec les poids experts issus de la Directive 2006/7/CE.
    Les sous-scores manquants sont ignorés et les poids renormalisés.
    """
    score = pd.Series(0.0, index=df.index)
    total_weight = pd.Series(0.0, index=df.index)

    for col, w in EXPERT_WEIGHTS.items():
        if col not in df.columns:
            continue
        valid = df[col].notna()
        score[valid] += df.loc[valid, col] * w
        total_weight[valid] += w

    # Renormalise si certains sous-scores sont manquants
    mask = total_weight > 0
    score[mask] = score[mask] / total_weight[mask] * 1.0
    score[~mask] = np.nan
    return score.rename("score_expert")


# ── Score appris ───────────────────────────────────────────────────────────────

def learned_score(df: pd.DataFrame, verbose: bool = True) -> tuple:
    """
    Apprend les poids des sous-scores par régression Ridge supervisée,
    en utilisant le classement officiel (inversé) comme cible.

    Retourne (series de scores appris, dict des poids appris).
    """
    # Cible : classement officiel inversé → 1=100, 2=75, 3=50, 4=25
    df = df.copy()
    df["target"] = df["classement"].map(CLASSEMENT_SCORE)

    feature_cols = [c for c in ["score_bacterio", "score_tendance",
                                "score_meteo", "score_ouverture"]
                    if c in df.columns]

    # Garde uniquement les lignes avec features ET target disponibles
    mask = df[feature_cols + ["target"]].notna().all(axis=1)
    X = df.loc[mask, feature_cols].values
    y = df.loc[mask, "target"].values

    if len(X) < 10:
        raise ValueError("Pas assez de données pour entraîner le modèle appris.")

    # Normalisation 0-100 → les features sont déjà sur [0,100], mais Ridge
    # bénéficie d'une normalisation pour comparer les coefs.
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    model = Ridge(alpha=1.0, positive=True, fit_intercept=True)
    model.fit(X_scaled, y)

    # Prédit sur toutes les lignes avec features disponibles
    mask_pred = df[feature_cols].notna().all(axis=1)
    X_pred = scaler.transform(df.loc[mask_pred, feature_cols].values)
    preds = pd.Series(np.nan, index=df.index)
    preds[mask_pred] = np.clip(model.predict(X_pred), 0, 100)

    # Rapport des poids
    raw_coefs  = model.coef_
    coef_sum   = raw_coefs.sum()
    poids_norm = raw_coefs / coef_sum if coef_sum > 0 else raw_coefs
    weights = {col: round(float(w), 4) for col, w in zip(feature_cols, poids_norm)}

    if verbose:
        y_pred_train = model.predict(X_scaled)
        mae = mean_absolute_error(y, y_pred_train)
        r2  = r2_score(y, y_pred_train)
        print("\n--- Modele appris (Ridge, alpha=1) ---")
        print(f"  Features    : {feature_cols}")
        print(f"  N lignes    : {mask.sum():,}")
        print(f"  MAE (train) : {mae:.2f}")
        print(f"  R2  (train) : {r2:.4f}")
        print(f"  Poids normalises : {weights}")
        print(f"  Poids experts    : {EXPERT_WEIGHTS}")

    return preds.rename("score_appris"), weights


# ── Pipeline complet ───────────────────────────────────────────────────────────

def compute_all_scores(df_site_year: pd.DataFrame,
                       df_analyses: pd.DataFrame,
                       df_meteo: pd.DataFrame | None = None) -> tuple:
    """
    Calcule les 4 sous-scores puis les 2 scores agrégés (expert et appris).

    Paramètres
    ----------
    df_site_year : DataFrame consolidé site × saison (ETL)
    df_analyses  : DataFrame granulaire des prélèvements
    df_meteo     : DataFrame optionnel issu de meteo.compute_meteo_scores()
                   - colonnes attendues : code_site, saison, score_meteo
                   Si None, le sous-score météo est absent (renormalisé automatiquement).

    Retourne (df_scored, learned_weights)
    """
    print("\nCalcul des statistiques bacteriologiques...")
    stats = compute_bacterio_stats(df_analyses)

    print("Calcul des sous-scores...")
    bact_scores = bacterio_score(stats, df_site_year)
    stats = stats.assign(score_bacterio=bact_scores.values)

    df = df_site_year.merge(
        stats[["code_site", "saison", "n_prelevements",
               "ec_p95", "ent_p95", "ec_p90", "ent_p90", "score_bacterio"]],
        on=["code_site", "saison"], how="left"
    )

    df["score_tendance"]  = tendance_score(df).values
    df["score_ouverture"] = ouverture_score(df).values

    # Intègre le sous-score météo si disponible
    if df_meteo is not None and "score_meteo" in df_meteo.columns:
        df = df.merge(
            df_meteo[["code_site", "saison", "score_meteo",
                       "precip_7j_median", "precip_7j_max"]].drop_duplicates(),
            on=["code_site", "saison"], how="left"
        )
        n_with_meteo = df["score_meteo"].notna().sum()
        print(f"  Scores meteo integres : {n_with_meteo:,}/{len(df):,} lignes")
    else:
        df["score_meteo"] = np.nan
        print("  [INFO] Sous-score meteo absent — poids renormalises sur les autres sous-scores.")

    df["score_expert"] = expert_score(df).values

    print("Entrainement du modele appris...")
    try:
        df["score_appris"], learned_weights = learned_score(df)
    except ValueError as e:
        print(f"  [WARN] {e}")
        df["score_appris"] = np.nan
        learned_weights = {}

    print("\nDistribution des scores experts :")
    print(df["score_expert"].describe().round(2).to_string())

    return df, learned_weights


if __name__ == "__main__":
    from etl import build_consolidated
    df_site_year, df_analyses = build_consolidated()
    df_scored, weights = compute_all_scores(df_site_year, df_analyses)
    df_scored.to_csv("outputs/scores.csv", index=False)
    print("\nFichier outputs/scores.csv créé.")
