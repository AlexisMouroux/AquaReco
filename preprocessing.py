"""
preprocessing.py - AquaReco Semaine 4
Filtrage des données, feature engineering et production du dataset ML.

Applique les filtres séquentiels, produit un rapport d'exclusions détaillé,
et retourne un DataFrame site×saison prêt pour models.py.
"""

import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import glob
import math
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

OUTPUT_DIR = Path(__file__).parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

DOM_TOM = {"971", "972", "973", "974", "976"}

# Mots-clés qui indiquent un nouveau site dans la colonne 'evolution'
_NEW_SITE_KW = ["nouveau", "identifi", "nouvelle"]


# ── Normalisation ──────────────────────────────────────────────────────────────

def _norm(s: str) -> str:
    s = unicodedata.normalize("NFD", str(s)).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")


def _contains_any(text: str, keywords: list) -> bool:
    t = _norm(str(text))
    return any(kw in t for kw in keywords)


# ── Chargement des fichiers informations (événements de saison) ────────────────

def _load_events() -> pd.DataFrame:
    """
    Charge tous les fichiers 'informations-sur-la-saison-*.csv'.
    Retourne un DataFrame normalisé avec les colonnes :
      code_site, saison, type_evenement, date_debut, date_fin
    """
    frames = []
    for path in sorted(glob.glob("saison-balneaire-*informations*.csv")):
        df = pd.read_csv(path, sep=";", encoding="latin-1")
        df.columns = [_norm(c) for c in df.columns]
        rename = {
            c: "saison"         for c in df.columns if "saison" in c and len(c) < 20
        }
        rename.update({
            c: "code_site"      for c in df.columns if "code_unique" in c
        })
        rename.update({
            c: "type_evenement" for c in df.columns if "type" in c and "even" in c
        })
        rename.update({
            c: "date_debut"     for c in df.columns if "debut" in c
        })
        rename.update({
            c: "date_fin"       for c in df.columns if "de_fin" in c or c.endswith("fin")
        })
        df = df.rename(columns=rename)
        for col in ["saison", "code_site", "type_evenement", "date_debut", "date_fin"]:
            if col not in df.columns:
                df[col] = None
        df["saison"] = df["saison"].astype(str).str[:4]
        frames.append(df[["code_site", "saison", "type_evenement", "date_debut", "date_fin"]])

    events = pd.concat(frames, ignore_index=True)
    # Parse dates DD/MM/YYYY
    for col in ["date_debut", "date_fin"]:
        events[col] = pd.to_datetime(events[col], format="%d/%m/%Y", errors="coerce")
    return events


# ── Construction du masque d'exclusion : interdictions sanitaires ──────────────

def _build_interdiction_exclusion(df_analyses: pd.DataFrame,
                                   events: pd.DataFrame) -> pd.Series:
    """
    Retourne un masque booléen (index df_analyses) :
      True = prélèvement à GARDER
      False = prélèvement pendant une interdiction sanitaire → EXCLURE
    """
    san = events[
        events["type_evenement"].str.contains("Interdiction sanitaire", case=False, na=False)
    ][["code_site", "date_debut", "date_fin"]].dropna()

    # Génère l'ensemble des (code_site, date) à exclure
    excl_pairs = set()
    for _, row in san.iterrows():
        dates = pd.date_range(row["date_debut"], row["date_fin"])
        for d in dates:
            excl_pairs.add((row["code_site"], d.normalize()))

    dates_norm = pd.to_datetime(df_analyses["date_prelevement"]).dt.normalize()
    mask_excl = pd.Series([
        (cs, d) not in excl_pairs
        for cs, d in zip(df_analyses["code_site"], dates_norm)
    ], index=df_analyses.index)
    return mask_excl


# ── Analyse de corrélation sites non classés vs nombre de prélèvements ────────

def correlation_unclassified_vs_samples(df_site_year: pd.DataFrame,
                                         df_analyses: pd.DataFrame) -> None:
    """
    Vérifie l'hypothèse : les sites non classés ont-ils moins de 4 prélèvements ?
    Affiche un tableau de contingence et une corrélation point-bisérial.
    """
    # Nombre de prélèvements IN-SAISON par site×saison
    n_prel = (
        df_analyses[~df_analyses["statut_prelevement"].str.contains(
            "pre", case=False, na=False)]
        .groupby(["code_site", "saison"])["date_prelevement"]
        .count()
        .reset_index(name="n_prelev")
    )
    merged = df_site_year[["code_site", "saison", "classement"]].merge(
        n_prel, on=["code_site", "saison"], how="left"
    )
    merged["n_prelev"] = merged["n_prelev"].fillna(0).astype(int)
    merged["non_classe"] = merged["classement"].isna().astype(int)

    print("\n  Correlation sites non classes vs n_prelevements :")
    print("  " + "-"*55)

    # Tableau de contingence (n_prelev buckets)
    bins = [-1, 0, 3, 7, 15, 1000]
    labels = ["0", "1-3", "4-7", "8-15", ">15"]
    merged["bucket"] = pd.cut(merged["n_prelev"], bins=bins, labels=labels)
    table = merged.groupby("bucket", observed=True).agg(
        total=("non_classe", "count"),
        non_classes=("non_classe", "sum"),
    )
    table["pct_non_classe"] = (table["non_classes"] / table["total"] * 100).round(1)
    print(f"  {'Nb prelevement':>18}  {'Total':>7}  {'Non classes':>12}  {'%':>6}")
    print("  " + "-"*55)
    for label, row in table.iterrows():
        print(f"  {str(label):>18}  {int(row['total']):>7}  {int(row['non_classes']):>12}  "
              f"{row['pct_non_classe']:>5.1f}%")

    # Corrélation point-bisérial
    from scipy import stats as _stats
    valid = merged.dropna(subset=["n_prelev"])
    corr, pval = _stats.pointbiserialr(valid["non_classe"], valid["n_prelev"])
    print(f"\n  Correlation point-biseriale : r={corr:.3f}  p={pval:.2e}")
    if corr < -0.2:
        print("  => Sites non classes ont significativement moins de prelevements.")
    else:
        print("  => Pas de lien fort entre classement manquant et nb de prelevements.")


# ── Filtres d'exclusion ────────────────────────────────────────────────────────

def apply_filters(df_site_year: pd.DataFrame,
                  df_analyses: pd.DataFrame,
                  events: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Applique les filtres séquentiels sur les prélèvements.

    Retourne (df_filtered_sites, df_filtered_samples, df_badges)
      - df_filtered_sites  : sites éligibles à l'entraînement (site×saison)
      - df_filtered_samples: prélèvements après exclusions (liés aux sites éligibles)
      - df_badges          : TOUS les sites avec leurs badges pour le dashboard
    """
    n0_sites   = len(df_site_year)
    n0_samples = len(df_analyses)

    report_lines = [
        f"\n  {'='*62}",
        f"  RAPPORT D'EXCLUSIONS",
        f"  {'='*62}",
        f"  Depart : {n0_sites:>7,} lignes site x saison  |  "
        f"{n0_samples:>9,} prelevements",
        f"  {'-'*62}",
    ]

    # ── 1. DOM-TOM ─────────────────────────────────────────────────
    m_dom = df_site_year["departement"].astype(str).isin(DOM_TOM)
    excl_sites_dom = df_site_year[m_dom]["code_site"].unique()
    report_lines.append(
        f"  DOM-TOM (971-976)         : {m_dom.sum():>6,} lignes site×saison  "
        f"  ({len(excl_sites_dom):,} sites uniques)"
    )
    df_sy = df_site_year[~m_dom].copy()
    df_an = df_analyses[~df_analyses["code_site"].isin(excl_sites_dom)].copy()

    # ── 2. Classement manquant (site×saison sans classement) ───────
    m_nocl = df_sy["classement"].isna()
    excl_codes_nocl = df_sy[m_nocl].set_index(["code_site","saison"]).index
    report_lines.append(
        f"  Classement manquant       : {m_nocl.sum():>6,} lignes site×saison"
    )
    df_sy_train = df_sy[~m_nocl].copy()

    # ── 3. Nouveaux sites (filtre par saison, PAS par site entier) ────
    m_new = df_sy_train["evolution"].map(
        lambda x: _contains_any(x, _NEW_SITE_KW) if pd.notna(x) else False
    )
    new_codes  = df_sy_train[m_new]["code_site"].unique()
    new_pairs  = set(zip(df_sy_train[m_new]["code_site"],
                         df_sy_train[m_new]["saison"]))

    # Comparaison explicite avant/après pour vérifier le comportement per-saison
    tmp_keys = pd.Series(list(zip(df_an["code_site"], df_an["saison"])),
                         index=df_an.index)
    n_excl_si_site_entier = int(df_an["code_site"].isin(new_codes).sum())
    n_excl_par_saison     = int(tmp_keys.isin(new_pairs).sum())

    report_lines.append(
        f"  Nouveaux sites            : {m_new.sum():>6,} lignes site×saison  "
        f"  ({len(new_codes):,} sites uniques)"
    )
    report_lines.append(
        f"    -> excl. par saison    : {n_excl_par_saison:>6,} prelevements"
        f"  (si site entier : {n_excl_si_site_entier:,})"
    )

    # Retire uniquement les (code_site, saison) marqués "nouveau",
    # PAS toutes les saisons du site - les autres saisons sont conservées.
    df_sy_train = df_sy_train[~m_new].copy()

    # Filtre les analyses sur les paires (code_site, saison) éligibles
    train_keys = set(zip(df_sy_train["code_site"], df_sy_train["saison"]))
    an_keys    = pd.Series(list(zip(df_an["code_site"], df_an["saison"])),
                           index=df_an.index)
    df_an_train = df_an[an_keys.isin(train_keys)].copy()

    n_after_site_filters = len(df_an_train)

    # ── 4. Prélèvements de pré-saison ──────────────────────────────
    m_presaison = df_an_train["statut_prelevement"].str.contains(
        r"pr[eé]-?saison|pre.?saison", case=False, na=False, regex=True
    )
    report_lines.append(
        f"  Prelevements pre-saison   : {m_presaison.sum():>6,} prelevements exclus"
    )
    df_an_train = df_an_train[~m_presaison].copy()

    # ── 5. Interdictions sanitaires ────────────────────────────────
    mask_keep = _build_interdiction_exclusion(df_an_train, events)
    n_excl_interdiction = (~mask_keep).sum()
    report_lines.append(
        f"  Interdictions sanitaires  : {n_excl_interdiction:>6,} prelevements exclus"
    )
    df_an_train = df_an_train[mask_keep].copy()

    # ── Résumé ─────────────────────────────────────────────────────
    report_lines += [
        f"  {'-'*62}",
        f"  APRES FILTRES : {len(df_sy_train):>6,} lignes site×saison (entrainement)  "
        f"|  {len(df_an_train):>8,} prelevements",
        f"  Sites uniques  : {df_sy_train['code_site'].nunique():,}",
        f"  Saisons        : {sorted(df_sy_train['saison'].unique())}",
        f"  {'='*62}",
    ]
    for line in report_lines:
        print(line)

    # ── Badges pour le dashboard (sur tous les sites métropole) ───
    badges = []
    for _, row in df_sy[df_sy["saison"] == df_sy["saison"].max()].iterrows():
        b = {"code_site": row["code_site"], "nom_site": row.get("nom_site", "")}
        b["badge_insuffisant"]     = pd.isna(row["classement"])
        b["badge_nouveau_site"]    = _contains_any(
            row.get("evolution", ""), _NEW_SITE_KW)
        # Badge interdiction sanitaire : site ayant eu une interdiction la dernière saison
        saison_max = df_sy["saison"].max()
        b["badge_interdiction"]    = bool(
            events[
                (events["code_site"] == row["code_site"]) &
                (events["saison"].astype(str) == str(saison_max)) &
                (events["type_evenement"].str.contains("Interdiction sanitaire", na=False))
            ].shape[0] > 0
        )
        badges.append(b)
    df_badges = pd.DataFrame(badges)
    df_badges.to_csv(OUTPUT_DIR / "site_badges.csv", index=False, encoding="utf-8-sig")

    return df_sy_train, df_an_train, df_badges


# ── Feature engineering ────────────────────────────────────────────────────────

def _one_hot_type_eau(df: pd.DataFrame) -> pd.DataFrame:
    """Encode type_eau (normalisé) en colonnes booléennes one-hot."""
    import unicodedata as _ud
    def _norm_type(s):
        s = _ud.normalize("NFD", str(s)).encode("ascii", "ignore").decode("ascii").lower()
        if "cotier" in s or "cotiere" in s: return "mer_cote"
        if "lac"    in s:                   return "lac"
        if "riviere" in s or "fleuve" in s or "cours" in s: return "riviere"
        if "mer"    in s:                   return "mer_cote"
        if "transit" in s:                  return "transition"
        return "autre"

    df = df.copy()
    df["_type_norm"] = df["type_eau"].map(_norm_type)
    for cat in ["lac", "mer_cote", "riviere", "transition", "autre"]:
        df[f"type_{cat}"] = (df["_type_norm"] == cat).astype(int)
    df = df.drop(columns=["_type_norm"])
    return df


def _add_trend(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tendance du classement sur les 3 saisons précédentes :
      +1 = amélioration (classement diminue), 0 = stable, -1 = dégradation.
    Calculé uniquement à partir des saisons T-1 à T-3 pour éviter toute
    fuite vers la cible (classement de la saison courante).
    """
    df = df.sort_values(["code_site", "saison"]).copy()
    cl_shifted_3 = df.groupby("code_site")["classement"].shift(3)
    cl_shifted_1 = df.groupby("code_site")["classement"].shift(1)
    delta = cl_shifted_3 - cl_shifted_1   # positif = dégradé dans le passé = amélioration récente
    df["tendance"] = np.sign(delta).fillna(0).astype(int)
    return df


def compute_bacterio_features(df_samples: pd.DataFrame) -> pd.DataFrame:
    """
    À partir des prélèvements filtrés, calcule les features bactériologiques
    agrégées au niveau site×saison.

    Features produites :
      p95_log_ecoli, p95_log_ent
      mean_log_ecoli, mean_log_ent
      std_log_ecoli, std_log_ent
      n_prelevements
    """
    df = df_samples.copy()
    df["log_ecoli"] = np.log10(df["ecoli"].clip(lower=0).fillna(0) + 1)
    df["log_ent"]   = np.log10(df["enterococci"].clip(lower=0).fillna(0) + 1)

    agg = (
        df.groupby(["code_site", "saison"])
          .agg(
              p95_log_ecoli = ("log_ecoli", lambda s: np.percentile(s, 95)),
              p95_log_ent   = ("log_ent",   lambda s: np.percentile(s, 95)),
              mean_log_ecoli= ("log_ecoli", "mean"),
              mean_log_ent  = ("log_ent",   "mean"),
              std_log_ecoli = ("log_ecoli", "std"),
              std_log_ent   = ("log_ent",   "std"),
              n_prelevements= ("log_ecoli", "count"),
          )
          .reset_index()
    )
    agg["std_log_ecoli"] = agg["std_log_ecoli"].fillna(0)
    agg["std_log_ent"]   = agg["std_log_ent"].fillna(0)
    return agg


def add_month_encoding(df_samples: pd.DataFrame) -> pd.DataFrame:
    """
    Encode le mois médian des prélèvements d'une saison en sin/cos.
    Retourne un DataFrame site×saison avec mois_sin et mois_cos.
    """
    df = df_samples.copy()
    df["mois"] = pd.to_datetime(df["date_prelevement"]).dt.month

    agg = (
        df.groupby(["code_site", "saison"])["mois"]
          .median()
          .reset_index(name="mois_median")
    )
    agg["mois_sin"] = np.sin(2 * np.pi * agg["mois_median"] / 12).round(4)
    agg["mois_cos"] = np.cos(2 * np.pi * agg["mois_median"] / 12).round(4)
    return agg[["code_site", "saison", "mois_sin", "mois_cos"]]


def add_meteo_features(df: pd.DataFrame,
                       df_analyses: pd.DataFrame,
                       df_sites: pd.DataFrame,
                       use_cache: bool = True) -> pd.DataFrame:
    """
    Appelle meteo.py pour récupérer les features météo site×saison et les fusionne.
    Ignoré gracieusement si meteo non disponible.
    """
    try:
        from meteo import get_meteo_features_for_ml
        df_meteo = get_meteo_features_for_ml(df_analyses, df_sites, use_cache=use_cache)
        meteo_cols = [c for c in df_meteo.columns if c not in ["code_site", "saison"]]
        df = df.merge(df_meteo[["code_site", "saison"] + meteo_cols],
                      on=["code_site", "saison"], how="left")
        print(f"  Features meteo ajoutees : {meteo_cols}")
    except Exception as exc:
        print(f"  [WARN] Features meteo indisponibles ({exc}). "
              f"Relancer avec use_cache=False pour reconstruire le cache v2.")
    return df


def add_osm_features(df: pd.DataFrame) -> pd.DataFrame:
    """Fusionne les équipements OSM (booléens) si disponibles."""
    eq_path = OUTPUT_DIR / "osm_equipements.csv"
    if not eq_path.exists():
        print("  [INFO] osm_equipements.csv absent — features OSM ignorees.")
        for col in ["parking", "sanitaires", "pmr", "douche", "poste_secours"]:
            df[col] = 0
        return df
    eq = pd.read_csv(eq_path, sep=";")[
        ["code_site", "parking", "sanitaires", "pmr", "douche", "poste_secours"]
    ]
    for col in ["parking", "sanitaires", "pmr", "douche", "poste_secours"]:
        eq[col] = eq[col].astype(int)
    return df.merge(eq, on="code_site", how="left").fillna(
        {c: 0 for c in ["parking", "sanitaires", "pmr", "douche", "poste_secours"]}
    )


# ── Pipeline complet ───────────────────────────────────────────────────────────

FEATURE_COLS = [
    # Bactériologiques
    "p95_log_ecoli", "p95_log_ent",
    "mean_log_ecoli", "mean_log_ent",
    "std_log_ecoli",  "std_log_ent",
    "n_prelevements",
    # Temporelles
    "mois_sin", "mois_cos", "tendance",
    # Type d'eau (one-hot)
    "type_lac", "type_mer_cote", "type_riviere", "type_transition", "type_autre",
    # Équipements OSM
    "parking", "sanitaires", "pmr", "douche", "poste_secours",
    # Météo (ajoutées si cache v2 disponible)
    "precip_7j_median", "temp_7j_median",
    "wind_speed_7j_median", "wind_sin_7j_median", "wind_cos_7j_median",
]

META_COLS = ["code_site", "saison", "classement",
             "nom_site", "region", "departement", "type_eau",
             "longitude", "latitude"]


def build_feature_matrix(use_cache: bool = True) -> pd.DataFrame:
    """
    Pipeline complet :
      1. Charge les données (etl.py)
      2. Analyse corrélation non-classés vs prélèvements
      3. Applique les filtres (rapport)
      4. Feature engineering
      5. Sauvegarde outputs/features.parquet

    Retourne le DataFrame final prêt pour models.py.
    """
    cache_path = OUTPUT_DIR / "features.parquet"
    if use_cache and cache_path.exists():
        print("Cache features charge.")
        return pd.read_parquet(cache_path)

    from etl import build_consolidated
    df_site_year, df_analyses = build_consolidated()

    print("\n--- Analyse sites non classes vs prelevements ---")
    correlation_unclassified_vs_samples(df_site_year, df_analyses)

    print("\n--- Chargement des evenements de saison ---")
    events = _load_events()
    print(f"  {len(events):,} evenements charges "
          f"({events['type_evenement'].value_counts().get('Interdiction sanitaire', 0):,} "
          f"interdictions sanitaires).")

    print("\n--- Application des filtres ---")
    df_sy_train, df_an_train, _ = apply_filters(df_site_year, df_analyses, events)

    # ── Features bactériologiques ──────────────────────────────────
    print("\n--- Feature engineering ---")
    feat_bact   = compute_bacterio_features(df_an_train)
    feat_mois   = add_month_encoding(df_an_train)

    # ── Site×saison metadata + type eau + tendance ──────────────────
    df_sy_feat = (
        df_sy_train[META_COLS + ["evolution"]]
        .drop_duplicates(["code_site", "saison"])
        .copy()
    )
    df_sy_feat = _one_hot_type_eau(df_sy_feat)
    df_sy_feat = _add_trend(df_sy_feat)

    # ── Fusion ─────────────────────────────────────────────────────
    df = (
        df_sy_feat
        .merge(feat_bact, on=["code_site", "saison"], how="left")
        .merge(feat_mois,  on=["code_site", "saison"], how="left")
    )
    df = add_meteo_features(df, df_an_train, df_site_year, use_cache=True)
    df = add_osm_features(df)

    # ── Rapport des features ───────────────────────────────────────
    feat_present = [c for c in FEATURE_COLS if c in df.columns]
    missing_pct = df[feat_present].isna().mean() * 100
    print(f"\n  Features produites : {len(feat_present)} colonnes")
    print(f"  Lignes             : {len(df):,}  site×saison")
    print(f"  Valeurs manquantes (top) :")
    top_miss = missing_pct[missing_pct > 0].sort_values(ascending=False).head(10)
    if top_miss.empty:
        print("    Aucune valeur manquante dans les features.")
    else:
        for col, pct in top_miss.items():
            print(f"    {col:<22} {pct:.1f}%")

    print(f"\n  Distribution de la cible (classement) :")
    target_dist = df["classement"].astype(int).value_counts().sort_index()
    labels = {1: "Excellent", 2: "Bon", 3: "Suffisant", 4: "Insuffisant"}
    for cl, cnt in target_dist.items():
        pct = cnt / len(df) * 100
        bar = "#" * int(pct / 2)
        print(f"    {labels.get(cl, cl):<14}  {bar:<40}  {cnt:>5,}  ({pct:.1f}%)")

    df.to_parquet(cache_path, index=False)
    print(f"\n  Feature matrix sauvegardee : {cache_path}")
    return df


# ── Pipeline per-sample (granularité prélèvement) ─────────────────────────────

def _norm_type_coastal(type_eau) -> bool:
    """True si eau côtière/marine/transition, False pour eaux intérieures."""
    import unicodedata as _ud
    s = _ud.normalize("NFD", str(type_eau)).encode("ascii", "ignore").decode("ascii").lower()
    return any(kw in s for kw in ("mer", "cotier", "cotiere", "marin", "transit"))


def _classify_eu(ecoli: float, ent: float, coastal: bool):
    """Classifie un prélèvement individuel selon la Directive 2006/7/CE (0-3)."""
    if pd.isna(ecoli) or pd.isna(ent):
        return np.nan
    if coastal:
        if ecoli < 250 and ent < 100: return 0
        if ecoli < 500 and ent < 200: return 1
        if ecoli < 500 and ent < 185: return 2
        return 3
    else:
        if ecoli < 500  and ent < 200: return 0
        if ecoli < 1000 and ent < 400: return 1
        if ecoli < 900  and ent < 330: return 2
        return 3


FEATURE_COLS_PER_SAMPLE = [
    "log_ecoli", "log_ent",
    "precip_7j", "temp_7j", "wind_speed_7j", "wind_sin_7j", "wind_cos_7j",
    "mois_sin", "mois_cos", "tendance",
    "type_lac", "type_mer_cote", "type_riviere", "type_transition", "type_autre",
    "parking", "sanitaires", "pmr", "douche", "poste_secours",
]


def build_feature_matrix_per_sample(use_cache: bool = True) -> pd.DataFrame:
    """
    Pipeline per-sample : chaque ligne = un prélèvement individuel.
    Variable cible : classement EU Directive 2006/7/CE (0-3).
    Sauvegarde outputs/features_per_sample.parquet (ne touche pas features.parquet).
    """
    cache_path = OUTPUT_DIR / "features_per_sample.parquet"
    if use_cache and cache_path.exists():
        print("Cache features_per_sample charge.")
        return pd.read_parquet(cache_path)

    from etl import build_consolidated
    df_site_year, df_analyses = build_consolidated()

    print("\n--- Chargement des evenements de saison ---")
    events = _load_events()

    print("\n--- Application des filtres (niveau prelevement) ---")
    df_sy_train, df_an_train, _ = apply_filters(df_site_year, df_analyses, events)

    # ── Join type_eau depuis les sites ────────────────────────────
    type_eau_map = (
        df_sy_train[["code_site", "saison", "type_eau"]]
        .drop_duplicates(["code_site", "saison"])
    )
    df = df_an_train.merge(type_eau_map, on=["code_site", "saison"], how="left")

    # ── Variable cible EU Directive ───────────────────────────────
    print("\n--- Calcul de la variable cible (Directive 2006/7/CE) ---")
    coastal = df["type_eau"].map(_norm_type_coastal).values
    ecoli   = pd.to_numeric(df["ecoli"],       errors="coerce").values
    ent     = pd.to_numeric(df["enterococci"], errors="coerce").values

    eu = np.full(len(df), 3, dtype=float)
    # Non conf. is the default (3); classify up where thresholds are met
    # Inland
    m_in = ~coastal
    eu[m_in & (ecoli < 900)  & (ent < 330)]  = 2
    eu[m_in & (ecoli < 1000) & (ent < 400)]  = 1
    eu[m_in & (ecoli < 500)  & (ent < 200)]  = 0
    # Coastal / transition
    m_co = coastal
    eu[m_co & (ecoli < 500) & (ent < 185)]   = 2
    eu[m_co & (ecoli < 500) & (ent < 200)]   = 1
    eu[m_co & (ecoli < 250) & (ent < 100)]   = 0
    # NaN where micro values are missing
    eu[np.isnan(ecoli) | np.isnan(ent)] = np.nan
    df["classement_eu"] = eu
    n_before = len(df)
    df = df.dropna(subset=["classement_eu"])
    df["classement_eu"] = df["classement_eu"].astype(int)
    n_excl = n_before - len(df)
    if n_excl:
        print(f"  Exclusion cible NaN (micro manquants) : {n_excl}")

    # ── Features bactériologiques ──────────────────────────────────
    df["log_ecoli"] = np.log10(df["ecoli"].clip(lower=0).fillna(0) + 1)
    df["log_ent"]   = np.log10(df["enterococci"].clip(lower=0).fillna(0) + 1)

    # ── Features temporelles ───────────────────────────────────────
    df["mois"] = pd.to_datetime(df["date_prelevement"]).dt.month
    df["mois_sin"] = np.sin(2 * np.pi * df["mois"] / 12).round(4)
    df["mois_cos"] = np.cos(2 * np.pi * df["mois"] / 12).round(4)
    df = df.drop(columns=["mois"])

    # ── Tendance site×saison ───────────────────────────────────────
    trend_src = df_sy_train[["code_site", "saison", "classement"]].drop_duplicates()
    trend_df  = _add_trend(trend_src)[["code_site", "saison", "tendance"]]
    df = df.merge(trend_df, on=["code_site", "saison"], how="left")
    df["tendance"] = df["tendance"].fillna(0).astype(int)

    # ── One-hot type_eau ───────────────────────────────────────────
    df = _one_hot_type_eau(df)

    # ── Météo per-sample ───────────────────────────────────────────
    meteo_path = OUTPUT_DIR / "meteo_features_ml.csv"
    meteo_candidates = ["precip_7j", "temp_7j", "wind_speed_7j", "wind_sin_7j", "wind_cos_7j"]
    meteo_cols_avail = []
    if meteo_path.exists():
        met = pd.read_csv(meteo_path, parse_dates=["date_prelevement"])
        meteo_cols_avail = [c for c in meteo_candidates if c in met.columns]
        df = df.merge(
            met[["code_site", "date_prelevement"] + meteo_cols_avail],
            on=["code_site", "date_prelevement"], how="left"
        )
        for col in meteo_cols_avail:
            df[col] = df.groupby("type_eau")[col].transform(
                lambda s: s.fillna(s.median())
            )
        print(f"  Features meteo jointes et imputees : {meteo_cols_avail}")
    else:
        print("  [INFO] meteo_features_ml.csv absent — features meteo absentes.")

    # ── Équipements OSM ────────────────────────────────────────────
    df = add_osm_features(df)

    # ── Rapport ────────────────────────────────────────────────────
    feat_present = [c for c in FEATURE_COLS_PER_SAMPLE if c in df.columns]
    missing_pct  = df[feat_present].isna().mean() * 100
    print(f"\n  Features : {len(feat_present)} colonnes  |  {len(df):,} prelevements")
    top_miss = missing_pct[missing_pct > 0].sort_values(ascending=False).head(10)
    if not top_miss.empty:
        print("  Valeurs manquantes (top) :")
        for col, pct in top_miss.items():
            print(f"    {col:<22} {pct:.1f}%")

    print("\n  Distribution de la cible (classement EU) :")
    labels_eu = {0: "Excellent", 1: "Bon", 2: "Suffisant", 3: "Non conf."}
    for cl, cnt in df["classement_eu"].value_counts().sort_index().items():
        pct = cnt / len(df) * 100
        bar = "#" * int(pct / 2)
        print(f"    {labels_eu.get(cl, cl):<14}  {bar:<40}  {cnt:>6,}  ({pct:.1f}%)")

    # ── Sauvegarde ─────────────────────────────────────────────────
    meta_ps   = ["code_site", "saison", "date_prelevement", "classement_eu", "type_eau"]
    keep_cols = [c for c in meta_ps + feat_present if c in df.columns]
    out = df[keep_cols]
    out.to_parquet(cache_path, index=False)
    print(f"\n  Dataset per-sample sauvegarde : {cache_path}")
    return out


# ── Pipeline temporel (features historiques, sans fuite de données) ───────────

def _add_historical_bacterio(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcule les features bactériologiques basées UNIQUEMENT sur les
    prélèvements antérieurs du même site.
    Le DataFrame doit déjà contenir ecoli et enterococci numériques.
    """
    df = df.sort_values(["code_site", "date_prelevement"]).copy()

    le = np.log10(pd.to_numeric(df["ecoli"],       errors="coerce").clip(lower=0).fillna(0) + 1)
    ld = np.log10(pd.to_numeric(df["enterococci"], errors="coerce").clip(lower=0).fillna(0) + 1)
    df["_le"] = le.values
    df["_ld"] = ld.values

    grp = df.groupby("code_site", sort=False)

    # Moyenne de tous les prélèvements précédents (expanding puis shift(1))
    df["ecoli_moy_hist"] = grp["_le"].transform(lambda s: s.expanding().mean().shift(1))
    df["ent_moy_hist"]   = grp["_ld"].transform(lambda s: s.expanding().mean().shift(1))

    # Dernier prélèvement uniquement (shift(1))
    df["ecoli_dernier"] = grp["_le"].shift(1)
    df["ent_dernier"]   = grp["_ld"].shift(1)

    # Nombre de prélèvements antérieurs disponibles
    df["n_prelevements_ant"] = (
        grp["_le"]
        .transform(lambda s: s.expanding().count().shift(1))
        .fillna(0).astype(int)
    )

    df = df.drop(columns=["_le", "_ld"])
    return df


def _classement_eu_correct(df_an_train: pd.DataFrame, type_eau_map: pd.DataFrame,
                           min_prelevements: int = 16) -> pd.DataFrame:
    """
    Calcule le classement officiel EU (Directive 2006/7/CE, Annexe II) par site×saison,
    à partir des prélèvements des 4 dernières saisons (saison courante + 3 précédentes).

    Méthode : log10(valeur+1) sur chaque prélèvement de la fenêtre, moyenne (mu) et
    écart-type (sigma) sur la fenêtre glissante, puis :
      p95 = antilog(mu + 1.65*sigma)   p90 = antilog(mu + 1.28*sigma)
    Classement (0=Excellent, 1=Bon, 2=Suffisant, 3=Insuffisant). NaN si moins de
    `min_prelevements` échantillons valides sur la fenêtre (donnée insuffisante).
    """
    df = df_an_train[["code_site", "saison", "ecoli", "enterococci"]].copy()
    df["saison"] = pd.to_numeric(df["saison"], errors="coerce").astype(int)
    ecoli = pd.to_numeric(df["ecoli"],       errors="coerce").clip(lower=0)
    ent   = pd.to_numeric(df["enterococci"], errors="coerce").clip(lower=0)
    df["log_ecoli"] = np.log10(ecoli + 1)
    df["log_ent"]   = np.log10(ent + 1)
    df = df.dropna(subset=["log_ecoli", "log_ent"])

    # Stats par site×saison individuelle (sommes, pour pooling sur la fenêtre)
    per_season = (
        df.groupby(["code_site", "saison"])
          .agg(n=("log_ecoli", "size"),
               sum_e=("log_ecoli", "sum"),
               sumsq_e=("log_ecoli", lambda s: float((s ** 2).sum())),
               sum_n=("log_ent", "sum"),
               sumsq_n=("log_ent", lambda s: float((s ** 2).sum())))
          .reset_index()
    )

    target = (
        type_eau_map[["code_site", "saison", "type_eau"]]
        .drop_duplicates(["code_site", "saison"])
        .copy()
    )
    target["saison"] = pd.to_numeric(target["saison"], errors="coerce").astype(int)

    acc = target[["code_site", "saison"]].copy()
    for col in ["n", "sum_e", "sumsq_e", "sum_n", "sumsq_n"]:
        acc[col] = 0.0

    # Fenêtre glissante de 4 saisons (offset 0 = saison courante, 1..3 = precedentes)
    for offset in range(4):
        shifted = per_season.copy()
        shifted["saison"] = shifted["saison"] + offset
        merged = acc[["code_site", "saison"]].merge(
            shifted, on=["code_site", "saison"], how="left"
        )
        for col in ["n", "sum_e", "sumsq_e", "sum_n", "sumsq_n"]:
            acc[col] = acc[col] + merged[col].fillna(0.0).values

    n = acc["n"].values
    with np.errstate(invalid="ignore", divide="ignore"):
        mu_e  = acc["sum_e"].values / n
        var_e = np.clip(acc["sumsq_e"].values / n - mu_e ** 2, 0, None)
        mu_n  = acc["sum_n"].values / n
        var_n = np.clip(acc["sumsq_n"].values / n - mu_n ** 2, 0, None)
    sigma_e, sigma_n = np.sqrt(var_e), np.sqrt(var_n)

    p95_ecoli = 10 ** (mu_e + 1.65 * sigma_e) - 1
    p90_ecoli = 10 ** (mu_e + 1.28 * sigma_e) - 1
    p95_ent   = 10 ** (mu_n + 1.65 * sigma_n) - 1
    p90_ent   = 10 ** (mu_n + 1.28 * sigma_n) - 1

    acc = acc.merge(target, on=["code_site", "saison"], how="left")
    coastal = acc["type_eau"].map(_norm_type_coastal).values
    m_in, m_co = ~coastal, coastal

    cl = np.full(len(acc), 3, dtype=float)   # Insuffisant par defaut
    # Suffisant (p90)
    cl[m_in & (p90_ecoli < 900) & (p90_ent < 330)] = 2
    cl[m_co & (p90_ecoli < 500) & (p90_ent < 185)] = 2
    # Bon (p95) - plus strict, écrase Suffisant si vérifié
    cl[m_in & (p95_ecoli < 1000) & (p95_ent < 400)] = 1
    cl[m_co & (p95_ecoli < 500)  & (p95_ent < 200)] = 1
    # Excellent (p95) - le plus strict, écrase Bon si vérifié
    cl[m_in & (p95_ecoli < 500) & (p95_ent < 200)] = 0
    cl[m_co & (p95_ecoli < 250) & (p95_ent < 100)] = 0

    donnees_insuffisantes = (n < min_prelevements) | acc["type_eau"].isna().values
    cl[donnees_insuffisantes] = np.nan

    acc["classement_eu_correct"]   = cl
    acc["n_prelevements_4saisons"] = n.astype(int)
    return acc[["code_site", "saison", "classement_eu_correct", "n_prelevements_4saisons"]]


FEATURE_COLS_TEMPORAL = [
    # Historique bactériologique (prélèvements antérieurs uniquement)
    "ecoli_moy_hist", "ent_moy_hist",
    "ecoli_dernier",  "ent_dernier",
    "n_prelevements_ant",
    # Météo (fenêtre 7j avant le prélèvement - pas de fuite)
    "precip_7j", "temp_7j", "wind_speed_7j", "wind_sin_7j", "wind_cos_7j",
    # Temporelles
    "mois_sin", "mois_cos",
    # Tendance site×saison (classement officiel, saisons précédentes)
    "tendance",
    # Type d'eau (one-hot)
    "type_lac", "type_mer_cote", "type_riviere", "type_transition", "type_autre",
    # Équipements OSM
    "parking", "sanitaires", "pmr", "douche", "poste_secours",
]


def build_feature_matrix_temporal(use_cache: bool = True) -> pd.DataFrame:
    """
    Pipeline temporel sans fuite de données.
    Variable cible : classement EU 3 classes (0=Excellent, 1=Bon, 3=Non conf.).
    Features : historique bactério (shift+expanding), météo, temporel, tendance, OSM.
    Sauvegarde outputs/features_temporal.parquet.
    """
    cache_path = OUTPUT_DIR / "features_temporal.parquet"
    if use_cache and cache_path.exists():
        print("Cache features_temporal charge.")
        return pd.read_parquet(cache_path)

    from etl import build_consolidated
    df_site_year, df_analyses = build_consolidated()

    print("\n--- Chargement des evenements de saison ---")
    events = _load_events()

    print("\n--- Application des filtres ---")
    df_sy_train, df_an_train, _ = apply_filters(df_site_year, df_analyses, events)

    # ── Join type_eau ─────────────────────────────────────────────
    type_eau_map = (
        df_sy_train[["code_site", "saison", "type_eau"]]
        .drop_duplicates(["code_site", "saison"])
    )
    df = df_an_train.merge(type_eau_map, on=["code_site", "saison"], how="left")

    # ── Variable cible EU 3 classes (sans Suffisant) ──────────────
    print("\n--- Calcul de la variable cible (3 classes) ---")
    coastal   = df["type_eau"].map(_norm_type_coastal).values
    ecoli_v   = pd.to_numeric(df["ecoli"],       errors="coerce").values
    ent_v     = pd.to_numeric(df["enterococci"], errors="coerce").values

    eu = np.full(len(df), 3, dtype=float)
    m_in = ~coastal
    eu[m_in & (ecoli_v < 1000) & (ent_v < 400)] = 1
    eu[m_in & (ecoli_v < 500)  & (ent_v < 200)] = 0
    m_co = coastal
    eu[m_co & (ecoli_v < 500)  & (ent_v < 200)] = 1
    eu[m_co & (ecoli_v < 250)  & (ent_v < 100)] = 0
    eu[np.isnan(ecoli_v) | np.isnan(ent_v)] = np.nan
    df["classement_eu"] = eu

    n_before = len(df)
    df = df.dropna(subset=["classement_eu"])
    df["classement_eu"] = df["classement_eu"].astype(int)
    if (n_before - len(df)):
        print(f"  Exclusion cible NaN : {n_before - len(df)}")

    # ── Classement EU officiel CORRECT (percentile, fenêtre glissante 4 saisons) ──
    print("\n--- Calcul du classement EU officiel correct (Directive 2006/7/CE, Annexe II) ---")
    cl_correct = _classement_eu_correct(df_an_train, type_eau_map)
    df = df.merge(cl_correct, on=["code_site", "saison"], how="left")
    n_valid = int(df["classement_eu_correct"].notna().sum())
    print(f"  Lignes avec classement_eu_correct valide : {n_valid:,} / {len(df):,} "
          f"(NaN = moins de 16 prelevements sur la fenetre 4 saisons)")

    # ── Features historiques bactériologiques ──────────────────────
    print("--- Feature engineering historique (shift+expanding) ---")
    df = _add_historical_bacterio(df)   # trie par (code_site, date_prelevement)

    # ── Features temporelles ───────────────────────────────────────
    df["mois"] = pd.to_datetime(df["date_prelevement"]).dt.month
    df["mois_sin"] = np.sin(2 * np.pi * df["mois"] / 12).round(4)
    df["mois_cos"] = np.cos(2 * np.pi * df["mois"] / 12).round(4)
    df = df.drop(columns=["mois"])

    # ── Tendance site×saison ───────────────────────────────────────
    trend_src = df_sy_train[["code_site", "saison", "classement"]].drop_duplicates()
    trend_df  = _add_trend(trend_src)[["code_site", "saison", "tendance"]]
    df = df.merge(trend_df, on=["code_site", "saison"], how="left")
    df["tendance"] = df["tendance"].fillna(0).astype(int)

    # ── One-hot type_eau ───────────────────────────────────────────
    df = _one_hot_type_eau(df)

    # ── Météo per-sample ───────────────────────────────────────────
    meteo_path = OUTPUT_DIR / "meteo_features_ml.csv"
    meteo_candidates = ["precip_7j", "temp_7j", "wind_speed_7j", "wind_sin_7j", "wind_cos_7j"]
    meteo_cols_avail = []
    if meteo_path.exists():
        met = pd.read_csv(meteo_path, parse_dates=["date_prelevement"])
        meteo_cols_avail = [c for c in meteo_candidates if c in met.columns]
        df = df.merge(
            met[["code_site", "date_prelevement"] + meteo_cols_avail],
            on=["code_site", "date_prelevement"], how="left"
        )
        print(f"  Features meteo jointes : {meteo_cols_avail}")
    else:
        print("  [INFO] meteo_features_ml.csv absent — features meteo absentes.")

    # ── Équipements OSM ────────────────────────────────────────────
    df = add_osm_features(df)

    # ── Rapport valeurs manquantes avant/après imputation ──────────
    hist_cols  = ["ecoli_moy_hist", "ent_moy_hist", "ecoli_dernier", "ent_dernier"]
    feat_all   = [c for c in FEATURE_COLS_TEMPORAL if c in df.columns]
    miss_before = df[feat_all].isna().mean() * 100
    top_before  = miss_before[miss_before > 0].sort_values(ascending=False)
    print(f"\n  Valeurs manquantes avant imputation :")
    for col, pct in top_before.items():
        print(f"    {col:<22} {pct:.1f}%")

    # Imputation mediane par type_eau, puis mediane globale si groupe entier NaN
    for col in hist_cols + meteo_cols_avail:
        if col not in df.columns:
            continue
        df[col] = df.groupby("type_eau")[col].transform(
            lambda s: s.fillna(s.median())
        )
        df[col] = df[col].fillna(df[col].median())

    miss_after = df[feat_all].isna().mean() * 100
    top_after  = miss_after[miss_after > 0]
    if top_after.empty:
        print("  Valeurs manquantes apres imputation : aucune.")
    else:
        print("  Valeurs manquantes apres imputation :")
        for col, pct in top_after.items():
            print(f"    {col:<22} {pct:.1f}%")

    # ── Distribution de la cible ───────────────────────────────────
    print("\n  Distribution de la cible (classement EU 3 classes, ANCIEN seuil instantane) :")
    labels_eu = {0: "Excellent", 1: "Bon", 3: "Non conf."}
    for cl, cnt in df["classement_eu"].value_counts().sort_index().items():
        pct = cnt / len(df) * 100
        bar = "#" * int(pct / 2)
        print(f"    {labels_eu.get(cl, cl):<14}  {bar:<40}  {cnt:>6,}  ({pct:.1f}%)")

    # ── Comparaison classement CORRECT (site×saison la plus recente) vs officiel CSV ──
    print("\n  --- Comparaison classement_eu_correct vs classement officiel CSV (saison la plus recente) ---")
    saison_max  = int(df["saison"].max())
    site_level  = df[df["saison"] == saison_max].drop_duplicates("code_site")
    correct_pct = (site_level["classement_eu_correct"].value_counts(normalize=True, dropna=True).sort_index() * 100)
    n_nan       = int(site_level["classement_eu_correct"].isna().sum())

    officiel = (
        trend_src[trend_src["saison"] == saison_max]
        .drop_duplicates("code_site")["classement"]
    )
    officiel_pct = (officiel.value_counts(normalize=True).sort_index() * 100)

    labels4     = {0: "Excellent", 1: "Bon", 2: "Suffisant", 3: "Insuffisant"}
    labels4_off = {1: "Excellent", 2: "Bon", 3: "Suffisant", 4: "Insuffisant"}
    print(f"    {'Classe':<14}{'Correct (%)':>14}{'Officiel CSV (%)':>20}")
    for k in [0, 1, 2, 3]:
        print(f"    {labels4[k]:<14}{correct_pct.get(k, 0.0):>13.1f}%{officiel_pct.get(k + 1, 0.0):>19.1f}%")
    print(f"    Sites avec donnees insuffisantes (NaN, < 16 prelevements / 4 saisons) : "
          f"{n_nan:,} / {len(site_level):,}")

    # ── Sauvegarde ─────────────────────────────────────────────────
    meta_t    = ["code_site", "saison", "date_prelevement", "classement_eu",
                 "classement_eu_correct", "n_prelevements_4saisons", "type_eau"]
    keep_cols = [c for c in meta_t + feat_all if c in df.columns]
    out       = df[keep_cols]
    out.to_parquet(cache_path, index=False)
    print(f"\n  Dataset temporal sauvegarde : {cache_path}")
    print(f"  {len(out):,} lignes x {out.shape[1]} colonnes")
    return out


if __name__ == "__main__":
    df_t = build_feature_matrix_temporal(use_cache=False)
    print(f"\nDataset ML (temporal) : {df_t.shape[0]} lignes x {df_t.shape[1]} colonnes")
