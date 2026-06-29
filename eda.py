"""
Analyse exploratoire des données - AquaReco
Produit 5 visualisations enregistrées dans outputs/.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # backend non-interactif (compatible serveur / CI)
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

OUTPUT_DIR = Path(__file__).parent / "outputs"
EDA_DIR    = OUTPUT_DIR / "eda"
OUTPUT_DIR.mkdir(exist_ok=True)
EDA_DIR.mkdir(exist_ok=True)

CLASSEMENT_LABELS = {1: "Excellent", 2: "Bon", 3: "Suffisant", 4: "Insuffisant"}
CLASSEMENT_COLORS = {1: "#2ecc71", 2: "#f1c40f", 3: "#e67e22", 4: "#e74c3c"}
PALETTE = list(CLASSEMENT_COLORS.values())

sns.set_theme(style="whitegrid", palette="muted", font_scale=1.05)


# ── 1. Distribution des classements par région ─────────────────────────────────

def plot_classement_by_region(df: pd.DataFrame, top_n: int = 15) -> Path:
    """
    Heatmap du classement moyen par région × année.
    Seules les `top_n` régions avec le plus de sites sont affichées.
    """
    df = df.dropna(subset=["classement", "region"])
    top_regions = df["region"].value_counts().head(top_n).index

    pivot = (
        df[df["region"].isin(top_regions)]
        .assign(classement=lambda d: d["classement"].astype("float"))
        .groupby(["region", "saison"])["classement"]
        .mean()
        .unstack("saison")
    )

    fig, ax = plt.subplots(figsize=(11, 6))
    sns.heatmap(
        pivot, annot=True, fmt=".2f", cmap="RdYlGn_r",
        vmin=1, vmax=4, linewidths=0.5, ax=ax,
        cbar_kws={"label": "Classement moyen (1=excellent, 4=insuffisant)"}
    )
    ax.set_title("Classement sanitaire moyen par région et par saison", fontweight="bold")
    ax.set_xlabel("Saison")
    ax.set_ylabel("")
    plt.tight_layout()
    out = EDA_DIR / "01_classement_region_heatmap.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Enregistré : {out}")
    return out


# ── 2. Évolution temporelle de la qualité (2020-2024) ─────────────────────────

def plot_evolution_temporelle(df: pd.DataFrame) -> Path:
    """
    Graphique en aires empilées : part (%) de chaque classe par saison.
    """
    df = df.dropna(subset=["classement"]).copy()
    df["classement"] = df["classement"].astype(int)

    counts = (
        df.groupby(["saison", "classement"])
        .size()
        .unstack("classement", fill_value=0)
    )
    pcts = counts.div(counts.sum(axis=1), axis=0) * 100

    fig, ax = plt.subplots(figsize=(9, 5))
    colors = [CLASSEMENT_COLORS.get(c, "#aaa") for c in pcts.columns]
    labels = [CLASSEMENT_LABELS.get(c, str(c)) for c in pcts.columns]

    ax.stackplot(pcts.index, [pcts[c] for c in pcts.columns],
                 labels=labels, colors=colors, alpha=0.85)

    ax.set_xlim(pcts.index.min(), pcts.index.max())
    ax.set_ylim(0, 100)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter())
    ax.set_xlabel("Saison")
    ax.set_ylabel("Part des sites (%)")
    ax.set_title("Évolution de la qualité sanitaire des eaux de baignade (2020–2024)",
                 fontweight="bold")
    ax.legend(loc="lower left", framealpha=0.9)
    plt.tight_layout()
    out = EDA_DIR / "02_evolution_temporelle.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Enregistré : {out}")
    return out


# ── 3. Répartition par type d'eau ─────────────────────────────────────────────

def plot_type_eau(df: pd.DataFrame) -> Path:
    """
    Double graphique : répartition des sites par type d'eau (camembert)
    et classement moyen par type d'eau (barres).
    """
    df = df.dropna(subset=["type_eau"])
    site_types = df.drop_duplicates("code_site")["type_eau"].value_counts()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Camembert
    wedge_colors = ["#3498db", "#2ecc71", "#e67e22", "#9b59b6", "#1abc9c"]
    ax1.pie(site_types.values, labels=site_types.index,
            autopct="%1.1f%%", startangle=90,
            colors=wedge_colors[:len(site_types)])
    ax1.set_title("Répartition des sites par type d'eau", fontweight="bold")

    # Classement moyen par type d'eau
    mean_cl = df.dropna(subset=["classement"]).groupby("type_eau")["classement"].mean().sort_values()
    colors_bar = [CLASSEMENT_COLORS.get(round(v), "#aaa") for v in mean_cl.values]
    bars = ax2.barh(mean_cl.index, mean_cl.values, color=colors_bar, edgecolor="white")
    ax2.set_xlim(1, 4)
    ax2.set_xlabel("Classement moyen (1=excellent, 4=insuffisant)")
    ax2.set_title("Classement moyen par type d'eau", fontweight="bold")
    ax2.bar_label(bars, fmt="%.2f", padding=3)
    ax2.invert_xaxis()

    plt.tight_layout()
    out = EDA_DIR / "03_type_eau.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Enregistré : {out}")
    return out


# ── 4. Taux de valeurs manquantes ─────────────────────────────────────────────

def plot_missing_values(df: pd.DataFrame) -> Path:
    """
    Barplot horizontal des colonnes triées par taux de valeurs manquantes.
    """
    pct = (df.isnull().mean() * 100).sort_values(ascending=True)
    pct = pct[pct > 0]   # ne montre que les colonnes avec au moins 1 valeur manquante

    if len(pct) == 0:
        print("  Aucune valeur manquante détectée.")
        return None

    fig, ax = plt.subplots(figsize=(9, max(4, len(pct) * 0.35)))
    colors = ["#e74c3c" if v > 20 else "#f39c12" if v > 5 else "#3498db" for v in pct.values]
    bars = ax.barh(pct.index, pct.values, color=colors, edgecolor="white")
    ax.set_xlabel("Valeurs manquantes (%)")
    ax.set_title("Taux de valeurs manquantes par colonne", fontweight="bold")
    ax.bar_label(bars, fmt="%.1f%%", padding=3, fontsize=9)
    ax.set_xlim(0, min(100, pct.max() + 15))
    plt.tight_layout()
    out = EDA_DIR / "04_valeurs_manquantes.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Enregistré : {out}")
    return out


# ── 5. Distribution des scores qualité ───────────────────────────────────────

def plot_score_distribution(df: pd.DataFrame) -> Path:
    """
    Distribution (violon + boîte) des scores expert et appris par saison.
    """
    score_cols = [c for c in ["score_expert", "score_appris"] if c in df.columns]
    if not score_cols:
        print("  Aucun score disponible pour ce graphique.")
        return None

    fig, axes = plt.subplots(1, len(score_cols), figsize=(6 * len(score_cols), 5), squeeze=False)
    titles = {"score_expert": "Score expert (pondération réglementaire)",
              "score_appris": "Score appris (régression Ridge)"}

    for ax, col in zip(axes[0], score_cols):
        data = df.dropna(subset=[col, "saison"])
        saisons = sorted(data["saison"].unique())
        parts = [data.loc[data["saison"] == s, col].dropna().values for s in saisons]
        parts = [p for p in parts if len(p) > 1]

        vp = ax.violinplot(parts, positions=range(len(parts)), showmedians=True)
        for body in vp["bodies"]:
            body.set_alpha(0.7)
            body.set_facecolor("#3498db")
        ax.set_xticks(range(len(parts)))
        ax.set_xticklabels([str(s) for s in saisons[:len(parts)]])
        ax.set_ylim(0, 105)
        ax.set_ylabel("Score Q in [0, 100]")
        ax.set_xlabel("Saison")
        ax.set_title(titles.get(col, col), fontweight="bold")

    plt.tight_layout()
    out = EDA_DIR / "05_distribution_scores.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Enregistré : {out}")
    return out


# ── 6. Comparaison pondérations expert vs appris ──────────────────────────────

def plot_weight_comparison(expert_weights: dict, learned_weights: dict) -> Path:
    """
    Barplot côte à côte des poids expert et appris pour chaque sous-score.
    """
    features = sorted(set(list(expert_weights) + list(learned_weights)))
    x = np.arange(len(features))
    w_exp = [expert_weights.get(f, 0) for f in features]
    w_lea = [learned_weights.get(f, 0) for f in features]

    fig, ax = plt.subplots(figsize=(8, 4))
    width = 0.35
    ax.bar(x - width / 2, w_exp, width, label="Expert", color="#3498db", alpha=0.85)
    ax.bar(x + width / 2, w_lea, width, label="Appris (Ridge)", color="#e67e22", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels([f.replace("score_", "").capitalize() for f in features])
    ax.set_ylabel("Poids normalisé")
    ax.set_title("Comparaison des pondérations expert vs appris", fontweight="bold")
    ax.legend()
    ax.set_ylim(0, 1)
    plt.tight_layout()
    out = EDA_DIR / "06_comparaison_poids.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Enregistré : {out}")
    return out


# ── 7. Carte choroplèthe par département ──────────────────────────────────────

def plot_choropleth_departement(df: pd.DataFrame) -> Path | None:
    """
    Carte choroplèthe du classement moyen par département via GeoPandas.
    Nécessite geopandas et le fichier GeoJSON des départements (téléchargé
    automatiquement depuis data.gouv.fr si absent).
    """
    try:
        import geopandas as gpd
    except ImportError:
        print("  [SKIP] geopandas non installe - ignoré pour la carte choroplèthe.")
        return None

    import urllib.request, json

    GEOJSON_URL  = (
        "https://raw.githubusercontent.com/gregoiredavid/france-geojson/"
        "master/departements.geojson"
    )
    GEOJSON_FILE = OUTPUT_DIR / "departements.geojson"

    if not GEOJSON_FILE.exists():
        print("  Telechargement du GeoJSON departements...")
        try:
            urllib.request.urlretrieve(GEOJSON_URL, GEOJSON_FILE)
        except Exception as e:
            print(f"  [WARN] Telechargement echoue : {e}")
            return None

    gdf = gpd.read_file(GEOJSON_FILE)

    # Moyenne du classement par département (toutes saisons)
    dept_stats = (
        df.dropna(subset=["classement", "departement"])
          .assign(dept_norm=lambda d: d["departement"].astype(str).str.zfill(2))
          .groupby("dept_norm")["classement"]
          .mean()
          .reset_index()
          .rename(columns={"dept_norm": "code", "classement": "cl_moyen"})
    )

    gdf = gdf.merge(dept_stats, on="code", how="left")

    fig, ax = plt.subplots(1, 1, figsize=(10, 9))
    gdf.plot(
        column="cl_moyen", ax=ax, cmap="RdYlGn_r",
        vmin=1, vmax=4, legend=True, missing_kwds={"color": "#cccccc"},
        legend_kwds={"label": "Classement moyen (1=excellent)", "shrink": 0.7},
    )
    ax.set_axis_off()
    ax.set_title("Classement sanitaire moyen par département (2020–2024)",
                 fontweight="bold", fontsize=13)
    plt.tight_layout()
    out = EDA_DIR / "07_carte_choropleth_dept.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Enregistré : {out}")
    return out


# ── 8. Distribution log-scale des indicateurs bactériologiques ────────────────

def plot_bacterio_distributions(df_analyses: pd.DataFrame) -> Path:
    """
    Histogrammes en échelle logarithmique de E. coli et entérocoques.
    Superpose les seuils réglementaires EU pour eaux douces et côtières.
    """
    df = df_analyses.dropna(subset=["ecoli", "enterococci"]).copy()
    # Filtre les valeurs à la limite de détection (15 UFC) pour les deux axes
    df_ec  = df[df["ecoli"]       > 15]["ecoli"]
    df_ent = df[df["enterococci"] > 15]["enterococci"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Seuils EU (P95 eaux douces)
    thresholds_ec  = {"Excellent (500)":  500,  "Bon (1000)": 1000}
    thresholds_ent = {"Excellent (200)":  200,  "Bon (400)":  400}
    colors_th = {"Excellent (500)": "#2ecc71", "Bon (1000)": "#f1c40f",
                 "Excellent (200)": "#2ecc71", "Bon (400)":  "#f1c40f"}

    for ax, data, label, thresholds in [
        (ax1, df_ec,  "E. coli (UFC/100 ml)",         thresholds_ec),
        (ax2, df_ent, "Enterococci (UFC/100 ml)", thresholds_ent),
    ]:
        ax.hist(np.log10(data.clip(lower=1)), bins=60, color="#3498db",
                alpha=0.7, edgecolor="white", linewidth=0.3)
        ax.set_xlabel(f"log10({label})")
        ax.set_ylabel("Nombre de prelevements")
        # Seuils
        for name, val in thresholds.items():
            ax.axvline(np.log10(val), color=colors_th[name],
                       linestyle="--", linewidth=1.5, label=name)
        ax.legend(fontsize=9)

        # Axe X secondaire en unités naturelles
        xticks = [1, 10, 100, 1000, 10000]
        ax.set_xticks([np.log10(v) for v in xticks])
        ax.set_xticklabels([str(v) for v in xticks])

    ax1.set_title("Distribution E. coli (valeurs > 15 UFC)", fontweight="bold")
    ax2.set_title("Distribution Enterococci (valeurs > 15 UFC)", fontweight="bold")

    fig.suptitle(
        f"Indicateurs bacteriologiques 2020-2024  |  "
        f"n={len(df_ec):,} (E.coli) / {len(df_ent):,} (Entero) prelevement detectes",
        fontsize=10, y=1.01
    )
    plt.tight_layout()
    out = EDA_DIR / "08_bacterio_distribution_logscale.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Enregistré : {out}")
    return out


# ── 9. Corrélation précipitations vs taux de bactéries ────────────────────────

def plot_precip_vs_bacterio(df_scored: pd.DataFrame,
                             df_analyses: pd.DataFrame) -> Path | None:
    """
    Scatter plot de la somme de précipitations 7 jours (P7j médiane par site-saison)
    vs médiane E. coli / entérocoques. Sépare les types d'eau par couleur.
    Nécessite que df_scored contienne 'precip_7j_median' (calculé par meteo.py).
    """
    if "precip_7j_median" not in df_scored.columns:
        print("  [SKIP] precip_7j_median absent — lancez d'abord meteo.py.")
        return None

    # Médiane des bactéries par site-saison
    bact_agg = (
        df_analyses
        .dropna(subset=["ecoli", "enterococci"])
        .groupby(["code_site", "saison"])
        .agg(ec_median=("ecoli", "median"), ent_median=("enterococci", "median"))
        .reset_index()
    )

    df = df_scored.merge(bact_agg, on=["code_site", "saison"], how="inner")
    df = df.dropna(subset=["precip_7j_median", "ec_median"])

    # Classification eau douce vs côtière
    def _cat(s):
        if pd.isna(s):
            return "Autre"
        s = s.lower()
        return "Eau cotiere / transition" if "coti" in s or "transit" in s else "Eau douce"

    df["cat_eau"] = df.get("origine_eau", pd.Series("Autre", index=df.index)).map(_cat)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    palette = {"Eau douce": "#3498db", "Eau cotiere / transition": "#e67e22", "Autre": "#95a5a6"}
    markers = {"Eau douce": "o", "Eau cotiere / transition": "s", "Autre": "^"}

    titles = [("ec_median",  "E. coli (UFC/100 ml)"),
              ("ent_median", "Enterococci (UFC/100 ml)")]

    for ax, (col, ylabel) in zip(axes, titles):
        for cat, grp in df.groupby("cat_eau"):
            x = grp["precip_7j_median"]
            y = np.log10(grp[col].clip(lower=1))
            ax.scatter(x, y, alpha=0.3, s=10,
                       color=palette.get(cat, "#95a5a6"),
                       marker=markers.get(cat, "o"),
                       label=cat)

        # Ligne de tendance globale
        mask_valid = df["precip_7j_median"].notna() & df[col].notna() & (df[col] > 0)
        x_all = df.loc[mask_valid, "precip_7j_median"].values
        y_all = np.log10(df.loc[mask_valid, col].clip(lower=1)).values
        if len(x_all) > 10:
            z = np.polyfit(x_all, y_all, 1)
            x_line = np.linspace(x_all.min(), x_all.max(), 100)
            ax.plot(x_line, np.polyval(z, x_line), "r-", linewidth=1.5,
                    label=f"Tendance (pente={z[0]:.3f})")
            r = np.corrcoef(x_all, y_all)[0, 1]
            ax.annotate(f"r = {r:.3f}", xy=(0.05, 0.92), xycoords="axes fraction",
                        fontsize=10, color="red")

        yticks = [1, 10, 100, 1000, 10000]
        ax.set_yticks([np.log10(v) for v in yticks])
        ax.set_yticklabels([str(v) for v in yticks])
        ax.set_xlabel("Precipitations 7 jours (mm, mediane saisonniere)")
        ax.set_ylabel(f"log10({ylabel})")
        ax.set_title(f"Precipitations vs {ylabel.split()[0]}", fontweight="bold")
        ax.legend(fontsize=8, markerscale=2)

    fig.suptitle("Impact des precipitations sur la contamination bacteriologique (2020-2024)",
                 fontweight="bold")
    plt.tight_layout()
    out = EDA_DIR / "09_precip_vs_bacterio.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Enregistré : {out}")
    return out


# ── Pipeline EDA ───────────────────────────────────────────────────────────────

def run_eda(df_site_year: pd.DataFrame,
            expert_weights: dict | None = None,
            learned_weights: dict | None = None,
            df_analyses: pd.DataFrame | None = None) -> None:
    """Lance l'ensemble des visualisations EDA (9 graphiques)."""
    from score import EXPERT_WEIGHTS
    expert_weights  = expert_weights  or EXPERT_WEIGHTS
    learned_weights = learned_weights or {}

    print("\nGeneration des visualisations EDA...")
    plot_classement_by_region(df_site_year)
    plot_evolution_temporelle(df_site_year)
    plot_type_eau(df_site_year)
    plot_missing_values(df_site_year)

    if "score_expert" in df_site_year.columns:
        plot_score_distribution(df_site_year)

    if expert_weights and learned_weights:
        plot_weight_comparison(expert_weights, learned_weights)

    # Visualisations nécessitant les données granulaires (analyses)
    if df_analyses is not None:
        plot_bacterio_distributions(df_analyses)
        plot_precip_vs_bacterio(df_site_year, df_analyses)

    # Carte choroplèthe (nécessite geopandas)
    plot_choropleth_departement(df_site_year)

    print(f"\nTous les graphiques sont dans : {EDA_DIR}")


if __name__ == "__main__":
    from etl import build_consolidated
    from score import compute_all_scores, EXPERT_WEIGHTS

    df_site_year, df_analyses = build_consolidated()
    df_scored, learned_weights = compute_all_scores(df_site_year, df_analyses)
    run_eda(df_scored, EXPERT_WEIGHTS, learned_weights)
