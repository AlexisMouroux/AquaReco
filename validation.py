"""
Validation du score expert AquaReco - Semaine 4
Évalue la cohérence entre le score Q et le classement officiel EU 2006/7/CE.
"""

import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
)

OUTPUT_DIR = Path(__file__).parent / "outputs"
EDA_DIR    = OUTPUT_DIR / "eda"
EDA_DIR.mkdir(parents=True, exist_ok=True)

# Seuils de conversion score → classement dérivé
SCORE_THRESHOLDS = {1: 80, 2: 60, 3: 40}   # Q≥80→1, 60≤Q<80→2, 40≤Q<60→3, Q<40→4
LABEL_MAP = {1: "Excellent", 2: "Bon", 3: "Suffisant", 4: "Insuffisant"}


# ── Helpers ────────────────────────────────────────────────────────────────────

def score_to_class(q: float) -> int:
    if q >= 80: return 1
    if q >= 60: return 2
    if q >= 40: return 3
    return 4


def load_data() -> pd.DataFrame:
    path = OUTPUT_DIR / "sites_scores.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} introuvable — lancez main.py d'abord.")
    df = pd.read_csv(path)
    df = df.dropna(subset=["classement", "score_expert"]).copy()
    df["classement"]     = df["classement"].astype(int)
    df["classement_der"] = df["score_expert"].map(score_to_class)
    return df


# ── 1. Métriques globales ──────────────────────────────────────────────────────

def compute_metrics(df: pd.DataFrame) -> dict:
    y_true = df["classement"]
    y_pred = df["classement_der"]

    acc    = accuracy_score(y_true, y_pred)
    report = classification_report(
        y_true, y_pred,
        labels=[1, 2, 3, 4],
        target_names=[LABEL_MAP[i] for i in [1,2,3,4]],
        output_dict=True,
        zero_division=0,
    )

    # Taux de concordance directionnelle : pred ≤ vrai (score pas trop optimiste)
    # On mesure aussi la distance absolue moyenne entre classes
    dist  = (y_pred - y_true).abs()
    exact = (dist == 0).mean()
    off1  = (dist <= 1).mean()

    print(f"\n{'='*60}")
    print("  VALIDATION DU SCORE EXPERT")
    print(f"{'='*60}")
    print(f"  Lignes avec classement + score : {len(df):,}")
    print(f"  Accuracy exacte                : {acc*100:.1f}%")
    print(f"  A ±1 classe pres               : {off1*100:.1f}%")
    print(f"  Distance moyenne entre classes : {dist.mean():.2f}")
    print()
    print("  Metriques par classe :")
    for cl_name in [LABEL_MAP[i] for i in [1,2,3,4]]:
        r = report[cl_name]
        print(f"    {cl_name:<14}  precision={r['precision']:.2f}  "
              f"recall={r['recall']:.2f}  f1={r['f1-score']:.2f}  "
              f"n={r['support']:.0f}")

    return {"accuracy": acc, "off1": off1, "dist_mean": dist.mean(), "report": report}


# ── 2. Matrice de confusion ────────────────────────────────────────────────────

def plot_confusion(df: pd.DataFrame) -> None:
    y_true = df["classement"]
    y_pred = df["classement_der"]
    cm     = confusion_matrix(y_true, y_pred, labels=[1,2,3,4])
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, data, fmt, title in zip(
        axes,
        [cm, cm_pct],
        ["d", ".1f"],
        ["Effectifs", "% par classement officiel (ligne)"],
    ):
        im = ax.imshow(data, cmap="Blues", vmin=0, vmax=data.max())
        labels = [LABEL_MAP[i] for i in [1,2,3,4]]
        ax.set_xticks(range(4)); ax.set_xticklabels(labels, rotation=30, ha="right")
        ax.set_yticks(range(4)); ax.set_yticklabels(labels)
        ax.set_xlabel("Classement derive du score Q")
        ax.set_ylabel("Classement officiel")
        ax.set_title(title)
        for i in range(4):
            for j in range(4):
                val = f"{data[i,j]:{fmt}}"
                if fmt == ".1f": val += "%"
                color = "white" if data[i,j] > data.max() * 0.6 else "black"
                ax.text(j, i, val, ha="center", va="center", fontsize=9, color=color)
        fig.colorbar(im, ax=ax, shrink=0.8)

    fig.suptitle("Matrice de confusion : Score Q vs Classement officiel", fontsize=13)
    fig.tight_layout()
    out = EDA_DIR / "validation_score.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Matrice de confusion : {out}")


# ── 3. Boxplot scores par classement ──────────────────────────────────────────

def plot_score_by_classement(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))

    data   = [df[df["classement"] == cl]["score_expert"].values for cl in [1,2,3,4]]
    labels = [f"{LABEL_MAP[cl]}\n(n={len(d):,})" for cl, d in zip([1,2,3,4], data)]
    colors = ["#2ecc71", "#3498db", "#f39c12", "#e74c3c"]

    bp = ax.boxplot(data, labels=labels, patch_artist=True, notch=True,
                    medianprops=dict(color="black", linewidth=2))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color); patch.set_alpha(0.7)

    # Seuils du score
    for yval, label, style in [(80, "Q=80 (seuil Excellent)", "--"),
                                (60, "Q=60 (seuil Bon)",      ":"),
                                (40, "Q=40 (seuil Suffisant)","-.")]:
        ax.axhline(yval, color="grey", linestyle=style, linewidth=1, alpha=0.8)
        ax.text(4.45, yval + 1, label, fontsize=7.5, color="grey", va="bottom")

    ax.set_ylabel("Score expert Q [0–100]")
    ax.set_title("Distribution du score Q par classement officiel (2020–2024)")
    ax.set_ylim(0, 108)
    fig.tight_layout()
    out = EDA_DIR / "validation_boxplot.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Boxplot scores        : {out}")


# ── 4. Cas aberrants ──────────────────────────────────────────────────────────

def show_outliers(df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """
    Sites avec un score Q incohérent par rapport au classement officiel.
    Score d'aberrance = |score_expert - score_attendu_médian_du_classement|
    """
    medians = df.groupby("classement")["score_expert"].median().to_dict()

    df = df.copy()
    df["score_attendu"] = df["classement"].map(medians)
    df["ecart"]         = (df["score_expert"] - df["score_attendu"]).abs()

    # Garde uniquement les vrais désaccords (classements différents)
    desaccords = df[df["classement"] != df["classement_der"]].copy()
    desaccords = desaccords.nlargest(top_n, "ecart")

    cols = ["nom_site", "saison", "classement", "classement_der",
            "score_expert", "score_bacterio", "score_tendance",
            "score_meteo", "ecart"]
    present = [c for c in cols if c in desaccords.columns]
    out     = desaccords[present].reset_index(drop=True)

    print(f"\n  Top {top_n} cas aberrants (classement officiel != classement derive) :")
    print("  " + "-"*80)
    for _, r in out.iterrows():
        cl_off = LABEL_MAP.get(int(r["classement"]), "?")
        cl_der = LABEL_MAP.get(int(r["classement_der"]), "?")
        print(f"  {str(r['nom_site'])[:35]:<36} "
              f"({int(r['saison'])})  "
              f"officiel={cl_off:<12} derive={cl_der:<12} "
              f"Q={r['score_expert']:5.1f}  "
              f"bacterio={r.get('score_bacterio', float('nan')):.0f}  "
              f"tendance={r.get('score_tendance', float('nan')):.0f}  "
              f"ecart={r['ecart']:.1f}")

    # Bilan directionnel
    over = df[df["classement_der"] < df["classement"]]   # score trop optimiste
    under = df[df["classement_der"] > df["classement"]]  # score trop pessimiste
    print(f"\n  Score trop optimiste (derive meilleur qu'officiel) : "
          f"{len(over):,} cas ({len(over)/len(df)*100:.1f}%)")
    print(f"  Score trop pessimiste (derive moins bon qu'officiel): "
          f"{len(under):,} cas ({len(under)/len(df)*100:.1f}%)")

    return out


# ── Point d'entrée ────────────────────────────────────────────────────────────

def run_validation() -> None:
    df = load_data()
    compute_metrics(df)
    plot_confusion(df)
    plot_score_by_classement(df)
    show_outliers(df, top_n=10)

    print(f"\n  Graphiques sauvegardes dans {EDA_DIR}/")
    print("="*60)


if __name__ == "__main__":
    run_validation()
