"""
smote.py - AquaReco Semaine 5
Rééquilibrage du dataset avec SMOTE et réentraînement des modèles.
"""

import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso, Ridge
from sklearn.metrics import f1_score, accuracy_score, classification_report
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

import xgboost as xgb

OUTPUT_DIR = Path(__file__).parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

LABEL_MAP = {0: "Excellent", 1: "Bon", 2: "Non conf."}
CLASS_ORDER = [0, 1, 2]


def load_features_temporal() -> tuple[np.ndarray, np.ndarray, list, list, dict]:
    """
    Charge features_temporal.parquet et retourne (X, y, feature_names, class_order, label_map).

    Cible : classement_eu_correct (percentile EU officiel, fenêtre glissante 4 saisons -
    voir preprocessing.py::_classement_eu_correct), PAS l'ancien classement_eu (seuil
    instantané par prélèvement, biaisé vers "Excellent"). Suffisant (2) et Insuffisant (3)
    sont fusionnés en "Non conf." pour rester cohérent avec le schéma 3 classes du reste
    du pipeline (Excellent/Bon/Non conf.).
    Les lignes sans classement_eu_correct valide (< 16 prélèvements sur la fenêtre 4
    saisons) sont exclues de l'entraînement.
    """
    path = OUTPUT_DIR / "features_temporal.parquet"
    if not path.exists():
        raise FileNotFoundError(
            "features_temporal.parquet introuvable. Lancez preprocessing.py d'abord."
        )
    df = pd.read_parquet(path)
    df = df.dropna(subset=["classement_eu_correct"]).copy()
    df["classement_eu_3c"] = df["classement_eu_correct"].astype(int).clip(upper=2)

    CANDIDATE_FEATURES = [
        "ecoli_moy_hist", "ent_moy_hist", "ecoli_dernier", "ent_dernier",
        "n_prelevements_ant",
        "precip_7j", "temp_7j", "wind_speed_7j", "wind_sin_7j", "wind_cos_7j",
        "mois_sin", "mois_cos", "tendance",
        "type_lac", "type_mer_cote", "type_riviere", "type_transition", "type_autre",
        "parking", "sanitaires", "pmr", "douche", "poste_secours",
    ]
    feat_cols = [c for c in CANDIDATE_FEATURES if c in df.columns]

    raw_classes = sorted(df["classement_eu_3c"].unique().tolist())
    remap = {c: i for i, c in enumerate(raw_classes)}
    class_order = list(range(len(raw_classes)))
    # Mapper les labels en fonction des classes réelles (0, 1, 2 → 0, 1, 2)
    label_map_orig = {0: "Excellent", 1: "Bon", 2: "Non conf."}
    label_map = {remap[c]: label_map_orig.get(c, f"Class {c}") for c in raw_classes}

    X = df[feat_cols].values.astype(float)
    y = np.array([remap[c] for c in df["classement_eu_3c"].values])
    return X, y, feat_cols, class_order, label_map


def get_models_no_smote() -> dict:
    """Modèles sans SMOTE."""
    return {
        "Ridge": Pipeline([
            ("imp",    SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf",    Ridge(alpha=1.0)),
        ]),
        "Lasso": Pipeline([
            ("imp",    SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf",    Lasso(alpha=0.01, max_iter=5000)),
        ]),
        "HistGradientBoosting": HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.05,
            max_leaf_nodes=31, random_state=42,
        ),
        "XGBoost": Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("clf", xgb.XGBClassifier(
                n_estimators=300, learning_rate=0.05, max_depth=6,
                subsample=0.8, colsample_bytree=0.8,
                eval_metric="mlogloss", random_state=42,
            )),
        ]),
    }


def get_models_with_smote() -> dict:
    """Modèles avec SMOTE intégré dans le pipeline (par fold)."""
    return {
        "Ridge": ImbPipeline([
            ("imp",    SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("smote",  SMOTE(sampling_strategy="not majority", random_state=42)),
            ("clf",    Ridge(alpha=1.0)),
        ]),
        "Lasso": ImbPipeline([
            ("imp",    SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("smote",  SMOTE(sampling_strategy="not majority", random_state=42)),
            ("clf",    Lasso(alpha=0.01, max_iter=5000)),
        ]),
        "HistGradientBoosting": ImbPipeline([
            ("imp",    SimpleImputer(strategy="median")),
            ("smote",  SMOTE(sampling_strategy="not majority", random_state=42)),
            ("clf",    HistGradientBoostingClassifier(
                max_iter=300, learning_rate=0.05,
                max_leaf_nodes=31, random_state=42,
            )),
        ]),
        "XGBoost": ImbPipeline([
            ("imp",    SimpleImputer(strategy="median")),
            ("smote",  SMOTE(sampling_strategy="not majority", random_state=42)),
            ("clf",    xgb.XGBClassifier(
                n_estimators=300, learning_rate=0.05, max_depth=6,
                subsample=0.8, colsample_bytree=0.8,
                eval_metric="mlogloss", random_state=42,
            )),
        ]),
    }


def evaluate_model(name: str, model, X: np.ndarray, y: np.ndarray,
                   use_smote: bool = False) -> dict:
    """Évaluation par CV k=5 stratifiée."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    y_pred = np.empty(len(y), dtype=int)
    for train_idx, test_idx in cv.split(X, y):
        model.fit(X[train_idx], y[train_idx])
        y_pred[test_idx] = model.predict(X[test_idx])

    acc = accuracy_score(y, y_pred)
    report = classification_report(
        y, y_pred, labels=CLASS_ORDER,
        target_names=[LABEL_MAP[c] for c in CLASS_ORDER],
        output_dict=True, zero_division=0,
    )
    f1_macro = f1_score(y, y_pred, average="macro", zero_division=0)

    print(f"\n  [{name}]")
    print(f"  Accuracy : {acc*100:.1f}%   F1-macro : {f1_macro:.3f}")
    for cl in CLASS_ORDER:
        r = report[LABEL_MAP[cl]]
        print(f"    {LABEL_MAP[cl]:<14} P={r['precision']:.2f}  R={r['recall']:.2f}  F1={r['f1-score']:.2f}")

    return {
        "model":     name,
        "smote":     "Oui" if use_smote else "Non",
        "accuracy":  round(acc, 4),
        "f1_macro":  round(f1_macro, 4),
        "f1_bon":    round(report[LABEL_MAP[1]]["f1-score"], 4),
        "f1_non_conf": round(report[LABEL_MAP[2]]["f1-score"], 4),
    }


def run_smote_comparison() -> None:
    """Entraîne les modèles avec et sans SMOTE, compare les résultats."""
    print("=" * 70)
    print("  SMOTE — RÉÉQUILIBRAGE DU DATASET")
    print("=" * 70)

    X, y, feat_cols, class_order, label_map = load_features_temporal()

    # Distribution initiale
    print(f"\n  Distribution avant SMOTE :")
    for cl in class_order:
        cnt = (y == cl).sum()
        pct = cnt / len(y) * 100
        print(f"    {label_map[cl]:<14} {cnt:>7,}  ({pct:.1f}%)")

    print(f"\n  Dataset : {X.shape[0]:,} lignes x {X.shape[1]} features")
    print(f"  Validation croisee k=5 stratifiee en cours...")

    results = []

    # ── Sans SMOTE ────────────────────────────────────────────────
    print("\n  --- SANS SMOTE ---")
    models_no = get_models_no_smote()
    for name, model in models_no.items():
        res = evaluate_model(name, model, X, y, use_smote=False)
        results.append(res)

    # ── Avec SMOTE ────────────────────────────────────────────────
    print("\n  --- AVEC SMOTE ---")
    models_yes = get_models_with_smote()
    for name, model in models_yes.items():
        res = evaluate_model(name, model, X, y, use_smote=True)
        results.append(res)

    # ── Tableau comparatif ────────────────────────────────────────
    df_comp = pd.DataFrame(results)
    out_csv = OUTPUT_DIR / "models_comparison_smote.csv"
    df_comp.to_csv(out_csv, index=False, encoding="utf-8-sig")

    print(f"\n  {'='*70}")
    print(f"  TABLEAU COMPARATIF (SMOTE)")
    print(f"  {'='*70}")
    print(f"  {'Modele':<20} {'SMOTE':<6} {'Acc':>6} {'F1mac':>7} {'F1Bon':>7} {'F1NCf':>7}")
    print("  " + "-" * 70)
    for _, r in df_comp.iterrows():
        print(f"  {r['model']:<20} {r['smote']:<6} "
              f"{r['accuracy']*100:>5.1f}% "
              f"{r['f1_macro']:>7.3f} "
              f"{r['f1_bon']:>7.3f} "
              f"{r['f1_non_conf']:>7.3f}")
    print(f"  {'='*70}")
    print(f"  Tableau sauvegarde : {out_csv}")

    # Identifier le meilleur modèle
    best_idx = df_comp["f1_macro"].idxmax()
    best_row = df_comp.loc[best_idx]
    print(f"\n  Meilleur modele (F1-macro) : {best_row['model']} (SMOTE: {best_row['smote']}) "
          f"F1={best_row['f1_macro']:.3f}")

    # ── Persistance du modele final (HistGradientBoosting + SMOTE) ─────
    print(f"\n  Reentrainement de HistGradientBoosting + SMOTE sur l'ensemble des donnees...")
    final_model = get_models_with_smote()["HistGradientBoosting"]
    final_model.fit(X, y)

    model_path = OUTPUT_DIR / "model_qualite.pkl"
    cols_path  = OUTPUT_DIR / "feature_columns.pkl"
    joblib.dump(final_model, model_path)
    joblib.dump(feat_cols, cols_path)
    print(f"  Modele sauvegarde            : {model_path}")
    print(f"  Colonnes de features (ordre) : {cols_path}")
    print(f"    {feat_cols}")


if __name__ == "__main__":
    run_smote_comparison()
