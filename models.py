"""
models.py - AquaReco Semaine 4
Entraîne et compare 5 modèles ML sur les features issues de preprocessing.py.
Produit un tableau comparatif + SHAP values pour l'interprétabilité.
"""

import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso, LogisticRegression, Ridge, RidgeClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, f1_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

OUTPUT_DIR = Path(__file__).parent / "outputs"
EDA_DIR    = OUTPUT_DIR / "eda"
EDA_DIR.mkdir(parents=True, exist_ok=True)

LABEL_MAP   = {0: "Excellent", 1: "Bon", 2: "Suffisant", 3: "Insuffisant"}
CLASS_ORDER = [0, 1, 2, 3]

# Features à utiliser depuis preprocessing.py
# (les colonnes manquantes sont ignorées automatiquement)
CANDIDATE_FEATURES = [
    "p95_log_ecoli", "p95_log_ent",
    "mean_log_ecoli", "mean_log_ent",
    "std_log_ecoli",  "std_log_ent",
    "n_prelevements",
    "mois_sin", "mois_cos", "tendance",
    "type_lac", "type_mer_cote", "type_riviere", "type_transition", "type_autre",
    "parking", "sanitaires", "pmr", "douche", "poste_secours",
    # Météo si cache v2 disponible
    "precip_7j_median", "temp_7j_median",
    "wind_speed_7j_median", "wind_sin_7j_median", "wind_cos_7j_median",
]


# ── Chargement des données ────────────────────────────────────────────────────

def load_features() -> tuple[np.ndarray, np.ndarray, list]:
    """Charge features.parquet et retourne (X, y, feature_names)."""
    path = OUTPUT_DIR / "features.parquet"
    if not path.exists():
        raise FileNotFoundError(
            "features.parquet introuvable. Lancez preprocessing.py d'abord."
        )
    df = pd.read_parquet(path)
    df = df.dropna(subset=["classement"]).copy()
    df["classement"] = df["classement"].astype(int)

    feat_cols = [c for c in CANDIDATE_FEATURES if c in df.columns]
    missing   = [c for c in CANDIDATE_FEATURES if c not in df.columns]
    if missing:
        print(f"  [INFO] Features absentes (cache meteo v2 ?) : {missing}")

    # Supprime les lignes sans aucune feature bactério (cas extrêmes)
    bact_cols = [c for c in feat_cols if "ecoli" in c or "ent" in c]
    df = df.dropna(subset=bact_cols[:2])

    X = df[feat_cols].values.astype(float)
    y = df["classement"].values - 1    # [1,2,3,4] → [0,1,2,3] (XGBoost + uniformité)
    return X, y, feat_cols


# ── Définition des modèles ────────────────────────────────────────────────────

def get_models() -> dict:
    """
    Retourne un dict {nom: pipeline scikit-learn}.
    Ridge et Lasso sont encapsulés en classifieurs multiclasse (OvR + seuil).
    """
    import xgboost as xgb

    # Imputation médiane pour les modèles ne gérant pas nativement les NaN
    imputer = SimpleImputer(strategy="median")

    models = {
        "Ridge (OvR)": Pipeline([
            ("imp",    SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf",    OneVsRestClassifier(Ridge(alpha=1.0))),
        ]),
        "Lasso (OvR)": Pipeline([
            ("imp",    SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf",    OneVsRestClassifier(Lasso(alpha=0.01, max_iter=5000))),
        ]),
        # HistGB gère nativement les NaN - pas d'imputation
        "HistGradientBoosting": HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.05,
            max_leaf_nodes=31, random_state=42,
            class_weight="balanced",
        ),
        "XGBoost": Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("clf", xgb.XGBClassifier(
                n_estimators=300, learning_rate=0.05, max_depth=6,
                subsample=0.8, colsample_bytree=0.8,
                eval_metric="mlogloss",
                n_jobs=-1, random_state=42,
            )),
        ]),
    }
    return models


# ── Métrique ±1 classe ────────────────────────────────────────────────────────

def accuracy_within_1(y_true, y_pred) -> float:
    return np.mean(np.abs(np.array(y_true) - np.array(y_pred)) <= 1)


# ── Évaluation par validation croisée ─────────────────────────────────────────

def evaluate_model(name: str, model, X: np.ndarray, y: np.ndarray,
                   k: int = 5,
                   class_order: list = None,
                   label_map: dict = None,
                   fit_params: dict = None) -> dict:
    """Cross-validation k=5 stratifiée, retourne les métriques."""
    if class_order is None: class_order = CLASS_ORDER
    if label_map   is None: label_map   = LABEL_MAP

    cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)

    if fit_params:
        # Boucle CV manuelle pour passer sample_weight correctement par fold
        y_pred = np.empty(len(y), dtype=int)
        for train_idx, test_idx in cv.split(X, y):
            fold_kw = {key: val[train_idx] for key, val in fit_params.items()}
            model.fit(X[train_idx], y[train_idx], **fold_kw)
            y_pred[test_idx] = model.predict(X[test_idx])
    else:
        y_pred = cross_val_predict(model, X, y, cv=cv, n_jobs=-1)

    acc  = accuracy_score(y, y_pred)
    off1 = accuracy_within_1(y, y_pred)
    report = classification_report(
        y, y_pred, labels=class_order,
        target_names=[label_map[c] for c in class_order],
        output_dict=True, zero_division=0,
    )
    f1_macro = f1_score(y, y_pred, average="macro", zero_division=0)
    f1_w     = f1_score(y, y_pred, average="weighted", zero_division=0)

    print(f"\n  [{name}]")
    print(f"  Accuracy : {acc*100:.1f}%   ±1 classe : {off1*100:.1f}%   "
          f"F1-macro : {f1_macro:.3f}   F1-weighted : {f1_w:.3f}")
    for cl in class_order:
        r = report[label_map[cl]]
        print(f"    {label_map[cl]:<14} P={r['precision']:.2f}  "
              f"R={r['recall']:.2f}  F1={r['f1-score']:.2f}  "
              f"n={r['support']:.0f}")

    def _get_f1(lbl_candidates):
        for lbl in lbl_candidates:
            if lbl in report:
                return round(report[lbl]["f1-score"], 4)
        return 0.0

    return {
        "model":          name,
        "accuracy":       round(acc,  4),
        "acc_pm1":        round(off1, 4),
        "f1_macro":       round(f1_macro, 4),
        "f1_weighted":    round(f1_w, 4),
        "f1_excellent":   _get_f1(["Excellent"]),
        "f1_bon":         _get_f1(["Bon"]),
        "f1_suffisant":   _get_f1(["Suffisant"]),
        "f1_insuffisant": _get_f1(["Insuffisant", "Non conf."]),
        "y_pred":         y_pred,
    }


# ── Graphiques ────────────────────────────────────────────────────────────────

def plot_confusion_all(results: list, y: np.ndarray, suffix: str = "") -> None:
    """Matrice de confusion pour chaque modèle (sauf Ridge/Lasso)."""
    linear_names = ("Ridge (OvR)", "Lasso (OvR)", "Ridge (balanced)", "Lasso/LR-L1 (balanced)")
    tree_models = [r for r in results if r["model"] not in linear_names]
    if not tree_models:
        return

    fig, axes = plt.subplots(1, len(tree_models),
                              figsize=(5 * len(tree_models), 4.5))
    if len(tree_models) == 1:
        axes = [axes]

    for ax, r in zip(axes, tree_models):
        cm = confusion_matrix(y, r["y_pred"], labels=CLASS_ORDER)
        cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100
        im = ax.imshow(cm_pct, cmap="Blues", vmin=0, vmax=100)
        labels = [LABEL_MAP[c] for c in CLASS_ORDER]
        ax.set_xticks(range(4)); ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
        ax.set_yticks(range(4)); ax.set_yticklabels(labels, fontsize=8)
        ax.set_title(r["model"], fontsize=10)
        ax.set_xlabel("Prédit"); ax.set_ylabel("Réel")
        for i in range(4):
            for j in range(4):
                color = "white" if cm_pct[i,j] > 55 else "black"
                ax.text(j, i, f"{cm_pct[i,j]:.0f}%",
                        ha="center", va="center", fontsize=8, color=color)

    fig.suptitle("Matrices de confusion (% par ligne, CV k=5)", fontsize=12)
    fig.tight_layout()
    out = EDA_DIR / f"models_confusion{suffix}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Matrices de confusion : {out}")


def plot_comparison_bar(df_comp: pd.DataFrame, suffix: str = "") -> None:
    """Barplot comparatif des métriques pour tous les modèles."""
    metrics = ["accuracy", "acc_pm1", "f1_macro", "f1_weighted"]
    labels  = ["Accuracy", "Acc ±1 cl.", "F1 macro", "F1 weighted"]
    x = np.arange(len(df_comp))
    width = 0.18

    fig, ax = plt.subplots(figsize=(12, 5))
    for i, (met, lab) in enumerate(zip(metrics, labels)):
        offset = (i - 1.5) * width
        bars = ax.bar(x + offset, df_comp[met], width, label=lab, alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(df_comp["model"], rotation=15, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Comparaison des modèles — Validation croisée k=5")
    ax.legend(loc="lower right")
    ax.axhline(0.8, color="grey", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.axhline(0.9, color="grey", linestyle=":",  linewidth=0.8, alpha=0.6)
    fig.tight_layout()
    out = EDA_DIR / f"models_comparison{suffix}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Barplot comparaison : {out}")


# ── SHAP values ───────────────────────────────────────────────────────────────

def plot_shap(model_name: str, model, X: np.ndarray,
              y: np.ndarray, feature_names: list) -> None:
    """
    Calcule et trace les SHAP values pour un modèle arborescent.
    Entraîne sur l'ensemble complet (interprétatibilité globale, pas de CV).
    """
    try:
        import shap
    except ImportError:
        print(f"  [WARN] shap non installe — SHAP values ignorees pour {model_name}.")
        return

    print(f"  Calcul SHAP pour {model_name}...")
    model.fit(X, y)

    if hasattr(model, "named_steps"):
        clf = model.named_steps.get("clf", model)
        # Transforme X si pipeline avec imputer/scaler avant le clf
        X_tf = X.copy()
        for step_name, step in model.named_steps.items():
            if step_name == "clf":
                break
            X_tf = step.transform(X_tf)
    else:
        clf  = model
        X_tf = X

    idx   = np.random.default_rng(0).choice(len(X_tf), min(3000, len(X_tf)), replace=False)
    X_sub = X_tf[idx]

    try:
        explainer = shap.TreeExplainer(clf)
        shap_vals = explainer.shap_values(X_sub)
        if isinstance(shap_vals, list):
            imp = np.mean([np.abs(sv) for sv in shap_vals], axis=0).mean(axis=0)
        elif shap_vals.ndim == 3:
            imp = np.abs(shap_vals).mean(axis=(0, 2))
        else:
            imp = np.abs(shap_vals).mean(axis=0)
    except Exception as exc:
        # Fallback : feature_importances_ du modèle (XGBoost multiclass + SHAP compat)
        print(f"  [INFO] SHAP TreeExplainer indisponible ({exc.__class__.__name__}), "
              f"utilisation de feature_importances_.")
        if hasattr(clf, "feature_importances_"):
            imp = clf.feature_importances_
        else:
            print(f"  [WARN] Pas de feature_importances_ — SHAP ignore pour {model_name}.")
            return

    order = np.argsort(imp)[::-1][:15]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(range(len(order)), imp[order][::-1], color="steelblue", alpha=0.8)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([feature_names[i] for i in order[::-1]], fontsize=9)
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title(f"Feature importance SHAP — {model_name}")
    fig.tight_layout()
    slug = model_name.lower().replace(" ", "_").replace("(", "").replace(")", "")
    out  = EDA_DIR / f"shap_{slug}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  SHAP {model_name} : {out}")


# ── Pipeline principal ────────────────────────────────────────────────────────

def run_models() -> pd.DataFrame:
    global y_shap

    print("="*62)
    print("  COMPARAISON DES MODELES ML")
    print("="*62)

    X, y, feat_cols = load_features()

    print(f"\n  Dataset : {X.shape[0]:,} lignes x {X.shape[1]} features")
    print(f"  Classes : {dict(zip(*np.unique(y, return_counts=True)))}")
    print(f"\n  Validation croisee k=5 stratifiee en cours...")

    models  = get_models()
    results = []
    for name, model in models.items():
        res = evaluate_model(name, model, X, y)
        results.append(res)

    # ── Tableau comparatif ─────────────────────────────────────────
    df_comp = pd.DataFrame([
        {k: v for k, v in r.items() if k != "y_pred"}
        for r in results
    ]).sort_values("f1_macro", ascending=False)

    out_csv = OUTPUT_DIR / "models_comparison.csv"
    df_comp.to_csv(out_csv, index=False, encoding="utf-8-sig")

    print(f"\n  {'='*62}")
    print(f"  TABLEAU RECAPITULATIF (trie par F1 macro)")
    print(f"  {'='*62}")
    print(f"  {'Modele':<26} {'Acc':>6} {'±1cl':>6} "
          f"{'F1mac':>7} {'F1Exc':>7} {'F1Bon':>7} "
          f"{'F1Suf':>7} {'F1Ins':>7}")
    print("  " + "-"*62)
    for _, r in df_comp.iterrows():
        print(f"  {r['model']:<26} "
              f"{r['accuracy']*100:>5.1f}% "
              f"{r['acc_pm1']*100:>5.1f}% "
              f"{r['f1_macro']:>7.3f} "
              f"{r['f1_excellent']:>7.3f} "
              f"{r['f1_bon']:>7.3f} "
              f"{r['f1_suffisant']:>7.3f} "
              f"{r['f1_insuffisant']:>7.3f}")
    print(f"  {'='*62}")
    print(f"  Tableau sauvegarde : {out_csv}")

    # ── Graphiques ────────────────────────────────────────────────
    plot_confusion_all(results, y)
    plot_comparison_bar(df_comp)

    # ── SHAP values (modèles arborescents) ────────────────────────
    print("\n  --- SHAP feature importance ---")
    for name in ["HistGradientBoosting", "XGBoost"]:
        if name in models:
            plot_shap(name, models[name], X, y, feat_cols)

    print(f"\n  Graphiques sauvegardes dans {EDA_DIR}/")
    print("="*62)
    return df_comp


# ── Pipeline per-sample ───────────────────────────────────────────────────────

CANDIDATE_FEATURES_PER_SAMPLE = [
    "log_ecoli", "log_ent",
    "precip_7j", "temp_7j", "wind_speed_7j", "wind_sin_7j", "wind_cos_7j",
    "mois_sin", "mois_cos", "tendance",
    "type_lac", "type_mer_cote", "type_riviere", "type_transition", "type_autre",
    "parking", "sanitaires", "pmr", "douche", "poste_secours",
]


_EU_LABELS = {0: "Excellent", 1: "Bon", 2: "Suffisant", 3: "Non conf."}


def load_features_per_sample() -> tuple[np.ndarray, np.ndarray, list, list, dict]:
    """
    Charge features_per_sample.parquet.
    Retourne (X, y, feature_names, class_order, label_map).
    y est remappé en entiers consécutifs 0..n-1 (XGBoost l'exige).
    """
    path = OUTPUT_DIR / "features_per_sample.parquet"
    if not path.exists():
        raise FileNotFoundError(
            "features_per_sample.parquet introuvable. "
            "Lancez preprocessing.py (build_feature_matrix_per_sample) d'abord."
        )
    df = pd.read_parquet(path)
    df = df.dropna(subset=["classement_eu"]).copy()
    df["classement_eu"] = df["classement_eu"].astype(int)

    feat_cols = [c for c in CANDIDATE_FEATURES_PER_SAMPLE if c in df.columns]
    missing   = [c for c in CANDIDATE_FEATURES_PER_SAMPLE if c not in df.columns]
    if missing:
        print(f"  [INFO] Features absentes : {missing}")

    raw_classes = sorted(df["classement_eu"].unique().tolist())
    remap = {c: i for i, c in enumerate(raw_classes)}
    if raw_classes != list(range(len(raw_classes))):
        print(f"  [INFO] Classes presentes : {raw_classes} — remappage vers "
              f"{list(range(len(raw_classes)))}")
    class_order = list(range(len(raw_classes)))
    label_map   = {remap[c]: _EU_LABELS[c] for c in raw_classes}

    X = df[feat_cols].values.astype(float)
    y = np.array([remap[c] for c in df["classement_eu"].values])
    return X, y, feat_cols, class_order, label_map


def get_models_per_sample() -> dict:
    """
    Modèles avec class_weight='balanced'.
    Ridge → RidgeClassifier, Lasso → LogisticRegression(L1).
    """
    import xgboost as xgb

    return {
        "Ridge (balanced)": Pipeline([
            ("imp",    SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf",    RidgeClassifier(alpha=1.0, class_weight="balanced")),
        ]),
        "Lasso/LR-L1 (balanced)": Pipeline([
            ("imp",    SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf",    LogisticRegression(
                penalty="l1", solver="saga", C=100,
                class_weight="balanced", max_iter=5000, random_state=42,
            )),
        ]),
        "HistGradientBoosting": HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.05,
            max_leaf_nodes=31, random_state=42,
            class_weight="balanced",
        ),
        "XGBoost": Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("clf", xgb.XGBClassifier(
                n_estimators=300, learning_rate=0.05, max_depth=6,
                subsample=0.8, colsample_bytree=0.8,
                eval_metric="mlogloss",
                n_jobs=-1, random_state=42,
            )),
        ]),
    }


def run_models_per_sample() -> pd.DataFrame:
    """Run models on features_per_sample.parquet, save to models_comparison_per_sample.csv."""
    print("=" * 62)
    print("  COMPARAISON DES MODELES ML (per-sample)")
    print("=" * 62)

    X, y, feat_cols, class_order, label_map = load_features_per_sample()

    cls_counts = dict(zip(*np.unique(y, return_counts=True)))
    cls_display = {label_map[k]: v for k, v in cls_counts.items()}
    print(f"\n  Dataset : {X.shape[0]:,} lignes x {X.shape[1]} features")
    print(f"  Classes : {cls_display}")
    print(f"\n  Validation croisee k=5 stratifiee en cours...")

    models  = get_models_per_sample()
    results = []
    for name, model in models.items():
        res = evaluate_model(name, model, X, y,
                             class_order=class_order, label_map=label_map)
        results.append(res)

    df_comp = pd.DataFrame([
        {k: v for k, v in r.items() if k != "y_pred"}
        for r in results
    ]).sort_values("f1_macro", ascending=False)

    out_csv = OUTPUT_DIR / "models_comparison_per_sample.csv"
    df_comp.to_csv(out_csv, index=False, encoding="utf-8-sig")

    print(f"\n  {'='*62}")
    print(f"  TABLEAU RECAPITULATIF — per-sample (trie par F1 macro)")
    print(f"  {'='*62}")
    print(f"  {'Modele':<26} {'Acc':>6} {'±1cl':>6} "
          f"{'F1mac':>7} {'F1Exc':>7} {'F1Bon':>7} "
          f"{'F1Suf':>7} {'F1NCf':>7}")
    print("  " + "-" * 62)
    for _, r in df_comp.iterrows():
        print(f"  {r['model']:<26} "
              f"{r['accuracy']*100:>5.1f}% "
              f"{r['acc_pm1']*100:>5.1f}% "
              f"{r['f1_macro']:>7.3f} "
              f"{r['f1_excellent']:>7.3f} "
              f"{r['f1_bon']:>7.3f} "
              f"{r['f1_suffisant']:>7.3f} "
              f"{r['f1_insuffisant']:>7.3f}")
    print(f"  {'='*62}")
    print(f"  Tableau sauvegarde : {out_csv}")

    plot_confusion_all(results, y, suffix="_per_sample")
    plot_comparison_bar(df_comp, suffix="_per_sample")

    print("\n  --- SHAP feature importance ---")
    for name in ["HistGradientBoosting", "XGBoost"]:
        if name in models:
            plot_shap(name, models[name], X, y, feat_cols)

    print(f"\n  Graphiques sauvegardes dans {EDA_DIR}/")
    print("=" * 62)
    return df_comp


# ── Pipeline temporel ────────────────────────────────────────────────────────

CANDIDATE_FEATURES_TEMPORAL = [
    "ecoli_moy_hist", "ent_moy_hist",
    "ecoli_dernier",  "ent_dernier",
    "n_prelevements_ant",
    "precip_7j", "temp_7j", "wind_speed_7j", "wind_sin_7j", "wind_cos_7j",
    "mois_sin", "mois_cos", "tendance",
    "type_lac", "type_mer_cote", "type_riviere", "type_transition", "type_autre",
    "parking", "sanitaires", "pmr", "douche", "poste_secours",
]


def load_features_temporal() -> tuple[np.ndarray, np.ndarray, list, list, dict]:
    """
    Charge features_temporal.parquet.
    Retourne (X, y, feature_names, class_order, label_map).
    y remappé en entiers consécutifs (XGBoost l'exige).
    """
    path = OUTPUT_DIR / "features_temporal.parquet"
    if not path.exists():
        raise FileNotFoundError(
            "features_temporal.parquet introuvable. "
            "Lancez preprocessing.py (build_feature_matrix_temporal) d'abord."
        )
    df = pd.read_parquet(path)
    df = df.dropna(subset=["classement_eu"]).copy()
    df["classement_eu"] = df["classement_eu"].astype(int)

    feat_cols = [c for c in CANDIDATE_FEATURES_TEMPORAL if c in df.columns]
    missing   = [c for c in CANDIDATE_FEATURES_TEMPORAL if c not in df.columns]
    if missing:
        print(f"  [INFO] Features absentes : {missing}")

    raw_classes = sorted(df["classement_eu"].unique().tolist())
    remap = {c: i for i, c in enumerate(raw_classes)}
    if raw_classes != list(range(len(raw_classes))):
        print(f"  [INFO] Classes {raw_classes} → remappage {list(range(len(raw_classes)))}")
    class_order = list(range(len(raw_classes)))
    label_map   = {remap[c]: _EU_LABELS[c] for c in raw_classes}

    X = df[feat_cols].values.astype(float)
    y = np.array([remap[c] for c in df["classement_eu"].values])
    return X, y, feat_cols, class_order, label_map


def get_models_temporal() -> dict:
    """
    Modèles avec class_weight='balanced'.
    HistGB et XGBoost reçoivent sample_weight via fit_params dans run_models_temporal.
    """
    import xgboost as xgb

    return {
        "Ridge (balanced)": Pipeline([
            ("imp",    SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf",    RidgeClassifier(alpha=1.0, class_weight="balanced")),
        ]),
        "Lasso/LR-L1 (balanced)": Pipeline([
            ("imp",    SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf",    LogisticRegression(
                penalty="l1", solver="saga", C=100,
                class_weight="balanced", max_iter=5000, random_state=42,
            )),
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
                eval_metric="mlogloss",
                n_jobs=-1, random_state=42,
            )),
        ]),
    }


def run_models_temporal() -> pd.DataFrame:
    """Run models sur features_temporal.parquet, sauvegarde models_comparison_temporal.csv."""
    from sklearn.utils.class_weight import compute_sample_weight

    print("=" * 62)
    print("  COMPARAISON DES MODELES ML (temporal — sans fuite)")
    print("=" * 62)

    X, y, feat_cols, class_order, label_map = load_features_temporal()
    sw = compute_sample_weight("balanced", y)

    cls_counts  = dict(zip(*np.unique(y, return_counts=True)))
    cls_display = {label_map[k]: v for k, v in cls_counts.items()}
    print(f"\n  Dataset : {X.shape[0]:,} lignes x {X.shape[1]} features")
    print(f"  Classes : {cls_display}")
    print(f"\n  Validation croisee k=5 stratifiee en cours...")

    models = get_models_temporal()
    results = []
    for name, model in models.items():
        if name == "HistGradientBoosting":
            fp = {"sample_weight": sw}
        elif name == "XGBoost":
            fp = {"clf__sample_weight": sw}
        else:
            fp = None
        res = evaluate_model(name, model, X, y,
                             class_order=class_order, label_map=label_map,
                             fit_params=fp)
        results.append(res)

    df_comp = pd.DataFrame([
        {k: v for k, v in r.items() if k != "y_pred"}
        for r in results
    ]).sort_values("f1_macro", ascending=False)

    out_csv = OUTPUT_DIR / "models_comparison_temporal.csv"
    df_comp.to_csv(out_csv, index=False, encoding="utf-8-sig")

    print(f"\n  {'='*62}")
    print(f"  TABLEAU RECAPITULATIF — temporel (trie par F1 macro)")
    print(f"  {'='*62}")
    print(f"  {'Modele':<26} {'Acc':>6} {'±1cl':>6} "
          f"{'F1mac':>7} {'F1Exc':>7} {'F1Bon':>7} {'F1NCf':>7}")
    print("  " + "-" * 62)
    for _, r in df_comp.iterrows():
        print(f"  {r['model']:<26} "
              f"{r['accuracy']*100:>5.1f}% "
              f"{r['acc_pm1']*100:>5.1f}% "
              f"{r['f1_macro']:>7.3f} "
              f"{r['f1_excellent']:>7.3f} "
              f"{r['f1_bon']:>7.3f} "
              f"{r['f1_insuffisant']:>7.3f}")
    print(f"  {'='*62}")
    print(f"  Tableau sauvegarde : {out_csv}")

    plot_confusion_all(results, y, suffix="_temporal")
    plot_comparison_bar(df_comp, suffix="_temporal")

    print("\n  --- SHAP feature importance ---")
    for name in ["HistGradientBoosting", "XGBoost"]:
        if name in models:
            slug = name.lower().replace(" ", "_")
            plot_shap(name, models[name], X, y, feat_cols)

    print(f"\n  Graphiques sauvegardes dans {EDA_DIR}/")
    print("=" * 62)
    return df_comp


if __name__ == "__main__":
    run_models_temporal()
