"""
AquaReco - Point d'entree principal
Execute l'ETL, le scoring meteo, le scoring multicritere et l'EDA.
"""

import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from pathlib import Path

from etl              import build_consolidated, missing_report
from meteo            import compute_meteo_scores
from osm              import fetch_osm_equipements
from score            import compute_all_scores, EXPERT_WEIGHTS
from eda              import run_eda
from preprocessing    import build_feature_matrix
from models           import run_models
from validation       import run_validation
from synthetic_users  import run_synthetic

OUTPUT_DIR = Path(__file__).parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


def main(use_meteo_cache: bool = True, use_osm_cache: bool = True) -> None:
    print("=" * 60)
    print("  AquaReco - Systeme de recommandation de baignade")
    print("  Semaines 1-4 : ETL + OSM + Meteo + Score + EDA + Validation + Synthese")
    print("=" * 60)

    # ── 1. ETL ────────────────────────────────────────────────────
    df_site_year, df_analyses = build_consolidated()

    print("\nRapport qualite donnees :")
    print(missing_report(df_site_year).to_string())

    # ── 2. Enrichissement OSM (Overpass API) ─────────────────────
    print("\n--- Equipements de proximite (OpenStreetMap) ---")
    try:
        df_equip = fetch_osm_equipements(df_site_year, use_cache=use_osm_cache)
        print(f"Equipements OSM recuperes pour {len(df_equip):,} sites.")
    except Exception as e:
        print(f"[WARN] OSM indisponible ({e}). Pipeline continue sans equipements.")
        df_equip = None

    # ── 3. Sous-score météo (OpenMeteo) ───────────────────────────
    print("\n--- Sous-score meteo ---")
    try:
        df_meteo = compute_meteo_scores(
            df_analyses, df_site_year, use_cache=use_meteo_cache
        )
        print(f"Score meteo calcule pour {len(df_meteo):,} lignes site x saison.")
    except Exception as e:
        print(f"[WARN] Meteo indisponible ({e}). Pipeline continue sans sous-score meteo.")
        df_meteo = None

    # ── 4. Scoring multicritère ───────────────────────────────────
    df_scored, learned_weights = compute_all_scores(
        df_site_year, df_analyses, df_meteo=df_meteo
    )

    out_csv = OUTPUT_DIR / "sites_scores.csv"
    df_scored.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"\nFichier exporte : {out_csv}")

    print("\n--- Synthese par saison ---")
    summary_cols = [c for c in ["score_expert", "score_appris",
                                "score_meteo", "classement"] if c in df_scored.columns]
    print(
        df_scored.groupby("saison")[summary_cols]
        .agg(["mean", "count"])
        .round(2)
        .to_string()
    )

    print("\n--- Ponderations comparees ---")
    print(f"  Expert : {EXPERT_WEIGHTS}")
    print(f"  Appris : {learned_weights}")

    # ── 5. EDA ────────────────────────────────────────────────────
    run_eda(df_scored, EXPERT_WEIGHTS, learned_weights, df_analyses=df_analyses)

    # ── 6. Validation du score expert ────────────────────────────
    print("\n--- Validation du score expert ---")
    run_validation()

    # ── 7. Feature engineering ML (preprocessing) ────────────────
    print("\n--- Feature engineering ML ---")
    build_feature_matrix(use_cache=True)

    # ── 8. Comparaison des modeles ML ─────────────────────────────
    print("\n--- Comparaison des modeles ML ---")
    run_models()

    # ── 9. Profils utilisateurs synthetiques ─────────────────────
    print("\n--- Generation des profils synthetiques ---")
    run_synthetic()

    print("\n" + "=" * 60)
    print("  Pipeline termine. Resultats dans outputs/")
    print("=" * 60)


if __name__ == "__main__":
    # use_meteo_cache=False / use_osm_cache=False pour forcer le re-telechargement
    main(use_meteo_cache=True, use_osm_cache=True)
