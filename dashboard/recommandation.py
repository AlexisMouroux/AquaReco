"""Onglet Recommandation - hero banner, profil cards, result cards enrichis."""
from __future__ import annotations

import unicodedata
from datetime import date as _date
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from auth import require_login
from components.fiche_site import show_fiche_dialog
from database.db import add_favori, get_preferences, remove_favori, save_preferences
from data_loader import (
    CLASSEMENT_COLORS,
    CLASSEMENT_LABELS,
    geocode_ville,
    load_historique_officiel,
    load_sites,
)

DEFAULT_LAT = 46.603354
DEFAULT_LON = 1.888334

# ── Constantes ────────────────────────────────────────────────────────────────

PROFIL_INFO: list[tuple[str, str, str]] = [
    ("Famille",          "👨‍👩‍👧", "Sites surveillés, sable, équipements enfants"),
    ("Sportif",          "🏊",    "Eau profonde, conditions pour la nage et la voile"),
    ("Senior",           "🧘",    "Accès facile, calme, eau tempérée"),
    ("Aventurier",       "🧗",    "Sites remarquables et nature préservée"),
    ("Vacancier côtier", "🌊",    "Plages de mer, ensoleillement maximum"),
    ("Touriste étranger","🌍",    "Sites emblématiques, top qualité"),
]

SCORE_MIN_MAP = {"tous": 0.0, "bonne": 0.50, "excellente": 0.70}
TYPE_OPTIONS = {
    "Tous les types": "tous",
    "Lac": "lac",
    "Rivière / fluvial": "riviere",
    "Côtier / mer": "mer",
}
TYPE_OPTIONS_INV = {v: k for k, v in TYPE_OPTIONS.items()}
SCORE_OPTIONS = {
    "Tous": "tous",
    "Bonne qualité (≥ 50/100)": "bonne",
    "Excellente qualité (≥ 70/100)": "excellente",
}
SCORE_OPTIONS_INV = {v: k for k, v in SCORE_OPTIONS.items()}

RANK_COLORS = ["#E6B800", "#B0B0B0", "#CD7F32"]   # or, argent, bronze
RANK_TEXT   = ["#333",    "white",   "white"]

_TYPE_EXACT   = {"lac": "Lac", "mer": "Mer", "riviere": "Riviere"}
_TYPE_PARTIAL = {"mer": "Cote", "riviere": "Transition"}

WEIGHTS = (0.4, 0.2, 0.1, 0.3)   # wq, wt, we, wd

_DESCS: dict[tuple[str, int], str] = {
    ("Eau côtière",      0): "Eau de mer transparente, site de qualité supérieure",
    ("Eau côtière",      1): "Plage aux conditions favorables, qualité conforme",
    ("Eau côtière",      3): "Site côtier — vérifier les conditions avant la baignade",
    ("Lac",              0): "Eau douce cristalline, idéal pour la natation et la détente",
    ("Lac",              1): "Plan d'eau de bonne qualité, agréable pour la famille",
    ("Lac",              3): "Plan d'eau — respecter les éventuels avis de fermeture",
    ("Rivière",          0): "Eau vive de qualité supérieure, site naturel préservé",
    ("Rivière",          1): "Rivière praticable, qualité conforme aux normes européennes",
    ("Rivière",          3): "Rivière — consulter les analyses récentes avant la baignade",
    ("Eau de transition",0): "Zone estuarienne de qualité remarquable, cadre naturel exceptionnel",
    ("Eau de transition",1): "Site de transition conforme, baignade généralement possible",
    ("Eau de transition",3): "Zone de transition — prudence recommandée",
}


# ── Chargement ────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def _load_scores() -> pd.DataFrame:
    path = Path(__file__).parent.parent / "outputs" / "sites_scores.csv"
    return (
        pd.read_csv(path)[["code_site", "saison", "score_expert"]]
        .dropna(subset=["score_expert"])
        .sort_values("saison", ascending=False)
        .drop_duplicates("code_site")[["code_site", "score_expert"]]
    )


@st.cache_data(show_spinner=False)
def _load_sites_enriched() -> pd.DataFrame:
    sites = load_sites()
    scores = _load_scores()
    sites = sites.merge(scores, on="code_site", how="left")
    sites["score_expert"] = sites["score_expert"].fillna(50.0)
    sites["type_norm"] = sites["type_eau"].apply(_norm_type)
    return sites


def _norm_type(s: str) -> str:
    s = unicodedata.normalize("NFD", str(s)).encode("ascii", "ignore").decode().lower()
    if "cotier" in s or "cotiere" in s or "cotie" in s:
        return "Cote"
    if "lac" in s:
        return "Lac"
    if "riviere" in s or "fleuve" in s:
        return "Riviere"
    if "mer" in s:
        return "Mer"
    if "transit" in s:
        return "Transition"
    return "Autre"


# ── Moteur content-based (utilisateur unique) ─────────────────────────────────

def _haversine_arr(lat0: float, lon0: float,
                   lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
    R = 6371.0
    lat0_r = np.radians(lat0)
    lat2_r = np.radians(lats)
    dlat = lat2_r - lat0_r
    dlon = np.radians(lons) - np.radians(lon0)
    a = np.sin(dlat / 2) ** 2 + np.cos(lat0_r) * np.cos(lat2_r) * np.sin(dlon / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def _recommend(
    sites: pd.DataFrame,
    lat: float,
    lon: float,
    distance_max: int,
    type_pref: str,
    score_min_val: float,
    k: int = 10,
) -> pd.DataFrame:
    wq, wt, we, wd = WEIGHTS
    site_q_raw  = (sites["score_expert"].values / 100.0).astype(float)
    site_types  = sites["type_norm"].values

    q_sc = np.where(site_q_raw >= score_min_val, site_q_raw, site_q_raw * 0.5)

    if type_pref == "tous":
        type_sc = np.ones(len(sites))
    else:
        exact   = _TYPE_EXACT.get(type_pref, "")
        partial = _TYPE_PARTIAL.get(type_pref, "")
        type_sc = np.zeros(len(sites))
        type_sc = np.where(site_types == exact,   1.0, type_sc)
        if partial:
            type_sc = np.where(site_types == partial, 0.5, type_sc)

    equip_sc = np.ones(len(sites))

    dists   = _haversine_arr(lat, lon, sites["latitude"].values, sites["longitude"].values)
    dist_sc = np.exp(-dists / max(distance_max, 1.0))
    final   = wq * q_sc + wt * type_sc + we * equip_sc + wd * dist_sc

    top_idx = np.argsort(-final)[:k]
    result  = sites.iloc[top_idx].copy()
    result["distance_km"] = dists[top_idx]
    result["_score"]      = final[top_idx]
    return result.reset_index(drop=True)


# ── Helpers HTML ──────────────────────────────────────────────────────────────

def _chip(icon: str, label: str, value: str) -> str:
    return (
        f'<div>'
        f'<div style="font-size:0.67em;color:#7ec8e3;text-transform:uppercase;'
        f'letter-spacing:0.11em;font-weight:600;margin-bottom:2px">'
        f'{icon}&nbsp;{label}</div>'
        f'<div style="font-weight:600;font-size:0.9em">{value}</div>'
        f'</div>'
    )


def _badge_cl(classement: int) -> str:
    label = CLASSEMENT_LABELS.get(classement, str(classement))
    color = CLASSEMENT_COLORS.get(classement, "#888")
    return (
        f'<span style="background:{color};color:white;padding:2px 8px;'
        f'border-radius:10px;font-size:0.78em;font-weight:600">{label}</span>'
    )


def _describe(type_eau: str, classement: int) -> str:
    return _DESCS.get((type_eau, classement), "Site de baignade surveillé")


def _profil_card(name: str, icon: str, desc: str, is_sel: bool) -> bool:
    """Affiche la carte profil (HTML) + bouton Streamlit. Retourne True si cliqué."""
    if is_sel:
        bg      = "linear-gradient(135deg,#1a3a5c,#0a2440)"
        border  = "#1a6b8a"
        nc      = "white"
        dc      = "rgba(255,255,255,0.7)"
        check   = '<span style="float:right;color:#7ec8e3;font-size:0.85em">✓</span>'
    else:
        bg      = "rgba(128,128,128,0.06)"
        border  = "#5ba3c9"
        nc      = "inherit"
        dc      = "#888"
        check   = ""

    st.markdown(
        f'<div style="background:{bg};border:2px solid {border};border-radius:12px;'
        f'padding:16px;margin-bottom:6px;min-height:98px">'
        f'<div style="font-size:1.4em;margin-bottom:6px">{icon}{check}</div>'
        f'<div style="font-weight:700;color:{nc};font-size:0.9em">{name}</div>'
        f'<div style="font-size:0.76em;color:{dc};margin-top:4px;line-height:1.4">'
        f'{desc}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )
    label = "✓ Actif" if is_sel else "Choisir"
    btype = "primary" if is_sel else "secondary"
    return bool(st.button(label, key=f"profil_{name}", type=btype, use_container_width=True))


# ── Affichage principal ───────────────────────────────────────────────────────

def show() -> None:
    """Affiche l'onglet Recommandation."""
    if not require_login("recommandation"):
        return

    user_id: int  = st.session_state.user_id
    user_profil: str = st.session_state.user_profil
    prefs    = get_preferences(user_id)
    dist_max = int(prefs.get("distance_max", 50))

    # ── CSS : champ localisation intégré dans le hero ────────────────────────
    st.markdown(
        """
        <style>
        [data-testid="stTextInput"]:has(input[placeholder="Entrez une ville..."]) {
            background: linear-gradient(135deg,#0a1628 0%,#1a3a5c 55%,#0d2137 100%);
            padding: 0 32px 14px;
            margin-top: -1rem;
        }
        [data-testid="stTextInput"]:has(input[placeholder="Entrez une ville..."]) label p {
            color: #7ec8e3 !important;
            font-size: 0.67em !important;
            text-transform: uppercase !important;
            letter-spacing: 0.11em !important;
            font-weight: 600 !important;
            margin-bottom: 4px !important;
        }
        input[placeholder="Entrez une ville..."] {
            background: rgba(13,33,55,0.92) !important;
            border: 1px solid rgba(255,255,255,0.25) !important;
            color: white !important;
        }
        input[placeholder="Entrez une ville..."]:focus {
            border-color: #7ec8e3 !important;
            box-shadow: 0 0 0 1px rgba(126,200,227,0.4) !important;
        }
        input[placeholder="Entrez une ville..."]::placeholder {
            color: rgba(255,255,255,0.45) !important;
        }
        [data-testid="stMarkdownContainer"]:has(#reco-hero-chips) {
            margin-top: -1rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # ── Date (compacte, droite) ───────────────────────────────────────────────
    _, date_col = st.columns([4, 1])
    with date_col:
        swim_date = st.date_input(
            "Date souhaitée",
            value=_date.today(),
            key="reco_date",
            label_visibility="collapsed",
        )

    # ── Hero : titre + sous-titre ─────────────────────────────────────────────
    st.markdown(
        """
        <div style="
          background:linear-gradient(135deg,#0a1628 0%,#1a3a5c 55%,#0d2137 100%);
          border-radius:16px 16px 0 0;padding:28px 32px 16px;color:white
        ">
          <p style="font-size:0.7em;color:#7ec8e3;text-transform:uppercase;
                    letter-spacing:0.14em;margin:0 0 8px;font-weight:600">
            RECOMMANDATION PERSONNALISÉE
          </p>
          <h2 style="font-size:1.75em;font-weight:700;margin:0 0 10px;line-height:1.25">
            Trouvez le spot idéal selon votre profil et votre date.
          </h2>
          <p style="color:rgba(255,255,255,0.68);font-size:0.87em;margin:0;line-height:1.55">
            Notre moteur croise qualité bactériologique, météo J+3 et tendance saisonnière
            pour vous proposer les meilleurs sites de baignade.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Champ localisation (dans le hero via CSS) ─────────────────────────────
    loc_query = st.text_input(
        "📍 Localisation",
        key="reco_loc_widget",
        placeholder="Entrez une ville...",
        label_visibility="visible",
    )

    # ── Résolution géographique ───────────────────────────────────────────────
    # last_location est partagé avec l'onglet Carte (niveau 2 de la localisation
    # par défaut) : {"lat":, "lon":, "nom":}, mis à jour par toute recherche de
    # ville réussie ici ou dans Carte.
    reco_q  = (loc_query or "").strip()
    ref_lat = float(st.session_state.get("ref_lat", DEFAULT_LAT))
    ref_lon = float(st.session_state.get("ref_lon", DEFAULT_LON))
    last_loc  = st.session_state.get("last_location")
    loc_label = last_loc["nom"] if last_loc else "France"

    if reco_q:
        coords = geocode_ville(reco_q)
        if coords:
            ref_lat, ref_lon = coords
            loc_label = reco_q.title()
            st.session_state["last_location"] = {"lat": ref_lat, "lon": ref_lon, "nom": reco_q}
        else:
            loc_label = f"« {reco_q} » introuvable"
            st.warning(f"Ville « {reco_q} » non trouvée — coordonnées par défaut utilisées.")

    # ── Hero : chips de contexte (bas du banner) ──────────────────────────────
    st.markdown(
        f"""
        <div id="reco-hero-chips" style="
          background:linear-gradient(135deg,#0a1628 0%,#1a3a5c 55%,#0d2137 100%);
          border-radius:0 0 16px 16px;padding:0 32px 24px;color:white;margin-bottom:22px
        ">
          <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:28px;
                      border-top:1px solid rgba(255,255,255,0.12);padding-top:16px">
            {_chip("📍", "Localisation", loc_label)}
            {_chip("📅", "Date", swim_date.strftime("%d/%m/%Y"))}
            {_chip("📏", "Rayon", f"{dist_max}&nbsp;km")}
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Sélection du profil ───────────────────────────────────────────────────
    selected_profil = st.session_state.get("reco_profil", user_profil)

    st.markdown(
        "<p style='font-size:0.72em;text-transform:uppercase;letter-spacing:0.12em;"
        "font-weight:600;color:#888;margin:0 0 10px'>VOTRE PROFIL</p>",
        unsafe_allow_html=True,
    )

    row1 = st.columns(3)
    row2 = st.columns(3)

    for i, (name, icon, desc) in enumerate(PROFIL_INFO):
        col = row1[i] if i < 3 else row2[i - 3]
        with col:
            if _profil_card(name, icon, desc, name == selected_profil):
                if name != selected_profil:
                    st.session_state["reco_profil"] = name
                    st.session_state["show_reco"] = False
                    st.rerun()

    # ── Préférences (expander compact) ───────────────────────────────────────
    with st.expander("⚙️ Modifier mes préférences de recherche"):
        with st.form("form_prefs_reco"):
            pa, pb, pc = st.columns(3)
            with pa:
                new_dist = st.slider(
                    "Rayon (km)",
                    5, 500, dist_max, 5,
                )
            with pb:
                type_cur = TYPE_OPTIONS_INV.get(prefs.get("type_eau_pref", "tous"), "Tous les types")
                new_type_lbl = st.selectbox("Type d'eau", list(TYPE_OPTIONS.keys()),
                                            index=list(TYPE_OPTIONS.keys()).index(type_cur))
            with pc:
                score_cur = SCORE_OPTIONS_INV.get(prefs.get("score_min", "tous"), "Tous")
                new_score_lbl = st.selectbox("Classement min", list(SCORE_OPTIONS.keys()),
                                             index=list(SCORE_OPTIONS.keys()).index(score_cur))
            if st.form_submit_button("💾 Sauvegarder", type="primary", use_container_width=True):
                save_preferences(user_id, new_dist,
                                 TYPE_OPTIONS[new_type_lbl],
                                 SCORE_OPTIONS[new_score_lbl])
                st.success("Préférences sauvegardées.")
                st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Bouton lancer ─────────────────────────────────────────────────────────
    if st.button("🎯 Obtenir mes recommandations", type="primary", use_container_width=True):
        st.session_state["show_reco"] = True

    if not st.session_state.get("show_reco"):
        st.caption(
            "Sélectionnez votre profil et cliquez pour générer vos recommandations."
        )
        return

    # ── Calcul ────────────────────────────────────────────────────────────────
    prefs        = get_preferences(user_id)
    type_pref    = prefs.get("type_eau_pref", "tous")
    score_min_val = SCORE_MIN_MAP.get(prefs.get("score_min", "tous"), 0.0)

    with st.spinner("Calcul des recommandations…"):
        sites    = _load_sites_enriched()
        results  = _recommend(sites, ref_lat, ref_lon, dist_max, type_pref, score_min_val)

    historique_all = load_historique_officiel()
    favoris: set   = st.session_state.setdefault("favoris", set())

    n = len(results)
    st.markdown(
        f"<b>{n} recommandation{'s' if n != 1 else ''}</b>"
        f"&nbsp;·&nbsp;profil <b>{selected_profil}</b>",
        unsafe_allow_html=True,
    )
    st.markdown("<br>", unsafe_allow_html=True)

    max_score = float(results["_score"].max()) if n > 0 else 1.0

    for rank, (_, row) in enumerate(results.iterrows(), start=1):
        code       = row["code_site"]
        is_fav     = code in favoris
        cl         = int(row["classement_predit"])
        score_pct  = max(1, int(99 * row["_score"] / max_score)) if max_score > 0 else 0
        rank_bg    = RANK_COLORS[rank - 1] if rank <= 3 else "#1a6b8a"
        rank_fg    = RANK_TEXT[rank - 1]   if rank <= 3 else "white"
        badge_html = _badge_cl(cl)
        desc       = _describe(row["type_eau"], cl)
        score_val  = int(row.get("score_expert") or 0)
        commune    = str(row.get("commune") or "").strip()
        heart      = "♥" if is_fav else "♡"

        card_col, btn_col = st.columns([11, 1])

        with card_col:
            st.markdown(
                f"""
                <div style="display:flex;gap:14px;padding:12px 0 4px">
                  <div style="
                    background:{rank_bg};color:{rank_fg};border-radius:8px;
                    min-width:36px;width:36px;height:36px;flex-shrink:0;
                    display:flex;align-items:center;justify-content:center;
                    font-weight:700;font-size:1em">{rank}</div>
                  <div style="flex:1;min-width:0">
                    <div style="font-weight:700;font-size:1.05em;margin-bottom:4px">
                      {row['nom_site']}
                    </div>
                    <div style="font-size:0.83em;color:#888;margin-bottom:6px">
                      {badge_html}&nbsp;·&nbsp;{score_val}pts
                      &nbsp;·&nbsp;{row['type_eau']}
                      {"&nbsp;·&nbsp;" + commune if commune else ""}
                      &nbsp;·&nbsp;{row['distance_km']:.0f}&nbsp;km
                    </div>
                    <div style="font-size:0.81em;color:#777;margin-bottom:8px;font-style:italic">
                      {desc}
                    </div>
                    <div style="background:rgba(128,128,128,0.15);border-radius:4px;
                                height:5px;overflow:hidden">
                      <div style="background:#1a6b8a;width:{score_pct}%;height:5px;
                                  border-radius:4px"></div>
                    </div>
                    <div style="font-size:0.73em;color:#888;margin-top:3px">
                      {score_pct}% pertinence
                    </div>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with btn_col:
            if st.button("📋", key=f"reco_fiche_{code}", help="Voir la fiche"):
                site_row = sites[sites["code_site"] == code].iloc[0]
                hist_s   = historique_all[historique_all["code_site"] == code]
                show_fiche_dialog(site_row, hist_s)
            if st.button(heart, key=f"reco_fav_{code}", help="Ajouter/retirer des favoris"):
                if is_fav:
                    favoris.discard(code)
                    if st.session_state.get("logged_in"):
                        remove_favori(st.session_state.user_id, code)
                else:
                    favoris.add(code)
                    if st.session_state.get("logged_in"):
                        add_favori(st.session_state.user_id, code)

        st.divider()
