"""Fiche détaillée d'un site de baignade."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from data_loader import CLASSEMENT_COLORS, CLASSEMENT_LABELS, HIST_COLORS, HIST_LABELS
from components.meteo import show_meteo
from database.db import add_favori, remove_favori

TENDANCE_ICON  = {1: "↑", 0: "→", -1: "↓"}
TENDANCE_LABEL = {1: "Amélioration", 0: "Stable", -1: "Dégradation"}
TENDANCE_COLOR = {1: "#2ECC71", 0: "#888888", -1: "#E74C3C"}

EQUIPEMENTS = [
    ("parking",       "Parking"),
    ("sanitaires",    "Sanitaires"),
    ("pmr",           "Accès PMR"),
    ("douche",        "Douche"),
    ("poste_secours", "Poste de secours"),
]


def _badge_eq(label: str, present: bool) -> str:
    bg  = "#2ECC71" if present else "#888888"
    return (
        f'<span style="background:{bg};color:white;padding:2px 8px;'
        f'border-radius:12px;font-size:0.78em;font-weight:600;margin:2px">{label}</span>'
    )


def _historique_chart(historique_site: pd.DataFrame) -> go.Figure:
    """Graphique d'évolution du classement officiel sur 5 saisons."""
    saisons = list(range(2020, 2025))
    hist_map = dict(zip(historique_site["saison"], historique_site["classement_num"]))

    x, y, colors, texts = [], [], [], []
    for s in saisons:
        cl = hist_map.get(s)
        if cl is not None:
            x.append(s)
            y.append(int(cl))
            colors.append(HIST_COLORS.get(int(cl), "#888"))
            texts.append(HIST_LABELS.get(int(cl), ""))

    fig = go.Figure()

    # Bandes de fond semi-transparentes (lisibles en dark et light mode)
    for val, rgba in [
        (1, "rgba(46,204,113,0.15)"),
        (2, "rgba(243,156,18,0.15)"),
        (3, "rgba(230,126,34,0.15)"),
        (4, "rgba(231,76,60,0.15)"),
    ]:
        fig.add_shape(
            type="rect",
            x0=2019.5, x1=2024.5,
            y0=val - 0.45, y1=val + 0.45,
            fillcolor=rgba, line_width=0, layer="below",
        )

    if x:
        fig.add_trace(go.Scatter(
            x=x, y=y,
            mode="lines+markers",
            line=dict(color="#888888", width=2, dash="dot"),
            marker=dict(size=14, color=colors, line=dict(color="white", width=2)),
            text=texts,
            hovertemplate="<b>%{x}</b> : %{text}<extra></extra>",
        ))

    fig.update_yaxes(
        autorange="reversed",
        tickvals=[1, 2, 3, 4],
        ticktext=["Excellente", "Bonne", "Suffisante", "Non conforme"],
        range=[0.5, 4.5],
        gridcolor="rgba(128,128,128,0.2)",
    )
    fig.update_xaxes(tickvals=saisons, dtick=1, tickformat="d",
                     gridcolor="rgba(128,128,128,0.2)")
    fig.update_layout(
        height=200,
        margin=dict(l=0, r=0, t=4, b=0),
        showlegend=False,
        # Fond transparent → s'adapte au thème Streamlit
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def show_fiche_content(site: pd.Series, historique_site: pd.DataFrame) -> None:
    """Contenu de la fiche site (appelé depuis le dialogue modal)."""
    code = site["code_site"]
    favoris: set = st.session_state.setdefault("favoris", set())
    is_fav = code in favoris

    cl = int(site["classement_predit"])
    cl_color = CLASSEMENT_COLORS.get(cl, "#888")
    cl_label = CLASSEMENT_LABELS.get(cl, "Inconnu")

    tend = int(site.get("tendance", 0))
    tend_icon  = TENDANCE_ICON.get(tend, "→")
    tend_label = TENDANCE_LABEL.get(tend, "Stable")
    tend_color = TENDANCE_COLOR.get(tend, "#888")

    # ── En-tête : nom + bouton favori ────────────────────────────────
    col_info, col_fav = st.columns([10, 1])
    with col_info:
        st.markdown(f"### {site['nom_site']}")
        st.markdown(
            f"{site.get('commune', '')} · {site['type_eau']}  \n"
            f"<span style='background:{cl_color};color:white;padding:2px 10px;"
            f"border-radius:12px;font-size:0.85em;font-weight:600'>{cl_label}</span>"
            f"&nbsp;&nbsp;"
            f"<span style='color:{tend_color};font-size:1.2em;font-weight:700'>{tend_icon}</span>"
            f" <span style='color:{tend_color};font-size:0.85em'>{tend_label} (3 saisons)</span>",
            unsafe_allow_html=True,
        )
    with col_fav:
        fav_icon = "♥" if is_fav else "♡"
        # Le clic modifie les favoris ; Streamlit relance le dialogue automatiquement
        if st.button(fav_icon, key=f"fav_dialog_{code}", help="Ajouter/retirer des favoris"):
            if is_fav:
                st.session_state["favoris"].discard(code)
                if st.session_state.get("logged_in"):
                    remove_favori(st.session_state.user_id, code)
            else:
                st.session_state["favoris"].add(code)
                if st.session_state.get("logged_in"):
                    add_favori(st.session_state.user_id, code)

    # ── Équipements ──────────────────────────────────────────────────
    st.markdown("**Équipements**")
    badges_html = " ".join(
        _badge_eq(label, bool(site.get(col, 0)))
        for col, label in EQUIPEMENTS
    )
    st.markdown(badges_html, unsafe_allow_html=True)

    st.markdown("")

    # ── Historique des classements ───────────────────────────────────
    st.markdown("**Évolution du classement officiel 2020–2024**")
    if historique_site.empty:
        st.caption("Historique non disponible pour ce site.")
    else:
        st.plotly_chart(
            _historique_chart(historique_site),
            use_container_width=True,
            config={"displayModeBar": False},
        )

    # ── Météo J+3 ────────────────────────────────────────────────────
    st.markdown("**Prévisions météo J+3**")
    show_meteo(float(site["latitude"]), float(site["longitude"]))


@st.dialog("Fiche site", width="large")
def show_fiche_dialog(site: pd.Series, hist_site: pd.DataFrame) -> None:
    """Dialogue modal - s'ouvre via un appel direct depuis carte.py ou favoris.py."""
    show_fiche_content(site, hist_site)
