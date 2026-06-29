"""Onglet Favoris du dashboard AquaReco."""
from __future__ import annotations

import streamlit as st

from auth import require_login
from components.fiche_site import show_fiche_dialog
from database.db import remove_favori
from data_loader import (
    CLASSEMENT_COLORS,
    CLASSEMENT_LABELS,
    CLASSEMENT_ORDER,
    haversine_km,
    load_historique_officiel,
    load_sites,
)

DEFAULT_LAT = 46.603354
DEFAULT_LON = 1.888334


def _badge_cl(classement: int) -> str:
    label = CLASSEMENT_LABELS.get(classement, str(classement))
    color = CLASSEMENT_COLORS.get(classement, "#888")
    return (
        f'<span style="background:{color};color:white;padding:2px 8px;'
        f'border-radius:12px;font-size:0.78em;font-weight:600">{label}</span>'
    )


def show() -> None:
    """Affiche l'onglet Favoris."""
    if not require_login("favoris"):
        return

    sites_all = load_sites()
    historique_all = load_historique_officiel()
    favoris: set = st.session_state.setdefault("favoris", set())

    ref_lat = float(st.session_state.get("ref_lat", DEFAULT_LAT))
    ref_lon = float(st.session_state.get("ref_lon", DEFAULT_LON))

    if not favoris:
        st.info("Aucun favori ajouté. Cliquez sur ♡ dans l'onglet Carte pour ajouter des sites.")
        return

    fav_df = sites_all[sites_all["code_site"].isin(favoris)].copy()
    fav_df["distance_km"] = haversine_km(ref_lat, ref_lon, fav_df["latitude"], fav_df["longitude"])
    fav_df["_cl_order"] = fav_df["classement_predit"].map(CLASSEMENT_ORDER).fillna(99)
    fav_df = fav_df.sort_values(["_cl_order", "distance_km"]).reset_index(drop=True)

    n = len(fav_df)
    st.markdown(f"**{n} site{'s' if n != 1 else ''} en favori**")

    for _, row in fav_df.iterrows():
        code = row["code_site"]
        c1, c2, c3 = st.columns([4, 1, 1])
        with c1:
            st.markdown(
                f"**{row['nom_site']}**  \n"
                f"{row.get('commune', '')} · {row['type_eau']}  \n"
                f"{_badge_cl(int(row['classement_predit']))} &nbsp; "
                f"<span style='color:#888;font-size:0.85em'>{row['distance_km']:.0f} km</span>",
                unsafe_allow_html=True,
            )
        with c2:
            if st.button("📋", key=f"fav_fiche_{code}", help="Voir la fiche"):
                st.session_state["site_selectionne"] = code
                site_row = sites_all[sites_all["code_site"] == code].iloc[0]
                hist_site = historique_all[historique_all["code_site"] == code]
                show_fiche_dialog(site_row, hist_site)
        with c3:
            if st.button("💔", key=f"unfav_{code}", help="Retirer des favoris"):
                st.session_state["favoris"].discard(code)
                if st.session_state.get("logged_in"):
                    remove_favori(st.session_state.user_id, code)
                st.rerun()
        st.divider()
