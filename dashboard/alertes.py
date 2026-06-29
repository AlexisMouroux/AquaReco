"""Onglet Alertes - météo J+3 pour les sites favoris."""
from __future__ import annotations

import streamlit as st

from auth import require_login
from components.meteo import get_forecast
from data_loader import CLASSEMENT_COLORS, CLASSEMENT_LABELS, load_sites

WEATHER_ICONS = {
    "précipitations": "🌧️",
    "température": "🌡️",
    "vent": "💨",
}


def _badge_cl(classement: int) -> str:
    label = CLASSEMENT_LABELS.get(classement, str(classement))
    color = CLASSEMENT_COLORS.get(classement, "#888")
    return (
        f'<span style="background:{color};color:white;padding:2px 8px;'
        f'border-radius:12px;font-size:0.78em;font-weight:600">{label}</span>'
    )


def _max_precip(data: dict) -> float:
    """Précipitations maximales sur les 3 prochains jours."""
    precip = data.get("daily", {}).get("precipitation_sum", [])
    return max((p or 0.0 for p in precip), default=0.0)


def show() -> None:
    """Affiche l'onglet Alertes."""
    if not require_login("alertes"):
        return

    favoris: set = st.session_state.setdefault("favoris", set())

    if not favoris:
        st.info(
            "Aucun site en favori. Ajoutez des sites depuis l'onglet Carte "
            "pour recevoir des alertes météo."
        )
        return

    sites_all = load_sites()
    fav_df = sites_all[sites_all["code_site"].isin(favoris)].reset_index(drop=True)

    st.markdown(
        f"**{len(fav_df)} site{'s' if len(fav_df) != 1 else ''} surveillé{'s' if len(fav_df) != 1 else ''}**"
    )

    alerts_found = False

    for _, row in fav_df.iterrows():
        data = get_forecast(float(row["latitude"]), float(row["longitude"]))

        has_alert = False
        if data:
            max_p = _max_precip(data)
            has_alert = max_p > 10

        if has_alert:
            alerts_found = True

        with st.expander(
            f"{'🔴 ' if has_alert else '🟢 '}{row['nom_site']}  —  "
            f"{row.get('commune', '')} · {row['type_eau']}",
            expanded=has_alert,
        ):
            st.markdown(
                f"{_badge_cl(int(row['classement_predit']))}",
                unsafe_allow_html=True,
            )
            st.markdown("")

            if data is None:
                st.caption("Prévisions météo indisponibles.")
                continue

            daily = data.get("daily", {})
            dates = daily.get("time", [])
            tmax = daily.get("temperature_2m_max", [])
            tmin = daily.get("temperature_2m_min", [])
            precip = daily.get("precipitation_sum", [])
            wind = daily.get("wind_speed_10m_max", [])

            cols = st.columns(3)
            for i in range(min(3, len(dates))):
                with cols[i]:
                    p = precip[i] if i < len(precip) else 0.0
                    p = p or 0.0
                    st.markdown(
                        f"**{dates[i]}**  \n"
                        f"🌡️ {tmin[i]:.0f}° / {tmax[i]:.0f}°C  \n"
                        f"🌧️ {p:.1f} mm  \n"
                        f"💨 {wind[i]:.0f} km/h"
                    )

            if has_alert:
                st.warning(
                    f"Risque de dégradation de la qualité de l'eau "
                    f"(précipitations max : {_max_precip(data):.1f} mm sur 3 jours)",
                    icon="⚠️",
                )

    if not alerts_found:
        st.success("Aucune alerte météo sur vos sites favoris dans les 3 prochains jours.", icon="✅")
