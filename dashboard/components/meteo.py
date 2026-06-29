"""Prévisions météo J+3 via l'API OpenMeteo (gratuite, sans clé)."""
from __future__ import annotations

import json
import urllib.parse
import urllib.request

import streamlit as st

_URL = "https://api.open-meteo.com/v1/forecast"


@st.cache_data(ttl=3 * 3600, show_spinner=False)
def get_forecast(lat: float, lon: float) -> dict | None:
    """Retourne les prévisions J+3 pour (lat, lon) ou None en cas d'échec.

    Clé du résultat : daily → {time, temperature_2m_max, temperature_2m_min,
    precipitation_sum, wind_speed_10m_max}.
    Les coordonnées sont arrondies à 2 décimales pour maximiser les hits de cache.
    """
    params = urllib.parse.urlencode({
        "latitude": round(lat, 2),
        "longitude": round(lon, 2),
        "daily": ",".join([
            "precipitation_sum",
            "temperature_2m_max",
            "temperature_2m_min",
            "wind_speed_10m_max",
        ]),
        "timezone": "Europe/Paris",
        "forecast_days": 3,
    })
    url = f"{_URL}?{params}"
    req = urllib.request.Request(url, headers={"User-Agent": "AquaReco-TER/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=6) as r:
            return json.loads(r.read())
    except Exception:
        return None


def show_meteo(lat: float, lon: float) -> None:
    """Affiche le bloc météo J+3 dans la fiche site."""
    data = get_forecast(lat, lon)
    if data is None:
        st.caption("Prévisions météo indisponibles.")
        return

    daily = data.get("daily", {})
    dates = daily.get("time", [])
    tmax  = daily.get("temperature_2m_max", [])
    tmin  = daily.get("temperature_2m_min", [])
    precip = daily.get("precipitation_sum", [])
    wind  = daily.get("wind_speed_10m_max", [])

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
            if p > 10:
                st.warning("Risque de dégradation de la qualité de l'eau", icon="⚠️")
