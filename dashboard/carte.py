"""Onglet Carte du dashboard AquaReco."""
from __future__ import annotations

import folium
import pandas as pd
import streamlit as st
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium
from streamlit_js_eval import get_geolocation

from auth import open_auth_dialog
from components.fiche_site import show_fiche_dialog
from database.db import add_favori, get_preferences, remove_favori, save_location_pref
from data_loader import (
    CLASSEMENT_COLORS,
    CLASSEMENT_LABELS,
    CLASSEMENT_ORDER,
    geocode_ville,
    haversine_km,
    load_historique_officiel,
    load_sites,
)

# Niveau 3 - fallback localisation par défaut (voir _resolve_default_location)
PARIS_LAT = 48.8566
PARIS_LON = 2.3522
PARIS_NOM = "Paris"

CLASSEMENT_FILTER_OPTIONS = [
    "Tous les sites",
    "Bonne qualité minimum",
    "Excellente qualité uniquement",
]

# Couleur du cluster = pire classement parmi les marqueurs regroupés (pas la
# taille du cluster, comportement par défaut de Leaflet.markercluster qui
# colore en jaune/rouge selon le NOMBRE de marqueurs, sans rapport avec leur
# couleur réelle - d'où des clusters jaunes ne contenant que des sites verts).
# Couleur = celle majoritaire parmi les marqueurs regroupés (pas la pire -
# un seul rouge perdu parmi 20 verts ne doit pas teindre tout le cluster en
# rouge). En cas d'égalité stricte de comptage, on départage en faveur de la
# couleur la plus positive (ex: 10 vert / 10 rouge -> vert).
_CLUSTER_ICON_JS = f"""
function(cluster) {{
    var children = cluster.getAllChildMarkers();
    var rank = {{'{CLASSEMENT_COLORS[3]}': 2, '{CLASSEMENT_COLORS[1]}': 1, '{CLASSEMENT_COLORS[0]}': 0}};
    var counts = {{}};
    for (var i = 0; i < children.length; i++) {{
        var c = children[i].options.fillColor || children[i].options.color;
        counts[c] = (counts[c] || 0) + 1;
    }}
    var majorityColor = '{CLASSEMENT_COLORS[0]}';
    var maxCount = -1;
    for (var color in counts) {{
        var cnt = counts[color];
        if (cnt > maxCount || (cnt === maxCount && (rank[color] || 0) < (rank[majorityColor] || 0))) {{
            maxCount = cnt;
            majorityColor = color;
        }}
    }}
    var count = cluster.getChildCount();
    return L.divIcon({{
        html: '<div style="background:' + majorityColor + ';width:36px;height:36px;' +
              'border-radius:50%;display:flex;align-items:center;justify-content:center;' +
              'color:white;font-weight:600;font-size:13px;border:2px solid white;' +
              'box-shadow:0 1px 4px rgba(0,0,0,.4)">' + count + '</div>',
        className: '',
        iconSize: [40, 40],
    }});
}}
"""


def _badge_cl(classement: int) -> str:
    label = CLASSEMENT_LABELS.get(classement, str(classement))
    color = CLASSEMENT_COLORS.get(classement, "#888")
    return (
        f'<span style="background:{color};color:white;padding:2px 8px;'
        f'border-radius:12px;font-size:0.78em;font-weight:600">{label}</span>'
    )


def _build_map(
    df: pd.DataFrame,
    ref_lat: float,
    ref_lon: float,
    selected_code: str | None,
) -> folium.Map:
    if df.empty:
        m = folium.Map(location=[ref_lat, ref_lon], zoom_start=9, tiles="CartoDB positron")
        return m

    m = folium.Map(
        location=[df["latitude"].mean(), df["longitude"].mean()],
        zoom_start=7,
        tiles="CartoDB positron",
    )
    cluster = MarkerCluster(icon_create_function=_CLUSTER_ICON_JS).add_to(m)

    for _, row in df.iterrows():
        code = row["code_site"]
        cl = int(row["classement_predit"])
        color = CLASSEMENT_COLORS.get(cl, "#888")
        label = CLASSEMENT_LABELS.get(cl, "Inconnu")
        is_sel = code == selected_code

        folium.CircleMarker(
            location=[row["latitude"], row["longitude"]],
            radius=10 if is_sel else 7,
            color="black" if is_sel else color,
            weight=3 if is_sel else 1,
            fill=True,
            fill_color=color,
            fill_opacity=0.9 if is_sel else 0.85,
            popup=folium.Popup(
                f"<b>{row['nom_site']}</b><br>"
                f"{row.get('commune','')} · {row['type_eau']}<br>"
                f"<span style='color:{color};font-weight:600'>{label}</span>",
                max_width=250,
            ),
            tooltip=row["nom_site"],
        ).add_to(cluster)

    m.fit_bounds(
        [[df["latitude"].min(), df["longitude"].min()],
         [df["latitude"].max(), df["longitude"].max()]]
    )

    m.get_root().html.add_child(folium.Element("""
    <div style="position:fixed;bottom:30px;left:30px;z-index:1000;
        background:white;padding:10px 14px;border-radius:8px;
        box-shadow:0 2px 8px rgba(0,0,0,.2);font-size:13px;line-height:1.8">
        <b>Qualité de l'eau</b><br>
        <span style="color:#2ECC71">&#9679;</span> Excellente<br>
        <span style="color:#F39C12">&#9679;</span> Bonne<br>
        <span style="color:#E74C3C">&#9679;</span> Non conforme
    </div>
    """))
    return m


def _resolve_default_location() -> tuple[float, float, str]:
    """Localisation par défaut au chargement, par ordre de priorité :
    1) géoloc navigateur acceptée  2) préférence SQLite (ville_pref) si connecté
    3) Paris en dernier recours."""
    geo = st.session_state.get("geoloc_coords")
    if geo:
        return geo["lat"], geo["lon"], "ma position"

    if st.session_state.get("logged_in", False):
        prefs = get_preferences(st.session_state.user_id)
        if prefs.get("ville_pref") and prefs.get("lat_pref") is not None and prefs.get("lon_pref") is not None:
            return float(prefs["lat_pref"]), float(prefs["lon_pref"]), prefs["ville_pref"]

    return PARIS_LAT, PARIS_LON, PARIS_NOM


def _handle_geolocation() -> None:
    """Gère le cycle de la demande de géolocalisation navigateur (Niveau 1).

    A appeler en tout début de show(). Tant que `geoloc_requested` est vrai et
    qu'aucun résultat n'est encore connu, on monte le composant JS ; une fois
    résolu (coordonnées ou refus), on arrête de l'appeler.
    """
    if not st.session_state.get("geoloc_requested", False):
        return
    if st.session_state.get("geoloc_coords") or st.session_state.get("geoloc_denied"):
        return

    result = get_geolocation()
    if result is None:
        return  # en attente de la réponse du navigateur (permission GPS)

    if "coords" in result:
        st.session_state["geoloc_coords"] = {
            "lat": result["coords"]["latitude"],
            "lon": result["coords"]["longitude"],
        }
        st.session_state["geoloc_requested"] = False
    else:
        st.session_state["geoloc_denied"] = True
        st.session_state["geoloc_requested"] = False
        st.toast(
            "Géolocalisation refusée ou indisponible — dernière ville ou Paris utilisés.",
            icon="⚠️",
        )


def show() -> None:
    """Affiche l'onglet Carte."""
    sites_all = load_sites()
    historique_all = load_historique_officiel()
    type_eau_options = sorted(sites_all["type_eau"].dropna().unique().tolist())

    st.session_state.setdefault("favoris", set())
    st.session_state.setdefault("site_selectionne", None)

    _handle_geolocation()

    # ── Colonnes : gauche (filtres + liste) / droite (carte) ─────────
    col_left, col_right = st.columns([1, 1.6], gap="medium")

    # ── Widgets de filtrage (col gauche, 1ère passe) ──────────────────
    with col_left:
        st.markdown("### Recherche")
        if st.button("📍 Utiliser ma position", key="btn_geoloc", use_container_width=True):
            st.session_state["geoloc_requested"] = True
            st.session_state["geoloc_denied"] = False
            st.rerun()
        search = st.text_input(
            "Nom du site ou ville",
            placeholder="Ex : plage de… ou Annecy",
            key="carte_search",
            label_visibility="collapsed",
        )
        st.markdown("### Filtres")

        # Slider avec debounce : la carte n'utilise la nouvelle valeur qu'au
        # relâchement (on_change), pas pendant le glissement.
        st.session_state.setdefault("distance_max", 25)
        st.session_state.setdefault("last_distance_max", st.session_state["distance_max"])

        def _commit_distance() -> None:
            st.session_state["distance_max"] = st.session_state["slider_distance"]
            st.session_state["last_distance_max"] = st.session_state["slider_distance"]

        st.slider(
            "Distance maximale (km)",
            min_value=5, max_value=1500,
            value=st.session_state["distance_max"],
            step=5,
            key="slider_distance",
            on_change=_commit_distance,
        )
        distance_max = st.session_state["last_distance_max"]

        classement_filter = st.selectbox(
            "Classement minimum",
            CLASSEMENT_FILTER_OPTIONS,
            key="carte_classement",
        )
        types_selected = st.multiselect(
            "Type de plan d'eau",
            options=type_eau_options,
            default=type_eau_options,
            key="carte_types",
        )

    # ── Filtrage ─────────────────────────────────────────────────────
    df = sites_all.copy()
    search_q = search.strip()
    ref_lat, ref_lon, ref_label = _resolve_default_location()
    geo_mode = False
    geo_not_found = False

    if search_q:
        mask = df["nom_site"].str.contains(search_q, case=False, na=False)
        site_matches = df[mask]
        if not site_matches.empty:
            df = site_matches
            if len(df) == 1:
                ref_lat = float(df["latitude"].iloc[0])
                ref_lon = float(df["longitude"].iloc[0])
        else:
            coords = geocode_ville(search_q)
            if coords:
                ref_lat, ref_lon = coords
                geo_mode = True
                # Garde le chip "Localisation" de l'onglet Recommandation à jour
                # (n'intervient plus dans la résolution de la localisation par
                # défaut de Carte - seule la préférence SQLite ville_pref compte).
                st.session_state["last_location"] = {
                    "lat": ref_lat, "lon": ref_lon, "nom": search_q,
                }
                if st.session_state.get("logged_in", False):
                    save_location_pref(st.session_state.user_id, search_q, ref_lat, ref_lon)
            else:
                geo_not_found = True
                df = df.iloc[0:0]

    df = df.copy()
    df["distance_km"] = haversine_km(ref_lat, ref_lon, df["latitude"], df["longitude"])

    apply_distance = (not search_q) or geo_mode or (len(df) == 1)
    if apply_distance:
        df = df[df["distance_km"] <= distance_max]

    if classement_filter == "Bonne qualité minimum":
        df = df[df["classement_predit"].isin([0, 1])]
    elif classement_filter == "Excellente qualité uniquement":
        df = df[df["classement_predit"] == 0]

    if types_selected:
        df = df[df["type_eau"].isin(types_selected)]

    df = df.copy()
    df["_cl_order"] = df["classement_predit"].map(CLASSEMENT_ORDER).fillna(99)
    df = df.sort_values(["_cl_order", "distance_km"]).reset_index(drop=True)

    # Stocke le point de référence pour l'onglet Favoris
    st.session_state["ref_lat"] = ref_lat
    st.session_state["ref_lon"] = ref_lon

    selected_code = st.session_state.get("site_selectionne")
    if selected_code and selected_code not in df["code_site"].values:
        st.session_state["site_selectionne"] = None
        selected_code = None

    # Clé stable : change si les sites filtrés OU le point de référence changent.
    # Sans ref_lat/ref_lon, un changement de niveau de localisation (ex: DB pref
    # -> dernière ville) qui produirait le même ensemble de sites ne forcerait
    # pas st_folium à recréer la carte, et l'ancien zoom/pan resterait affiché
    # au lieu du nouveau fit_bounds.
    map_key = "folium_" + str(abs(hash((
        round(ref_lat, 4), round(ref_lon, 4),
        tuple(sorted(df["code_site"].tolist())),
    ))))

    # ── Liste de résultats (col gauche, 2ème passe) ───────────────────
    with col_left:
        if geo_not_found:
            st.warning(f"Ville « {search_q} » non trouvée.")
        n = len(df)
        if geo_mode and n > 0:
            st.markdown(
                f"**{n} site{'s' if n != 1 else ''} "
                f"dans un rayon de {distance_max} km autour de « {search_q} »**"
            )
        elif not search_q:
            st.markdown(
                f"**{n} site{'s' if n != 1 else ''} "
                f"dans un rayon de {distance_max} km autour de {ref_label}**"
            )
        else:
            st.markdown(f"**{n} site{'s' if n != 1 else ''} trouvé{'s' if n != 1 else ''}**")

        with st.container(height=420):
            for _, row in df.iterrows():
                code = row["code_site"]
                is_fav = code in st.session_state["favoris"]
                c1, c2, c3 = st.columns([4, 1, 1])
                with c1:
                    st.markdown(
                        f"**{row['nom_site']}**  \n"
                        f"{row.get('commune','')} · {row['type_eau']}  \n"
                        f"{_badge_cl(int(row['classement_predit']))} &nbsp; "
                        f"<span style='color:#888;font-size:0.85em'>"
                        f"{row['distance_km']:.0f} km</span>",
                        unsafe_allow_html=True,
                    )
                with c2:
                    btn_type = "primary" if code == selected_code else "secondary"
                    if st.button("📋", key=f"sel_{code}", help="Voir la fiche", type=btn_type):
                        st.session_state["site_selectionne"] = code
                        site_r = sites_all[sites_all["code_site"] == code].iloc[0]
                        hist_s = historique_all[historique_all["code_site"] == code]
                        show_fiche_dialog(site_r, hist_s)
                with c3:
                    heart = "♥" if is_fav else "♡"
                    if st.button(heart, key=f"fav_{code}", help="Ajouter/retirer des favoris"):
                        if not st.session_state.get("logged_in", False):
                            open_auth_dialog("login")
                        elif is_fav:
                            st.session_state["favoris"].discard(code)
                            remove_favori(st.session_state.user_id, code)
                        else:
                            st.session_state["favoris"].add(code)
                            add_favori(st.session_state.user_id, code)
                st.divider()

    # ── Carte Folium (col droite) ─────────────────────────────────────
    with col_right:
        folium_map = _build_map(df, ref_lat, ref_lon, selected_code)
        map_data = st_folium(
            folium_map,
            use_container_width=True,
            height=720,
            returned_objects=["last_object_clicked_tooltip"],
            key=map_key,
        )

        # Clic sur un marqueur → ouvre la fiche dialog
        if map_data:
            tip = map_data.get("last_object_clicked_tooltip")
            if tip:
                match = df[df["nom_site"] == tip]
                if not match.empty:
                    code = match.iloc[0]["code_site"]
                    if code != selected_code:
                        st.session_state["site_selectionne"] = code
                        site_r = sites_all[sites_all["code_site"] == code].iloc[0]
                        hist_s = historique_all[historique_all["code_site"] == code]
                        show_fiche_dialog(site_r, hist_s)
