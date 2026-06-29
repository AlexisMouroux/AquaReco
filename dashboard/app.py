"""Point d'entrée Streamlit du dashboard AquaReco."""
import sys
import pathlib

_HERE = pathlib.Path(__file__).parent
sys.path.insert(0, str(_HERE))           # dashboard/  → data_loader, carte, …
sys.path.insert(0, str(_HERE.parent))    # aquareco/   → database.db

import streamlit as st

import auth
from database.db import init_db

init_db()

st.set_page_config(
    page_title="AquaReco",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
      .block-container { padding-top: 1rem; }
      .stMarkdown span { display: inline; }
      div[data-testid="stModal"] > div[role="dialog"] {
          width: min(42vw, 700px) !important;
          max-width: min(42vw, 700px) !important;
      }
      /* Colonne email/profil du header : centree verticalement plutot qu'en haut */
      div[data-testid="stVerticalBlock"]:has(p[style*="text-align"]) {
          justify-content: center !important;
      }
      /* Colonne bouton Se connecter/Deconnexion du header : meme espacement vertical */
      div[data-testid="stVerticalBlock"]:has([style*="margin-top: 10px"]) {
          gap: 2rem !important;
          justify-content: center !important;
      }

      /* ── Palette bleue (remplace le rouge par defaut Streamlit) ──────────
         Cible les attributs stables (data-testid / role / data-baseweb) plutot
         que les classes "theme" de config.toml, pour ne pas desactiver le
         selecteur clair/sombre de Streamlit (menu ⋮ > Settings). */

      /* Boutons primaires (Se connecter, Creer un compte, Obtenir mes
         recommandations, Sauvegarder, onglet profil actif) */
      button[data-testid^="stBaseButton-primary"] {
          background-color: #1a6b8a !important;
          border-color: #1a6b8a !important;
      }
      button[data-testid^="stBaseButton-primary"]:hover {
          background-color: #5ba3c9 !important;
          border-color: #5ba3c9 !important;
          color: white !important;
      }

      /* Onglet de navigation actif (texte + barre de soulignement) */
      button[data-testid="stTab"][aria-selected="true"] {
          color: #1a6b8a !important;
      }
      [data-baseweb="tab-highlight"] {
          background-color: #1a6b8a !important;
      }

      /* Curseur des sliders (ex: distance max, rayon de recherche) */
      div[role="slider"] {
          background-color: #1a6b8a !important;
      }
      [data-testid="stSliderThumbValue"] {
          color: #1a6b8a !important;
          border-color: #1a6b8a !important;
      }
      /* Portion "remplie" du rail du slider (gradient rouge par defaut du theme) */
      [data-testid="stSlider"] [data-baseweb="slider"] > div:first-child > div:first-child > div:nth-child(2) {
          background: rgba(172, 177, 195, 0.25) !important;
      }

      /* Chips des multiselect (ex: type de plan d'eau) - badges secondaires */
      span[data-baseweb="tag"] {
          background-color: #5ba3c9 !important;
      }

      /* Liens cliquables dans le contenu principal */
      [data-testid="stMain"] a {
          color: #1a6b8a !important;
      }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── En-tête (connecté ou non - l'auth n'est plus une porte bloquante) ────────
header_l, header_m, header_r = st.columns([4, 2, 1])
with header_l:
    st.markdown(
        "<h1 style='margin:0;color:#1a6b8a'>💧 AquaReco</h1>"
        "<p style='margin:0;font-size:0.9em'>Recommandation de sites de baignade</p>",
        unsafe_allow_html=True,
    )
with header_m:
    if st.session_state.get("logged_in", False):
        st.markdown(
            f"<p style='text-align:right;font-size:0.82em;margin-top:14px'>"
            f"<b>{st.session_state.user_email}</b><br>"
            f"<span style='color:#888'>{st.session_state.user_profil}</span></p>",
            unsafe_allow_html=True,
        )
with header_r:
    st.markdown("<div style='margin-top:10px'>", unsafe_allow_html=True)
    if st.session_state.get("logged_in", False):
        if st.button("Déconnexion", use_container_width=True):
            for key in [
                "logged_in", "user_id", "user_email", "user_profil",
                "favoris", "auth_mode", "site_selectionne",
            ]:
                st.session_state.pop(key, None)
            st.rerun()
    else:
        if st.button("Se connecter", use_container_width=True, type="primary"):
            auth.open_auth_dialog("login")
    st.markdown("</div>", unsafe_allow_html=True)

# Ré-ouvre la modale d'auth tant qu'elle est marquée ouverte (cf. auth.py)
auth.render_auth_dialog()

st.divider()

# ── Navigation ────────────────────────────────────────────────────────────────
tabs = st.tabs(["🗺️ Carte", "❤️ Favoris", "🔔 Alertes", "🎯 Recommandation"])

with tabs[0]:
    import carte
    carte.show()

with tabs[1]:
    import favoris
    favoris.show()

with tabs[2]:
    import alertes
    alertes.show()

with tabs[3]:
    import recommandation
    recommandation.show()
