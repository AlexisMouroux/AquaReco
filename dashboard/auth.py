"""Formulaires d'authentification AquaReco."""
from __future__ import annotations

import re

import streamlit as st
import streamlit.components.v1 as components

from database.db import authenticate_user, create_user, get_favoris

PROFILS = [
    "Famille",
    "Sportif",
    "Senior",
    "Aventurier",
    "Vacancier côtier",
    "Touriste étranger",
]


def _valid_email(email: str) -> bool:
    return bool(re.match(r"[^@\s]+@[^@\s]+\.[^@\s]+", email.strip()))


# Critères de robustesse du mot de passe (ordre d'affichage de la checklist).
_PASSWORD_CRITERIA: list[tuple[str, callable]] = [
    ("8 caractères minimum", lambda pw: len(pw) >= 8),
    ("Une majuscule",        lambda pw: any(c.isupper() for c in pw)),
    ("Une minuscule",        lambda pw: any(c.islower() for c in pw)),
    ("Un chiffre",           lambda pw: any(c.isdigit() for c in pw)),
    ("Un caractère spécial", lambda pw: bool(re.search(r"[^A-Za-z0-9]", pw))),
]


def _password_valid(password: str) -> bool:
    return all(check(password) for _, check in _PASSWORD_CRITERIA)


def _password_checklist_html(password: str) -> str:
    items = []
    for label, check in _PASSWORD_CRITERIA:
        ok = check(password)
        icon, color = ("✓", "#2ECC71") if ok else ("✗", "#E74C3C")
        items.append(
            f"<li style='color:{color}'><span style='font-weight:700'>{icon}</span> {label}</li>"
        )
    return (
        "<ul style='margin:2px 0 14px;padding-left:4px;list-style:none;"
        "font-size:0.85em;line-height:1.6'>" + "".join(items) + "</ul>"
    )


# st.text_input ne committe sa valeur côté Python qu'au blur/Entrée - pour que
# la checklist se rafraîchisse pendant la frappe (avec un léger debounce), on
# simule un appui sur Entrée 1s après la dernière frappe, ce qui déclenche le
# commit + rerun normal de Streamlit sans bricoler de canal de communication
# séparé (pattern "hidden input mis à jour par JS", ici appliqué au champ
# réel lui-même via un événement clavier synthétique).
_PASSWORD_DEBOUNCE_JS = """
<script>
(function() {
    function attach() {
        const doc = window.parent.document;
        const inputs = doc.querySelectorAll('input[aria-label="Mot de passe"]');
        const target = inputs[inputs.length - 1];
        if (!target || target._aquarecoDebounce) return;
        target._aquarecoDebounce = true;
        let timer;
        target.addEventListener('input', function() {
            clearTimeout(timer);
            timer = setTimeout(function() {
                ['keydown', 'keypress', 'keyup'].forEach(function(type) {
                    target.dispatchEvent(new KeyboardEvent(type, {
                        bubbles: true, cancelable: true,
                        key: 'Enter', code: 'Enter', keyCode: 13, which: 13,
                    }));
                });
            }, 1000);
        });
    }
    attach();
    setTimeout(attach, 300);
})();
</script>
"""


def _set_session(user: dict) -> None:
    st.session_state.logged_in = True
    st.session_state.user_id = user["id"]
    st.session_state.user_email = user["email"]
    st.session_state.user_profil = user["profil"]
    st.session_state.favoris = set(get_favoris(user["id"]))
    st.session_state["_auth_dialog_open"] = False


def show() -> None:
    """Affiche le formulaire de connexion ou d'inscription selon l'état."""
    st.session_state.setdefault("auth_mode", "login")

    _, col, _ = st.columns([1, 2, 1])
    with col:
        st.markdown(
            "<h1 style='text-align:center;color:#1a6b8a'>💧 AquaReco</h1>"
            "<p style='text-align:center;margin-bottom:1.5rem'>"
            "Recommandation de sites de baignade</p>",
            unsafe_allow_html=True,
        )

        if st.session_state.auth_mode == "login":
            _login_form()
        else:
            _register_form()


def _login_form() -> None:
    st.markdown("### Connexion")
    with st.form("form_login", clear_on_submit=False):
        email = st.text_input("Email", placeholder="votre@email.fr")
        password = st.text_input("Mot de passe", type="password")
        submitted = st.form_submit_button(
            "Se connecter", type="primary", use_container_width=True
        )

    if submitted:
        if not email or not password:
            st.error("Renseignez votre email et mot de passe.")
        else:
            user = authenticate_user(email, password)
            if user:
                _set_session(user)
                st.rerun()
            else:
                st.error("Email ou mot de passe incorrect")

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button(
        "Pas encore de compte ? **Créer un compte**",
        use_container_width=True,
        key="to_register",
    ):
        open_auth_dialog("register")


def _mark_password_touched() -> None:
    st.session_state["password_touched"] = True


def _register_form() -> None:
    # Pas de st.form ici : le mot de passe doit rester hors formulaire pour que
    # la checklist se mette à jour à la validation du champ (un st.form ne
    # réexécute le script qu'à la soumission, donc figerait la checklist).
    st.markdown("### Créer mon compte")
    email = st.text_input("Email", placeholder="votre@email.fr", key="register_email")
    password = st.text_input(
        "Mot de passe", type="password", key="register_password",
        on_change=_mark_password_touched,
    )
    components.html(_PASSWORD_DEBOUNCE_JS, height=0)
    # La checklist n'apparaît qu'après une première saisie validée (clic dans le
    # champ puis ailleurs, ou 1s sans frappe), et reste ensuite affichée
    # jusqu'à la fin.
    if st.session_state.get("password_touched", False):
        st.markdown(_password_checklist_html(password), unsafe_allow_html=True)
    confirm = st.text_input("Confirmer le mot de passe", type="password", key="register_confirm")
    profil = st.selectbox("Mon profil", PROFILS, key="register_profil")
    submitted = st.button(
        "Créer mon compte", type="primary", use_container_width=True, key="register_submit"
    )

    if submitted:
        if not _valid_email(email):
            st.error("Adresse email invalide.")
        elif not _password_valid(password):
            st.error("Le mot de passe ne respecte pas tous les critères ci-dessus.")
        elif password != confirm:
            st.error("Les mots de passe ne correspondent pas.")
        else:
            user_id = create_user(email, password, profil)
            if user_id is None:
                st.error("Cette adresse email est déjà utilisée.")
            else:
                _set_session({"id": user_id, "email": email.lower().strip(), "profil": profil})
                st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button(
        "Déjà un compte ? **Se connecter**",
        use_container_width=True,
        key="to_login",
    ):
        open_auth_dialog("login")


# ── Modale réutilisable (auth à la demande) ────────────────────────────────────
#
# st.dialog ne reste "ouvert" d'un rerun à l'autre que si le script rappelle
# la fonction décorée à chaque exécution - un st.rerun() déclenché par un widget
# A L'INTERIEUR de la modale (ex: le lien "Créer un compte") referme sinon la
# modale, puisque rien ne la rouvre sur ce nouveau run. On rend donc l'ouverture
# persistante via un flag de session_state, vérifié sans condition à chaque run
# (voir render_auth_dialog, appelé depuis app.py).

@st.dialog("Mon compte AquaReco", width="large")
def show_auth_dialog() -> None:
    """Contenu de la modale de connexion / inscription. Réutilise show()."""
    show()


def open_auth_dialog(mode: str = "login") -> None:
    """Programme l'ouverture de la modale au prochain rerun et le force.

    A appeler depuis un callback de bouton (header, garde d'onglet, bouton
    favori). Le rerun est nécessaire : sans lui, le flag ne serait pris en
    compte qu'au prochain clic de l'utilisateur ailleurs dans l'app.
    """
    st.session_state.auth_mode = mode
    st.session_state["_auth_dialog_open"] = True
    st.rerun()


def render_auth_dialog() -> None:
    """À appeler sans condition à un point fixe du script (app.py).

    Le flag est consommé (remis à False) dès cet appel : pour que la modale
    reste affichée au prochain rerun (bascule login/register), le code qui
    déclenche ce rerun doit explicitement rappeler open_auth_dialog(). Ainsi,
    une fermeture native (bouton "X") n'est jamais ré-ouverte par un rerun
    sans rapport survenant ailleurs dans l'app par la suite.
    """
    if st.session_state.get("_auth_dialog_open", False):
        st.session_state["_auth_dialog_open"] = False
        show_auth_dialog()


def require_login(key_prefix: str,
                  message: str = "Cette fonctionnalité nécessite d'être connecté.") -> bool:
    """Garde d'accès pour les onglets nécessitant une connexion.

    Retourne True si l'utilisateur est connecté (l'appelant affiche alors son
    contenu habituel). Sinon, affiche un message d'invitation centré avec deux
    boutons ouvrant la modale d'auth, et retourne False (l'appelant doit
    s'arrêter immédiatement).
    """
    if st.session_state.get("logged_in", False):
        return True

    _, col, _ = st.columns([1, 2, 1])
    with col:
        st.markdown(
            f"<div style='text-align:center;padding:48px 16px'>"
            f"<p style='font-size:2.2em;margin:0 0 8px'>🔒</p>"
            f"<p style='font-size:1.05em;color:#888;margin:0 0 24px'>{message}</p>"
            f"</div>",
            unsafe_allow_html=True,
        )
        b1, b2 = st.columns(2)
        with b1:
            if st.button("Se connecter", use_container_width=True,
                        type="primary", key=f"{key_prefix}_gate_login"):
                open_auth_dialog("login")
        with b2:
            if st.button("Créer un compte", use_container_width=True,
                        key=f"{key_prefix}_gate_register"):
                open_auth_dialog("register")
    return False
