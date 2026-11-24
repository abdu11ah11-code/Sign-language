"""SignBridge Streamlit application entry point.

Run via the console script ``sign-bridge`` (see :mod:`sign_bridge.main`) or
directly with ``streamlit run src/sign_bridge/ui/app.py``. Wires the theme,
global sidebar, and the three tabs together; all heavy ML work is lazy-loaded
inside the tabs so the page paints immediately. The app opens on a welcome screen
(the landing page, :mod:`sign_bridge.ui.landing`); **Start signing** goes to the tabs
and **Home page** in the sidebar comes back.

The live camera loop runs *last*, after every tab has been drawn, so all
tabs stay usable while the camera is on (see ``camera_view``).
"""

from __future__ import annotations

import streamlit as st

from sign_bridge.config import load_config
from sign_bridge.ui.components import camera_view, controls, conversation, sign_player
from sign_bridge.ui.theme import inject_theme, masthead

#: The welcome page fills the whole window: Streamlit's header, margins and column width are
#: switched off for this screen only, and the page's frame is as tall as the window.
HOME_CSS = """<style>
[data-testid="stHeader"], [data-testid="stToolbar"], [data-testid="stDecoration"] {
  display: none !important; }
[data-testid="stMainBlockContainer"], .block-container {
  padding: 0 !important; max-width: none !important; }
[data-testid="stMain"] [data-testid="stVerticalBlock"] { gap: 0 !important; }
[data-testid="stMain"] iframe {
  display: block; width: 100% !important; height: 100vh !important; border: 0; }
[data-testid="stAppViewContainer"], [data-testid="stMain"] { overflow: hidden !important; }
.st-key-start_signing { position: fixed !important; left: -10000px; top: 0; }
</style>"""


def home() -> None:
    """The welcome screen: the landing page, full window. Its Start signing buttons press
    the hidden Streamlit button below, which switches to the signing tabs."""
    import streamlit.components.v1 as components

    from sign_bridge.ui.landing import landing_page

    st.markdown(HOME_CSS, unsafe_allow_html=True)
    if st.button("Start signing", key="start_signing"):
        st.session_state["view"] = "app"
        st.rerun()
    components.html(landing_page("app"), height=900, scrolling=True)


def main() -> None:
    """Compose and render the full application."""
    config = load_config()
    st.set_page_config(
        page_title=f"{config.app.name} — Sign Translator",
        page_icon="👐",
        layout="wide",
    )

    controls.init_state(config)
    inject_theme(
        high_contrast=st.session_state["high_contrast"],
        base_font_px=st.session_state["base_font_px"],
    )
    st.session_state.setdefault("view", "home")
    if st.session_state["view"] == "home":
        home()
        return
    controls.sidebar(config)
    if st.sidebar.button("Home page", key="go_home", width="stretch"):
        st.session_state["view"] = "home"
        st.rerun()
    masthead(st.session_state["language"])

    tab_sign, tab_speech, tab_conv = st.tabs(
        ["Sign → Speech", "Speech → Sign", "Conversation"]
    )
    views: list[camera_view.LiveView] = []
    with tab_sign:
        view = camera_view.render(config, context="tab_sign")
        if view is not None:
            views.append(view)
    with tab_speech:
        sign_player.render(config, context="tab_speech")
    with tab_conv:
        conv_view = conversation.render(config)
        if conv_view is not None:
            views.append(conv_view)

    camera_view.run_live_loop(config, views)


if __name__ == "__main__":
    main()
