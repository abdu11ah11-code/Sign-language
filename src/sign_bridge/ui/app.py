"""SignBridge Streamlit application entry point.

Run via the console script ``sign-bridge`` (see :mod:`sign_bridge.main`) or
directly with ``streamlit run src/sign_bridge/ui/app.py``. Wires the theme,
global sidebar, and the three tabs together; all heavy ML work is lazy-loaded
inside the tabs so the page paints immediately.

The live camera loop runs *last*, after every tab has been drawn, so all
tabs stay usable while the camera is on (see ``camera_view``).
"""

from __future__ import annotations

import streamlit as st

from sign_bridge.config import load_config
from sign_bridge.ui.components import camera_view, controls, conversation, sign_player
from sign_bridge.ui.theme import inject_theme, masthead


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
    controls.sidebar(config)
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
