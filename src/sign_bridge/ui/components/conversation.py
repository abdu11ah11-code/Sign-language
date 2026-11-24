"""Conversation — the combined two-way communication tab.

Stacks both directions for a face-to-face exchange between a Deaf signer and
a hearing speaker. Both sides write to one shared, speaker-tagged log at the
top: the signer's recognized sentences (live, from the camera loop) and the
hearing person's typed/spoken replies (rendered to signs below). The shared
log makes turn-taking legible to both parties — the accessibility point of a
"conversation" mode.
"""

from __future__ import annotations

import streamlit as st

from sign_bridge.config import AppConfig
from sign_bridge.ui.components import camera_view, readout, sign_player
from sign_bridge.ui.components.camera_view import LiveView


def render(config: AppConfig) -> LiveView | None:
    """Render the Conversation tab.

    Args:
        config: Loaded application config.

    Returns:
        The camera :class:`LiveView` (its transcript slot is the shared log),
        or None if no model is loaded.
    """
    log_slot = st.empty()
    log_slot.markdown(readout.transcript_html(st.session_state["transcript"],
                                              "Conversation log"), unsafe_allow_html=True)

    st.divider()
    st.markdown('<div class="fg-label">Signer → Speech</div>', unsafe_allow_html=True)
    view = camera_view.render(config, context="conv_cam", transcript_slot=log_slot,
                              transcript_title="Conversation log")

    st.markdown('<div class="fg-label" style="margin-top:1.4rem">Speech → Signer'
                '</div>', unsafe_allow_html=True)
    sign_player.render(config, context="conv_play")
    return view
