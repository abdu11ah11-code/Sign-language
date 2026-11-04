"""Conversation — the combined two-way communication tab.

Stacks both directions for a face-to-face exchange between a Deaf signer and
a hearing speaker: the signer's recognized speech accrues to a shared log
(top), and the hearing person's typed/spoken replies are rendered to signs
(bottom). The shared log makes the turn-taking legible to both parties,
which is the accessibility point of a "conversation" mode.
"""

from __future__ import annotations

import html

import streamlit as st

from sign_bridge.config import AppConfig
from sign_bridge.ui.components import camera_view, sign_player


def render(config: AppConfig) -> None:
    """Render the Conversation tab.

    Args:
        config: Loaded application config.
    """
    st.markdown('<div class="fg-label">Conversation log</div>', unsafe_allow_html=True)
    log = st.session_state["transcript"]
    if not log:
        st.markdown('<div class="fg-tagline">The shared transcript appears here as '
                    'each side communicates.</div>', unsafe_allow_html=True)
    else:
        for i, (ts, sentence) in enumerate(log, start=1):
            st.markdown(
                f'<div class="fg-entry"><span class="idx">{i:02d}</span>'
                f'<span style="flex:1">{html.escape(sentence)}</span>'
                f'<span class="time">{html.escape(ts)}</span></div>',
                unsafe_allow_html=True)

    st.divider()
    top, bottom = st.container(), st.container()

    with top:
        st.markdown('<div class="fg-label">Signer → Speech</div>', unsafe_allow_html=True)
        camera_view.render(config, context="conv_cam")

    with bottom:
        st.markdown('<div class="fg-label" style="margin-top:1rem">Speech → Signer'
                    '</div>', unsafe_allow_html=True)
        sign_player.render(config, context="conv_play")
