"""Sidebar controls and shared session-state helpers.

Houses the global controls that apply across all three tabs: language
(ASL/PSL), accessibility settings (high contrast, base font size, always-on
captions), and audio behavior (auto-speak, TTS engine). Also centralizes
Streamlit ``session_state`` initialization so every tab reads consistent
keys.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import streamlit as st

from sign_bridge.config import AppConfig


def init_state(config: AppConfig) -> None:
    """Initialize session-state defaults once per session.

    Args:
        config: Loaded application config providing initial values.
    """
    defaults: dict[str, Any] = {
        "language": config.app.default_language,
        "high_contrast": config.ui.high_contrast_default,
        "base_font_px": config.ui.base_font_px,
        "captions_on": config.ui.captions_always_on,
        "auto_speak": False,
        "transcript": [],          # list[(time, text, "signer" | "speaker")]
        "running": False,          # camera loop active
        "tts_engine": config.tts.engine,
        "ai_enabled": config.ai.enabled and config.ai.provider != "none",
        "ai_last": None,           # latest AI sentence (see camera_view)
        "_ai_jobs": [],            # AI sentences still being written
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def sidebar(config: AppConfig) -> None:
    """Render the global sidebar; writes selections into session_state.

    Args:
        config: Loaded application config (for available options).
    """
    with st.sidebar:
        st.markdown('<div class="fg-label">Language</div>', unsafe_allow_html=True)
        lang = st.radio(
            "Sign language",
            options=["asl", "psl"],
            format_func=lambda x: "ASL — American" if x == "asl" else "PSL — Pakistan",
            index=0 if st.session_state["language"] == "asl" else 1,
            label_visibility="collapsed",
        )
        st.session_state["language"] = lang

        st.markdown('<div class="fg-label">Accessibility</div>', unsafe_allow_html=True)
        st.session_state["high_contrast"] = st.toggle(
            "High-contrast mode", value=st.session_state["high_contrast"],
            help="Maximum contrast palette and heavier rules for low vision.",
        )
        st.session_state["base_font_px"] = st.slider(
            "Text size", min_value=14, max_value=28,
            value=st.session_state["base_font_px"], step=1,
            help="Scales the entire interface, including the detected sign.",
        )
        st.session_state["captions_on"] = st.toggle(
            "Captions always on", value=st.session_state["captions_on"],
            help="Keep captions visible during sign-video playback.",
        )

        st.markdown('<div class="fg-label">Audio</div>', unsafe_allow_html=True)
        st.session_state["auto_speak"] = st.toggle(
            "Auto-speak sentences", value=st.session_state["auto_speak"],
            help="Speak each sentence aloud as soon as it is finalized.",
        )
        st.session_state["tts_engine"] = st.selectbox(
            "Voice engine", options=["gtts", "pyttsx3"],
            index=0 if st.session_state["tts_engine"] == "gtts" else 1,
            format_func=lambda e: "Google TTS (online)" if e == "gtts"
            else "Offline (server speakers)",
        )

        _ai_settings(config)

        st.markdown(
            '<div class="fg-tagline" style="margin-top:1rem">'
            "Prototype — not a certified interpreter. See README §Ethics."
            "</div>",
            unsafe_allow_html=True,
        )


_KEY_HELP = {
    "gemini": ("Google Gemini", "https://aistudio.google.com/apikey", "free tier available"),
    "anthropic": ("Claude", "https://console.anthropic.com/", "paid, very cheap per sentence"),
    "openai": ("OpenAI", "https://platform.openai.com/api-keys", "paid, very cheap per sentence"),
}


def _save_key(provider: str, secrets_path: str) -> None:
    from sign_bridge.language.sentence_ai import save_api_key

    key = str(st.session_state.get("_ai_key_input") or "").strip()
    if key:
        save_api_key(provider, key, Path(secrets_path))
        st.session_state["_ai_key_input"] = ""
        st.session_state["_ai_key_saved"] = True
        st.session_state.pop("_ai_writer", None)


def _ai_settings(config: AppConfig) -> None:
    """Sidebar block: AI sentence writer on/off and API key."""
    from sign_bridge.language.sentence_ai import find_api_key

    provider = config.ai.provider
    st.markdown('<div class="fg-label">AI sentences</div>', unsafe_allow_html=True)
    if provider == "none":
        st.caption("Turned off in config.yaml (ai.provider: none).")
        st.session_state["ai_enabled"] = False
        return
    st.session_state["ai_enabled"] = st.toggle(
        "Write sentences with AI", value=st.session_state["ai_enabled"],
        help="Turns the signed words into natural English, e.g. MOM DAD NO DRINK → "
             "\"Mom and Dad don't drink.\" Only the sign words are sent to the AI.")
    name, url, cost = _KEY_HELP[provider]
    secrets_path = config.resolve(Path(config.ai.secrets_file))
    if find_api_key(provider, secrets_path):
        st.caption(f"{name} key: saved ✓")
        if st.session_state.pop("_ai_key_saved", False):
            st.success("Key saved. Finish a sentence to try it.")
        return
    st.caption(f"Needs a {name} API key ({cost}). Get one at [{url}]({url}), then paste it here.")
    st.text_input(f"{name} API key", type="password", key="_ai_key_input",
                  placeholder="paste your key", label_visibility="collapsed")
    st.button("Save key", key="_ai_key_save", on_click=_save_key,
              args=(provider, str(secrets_path)))
