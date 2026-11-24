"""Field Guide theme — CSS injected into the Streamlit app.

Design direction #4 ("The Field Guide"): editorial paper-and-ink, Fraunces
display italic for the focal sign, Inter for UI, hairline rules, a deep-green
accent (#1E5A45) and ochre secondary (#A8731E). Accessibility is first-class,
not bolted on: a high-contrast mode and an adjustable base font size both
rewrite the same CSS variables the whole UI reads from.
"""

from __future__ import annotations

import streamlit as st

# Palette (light "paper" theme) and its high-contrast counterpart.
_PALETTE_LIGHT = {
    "paper": "#F7F5F0",
    "ink": "#181611",
    "green": "#1E5A45",
    "ochre": "#A8731E",
    "hairline": "#D9D3C4",
    "soft": "#6E6857",
    "card": "#FFFFFF",
}
_PALETTE_HC = {
    "paper": "#FFFFFF",
    "ink": "#000000",
    "green": "#005C3A",
    "ochre": "#8A4B00",
    "hairline": "#000000",
    "soft": "#1A1A1A",
    "card": "#FFFFFF",
}

_FONT_IMPORT = (
    "@import url('https://fonts.googleapis.com/css2?"
    "family=Fraunces:ital,opsz,wght@0,9..144,300..600;1,9..144,300..600"
    "&family=Inter:wght@400;500;600&display=swap');"
)


def inject_theme(high_contrast: bool = False, base_font_px: int = 18) -> None:
    """Inject the Field Guide CSS into the current Streamlit page.

    Args:
        high_contrast: Switch to the maximum-contrast palette and heavier
            rules for low-vision users.
        base_font_px: Root font size driving the whole type scale.
    """
    p = _PALETTE_HC if high_contrast else _PALETTE_LIGHT
    rule_w = "2px" if high_contrast else "1px"

    css = f"""
    <style>
    {_FONT_IMPORT}
    :root {{
        --paper:{p['paper']}; --ink:{p['ink']}; --green:{p['green']};
        --ochre:{p['ochre']}; --hairline:{p['hairline']}; --soft:{p['soft']};
        --card:{p['card']}; --rule:{rule_w}; --base:{base_font_px}px;
    }}
    .stApp {{ background:var(--paper); color:var(--ink);
        font-family:'Inter',sans-serif; font-size:var(--base); }}
    .main .block-container, [data-testid="stMainBlockContainer"] {{
        max-width:1180px; padding-top:3.6rem; }}

    /* Masthead */
    .fg-masthead {{ display:flex; align-items:baseline; justify-content:space-between;
        gap:1rem; flex-wrap:wrap; border-bottom:2px solid var(--ink);
        padding-bottom:.7rem; margin-bottom:.4rem; }}
    .fg-title {{ font-family:'Fraunces',serif; font-weight:500;
        font-size:calc(var(--base) * 1.7); letter-spacing:-.01em; }}
    .fg-title i {{ color:var(--green); font-style:normal; }}
    .fg-tagline {{ font-family:'Fraunces',serif; font-style:italic;
        color:var(--soft); font-size:calc(var(--base) * .82); }}

    /* Small-caps labels */
    .fg-label {{ font-size:calc(var(--base) * .62); font-weight:600;
        letter-spacing:.2em; text-transform:uppercase; color:var(--soft);
        margin-bottom:.5rem; }}

    /* Focal detected sign */
    .fg-sign {{ font-family:'Fraunces',serif; font-style:italic; font-weight:340;
        font-size:calc(var(--base) * 4.6); line-height:.92; letter-spacing:-.02em;
        color:var(--ink); margin:.1rem 0; }}
    .fg-sign.locked::after {{ content:"."; color:var(--green); }}
    .fg-sign.idle {{ color:var(--hairline); font-size:calc(var(--base) * 2.4); }}
    .fg-sign.unsure {{ color:var(--soft); font-size:calc(var(--base) * 2.4); }}
    .fg-howto {{ font-size:calc(var(--base) * .78); color:var(--green); }}
    .fg-howto a {{ color:var(--green); }}
    .fg-hint {{ font-size:calc(var(--base) * .78); color:var(--soft); }}
    .sa-svg {{ display:block; width:100%; }}
    .sa-head, .sa-body {{ stroke:var(--ink); }}
    .sa-hand {{ stroke:var(--green); }}
    .sa-lips {{ stroke:var(--ochre); }}
    .sa-box {{ border:1px solid var(--hairline); padding:.4rem .4rem .1rem; }}
    .fg-ai {{ font-family:Fraunces,serif; font-size:calc(var(--base) * 1.35);
              color:var(--ink); line-height:1.25; margin:.15rem 0 .2rem; }}

    /* Word in progress + transcript */
    .fg-building {{ font-size:calc(var(--base) * 1.4); font-weight:500;
        letter-spacing:.3em; min-height:1.6em; }}
    .fg-building .pending {{ color:var(--green); }}
    .fg-entry {{ font-family:'Fraunces',serif; font-size:calc(var(--base) * 1.05);
        padding:.6rem 0; border-bottom:1px solid var(--hairline); display:flex;
        gap:1rem; align-items:baseline; }}
    .fg-entry .idx {{ font-family:'Fraunces',serif; font-style:italic;
        color:var(--ochre); min-width:1.6rem; }}
    .fg-entry .time {{ color:var(--soft); font-size:.7em; font-family:'Inter'; }}
    .fg-entry .who {{ font-family:'Inter'; font-size:.62em; font-weight:600;
        letter-spacing:.14em; text-transform:uppercase; color:var(--green);
        min-width:4.6rem; }}
    .fg-entry.speaker .who {{ color:var(--ochre); }}

    /* Ranked candidate signs (top-3 live predictions) */
    .fg-cand {{ display:flex; align-items:center; gap:.6rem; margin:.25rem 0;
        font-size:calc(var(--base) * .8); }}
    .fg-cand .name {{ min-width:6.5rem; font-family:'Fraunces',serif; }}
    .fg-cand .bar {{ position:relative; flex:1; height:6px; background:var(--hairline); }}
    .fg-cand .bar .fill {{ position:absolute; left:0; top:0; bottom:0;
        background:var(--green); }}
    .fg-cand .bar .tick {{ position:absolute; top:-3px; bottom:-3px; width:var(--rule);
        background:var(--ochre); }}
    .fg-cand .pct {{ min-width:2.6rem; text-align:right; color:var(--soft); }}

    /* Confidence gauge (thin editorial line + dot) */
    .fg-gauge {{ position:relative; height:2px; background:var(--hairline);
        margin:.7rem 0 .5rem; }}
    .fg-gauge .fill {{ position:absolute; left:0; top:0; bottom:0;
        background:var(--green); }}
    .fg-gauge .dot {{ position:absolute; top:-5px; width:12px; height:12px;
        border-radius:50%; background:var(--paper); border:2px solid var(--green);
        transform:translateX(-50%); }}
    .fg-gauge-cap {{ font-size:calc(var(--base) * .78); color:var(--soft); }}
    .fg-gauge-cap em {{ font-family:'Fraunces',serif; font-style:italic;
        color:var(--ink); }}

    /* Figure caption under camera/player */
    .fg-figcap {{ display:flex; justify-content:space-between; gap:.8rem;
        flex-wrap:wrap; font-family:'Fraunces',serif; font-style:italic;
        color:var(--soft); font-size:calc(var(--base) * .8); margin-top:.5rem; }}
    .fg-figcap .live {{ color:var(--green); }}

    /* Status footer line */
    .fg-status {{ border-top:2px solid var(--ink); margin-top:1.4rem;
        padding-top:.6rem; color:var(--soft); font-size:calc(var(--base) * .78);
        display:flex; gap:.6rem; flex-wrap:wrap; }}
    .fg-status .on {{ color:var(--green); }}
    .fg-status .sep {{ color:var(--hairline); }}

    /* Buttons → ink slab that greens on hover */
    .stButton > button {{ border-radius:0; border:var(--rule) solid var(--ink);
        background:var(--ink); color:var(--paper); font-weight:600;
        letter-spacing:.08em; text-transform:uppercase; white-space:nowrap;
        font-size:calc(var(--base) * .72); padding:.6rem .7rem; transition:.15s; }}
    .stButton > button p {{ white-space:nowrap; }}
    .stButton > button:hover {{ background:var(--green); border-color:var(--green);
        color:var(--paper); }}
    .stButton > button:focus-visible {{ outline:3px solid var(--ochre);
        outline-offset:2px; }}

    /* Tabs as ruled nav */
    .stTabs [data-baseweb="tab-list"] {{ gap:1.8rem; border-bottom:1px solid var(--hairline); }}
    .stTabs [data-baseweb="tab"] {{ font-family:'Inter'; font-weight:500;
        color:var(--soft); padding:.4rem 0; }}
    .stTabs [aria-selected="true"] {{ color:var(--ink) !important;
        box-shadow:inset 0 -2px 0 var(--green); }}

    /* Inputs */
    .stTextInput input, .stTextArea textarea {{ border-radius:0;
        border:1px solid var(--ink); background:var(--card); color:var(--ink);
        font-family:'Inter'; }}
    [data-testid="stImage"] img {{ border:1px solid var(--ink); }}

    @media (prefers-reduced-motion: reduce) {{
        * {{ animation:none !important; transition:none !important; }}
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


def masthead(language: str) -> None:
    """Render the editorial masthead with the active language noted."""
    st.markdown(
        f"""
        <div class="fg-masthead">
          <div class="fg-title">Sign<i>Bridge</i></div>
          <div class="fg-tagline">A bidirectional interpreter for American
          &amp; Pakistani Sign Language · <b>{language.upper()}</b></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
