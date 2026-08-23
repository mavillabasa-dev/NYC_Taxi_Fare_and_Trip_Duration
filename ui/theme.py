"""Visual theme for the dashboard, matched to the T-115 presentation deck.

The palette, typefaces and the taxi-checker motif are lifted from `ppt/index.html` so
that the live demo and the slides behind it read as one piece of work rather than two.
Values are copied deliberately rather than imported: `ui/` is an independent tree and
must not reach into `ppt/`, which is a build artefact of a different tool.

Streamlit is themed in two layers, and both are needed:

* `.streamlit/config.toml` sets the base palette, which is what Streamlit's own widgets
  (selectboxes, inputs, the sidebar) read. CSS alone cannot reach inside those.
* The stylesheet below covers everything config.toml has no vocabulary for: typography,
  the checker rules, metric cards, and the states of the alert boxes.
"""

import streamlit as st

# --- palette, from ppt/index.html -------------------------------------------------
INK = "#0B0A0C"  # page surface
PANEL = "#141216"  # raised card
PANEL_2 = "#1A171C"  # raised card, second level
RULE = "#332F36"  # hairline, deliberately recessive
GOLD = "#F0BE4A"  # primary accent
GOLD_DEEP = "#B98C2C"  # secondary mark
TAXI = "#FFC72C"  # NYC taxi yellow, highlights only
CREAM = "#F2ECE0"  # primary text
TEXT_2 = "#A9A294"  # secondary text
TEXT_3 = "#7C7568"  # muted text, captions

FONT_DISPLAY = "'Zodiak', 'Iowan Old Style', Georgia, serif"
FONT_SANS = "'Supreme', 'Helvetica Neue', Arial, sans-serif"
FONT_MONO = "'JetBrains Mono', Consolas, monospace"

# Zodiak and Supreme come from Fontshare, JetBrains Mono from Google. Both have real
# fallback stacks above, so an offline venue degrades to a serif/grotesk pair rather
# than to whatever the browser picks.
_FONT_LINKS = """
<link rel="preconnect" href="https://api.fontshare.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet"
      href="https://api.fontshare.com/v2/css?f[]=zodiak@400,700,800&f[]=supreme@400,500,700&display=swap">
<link rel="stylesheet"
      href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&display=swap">
"""

_CSS = f"""
<style>
  :root {{
    --ink: {INK}; --panel: {PANEL}; --panel-2: {PANEL_2}; --rule: {RULE};
    --gold: {GOLD}; --gold-deep: {GOLD_DEEP}; --taxi: {TAXI};
    --cream: {CREAM}; --text-2: {TEXT_2}; --text-3: {TEXT_3};
    --gold-wash: rgba(240,190,74,.10);
    --gold-edge: rgba(240,190,74,.26);
  }}

  .stApp {{ background: var(--ink); }}

  html, body, [class*="css"], .stMarkdown, .stText {{
    font-family: {FONT_SANS};
    color: var(--cream);
  }}

  /* Streamlit centres the block container with a wide top pad; reclaim it, the
     checker band below becomes the top edge instead. */
  .block-container {{ padding-top: 2.2rem; max-width: 1100px; }}

  h1, h2, h3 {{ font-family: {FONT_DISPLAY}; color: var(--cream); letter-spacing: -.01em; }}
  h1 {{ font-weight: 800; font-size: 2.5rem; line-height: 1.06; margin-bottom: .2rem; }}
  h2 {{ font-weight: 700; font-size: 1.5rem; }}
  h3 {{ font-weight: 700; font-size: 1.15rem; }}

  /* --- the taxi checker, the deck's recurring device ----------------------- */
  .checker {{
    height: 14px;
    background-image:
      linear-gradient(45deg, var(--gold) 25%, transparent 25%, transparent 75%, var(--gold) 75%),
      linear-gradient(45deg, var(--gold) 25%, transparent 25%, transparent 75%, var(--gold) 75%);
    background-size: 14px 14px;
    background-position: 0 0, 7px 7px;
    margin-bottom: 1.6rem;
  }}
  .checker-rule {{
    height: 8px;
    background-image:
      linear-gradient(45deg, var(--gold-deep) 25%, transparent 25%, transparent 75%, var(--gold-deep) 75%),
      linear-gradient(45deg, var(--gold-deep) 25%, transparent 25%, transparent 75%, var(--gold-deep) 75%);
    background-size: 8px 8px;
    background-position: 0 0, 4px 4px;
    opacity: .55;
    margin: .1rem 0 1.4rem;
  }}

  .eyebrow {{
    font-family: {FONT_MONO};
    font-size: 11px; font-weight: 600;
    letter-spacing: .18em; text-transform: uppercase;
    color: var(--text-3);
    margin: 0 0 .5rem;
  }}

  /* --- metrics: the two numbers the whole project exists to produce -------- */
  [data-testid="stMetric"] {{
    background: var(--panel);
    border: 1px solid var(--rule);
    border-top: 3px solid var(--gold);
    border-radius: 4px;
    padding: 18px 22px 14px;
  }}
  [data-testid="stMetricLabel"] p {{
    font-family: {FONT_MONO} !important;
    font-size: 10.5px !important;
    letter-spacing: .14em; text-transform: uppercase;
    color: var(--text-3) !important;
  }}
  [data-testid="stMetricValue"] {{
    font-family: {FONT_SANS};
    font-weight: 700;
    font-size: 2.6rem !important;
    color: var(--taxi) !important;
    font-variant-numeric: tabular-nums;
  }}

  /* --- inputs ------------------------------------------------------------- */
  [data-testid="stWidgetLabel"] p {{
    font-family: {FONT_MONO} !important;
    font-size: 10.5px !important;
    letter-spacing: .12em; text-transform: uppercase;
    color: var(--text-2) !important;
  }}
  [data-baseweb="select"] > div, .stTextInput input, .stNumberInput input {{
    background: var(--panel) !important;
    border-color: var(--rule) !important;
    color: var(--cream) !important;
    border-radius: 3px !important;
  }}
  /* A disabled field here means "computed for you", not "broken" — keep it legible
     rather than greyed to the point of looking like an error. */
  .stNumberInput input:disabled, [data-baseweb="select"][aria-disabled="true"] > div {{
    color: var(--text-2) !important;
    background: var(--panel-2) !important;
    -webkit-text-fill-color: {TEXT_2} !important;
  }}

  .stButton button, .stFormSubmitButton button {{
    background: var(--gold) !important;
    color: {INK} !important;
    border: none !important;
    border-radius: 3px !important;
    font-family: {FONT_MONO} !important;
    font-weight: 600;
    letter-spacing: .1em; text-transform: uppercase; font-size: 12px !important;
    padding: .55rem 1.6rem !important;
  }}
  .stButton button:hover, .stFormSubmitButton button:hover {{
    background: var(--taxi) !important;
    color: {INK} !important;
  }}

  [data-testid="stForm"] {{
    background: var(--panel);
    border: 1px solid var(--rule);
    border-radius: 4px;
    padding: 20px 22px 8px;
  }}

  /* --- status boxes ------------------------------------------------------- */
  [data-testid="stAlert"] {{
    border-radius: 3px;
    font-size: .92rem;
    border-left-width: 4px;
  }}

  .stCheckbox label p, [data-testid="stCaptionContainer"] p {{
    color: var(--text-2) !important;
    font-size: .88rem;
  }}

  hr {{ border-color: var(--rule); }}

  /* Streamlit's chrome adds nothing to a demo and draws the eye away from it. */
  #MainMenu, footer, [data-testid="stDecoration"] {{ visibility: hidden; }}
</style>
"""


def apply() -> None:
    """Injects the fonts and stylesheet. Call once, immediately after set_page_config."""
    st.markdown(_FONT_LINKS + _CSS, unsafe_allow_html=True)


def checker(small: bool = False) -> None:
    """Draws the taxi-checker band used as a rule throughout the deck."""
    st.markdown(
        f'<div class="{"checker-rule" if small else "checker"}"></div>',
        unsafe_allow_html=True,
    )


def eyebrow(text: str) -> None:
    """Small uppercase mono label, the deck's section marker."""
    st.markdown(f'<p class="eyebrow">{text}</p>', unsafe_allow_html=True)
