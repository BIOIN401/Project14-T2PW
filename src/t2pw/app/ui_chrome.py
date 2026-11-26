"""Shared Streamlit chrome mirroring the public FastAPI site (brand, top nav, funders).

No pipeline / biology logic.
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

_PB_MAGENTA = "#9d4485"
_PB_MAGENTA_DEEP = "#7a3368"
_PB_MAGENTA_DARKER = "#5c2650"
_PB_CREAM = "#fff9e5"
_PB_NAVY = "#001c55"
_PB_ORANGE = "#ff8c00"
_PB_ORANGE_DARK = "#cc7000"
_PB_GREEN = "#76c043"
_PB_GREEN_DARK = "#5a9433"
_PB_SKY = "#5dade2"

_FUNDER_DIR = Path(__file__).resolve().parents[1] / "web" / "static" / "img" / "funders"

_BRAND_CSS = f"""
<style>
  :root {{
    --pb-magenta: {_PB_MAGENTA};
    --pb-magenta-deep: {_PB_MAGENTA_DEEP};
    --pb-cream: {_PB_CREAM};
    --pb-navy: {_PB_NAVY};
    --pb-orange: {_PB_ORANGE};
    --pb-green: {_PB_GREEN};
    --pb-sky: {_PB_SKY};
    --t2pw-band-pad-x: clamp(1.1rem, 3vw, 1.75rem);
  }}
  [data-testid="stSidebar"],
  [data-testid="stSidebarCollapsedControl"] {{
    display: none !important;
  }}
  [data-testid="stHeader"] {{
    background: var(--pb-magenta) !important;
  }}
  [data-testid="stHeader"] * {{
    color: #ffffff !important;
  }}

  /* Drop Streamlit top padding so magenta nav meets the app header with no white strip */
  .stApp [data-testid="stAppViewContainer"] .main .block-container {{
    padding-top: 0 !important;
    max-width: 100% !important;
  }}

  /* True viewport full-bleed (same width for nav + hero) */
  .t2pw-topnav,
  .t2pw-hero,
  .t2pw-hero-cta-band {{
    width: 100vw !important;
    max-width: 100vw !important;
    margin-left: calc(50% - 50vw) !important;
    margin-right: calc(50% - 50vw) !important;
    box-sizing: border-box;
  }}
  div[data-testid="element-container"]:has(.t2pw-topnav) {{
    margin-bottom: 0 !important;
  }}

  /* Magenta top nav with desktop row + mobile hamburger */
  .t2pw-topnav {{
    position: relative;
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    justify-content: space-between;
    gap: 0.5rem 1rem;
    margin-top: 0 !important;
    margin-bottom: 0 !important;
    padding: 0.7rem var(--t2pw-band-pad-x);
    background: {_PB_MAGENTA};
    border-bottom: 1px solid {_PB_MAGENTA_DEEP};
    color: #ffffff;
  }}
  .t2pw-topnav__brand {{
    display: flex;
    align-items: baseline;
    gap: 0.55rem;
    margin: 0;
    color: #ffffff !important;
    text-decoration: none !important;
    font-size: 1.25rem;
    font-weight: 700;
    letter-spacing: 0.02em;
    white-space: nowrap;
  }}
  .t2pw-topnav__brand span {{
    font-size: 0.92rem;
    font-weight: 500;
    color: rgba(255, 255, 255, 0.88) !important;
  }}
  .t2pw-nav-cb {{
    position: absolute;
    opacity: 0;
    pointer-events: none;
  }}
  .t2pw-nav-burger {{
    display: none;
    flex-direction: column;
    justify-content: center;
    gap: 5px;
    width: 2.5rem;
    height: 2.5rem;
    margin: 0;
    padding: 0.45rem;
    border: 1px solid rgba(255, 255, 255, 0.35);
    border-radius: 0.4rem;
    background: transparent;
    cursor: pointer;
  }}
  .t2pw-nav-burger span {{
    display: block;
    height: 2px;
    width: 100%;
    background: #ffffff;
    border-radius: 1px;
  }}
  .t2pw-topnav__menu {{
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 0.15rem;
    margin: 0;
    padding: 0;
    list-style: none;
  }}
  .t2pw-topnav__menu a {{
    display: block;
    padding: 0.45rem 0.8rem;
    border-radius: 0.4rem;
    color: #ffffff !important;
    font-weight: 500;
    text-decoration: none !important;
  }}
  .t2pw-topnav__menu a:hover {{
    background: {_PB_MAGENTA_DEEP} !important;
    color: #ffffff !important;
  }}
  @media (max-width: 48rem) {{
    .t2pw-nav-burger {{
      display: inline-flex;
      margin-left: auto;
    }}
    .t2pw-topnav__menu {{
      display: none;
      flex-basis: 100%;
      flex-direction: column;
      align-items: stretch;
      gap: 0.2rem;
      padding: 0.35rem 0 0.5rem;
      border-top: 1px solid rgba(255, 255, 255, 0.22);
    }}
    .t2pw-nav-cb:checked ~ .t2pw-topnav__menu {{
      display: flex;
    }}
    .t2pw-nav-cb:checked + .t2pw-nav-burger {{
      background: rgba(255, 255, 255, 0.14);
    }}
  }}

  .t2pw-hero {{
    position: relative;
    overflow: hidden;
    margin-top: 0 !important;
    margin-bottom: 0 !important;
    padding: 2.75rem var(--t2pw-band-pad-x) 1rem;
    color: #f4f8fb;
    background:
      radial-gradient(ellipse 70% 55% at 75% 35%, rgba(157, 68, 133, 0.35), transparent 55%),
      radial-gradient(ellipse 50% 40% at 20% 70%, rgba(118, 192, 67, 0.12), transparent 50%),
      linear-gradient(165deg, {_PB_NAVY} 0%, #0a2f6e 48%, #001433 100%);
  }}
  .t2pw-hero::after {{
    content: "";
    position: absolute;
    inset: 0;
    background: repeating-linear-gradient(
      -18deg,
      transparent,
      transparent 28px,
      rgba(160, 207, 255, 0.04) 28px,
      rgba(160, 207, 255, 0.04) 29px
    );
    pointer-events: none;
  }}
  .t2pw-hero__inner {{
    position: relative;
    z-index: 1;
    max-width: 42rem;
  }}
  .t2pw-hero .eyebrow {{
    margin: 0 0 0.5rem 0;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    font-size: 0.78rem;
    color: rgba(232, 238, 247, 0.85);
  }}
  .t2pw-hero h1 {{
    margin: 0 0 0.35rem 0;
    font-size: clamp(2.8rem, 8vw, 4.2rem);
    font-style: italic;
    font-weight: 800;
    letter-spacing: 0.02em;
    color: #f0f6ff;
    line-height: 1;
  }}
  .t2pw-hero .expand {{
    margin: 0 0 1rem 0;
    font-size: 1.15rem;
    color: rgba(240, 246, 255, 0.92);
  }}
  .t2pw-hero .expand .t {{ color: {_PB_SKY}; font-weight: 700; }}
  .t2pw-hero .expand .p {{ color: {_PB_ORANGE}; font-weight: 700; }}
  .t2pw-hero .expand .w {{ color: {_PB_GREEN}; font-weight: 700; }}
  .t2pw-hero p.lead {{
    margin: 0;
    max-width: 34rem;
    font-size: 1.05rem;
    line-height: 1.5;
    color: #e8eef7;
  }}

  .t2pw-hero-cta-band {{
    background:
      radial-gradient(ellipse 70% 55% at 75% 35%, rgba(157, 68, 133, 0.35), transparent 55%),
      linear-gradient(165deg, {_PB_NAVY} 0%, #0a2f6e 48%, #001433 100%);
    margin-top: 0 !important;
    margin-bottom: 1.5rem !important;
    padding: 0.75rem var(--t2pw-band-pad-x) 2.5rem;
  }}
  /* Collapse gaps between hero markdown blocks and the CTA band */
  div[data-testid="element-container"]:has(.t2pw-hero),
  div[data-testid="element-container"]:has(.t2pw-hero-cta-band) {{
    margin: 0 !important;
    padding: 0 !important;
  }}
  .t2pw-hero-ctas {{
    display: flex;
    flex-direction: column;
    align-items: flex-end; /* right-aligned; widths set per button */
    gap: 0.75rem;
    width: 100%;
    margin-left: auto;
  }}
  a.t2pw-hero-cta {{
    display: flex;
    align-items: center;
    justify-content: flex-start;
    box-sizing: border-box;
    min-height: 2.9rem;
    padding: 0.75rem 1.2rem;
    border-radius: 0.45rem;
    font-family: inherit;
    font-weight: 600;
    font-size: 1.02rem;
    line-height: 1.2;
    text-decoration: none !important;
    color: #ffffff !important;
    box-shadow: 0 2px 10px rgba(0, 0, 0, 0.25);
    transition: background-color 0.15s ease, transform 0.15s ease;
  }}
  a.t2pw-hero-cta:hover {{
    transform: translateY(-1px);
    color: #ffffff !important;
    text-decoration: none !important;
    filter: none !important;
  }}
  a.t2pw-hero-cta--submit {{ background: {_PB_MAGENTA} !important; }}
  a.t2pw-hero-cta--submit:hover {{ background: {_PB_MAGENTA_DEEP} !important; }}
  a.t2pw-hero-cta--tutorial {{ background: {_PB_ORANGE} !important; }}
  a.t2pw-hero-cta--tutorial:hover {{ background: {_PB_ORANGE_DARK} !important; }}
  a.t2pw-hero-cta--examples {{ background: {_PB_GREEN} !important; }}
  a.t2pw-hero-cta--examples:hover {{ background: {_PB_GREEN_DARK} !important; }}

  /* Home only: collapse gap between nav and hero. */
  div[data-testid="stVerticalBlock"]:has(.t2pw-hero):has(.t2pw-topnav) {{
    gap: 0 !important;
  }}
  div[data-testid="stVerticalBlock"]:has(.t2pw-hero) > div:has(.t2pw-intro),
  div[data-testid="stVerticalBlock"]:has(.t2pw-hero) > div:has(.t2pw-funders),
  div[data-testid="stVerticalBlock"]:has(.t2pw-hero) > div:has(.t2pw-funders-logos) {{
    margin-top: 1.25rem !important;
  }}

  .t2pw-intro {{
    background: #eef2f7;
    border: 1px solid #c5d0dc;
    border-radius: 0.5rem;
    padding: 1.25rem 1.4rem;
    margin-bottom: 1.25rem;
    color: #1a2332;
  }}
  .t2pw-intro h2 {{
    margin-top: 0;
    color: {_PB_NAVY};
  }}
  .t2pw-funders {{
    margin-top: 2rem;
    padding-top: 1.25rem;
    border-top: 1px solid #c5d0dc;
    color: #444;
    font-size: 0.9rem;
  }}
  .t2pw-funders a {{ color: {_PB_MAGENTA}; }}
  .t2pw-funders-logos {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(min(9.5rem, 100%), 1fr));
    gap: clamp(0.75rem, 2vw, 1.5rem);
    align-items: center;
    justify-items: center;
    margin-top: 1.25rem;
  }}
  .t2pw-funders-logos a {{
    display: flex;
    align-items: center;
    justify-content: center;
    width: 100%;
    max-width: 11rem;
    padding: 0.35rem;
  }}
  .t2pw-funders-logos img {{
    width: 100%;
    height: auto;
    max-height: clamp(2.75rem, 8vw, 4.5rem);
    object-fit: contain;
  }}
</style>
"""


def inject_brand_css() -> None:
    st.markdown(_BRAND_CSS, unsafe_allow_html=True)


def render_top_nav(
    *,
    home_page: str,
    submit_page: str,
    tutorial_page: str,
    about_page: str,
    examples_page: str,
) -> None:
    """Magenta top bar: brand + links; hamburger collapses links under 48rem."""
    # Absolute url_paths so brand/links resolve from any page (not relative to /submit).
    def _href(page: str) -> str:
        mapping = {
            home_page: "/",  # default landing
            submit_page: "/submit",
            tutorial_page: "/tutorial",
            about_page: "/about",
            examples_page: "/examples",
        }
        return mapping.get(page, "/")

    st.html(
        f"""
        <header class="t2pw-topnav" role="banner">
          <a class="t2pw-topnav__brand" href="/" title="Home">
            T2PW<span>Text to PathWhiz</span>
          </a>
          <input type="checkbox" id="t2pw-nav-cb" class="t2pw-nav-cb" aria-hidden="true">
          <label for="t2pw-nav-cb" class="t2pw-nav-burger" aria-label="Open menu">
            <span></span><span></span><span></span>
          </label>
          <nav class="t2pw-topnav__menu" aria-label="Site">
            <a href="{_href(home_page)}">Home</a>
            <a href="{_href(submit_page)}">Submit</a>
            <a href="{_href(tutorial_page)}">Tutorial</a>
            <a href="{_href(about_page)}">About</a>
            <a href="{_href(examples_page)}">Examples</a>
          </nav>
        </header>
        """
    )


def render_landing_hero() -> None:
    st.markdown(
        """
        <section class="t2pw-hero" aria-label="T2PW">
          <div class="t2pw-hero__inner">
            <p class="eyebrow">Text to PathWhiz</p>
            <h1>T2PW</h1>
            <p class="expand">
              <span class="t">T</span>ext to
              <span class="p">P</span>ath
              <span class="w">W</span>hiz
            </p>
            <p class="lead">
              Turn pathway descriptions from the literature into draft PathWhiz PWML —
              ready for a person to review.
            </p>
          </div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def render_landing_ctas(
    *,
    submit_page: str,
    tutorial_page: str,
    examples_page: str,
    about_page: str | None = None,
) -> None:
    """Stacked magenta / orange / green CTAs, right-aligned; stepped widths."""
    del about_page
    _ = (submit_page, tutorial_page, examples_page)
    # Inline widths so Streamlit theme CSS cannot force equal 100% button widths.
    st.html(
        """
        <div class="t2pw-hero-cta-band">
          <nav class="t2pw-hero-ctas" aria-label="Get started">
            <a class="t2pw-hero-cta t2pw-hero-cta--submit" href="submit"
               style="width:20rem;max-width:100%;flex:0 0 auto">Submit a pathway →</a>
            <a class="t2pw-hero-cta t2pw-hero-cta--tutorial" href="tutorial"
               style="width:15.5rem;max-width:92%;flex:0 0 auto">Read the tutorial →</a>
            <a class="t2pw-hero-cta t2pw-hero-cta--examples" href="examples"
               style="width:12rem;max-width:78%;flex:0 0 auto">Browse examples →</a>
          </nav>
        </div>
        """
    )


def render_funders() -> None:
    st.markdown(
        """
        <div class="t2pw-funders">
          <p>
            This project is supported by the
            <a href="https://cihr-irsc.gc.ca/" rel="noopener noreferrer" target="_blank">Canadian Institutes of Health Research</a>,
            <a href="https://www.innovation.ca/" rel="noopener noreferrer" target="_blank">Canada Foundation for Innovation</a>,
            and by
            <a href="https://www.metabolomicscentre.ca/" rel="noopener noreferrer" target="_blank">The Metabolomics Innovation Centre (TMIC)</a>,
            a nationally-funded research and core facility that supports a wide range of cutting-edge metabolomic studies.
            TMIC is funded by
            <a href="https://www.genomecanada.ca/" rel="noopener noreferrer" target="_blank">Genome Canada</a>,
            <a href="https://www.genomealberta.ca/" rel="noopener noreferrer" target="_blank">Genome Alberta</a>,
            and
            <a href="https://www.genomebc.ca/" rel="noopener noreferrer" target="_blank">Genome British Columbia</a>.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    logos = [
        ("tmic.png", "The Metabolomics Innovation Centre (TMIC)", "https://www.metabolomicscentre.ca/"),
        ("cfi.png", "Canada Foundation for Innovation", "https://www.innovation.ca/"),
        ("genome-canada.png", "Genome Canada", "https://www.genomecanada.ca/"),
        ("genome-alberta.png", "Genome Alberta", "https://www.genomealberta.ca/"),
        ("genome-bc.png", "Genome British Columbia", "https://www.genomebc.ca/"),
        ("cihr.png", "Canadian Institutes of Health Research (CIHR IRSC)", "https://cihr-irsc.gc.ca/"),
    ]
    # file:// URLs do not work in the browser; Streamlit media serves local paths via st.image.
    # Use a responsive HTML grid fed by Streamlit-hosted media through base64 for reliability.
    import base64

    cells: list[str] = []
    for name, alt, href in logos:
        path = _FUNDER_DIR / name
        if not path.is_file():
            continue
        b64 = base64.b64encode(path.read_bytes()).decode("ascii")
        cells.append(
            f'<a href="{href}" rel="noopener noreferrer" target="_blank">'
            f'<img src="data:image/png;base64,{b64}" alt="{alt}" loading="lazy"></a>'
        )
    if cells:
        st.markdown(
            '<div class="t2pw-funders-logos">' + "".join(cells) + "</div>",
            unsafe_allow_html=True,
        )


def page_chrome(_page_label: str = "") -> None:
    """Pages under t2pw_ui: CSS only (nav is drawn by the entrypoint)."""
    inject_brand_css()


def inject_submit_page_css() -> None:
    """Submit-page polish: sections, pathway cards, primary run button (no biology change)."""
    st.markdown(
        f"""
        <style>
          .t2pw-submit-lead {{
            max-width: 42rem;
            margin: 0 0 1.25rem 0;
            color: #3a4553;
            font-size: 1.02rem;
            line-height: 1.45;
          }}
          .t2pw-section-label {{
            display: block;
            margin: 1.5rem 0 0 !important;
            padding: 0 0 0.45rem 0 !important;
            border: none !important;
            border-bottom: 2px solid {_PB_MAGENTA} !important;
            color: {_PB_NAVY};
            font-size: 1.05rem;
            font-weight: 700;
            letter-spacing: 0.01em;
            line-height: 1.35;
          }}
          .t2pw-section-hint {{
            display: block;
            margin: 0.65rem 0 0.85rem 0 !important;
            padding: 0 !important;
            border: none !important;
            color: #5a6573;
            font-size: 0.9rem;
            line-height: 1.45;
            text-decoration: none !important;
          }}
          /* Space below the rule so the hint never sits on the bar. */
          div[data-testid="element-container"]:has(.t2pw-section-label) {{
            margin-bottom: 0.55rem !important;
          }}
          div[data-testid="element-container"]:has(.t2pw-section-hint) {{
            margin-top: 0.15rem !important;
            margin-bottom: 0.5rem !important;
          }}
          [data-testid="stCaption"],
          [data-testid="stCaption"] *,
          [data-testid="stWidgetLabel"],
          [data-testid="stWidgetLabel"] * {{
            text-decoration: none !important;
          }}
          div[data-testid="stRadio"] > label[data-testid="stWidgetLabel"],
          div[data-testid="stFileUploader"] > label[data-testid="stWidgetLabel"] {{
            display: none !important;
          }}
          /* Right-align per-pathway remove; allow full label (no ellipsis) */
          div[class*="st-key-remove_pathway_entry"] {{
            display: flex;
            justify-content: flex-end;
          }}
          div[class*="st-key-remove_pathway_entry"] button {{
            white-space: nowrap !important;
            overflow: visible !important;
            text-overflow: clip !important;
            min-width: 11rem;
          }}
          .t2pw-pathway-card {{
            margin: 0.75rem 0 1rem 0;
            padding: 0.85rem 1rem 0.25rem;
            border: 1px solid #d5dde8;
            border-radius: 0.5rem;
            background: #f7f9fc;
          }}
          .t2pw-pathway-card__title {{
            margin: 0 0 0.35rem 0;
            color: {_PB_MAGENTA};
            font-weight: 700;
            font-size: 0.95rem;
          }}
          /* Primary form submit */
          div[data-testid="stForm"] button[kind="primaryFormSubmit"],
          div[data-testid="stForm"] button[type="submit"] {{
            background-color: {_PB_MAGENTA} !important;
            border-color: {_PB_MAGENTA_DEEP} !important;
            color: #ffffff !important;
            font-weight: 600 !important;
            min-height: 2.75rem;
            padding-left: 1.4rem !important;
            padding-right: 1.4rem !important;
          }}
          div[data-testid="stForm"] button[kind="primaryFormSubmit"]:hover,
          div[data-testid="stForm"] button[type="submit"]:hover {{
            background-color: {_PB_MAGENTA_DEEP} !important;
            border-color: {_PB_MAGENTA_DARKER} !important;
            color: #ffffff !important;
          }}
          div[class*="st-key-add_pathway_entry"] button {{
            background-color: {_PB_MAGENTA} !important;
            border-color: {_PB_MAGENTA_DEEP} !important;
            color: #ffffff !important;
            font-weight: 600 !important;
          }}
          div[class*="st-key-add_pathway_entry"] button:hover {{
            background-color: {_PB_MAGENTA_DEEP} !important;
            color: #ffffff !important;
          }}
          div[class*="st-key-remove_pathway_entry"] button {{
            background: transparent !important;
            border: 1px solid #c5d0dc !important;
            color: {_PB_MAGENTA_DEEP} !important;
          }}
          div[class*="st-key-remove_pathway_entry"] button:hover {{
            background: #eef2f7 !important;
            border-color: {_PB_MAGENTA} !important;
            color: {_PB_MAGENTA_DEEP} !important;
          }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_submit_header() -> None:
    """Title + lead for the Submit page."""
    st.title("Submit a pathway")
    st.markdown(
        '<p class="t2pw-submit-lead">'
        "Paste a pathway description or upload PDFs. T2PW runs the local pipeline and "
        "returns a draft PathWhiz PWML for review."
        "</p>",
        unsafe_allow_html=True,
    )
