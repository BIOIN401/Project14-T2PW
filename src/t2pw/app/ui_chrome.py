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

  /* True viewport full-bleed (same width for nav + hero; ignores block-container padding) */
  .t2pw-fullbleed,
  div[data-testid="stHorizontalBlock"]:has(.t2pw-nav-brand),
  .t2pw-hero,
  .t2pw-hero-cta-band {{
    width: 100vw !important;
    max-width: 100vw !important;
    margin-left: calc(50% - 50vw) !important;
    margin-right: calc(50% - 50vw) !important;
    box-sizing: border-box;
  }}

  /* Full-bleed magenta nav */
  div[data-testid="stHorizontalBlock"]:has(.t2pw-nav-brand),
  div[data-testid="stHorizontalBlock"]:has(.t2pw-nav-brand) > div[data-testid="stColumn"] {{
    background: {_PB_MAGENTA} !important;
  }}
  div[data-testid="stHorizontalBlock"]:has(.t2pw-nav-brand) {{
    margin-top: 0 !important;
    margin-bottom: 0 !important;
    padding: 0.65rem var(--t2pw-band-pad-x) !important;
    border-bottom: 1px solid {_PB_MAGENTA_DEEP};
    gap: 0.25rem !important;
  }}
  /* Collapse Streamlit vertical gaps under the nav so navy hero butts up against it */
  div[data-testid="element-container"]:has(.t2pw-nav-brand),
  div[data-testid="stHorizontalBlock"]:has(.t2pw-nav-brand) {{
    margin-bottom: 0 !important;
  }}
  div[data-testid="stVerticalBlockBorderWrapper"]:has(.t2pw-nav-brand),
  div[data-testid="stVerticalBlock"]:has(> div .t2pw-nav-brand) {{
    gap: 0 !important;
  }}
  div[data-testid="stHorizontalBlock"]:has(.t2pw-nav-brand) [data-testid="stPageLink"] a,
  div[data-testid="stHorizontalBlock"]:has(.t2pw-nav-brand) a {{
    color: #ffffff !important;
    font-weight: 500 !important;
    text-decoration: none !important;
    padding: 0.45rem 0.8rem !important;
    border-radius: 0.4rem !important;
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
  }}
  div[data-testid="stHorizontalBlock"]:has(.t2pw-nav-brand) [data-testid="stPageLink"] a:hover,
  div[data-testid="stHorizontalBlock"]:has(.t2pw-nav-brand) a:hover {{
    background: {_PB_MAGENTA_DEEP} !important;
    color: #ffffff !important;
  }}
  [data-testid="stPageLink"] a:hover,
  [data-testid="stPageLink"] a:focus {{
    background: {_PB_MAGENTA_DEEP} !important;
    color: #ffffff !important;
  }}
  button:hover,
  button[kind="secondary"]:hover,
  button[kind="primary"]:hover {{
    background-color: {_PB_MAGENTA_DEEP} !important;
    border-color: {_PB_MAGENTA_DARKER} !important;
    color: #ffffff !important;
  }}

  .t2pw-nav-brand {{
    margin: 0;
    padding: 0.35rem 0.25rem;
    color: #ffffff !important;
    font-size: 1.25rem;
    font-weight: 700;
    letter-spacing: 0.02em;
    white-space: nowrap;
  }}
  .t2pw-nav-brand span {{
    margin-left: 0.55rem;
    font-size: 0.92rem;
    font-weight: 500;
    color: rgba(255, 255, 255, 0.88) !important;
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
    align-items: stretch;
    gap: 0.75rem;
    max-width: min(17.5rem, 100%);
    width: min(17.5rem, 100%);
    margin-left: auto;
  }}
  a.t2pw-hero-cta {{
    display: flex;
    align-items: center;
    justify-content: flex-start;
    box-sizing: border-box;
    width: 100%;
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

  /* Nav + hero share one column: kill Streamlit's default vertical gap so navy
     meets magenta with no white seam; keep spacing for later page content. */
  div[data-testid="stVerticalBlock"]:has(.t2pw-nav-brand) {{
    gap: 0 !important;
  }}
  div[data-testid="stVerticalBlock"]:has(.t2pw-hero) > div:has(.t2pw-intro),
  div[data-testid="stVerticalBlock"]:has(.t2pw-hero) > div:has(.t2pw-funders),
  div[data-testid="stVerticalBlock"]:has(.t2pw-hero) > div:has(.t2pw-funders-logos) {{
    margin-top: 1.25rem !important;
  }}

  .t2pw-intro {{
    background: {_PB_CREAM};
    border: 1px solid #e0c48a;
    border-radius: 0.5rem;
    padding: 1.25rem 1.4rem;
    margin-bottom: 1.25rem;
  }}
  .t2pw-funders {{
    margin-top: 2rem;
    padding-top: 1.25rem;
    border-top: 1px solid #e0c48a;
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
    """Magenta top bar: brand left, links right — matches public site chrome."""
    brand, home, submit, tutorial, about, examples = st.columns(
        [3.4, 1, 1, 1.1, 1, 1.1]
    )
    with brand:
        st.markdown(
            '<p class="t2pw-nav-brand">T2PW<span>Text to PathWhiz</span></p>',
            unsafe_allow_html=True,
        )
    with home:
        st.page_link(home_page, label="Home")
    with submit:
        st.page_link(submit_page, label="Submit")
    with tutorial:
        st.page_link(tutorial_page, label="Tutorial")
    with about:
        st.page_link(about_page, label="About")
    with examples:
        st.page_link(examples_page, label="Examples")


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
    """Stacked magenta / orange / green CTAs, right-aligned on the navy band."""
    del about_page
    _ = (submit_page, tutorial_page, examples_page)
    st.markdown(
        """
        <div class="t2pw-hero-cta-band">
          <nav class="t2pw-hero-ctas" aria-label="Get started">
            <a class="t2pw-hero-cta t2pw-hero-cta--submit" href="submit">Submit a pathway →</a>
            <a class="t2pw-hero-cta t2pw-hero-cta--tutorial" href="tutorial">Read the tutorial →</a>
            <a class="t2pw-hero-cta t2pw-hero-cta--examples" href="examples">Browse examples →</a>
          </nav>
        </div>
        """,
        unsafe_allow_html=True,
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
