"""Landing page — default for ``t2pw_ui`` (mirrors public FastAPI home)."""

from __future__ import annotations

import streamlit as st

from t2pw.app.ui_chrome import (
    page_chrome,
    render_funders,
    render_landing_ctas,
    render_landing_hero,
)

page_chrome("Home")
render_landing_hero()
render_landing_ctas(
    submit_page="streamlit_app.py",
    tutorial_page="ui_pages/tutorial.py",
    examples_page="ui_pages/examples.py",
)

st.markdown(
    """
    <div class="t2pw-intro">
      <h2>About Text to Pathways</h2>
      <p>
        T2PW (Text to PathWhiz) reads scientific papers and drafts a metabolic or
        signalling pathway in PWML, the file format PathWhiz uses. Give a pathway
        description or PDF files; the pipeline returns a draft for a curator to review.
      </p>
    </div>
    """,
    unsafe_allow_html=True,
)

render_funders()
