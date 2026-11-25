"""About / cite — mirrors the public /about page."""

from __future__ import annotations

import streamlit as st

from t2pw.app.ui_chrome import page_chrome, render_funders

page_chrome("About")

st.header("About T2PW")
st.markdown(
    """
T2PW reads scientific papers and drafts a pathway in PWML, the file format PathWhiz uses.
Every result is a draft for a person to review.
"""
)

st.header("How to cite")
st.markdown(
    "[Placeholder for the product owner: citation text, DOI and year, added on publication.]"
)

st.header("Credits and contact")
st.markdown(
    "[Placeholder for the product owner: laboratory name, funding line, contact address and licence.]"
)

render_funders()
