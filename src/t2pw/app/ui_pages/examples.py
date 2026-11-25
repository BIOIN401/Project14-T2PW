"""Examples placeholder — mirrors the public /examples page."""

from __future__ import annotations

import streamlit as st

from t2pw.app.ui_chrome import page_chrome, render_funders

page_chrome("Examples")

st.header("Examples")
st.info(
    "[TODO for the product owner: 3 to 4 curated, verified examples will be listed here, "
    "each with its input and a finished result you can open.]"
)
st.markdown("Until then, open **Submit a pathway** and paste a description or upload a PDF.")
st.page_link("streamlit_app.py", label="Submit a pathway →")

render_funders()
