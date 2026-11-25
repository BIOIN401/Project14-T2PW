"""Tutorial / help — Streamlit-oriented copy of the public /help page."""

from __future__ import annotations

import streamlit as st

from t2pw.app.ui_chrome import page_chrome, render_funders

page_chrome("Tutorial")

st.header("Help")

st.subheader("How to start")
st.markdown(
    """
- **Paste text.** One or more pathway descriptions (use *Add another pathway* on Submit).
- **Upload PDF.** One or more scientific PDFs; optional page range and OCR.
- **Text + PDF.** Combine pasted text with uploaded papers.

Export mode:

- **Strict PWML** — PathBank-ready checks; use for anything destined for PathBank.
- **Research (relaxed)** — formatting rules relaxed; output is a candidate for review, not an importable PWML file.
"""
)

st.subheader("What happens next")
st.markdown(
    """
The pipeline runs in this process (Stages 0–2, enrichment, export). Watch the page for
progress and download artifacts when finished.

The public site queues up to five pathways with pathway name, PubMed IDs, or PDFs and
offers a private job link when done.
"""
)

st.page_link("streamlit_app.py", label="Submit a pathway →")

render_funders()
