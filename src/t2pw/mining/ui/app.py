"""Standalone Streamlit entry point for Pathway Mining.

    streamlit run src/t2pw/mining/ui/app.py

The page also appears inside the main app when the feature flag is on (see
:mod:`t2pw.mining.ui.entry`); this file exists so the experimental mode can be
opened on its own, without loading the PWML pipeline page at all.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Streamlit executes this file directly, so Python puts its own directory on
# sys.path instead of the src root that holds the t2pw package. Same fix the
# main app uses.
SRC_ROOT = Path(__file__).resolve().parents[3]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import streamlit as st  # noqa: E402

from t2pw.mining.ui.page import render_page  # noqa: E402

st.set_page_config(page_title="T2PW Pathway Mining (experimental)", layout="wide")
render_page(st)
