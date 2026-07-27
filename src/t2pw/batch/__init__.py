"""Unattended batch-run support for the T2PW pipeline.

This package drives the *real* Streamlit app headlessly over many papers so an
overnight run exercises exactly the code a human would exercise in the browser.
Nothing here is a second implementation of the pipeline: acquisition reuses
``t2pw.rag.acquire`` and execution reuses ``src/t2pw/app/streamlit_app.py``, so
any change to the app is automatically reflected in the batch runner.

Unlike the pipeline/mapping/curation/pwml packages (which must never import
``t2pw.rag``), this package sits above every layer and may import anything.
"""

from __future__ import annotations
