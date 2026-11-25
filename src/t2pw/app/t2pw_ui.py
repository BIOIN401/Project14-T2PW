"""T2PW Streamlit entry — landing first, top navigation (public-site chrome).

Launch::

    PYTHONPATH=src streamlit run src/t2pw/app/t2pw_ui.py

``streamlit run …/streamlit_app.py`` also hands off here for humans; AppTest still
loads the submit console directly.
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from t2pw.app.ui_chrome import inject_brand_css, render_top_nav

_APP_DIR = Path(__file__).resolve().parent

# Paths relative to this entrypoint (required by st.page_link / st.Page).
HOME = "ui_pages/home.py"
SUBMIT = "streamlit_app.py"
TUTORIAL = "ui_pages/tutorial.py"
ABOUT = "ui_pages/about.py"
EXAMPLES = "ui_pages/examples.py"


def run_app() -> None:
    st.set_page_config(
        page_title="T2PW",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    home = st.Page(_APP_DIR / HOME, title="Home", default=True, url_path="home")
    submit = st.Page(_APP_DIR / SUBMIT, title="Submit", url_path="submit")
    tutorial = st.Page(_APP_DIR / TUTORIAL, title="Tutorial", url_path="tutorial")
    about = st.Page(_APP_DIR / ABOUT, title="About", url_path="about")
    examples = st.Page(_APP_DIR / EXAMPLES, title="Examples", url_path="examples")

    pg = st.navigation(
        [home, submit, tutorial, about, examples],
        position="hidden",  # custom magenta bar below
    )

    inject_brand_css()
    render_top_nav(
        home_page=HOME,
        submit_page=SUBMIT,
        tutorial_page=TUTORIAL,
        about_page=ABOUT,
        examples_page=EXAMPLES,
    )
    pg.run()


if __name__ == "__main__":
    run_app()
