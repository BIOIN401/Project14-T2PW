"""The gated hook the main Streamlit app calls.

Designed so that the entire cost of this feature, when it is disabled, is one
function call that returns ``False`` after reading one environment variable. It
imports nothing heavy at module scope: :mod:`t2pw.mining.ui.page` -- and through
it the controller, the model client and the vector store -- is imported only
after the gate has already said yes.

The main app's integration is three lines and a no-op by default. See
``docs/pathway_mining/streamlit.md``.
"""

from __future__ import annotations

from typing import Any, Optional

#: The session-state key the main app's toggle writes.
MODE_KEY = "mining_mode_active"


def mining_available(override: Optional[bool] = None) -> bool:
    """Cheap gate. ``False`` unless the feature flag (or an override) says yes."""
    from t2pw.mining.flags import mining_enabled

    return mining_enabled(override)


def render_entry(st: Any, *, override: Optional[bool] = None) -> bool:
    """Offer the Pathway Mining mode, and render it if the user is in it.

    Returns ``True`` when the mining page took over the render, so the caller can
    skip the rest of its own page. Returns ``False`` in every other case,
    including whenever the feature is disabled -- which is what keeps the default
    path byte-for-byte unchanged.
    """
    if not mining_available(override):
        return False

    active = bool(st.session_state.get(MODE_KEY, False))
    choice = st.sidebar.radio(
        "Mode",
        ["PWML pipeline", "Pathway mining (experimental)"],
        index=1 if active else 0,
        key="mining_mode_radio",
        help=(
            "Pathway mining is a separate, experimental research-graph mode. It does not "
            "produce PWML and does not touch the pipeline on this page."
        ),
    )
    now_active = choice.startswith("Pathway mining")
    st.session_state[MODE_KEY] = now_active
    if not now_active:
        return False

    from t2pw.mining.ui.page import render_page

    render_page(st)
    return True


__all__ = ["MODE_KEY", "mining_available", "render_entry"]
