"""Streamlit surface for pathway mining.

Importing this package pulls in nothing heavy: :mod:`t2pw.mining.ui.entry` is the
only module the main app touches, and it imports the page (and through it the
controller) lazily, after the feature gate has already said yes.
"""

from __future__ import annotations

from t2pw.mining.ui.entry import MODE_KEY, mining_available, render_entry

__all__ = ["MODE_KEY", "mining_available", "render_entry"]
