"""R4b verbatim proof: diff the base handler's body (from the reenter line through the finally) against
the service body, dedented. Prints only the changed lines.

usage: verbatim_diff.py BASE_TREE TIP_TREE
"""
import difflib
import sys
import textwrap
from pathlib import Path

base = Path(sys.argv[1], "src/t2pw/app/streamlit_app.py").read_text(encoding="utf-8")
tip = Path(sys.argv[2], "src/t2pw/service/post_run.py").read_text(encoding="utf-8")


def rng(text, first, last):
    lines = text.splitlines()
    start = next(i for i, l in enumerate(lines) if l.strip().startswith(first))
    end = next(i for i, l in enumerate(lines[start:], start) if l.strip().startswith(last)) + 1
    return textwrap.dedent("\n".join(lines[start:end]))


a = rng(base, "reenter_extraction_diagnostics(st.session_state", "_sync_extraction_diagnostics()").splitlines()
b = rng(tip, "_diag.reenter(state.get", "_extraction._sync_extraction_diagnostics(state=state)").splitlines()
for line in difflib.unified_diff(a, b, "base-handler", "post_run", lineterm="", n=0):
    print(line)
