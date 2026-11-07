"""Unified diff of the moved base ranges against their new home, CR stripped.

The submit-block range is dedented by 4 (it sat inside ``if submit:``); the service body
is inside ``run_stage_one``, so it is dedented by 4 too.

usage: verbatim_diff.py BASE_TREE TIP_TREE > verbatim-diff.txt
"""
import difflib
import sys
import textwrap
from pathlib import Path

base, tip = sys.argv[1], sys.argv[2]
app = Path(base, "src/t2pw/app/streamlit_app.py").read_text(encoding="utf-8").splitlines()
ext = Path(tip, "src/t2pw/service/extraction.py").read_text(encoding="utf-8").splitlines()


def block(lines, first, last):
    i = lines.index(first)
    j = lines.index(last, i)
    return lines[i:j + 1]


def dedent(lines):
    return textwrap.dedent("\n".join(lines)).splitlines()


pairs = [
    ("render_attempts", block(app, "def render_attempts(label: str, attempts: List[Dict[str, Any]]) -> None:",
                              '                st.caption(log["error"])'),
     block(ext, "def render_attempts(label: str, attempts: List[Dict[str, Any]], *, reporter: Reporter) -> None:",
           '                reporter.caption(log["error"])')),
    ("_last_cleaning_report", block(app, "def _last_cleaning_report() -> Dict[str, Any]:",
                                    "    return _safe_dict(passes[-1]) if passes else {}"),
     block(ext, "def _last_cleaning_report() -> Dict[str, Any]:",
           "    return _safe_dict(passes[-1]) if passes else {}")),
    ("submit block: Stage 1 through the RAG seed",
     dedent(block(app, "    # Stage 1: strict extraction with auto-repair",
                  '                st.session_state["rag_evidence_context"] = rag_result.evidence_context  # S2')),
     dedent(block(ext, "    # Stage 1: strict extraction with auto-repair",
                  '                state["rag_evidence_context"] = rag_result.evidence_context  # S2'))),
]
for title, old, new in pairs:
    print(f"===== {title}: base APP vs tip EXT")
    sys.stdout.writelines(line + "\n" for line in difflib.unified_diff(
        old, new, "base/streamlit_app.py", "tip/service/extraction.py", lineterm="", n=1))
    print()
