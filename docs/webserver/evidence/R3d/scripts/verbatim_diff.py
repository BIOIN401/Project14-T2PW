"""R3d: unified diff of the moved base ranges against their new home, CR stripped.

The submit-block range is dedented by 4 (it sat inside ``if submit:``); the service body is
inside ``run_stage_two``, so it is dedented by 4 too.

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
    ("emit_rag_loop_messages + rag_loop_session_record",
     block(app, "def emit_rag_loop_messages(messages: Any) -> None:", '        if key not in {"payload", "messages"}'),
     block(ext, "def emit_rag_loop_messages(messages: Any, *, reporter: Reporter) -> None:",
           '        if key not in {"payload", "messages"}')),
    ("submit block: R3c rebinding, Stage 2 through the session-state publish",
     dedent(block(app, "    stage_one = _stage_one.stage_one", '    st.session_state.pop("post_pipeline_artifacts", None)')),
     dedent(block(ext, "    stage_one = stage_one_result.stage_one", '    state.pop("post_pipeline_artifacts", None)'))),
]
for title, old, new in pairs:
    print(f"===== {title}: base APP vs tip EXT")
    sys.stdout.writelines(line + "\n" for line in difflib.unified_diff(
        old, new, "base/streamlit_app.py", "tip/service/extraction.py", lineterm="", n=0))
    print()
