"""C-075 -- does C-060a's AST instrument survive the orchestrator's app hunk?

``tests/test_c060a_requested_pathway_wiring.py`` lifts the app's two
``merge_additions`` statements out of ``streamlit_app.py`` BY AST and ``exec``s
the real source bytes in a namespace it builds by hand (``:149-160``). C-075
arms the Stage-2 site with ``source_text=text``, so that namespace must carry
``text`` or every arm in that file dies with ``NameError`` before reaching its
assertion -- disarming C-060a's protection without changing a line of C-060a's
test.

This probe proves the namespace entry is sufficient BEFORE the hunk lands. It
never edits the repo's ``streamlit_app.py`` -- that file belongs to the
orchestrator and carries an uncommitted product-owner edit. It writes a PATCHED
COPY to a scratch directory and points the test module's ``APP_PATH`` at it,
which is exactly the file the AST lift reads.

Usage::

    <python> c075_c060a_hunk_probe.py <scratch-dir>

Measured 2026-08-23 at C-075's tip:

    app merge sites lifted : [5573, 5628]
    stage-2 statement passes source_text : True
    lifted statement EXECUTED, no NameError
    compounds : ['isochorismate', '2,3-diDHB', 'pyruvate']
    reactions : ['EntB isochorismatase reaction']
    top-level keys added by the armed gate : ['biological_states', 'element_locations']

The last two lines are C-060a's own properties, intact: NADH and NAD are gone,
the synthesized composite reaction is gone, and no source index appears -- because
``APP_SOURCE_TEXT`` is deliberately empty, so this instrument keeps measuring the
``pathway_context`` wiring it owns rather than another card's rule.
"""
import ast, io, sys, types
from pathlib import Path

ROOT = Path("C:/t/c075")
SCRATCH = Path(sys.argv[1])
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

app = (ROOT / "src/t2pw/app/streamlit_app.py").read_text(encoding="utf-8")
OLD = """        final_payload = merge_additions(
            stage_one_in_scope,
            stage_two if isinstance(stage_two, dict) else {},
            pathway_context=pathway_context_from_stage_zero(pathway_context),
        )"""
NEW = """        final_payload = merge_additions(
            stage_one_in_scope,
            stage_two if isinstance(stage_two, dict) else {},
            pathway_context=pathway_context_from_stage_zero(pathway_context),
            source_text=text,
        )"""
assert OLD in app, "the Stage-2 merge statement is not where the probe expects it"
patched = SCRATCH / "streamlit_app_with_hunk.py"
patched.write_text(app.replace(OLD, NEW), encoding="utf-8")

import test_c060a_requested_pathway_wiring as t
t.APP_PATH = patched

sites = t._app_sites()
print("app merge sites lifted :", [s.lineno for s in sites])
stmt = ast.unparse(sites[0])
print("stage-2 statement passes source_text :", "source_text" in stmt)

payload = t._stage_two_site(t._f4_base(), t._f4_additions())
print("lifted statement EXECUTED, no NameError")
print("compounds :", t._names(payload))
print("reactions :", t._reactions(payload))
print("top-level keys added by the armed gate :",
      sorted(set(payload) - set(t._f4_base()) - {"entity_admission_report"}))
