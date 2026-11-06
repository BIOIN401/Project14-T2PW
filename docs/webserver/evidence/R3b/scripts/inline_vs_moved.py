"""R3b display-equivalence proof: the BASE app's inline Stage-0 code (plus the base app's
own helper functions) vs the TIP app's call site + run_stage_zero + StreamlitReporter,
against the same recording fake ``st`` and the same fake ``preprocess``, over every
Stage-0 branch.

Both snippets run from ``reset_refinement_state()`` up to the Stage-1 comment, cut out
of the real app files and exec'd at module level, as Streamlit runs the script. The
base helpers (``_run_stage_zero_with_retry``, ``_start_extraction_diagnostics`` ...) are
lifted out of the BASE app by AST; the tip uses the committed service module.

Compared per scenario: the ordered st.* calls (spinner enter/exit included), where
st.stop() fired (or what was raised), every session_state write in order with its value,
every preprocess call (text + kwargs), every recorder activation (run_id, artifact_dir,
unique), the final ``pathway_context`` and ``_diagnostics_dir``. The random
``run-<hex>`` subdirectory of a unique recorder is normalised to ``run-X``.

usage: inline_vs_moved.py BASE_TREE TIP_TREE     (PYTHONPATH must be TIP_TREE/src)
"""
import ast
import json
import logging
import re
import sys
import tempfile
import textwrap
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Optional

base_tree, tip_tree = sys.argv[1], sys.argv[2]

import t2pw.paths as paths  # noqa: E402
import t2pw.pipeline.extraction_diagnostics as xd  # noqa: E402
import t2pw.service.extraction as ex  # noqa: E402
from t2pw.pipeline.preprocessor import (  # noqa: E402
    PREPROCESS_STATUS_KEY, describe_preprocess_status, is_ambiguous_multi_example_review_context,
    preprocess_was_recovered)
from t2pw.service.reporter import StreamlitReporter  # noqa: E402

assert ex.__file__.replace("\\", "/").startswith(tip_tree.replace("\\", "/")), ex.__file__
ROOT = Path(tempfile.mkdtemp(prefix="r3b-ivm-"))
REAL_ACTIVATE, REAL_CURRENT = xd.activate, xd.current
HEX = re.compile(r"run-[0-9a-f]{12}")


def norm(value):
    return HEX.sub("run-X", json.dumps(value, sort_keys=True, default=str))


def app_lines(tree):
    return Path(tree, "src/t2pw/app/streamlit_app.py").read_text(encoding="utf-8").splitlines()


def cut(tree):
    lines = app_lines(tree)
    start = lines.index("    reset_refinement_state()")
    end = lines.index("    # Stage 1: strict extraction with auto-repair")
    return textwrap.dedent("\n".join(lines[start:end]))


def lift(tree, names):
    """The named top-level defs / assignments of a tree's app, as one module."""
    src = "\n".join(app_lines(tree))
    keep = []
    for node in ast.parse(src).body:
        if isinstance(node, ast.FunctionDef) and node.name in names:
            keep.append(node)
        elif isinstance(node, ast.Assign) and any(getattr(t, "id", None) in names for t in node.targets):
            keep.append(node)
    found = {getattr(n, "name", None) or n.targets[0].id for n in keep}
    assert found == set(names), set(names) - found
    return compile(ast.Module(body=keep, type_ignores=[]), f"{tree}/streamlit_app.py", "exec")


BASE_HELPERS = lift(base_tree, {
    "REFINEMENT_STATE_DEFAULTS", "_PREPROCESS_RETRY_CHARS", "_has_usable_context",
    "_seed_context_from_user_scope", "_stage0_context_rank", "_run_stage_zero_with_retry",
    "reset_refinement_state", "EXTRACTION_DIAGNOSTICS_KEY", "EXTRACTION_DIAGNOSTICS_DIR_KEY",
    "DIAGNOSTICS_DIRNAME", "_start_extraction_diagnostics", "_sync_extraction_diagnostics"})
TIP_HELPERS = lift(tip_tree, {"REFINEMENT_STATE_DEFAULTS", "reset_refinement_state"})


class Stop(Exception):
    pass


class Ctx:
    def __init__(self, log, name):
        self.log, self.name = log, name

    def __enter__(self):
        self.log.append([self.name + ":enter"])

    def __exit__(self, *a):
        self.log.append([self.name + ":exit"])


class State(dict):
    def __init__(self, log):
        super().__init__()
        self._log = log

    def __setitem__(self, key, value):
        self._log.append(["session_state", key, norm(value)])
        super().__setitem__(key, value)


class FakeSt:
    def __init__(self):
        self.log = []
        self.session_state = State(self.log)

    def __getattr__(self, name):
        def call(*a, **k):
            self.log.append([name, [norm(x) for x in a], {kk: norm(v) for kk, v in k.items()}])
            if name == "stop":
                raise Stop()
            if name in ("spinner", "expander"):
                return Ctx(self.log, name)
        return call


def failed(status="llm_error", detail="timed out"):
    return {"pathway_name": "", "likely_organism": "", "key_compounds": [], "key_proteins": [],
            "gap_terms": [], PREPROCESS_STATUS_KEY: {"status": status, "detail": detail}}


def usable():
    return {"pathway_name": "Menaquinone biosynthesis", "likely_organism": "E. coli",
            PREPROCESS_STATUS_KEY: {"status": "ok"}}


def case_c(candidates=True, warning="Pick one of the examples."):
    ctx = {"document_type": "multi_example_review", "selected_example": "", "pathway_name": ""}
    if candidates:
        ctx["candidate_examples"] = [{"name": "Lipid A biosynthesis"}, {"name": "Menaquinone"}]
    if warning is not None:
        ctx["warning"] = warning
    return ctx


def recovered():
    return {"pathway_name": "Heme", PREPROCESS_STATUS_KEY: {"status": "ok", "recovered": True,
                                                           "detail": "cut at 800"}}


SHORT = "Menaquinone is made from chorismate by MenF."
LONG = "Lipid A review. " * 2000  # 32,000 chars > the 20,000-char retry head
RAISE = "raise"

S = [  # (name, text, draws, user_task_context, temperature, sync_raises)
    ("usable", SHORT, [usable()], "menaquinone", 0.0, False),
    ("failed-no-scope", SHORT, [failed()], None, 0.0, False),
    ("failed-unparseable", SHORT, [{PREPROCESS_STATUS_KEY: {"status": "unparseable", "raw_len": 9,
                                                            "raw_preview": "{oops"}}], None, 0.3, False),
    ("seeded-organism", SHORT, [failed()], "menaquinone biosynthesis in Lactococcus lactis", 0.0, False),
    ("seeded-no-organism", SHORT, [failed()], "lipid A biosynthesis", 0.0, False),
    ("seeded-long-tail", SHORT, [failed()], "x in vitro reconstitution of y", 0.0, False),
    ("seeded-keeps-organism", SHORT, [dict(failed(), likely_organism="")], "heme in yeast", 0.0, False),
    ("not-a-dict", SHORT, [None], None, 0.0, False),
    ("not-a-dict-seeded", SHORT, [None], "heme in yeast", 0.0, False),
    ("recovered", SHORT, [recovered()], None, 0.0, False),
    ("ambiguous-exit", SHORT, [case_c()], "lipid A", 0.0, False),
    ("ambiguous-bare", SHORT, [case_c(candidates=False, warning=None)], None, 0.0, False),
    ("ambiguous-blank-warning", SHORT, [case_c(warning="   ")], None, 0.0, False),
    ("ambiguous-nonstr-warning", SHORT, [case_c(warning=["x"])], None, 0.0, False),
    ("long-case-c-no-retry", LONG, [case_c()], "lipid A biosynthesis", 0.0, False),
    ("long-retry-adopted", LONG, [failed(), usable()], "u", 0.2, False),
    ("long-retry-rejected", LONG, [failed(), failed("empty_reply", "")], None, 0.0, False),
    ("long-retry-rejected-seeded", LONG, [failed(), failed()], "heme biosynthesis in yeast", 0.0, False),
    ("long-retry-case-c-exit", LONG, [failed(), case_c()], "lipid A", 0.0, False),
    ("long-usable-no-retry", LONG, [usable()], None, 0.0, False),
    ("preprocess-raises", SHORT, [RAISE], None, 0.0, False),
    ("retry-raises", LONG, [failed(), RAISE], None, 0.0, False),
    ("sync-raises", SHORT, [usable()], None, 0.0, True),
    ("sync-raises-ambiguous", SHORT, [case_c()], None, 0.0, True),
]


def run(arm, scen):
    name, text, draws, utc, temp, sync_raises = scen
    st, calls, activations, queue = FakeSt(), [], [], [deepcopy(d) for d in draws]

    def preprocess(t, **kw):
        calls.append([t[:40], len(t), kw])
        d = queue.pop(0)
        if d == RAISE:
            raise RuntimeError("LLM down")
        return d

    def activate(**kw):
        rec = REAL_ACTIVATE(**kw)
        activations.append([norm(kw), norm(str(rec.artifact_dir))])
        return rec

    def current():
        if sync_raises:
            raise RuntimeError("no recorder")
        return REAL_CURRENT()

    ns = dict(st=st, text=text, temperature=temp, user_task_context=utc, preprocess=preprocess,
              is_ambiguous_multi_example_review_context=is_ambiguous_multi_example_review_context,
              describe_preprocess_status=describe_preprocess_status,
              preprocess_was_recovered=preprocess_was_recovered, deepcopy=deepcopy, Any=Any,
              Dict=Dict, Optional=Optional, Path=Path, logger=logging.getLogger("r3b-ivm"))
    paths.PROJECT_ROOT = ROOT
    if arm == "base":
        ns.update(PROJECT_ROOT=ROOT, activate_extraction_diagnostics=activate,
                  diagnostic_payload_hash=xd.payload_hash, current_extraction_diagnostics=current)
        exec(BASE_HELPERS, ns)
    else:
        xd.activate, xd.current = activate, current  # the tip looks these up at call time
        ns.update(run_stage_zero=ex.run_stage_zero, StreamlitReporter=StreamlitReporter)
        exec(TIP_HELPERS, ns)
    outcome = "completed"
    try:
        exec(compile(cut(base_tree if arm == "base" else tip_tree), f"<{arm}:{name}>", "exec"), ns)
    except Stop:
        outcome = "stopped"
    except Exception as exc:  # noqa: BLE001 - the raise itself is compared
        outcome = f"raised {type(exc).__name__}: {exc}"
    finally:
        xd.activate, xd.current = REAL_ACTIVATE, REAL_CURRENT
        xd.deactivate()
    return {"st": st.log, "outcome": outcome, "preprocess": calls, "activate": activations,
            "pathway_context": norm(ns.get("pathway_context")),
            "_diagnostics_dir": norm(str(ns["_diagnostics_dir"])) if "_diagnostics_dir" in ns else None}


old, new = cut(base_tree), cut(tip_tree)
assert "= run_stage_zero(" not in old and "_run_stage_zero_with_retry(" in old
assert "= run_stage_zero(" in new and "_run_stage_zero_with_retry" not in new
bad = 0
for scen in S:
    a, b = run("base", scen), run("tip", scen)
    same = norm(a) == norm(b)
    diff = [k for k in a if norm(a[k]) != norm(b[k])]
    # Known, reported difference: when Stage 0 RAISES, the base had already bound the
    # script global _diagnostics_dir; the tip binds it only from the returned result.
    # The exception ends the script run either way, and Streamlit re-execs the script
    # in a fresh namespace, so nothing can read it. Everything else must match.
    allowed = not same and a["outcome"].startswith("raised") and a["outcome"] == b["outcome"]         and diff == ["_diagnostics_dir"]
    bad += not (same or allowed)
    n_state = sum(1 for e in a["st"] if e[0] == "session_state")
    print(f"{scen[0]:28s} {'IDENTICAL' if same else ('SAME-EXCEPT-UNBOUND-GLOBAL' if allowed else 'DIFFERENT')}  outcome={a['outcome']:<28s} "
          f"st-events={len(a['st'])} state-writes={n_state} preprocess-calls={len(a['preprocess'])}")
    if not same:
        print("  differing fields:", diff, "| base _diagnostics_dir:", a["_diagnostics_dir"] and "bound",
              "| tip _diagnostics_dir:", b["_diagnostics_dir"])
    if not (same or allowed):
        print("  base:", norm(a)[:3000])
        print("  tip :", norm(b)[:3000])
print(f"{len(S) - bad}/{len(S)} scenarios identical or known-difference only; {bad} unexplained")
sys.exit(1 if bad else 0)
