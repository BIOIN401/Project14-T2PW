"""C-077 non-collateral census: manifest rows for legs with NO Stage-0 conflict.

Runs a fixed corpus of driver legs through the PRODUCTION entry point
(``t2pw.batch.driver.run_one``) against the same throwaway fixture apps
``tests/test_batch_driver.py`` uses, and prints each leg's manifest row as
canonical JSON.

Run it on the base tree and on the tip and diff the two outputs. Any difference
on a leg that has no Stage-0 conflict is collateral from C-077, by construction:
the corpus deliberately covers every terminal path the driver has -- a strict
pass with a frozen release record, a research pass, a gate-blocked strict leg, an
empty payload, a Stage-1 refusal, an agreeing Stage-0 reading in both modes, a
pinned paper with no request to contradict, a legacy plan record, and the
annotate-only conflict branch (``stage0_conflict_aborts=false``) which records the
conflict and carries on.

The ABORTING conflict branch is deliberately NOT in this corpus. That branch is
what the card changes and its rows are supposed to differ; ``tests/
test_c077_stage0_conflict_disposition.py`` is where that difference is asserted.

NORMALISATION. Two fields move between identical runs for reasons unrelated to
any code change and are replaced rather than dropped, so a field that vanished
entirely would still show up in the diff:

* ``seconds`` -- wall clock.
* every absolute path inside ``detail`` -- the pytest/OS temp directory differs
  per invocation and per tree.

Usage::

    PYTHONPATH=<tree>/src <py> docs/pwml_recovery_sprint/evidence/c077_row_census.py
"""

from __future__ import annotations

import json
import os
import re
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
for _path in (ROOT / "src", ROOT / "tests"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from test_batch_driver import (  # noqa: E402
    _RELEASE_READY,
    _Paper,
    _artifacts,
    _lit,
    _post_pipeline_body,
    _run,
    _write_app,
)
from t2pw.batch.driver import RESEARCH, STRICT  # noqa: E402

_PATHISH = re.compile(r"[A-Za-z]:[\\/][^\s\"']*|/tmp/[^\s\"']*")


def _stage0(context: object) -> str:
    return f'    st.session_state["pathway_context"] = {_lit(context)}\n'


def _scoped(pathway: str, organism: str, paper_id: str = "PMC1") -> _Paper:
    paper = _Paper(paper_id=paper_id, topic=pathway, organism=organism)
    paper.requested_pathway = pathway
    paper.requested_organism = organism
    return paper


def _normalise(row: dict) -> dict:
    out = dict(row)
    out["seconds"] = "<elapsed>"
    if isinstance(out.get("detail"), str):
        out["detail"] = _PATHISH.sub("<path>", out["detail"])
    if isinstance(out.get("message"), str):
        out["message"] = _PATHISH.sub("<path>", out["message"])
    return out


#: The full-strict PWML result a passing strict leg's app publishes.
_PWML_OK = {
    "ok": True,
    "_xml": "<pathway><name>menaquinone</name></pathway>",
    "pwml_ir": {"pathway": {"name": "menaquinone"}},
    "pwml_ir_report": {"counts": {"reactions": 1}},
    "validation_report": {"issues": 0},
    "qa": {"ok": True},
    "counts": {"reactions": 1, "compounds": 2},
    "output_path": "outputs/pathway.pwml",
    "quarantine_report": {"release": _RELEASE_READY},
}


def _legs(tmp: Path):
    """``(name, app_body, mode, paper, env)`` for every non-conflict leg."""

    agree = _stage0({"pathway_name": "lipid A biosynthesis", "likely_organism": "Escherichia coli"})
    conflict = _stage0({"pathway_name": "menaquinone biosynthesis", "likely_organism": "Lactococcus lactis"})
    return [
        (
            "strict_pass",
            _post_pipeline_body(_artifacts("pathwhiz"), pwml=_PWML_OK),
            STRICT,
            _Paper(),
            {},
        ),
        (
            "research_pass",
            _post_pipeline_body(_artifacts("research")),
            RESEARCH,
            _Paper(),
            {},
        ),
        (
            "strict_gate_fail",
            _post_pipeline_body(
                _artifacts(
                    "pathwhiz",
                    gate_failed=True,
                    gate_fail_report={
                        "status": "failed",
                        "stage": "post_normalization_hard_gates",
                        "errors": [{"path": "/x", "reason": "Forbidden complex reference detected: z"}],
                    },
                ),
                pwml=_PWML_OK,
            ),
            STRICT,
            _Paper(),
            {},
        ),
        (
            "empty_payload",
            '''
if submitted:
    st.session_state["pipeline_ready"] = True
    st.session_state["stage_one"] = {"entities": {}, "processes": {}}
    st.session_state["final_payload"] = {"entities": {}, "processes": {}}
''',
            RESEARCH,
            _Paper(),
            {},
        ),
        (
            "stage1_refused",
            '''
if submitted:
    st.error("Payload must include a processes object")
    st.stop()
''',
            STRICT,
            _Paper(),
            {},
        ),
        (
            "agreeing_stage0_research",
            _post_pipeline_body(_artifacts("research"), extra=agree),
            RESEARCH,
            _scoped("lipid A biosynthesis", "Escherichia coli"),
            {},
        ),
        (
            "agreeing_stage0_strict",
            _post_pipeline_body(_artifacts("pathwhiz"), pwml=_PWML_OK, extra=agree),
            STRICT,
            _scoped("lipid A biosynthesis", "Escherichia coli"),
            {},
        ),
        (
            "pinned_paper_no_request",
            _post_pipeline_body(_artifacts("research"), extra=conflict),
            RESEARCH,
            _Paper(paper_id="PMC9"),
            {},
        ),
        (
            "legacy_plan_record",
            _post_pipeline_body(_artifacts("research"), extra=agree),
            RESEARCH,
            _Paper(paper_id="PMC5", topic="lipid A biosynthesis", organism="Escherichia coli"),
            {},
        ),
        (
            "annotate_only_conflict",
            _post_pipeline_body(_artifacts("research"), extra=conflict),
            RESEARCH,
            _scoped("menaquinone biosynthesis", "Bacillus subtilis", "PMC12421875"),
            {"RAG_ELIGIBILITY_STAGE0_CONFLICT_ABORTS": "false"},
        ),
    ]


def main() -> int:
    census = {}
    with tempfile.TemporaryDirectory(prefix="c077census") as raw:
        tmp = Path(raw)
        for name, body, mode, paper, env in _legs(tmp):
            saved = {key: os.environ.get(key) for key in env}
            os.environ.update(env)
            try:
                app = _write_app(tmp, name, body)
                outcome = _run(app, mode, paper)
            finally:
                for key, value in saved.items():
                    if value is None:
                        os.environ.pop(key, None)
                    else:
                        os.environ[key] = value
            census[name] = _normalise(outcome.to_dict())
    json.dump(census, sys.stdout, indent=1, sort_keys=True, default=str)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
