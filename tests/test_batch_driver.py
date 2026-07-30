"""Tests for the batch driver's contract with the Streamlit app.

The real app cannot run here: it needs an LLM, a PathBank MySQL instance and the
network. What *can* be tested -- and is the part that actually breaks -- is the
driver's side of the contract: which widgets it drives, which session_state keys
it reads, how it tells a pass from a fail, and how it classifies what went wrong.

So every test points ``AppTest`` at a tiny throwaway script that speaks the app's
contract (same widget keys, same labels, same session_state keys, same
st.error+st.stop failure convention) and nothing else. These scripts are the
executable record of what the driver assumes about the app; if the app changes,
one of these fixtures should change with it.
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from t2pw.batch.driver import (  # noqa: E402
    DETAIL_LIMIT,
    KIND_TIMEOUT,
    MODE_RESEARCH,
    MODE_STRICT,
    RESEARCH,
    RUNTIME_SCHEMA_WARN_LIMIT,
    STRICT,
    WARN_NO_FOCUS_BOX,
    WARN_NO_RESEARCH_REPORT,
    WARN_RUNTIME_SCHEMA_PREFIX,
    RunOutcome,
    run_one,
)


# ---------------------------------------------------------------------------
# Real streamlit, whatever the rest of the suite did to sys.modules.
# ---------------------------------------------------------------------------
def _is_real_streamlit(module: object) -> bool:
    """A ``MagicMock`` is not a module; a bare ``ModuleType`` stub has no path."""

    return isinstance(module, types.ModuleType) and hasattr(module, "__path__")


@pytest.fixture(autouse=True)
def real_streamlit():
    """Guarantee the genuine ``streamlit`` package for the duration of each test.

    ``tests/test_stage0_scope_fallback.py`` and the two ``test_rag_*`` modules
    install a ``MagicMock`` as ``sys.modules["streamlit"]`` at *import* time --
    they must, because ``t2pw.app.streamlit_app`` executes a Streamlit script on
    import. Pytest imports every test module during collection, so by the time
    this module's tests run the stub is already in place and ``AppTest`` would be
    a MagicMock returning MagicMocks: every assertion here would then pass or
    fail for reasons unrelated to the driver.

    Whatever was in ``sys.modules`` is restored afterwards, so those modules keep
    the stub they captured and this file stays invisible to the rest of the suite
    regardless of collection order.
    """

    def _streamlit_keys():
        return [n for n in list(sys.modules) if n == "streamlit" or n.startswith("streamlit.")]

    saved = {name: sys.modules[name] for name in _streamlit_keys()}
    stubbed = not _is_real_streamlit(sys.modules.get("streamlit"))
    if stubbed:
        for name in saved:
            del sys.modules[name]
    import streamlit  # noqa: F401,PLC0415
    import streamlit.testing.v1  # noqa: F401,PLC0415

    assert _is_real_streamlit(sys.modules["streamlit"]), "the real streamlit did not load"
    try:
        yield
    finally:
        if stubbed:
            for name in _streamlit_keys():
                del sys.modules[name]
            sys.modules.update(saved)


# ---------------------------------------------------------------------------
# Fixture-app construction.
# ---------------------------------------------------------------------------
#: The widgets the driver drives, in the order and with the keys the app uses.
#: ``st.session_state.get`` is legal *inside* an app script (it is real streamlit
#: session state there); it is only ``AppTest.session_state`` that lacks ``.get``.
_PREAMBLE_HEAD = '''\
import streamlit as st
from types import SimpleNamespace

st.radio("Export mode", ["Strict PWML", "Research (relaxed)"], key="export_mode_radio")
st.radio("Input mode", ["Paste text", "Upload PDF", "Text + PDF"], key="input_mode_radio")
with st.form("pwml_pipeline"):
    st.text_area("Paste pathway description:", key="pathway_text_0")
'''

#: The extraction-focus box, reproduced with the app's exact signature: NO key.
#: That is the whole point -- ``at.text_area(key=...)`` cannot reach it, so the
#: driver has to find it by label. The observed value is echoed into session_state
#: so a test can assert what the app actually received.
_PREAMBLE_FOCUS = '''\
    user_task_context = st.text_area(
        "Optional extraction focus / task context",
        height=100,
        help="Use this to tell the model what pathway or scope you want extracted.",
    )
    st.session_state["observed_focus"] = user_task_context
'''

_PREAMBLE_TAIL = '''\
    submitted = st.form_submit_button("Run pipeline")
'''


def _write_app(tmp_path: Path, name: str, body: str, *, focus_box: bool = True) -> Path:
    """Write a fixture app script. utf-8 always: entity names are not ASCII.

    ``focus_box=False`` reproduces an app that stopped rendering the unkeyed
    extraction-focus text area, which the driver must survive with a warning.
    """

    preamble = _PREAMBLE_HEAD + (_PREAMBLE_FOCUS if focus_box else "") + _PREAMBLE_TAIL
    target = tmp_path / f"{name}.py"
    target.write_text(preamble + body, encoding="utf-8")
    return target


def _lit(value: object) -> str:
    """Embed a Python value in a generated script.

    ``repr``, not ``json.dumps``: JSON's ``false``/``true``/``null`` are not
    Python names and the generated script would die on a ``NameError``.
    """

    return repr(value)


class _Paper:
    """A stand-in for ``t2pw.batch.fetch.BatchPaper`` (duck-typed by the driver).

    ``topic`` and ``organism`` are the two fields the extraction focus is composed
    from. ``topic == ""`` is how :class:`~t2pw.batch.fetch.BatchPaper` records a
    *pinned* paper -- one given by id, with no search behind it.
    """

    def __init__(
        self,
        paper_id: str = "PMC1",
        full_text: str = "Some pathway prose.",
        topic: str = "",
        organism: str = "",
    ) -> None:
        self.paper_id = paper_id
        self.full_text = full_text
        self.topic = topic
        self.organism = organism


PAPER = _Paper()


def _run(app: Path, mode: str, paper: object = PAPER) -> RunOutcome:
    return run_one(paper, mode, app_path=app, timeout=120.0, app_timeout=45.0)


# ---------------------------------------------------------------------------
# Payload / artifact shapes, trimmed to what the driver reads.
# ---------------------------------------------------------------------------
_STAGE1 = {
    "entities": {"compounds": [{"name": "menaquinone"}], "proteins": [{"name": "MenG"}]},
    "processes": {"reactions": [{"name": "menaquinone demethylation"}]},
}
_MERGED = {
    "entities": {
        "compounds": [{"name": "menaquinone"}, {"name": "demethylmenaquinone"}],
        "proteins": [{"name": "MenG"}],
    },
    "processes": {
        "reactions": [
            {
                "name": "menaquinone demethylation",
                "inputs": ["menaquinone"],
                "outputs": ["demethylmenaquinone"],
            }
        ]
    },
}


def _artifacts(mode_value: str, **overrides: object) -> dict:
    base = {
        "gate_failed": False,
        "normalization_gate_failed": False,
        "gate_fail_report": {},
        "export_mode": mode_value,
        "research_review_flags": [],
        "research_skipped_format_rules": [],
        "research_normalization_actions": [],
        "final_mapped_db": _MERGED,
        "final_mapped": _MERGED,
        "mapping_report": {"summary": {"mapped": 3}},
        "post_extraction_contract_report": {"ok": True, "errors": [], "warnings": []},
    }
    base.update(overrides)
    return base


def _post_pipeline_body(artifacts: dict, *, pwml: object = None, extra: str = "") -> str:
    """The stage-1 → post-pipeline → PWML click chain the driver walks."""

    pwml_block = ""
    if pwml is not None:
        pwml_block = f'''
    if st.session_state.get("post_pipeline_artifacts"):
        if st.button("Generate PWML", key="refinement_generate_pwml"):
            result = {_lit(pwml)}
            xml = result.pop("_xml", "")
            if xml:
                result["xml_bytes"] = xml.encode("utf-8")
            st.session_state["pwml_export_result"] = result
'''
    return f'''
if submitted:
    st.session_state["pipeline_ready"] = True
    st.session_state["stage_one"] = {_lit(_STAGE1)}
    st.session_state["final_payload"] = {_lit(_MERGED)}
{extra}
if st.session_state.get("pipeline_ready"):
    if st.button("Run audit and DB mapping", key="pwml_generate_btn"):
        st.session_state["post_pipeline_artifacts"] = {_lit(artifacts)}
{pwml_block}
'''


# ---------------------------------------------------------------------------
# A passing strict run.
# ---------------------------------------------------------------------------
def test_strict_pass_writes_pwml_and_reports(tmp_path: Path) -> None:
    app = _write_app(
        tmp_path,
        "strict_pass",
        _post_pipeline_body(
            _artifacts("pathwhiz"),
            pwml={
                "ok": True,
                "_xml": "<pathway><name>menaquinone</name></pathway>",
                "pwml_ir": {"pathway": {"name": "menaquinone"}},
                "pwml_ir_report": {"counts": {"reactions": 1}},
                "validation_report": {"issues": 0},
                "qa": {"ok": True},
                "counts": {"reactions": 1, "compounds": 2},
                "output_path": "outputs/pathway.pwml",
            },
        ),
    )

    outcome = _run(app, STRICT)

    assert outcome.status == "pass", outcome.detail
    assert outcome.mode == MODE_STRICT
    assert outcome.stage == "pwml_export"
    assert outcome.failure_kind == ""
    assert outcome.paper_id == "PMC1"
    assert outcome.seconds >= 0.0

    # The .pwml file is the whole point of strict mode: it must be real bytes.
    assert outcome.artifacts["pathway.pwml"].startswith(b"<pathway>")
    assert json.loads(outcome.artifacts["pwml_ir.json"])["pathway"]["name"] == "menaquinone"
    assert json.loads(outcome.artifacts["final_mapped.json"]) == _MERGED
    assert json.loads(outcome.artifacts["stage1_payload.json"]) == _STAGE1
    assert json.loads(outcome.artifacts["merged_payload.json"]) == _MERGED
    assert "pwml_validation_report.json" in outcome.artifacts
    assert "research_pathway_report.txt" not in outcome.artifacts

    assert outcome.counts["reactions"] == 1
    assert outcome.counts["pwml_bytes"] == len(outcome.artifacts["pathway.pwml"])
    assert outcome.counts["pwml"]["reactions"] == 1

    row = outcome.to_dict()
    assert row["status"] == "pass" and row["mode"] == "strict"
    assert {"name": "pathway.pwml", "bytes": len(outcome.artifacts["pathway.pwml"])} in row["files"]


def test_strict_fails_when_pwml_export_reports_not_ok(tmp_path: Path) -> None:
    """A pass is never fabricated: no XML means fail, whatever else succeeded."""

    app = _write_app(
        tmp_path,
        "strict_no_xml",
        _post_pipeline_body(
            _artifacts("pathwhiz"),
            pwml={
                "ok": False,
                "error": "Required-field gate rejected the payload",
                "required_gate_report": {
                    "ok": False,
                    "errors": [
                        {"code": "missing_external_identity", "pointer": "/entities/proteins/0"}
                    ],
                },
            },
        ),
    )

    outcome = _run(app, STRICT)

    assert outcome.status == "fail"
    assert outcome.failure_kind == "contract"
    assert "missing_external_identity" in outcome.issue_codes
    assert "pathway.pwml" not in outcome.artifacts
    assert "Required-field gate" in outcome.message


def test_strict_fails_when_review_step_never_unlocks_pwml(tmp_path: Path) -> None:
    """No ``refinement_generate_pwml`` button means no PWML -- and strict needs one."""

    app = _write_app(tmp_path, "strict_no_button", _post_pipeline_body(_artifacts("pathwhiz")))

    outcome = _run(app, STRICT)

    assert outcome.status == "fail"
    assert outcome.stage == "pwml_export"
    assert "refinement_generate_pwml" in outcome.message


# ---------------------------------------------------------------------------
# A passing research run.
# ---------------------------------------------------------------------------
#: A pre-merge synthesized payload: RAG provenance still attached, which is
#: exactly why the report is built from ``rag_result.payload`` and not the merged
#: payload (the merge strips these carriers, tiering everything Tier D).
_PROV = {
    "source_id": "PMC1",
    "source_title": "Menaquinone study",
    "source_type": "paper",
    "source_uri": "https://example.org/PMC1",
    "section": "results",
    "chunk_id": "a" * 32,
}
_SYNTHESIZED = {
    "metadata": {"pathway_name": "Menaquinone biosynthesis"},
    "entities": {
        "compounds": [{"name": "menaquinone", "rag_provenance": _PROV}],
        "proteins": [{"name": "MenG", "rag_provenance": _PROV}],
    },
    "processes": {
        "reactions": [
            {
                "name": "menaquinone demethylation",
                "inputs": ["menaquinone"],
                "outputs": ["demethylmenaquinone"],
                "rag_provenance": _PROV,
                "source_refs": ["PMC1"],
                "source_papers": [
                    {
                        "source_id": "PMC1",
                        "title": "Menaquinone study",
                        "uri": "https://example.org/PMC1",
                        "source_type": "paper",
                    }
                ],
                "rag_confidence": 0.83,
            }
        ]
    },
}

_RAG_RESULT_STANZA = f'''
    st.session_state["rag_result"] = SimpleNamespace(
        payload={_lit(_SYNTHESIZED)},
        candidates=[],
        selection=None,
        evidence_context="PMC1: MenG demethylates menaquinone.",
    )
'''

_RESEARCH_FLAGS = [
    {
        "code": "reaction_enzyme_unresolved",
        "pointer": "/processes/reactions/0/enzymes/0",
        "message": "enzyme has no external identity",
        "research_category": "biology",
    }
]
_RESEARCH_GAPS = [
    {
        "code": "missing_external_identity",
        "pointer": "/entities/proteins/0",
        "message": "PathWhiz requires an external DB id",
    }
]


def test_research_pass_builds_citation_report_from_pre_merge_payload(tmp_path: Path) -> None:
    app = _write_app(
        tmp_path,
        "research_pass",
        _post_pipeline_body(
            _artifacts(
                "research",
                research_review_flags=_RESEARCH_FLAGS,
                research_skipped_format_rules=_RESEARCH_GAPS,
                research_normalization_actions=[
                    {"type": "research_mode_kept_colon_name", "name": "a:b", "reason": "format only"}
                ],
            ),
            extra=_RAG_RESULT_STANZA,
        ),
    )

    outcome = _run(app, RESEARCH)

    assert outcome.status == "pass", outcome.detail
    assert outcome.mode == MODE_RESEARCH
    assert outcome.stage == "research_report"
    assert outcome.failure_kind == ""

    # Research mode deliberately produces no PWML; its absence is not a failure.
    assert "pathway.pwml" not in outcome.artifacts

    report = outcome.artifacts["research_pathway_report.txt"]
    assert isinstance(report, str) and "menaquinone demethylation" in report
    citations = json.loads(outcome.artifacts["research_pathway_citations.json"])
    assert citations["summary"]["reaction_count"] == 1
    csv_text = outcome.artifacts["research_pathway_elements.csv"]
    assert csv_text.splitlines()[0].count(",") > 3
    flags = json.loads(outcome.artifacts["review_flags.json"])
    assert flags["biology_provenance_flags"] == _RESEARCH_FLAGS
    assert flags["format_rules_skipped"] == _RESEARCH_GAPS

    assert outcome.counts["review_flags"] == 1
    assert outcome.counts["skipped_format_rules"] == 1
    assert outcome.counts["normalization_actions"] == 1
    assert sum(outcome.counts["tier"].values()) >= 1


def test_research_without_rag_still_passes_and_says_why(tmp_path: Path) -> None:
    """RAG not triggering is normal. It must not be reported as a failure.

    But it must also not be mistakable for a clean run: with no report, no
    citations and no tiers in the folder, a night of these would otherwise be
    logged as ALL GREEN having produced no research deliverable at all.
    """

    app = _write_app(
        tmp_path,
        "research_no_rag",
        _post_pipeline_body(_artifacts("research", research_review_flags=_RESEARCH_FLAGS)),
    )

    outcome = _run(app, RESEARCH)

    assert outcome.status == "pass", outcome.detail
    assert outcome.failure_kind == ""
    assert "no RAG synthesis" in outcome.message
    assert "research_pathway_report.txt" not in outcome.artifacts
    # The flags are the one research deliverable that never depends on RAG.
    assert json.loads(outcome.artifacts["review_flags.json"])["biology_provenance_flags"]

    # ...and the warning travels into the manifest row.
    assert outcome.warnings == [WARN_NO_RESEARCH_REPORT]
    assert outcome.to_dict()["warnings"] == [WARN_NO_RESEARCH_REPORT]


def test_a_research_run_with_a_report_carries_no_warning(tmp_path: Path) -> None:
    app = _write_app(
        tmp_path,
        "research_warn_free",
        _post_pipeline_body(_artifacts("research"), extra=_RAG_RESULT_STANZA),
    )

    outcome = _run(app, RESEARCH)

    assert outcome.status == "pass", outcome.detail
    assert "research_pathway_report.txt" in outcome.artifacts
    assert outcome.warnings == []
    assert outcome.to_dict()["warnings"] == []


# ---------------------------------------------------------------------------
# Stage-1 failure reported the way the app reports it: st.error then st.stop.
# ---------------------------------------------------------------------------
def test_stage1_error_and_stop_is_a_classified_failure(tmp_path: Path) -> None:
    app = _write_app(
        tmp_path,
        "stage1_error",
        '''
if submitted:
    st.error("Stage 1 extraction failed: the LLM returned invalid JSON after 3 repair attempts.")
    st.stop()
st.write("never reached on submit")
''',
    )

    outcome = _run(app, STRICT)

    assert outcome.status == "fail"
    assert outcome.stage == "stage1"
    assert outcome.failure_kind == "llm"
    assert "invalid JSON" in outcome.message
    assert "invalid JSON" in outcome.detail
    assert outcome.artifacts == {}


def test_network_wording_outranks_llm_wording(tmp_path: Path) -> None:
    """An LLM call that cannot open a socket is a network failure, not an LLM one."""

    app = _write_app(
        tmp_path,
        "stage1_network",
        '''
if submitted:
    st.error("LLM request failed: ConnectionError contacting the UniProt service.")
    st.stop()
''',
    )

    outcome = _run(app, STRICT)

    assert outcome.status == "fail"
    assert outcome.failure_kind == "network"


def test_empty_pathway_is_no_reactions_not_unknown(tmp_path: Path) -> None:
    app = _write_app(
        tmp_path,
        "empty_pathway",
        '''
if submitted:
    st.session_state["pipeline_ready"] = True
    st.session_state["stage_one"] = {"processes": {"reactions": []}}
    st.session_state["final_payload"] = {"processes": {"reactions": []}, "entities": {}}
''',
    )

    outcome = _run(app, STRICT)

    assert outcome.status == "fail"
    assert outcome.failure_kind == "no_reactions"
    assert outcome.stage == "stage1"
    # Even a failed run hands back what the app did produce.
    assert "stage1_payload.json" in outcome.artifacts


def test_paper_without_full_text_never_starts_the_app(tmp_path: Path) -> None:
    app = _write_app(tmp_path, "unused", _post_pipeline_body(_artifacts("pathwhiz")))

    outcome = _run(app, STRICT, paper=_Paper(paper_id="PMC9", full_text="   "))

    assert outcome.status == "fail"
    assert outcome.failure_kind == "no_reactions"
    assert outcome.paper_id == "PMC9"


# ---------------------------------------------------------------------------
# The app's one real, expected refusal.
# ---------------------------------------------------------------------------
def test_ambiguous_review_scope_is_read_from_pipeline_error(tmp_path: Path) -> None:
    app = _write_app(
        tmp_path,
        "ambiguous",
        '''
if submitted:
    st.session_state["pipeline_ready"] = False
    st.session_state["pipeline_error"] = {
        "status": "ambiguous_review_scope",
        "message": "multi_example_review detected with no selected_example.",
        "candidate_examples": ["menaquinone biosynthesis", "heme biosynthesis"],
    }
    st.error("Ambiguous review scope: name the example you want, then re-run.")
    st.stop()
''',
    )

    outcome = _run(app, STRICT)

    assert outcome.status == "fail"
    assert outcome.failure_kind == "ambiguous_review_scope"
    assert outcome.issue_codes == ["ambiguous_review_scope"]
    assert "multi_example_review" in outcome.message
    assert "heme biosynthesis" in outcome.detail


# ---------------------------------------------------------------------------
# Contract / gate failures -- the codes are what the aggregator ranks on.
# ---------------------------------------------------------------------------
def test_contract_report_issue_codes_are_extracted(tmp_path: Path) -> None:
    app = _write_app(
        tmp_path,
        "contract_fail",
        _post_pipeline_body(
            _artifacts(
                "pathwhiz",
                post_extraction_contract_report={
                    "ok": False,
                    "stage": "post_extraction",
                    "errors": [
                        {
                            "code": "reaction_enzyme_unresolved",
                            "pointer": "/processes/reactions/0/enzymes/0",
                            "message": "enzyme has no external identity",
                        },
                        {
                            "code": "generated_wrapper_missing_components",
                            "pointer": "/entities/protein_complexes/0",
                            "message": "wrapper has no components",
                        },
                    ],
                    "warnings": [],
                },
            )
        ),
    )

    outcome = _run(app, STRICT)

    assert outcome.status == "fail"
    assert outcome.stage == "post_pipeline"
    assert outcome.failure_kind == "contract"
    assert outcome.issue_codes == [
        "reaction_enzyme_unresolved",
        "generated_wrapper_missing_components",
    ]
    assert outcome.counts["contract_errors"] == 2
    written = json.loads(outcome.artifacts["contract_reports.json"])
    assert written["post_extraction_contract_report"]["ok"] is False
    assert "/processes/reactions/0/enzymes/0" in outcome.detail


def test_gate_failure_without_codes_gets_groupable_synthetic_codes(tmp_path: Path) -> None:
    """Gate errors carry ``path``/``reason`` and no code, so one is derived.

    The derived code must collapse every instance of one rule onto one code --
    otherwise the ranked fix-list is a list of entity names, not of bugs.
    """

    app = _write_app(
        tmp_path,
        "gate_fail",
        _post_pipeline_body(
            _artifacts(
                "pathwhiz",
                normalization_gate_failed=True,
                gate_fail_report={
                    "status": "failed",
                    "stage": "post_normalization_hard_gates",
                    "error": "Hard-gate validation failed after normalization.",
                    "errors": [
                        {
                            "path": "/entities/protein_complexes/0/name",
                            "reason": "Forbidden complex reference detected: thyroglobulin:2-aminoacrylic acid",
                        },
                        {
                            "path": "/entities/protein_complexes/3/name",
                            "reason": "Forbidden complex reference detected: beta-hydroxymyristoyl-ACP",
                        },
                    ],
                },
            )
        ),
    )

    outcome = _run(app, STRICT)

    assert outcome.status == "fail"
    assert outcome.failure_kind == "contract"
    assert outcome.issue_codes == ["gate.forbidden_complex_reference_detected"]
    assert outcome.counts["gate_errors"] == 2
    # Gate errors are not also reported as contract errors: the ranked fix-list
    # would otherwise double every gate failure.
    assert "contract_errors" not in outcome.counts
    gate = json.loads(outcome.artifacts["gate_fail_report.json"])
    assert len(gate["errors"]) == 2


def test_missing_post_pipeline_artifacts_reads_the_stage_contract_error(tmp_path: Path) -> None:
    """A StageContractError is caught and rendered by the app, storing nothing."""

    app = _write_app(
        tmp_path,
        "stage_contract_error",
        f'''
if submitted:
    st.session_state["pipeline_ready"] = True
    st.session_state["stage_one"] = {_lit(_STAGE1)}
    st.session_state["final_payload"] = {_lit(_MERGED)}
if st.session_state.get("pipeline_ready"):
    if st.button("Run audit and DB mapping", key="pwml_generate_btn"):
        st.error("Audit and DB mapping stopped at the post_extraction stage boundary.")
''',
    )

    outcome = _run(app, STRICT)

    assert outcome.status == "fail"
    assert outcome.stage == "post_pipeline"
    assert outcome.failure_kind == "contract"
    assert "stage boundary" in outcome.message
    assert "merged_payload.json" in outcome.artifacts


# ---------------------------------------------------------------------------
# run_one must never raise.
# ---------------------------------------------------------------------------
def test_app_script_raising_becomes_a_crash_outcome(tmp_path: Path) -> None:
    app = _write_app(
        tmp_path,
        "boom",
        '''
if submitted:
    raise RuntimeError("streamlit_app blew up mid-render")
''',
    )

    outcome = _run(app, STRICT)

    assert isinstance(outcome, RunOutcome)
    assert outcome.status == "error"
    assert outcome.failure_kind == "crash"
    assert "streamlit_app blew up mid-render" in outcome.detail


def test_app_raising_on_first_render_is_reported_not_raised(tmp_path: Path) -> None:
    target = tmp_path / "import_boom.py"
    target.write_text(
        'import streamlit as st\nraise ImportError("no module named t2pw")\n',
        encoding="utf-8",
    )

    outcome = _run(target, STRICT)

    assert outcome.status == "error"
    assert "no module named t2pw" in outcome.detail
    assert outcome.stage == "app_load"


def test_a_hanging_extraction_becomes_a_timeout_not_a_crash(tmp_path: Path) -> None:
    """A wedged LLM call is the single most likely overnight failure mode."""

    app = _write_app(
        tmp_path,
        "hang",
        '''
if submitted:
    import time
    time.sleep(4)
    st.session_state["pipeline_ready"] = True
''',
    )

    outcome = run_one(PAPER, STRICT, app_path=app, timeout=20.0, app_timeout=1.0)

    assert outcome.status == "timeout"
    assert outcome.stage == "input"
    assert "time budget" in outcome.message
    assert "timed out" in outcome.detail.lower()
    # One bucket for every hang: the parent's synthesized timeout row uses this
    # same kind, so the ranked fix-list does not split timeouts across "timeout"
    # and "unknown" depending on who noticed.
    assert outcome.failure_kind == KIND_TIMEOUT


# ---------------------------------------------------------------------------
# detail is a pointer, not the record.
# ---------------------------------------------------------------------------
def test_detail_is_capped_and_says_what_was_dropped() -> None:
    """``detail`` rides one stdout line, then the manifest, then RESULT.txt.

    Full IR + gate reports run to 70-100 KB, which bloats all three -- and the
    same reports are already written as their own artifact files.
    """

    outcome = RunOutcome(
        paper_id="PMC1",
        mode=MODE_STRICT,
        status="fail",
        detail="x" * (DETAIL_LIMIT + 5000),
        artifacts={"pwml_ir.json": "{}", "gate_fail_report.json": "{}"},
    )

    capped = outcome.to_dict()["detail"]
    assert len(capped) < DETAIL_LIMIT + 500
    assert capped.startswith("x" * 100)
    assert "detail truncated" in capped
    assert "5000 more character(s) dropped" in capped
    # Nothing is lost: the marker names the files the content lives in.
    assert "pwml_ir.json" in capped
    assert "gate_fail_report.json" in capped
    # The outcome object itself is untouched, so nothing else is misled.
    assert len(outcome.detail) == DETAIL_LIMIT + 5000


def test_a_short_detail_is_passed_through_untouched() -> None:
    outcome = RunOutcome(detail="Traceback: boom")
    assert outcome.to_dict()["detail"] == "Traceback: boom"


def test_missing_app_path_is_reported_not_raised(tmp_path: Path) -> None:
    outcome = _run(tmp_path / "does_not_exist.py", STRICT)

    assert outcome.status == "error"
    assert outcome.failure_kind == "crash"
    assert "app script not found" in outcome.message


def test_garbage_paper_object_is_reported_not_raised() -> None:
    """The driver is handed whatever the fetch stage produced, including junk."""

    outcome = run_one(object(), "strict", timeout=5.0, app_timeout=5.0)

    assert isinstance(outcome, RunOutcome)
    assert outcome.status in {"fail", "error"}
    assert outcome.paper_id == "(unknown-paper)"


# ---------------------------------------------------------------------------
# Mode resolution.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("given", "expected"),
    [
        (STRICT, MODE_STRICT),
        ("strict", MODE_STRICT),
        ("pathwhiz", MODE_STRICT),
        ("", MODE_STRICT),
        (None, MODE_STRICT),
        (RESEARCH, MODE_RESEARCH),
        ("research", MODE_RESEARCH),
        ("Research (relaxed)", MODE_RESEARCH),
    ],
)
def test_unknown_mode_never_silently_relaxes_the_gates(
    tmp_path: Path, given: object, expected: str
) -> None:
    app = _write_app(tmp_path, "mode_probe", _post_pipeline_body(_artifacts("pathwhiz")))

    outcome = run_one(PAPER, given, app_path=app, timeout=60.0, app_timeout=30.0)

    assert outcome.mode == expected


# ---------------------------------------------------------------------------
# The extraction-focus box (unkeyed in the app, so addressed by label).
#
# The first overnight run failed nearly every topic-fetched paper with
# ``ambiguous_review_scope``: topic search returns reviews, and the app refuses to
# extract from a multi-example review unless a target example is named. The app's
# own remedy is this box, and the runner already knows the topic -- so it types it
# in. These tests pin that wiring, because it is invisible from the outside: the
# only proof the focus arrived is what the app script received.
# ---------------------------------------------------------------------------
#: Echoes the widget value the app script actually saw into an artifact the
#: outcome carries, so the assertion is end-to-end rather than on driver internals.
_FOCUS_ECHO = '''
    st.session_state["stage_one"] = dict(
        st.session_state["stage_one"], extraction_focus=user_task_context
    )
'''


def _observed_focus(outcome: RunOutcome) -> str:
    return json.loads(outcome.artifacts["stage1_payload.json"])["extraction_focus"]


@pytest.mark.parametrize(
    ("topic", "organism", "expected"),
    [
        # The shape the app's Stage-0 scope resolver reads best.
        ("lipid A biosynthesis", "Escherichia coli", "lipid A biosynthesis in Escherichia coli"),
        # No organism on the record: the topic alone is still a scope.
        ("menaquinone biosynthesis", "", "menaquinone biosynthesis"),
        # The topic already names the organism; appending it again would read as
        # "... in Escherichia coli in Escherichia coli".
        (
            "lipid A biosynthesis in Escherichia coli",
            "Escherichia coli",
            "lipid A biosynthesis in Escherichia coli",
        ),
    ],
)
def test_the_paper_topic_is_typed_into_the_unkeyed_focus_box(
    tmp_path: Path, topic: str, organism: str, expected: str
) -> None:
    app = _write_app(
        tmp_path,
        "focus_box",
        _post_pipeline_body(_artifacts("pathwhiz"), extra=_FOCUS_ECHO),
    )

    outcome = _run(app, STRICT, paper=_Paper(topic=topic, organism=organism))

    assert _observed_focus(outcome) == expected
    # Naming the scope is not a warning-worthy compromise; it is the normal path.
    assert outcome.warnings == []


def test_a_pinned_paper_leaves_the_focus_box_empty(tmp_path: Path) -> None:
    """A pinned paper has no topic, and inventing one would bias extraction.

    ``topic == ""`` means a human named this paper by id. There is no searched-for
    phrase that is known to be true of it, so the box stays empty rather than being
    filled with a guess the extraction would then be steered by.
    """

    app = _write_app(
        tmp_path,
        "focus_pinned",
        _post_pipeline_body(_artifacts("pathwhiz"), extra=_FOCUS_ECHO),
    )

    outcome = _run(app, STRICT, paper=_Paper(paper_id="PMC42", topic="", organism="Escherichia coli"))

    assert _observed_focus(outcome) == ""
    assert outcome.warnings == []


def test_a_missing_focus_box_degrades_to_a_warning(tmp_path: Path) -> None:
    """The box has no key, so it can only be found by label -- and labels move.

    Losing it costs the run its *scope*, not its life: the run continues (and may
    well still succeed on a single-example paper), but the manifest has to say the
    focus never arrived, otherwise a night of ``ambiguous_review_scope`` failures
    looks like an app bug again.
    """

    app = _write_app(
        tmp_path,
        "focus_missing",
        _post_pipeline_body(
            _artifacts("pathwhiz"),
            pwml={"ok": True, "_xml": "<pathway/>", "pwml_ir": {"pathway": {}}},
        ),
        focus_box=False,
    )

    outcome = _run(app, STRICT, paper=_Paper(topic="lipid A biosynthesis"))

    assert outcome.status == "pass", outcome.detail
    assert outcome.warnings == [WARN_NO_FOCUS_BOX]
    assert outcome.to_dict()["warnings"] == [WARN_NO_FOCUS_BOX]


def test_a_missing_focus_box_is_silent_when_there_was_nothing_to_type(tmp_path: Path) -> None:
    """No topic means nothing was lost, so there is nothing to warn about."""

    app = _write_app(
        tmp_path,
        "focus_missing_pinned",
        _post_pipeline_body(
            _artifacts("pathwhiz"),
            pwml={"ok": True, "_xml": "<pathway/>", "pwml_ir": {"pathway": {}}},
        ),
        focus_box=False,
    )

    outcome = _run(app, STRICT, paper=_Paper(topic=""))

    assert outcome.status == "pass", outcome.detail
    assert outcome.warnings == []


# ---------------------------------------------------------------------------
# A nested runtime_schema_report is information, never a verdict.
#
# Shapes copied from the real run that this regression protects:
# runs/2026-07-27_1623/papers/PMC12444477__the-regulation-of-lipid-a-biosynthesis/
# research/contract_reports.json -- 6 reactions, 29 entities, 0 gate errors, every
# real contract report ok=True/errors=0, and the run was nevertheless reported
# FAIL/contract because the nested report says ok=False/errors=1.
# ---------------------------------------------------------------------------
_SCHEMA_FINDING = {
    "code": "entity_missing_mapping_meta",
    "pointer": "/entities/subcellular_locations/1/mapping_meta",
    "message": "Mapped entity with a name is missing mapping_meta.",
    "expected": "object",
}


def _nested_schema_report(boundary: str) -> dict:
    """``RUNTIME_SCHEMA_MODE="report"`` keeps its own ok/errors, informationally."""

    return {
        "ok": False,
        "boundary": boundary,
        "mode": "report",
        "errors": [_SCHEMA_FINDING],
        "warnings": [],
        "summary": {"error_count": 1, "warning_count": 0},
    }


def _parent_contract_report(boundary: str) -> dict:
    """The stage's actual verdict: the finding arrived here as a WARNING."""

    return {
        "stage": boundary,
        "contract_type": "semantic",
        "effect_on_failure": "annotate_only",
        "runtime_schema_report": _nested_schema_report(boundary),
        "errors": [],
        "warnings": [_SCHEMA_FINDING],
        "ok": True,
        "export_mode": "research",
        "summary": {"error_count": 0, "warning_count": 1},
    }


_SCHEMA_ONLY_ARTIFACTS = {
    "post_normalization_contract_report": _parent_contract_report("post_normalization"),
    "post_audit_contract_report": _parent_contract_report("post_audit"),
    # A standalone top-level runtime-schema report is not blocking either.
    "pre_export_runtime_schema_report": _nested_schema_report("pre_export"),
}


def test_a_nested_runtime_schema_report_never_fails_a_run(tmp_path: Path) -> None:
    app = _write_app(
        tmp_path,
        "nested_schema_report",
        _post_pipeline_body(
            _artifacts("research", **_SCHEMA_ONLY_ARTIFACTS),
            extra=_RAG_RESULT_STANZA,
        ),
    )

    outcome = _run(app, RESEARCH)

    # The run had succeeded. Failing it threw the research deliverable away.
    assert outcome.status == "pass", outcome.detail
    assert outcome.failure_kind == ""
    assert "research_pathway_report.txt" in outcome.artifacts
    assert "contract_errors" not in outcome.counts
    assert "entity_missing_mapping_meta" not in outcome.issue_codes

    # ...but the findings stay visible, as warnings.
    assert outcome.counts["runtime_schema_findings"] == 3
    warned = outcome.warnings
    assert len(warned) == 3
    assert all(item.startswith(WARN_RUNTIME_SCHEMA_PREFIX) for item in warned)
    assert any("post_normalization_contract_report.runtime_schema_report" in i for i in warned)
    assert any("pre_export_runtime_schema_report" in i for i in warned)
    assert all("entity_missing_mapping_meta" in item for item in warned)
    assert outcome.to_dict()["warnings"] == warned

    # And every report -- nested included -- is still on disk to read.
    written = json.loads(outcome.artifacts["contract_reports.json"])
    assert written["post_audit_contract_report.runtime_schema_report"]["ok"] is False
    assert written["post_audit_contract_report"]["ok"] is True


def test_many_schema_findings_are_capped_but_counted_in_full(tmp_path: Path) -> None:
    """One bad bucket can produce dozens; the warnings list rides the manifest."""

    many = dict(_nested_schema_report("pre_export"))
    many["errors"] = [
        dict(_SCHEMA_FINDING, pointer=f"/entities/subcellular_locations/{i}/mapping_meta")
        for i in range(RUNTIME_SCHEMA_WARN_LIMIT + 2)
    ]

    app = _write_app(
        tmp_path,
        "many_schema_findings",
        _post_pipeline_body(
            _artifacts("research", pre_export_runtime_schema_report=many),
            extra=_RAG_RESULT_STANZA,
        ),
    )

    outcome = _run(app, RESEARCH)

    assert outcome.status == "pass", outcome.detail
    # The count is exact even though the list is not.
    assert outcome.counts["runtime_schema_findings"] == RUNTIME_SCHEMA_WARN_LIMIT + 2
    assert len(outcome.warnings) == RUNTIME_SCHEMA_WARN_LIMIT + 1
    assert outcome.warnings[-1].endswith("all of them are in contract_reports.json")
    assert "and 2 more finding(s)" in outcome.warnings[-1]


def test_a_real_contract_error_still_fails_alongside_schema_findings(tmp_path: Path) -> None:
    """Only the top-level report's own ``errors`` decide; the nested one is noise."""

    failing_parent = _parent_contract_report("post_normalization")
    failing_parent["ok"] = False
    failing_parent["errors"] = [
        {
            "code": "reaction_enzyme_unresolved",
            "pointer": "/processes/reactions/0/enzymes/0",
            "message": "enzyme has no external identity",
        }
    ]

    app = _write_app(
        tmp_path,
        "real_contract_error",
        _post_pipeline_body(
            _artifacts(
                "research",
                post_normalization_contract_report=failing_parent,
                post_audit_contract_report=_parent_contract_report("post_audit"),
            )
        ),
    )

    outcome = _run(app, RESEARCH)

    assert outcome.status == "fail"
    assert outcome.failure_kind == "contract"
    assert outcome.issue_codes == ["reaction_enzyme_unresolved"]
    assert outcome.counts["contract_errors"] == 1
    assert outcome.counts["runtime_schema_findings"] == 2
    assert any(item.startswith(WARN_RUNTIME_SCHEMA_PREFIX) for item in outcome.warnings)


# ---------------------------------------------------------------------------
# Structural guards are contract failures, not LLM failures.
#
# From the real run: PMC13278307/research was filed as failure_kind="llm" with
# "Extraction boundary failed: Payload must include a processes object." That is a
# StageContractError with code ``processes_required``, a STRUCTURAL_GUARD_CODES
# member that correctly aborts in both modes -- there is genuinely no pathway to
# review. Only the label was wrong, and it was wrong because the app prints a
# "usually a transient LLM failure" hint right next to it.
# ---------------------------------------------------------------------------
_LLM_HINT = (
    "Stage 0 failed: empty_reply - The model returned an empty reply (0 chars). "
    "Re-run (usually a transient LLM failure), or type the pathway/organism into "
    "the focus box so the pipeline can proceed without the preprocessor."
)


def test_structural_guard_wording_is_a_contract_failure_with_its_code(tmp_path: Path) -> None:
    app = _write_app(
        tmp_path,
        "structural_guard",
        f'''
if submitted:
    st.error("Extraction boundary failed: Payload must include a processes object.")
    st.warning({_lit(_LLM_HINT)})
    st.stop()
''',
    )

    outcome = _run(app, RESEARCH)

    assert outcome.status == "fail"
    assert outcome.stage == "stage1"
    assert outcome.failure_kind == "contract"
    assert outcome.issue_codes == ["processes_required"]
    assert "Payload must include a processes object" in outcome.message


@pytest.mark.parametrize(
    ("message", "code"),
    [
        ("Extraction boundary failed: Payload must be a dict.", "invalid_payload"),
        (
            "Extraction boundary failed: Payload must include an entities object.",
            "entities_required",
        ),
        ("Extraction boundary failed: Entity rows must be objects.", "entity_not_object"),
        ("Extraction boundary failed: Process rows must be objects.", "process_not_object"),
    ],
)
def test_every_structural_guard_message_maps_to_its_code(
    tmp_path: Path, message: str, code: str
) -> None:
    app = _write_app(
        tmp_path,
        "structural_guard_variants",
        f'''
if submitted:
    st.error({_lit(message)})
    st.stop()
''',
    )

    outcome = _run(app, STRICT)

    assert outcome.failure_kind == "contract"
    assert outcome.issue_codes == [code]


def test_genuine_llm_wording_is_not_reclassified_as_a_contract_failure(tmp_path: Path) -> None:
    """The other direction: an empty reply with no structural guard is still LLM.

    The hint that mislabelled PMC13278307 is present here *on its own*. Without a
    structural-guard sentence there is no contract evidence, and blaming the
    contract layer for a model that returned nothing would be the mirror-image bug.
    """

    app = _write_app(
        tmp_path,
        "genuine_llm",
        f'''
if submitted:
    st.error("Extraction failed: the model returned an empty reply after 3 attempts.")
    st.warning({_lit(_LLM_HINT)})
    st.stop()
''',
    )

    outcome = _run(app, RESEARCH)

    assert outcome.status == "fail"
    assert outcome.failure_kind == "llm"
    assert outcome.issue_codes == []


def test_dict_and_candidate_paper_shapes_are_both_accepted(tmp_path: Path) -> None:
    app = _write_app(
        tmp_path,
        "dict_paper",
        _post_pipeline_body(
            _artifacts("pathwhiz"),
            pwml={"ok": True, "_xml": "<pathway/>", "pwml_ir": {"pathway": {}}},
        ),
    )

    from types import SimpleNamespace

    as_dict = _run(app, STRICT, paper={"paper_id": "PMC7", "full_text": "prose"})
    as_candidate = _run(app, STRICT, paper=SimpleNamespace(id="PMC8", full_text="prose"))

    assert as_dict.paper_id == "PMC7" and as_dict.status == "pass"
    assert as_candidate.paper_id == "PMC8" and as_candidate.status == "pass"


# ---------------------------------------------------------------------------
# Stage-0 scope reconciliation, driven through the real app boundary.
#
# ``t2pw.rag.eligibility.apply_stage0_observation`` encodes the rule that the
# batch's requested pathway/organism is immutable and that Stage 0 may only add
# observations. Its production caller is ``driver._reconcile_stage0_scope``, which
# runs right after stage 1 succeeds and reads the app's own
# ``st.session_state["pathway_context"]``. These tests exercise that caller by
# driving a fixture app; calling the helper directly would prove nothing about
# whether the production path is wired.
# ---------------------------------------------------------------------------
def _stage0_stanza(context: object) -> str:
    """Make the fixture app store a Stage-0 context, as the real app does."""
    return f'    st.session_state["pathway_context"] = {_lit(context)}\n'


def _scoped_paper(pathway: str, organism: str, paper_id: str = "PMC1") -> _Paper:
    paper = _Paper(paper_id=paper_id, topic=pathway, organism=organism)
    paper.requested_pathway = pathway
    paper.requested_organism = organism
    return paper


def test_a_contradictory_stage0_pathway_persists_a_scope_conflict(tmp_path: Path) -> None:
    app = _write_app(
        tmp_path,
        "stage0_pathway_conflict",
        _post_pipeline_body(
            _artifacts("research"),
            extra=_stage0_stanza(
                {
                    "pathway_name": "cholesterol biosynthesis",
                    "likely_organism": "Homo sapiens",
                }
            ),
        ),
    )
    paper = _scoped_paper("lipid A biosynthesis", "Escherichia coli")

    outcome = _run(app, RESEARCH, paper)

    # An explicit, persisted outcome: not a failure, and not a silent pass.
    assert outcome.status == "scope_conflict"
    assert outcome.scope_conflicted is True
    assert outcome.failure_kind == ""
    assert "scope_conflict" in outcome.issue_codes
    assert outcome.counts["scope_conflicts"] == 2
    assert any("cholesterol biosynthesis" in c for c in outcome.scope_conflicts)
    assert any("Homo sapiens" in c for c in outcome.scope_conflicts)

    # The REQUEST is unchanged, and is written down next to the observation.
    artifact = json.loads(outcome.artifacts["scope_conflict.json"])
    assert artifact["requested"]["requested_pathway"] == "lipid A biosynthesis"
    assert artifact["requested"]["requested_organism"] == "Escherichia coli"
    assert artifact["observed"]["observed_pathways"] == ["cholesterol biosynthesis"]
    assert artifact["observed"]["observed_organisms"] == ["Homo sapiens"]

    # And it reaches the manifest row.
    row = outcome.to_dict()
    assert row["status"] == "scope_conflict"
    assert row["scope_conflicts"] == outcome.scope_conflicts
    assert row["observed_context"]["observed_pathways"] == ["cholesterol biosynthesis"]


def test_a_contradictory_stage0_organism_alone_is_a_scope_conflict(tmp_path: Path) -> None:
    app = _write_app(
        tmp_path,
        "stage0_organism_conflict",
        _post_pipeline_body(
            _artifacts("research"),
            extra=_stage0_stanza(
                {
                    "pathway_name": "lipid A biosynthesis",
                    "likely_organism": "Homo sapiens",
                }
            ),
        ),
    )
    paper = _scoped_paper("lipid A biosynthesis", "Escherichia coli")

    outcome = _run(app, RESEARCH, paper)

    assert outcome.status == "scope_conflict"
    assert len(outcome.scope_conflicts) == 1
    assert "Homo sapiens" in outcome.scope_conflicts[0]


def test_a_scope_conflict_is_not_counted_as_a_failure(tmp_path: Path) -> None:
    """The paper is not the requested paper. Nothing is broken, so nothing failed."""
    from t2pw.batch import report as batch_report

    app = _write_app(
        tmp_path,
        "stage0_conflict_not_failure",
        _post_pipeline_body(
            _artifacts("research"),
            extra=_stage0_stanza({"pathway_name": "cholesterol biosynthesis"}),
        ),
    )
    outcome = _run(
        app, RESEARCH, _scoped_paper("lipid A biosynthesis", "Escherichia coli")
    )

    status = batch_report._norm_status(outcome.status)
    assert status == batch_report.STATUS_INELIGIBLE
    assert batch_report._is_failure(status) is False
    run = batch_report._to_run(outcome.to_dict())
    assert run.ineligible is True
    assert run.failed is False


def test_an_agreeing_stage0_reading_records_observations_and_runs_on(
    tmp_path: Path,
) -> None:
    app = _write_app(
        tmp_path,
        "stage0_agrees",
        _post_pipeline_body(
            _artifacts("research"),
            extra=_stage0_stanza(
                {
                    "pathway_name": "lipid A biosynthesis",
                    "likely_organism": "Escherichia coli",
                    "aliases": ["Raetz pathway"],
                    "candidate_examples": [{"name": "E. coli K-12"}],
                }
            ),
        ),
    )
    outcome = _run(
        app, RESEARCH, _scoped_paper("lipid A biosynthesis", "Escherichia coli")
    )

    assert outcome.status != "scope_conflict"
    assert outcome.scope_conflicts == []
    # Stage 0's reading is recorded as an OBSERVATION regardless.
    assert outcome.observed_context["observed_pathways"] == ["lipid A biosynthesis"]
    assert outcome.observed_context["observed_organisms"] == ["Escherichia coli"]
    assert outcome.observed_context["aliases"] == ["Raetz pathway"]
    assert outcome.observed_context["ambiguities"] == ["E. coli K-12"]
    assert outcome.counts["stage0_observed_pathways"] == 1
    assert "scope_conflict.json" not in outcome.artifacts


def test_a_related_species_from_stage0_is_not_a_scope_conflict(tmp_path: Path) -> None:
    """Genus-level is permitted evidence, so it must not stop a run."""
    app = _write_app(
        tmp_path,
        "stage0_genus_level",
        _post_pipeline_body(
            _artifacts("research"),
            extra=_stage0_stanza(
                {
                    "pathway_name": "lipid A biosynthesis",
                    "likely_organism": "Escherichia fergusonii",
                }
            ),
        ),
    )
    outcome = _run(
        app, RESEARCH, _scoped_paper("lipid A biosynthesis", "Escherichia coli")
    )

    assert outcome.status != "scope_conflict"
    assert outcome.scope_conflicts == []
    assert outcome.observed_context["observed_organisms"] == ["Escherichia fergusonii"]


def test_a_pinned_paper_has_no_request_to_contradict(tmp_path: Path) -> None:
    app = _write_app(
        tmp_path,
        "stage0_pinned",
        _post_pipeline_body(
            _artifacts("research"),
            extra=_stage0_stanza({"pathway_name": "cholesterol biosynthesis"}),
        ),
    )
    # A pinned paper has topic == "": a human chose it, so nothing to conflict with.
    outcome = _run(app, RESEARCH, _Paper(topic="", organism=""))

    assert outcome.status != "scope_conflict"
    assert outcome.scope_conflicts == []


def test_an_app_that_stores_no_stage0_context_changes_nothing(tmp_path: Path) -> None:
    app = _write_app(
        tmp_path, "stage0_absent", _post_pipeline_body(_artifacts("research"))
    )

    outcome = _run(
        app, RESEARCH, _scoped_paper("lipid A biosynthesis", "Escherichia coli")
    )

    assert outcome.status != "scope_conflict"
    assert outcome.observed_context == {}
    assert "scope_conflicts" not in outcome.to_dict()


def test_the_conflict_can_be_downgraded_to_a_warning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``RAG_ELIGIBILITY_STAGE0_CONFLICT_ABORTS=false`` annotates and carries on."""
    monkeypatch.setenv("RAG_ELIGIBILITY_STAGE0_CONFLICT_ABORTS", "false")
    app = _write_app(
        tmp_path,
        "stage0_conflict_warn",
        _post_pipeline_body(
            _artifacts("research"),
            extra=_stage0_stanza({"pathway_name": "cholesterol biosynthesis"}),
        ),
    )
    outcome = _run(
        app, RESEARCH, _scoped_paper("lipid A biosynthesis", "Escherichia coli")
    )

    assert outcome.status != "scope_conflict"
    assert outcome.scope_conflicts  # still recorded
    assert any("stage-0 scope conflict" in w for w in outcome.warnings)
    assert "scope_conflict.json" in outcome.artifacts


def test_a_legacy_plan_record_still_supplies_the_requested_scope(tmp_path: Path) -> None:
    """Pre-2026-07-29 plans spell it ``topic`` / ``organism``; both are read."""
    app = _write_app(
        tmp_path,
        "stage0_legacy_record",
        _post_pipeline_body(
            _artifacts("research"),
            extra=_stage0_stanza(
                {
                    "pathway_name": "cholesterol biosynthesis",
                    "likely_organism": "Homo sapiens",
                }
            ),
        ),
    )
    # A dict shaped exactly like an old plan record: no requested_* keys at all.
    legacy = {
        "paper_id": "PMC1",
        "full_text": "Some pathway prose.",
        "topic": "lipid A biosynthesis",
        "organism": "Escherichia coli",
    }
    outcome = _run(app, RESEARCH, legacy)

    assert outcome.status == "scope_conflict"
    artifact = json.loads(outcome.artifacts["scope_conflict.json"])
    assert artifact["requested"]["requested_pathway"] == "lipid A biosynthesis"
    assert artifact["requested"]["requested_organism"] == "Escherichia coli"
