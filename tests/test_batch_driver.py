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
    STRICT,
    WARN_NO_RESEARCH_REPORT,
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
_PREAMBLE = '''\
import streamlit as st
from types import SimpleNamespace

st.radio("Export mode", ["Strict PWML", "Research (relaxed)"], key="export_mode_radio")
st.radio("Input mode", ["Paste text", "Upload PDF", "Text + PDF"], key="input_mode_radio")
with st.form("pwml_pipeline"):
    st.text_area("Paste pathway description:", key="pathway_text_0")
    submitted = st.form_submit_button("Run pipeline")
'''


def _write_app(tmp_path: Path, name: str, body: str) -> Path:
    """Write a fixture app script. utf-8 always: entity names are not ASCII."""

    target = tmp_path / f"{name}.py"
    target.write_text(_PREAMBLE + body, encoding="utf-8")
    return target


def _lit(value: object) -> str:
    """Embed a Python value in a generated script.

    ``repr``, not ``json.dumps``: JSON's ``false``/``true``/``null`` are not
    Python names and the generated script would die on a ``NameError``.
    """

    return repr(value)


class _Paper:
    """A stand-in for ``t2pw.batch.fetch.BatchPaper`` (duck-typed by the driver)."""

    def __init__(self, paper_id: str = "PMC1", full_text: str = "Some pathway prose.") -> None:
        self.paper_id = paper_id
        self.full_text = full_text


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
