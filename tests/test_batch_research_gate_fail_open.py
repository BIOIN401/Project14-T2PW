"""A research run is never failed by the strict Stage-3 gate.

THE RUN THIS FILE EXISTS FOR
----------------------------
Batch run 2026-07-28_0919, paper PMC12444477 "The regulation of lipid A
biosynthesis". Its **strict** run passed all gates with 0 gate errors and exported
a 433 KB PWML. Its **research** run was filed FAIL with 30 gate errors -- while
every contract report that run produced said ``ok=true``, ``still_blocking=0``,
``research_blocked=[]``. ``batch/report.py`` ranks exactly that pair (strict PASS
+ research FAIL) as a RESEARCH-MODE DEFECT to be fixed before anything else.

Two independent defects produced it, and this module pins both:

1. ``batch/driver.py``'s gate leg was mode-blind. It read ``gate_fail_report`` /
   ``final_stage3_gate_report`` out of the artifacts dict and failed the run on
   them with no mode check -- and it runs *before* the ``mode == MODE_RESEARCH``
   branch, so a research run could never reach its own fail-open policy. Same
   class as the nested-``runtime_schema_report`` misread that was fixed earlier on
   this same paper: a non-authoritative signal used as the verdict.

2. ``process_normalizer.normalize_process_payload`` skipped
   ``drop_process_orphan_proteins`` / ``prune_disconnected_proteins`` in research
   mode and then called ``run_strict_post_normalization_gates`` with an
   unconditional ``enforce_all_proteins_connected=True``, so the gate raised on
   exactly the orphans the relaxation had chosen to keep. Measured on a payload
   with four preserved orphans: 4 gate errors before the fix, 0 after, with all
   four proteins still present.

Every research-mode assertion here has a strict-mode twin, because the point is
the *difference*: strict behaviour had to stay byte-for-byte what it was.
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
    RESEARCH,
    RESEARCH_GATE_FLAG_KEY,
    RESEARCH_GATE_WARN_LIMIT,
    STRICT,
    WARN_RESEARCH_GATE_PREFIX,
    run_one,
)
from t2pw.pipeline.export_mode import PATHWHIZ, RESEARCH as RESEARCH_MODE  # noqa: E402
from t2pw.pipeline.process_normalizer import (  # noqa: E402
    GateValidationError,
    normalize_process_payload,
    run_strict_post_normalization_gates,
)


# ---------------------------------------------------------------------------
# Real streamlit, whatever the rest of the suite did to sys.modules.
# ---------------------------------------------------------------------------
def _is_real_streamlit(module: object) -> bool:
    return isinstance(module, types.ModuleType) and hasattr(module, "__path__")


@pytest.fixture
def real_streamlit():
    """The genuine ``streamlit`` package for the duration of one test.

    Deliberately not autouse and not shared with ``tests/test_batch_driver.py``:
    the normalizer tests below need no Streamlit at all, and the other batch-driver
    module keeps its own copy so neither file can silently change the other's
    import environment. ``tests/test_stage0_scope_fallback.py`` and the
    ``test_rag_*`` modules install a ``MagicMock`` as ``sys.modules["streamlit"]``
    at import time (they must -- importing ``t2pw.app.streamlit_app`` executes a
    Streamlit script), and pytest imports every module during collection, so
    without this the ``AppTest`` used here would be a MagicMock returning
    MagicMocks and every assertion would pass for the wrong reason.
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
# A fixture app that speaks the driver's half of the app contract.
# ---------------------------------------------------------------------------
_PREAMBLE = '''\
import streamlit as st
from types import SimpleNamespace

st.radio("Export mode", ["Strict PWML", "Research (relaxed)"], key="export_mode_radio")
st.radio("Input mode", ["Paste text", "Upload PDF", "Text + PDF"], key="input_mode_radio")
with st.form("pwml_pipeline"):
    st.text_area("Paste pathway description:", key="pathway_text_0")
    st.text_area(
        "Optional extraction focus / task context",
        height=100,
        help="Use this to tell the model what pathway or scope you want extracted.",
    )
    submitted = st.form_submit_button("Run pipeline")
'''

_STAGE1 = {
    "entities": {"compounds": [{"name": "lipid IVA"}], "proteins": [{"name": "LpxK"}]},
    "processes": {"reactions": [{"name": "lipid IVA phosphorylation"}]},
}
_MERGED = {
    "entities": {
        "compounds": [{"name": "lipid IVA"}, {"name": "lipid IVA 4'-phosphate"}],
        "proteins": [{"name": "LpxK"}, {"name": "LpxL (not yet wired)"}],
    },
    "processes": {
        "reactions": [
            {
                "name": "lipid IVA phosphorylation",
                "inputs": ["lipid IVA"],
                "outputs": ["lipid IVA 4'-phosphate"],
            }
        ]
    },
}

#: Two gate errors of one rule plus two of another, shaped exactly as
#: ``run_strict_post_normalization_gates`` emits them: ``path``/``reason``, no
#: ``code``. Two of the four are the self-inflicted degree-0 kind.
_GATE_ERRORS = [
    {
        "path": "/entities/proteins/1/name",
        "reason": "Protein has degree 0 after normalization: LpxL (not yet wired)",
    },
    {
        "path": "/entities/proteins/2/name",
        "reason": "Protein has degree 0 after normalization: LpxM (not yet wired)",
    },
    {
        "path": "/processes/reactions/0/enzymes/0",
        "reason": "Unknown protein/modifier reference: LpxK holoenzyme",
    },
]

_GATE_FAIL_REPORT = {
    "status": "failed",
    "stage": "post_normalization_hard_gates",
    "error": "Hard-gate validation failed after normalization.",
    "errors": _GATE_ERRORS,
}


def _artifacts(mode_value: str, **overrides: object) -> dict:
    """Post-pipeline artifacts with EVERY contract report green.

    That combination is the whole point: in the real failing run the contract
    reports were all ``ok=true`` and the gate reports were the only thing red, so
    a driver that fails on gate reports alone is the only possible cause.
    """

    base = {
        "gate_failed": False,
        "normalization_gate_failed": True,
        "gate_fail_report": _GATE_FAIL_REPORT,
        "export_mode": mode_value,
        "research_review_flags": [],
        "research_skipped_format_rules": [],
        "research_normalization_actions": [],
        "final_mapped_db": _MERGED,
        "final_mapped": _MERGED,
        "mapping_report": {"summary": {"mapped": 3}},
        "post_extraction_contract_report": {"ok": True, "errors": [], "warnings": []},
        "post_normalization_contract_report": {"ok": True, "errors": [], "warnings": []},
    }
    base.update(overrides)
    return base


def _write_app(tmp_path: Path, name: str, artifacts: dict) -> Path:
    body = f'''
if submitted:
    st.session_state["pipeline_ready"] = True
    st.session_state["stage_one"] = {_STAGE1!r}
    st.session_state["final_payload"] = {_MERGED!r}
if st.session_state.get("pipeline_ready"):
    if st.button("Run audit and DB mapping", key="pwml_generate_btn"):
        st.session_state["post_pipeline_artifacts"] = {artifacts!r}
'''
    target = tmp_path / f"{name}.py"
    target.write_text(_PREAMBLE + body, encoding="utf-8")
    return target


class _Paper:
    paper_id = "PMC12444477"
    full_text = "Lipid A biosynthesis proceeds through the Raetz pathway."
    topic = ""
    organism = ""


def _run(app: Path, mode: str):
    return run_one(_Paper(), mode, app_path=app, timeout=120.0, app_timeout=45.0)


# ---------------------------------------------------------------------------
# 1. The driver's gate leg, which used to be mode-blind.
# ---------------------------------------------------------------------------
def test_research_run_with_gate_errors_and_green_contracts_does_not_fail(
    tmp_path: Path, real_streamlit
) -> None:
    """The exact PMC12444477 shape: gate errors present, every contract ok=true."""

    app = _write_app(tmp_path, "research_gate_errors", _artifacts("research"))

    outcome = _run(app, RESEARCH)

    assert outcome.status == "pass", outcome.detail
    assert outcome.failure_kind == ""
    assert outcome.stage == "research_report"


def test_research_gate_errors_are_counted_not_hidden(tmp_path: Path, real_streamlit) -> None:
    """Fail-open, never fail-silent: the count survives into the manifest row."""

    app = _write_app(tmp_path, "research_gate_counts", _artifacts("research"))

    outcome = _run(app, RESEARCH)

    assert outcome.counts["gate_errors"] == len(_GATE_ERRORS)
    assert outcome.counts["gate_errors_not_blocking"] == len(_GATE_ERRORS)
    assert outcome.to_dict()["counts"]["gate_errors_not_blocking"] == len(_GATE_ERRORS)


def test_research_gate_errors_become_warnings_with_their_reasons(
    tmp_path: Path, real_streamlit
) -> None:
    app = _write_app(tmp_path, "research_gate_warnings", _artifacts("research"))

    outcome = _run(app, RESEARCH)

    gate_warnings = [w for w in outcome.warnings if w.startswith(WARN_RESEARCH_GATE_PREFIX)]
    assert len(gate_warnings) == len(_GATE_ERRORS)
    assert any("degree 0" in w for w in gate_warnings)
    assert any("/processes/reactions/0/enzymes/0" in w for w in gate_warnings)


def test_research_gate_errors_land_in_the_review_flag_channel(
    tmp_path: Path, real_streamlit
) -> None:
    """``review_flags.json`` is the research deliverable a reviewer opens.

    The gate rows get their own key: ``biology_provenance_flags`` means "the
    biology is unverified" and ``format_rules_skipped`` is the app's own
    relaxation ledger, so folding 30 importer-shape errors into either would make
    both unreadable.
    """

    app = _write_app(tmp_path, "research_gate_flags", _artifacts("research"))

    outcome = _run(app, RESEARCH)

    flags = json.loads(outcome.artifacts["review_flags.json"])
    gate_rows = flags[RESEARCH_GATE_FLAG_KEY]
    assert len(gate_rows) == len(_GATE_ERRORS)
    assert flags["biology_provenance_flags"] == []
    assert flags["format_rules_skipped"] == []
    # Gate errors carry path/reason and no code; the channel is normalized.
    assert gate_rows[0]["pointer"] == "/entities/proteins/1/name"
    assert gate_rows[0]["code"] == "gate.protein_has_degree_after_normalization"
    assert gate_rows[0]["source"] == "strict_post_normalization_gate"
    assert outcome.counts["gate_review_flags"] == len(_GATE_ERRORS)


def test_research_gate_error_codes_still_reach_the_ranked_fix_list(
    tmp_path: Path, real_streamlit
) -> None:
    """Not failing is not the same as not reporting: the aggregator ranks codes."""

    app = _write_app(tmp_path, "research_gate_codes", _artifacts("research"))

    outcome = _run(app, RESEARCH)

    assert "gate.protein_has_degree_after_normalization" in outcome.issue_codes
    assert "gate.unknown_protein_modifier_reference" in outcome.issue_codes
    # ...and the reports themselves are archived verbatim.
    assert json.loads(outcome.artifacts["gate_fail_report.json"])["errors"] == _GATE_ERRORS


def test_research_warning_list_is_capped_so_one_run_cannot_flood_the_manifest(
    tmp_path: Path, real_streamlit
) -> None:
    """PMC12444477 produced 30 gate errors; 30 warning lines bury the useful one."""

    many = [
        {"path": f"/entities/proteins/{i}/name", "reason": f"Protein has degree 0 after normalization: P{i}"}
        for i in range(30)
    ]
    app = _write_app(
        tmp_path,
        "research_gate_many",
        _artifacts("research", gate_fail_report={**_GATE_FAIL_REPORT, "errors": many}),
    )

    outcome = _run(app, RESEARCH)

    assert outcome.status == "pass", outcome.detail
    assert outcome.counts["gate_errors_not_blocking"] == 30
    gate_warnings = [w for w in outcome.warnings if w.startswith(WARN_RESEARCH_GATE_PREFIX)]
    assert len(gate_warnings) == RESEARCH_GATE_WARN_LIMIT + 1
    assert gate_warnings[-1].endswith(
        "all of them are in review_flags.json and the gate report files"
    )
    # Nothing is lost: the flag channel is never capped.
    assert len(json.loads(outcome.artifacts["review_flags.json"])[RESEARCH_GATE_FLAG_KEY]) == 30


def test_a_stage3_gate_report_that_is_not_ok_also_does_not_fail_research(
    tmp_path: Path, real_streamlit
) -> None:
    """The second gate artifact the driver reads takes the same route."""

    app = _write_app(
        tmp_path,
        "research_stage3_gate",
        _artifacts(
            "research",
            normalization_gate_failed=False,
            gate_fail_report={},
            final_stage3_gate_report={
                "ok": False,
                "stage": "final_stage3",
                "errors": _GATE_ERRORS,
            },
        ),
    )

    outcome = _run(app, RESEARCH)

    assert outcome.status == "pass", outcome.detail
    assert outcome.counts["gate_errors_not_blocking"] == len(_GATE_ERRORS)


def test_a_failed_gate_with_no_itemised_errors_is_still_visible(
    tmp_path: Path, real_streamlit
) -> None:
    """``gate_failed`` alone must not become an unrecorded pass."""

    app = _write_app(
        tmp_path,
        "research_gate_no_items",
        _artifacts("research", gate_failed=True, normalization_gate_failed=False, gate_fail_report={}),
    )

    outcome = _run(app, RESEARCH)

    assert outcome.status == "pass", outcome.detail
    assert any("with no itemised errors" in w for w in outcome.warnings)


# ---------------------------------------------------------------------------
# The strict-mode firewall: none of the above may have moved strict behaviour.
# ---------------------------------------------------------------------------
def test_strict_still_fails_on_the_same_gate_errors(tmp_path: Path, real_streamlit) -> None:
    app = _write_app(tmp_path, "strict_gate_errors", _artifacts("pathwhiz"))

    outcome = _run(app, STRICT)

    assert outcome.status == "fail"
    assert outcome.failure_kind == "contract"
    assert outcome.stage == "post_pipeline"
    assert outcome.counts["gate_errors"] == len(_GATE_ERRORS)
    assert f"{len(_GATE_ERRORS)} blocking issue(s)" in outcome.message
    assert "post_normalization_hard_gates" in outcome.message


def test_strict_never_gets_the_research_bookkeeping(tmp_path: Path, real_streamlit) -> None:
    app = _write_app(tmp_path, "strict_no_research_keys", _artifacts("pathwhiz"))

    outcome = _run(app, STRICT)

    assert "gate_errors_not_blocking" not in outcome.counts
    assert "gate_review_flags" not in outcome.counts
    assert [w for w in outcome.warnings if w.startswith(WARN_RESEARCH_GATE_PREFIX)] == []


# ---------------------------------------------------------------------------
# 2. Research mode must stop manufacturing the gate errors it then reports.
# ---------------------------------------------------------------------------
def _orphan_payload(count: int = 4) -> dict:
    """One wired enzyme plus ``count`` proteins the paper states but never wires.

    Four orphans on purpose: that is the number of degree-0 errors PMC12444477's
    research run carried in 2026-07-28_0919 out of its 30.
    """

    payload = {
        "entities": {
            "species": [{"name": "Escherichia coli"}],
            "compounds": [{"name": "ATP"}, {"name": "ADP"}],
            "proteins": [{"name": "LpxK", "species": "Escherichia coli", "uniprot_id": "P27300"}],
        },
        "processes": {
            "reactions": [
                {
                    "name": "lipid IVA phosphorylation",
                    "inputs": ["ATP"],
                    "outputs": ["ADP"],
                    "enzymes": [{"entity": "LpxK", "entity_type": "protein"}],
                }
            ]
        },
    }
    for index in range(count):
        payload["entities"]["proteins"].append(
            {
                "name": f"Orphan factor {index}",
                "species": "Escherichia coli",
                "uniprot_id": f"Q0000{index}",
            }
        )
    return payload


def test_research_mode_no_longer_fails_the_gate_it_relaxed() -> None:
    """Relax the passes, relax the rule those passes existed to satisfy."""

    data, report = normalize_process_payload(_orphan_payload(), mode=RESEARCH_MODE)

    assert report["gate"]["ok"] is True
    assert report["gate"]["errors"] == []
    assert report["summary"]["gate_error_count"] == 0
    # The relaxation is recorded, not silent.
    types_seen = {str(a.get("type")) for a in report["actions"] if isinstance(a, dict)}
    assert "research_mode_destructive_cleanup_skipped" in types_seen
    assert "research_mode_connectivity_gate_not_enforced" in types_seen


def test_the_four_orphans_are_still_preserved() -> None:
    """Relaxing the gate must not have been achieved by deleting the content."""

    data, _ = normalize_process_payload(_orphan_payload(), mode=RESEARCH_MODE)

    names = {str(row.get("name")) for row in data["entities"]["proteins"]}
    assert {f"Orphan factor {i}" for i in range(4)} <= names


def test_the_old_argument_is_what_manufactured_those_errors() -> None:
    """The measurement, pinned: same payload, enforcement on, 4 errors appear.

    This is the before/after that justifies the change -- run it against the
    research-mode output the fix now produces and the four degree-0 errors come
    straight back.
    """

    data, _ = normalize_process_payload(_orphan_payload(), mode=RESEARCH_MODE)

    with pytest.raises(GateValidationError) as excinfo:
        run_strict_post_normalization_gates(data, enforce_all_proteins_connected=True)

    errors = excinfo.value.details["errors"]
    assert len(errors) == 4
    assert all("degree 0 after normalization" in e["reason"] for e in errors)


def test_strict_mode_still_enforces_connectivity_on_the_same_payload() -> None:
    """The strict twin: nothing about strict mode's connectivity policy moved.

    Note which strict behaviour this pins. These orphans each carry a UniProt id,
    and ``prune_disconnected_proteins`` spares an externally identified protein --
    so strict mode does not delete them here, it *reports* them, and the degree-0
    rule is exactly what turns them into four gate errors. That report is the
    thing ``enforce_all_proteins_connected=not is_research(mode)`` must leave
    untouched in strict mode. (Strict deletion of an *unidentified* orphan is
    pinned separately by
    ``tests/test_research_mode_normalizer.py::test_strict_mode_deletes_the_unwired_protein``.)
    """

    data, report = normalize_process_payload(_orphan_payload(), mode=PATHWHIZ)

    assert report["gate"]["ok"] is False
    errors = report["gate"]["errors"]
    assert len(errors) == 4
    assert all("degree 0 after normalization" in e["reason"] for e in errors)
    assert report["summary"]["gate_error_count"] == 4
    # ...and a strict report still carries no research bookkeeping at all.
    assert "export_mode" not in report
    assert "research_mode_relaxations" not in report["summary"]
