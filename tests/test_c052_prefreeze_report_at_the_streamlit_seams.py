"""C-052: the pre-freeze report reaches the Streamlit seams, and A0-C8.

**Two kinds of test live here and they are labelled separately, because merge
rule G9 treats them differently.**

*Parts 1-4 are NEW ACCEPTANCE.* Both Streamlit seams captured a pre-freeze
resolution report and then **discarded** it, so ``rename_sources_collapsed``,
``review_required`` and every other compound-resolution diagnostic were
unreachable from the UI while the CLI persisted the same document. Nothing was
"wrong" at base -- the capability did not exist. These tests do fail at the base
SHA (no key, no file, no entry), and that is stated as a fact about a new
capability, **not** offered as a regression proof.

*Part 5 is a GUARD ON PRE-EXISTING BEHAVIOUR (A0-C8), and it PASSES AT BASE by
design.* ``LEDGER.md`` binds A0-C7/A0-C8 to *"test observable behaviour only"*:
the ``_freeze["canonical_json_path"] or sbml_input_path`` binding and the file
it names are unchanged by this card, and A0-C8 asks for them to be captured and
asserted rather than repaired. Claiming a base failure for them would be
fabricating one.

**What is deliberately NOT here.** ``run_pwml_export``'s seam is post-freeze --
it operates on a deepcopy taken after the hash -- and this card only *labels*
that fact in the record it writes. It adds no branch that changes whether a PWML
is produced, so it neither weakens nor strengthens any biological gate, and
acting on ``review_required`` is somebody else's work, registered as backlog.

No ``AppTest`` is needed: all three production surfaces under test --
``run_pwml_export``, ``run_post_pipeline_sbml_artifacts`` and
``_json_artifact_entries`` -- make zero ``st.`` calls.
"""

from __future__ import annotations

import ast
import json
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT / "tests") not in sys.path:
    sys.path.insert(0, str(ROOT / "tests"))

# ``t2pw.app.streamlit_app`` executes a Streamlit script on import and opens
# ``with st.form(...)`` at module scope, which under the real streamlit poisons
# every later ``AppTest`` in the process. The stub below is this repository's
# established convention, copied verbatim in shape from
# ``tests/test_prefreeze_third_export_seam.py``; it is sound here for the same
# reason it is sound there -- the functions under test make no Streamlit call at
# all, so the stub neutralizes the top-level script and nothing else. The guard
# matters: if the real streamlit is already loaded, stubbing it would break
# whatever loaded it.
if "streamlit" not in sys.modules:
    _st = MagicMock(name="streamlit")
    _st.columns.side_effect = lambda spec=1, **__: [
        MagicMock() for _ in range(spec if isinstance(spec, int) else len(spec))
    ]
    _st.tabs.side_effect = lambda labels, **__: [MagicMock() for _ in labels]
    _st.form_submit_button.return_value = False
    _st.checkbox.return_value = False
    _st.session_state = MagicMock()
    _st.session_state.get.return_value = None
    _st.session_state.__contains__ = lambda _s, _k: False
    sys.modules["streamlit"] = _st

from t2pw.pipeline.gate_reports import CANONICAL_PAYLOAD_KEY  # noqa: E402
from t2pw.pwml import prefreeze_resolution as prefreeze_module  # noqa: E402
import test_prefreeze_third_export_seam as third_seam  # noqa: E402
import test_streamlit_stage2_orchestration as orchestration  # noqa: E402

APP_REL = "src/t2pw/app/streamlit_app.py"
EP1 = "run_post_pipeline_sbml_artifacts"
EP3 = "run_pwml_export"

#: C-052's dispatch base -- the direct parent of this branch's first commit, so
#: it is reachable from any checkout that has the branch. Used only to read the
#: BEFORE key list of EP1's return dict out of git, which is the one way to state
#: "additions only" as a measurement rather than as a promise.
BASE_SHA = "a662c3f5ce994a1436fe62429c74a7db1144df14"

#: Exactly the keys C-052 appends to EP1's success return. The additions-only
#: test subtracts these and must recover the base list, in order.
_EP1_ADDED = (
    "prefreeze_resolution_report",
    "prefreeze_review_required",
    "canonical_json_path_name",
    "sbml_input_path_name",
)
#: Exactly the keys C-052 appends to EP3's success return.
_EP3_ADDED = (
    "prefreeze_resolution_report",
    "prefreeze_resolution_report_path",
    "prefreeze_review_required",
)

#: The third seam's own filename, and the CLI's. They must never be the same
#: string: ``writer.py`` defaults ``--out-dir`` to ``outputs`` and writes the
#: second one, so run from the project root a shared name means whichever writer
#: ran last wins, silently.
EP3_REPORT_NAME = "pwml_prefreeze_resolution_report.streamlit.json"
CLI_REPORT_NAME = "pwml_prefreeze_resolution_report.json"
EP3_SEAM_LABEL = "post_freeze_refinement_reexport"


# ── offline doubles ────────────────────────────────────────────────────────
#
# Passed explicitly so no test here ever falls through to
# ``PathBankDbResolver.from_env()`` and opens a real connection: a worktree now
# carries a ``.env``, so "offline by accident" is no longer available and has to
# be arranged.


class _OfflineResolver:
    """Reachable enough to be consulted, and reports itself unavailable."""

    last_error = "db_not_configured_in_test"

    def available(self) -> bool:
        return False


class _ChebiNameIndex:
    """Offline id -> canonical name index, keyed on ChEBI only."""

    def __init__(self, by_chebi: Dict[str, Dict[str, Any]]) -> None:
        self._by_chebi = by_chebi

    def compound_canonical(self, **ids: Any) -> Optional[Dict[str, Any]]:
        chebi = ids.get("chebi")
        return self._by_chebi.get(str(chebi)) if chebi else None


def _collapsing_index() -> _ChebiNameIndex:
    return _ChebiNameIndex(
        {"15428": {"id": 78, "name": "Glycerol 3-phosphate", "matched_on": "chebi"}}
    )


def _exportable_payload() -> Dict[str, Any]:
    """A payload the third seam can carry all the way to a written PWML.

    ``tests/test_prefreeze_third_export_seam.py``'s fixture, reused rather than
    re-typed: it is already pinned as a payload the quarantine boundary, the
    Stage-8 gate, the pre-export contract and the IR build all accept, so a
    refusal in the tests below is this card's seam and never the fixture.

    **It cannot be the collapsing population.** ``run_pwml_export`` quarantines
    BEFORE the pre-freeze seam, and the two-spellings fixture is unreferenced by
    construction -- referenced duplicate rows rename into one name and the
    connectivity guard stops the run, which is a different behaviour and not this
    card's. Unreferenced rows do not survive graph closure, so the collapse
    diagnostic is demonstrated end to end at the post-pipeline seam (part 2),
    whose boundary this fixture's own suite drives separately.
    """

    return third_seam._raw_extraction_payload()


def _export_kwargs(tmp_path: Path) -> Dict[str, Any]:
    return {
        "pathway_name": "Caffeine demethylation",
        "pathway_description": "",
        "pathway_subject": "Metabolic",
        "project_root": tmp_path,
        "ref_path": ROOT / "reference" / "PW000001.pwml",
        "strict_db": False,
    }


def _collapsing_payload() -> Dict[str, Any]:
    """Two spellings of one molecule -- the population that fills
    ``rename_sources_collapsed``.

    Neither row is referenced from ``processes``: a referenced pair renames into
    a shared name and the connectivity guard stops the run, which is a different
    behaviour and not the one this card is about. The shape is the one
    ``tests/test_prefreeze_compound_resolution.py`` already pins for this record.
    """

    return {
        "metadata": {"pathway_name": "C-052 fixture", "organism": "Homo sapiens"},
        "entities": {
            "species": [{"name": "Homo sapiens"}],
            "compounds": [
                {"name": "sn -glycerol 3-phosphate", "chebi_id": "CHEBI:15428",
                 "mapping_meta": {"query": {"name": "sn -glycerol 3-phosphate"}}},
                {"name": "sn-glycerol 3-phosphate", "chebi_id": "CHEBI:15428",
                 "mapping_meta": {"query": {"name": "sn-glycerol 3-phosphate"}}},
            ],
            "proteins": [{"name": "paper enzyme"}],
            "protein_complexes": [],
        },
        "processes": {
            "reactions": [{"name": "R1", "inputs": [], "outputs": []}],
            "transports": [],
            "interactions": [],
        },
    }


def _offline_prefreeze_spy(
    monkeypatch: pytest.MonkeyPatch,
    *,
    name_index: Any = None,
) -> List[Dict[str, Any]]:
    """Route the production seam through offline doubles and record its reports.

    The production call sites are untouched: this forwards to the real
    ``run_prefreeze_resolution`` with the arguments they pass, and only fills in
    the resolver and index they leave to the ambient environment. The returned
    list holds the exact report OBJECTS the seam handed back, so a caller can
    assert identity rather than equality.
    """

    real = prefreeze_module.run_prefreeze_resolution
    seen: List[Dict[str, Any]] = []

    def spy(payload: Dict[str, Any], **kwargs: Any) -> Dict[str, Any]:
        kwargs.setdefault("db_resolver", _OfflineResolver())
        if name_index is not None:
            kwargs.setdefault("name_index", name_index)
        report = real(payload, **kwargs)
        seen.append(report)
        return report

    monkeypatch.setattr(prefreeze_module, "run_prefreeze_resolution", spy)
    return seen


# ── EP1 harness ────────────────────────────────────────────────────────────


def _ep1(tmp_path: Path, payload: Dict[str, Any]) -> Any:
    """``run_post_pipeline_sbml_artifacts``, callable, with ``payload`` surviving.

    Reuses ``tests/test_streamlit_stage2_orchestration.py``'s harness -- the
    established vehicle for calling this orchestrator without a browser, a DB or
    an LLM -- and overrides its Stage-2 result so the compounds under test are
    the ones that reach the pre-freeze seam. The returned function's
    ``__globals__`` IS the harness namespace, which is how the two spies below
    reach ``freeze_canonical_payload`` and ``build_sbml``.
    """

    function, _observed = orchestration._orchestration_harness(
        tmp_path,
        stage2_result={"payload": deepcopy(payload), "report": {"summary": {}}},
    )
    return function


def _ep1_kwargs(tmp_path: Path, **overrides: Any) -> Dict[str, Any]:
    kwargs = dict(orchestration._invoke_kwargs(tmp_path))
    kwargs.update(overrides)
    return kwargs


def _dict_literal_keys(source: str, function_name: str) -> List[str]:
    """The key list of the LAST dict literal in ``function_name``'s success return.

    Rendered as the constant's value, or as the identifier for the three keys
    the module supplies through a name (``INITIAL_GATE_REPORT_KEY`` and friends),
    so a key that changed from a constant to a name -- or the reverse -- shows up
    as a change rather than as a match.
    """

    node = next(
        n for n in ast.parse(source).body
        if isinstance(n, ast.FunctionDef) and n.name == function_name
    )
    dicts = [d for d in ast.walk(node) if isinstance(d, ast.Dict) and len(d.keys) > 20]
    assert dicts, f"{function_name}: no artifact-set dict literal found"
    biggest = max(dicts, key=lambda d: len(d.keys))
    rendered: List[str] = []
    for key in biggest.keys:
        if isinstance(key, ast.Constant):
            rendered.append(str(key.value))
        elif isinstance(key, ast.Name):
            rendered.append(key.id)
        else:  # pragma: no cover - would be a new construct, and must be seen
            raise AssertionError(f"unrenderable key node {ast.dump(key)}")
    return rendered


def _source_at(ref: Optional[str]) -> str:
    if ref is None:
        return (ROOT / APP_REL).read_text(encoding="utf-8")
    shown = subprocess.run(
        ["git", "show", f"{ref}:{APP_REL}"], cwd=ROOT, capture_output=True, check=True
    )
    return shown.stdout.decode("utf-8")


# ══ Part 1 — NEW ACCEPTANCE: the third seam persists its record ═════════════


def test_new_acceptance_the_third_seam_writes_its_prefreeze_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Fails at base: the report is captured at the seam and then dropped.

    The record is the seam's own report, WHOLE -- not a summary of it -- so a
    diagnostic added to the report in future reaches the operator without this
    card being edited again. Asserted by comparing the file against the exact
    object the seam returned, which is why the spy records objects rather than
    the test restating an expected shape.
    """

    seen = _offline_prefreeze_spy(monkeypatch, name_index=_ChebiNameIndex({}))
    from t2pw.app.streamlit_app import run_pwml_export

    result = run_pwml_export(_exportable_payload(), **_export_kwargs(tmp_path))

    written = tmp_path / "outputs" / EP3_REPORT_NAME
    assert written.is_file(), result.get("error")
    record = json.loads(written.read_text(encoding="utf-8"))
    assert seen, "the pre-freeze seam never ran; this test would assert nothing"
    assert {k: v for k, v in record.items() if k != "seam"} == json.loads(
        json.dumps(seen[-1], default=str)
    )
    # The diagnostics D-029 names, reachable from a persisted record for the
    # first time. Their VALUES on this population are exercised at the
    # post-pipeline seam below, where the collapsing fixture survives.
    assert "rename_sources_collapsed" in record["compounds"]
    assert "review_required" in record
    assert record["stage"] == "prefreeze_resolution"


def test_new_acceptance_the_third_seams_record_says_it_is_post_freeze(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The label is mandatory, top level, and cannot be shadowed.

    This seam runs on a deepcopy taken after the hash; the post-pipeline seam's
    report describes a pre-freeze canonicalization of a different object.
    Persisting both under one undifferentiated name would tell the operator they
    are the same record. Asserted by writing a report that already carries a
    ``seam`` key: the label must still win.
    """

    real = prefreeze_module.run_prefreeze_resolution

    def shadowing(payload: Dict[str, Any], **kwargs: Any) -> Dict[str, Any]:
        kwargs.setdefault("db_resolver", _OfflineResolver())
        kwargs.setdefault("name_index", _ChebiNameIndex({}))
        report = real(payload, **kwargs)
        report["seam"] = "SHADOWED"
        return report

    monkeypatch.setattr(prefreeze_module, "run_prefreeze_resolution", shadowing)
    from t2pw.app.streamlit_app import run_pwml_export

    result = run_pwml_export(_exportable_payload(), **_export_kwargs(tmp_path))

    written = tmp_path / "outputs" / EP3_REPORT_NAME
    assert written.is_file(), result.get("error")
    record = json.loads(written.read_text(encoding="utf-8"))
    assert record["seam"] == EP3_SEAM_LABEL


def test_new_acceptance_the_third_seam_never_writes_the_cli_filename(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The CLI's default output path is ``outputs/`` + the name below.

    Sharing it means whichever writer ran last wins, silently, and a CLI test
    pins that path. Measured on disk rather than read off the source, and the
    two names are re-derived from the CLI module so a change there fails here.
    """

    _offline_prefreeze_spy(monkeypatch, name_index=_ChebiNameIndex({}))
    from t2pw.app.streamlit_app import run_pwml_export

    result = run_pwml_export(_exportable_payload(), **_export_kwargs(tmp_path))

    assert (tmp_path / "outputs" / EP3_REPORT_NAME).is_file(), result.get("error")
    assert not (tmp_path / "outputs" / CLI_REPORT_NAME).exists()
    cli_source = (ROOT / "src" / "t2pw" / "pwml" / "writer.py").read_text("utf-8")
    assert f'"{CLI_REPORT_NAME}"' in cli_source, (
        "the CLI no longer writes this name; the collision this test guards has "
        "moved and the guard has to move with it")
    assert EP3_REPORT_NAME not in cli_source
    assert EP3_REPORT_NAME != CLI_REPORT_NAME


def test_new_acceptance_the_record_survives_a_failure_below_the_seam(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The catch-all wraps everything below the seam.

    A failure in the IR build, the deterministic builder or the PWML write would
    lose the report entirely -- and the export that most needs a resolution
    record is exactly the one that failed. The write is therefore placed above
    ``build_pwml_ir``, which is what this measures: the export fails, and the
    record is on disk anyway.
    """

    _offline_prefreeze_spy(monkeypatch, name_index=_ChebiNameIndex({}))
    from t2pw.app import streamlit_app

    def exploding(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError("C-052: IR build blown up on purpose")

    monkeypatch.setattr(streamlit_app, "build_pwml_ir", exploding)

    result = streamlit_app.run_pwml_export(
        _exportable_payload(), **_export_kwargs(tmp_path))

    assert result["ok"] is False
    assert "blown up on purpose" in str(result.get("error"))
    record = json.loads((tmp_path / "outputs" / EP3_REPORT_NAME).read_text("utf-8"))
    assert record["seam"] == EP3_SEAM_LABEL
    assert record["stage"] == "prefreeze_resolution"


def test_new_acceptance_the_third_seams_result_surfaces_the_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Persisted AND surfaced: the caller gets the document and the path."""

    seen = _offline_prefreeze_spy(monkeypatch, name_index=_ChebiNameIndex({}))
    from t2pw.app.streamlit_app import run_pwml_export

    result = run_pwml_export(_exportable_payload(), **_export_kwargs(tmp_path))

    assert result["ok"] is True, result.get("error")
    assert set(_EP3_ADDED) <= set(result)
    assert result["prefreeze_resolution_report"] is seen[-1]
    assert result["prefreeze_resolution_report_path"] == str(
        tmp_path / "outputs" / EP3_REPORT_NAME)
    assert Path(result["prefreeze_resolution_report_path"]).is_file()
    assert result["prefreeze_review_required"] == seen[-1]["review_required"]


# ══ Part 2 — NEW ACCEPTANCE: the post-pipeline seam surfaces its record ═════


def test_new_acceptance_the_post_pipeline_seam_returns_its_prefreeze_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Fails at base: the report is captured, never returned, and the temp
    directory it could have been written into is swept in ``finally``."""

    seen = _offline_prefreeze_spy(monkeypatch, name_index=_collapsing_index())
    payload = _collapsing_payload()

    result = _ep1(tmp_path, payload)(payload, **_ep1_kwargs(tmp_path))

    assert seen, "the pre-freeze seam never ran; this test would assert nothing"
    assert result["prefreeze_resolution_report"] is seen[-1]
    assert result["prefreeze_review_required"] == seen[-1]["review_required"]
    assert result["prefreeze_resolution_report"]["compounds"][
        "rename_sources_collapsed"] == [{
            "target": "glycerol 3 phosphate",
            "sources": ["sn -glycerol 3-phosphate", "sn-glycerol 3-phosphate"],
            "source_key": "sn glycerol 3 phosphate",
        }]
    # The two seams' records are DIFFERENT documents about different objects, so
    # this one must not wear the other's label.
    assert "seam" not in result["prefreeze_resolution_report"]


def test_new_acceptance_the_post_pipeline_seam_adds_keys_and_changes_none(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The grant is key-ADDITION only. Measured against the base blob.

    Subtracting exactly C-052's four keys from the tip's return-dict literal
    must recover the base list **in order**, so a renamed key, a reordered key, a
    removed key or a fifth addition all fail here. Read out of git rather than
    out of a hand-copied list, because a hand-copied list is a promise.
    """

    base_keys = _dict_literal_keys(_source_at(BASE_SHA), EP1)
    tip_keys = _dict_literal_keys(_source_at(None), EP1)

    assert [k for k in tip_keys if k not in _EP1_ADDED] == base_keys
    assert [k for k in tip_keys if k in _EP1_ADDED] == list(_EP1_ADDED)
    assert len(tip_keys) == len(base_keys) + len(_EP1_ADDED)

    # And the same statement about the live object, not only about the source.
    seen = _offline_prefreeze_spy(monkeypatch, name_index=_collapsing_index())
    payload = _collapsing_payload()
    result = _ep1(tmp_path, payload)(payload, **_ep1_kwargs(tmp_path))
    assert seen
    assert set(_EP1_ADDED) <= set(result)


# ══ Part 3 — NEW ACCEPTANCE: no added key may contain ``pwml`` ══════════════


def test_new_acceptance_no_added_key_makes_the_batch_driver_misread_the_export(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The trap D-040 §1 names, measured through the driver itself.

    ``_find_pwml_result`` falls back to the first ``post_pipeline_artifacts``
    entry whose key satisfies ``"pwml" in key.lower()`` and returns it **as the
    PWML export result**. A key named ``pwml_prefreeze_resolution_report`` -- the
    natural "converge on the CLI's filename" instinct -- would make every leg
    with an empty ``pwml_export_result`` report the prefreeze report as its
    export. A string check on the key names would state the rule; this states the
    consequence, which is what actually has to stay true.
    """

    from t2pw.batch.driver import _find_pwml_result

    seen = _offline_prefreeze_spy(monkeypatch, name_index=_collapsing_index())
    payload = _collapsing_payload()
    result = _ep1(tmp_path, payload)(payload, **_ep1_kwargs(tmp_path))
    assert seen

    class _NoExport:
        session_state: Dict[str, Any] = {}

    where, found = _find_pwml_result(_NoExport(), result)
    assert (where, found) == ("", {}), where

    # Belt and braces, and it names the offending key when it fires.
    offenders = [key for key in result if "pwml" in key.lower()]
    assert offenders == [], offenders
    assert [key for key in _EP1_ADDED if "pwml" in key.lower()] == []
    assert [key for key in _EP3_ADDED if "pwml" in key.lower()] == []
    # The first fallback matches on VALUE, not on key name, so the report must
    # also not look like an export result.
    report = result["prefreeze_resolution_report"]
    assert "xml_bytes" not in report and "pwml_ir" not in report


# ══ Part 4 — NEW ACCEPTANCE: the report is rendered ════════════════════════


def test_new_acceptance_the_artifact_panel_offers_the_prefreeze_report() -> None:
    """The post-pipeline seam writes no file, so this panel is the only route."""

    from t2pw.app.streamlit_app import _json_artifact_entries

    entries = _json_artifact_entries(
        {"prefreeze_resolution_report": {"stage": "prefreeze_resolution", "ok": True}},
        None,
    )
    by_filename = {filename: (label, value) for label, filename, value in entries}
    assert "prefreeze_resolution_report.json" in by_filename
    label, value = by_filename["prefreeze_resolution_report.json"]
    assert label == "Pre-freeze resolution report"
    assert value == {"stage": "prefreeze_resolution", "ok": True}
    # Never the CLI's name and never the third seam's: three records, three names.
    assert CLI_REPORT_NAME not in by_filename
    assert EP3_REPORT_NAME not in by_filename
    # An absent report must not produce an empty entry to click on.
    assert not [
        entry for entry in _json_artifact_entries({"prefreeze_resolution_report": {}}, None)
        if entry[1] == "prefreeze_resolution_report.json"
    ]


# ══ Part 5 — A0-C8 GUARD on pre-existing behaviour. PASSES AT BASE. ════════


def test_a0c8_guard_the_sbml_build_reads_the_object_the_freeze_returned(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A0-C8, unit level. **This is not a regression proof and passes at base.**

    *"The actual ``canonical_json_path``"* cannot be asserted by equality: it is
    built under a ``uuid4()`` temp directory that the orchestrator sweeps before
    returning, and the orchestrator takes no ``tmp`` parameter, so no test can
    predict or supply the string. Identity can be asserted, and is stronger:
    the ``Path`` handed to ``build_sbml`` must be **the same object** the freeze
    returned, which no equality check on a reconstructed path could distinguish
    from a lookalike built somewhere else.

    ``sbml_input_source`` is recorded in A0-C8 as insufficient for this, and it
    is: it says which *kind* of payload was used, never which *file* was read.
    """

    _offline_prefreeze_spy(monkeypatch, name_index=_collapsing_index())
    payload = _collapsing_payload()
    function = _ep1(tmp_path, payload)
    namespace = function.__globals__

    frozen: List[Dict[str, Any]] = []
    real_freeze = namespace["freeze_canonical_payload"]

    def freeze_spy(*args: Any, **kwargs: Any) -> Any:
        out = real_freeze(*args, **kwargs)
        frozen.append(out)
        return out

    sbml_calls: List[Any] = []

    def build_sbml_spy(*args: Any, **_kwargs: Any) -> Dict[str, Any]:
        sbml_calls.append(args)
        return {}

    namespace["freeze_canonical_payload"] = freeze_spy
    namespace["build_sbml"] = build_sbml_spy

    result = function(payload, **_ep1_kwargs(tmp_path, build_legacy_sbml=True))

    assert len(frozen) == 1
    assert len(sbml_calls) == 1, "the legacy SBML build did not run; this is vacuous"
    supplied = sbml_calls[0][0]
    canonical = frozen[0]["canonical_json_path"]
    assert canonical is not None, "the freeze refused; this test would assert nothing"
    assert supplied is canonical
    assert Path(supplied).name == "final.canonical.json"
    assert result["sbml_input_source"] == CANONICAL_PAYLOAD_KEY
    # The observability half of the same requirement: both names are published.
    assert result["canonical_json_path_name"] == "final.canonical.json"
    assert result["sbml_input_path_name"] == "final.canonical.json"


def test_a0c8_guard_the_published_names_follow_the_freeze_not_a_constant(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The two published names are read off the freeze, not hard-coded.

    **Not a regression proof; the keys are new and this only pins how they are
    computed.** A refusing freeze publishes no canonical name and the SBML input
    falls back to the enriched file, and the pair must say so -- otherwise the
    keys would assert ``final.canonical.json`` on a run that never wrote one.
    """

    _offline_prefreeze_spy(monkeypatch, name_index=_collapsing_index())
    payload = _collapsing_payload()
    function = _ep1(tmp_path, payload)
    namespace = function.__globals__
    real_freeze = namespace["freeze_canonical_payload"]

    def refusing_freeze(*args: Any, **kwargs: Any) -> Any:
        out = dict(real_freeze(*args, **kwargs))
        out["canonical_json_path"] = None
        out["sbml_input_source"] = "enriched_pre_quarantine"
        return out

    namespace["freeze_canonical_payload"] = refusing_freeze

    result = function(payload, **_ep1_kwargs(tmp_path))

    assert result["canonical_json_path_name"] == ""
    assert result["sbml_input_path_name"] == "final.mapped.json"
    assert result["sbml_input_source"] == "enriched_pre_quarantine"


def test_a0c8_measured_limitation_the_cohort_never_exercises_the_sbml_build() -> None:
    """The honest record of what A0-C8's cohort clause is NOT discharged by.

    *"Including the path supplied to downstream SBML generation"* is unexercised
    on every cohort leg: the ``build_sbml`` call is guarded on
    ``build_legacy_sbml``, and the batch driver never binds the legacy SBML
    button. So the clause is discharged at unit level above, and this test
    records the limitation as a measurement so it cannot quietly stop being true
    -- binding the driver to that button would be a production-behaviour change
    this card does not own, and it would fail here rather than pass unnoticed.
    """

    app_source = _source_at(None)
    node = next(
        n for n in ast.parse(app_source).body
        if isinstance(n, ast.FunctionDef) and n.name == EP1
    )
    guarded = [
        branch for branch in ast.walk(node)
        if isinstance(branch, ast.If)
        and isinstance(branch.test, ast.Name)
        and branch.test.id == "build_legacy_sbml"
        and any(
            isinstance(call, ast.Call) and isinstance(call.func, ast.Name)
            and call.func.id == "build_sbml"
            for statement in branch.body for call in ast.walk(statement)
        )
    ]
    assert len(guarded) == 1, "the legacy SBML guard moved; re-measure the limitation"

    driver_source = (ROOT / "src" / "t2pw" / "batch" / "driver.py").read_text("utf-8")
    assert "run_legacy_sbml_export_btn" not in driver_source, (
        "the batch driver now binds the legacy SBML button. A0-C8's cohort clause "
        "may now be dischargeable at cohort level -- and C-052 did not do it.")
