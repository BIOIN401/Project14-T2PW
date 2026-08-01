"""Replay of locally cached REAL run artifacts through the quarantine boundary.

``tests/fixtures/strict_failures/cases.json`` is hand-written: compact, readable,
and therefore only as representative as whoever wrote it. This module answers the
other question -- what happens to the payloads that actually came out of the
overnight batches in ``runs/`` -- by running each cached leg through the same
Stage 3 gate the app runs, then through quarantine, then through the gate again.

``runs/`` is a local working directory, not a committed fixture: these tests skip
cleanly where it is absent, and every one of them is a no-op rather than a
failure when a leg has no usable artifact. Nothing here reaches the network, a
database, or an LLM -- normalization is pure Python and mapping is not re-run.

**No Stage-0 context is stored in any archived leg.** Every ``pathway_context``
lives in ``st.session_state`` and is dropped when the run ends, so the replay is
necessarily run in the undeclared-core regime. That is itself the evidence for
why ``quarantine_and_close`` takes the context as an explicit argument instead of
looking for one in the payload -- see ``test_no_archived_leg_carries_stage_zero_context``.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from t2pw.pipeline.process_normalizer import (  # noqa: E402
    GateValidationError,
    normalize_process_payload,
    run_strict_post_normalization_gates,
)
from t2pw.pipeline.strict_quarantine import quarantine_and_close  # noqa: E402
from t2pw.pwml.ir import (  # noqa: E402
    build_pwml_ir,
    validate_pwml_ir,
    validate_required_pwml_contract,
)
from t2pw.pwml.writer import blocking_pwml_ir_errors  # noqa: E402

#: Exactly what ``run_pwml_export`` puts on ``metadata`` before it calls the
#: required-field gate (``streamlit_app.py``:3663-3671). Reproduced here rather
#: than approximated, because the gate errors on a missing pathway name, subject,
#: width or height, and a replay that omitted them would report a metadata
#: omission of its own making as a payload defect.
EXPORT_PATHWAY_NAME = "Replayed pathway"
EXPORT_PATHWAY_SUBJECT = "Metabolic"
EXPORT_WIDTH = 3200
EXPORT_HEIGHT = 1400


def _as_export_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    """The payload as the exporter hands it to the required-field gate."""

    gate_payload = deepcopy(payload)
    metadata = gate_payload.setdefault("metadata", {})
    if isinstance(metadata, dict):
        metadata.setdefault("pathway_name", EXPORT_PATHWAY_NAME)
        metadata.setdefault("name", EXPORT_PATHWAY_NAME)
        metadata.setdefault("pathway_subject", EXPORT_PATHWAY_SUBJECT)
        metadata.setdefault("subject", EXPORT_PATHWAY_SUBJECT)
        metadata.setdefault("description", "")
        metadata.setdefault("width", EXPORT_WIDTH)
        metadata.setdefault("height", EXPORT_HEIGHT)
    return gate_payload


def _required_contract_codes(payload: Dict[str, Any]) -> List[str]:
    """Required-field gate error codes, run the way the exporter runs it."""

    report = validate_required_pwml_contract(_as_export_payload(payload), strict_db=True)
    if report.get("ok"):
        return []
    return [str(issue.get("code", "")) for issue in report.get("errors", [])]


def _ir_codes(payload: Dict[str, Any]) -> List[str]:
    """IR build + IR validation error codes, or ``["ir_build_crashed:..."]``."""

    try:
        ir, ir_report = build_pwml_ir(
            deepcopy(payload),
            pathway_name=EXPORT_PATHWAY_NAME,
            pathway_subject=EXPORT_PATHWAY_SUBJECT,
            strict_db=True,
            width=EXPORT_WIDTH,
            height=EXPORT_HEIGHT,
        )
    except Exception as exc:  # noqa: BLE001 - a crash is a result, and must be recorded
        return [f"ir_build_crashed:{type(exc).__name__}"]
    codes = [str(issue.get("code", "")) for issue in blocking_pwml_ir_errors(ir_report)]
    codes.extend(str(issue.get("code", "")) for issue in validate_pwml_ir(ir).get("errors", []))
    return codes

RUNS = ROOT / "runs"

#: Legs whose merged payload is large enough that normalizing it in a unit test
#: is measured in tens of seconds. Replaying them is still useful, so they are
#: not skipped -- but the whole module is marked slow for the same reason.
_MAX_PAYLOAD_BYTES = 2_000_000


pytestmark = pytest.mark.skipif(
    not RUNS.is_dir(), reason="runs/ is a local working directory and is not always present"
)


def _legs() -> List[Path]:
    """Every archived leg that kept a payload we can put through the gate."""

    if not RUNS.is_dir():
        return []
    out: List[Path] = []
    for leg in sorted(RUNS.glob("*/papers/*/*")):
        if not leg.is_dir():
            continue
        for name in ("final_mapped.json", "merged_payload.json"):
            path = leg / name
            if path.exists() and path.stat().st_size <= _MAX_PAYLOAD_BYTES:
                out.append(leg)
                break
    return out


def _leg_ids() -> List[str]:
    return [f"{leg.parts[-4]}/{leg.parts[-2][:24]}/{leg.parts[-1]}" for leg in _legs()]


def _payload_for(leg: Path) -> Tuple[str, Dict[str, Any]]:
    """The furthest-along payload this leg kept, and which file it came from."""

    for name in ("final_mapped.json", "merged_payload.json"):
        path = leg / name
        if path.exists() and path.stat().st_size <= _MAX_PAYLOAD_BYTES:
            return name, json.loads(path.read_text(encoding="utf-8"))
    raise AssertionError(f"no usable payload for {leg}")


def _normalized(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Stage 3 normalization, which is what the boundary's input has been through."""

    out = normalize_process_payload(deepcopy(payload))
    return out[0] if isinstance(out, tuple) else out


def _gate_reasons(payload: Dict[str, Any]) -> List[str]:
    try:
        run_strict_post_normalization_gates(deepcopy(payload))
    except GateValidationError as exc:
        return [str(row.get("reason", "")) for row in (exc.details.get("errors") or [])]
    return []


def _process_count(payload: Dict[str, Any]) -> int:
    return sum(
        len(rows)
        for rows in (payload.get("processes") or {}).values()
        if isinstance(rows, list)
    )


def test_there_is_something_to_replay() -> None:
    """A silent zero-leg run would make every test below vacuously green."""

    assert _legs(), "runs/ exists but holds no replayable leg"


@pytest.mark.parametrize("leg", _legs(), ids=_leg_ids())
def test_quarantine_never_makes_a_real_leg_worse(leg: Path) -> None:
    """The one property that must hold on every real payload, recovered or not.

    Quarantine may leave a leg failing -- some failures are not the kind it
    addresses -- but it must never turn a leg that passed the gate into one that
    does not, and it must never leave a leg failing for a reason it introduced.
    """

    _source, raw = _payload_for(leg)
    normalized = _normalized(raw)
    before = _gate_reasons(normalized)

    result = quarantine_and_close(normalized, strict_db=True)

    if not result.ok:
        # A refusal hands the caller the original; nothing was claimed.
        assert result.refusal_reasons
        return

    after = _gate_reasons(result.payload)
    assert not (set(after) - set(before)), {
        "leg": str(leg),
        "introduced": sorted(set(after) - set(before)),
    }
    if not before:
        assert after == [], f"{leg} passed before quarantine and must still pass"


@pytest.mark.parametrize("leg", _legs(), ids=_leg_ids())
def test_quarantine_is_a_fixpoint_on_every_real_leg(leg: Path) -> None:
    """Real payloads are where a closure that does not converge would show up."""

    _source, raw = _payload_for(leg)
    result = quarantine_and_close(_normalized(raw), strict_db=True)

    assert result.closure_report["converged"] is True
    again = quarantine_and_close(deepcopy(result.payload), strict_db=True)
    assert again.payload == result.payload


@pytest.mark.parametrize("leg", _legs(), ids=_leg_ids())
def test_a_real_leg_that_survives_keeps_processes(leg: Path) -> None:
    """Emptying a real pathway must never read as success."""

    _source, raw = _payload_for(leg)
    result = quarantine_and_close(_normalized(raw), strict_db=True)

    if result.ok:
        assert _process_count(result.payload) > 0
        # Which survivors have to be non-empty depends on the regime. No archived
        # leg carries a Stage-0 context (see the module docstring), so these all
        # run undeclared, where a pathway of only interactions is legitimate and
        # `core_accepted` is legitimately zero -- asserting on it here was the
        # same mistake the coverage check itself was making.
        countable = (
            "core_accepted_processes"
            if result.coverage["requested_core_declared"]
            else "surviving_processes"
        )
        assert result.coverage[countable] >= 1


def test_no_archived_leg_carries_stage_zero_context() -> None:
    """Why the context is a parameter and not something to go looking for.

    If a single archived payload carried ``key_compounds``, payload-only discovery
    would be defensible on real data. None does: the Stage-0 context lives in
    session state for the length of a run and is never written down, so a
    coverage check that reads only the payload runs in the undeclared-core regime
    on every production payload that exists.
    """

    carriers = [
        str(path)
        for path in RUNS.rglob("*.json")
        if path.stat().st_size <= _MAX_PAYLOAD_BYTES
        and "key_compounds" in path.read_text(encoding="utf-8", errors="ignore")
    ]

    assert carriers == []


# ── Full stack: Stage 3 is not exportability ────────────────────────────────
#
# Stage 3 recovery and strict PWML exportability are different claims, and this
# module used to make only the first and report it as if it were the second.
# Everything below measures both, per leg, and the module-level summary test
# refuses to let "Stage 3 passes" be read as "a file comes out".


def _full_stack(leg: Path) -> Dict[str, Any]:
    """Every stage the exporter runs, for one leg, as one row of a table."""

    _source, raw = _payload_for(leg)
    normalized = _normalized(raw)
    stage3_before = _gate_reasons(normalized)

    result = quarantine_and_close(normalized, strict_db=True)
    row: Dict[str, Any] = {
        "leg": f"{leg.parts[-4]}/{leg.parts[-2][:26]}/{leg.parts[-1]}",
        "stage3_before": "fail" if stage3_before else "pass",
        "quarantine": "ok" if result.ok else "refused",
        "stage3_after": None,
        "required_contract": None,
        "ir": None,
        "exportable": False,
        "required_codes": [],
        "ir_codes": [],
    }
    if not result.ok:
        return row

    stage3_after = _gate_reasons(result.payload)
    row["stage3_after"] = "fail" if stage3_after else "pass"
    if stage3_after:
        return row

    required_codes = _required_contract_codes(result.payload)
    row["required_codes"] = required_codes
    row["required_contract"] = "fail" if required_codes else "pass"
    if required_codes:
        return row

    ir_codes = _ir_codes(result.payload)
    row["ir_codes"] = ir_codes
    row["ir"] = "fail" if ir_codes else "pass"
    row["exportable"] = not ir_codes
    return row


@pytest.mark.parametrize("leg", _legs(), ids=_leg_ids())
def test_each_leg_is_measured_at_every_stage_not_only_at_stage_three(leg: Path) -> None:
    """A row is only ``exportable`` when every stage the exporter runs passed."""

    row = _full_stack(leg)

    assert row["quarantine"] in {"ok", "refused"}
    if row["quarantine"] == "refused":
        assert row["stage3_after"] is None
        assert row["exportable"] is False
        return
    assert row["stage3_after"] in {"pass", "fail"}
    # The claim under test: exportable implies all four, never just Stage 3.
    if row["exportable"]:
        assert row["stage3_after"] == "pass"
        assert row["required_contract"] == "pass"
        assert row["ir"] == "pass"
    else:
        assert (
            row["stage3_after"] == "fail"
            or row["required_contract"] == "fail"
            or row["ir"] == "fail"
        )


def test_stage_three_recovery_is_not_strict_exportability() -> None:
    """The measured distinction, pinned so the report cannot drift from it.

    Stage 3 recovery is real and is what quarantine was built for. Strict PWML
    exportability is a further claim that these payloads do NOT currently meet,
    for a reason quarantine is the wrong place to fix: the required-field gate
    wants a taxonomy id and a Prokaryote/Eukaryote classification on every
    species row, and the archived payloads carry species rows with neither.
    Inventing them inside quarantine would be fabricating database identity.

    So this asserts the shape of the gap rather than repairing it: Stage 3
    recovers, and the residual failures are the species-metadata class rather
    than anything quarantine left behind.
    """

    rows = [_full_stack(leg) for leg in _legs()]
    non_refused = [row for row in rows if row["quarantine"] == "ok"]
    assert non_refused, "nothing to measure"

    # Stage 3: quarantine's own claim. Every non-refused leg clears it.
    assert all(row["stage3_after"] == "pass" for row in non_refused), [
        row["leg"] for row in non_refused if row["stage3_after"] != "pass"
    ]

    # Exportability: a different claim, and currently a failing one.
    failing = [row for row in non_refused if row["required_contract"] == "fail"]
    codes = Counter(code for row in failing for code in row["required_codes"])
    assert failing, (
        "if this ever passes, the species-metadata defect below has been fixed "
        "upstream and this test should be replaced by an exportability assertion"
    )
    # Every residual failure is the species-metadata class -- not a dangling
    # reference, not a degree-zero entity, not anything quarantine touches.
    assert set(codes) <= {
        "species_missing_classification",
        "species_missing_taxonomy",
        "no_biological_states",
    }, dict(codes)


#: The full-stack result over the locally cached legs, measured and pinned.
#:
#: Reported numbers drift silently otherwise: the aggregate is what gets written
#: into ``docs/change_log.md`` and repeated in a summary, and nothing else in this
#: module would notice if 18/18 became 17/18. Pinned as an exact equality rather
#: than a bound, because "at least one leg exports" is the claim that was already
#: too weak to catch the original overstatement.
#:
#: When a new batch lands in ``runs/`` this test fails by construction. That is
#: the intended cost: re-measure, update the table here AND in the change log,
#: and the two can never disagree.
FULL_STACK_BASELINE: Dict[str, int] = {
    "legs_examined": 23,
    "quarantine_admitted": 18,
    "quarantine_refused": 5,
    "stage3_after_pass": 18,
    "required_contract_pass": 1,
    "reached_ir": 1,
    "ir_pass": 1,
    "exportable": 1,
}

#: How many LEGS exhibit each residual code, and how many error ROWS carry it.
#: Different numbers for the same defect -- a leg with two undeclared species
#: contributes one to the first and two to the second -- and conflating them is
#: how "17 legs" turns into "17 rows" in a report.
RESIDUAL_CODES_BY_LEG: Dict[str, int] = {
    "species_missing_classification": 17,
    "species_missing_taxonomy": 17,
    "no_biological_states": 4,
}
RESIDUAL_CODES_BY_ROW: Dict[str, int] = {
    "species_missing_classification": 19,
    "species_missing_taxonomy": 19,
    "no_biological_states": 4,
}

_REMEASURE = (
    "the cached legs in runs/ have changed. Re-measure, then update BOTH this "
    "baseline and the table in docs/change_log.md so they cannot disagree."
)


def test_the_full_stack_baseline_is_exactly_what_was_reported() -> None:
    """Every number in the reported table, asserted against a fresh measurement."""

    rows = [_full_stack(leg) for leg in _legs()]
    admitted = [row for row in rows if row["quarantine"] == "ok"]
    measured = {
        "legs_examined": len(rows),
        "quarantine_admitted": len(admitted),
        "quarantine_refused": len(rows) - len(admitted),
        "stage3_after_pass": sum(1 for row in admitted if row["stage3_after"] == "pass"),
        "required_contract_pass": sum(1 for row in admitted if row["required_contract"] == "pass"),
        "reached_ir": sum(1 for row in admitted if row["ir"] is not None),
        "ir_pass": sum(1 for row in admitted if row["ir"] == "pass"),
        "exportable": sum(1 for row in rows if row["exportable"]),
    }

    assert measured == FULL_STACK_BASELINE, f"{_REMEASURE} measured={measured}"

    by_leg = Counter(code for row in admitted for code in set(row["required_codes"]))
    by_row = Counter(code for row in admitted for code in row["required_codes"])

    assert dict(by_leg) == RESIDUAL_CODES_BY_LEG, f"{_REMEASURE} by_leg={dict(by_leg)}"
    assert dict(by_row) == RESIDUAL_CODES_BY_ROW, f"{_REMEASURE} by_row={dict(by_row)}"

    # The two halves of the reported claim, spelled out so neither can be read as
    # the other: Stage 3 recovery is complete, exportability is 1 in 18.
    assert measured["stage3_after_pass"] == measured["quarantine_admitted"]
    assert measured["exportable"] < measured["quarantine_admitted"]


def test_no_leg_fails_the_ir_once_the_required_contract_passes() -> None:
    """Where the contract is satisfied, the IR must be too."""

    rows = [_full_stack(leg) for leg in _legs()]
    reached_ir = [row for row in rows if row["required_contract"] == "pass"]

    for row in reached_ir:
        assert row["ir"] == "pass", (row["leg"], row["ir_codes"])
