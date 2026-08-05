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
import shutil
import sys
from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterator, List, Tuple

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

#: The frozen baseline cohort. **This is the only thing that decides membership.**
#:
#: The cohort used to be discovered by ``RUNS.glob("*/papers/*/*")``. That made the
#: merge gate a function of the working directory: every milestone benchmark
#: archives a new run directory under ``runs/``, and each one silently redefined
#: the population the gate was supposed to be guarding. It also went stale without
#: anyone noticing -- the pin said 23 legs while the glob returned 39.
#:
#: Reading a version-controlled manifest instead means a leg enters or leaves the
#: baseline only through a reviewed edit, and creating a run directory cannot move
#: the gate. See the ``what_this_is`` / ``additions_require_review`` keys in the
#: file itself.
#:
#: **This manifest is NOT C-010's per-leg delta allowlist**
#: (``docs/pwml_recovery_sprint/BASELINE.md`` § 6). That allowlist spans ``runs/``
#: *and* ``runs_verify/`` (32 legs) and lists legs whose *verdict* is expected to
#: change; this manifest lists which legs are *measured* and says nothing about
#: their verdicts. Two of C-010's six legs happen to sit inside this cohort. The
#: overlap is incidental and must never be used to pre-apply C-010's delta here.
BASELINE_COHORT_MANIFEST = ROOT / "tests" / "data" / "baseline_cohort_manifest.json"

#: Legs whose merged payload is large enough that normalizing it in a unit test
#: is measured in tens of seconds. Replaying them is still useful, so they are
#: not skipped -- but the whole module is marked slow for the same reason.
_MAX_PAYLOAD_BYTES = 2_000_000


pytestmark = pytest.mark.skipif(
    not RUNS.is_dir(), reason="runs/ is a local working directory and is not always present"
)


class BaselineCohortError(AssertionError):
    """A frozen manifest entry did not resolve.

    Raised, never swallowed and never downgraded to a skip. Silently dropping an
    unresolvable entry is exactly how the previous glob-driven gate broke: the
    cohort changed underneath the pinned totals and every aggregate moved with it.
    """


def _read_manifest(path: Path | None = None) -> Dict[str, Any]:
    """Load and structurally validate the manifest. Any defect is fatal."""

    manifest_path = BASELINE_COHORT_MANIFEST if path is None else path
    if not manifest_path.is_file():
        raise BaselineCohortError(
            f"the frozen baseline cohort manifest is missing: {manifest_path}. "
            "The replay merge gate has no cohort without it and must not fall "
            "back to globbing runs/."
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1:
        raise BaselineCohortError(
            f"{manifest_path}: unsupported schema_version "
            f"{manifest.get('schema_version')!r}, expected 1"
        )
    if manifest.get("max_payload_bytes") != _MAX_PAYLOAD_BYTES:
        raise BaselineCohortError(
            f"{manifest_path}: max_payload_bytes {manifest.get('max_payload_bytes')!r} "
            f"disagrees with the harness bound {_MAX_PAYLOAD_BYTES}. Widening the "
            "bound in one place only would change the cohort by stealth."
        )
    return manifest


def _manifest_legs(path: Path | None = None) -> List[Path]:
    """Resolve every manifest entry to a leg directory, or fail loudly."""

    manifest = _read_manifest(path)
    entries = manifest.get("legs")
    if not isinstance(entries, list) or not entries:
        raise BaselineCohortError(f"{path or BASELINE_COHORT_MANIFEST}: 'legs' is empty")
    declared = manifest.get("leg_count")
    if declared != len(entries):
        raise BaselineCohortError(
            f"{path or BASELINE_COHORT_MANIFEST}: leg_count {declared!r} disagrees "
            f"with {len(entries)} entries"
        )

    seen: set[str] = set()
    out: List[Path] = []
    unresolved: List[str] = []
    for entry in entries:
        rel = str(entry.get("path", ""))
        if rel in seen:
            raise BaselineCohortError(
                f"duplicate manifest entry {rel!r}: a leg counted twice would "
                "inflate every aggregate below"
            )
        seen.add(rel)
        leg = ROOT / rel
        payload_name = str(entry.get("payload", ""))
        payload = leg / payload_name
        if not leg.is_dir():
            unresolved.append(f"{rel} (leg directory absent)")
        elif not payload.is_file():
            unresolved.append(f"{rel} (payload {payload_name!r} absent)")
        elif payload.stat().st_size > _MAX_PAYLOAD_BYTES:
            unresolved.append(
                f"{rel} (payload {payload_name!r} is {payload.stat().st_size} bytes, "
                f"over the {_MAX_PAYLOAD_BYTES} bound)"
            )
        else:
            out.append(leg)

    if unresolved:
        raise BaselineCohortError(
            f"{len(unresolved)} of {len(entries)} frozen baseline cohort entries do "
            "not resolve on disk. The cohort must NEVER shrink silently -- restore "
            "the artifacts, or make the removal an explicit reviewed manifest edit "
            "together with a re-measured FULL_STACK_BASELINE and the table in "
            "docs/change_log.md. Unresolved: " + "; ".join(unresolved)
        )
    return out


def _legs() -> List[Path]:
    """The frozen baseline cohort: read from the manifest, never globbed.

    ``runs/`` is still allowed to be absent -- the whole module skips in that case
    (``pytestmark`` above) and returning an empty list keeps parametrization from
    erroring during collection. What is *not* allowed is ``runs/`` existing while a
    manifest entry does not: that is a shrinking cohort and it raises.
    """

    if not RUNS.is_dir():
        return []
    return _manifest_legs()


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
#: Measured over the FROZEN cohort in ``BASELINE_COHORT_MANIFEST`` -- 39 legs,
#: verified entry by entry on ``sprint/pwml-recovery`` @ ``2b786aa`` before any
#: C-branch landed. A new batch landing in ``runs/`` no longer moves these
#: numbers, which is the whole point: the gate measures a reviewed population,
#: not whatever happens to be on disk. Changing them requires a manifest edit,
#: a fresh measurement, and the matching edit to ``docs/change_log.md`` -- all in
#: one commit, so the three can never disagree.
#:
#: These are baseline *measurements*, not targets, and they are NOT C-010's
#: expected delta. C-010 changes what some legs' verdicts ARE; this pins which
#: legs are counted and what they measure today. Do not pre-apply one to the other.
FULL_STACK_BASELINE: Dict[str, int] = {
    "legs_examined": 39,
    "quarantine_admitted": 27,
    "quarantine_refused": 12,
    "stage3_after_pass": 27,
    "required_contract_pass": 8,
    "reached_ir": 8,
    "ir_pass": 8,
    "exportable": 8,
}

#: How many LEGS exhibit each residual code, and how many error ROWS carry it.
#: Different numbers for the same defect -- a leg with two undeclared species
#: contributes one to the first and two to the second -- and conflating them is
#: how "19 legs" turns into "19 rows" in a report.
RESIDUAL_CODES_BY_LEG: Dict[str, int] = {
    "species_missing_classification": 19,
    "species_missing_taxonomy": 19,
    "no_biological_states": 4,
}
RESIDUAL_CODES_BY_ROW: Dict[str, int] = {
    "species_missing_classification": 27,
    "species_missing_taxonomy": 27,
    "no_biological_states": 4,
}

_REMEASURE = (
    "the frozen baseline cohort no longer measures what was pinned. The cohort "
    "itself cannot drift (it is tests/data/baseline_cohort_manifest.json, not a "
    "glob), so either an artifact under runs/ changed or the pipeline's verdict "
    "on one of these legs changed. Establish which, then update the manifest if "
    "and only if membership was meant to change, and update BOTH this baseline "
    "and the table in docs/change_log.md so they cannot disagree."
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
    # the other: Stage 3 recovery is complete, exportability is 8 in 27.
    assert measured["stage3_after_pass"] == measured["quarantine_admitted"]
    assert measured["exportable"] < measured["quarantine_admitted"]

    # And the population it was measured over is the frozen one, not a glob.
    assert len(rows) == len(_manifest_legs())


def test_no_leg_fails_the_ir_once_the_required_contract_passes() -> None:
    """Where the contract is satisfied, the IR must be too."""

    rows = [_full_stack(leg) for leg in _legs()]
    reached_ir = [row for row in rows if row["required_contract"] == "pass"]

    for row in reached_ir:
        assert row["ir"] == "pass", (row["leg"], row["ir_codes"])


# ── The cohort is frozen, and the freeze is what is under test here ─────────
#
# Everything above measures legs. Everything below measures the thing that
# decides WHICH legs get measured. That used to be a filesystem glob, so the
# merge gate was a function of the working directory and moved whenever a
# benchmark archived a run. These tests exist so it can never do that again.


@pytest.fixture
def new_run_directory() -> Iterator[Path]:
    """A plausible freshly-archived batch, created under ``runs/`` and removed.

    Shaped to match ``*/papers/*/*`` exactly, because the point is that the
    *retired* glob would have swallowed it. Torn down in a finalizer so a failing
    assertion still cannot leave it behind in the working tree.
    """

    holder = RUNS / "_h001_synthetic_new_batch"
    leg = holder / "papers" / "PMC00000000" / "strict"
    shutil.rmtree(holder, ignore_errors=True)
    leg.mkdir(parents=True)
    (leg / "final_mapped.json").write_text(
        json.dumps({"processes": {}, "entities": {}}), encoding="utf-8"
    )
    try:
        yield holder
    finally:
        shutil.rmtree(holder, ignore_errors=True)


def test_a_new_run_directory_cannot_change_the_baseline_cohort(
    new_run_directory: Path,
) -> None:
    """The reason this branch exists.

    T-100..T-105 each archive a run directory. Under the old glob every one of
    them silently redefined the population the merge gate measures -- the gate
    guarding the sprint was rewritten by the sprint. The manifest ends that:
    membership is a reviewed file, not a directory listing.
    """

    # The fixture is real, and the retired glob WOULD have picked it up. Without
    # this the test could pass against a fixture that never existed.
    globbed = sorted(RUNS.glob("*/papers/*/*"))
    assert any(
        leg.is_relative_to(new_run_directory) for leg in globbed
    ), "fixture did not land where the retired glob would have found it"

    cohort = _legs()
    frozen = [ROOT / entry["path"] for entry in _read_manifest()["legs"]]

    assert cohort == frozen
    assert len(cohort) == FULL_STACK_BASELINE["legs_examined"]
    assert not any(leg.is_relative_to(new_run_directory) for leg in cohort)
    assert not any(new_run_directory.name in leg_id for leg_id in _leg_ids())


def test_a_manifest_entry_missing_from_disk_fails_loudly(tmp_path: Path) -> None:
    """A cohort that shrinks quietly is how the previous gate broke.

    The failure has to name the entry: "38 legs instead of 39" sends a reader
    looking for a pipeline regression, when what happened is that an artifact
    went missing.
    """

    manifest = _read_manifest()
    entries = [dict(entry) for entry in manifest["legs"]]
    vanished = "runs/2099-01-01_0000/papers/PMC99999999/strict"
    entries[7] = {"path": vanished, "payload": "final_mapped.json", "payload_bytes": 1}
    broken = tmp_path / "broken_manifest.json"
    broken.write_text(
        json.dumps({**manifest, "legs": entries, "leg_count": len(entries)}),
        encoding="utf-8",
    )

    with pytest.raises(BaselineCohortError) as excinfo:
        _manifest_legs(broken)

    message = str(excinfo.value)
    assert vanished in message, message
    assert "leg directory absent" in message, message
    assert "NEVER shrink silently" in message, message

    # The real manifest is untouched and still resolves in full.
    assert len(_manifest_legs()) == FULL_STACK_BASELINE["legs_examined"]


def test_a_duplicate_manifest_entry_fails_loudly(tmp_path: Path) -> None:
    """A leg listed twice would inflate every aggregate without moving the count check."""

    manifest = _read_manifest()
    entries = [dict(entry) for entry in manifest["legs"]]
    entries[3] = dict(entries[2])
    duplicated = tmp_path / "duplicate_manifest.json"
    duplicated.write_text(
        json.dumps({**manifest, "legs": entries, "leg_count": len(entries)}),
        encoding="utf-8",
    )

    with pytest.raises(BaselineCohortError) as excinfo:
        _manifest_legs(duplicated)

    assert entries[2]["path"] in str(excinfo.value)


def test_the_frozen_manifest_is_internally_consistent_and_fully_resolvable() -> None:
    """Every entry verified the way it was verified at freeze time."""

    manifest = _read_manifest()
    entries = manifest["legs"]

    assert manifest["schema_version"] == 1
    assert manifest["max_payload_bytes"] == _MAX_PAYLOAD_BYTES
    assert manifest["root"] == "runs"
    # Recorded so a reader can see the cohort is a reviewed population, not a scan.
    assert manifest["what_this_is"]
    assert manifest["additions_require_review"]
    assert manifest["this_is_not_the_c010_allowlist"]

    paths = [entry["path"] for entry in entries]
    assert len(entries) == manifest["leg_count"] == FULL_STACK_BASELINE["legs_examined"]
    assert len(set(paths)) == len(paths)
    assert paths == sorted(paths), "entries are stored in the order the harness reports"

    for entry in entries:
        leg = ROOT / entry["path"]
        assert entry["path"].startswith("runs/"), entry
        assert leg.is_dir(), entry
        payload = leg / entry["payload"]
        assert payload.is_file(), entry
        assert payload.stat().st_size == entry["payload_bytes"], entry
        assert payload.stat().st_size <= _MAX_PAYLOAD_BYTES, entry
        # The recorded payload is the one the unchanged preference order picks.
        resolved, _body = _payload_for(leg)
        assert resolved == entry["payload"], entry

    assert _legs() == [ROOT / path for path in paths]


def test_the_change_log_baseline_table_agrees_with_the_pinned_values() -> None:
    """The table in the log and the pin in this module cannot drift apart again.

    They already had. The pin said 23 legs, the glob returned 39, and the log
    still said 17 legs / 19 rows. Every number below is rendered from the pinned
    dicts, so updating one without the other fails here rather than in a report
    six weeks later.
    """

    # Whitespace-normalized: the log is hand-wrapped prose and a claim that gets
    # reflowed across a line break is still the same claim.
    log = " ".join((ROOT / "docs" / "change_log.md").read_text(encoding="utf-8").split())

    admitted = FULL_STACK_BASELINE["quarantine_admitted"]
    by_leg = RESIDUAL_CODES_BY_LEG
    by_row = RESIDUAL_CODES_BY_ROW
    fragments = [
        f"across the {FULL_STACK_BASELINE['legs_examined']} cached legs",
        f"| quarantine | {admitted} admitted, "
        f"{FULL_STACK_BASELINE['quarantine_refused']} refused |",
        f"| Stage 3 after quarantine | "
        f"**{FULL_STACK_BASELINE['stage3_after_pass']} / {admitted} pass** |",
        f"| required-field contract | "
        f"{FULL_STACK_BASELINE['required_contract_pass']} / {admitted} pass |",
        f"| IR build + validation | {FULL_STACK_BASELINE['ir_pass']} / "
        f"{FULL_STACK_BASELINE['reached_ir']} of those reaching it |",
        f"| **fully exportable** | **{FULL_STACK_BASELINE['exportable']} / {admitted}** |",
        # The one-line restatement earlier in the log, which is the same claim and
        # must therefore move with it.
        f"({FULL_STACK_BASELINE['legs_examined']} legs; {admitted} admitted / "
        f"{FULL_STACK_BASELINE['quarantine_refused']} refused; "
        f"{FULL_STACK_BASELINE['stage3_after_pass']}/{admitted} Stage 3; "
        f"{FULL_STACK_BASELINE['required_contract_pass']}/{admitted} required contract; "
        f"{FULL_STACK_BASELINE['ir_pass']}/{FULL_STACK_BASELINE['reached_ir']} IR; "
        f"{FULL_STACK_BASELINE['exportable']}/{admitted} exportable; "
        f"by leg {by_leg['species_missing_classification']}/"
        f"{by_leg['species_missing_taxonomy']}/{by_leg['no_biological_states']}, "
        f"by row {by_row['species_missing_classification']}/"
        f"{by_row['species_missing_taxonomy']}/{by_row['no_biological_states']})",
    ]
    # Legs and rows are different numbers for the same defect; the log has to say
    # both, because reporting one as the other is the error this pin was born from.
    for code in sorted(by_leg):
        fragments.append(f"`{code}` ({by_leg[code]} legs, {by_row[code]} rows)")

    missing = [fragment for fragment in fragments if fragment not in log]
    assert not missing, (
        "docs/change_log.md disagrees with the pinned baseline in this module. "
        f"Absent from the log: {missing}"
    )

    # And the log has to say the cohort is frozen, so a reader is not left to
    # assume it is still discovered by scanning runs/.
    assert "tests/data/baseline_cohort_manifest.json" in log
