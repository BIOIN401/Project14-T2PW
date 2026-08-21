"""C-057 -- PRODUCT_CONTRACT section 3 attribution for the strict quarantine stage.

**NEW ACCEPTANCE TEST (G9 clause 4, new capability).** Before this card
``strict_quarantine.py`` contained no lineage token of any kind and imported
nothing from ``t2pw.pipeline.lineage``, and no writer of ``stage="quarantine"``
existed anywhere in ``src/`` although the value has been in C-015's closed
``STAGES`` vocabulary since it was written. There is therefore no pre-existing
observable lineage behaviour at this stage for these tests to preserve, and none
of them is offered as a base-failing correction proof.

What they pin is one decision and its three consequences:

* **The decision.** This stage is a FILTER, not a source. It introduces no
  content, so the only section 3 category it can populate is the sixth --
  "unresolved or excluded". It attributes what it EXCLUDED and never what
  survived: naming quarantine as the provenance of a row it merely let through
  would be the exact false attribution section 3 exists to prevent.
* **Reachability.** A quarantined process row is deleted from the payload, so
  the attribution has to reach a reader some other way. It reaches one through
  ``quarantine_report.json`` -- ``original_process`` is the retained row a
  reviewer restores -- and, where the row is a locked reaction, through the
  saved payload as well.
* **Non-causation.** The pass runs after every admission decision is final and
  changes nothing except ``provenance_lineage``: not the surviving row set, not
  an admission state, not a removal reason, not a report field, and not
  ``canonical_graph_sha256``.
"""

from __future__ import annotations

import json
import logging
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from t2pw.pipeline import lineage  # noqa: E402
from t2pw.pipeline import strict_quarantine  # noqa: E402
from t2pw.pipeline.canonical_hash import (  # noqa: E402
    canonical_graph_sha256,
    canonical_payload_sha256,
)
from t2pw.pipeline.strict_quarantine import (  # noqa: E402
    ACCEPTED_STATES,
    QUARANTINE_REPORT_FILENAME,
    QUARANTINE_STATES,
    QUARANTINED_BROKEN_REFERENCE,
    QUARANTINED_OUT_OF_SCOPE,
    QUARANTINED_UNMAPPED_ENTITY,
    QUARANTINED_WEAK_EVIDENCE,
    quarantine_and_close,
    write_quarantine_artifacts,
)


# ── Fixtures ────────────────────────────────────────────────────────────────


def _payload() -> Dict[str, Any]:
    """Two-step glutathione biosynthesis: small, valid, fully declared."""

    return {
        "metadata": {
            "pathway_name": "Glutathione biosynthesis",
            "organism": "Homo sapiens",
            "key_compounds": ["L-glutamate", "glutathione"],
        },
        "entities": {
            "species": [{"name": "Homo sapiens", "taxonomy_id": "9606"}],
            "subcellular_locations": [{"name": "cytosol"}],
            "compounds": [
                {"name": "L-glutamate"},
                {"name": "L-cysteine"},
                {"name": "gamma-glutamylcysteine"},
                {"name": "glycine"},
                {"name": "glutathione"},
            ],
            "proteins": [
                {
                    "name": "glutamate-cysteine ligase",
                    "species": "Homo sapiens",
                    "mapped_ids": {"uniprot": "P48506"},
                },
                {
                    "name": "glutathione synthetase",
                    "species": "Homo sapiens",
                    "mapped_ids": {"uniprot": "P48637"},
                },
            ],
            "protein_complexes": [],
            "nucleic_acids": [],
        },
        "biological_states": [
            {"name": "cytosol", "species": "Homo sapiens", "subcellular_location": "cytosol"}
        ],
        "element_locations": {
            "compound_locations": [
                {"compound": name, "biological_state": "cytosol"}
                for name in (
                    "L-glutamate",
                    "L-cysteine",
                    "gamma-glutamylcysteine",
                    "glycine",
                    "glutathione",
                )
            ],
            "protein_locations": [
                {"protein": "glutamate-cysteine ligase", "biological_state": "cytosol"},
                {"protein": "glutathione synthetase", "biological_state": "cytosol"},
            ],
        },
        "processes": {
            "reactions": [
                {
                    "name": "gamma-glutamylcysteine synthesis",
                    "inputs": ["L-glutamate", "L-cysteine"],
                    "outputs": ["gamma-glutamylcysteine"],
                    "enzymes": [{"entity": "glutamate-cysteine ligase"}],
                    "biological_state": "cytosol",
                },
                {
                    "name": "glutathione synthesis",
                    "inputs": ["gamma-glutamylcysteine", "glycine"],
                    "outputs": ["glutathione"],
                    "enzymes": [{"entity": "glutathione synthetase"}],
                    "biological_state": "cytosol",
                },
            ],
            "transports": [],
            "interactions": [],
        },
    }


#: One payload edit per quarantine state, with the controls that trigger it. The
#: fifth state, ``QUARANTINED_DISCONNECTED``, is produced by closure rather than
#: by admission and is deliberately NOT attributed -- see
#: ``test_a_process_stranded_by_closure_is_documented_as_unattributed``.
_BAD_ROWS = {
    QUARANTINED_UNMAPPED_ENTITY: (
        {
            "name": "peripheral oxidation",
            "inputs": ["glutathione"],
            "outputs": ["glutathione disulfide"],
            "biological_state": "cytosol",
        },
        {},
    ),
    QUARANTINED_OUT_OF_SCOPE: (
        {
            "name": "off topic reaction",
            "inputs": ["glutathione"],
            "outputs": ["glycine"],
            "scope_membership": "out_of_scope",
            "biological_state": "cytosol",
        },
        {},
    ),
    QUARANTINED_WEAK_EVIDENCE: (
        {
            "name": "low confidence reaction",
            "inputs": ["glutathione"],
            "outputs": ["glycine"],
            "confidence": 0.1,
            "biological_state": "cytosol",
        },
        {"confidence_floor": 0.5},
    ),
    QUARANTINED_BROKEN_REFERENCE: (
        {
            "name": "hollow reaction",
            "inputs": [],
            "outputs": ["glutathione"],
            "biological_state": "cytosol",
        },
        {},
    ),
}


def _run_with(state: str):
    """A completed quarantine whose extra reaction landed in ``state``."""

    row, controls = _BAD_ROWS[state]
    payload = _payload()
    payload["processes"]["reactions"].append(deepcopy(row))
    result = quarantine_and_close(payload, strict_db=True, **controls)
    assert result.states()[row["name"]] == state, result.states()
    return result, row["name"]


def _quarantine_entries(row: Any) -> list:
    """The ``stage="quarantine"`` entries on ``row``, as plain mappings."""

    if not isinstance(row, dict):
        return []
    return [
        entry
        for entry in row.get(lineage.LINEAGE_KEY) or []
        if isinstance(entry, dict) and entry.get("stage") == "quarantine"
    ]


def _record(result, name: str) -> Dict[str, Any]:
    return next(row for row in result.quarantine_report["quarantined"] if row["name"] == name)


def _strip_lineage(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _strip_lineage(v) for k, v in value.items() if k != lineage.LINEAGE_KEY}
    if isinstance(value, list):
        return [_strip_lineage(item) for item in value]
    return value


# ── 1. The attribution itself ───────────────────────────────────────────────


@pytest.mark.parametrize("state", sorted(_BAD_ROWS))
def test_every_admission_time_quarantine_state_attributes_its_row(state: str) -> None:
    """NEW: one typed entry per excluded process, for each state admission emits."""

    result, name = _run_with(state)
    entries = _quarantine_entries(_record(result, name)["original_process"])

    assert len(entries) == 1, entries
    entry = entries[0]
    assert entry["stage"] == "quarantine"
    assert entry["origin"] == "excluded"
    assert entry["support"] == "unsupported"
    assert entry["paper_explicit"] == "not_evaluated"
    assert entry["review_required"] is True
    assert entry["uncertainty"].strip()
    assert entry["sources"] == []


@pytest.mark.parametrize("state", sorted(_BAD_ROWS))
def test_the_reason_names_the_admission_state_and_the_stage_s_own_reason(state: str) -> None:
    """NEW: the entry is readable without the record beside it."""

    result, name = _run_with(state)
    record = _record(result, name)
    reason = _quarantine_entries(record["original_process"])[0]["reason"]

    assert state in reason
    assert str(record["reason"]) in reason


def test_the_origin_is_the_contract_s_sixth_category_and_nothing_else() -> None:
    """NEW: section 3's "unresolved or excluded", through C-015's own map."""

    result, name = _run_with(QUARANTINED_UNMAPPED_ENTITY)
    entry = _quarantine_entries(_record(result, name)["original_process"])[0]

    assert entry["origin"] in lineage.CONTRACT_CATEGORIES["unresolved_or_excluded"]
    # ``excluded`` is "deliberately removed"; ``unresolved`` is "could not be
    # grounded" and describes the participant, which this stage never removes.
    assert entry["origin"] == "excluded"
    # ``excluded`` is not a SOURCED_ORIGINS member, so an unevidenced removal is
    # allowed to name nothing -- and must not invent a source it never retrieved.
    assert entry["origin"] not in lineage.SOURCED_ORIGINS


def test_the_entry_validates_against_the_frozen_vocabularies_and_round_trips() -> None:
    """NEW: written through the only supported writer, and re-readable."""

    result, name = _run_with(QUARANTINED_BROKEN_REFERENCE)
    row = _record(result, name)["original_process"]

    parsed = lineage.read(row)
    assert len(parsed.entries) == 1
    entry = parsed.entries[0]
    assert entry.stage in lineage.STAGES
    assert entry.origin in lineage.ORIGINS
    assert entry.support in lineage.SUPPORT_LEVELS
    assert entry.paper_explicit in lineage.PAPER_EXPLICIT_VALUES
    # JSON is what actually ships; a value the serializer cannot write is not
    # traceable however well it validates in memory.
    assert json.loads(json.dumps(row[lineage.LINEAGE_KEY])) == row[lineage.LINEAGE_KEY]


def test_an_accepted_process_is_never_attributed_to_this_stage() -> None:
    """NEW: a filter may not claim the provenance of what it let through."""

    result, _name = _run_with(QUARANTINED_UNMAPPED_ENTITY)

    for row in result.payload["processes"]["reactions"]:
        assert _quarantine_entries(row) == []
    for record in result.quarantine_report["accepted"]:
        assert record["state"] in ACCEPTED_STATES
        assert lineage.LINEAGE_KEY not in record


# ── 2. Reachability -- an artifact a reader can actually open ───────────────


def test_the_attribution_reaches_quarantine_report_json_on_disk(tmp_path: Path) -> None:
    """NEW: the reader path, end to end, off the filesystem rather than in memory."""

    result, name = _run_with(QUARANTINED_UNMAPPED_ENTITY)
    written = write_quarantine_artifacts(result, tmp_path)

    document = json.loads(
        Path(written[QUARANTINE_REPORT_FILENAME]).read_text(encoding="utf-8")
    )
    record = next(row for row in document["quarantined"] if row["name"] == name)
    entries = _quarantine_entries(record["original_process"])
    assert len(entries) == 1
    assert entries[0]["stage"] == "quarantine"
    # The row a reviewer restores is intact beside its exclusion history.
    assert record["original_process"]["outputs"] == ["glutathione disulfide"]


def test_the_dropped_row_leaves_no_attribution_in_the_saved_payload() -> None:
    """NEW: the payload carries the surviving graph, not a removal record."""

    result, name = _run_with(QUARANTINED_UNMAPPED_ENTITY)

    assert name not in [row["name"] for row in result.payload["processes"]["reactions"]]
    assert "quarantine" not in json.dumps(result.payload.get("processes"))


def test_a_quarantined_locked_reaction_carries_its_attribution_into_the_payload() -> None:
    """NEW: the one shape where the excluded row itself survives in the payload.

    ``_reconcile_locked_reactions`` files the retained row under
    ``quarantined_locked_reactions``, because a lock the run promised to preserve
    may not simply vanish. The attribution rides with it.
    """

    payload = _payload()
    payload["processes"]["reactions"].append(
        {
            "name": "peripheral oxidation",
            "locked_reaction_id": "LOCK-1",
            "inputs": ["glutathione"],
            "outputs": ["glutathione disulfide"],
            "biological_state": "cytosol",
        }
    )
    payload["locked_reaction_filter_report"] = {
        "locked_reactions_found": 1,
        "exported_locked_reactions": 1,
        "quarantined_locked_reactions": 0,
        "unaccounted_locked_reactions": 0,
    }

    result = quarantine_and_close(payload, strict_db=True)
    stranded = result.payload["quarantined_locked_reactions"]

    assert [row["locked_reaction_id"] for row in stranded] == ["LOCK-1"]
    entries = _quarantine_entries(stranded[0]["original_reaction"])
    assert len(entries) == 1
    assert entries[0]["origin"] == "excluded"


# ── 3. Non-causation: lineage describes decisions, it never makes them ──────


def test_attribution_changes_no_decision_the_stage_made(monkeypatch) -> None:
    """NEW: the control is the same tree with the pass replaced by a no-op.

    Everything the stage produces is compared with lineage stripped out. If the
    pass could move a keep/drop/admit decision, a removal reason, a report field
    or a hash, this is where it would show.
    """

    payload = _payload()
    payload["processes"]["reactions"].append(
        {
            "name": "peripheral oxidation",
            "inputs": ["glutathione"],
            "outputs": ["glutathione disulfide"],
            "biological_state": "cytosol",
        }
    )
    payload["processes"]["reactions"].append(
        {
            "name": "hollow reaction",
            "inputs": [],
            "outputs": ["glutathione"],
            "biological_state": "cytosol",
        }
    )

    attributed = quarantine_and_close(deepcopy(payload), strict_db=True)
    monkeypatch.setattr(
        strict_quarantine, "_attribute_quarantined_processes", lambda *a, **k: 0
    )
    control = quarantine_and_close(deepcopy(payload), strict_db=True)

    assert _strip_lineage(attributed.payload) == control.payload
    assert _strip_lineage(attributed.quarantine_report) == control.quarantine_report
    assert attributed.removed_entity_report == control.removed_entity_report
    assert attributed.closure_report == control.closure_report
    assert attributed.coverage == control.coverage
    # And the control is not vacuous: with the pass on, something was written.
    assert _quarantine_entries(_record(attributed, "hollow reaction")["original_process"])
    assert not _quarantine_entries(_record(control, "hollow reaction")["original_process"])


def test_graph_equivalence_does_not_move_but_the_lineage_stays_detectable() -> None:
    """NEW: PRODUCT_CONTRACT:178, both halves, on the one payload lineage reaches.

    ``canonical_graph_sha256`` is the exporters' hash and an allowlist, so
    ``provenance_lineage`` can never enter it. ``canonical_payload_sha256``
    covers the whole saved artifact, so the same write must be visible there.
    """

    payload = _payload()
    payload["processes"]["reactions"].append(
        {
            "name": "peripheral oxidation",
            "locked_reaction_id": "LOCK-1",
            "inputs": ["glutathione"],
            "outputs": ["glutathione disulfide"],
            "biological_state": "cytosol",
        }
    )
    payload["locked_reaction_filter_report"] = {
        "locked_reactions_found": 1,
        "exported_locked_reactions": 1,
        "quarantined_locked_reactions": 0,
        "unaccounted_locked_reactions": 0,
    }

    saved = quarantine_and_close(payload, strict_db=True).payload
    without = _strip_lineage(saved)

    assert canonical_graph_sha256(saved) == canonical_graph_sha256(without)
    assert canonical_payload_sha256(saved) != canonical_payload_sha256(without)


def test_the_admission_detail_digest_never_carries_the_attribution() -> None:
    """NEW: the pass runs after every ``row_digest`` is taken.

    ``failure_detail.row_digest`` is a DENYLIST, so it copies whatever it finds.
    Writing before the digest was taken would silently move an existing report
    field, ``admissions[*]["detail"]["row"]``.
    """

    result, name = _run_with(QUARANTINED_UNMAPPED_ENTITY)
    record = _record(result, name)

    assert record["detail"]["row"]["outputs"] == ["glutathione disulfide"]
    assert lineage.LINEAGE_KEY not in record["detail"]["row"]


def test_the_caller_s_payload_is_never_written_to() -> None:
    """NEW: the stage is transactional; attribution does not change that."""

    payload = _payload()
    payload["processes"]["reactions"].append(
        {
            "name": "peripheral oxidation",
            "inputs": ["glutathione"],
            "outputs": ["glutathione disulfide"],
            "biological_state": "cytosol",
        }
    )
    before = deepcopy(payload)

    quarantine_and_close(payload, strict_db=True)

    assert payload == before


def test_research_mode_returns_the_candidate_untouched_and_still_records_why() -> None:
    """NEW: research decides and applies nothing -- including this write.

    The graph comes back byte for byte, and the attribution lives only where the
    findings live.
    """

    payload = _payload()
    payload["processes"]["reactions"].append(
        {
            "name": "peripheral oxidation",
            "inputs": ["glutathione"],
            "outputs": ["glutathione disulfide"],
            "biological_state": "cytosol",
        }
    )
    before = deepcopy(payload)

    result = quarantine_and_close(payload, strict_db=True, mode="research")

    assert result.payload == before
    assert lineage.LINEAGE_KEY not in json.dumps(result.payload)
    entries = _quarantine_entries(_record(result, "peripheral oxidation")["original_process"])
    assert len(entries) == 1


# ── 4. The write guard, attacked ────────────────────────────────────────────


def test_an_equal_entry_is_never_appended_twice() -> None:
    """NEW: ``Lineage`` keeps duplicates by design, so the writer must not make one."""

    row: Dict[str, Any] = {"name": "peripheral oxidation"}
    entry = strict_quarantine._quarantine_lineage_entry(
        {"state": QUARANTINED_BROKEN_REFERENCE, "reason": "missing_inputs", "process_kind": "reaction"}
    )

    assert strict_quarantine._record_once(row, entry) is True
    assert strict_quarantine._record_once(row, entry) is False
    assert len(row[lineage.LINEAGE_KEY]) == 1


def test_the_dedupe_guard_is_not_a_blanket_skip() -> None:
    """NEW: non-vacuity. A DIFFERENT entry still gets written onto the same row."""

    row: Dict[str, Any] = {"name": "peripheral oxidation"}
    first = strict_quarantine._quarantine_lineage_entry(
        {"state": QUARANTINED_BROKEN_REFERENCE, "reason": "missing_inputs", "process_kind": "reaction"}
    )
    second = strict_quarantine._quarantine_lineage_entry(
        {"state": QUARANTINED_OUT_OF_SCOPE, "reason": "explicit_out_of_scope_verdict",
         "process_kind": "reaction"}
    )

    assert strict_quarantine._record_once(row, first) is True
    assert strict_quarantine._record_once(row, second) is True
    assert len(row[lineage.LINEAGE_KEY]) == 2


def test_an_earlier_stage_s_attribution_is_re_emitted_not_erased() -> None:
    """NEW: append-only. Quarantine may not overwrite where a row came from."""

    payload = _payload()
    payload["processes"]["reactions"].append(
        {
            "name": "peripheral oxidation",
            "inputs": ["glutathione"],
            "outputs": ["glutathione disulfide"],
            "biological_state": "cytosol",
            lineage.LINEAGE_KEY: [
                lineage.LineageEntry(
                    stage="paper_extraction",
                    origin="paper_stated",
                    support="derived",
                    paper_explicit="explicit",
                    reason="the supplied paper states this reaction",
                    review_required=False,
                ).as_dict()
            ],
        }
    )

    result = quarantine_and_close(payload, strict_db=True)
    stored = _record(result, "peripheral oxidation")["original_process"][lineage.LINEAGE_KEY]

    assert {entry["stage"] for entry in stored} == {"paper_extraction", "quarantine"}


def test_malformed_stored_lineage_is_logged_and_never_aborts_the_stage(caplog) -> None:
    """NEW: a lineage defect may not turn a valid verdict into a crashed run."""

    payload = _payload()
    payload["processes"]["reactions"].append(
        {
            "name": "peripheral oxidation",
            "inputs": ["glutathione"],
            "outputs": ["glutathione disulfide"],
            "biological_state": "cytosol",
            lineage.LINEAGE_KEY: "not a list of entries",
        }
    )

    with caplog.at_level(logging.WARNING, logger="t2pw.pipeline.strict_quarantine"):
        result = quarantine_and_close(payload, strict_db=True)

    assert result.states()["peripheral oxidation"] == QUARANTINED_UNMAPPED_ENTITY
    assert result.ok is True
    assert _quarantine_entries(_record(result, "peripheral oxidation")["original_process"]) == []
    assert any("malformed" in message for message in caplog.messages)


# ── 5. The two boundaries this card deliberately did not cross ──────────────


def test_closure_removed_rows_are_recorded_but_not_lineage_attributed() -> None:
    """NEW: pins the scope decision so a later card changes it deliberately.

    ``_prune_entities`` and its siblings discard the row object and keep only a
    summary, so the sole carrier that survives is a record inside
    ``removed_entity_report.json``. Growing that record is a report-schema
    change, not a lineage write, and it is not taken here.
    """

    result, _name = _run_with(QUARANTINED_UNMAPPED_ENTITY)
    report = result.removed_entity_report

    assert lineage.LINEAGE_KEY not in json.dumps(report)
    # The removal is still recorded -- this is a lineage gap, never a silent drop.
    assert report["schema_version"] == 1
    assert set(report) >= {
        "removed_entities",
        "removed_locations",
        "removed_biological_states",
        "counts",
    }


def test_a_process_stranded_by_closure_is_documented_as_unattributed() -> None:
    """NEW: ``QUARANTINED_DISCONNECTED`` has no attribution, and the map says so.

    Its row was deep-copied into ``originals`` while it was still accepted, and
    the only site that could write it, ``_revalidate_surviving_processes``, is
    outside this card's boundary. Pinned rather than hidden.
    """

    assert strict_quarantine.QUARANTINED_DISCONNECTED in QUARANTINE_STATES
    assert strict_quarantine.QUARANTINED_DISCONNECTED not in strict_quarantine._QUARANTINE_UNCERTAINTY
    assert set(strict_quarantine._QUARANTINE_UNCERTAINTY) == set(_BAD_ROWS)
    # The fallback exists for a state added later without an entry of its own.
    entry = strict_quarantine._quarantine_lineage_entry(
        {"state": strict_quarantine.QUARANTINED_DISCONNECTED, "reason": "stranded_by_closure"}
    )
    assert entry.uncertainty == strict_quarantine._QUARANTINE_UNCERTAINTY_FALLBACK
    assert entry.review_required is True
