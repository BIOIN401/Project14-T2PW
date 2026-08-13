"""NEW ACCEPTANCE (C-034): Stage-1 exit-gate provenance lineage.

**Every assertion in this file is new capability, not a regression fix.** At the
card's base SHA ``bcc0bfe`` ``settle_stage_one`` writes no lineage at all, so
every test here fails there for the one uninteresting reason that the feature is
absent. They are labelled as new acceptance deliberately: the behaviour that is
*preserved* rather than added -- outcome, status, diagnostics and every
non-lineage field -- is proved separately and behaviourally by
``docs/pwml_recovery_sprint/evidence/c034_boundary_golden.py``, which must
reproduce byte-identically at base AND at tip.

``PRODUCT_CONTRACT`` § 3 requires false content to be attributable *empirically*
to Stage 1. This boundary is the one place that can tell the three Stage-1
origins apart while they are still distinguishable: the rows the model drew, the
registry shells deterministic reconstruction added, and the rows localized
repair rewrote. Downstream they are indistinguishable payload rows.

Offline: reconstruction uses no model, repairs go through an injected ``chat_fn``.
"""

from __future__ import annotations

import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from t2pw.pipeline.canonical_hash import (  # noqa: E402
    canonical_graph_sha256,
    canonical_payload_sha256,
)
from t2pw.pipeline.extraction_diagnostics import (  # noqa: E402
    OUTCOME_CONTRACT_FAILED,
    OUTCOME_OK,
    activate,
    deactivate,
    payload_hash,
)
from t2pw.pipeline.lineage import LINEAGE_KEY, STAGES  # noqa: E402
from t2pw.pipeline.localized_repair import SHELL_PROVENANCE  # noqa: E402
from t2pw.pipeline.stage_one_boundary import settle_stage_one  # noqa: E402


@pytest.fixture(autouse=True)
def _clean_recorder():
    deactivate()
    yield
    deactivate()


class _Reply:
    def __init__(self, text: str, **diagnostics: Any) -> None:
        self.text = text
        self.diagnostics = {
            "model": "test-model",
            "finish_reason": "stop",
            "attempts": 1,
            "response_status": "ok",
            "terminal_reason": "",
            "attempt_log": [],
            **diagnostics,
        }


class _Provider:
    def __init__(self, *replies: _Reply) -> None:
        self._replies = list(replies)
        self.prompts: List[str] = []

    def __call__(self, messages: List[Dict[str, str]], **_: Any) -> _Reply:
        self.prompts.append(messages[-1]["content"])
        if not self._replies:
            raise AssertionError("the boundary made more repair draws than were scripted")
        return self._replies.pop(0)


def _multi_bucket() -> Dict[str, Any]:
    """Rows in three buckets across both sections, and one missing participant.

    ``ADP`` is named as a reaction output and is absent from the registry, so
    ``reconstruct_registry_shells`` adds exactly one shell -- which is what lets
    one payload exercise the paper-stated and the reconstructed attribution at
    the same time, on rows that must not be confused for one another.
    """

    return {
        "entities": {
            "compounds": [{"name": "ATP"}],
            "proteins": [{"name": "LpxA", "evidence": "LpxA acetylates UDP-GlcNAc."}],
        },
        "processes": {
            "reactions": [
                {
                    "name": "hydrolysis",
                    "inputs": ["ATP"],
                    "outputs": ["ADP"],
                    "evidence": "ATP is hydrolysed to ADP.",
                }
            ]
        },
    }


_NAMELESS = {
    "entities": {"proteins": [{"name": "", "evidence": "LpxA acetylates UDP-GlcNAc."}]},
    "processes": {
        "reactions": [
            {"name": "r", "inputs": ["UDP-GlcNAc"], "outputs": ["product"], "evidence": "e"}
        ]
    },
}


def _entries(row: Any) -> List[Dict[str, Any]]:
    return list(row.get(LINEAGE_KEY) or []) if isinstance(row, dict) else []


def _pairs(row: Any) -> List[tuple]:
    """``(stage, origin)`` for each of ``row``'s lineage entries, in stored order."""

    return [(e.get("stage"), e.get("origin")) for e in _entries(row)]


def _all_rows(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for section in ("entities", "processes"):
        for bucket in (payload.get(section) or {}).values():
            rows.extend(item for item in (bucket or []) if isinstance(item, dict))
    return rows


def _strip_lineage(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _strip_lineage(v) for k, v in value.items() if k != LINEAGE_KEY}
    if isinstance(value, list):
        return [_strip_lineage(item) for item in value]
    return value


# ---------------------------------------------------------------------------
# A1 -- every row the caller supplied is attributed to the Stage-1 draw.
# ---------------------------------------------------------------------------
def test_a1_every_inbound_entity_and_process_row_is_attributed_to_paper_extraction() -> None:
    outcome = settle_stage_one(_multi_bucket(), chat_fn=_Provider())

    assert outcome.ok and outcome.outcome == OUTCOME_OK
    settled = outcome.payload
    supplied = [
        settled["entities"]["compounds"][0],
        settled["entities"]["proteins"][0],
        settled["processes"]["reactions"][0],
    ]
    assert len(supplied) == 3  # rows in more than one bucket AND both sections
    for row in supplied:
        assert ("paper_extraction", "paper_stated") in _pairs(row), row.get("name")

    stated = [e for e in _entries(supplied[0]) if e["origin"] == "paper_stated"][0]
    # No source is invented for it: this seam never sees the paper.
    assert stated["sources"] == []
    assert stated["support"] == "unsupported"
    assert stated["paper_explicit"] == "explicit"
    assert stated["review_required"] is False
    assert stated["reason"]


def _self_declared() -> Dict[str, Any]:
    """Rows the extraction itself marked as not read off the page.

    ``pwml_system.txt:109/:116/:401`` tells the model to CREATE a complex whose
    subunit membership it could not read, and to mark it ``provenance:
    "inferred"`` with ``confidence < 1.0``; a process row carries a free-text
    ``inference`` note instead. Both markers reach this seam intact.
    """

    return {
        "entities": {
            "compounds": [{"name": "ATP"}, {"name": "ADP"}],
            "protein_complexes": [
                {"name": "ATP synthase", "provenance": "inferred", "confidence": 0.6}
            ],
            "proteins": [
                {"name": "LpxA", "provenance": "extracted", "confidence": 1.0}
            ],
        },
        "processes": {
            "reactions": [
                {
                    "name": "hydrolysis",
                    "inputs": ["ATP"],
                    "outputs": ["ADP"],
                    "evidence": "ATP is hydrolysed to ADP.",
                    "inference": "directionality inferred from context",
                    "confidence": 0.7,
                }
            ]
        },
    }


def _stated(row: Any) -> Dict[str, Any]:
    """The row's single ``paper_extraction`` entry."""

    entries = [e for e in _entries(row) if e["stage"] == "paper_extraction"]
    assert len(entries) == 1, entries
    return entries[0]


def test_a1_a_row_the_extraction_marked_inferred_is_not_reported_as_paper_explicit() -> None:
    """The prompt manufactures marked content; the lineage must not unmark it.

    A ``protein_complexes`` row with ``confidence: 0.6`` and ``provenance:
    "inferred"`` says, in the payload's own words, that the model reasoned it
    rather than read it. ``PRODUCT_CONTRACT`` § 3 exists so an investigator
    tracing a bad complex learns exactly that.
    """

    outcome = settle_stage_one(_self_declared(), chat_fn=_Provider())

    assert outcome.ok and outcome.outcome == OUTCOME_OK
    entry = _stated(outcome.payload["entities"]["protein_complexes"][0])
    assert entry["paper_explicit"] == "not_evaluated"   # never "not_explicit"
    assert entry["review_required"] is True
    assert entry["uncertainty"]
    # The § 2 origin table is fixed: HOW the model got there is not a reason to
    # move the row out of the Stage-1 draw.
    assert (entry["stage"], entry["origin"]) == ("paper_extraction", "paper_stated")


def test_a1_a_process_row_carrying_an_inference_note_is_not_reported_as_paper_explicit() -> None:
    outcome = settle_stage_one(_self_declared(), chat_fn=_Provider())

    entry = _stated(outcome.payload["processes"]["reactions"][0])
    assert entry["paper_explicit"] == "not_evaluated"
    assert entry["review_required"] is True
    assert entry["origin"] == "paper_stated"


def test_a1_an_extracted_row_and_an_unmarked_row_are_still_reported_as_explicit() -> None:
    """The downgrade is matched on specific values, not on "not extracted"."""

    outcome = settle_stage_one(_self_declared(), chat_fn=_Provider())

    for row in (
        outcome.payload["entities"]["proteins"][0],      # provenance: "extracted"
        outcome.payload["entities"]["compounds"][0],     # no marker at all
    ):
        entry = _stated(row)
        assert entry["paper_explicit"] == "explicit"
        assert entry["review_required"] is False


@pytest.mark.parametrize(
    "marker",
    [
        {"provenance": "enriched"},                       # from an API lookup
        {"provenance": "INFERRED"},                       # case is not a loophole
        {"rag_provenance": {"source_id": "PMC1"}},        # C-038's carrier
    ],
)
def test_a1_every_not_read_marker_downgrades_the_paper_explicit_claim(
    marker: Dict[str, Any]
) -> None:
    payload = _multi_bucket()
    payload["entities"]["proteins"][0].update(marker)

    outcome = settle_stage_one(payload, chat_fn=_Provider())

    entry = _stated(outcome.payload["entities"]["proteins"][0])
    assert entry["paper_explicit"] == "not_evaluated"
    assert entry["review_required"] is True


# ---------------------------------------------------------------------------
# A2 -- a reconstructed shell is inferred, and is never also paper-stated.
# ---------------------------------------------------------------------------
def test_a2_an_inbound_row_carrying_the_shell_marker_is_never_called_paper_stated() -> None:
    """The content-derived half of the partition, independent of the index half.

    ``_row_census`` is exact against ``localized_repair`` as it stands, but that
    module is not this card's to pin. A shell sitting BELOW the inbound count --
    which is what a future dedupe, sort or drop in reconstruction would produce,
    and what a resumed run that lost its lineage key already produces -- must
    still not be reported as something the paper stated.
    """

    payload = _multi_bucket()
    payload["entities"]["compounds"].insert(
        0,
        {"name": "ADP", "provenance": SHELL_PROVENANCE, "resolution_status": "unresolved"},
    )

    outcome = settle_stage_one(payload, chat_fn=_Provider())

    compounds = outcome.payload["entities"]["compounds"]
    assert compounds[0]["name"] == "ADP"
    assert _pairs(compounds[0]) == []        # index 0, inbound, and no claim made
    assert ("paper_extraction", "paper_stated") in _pairs(compounds[1])   # ATP still is


# ---------------------------------------------------------------------------
# A2 -- a reconstructed shell is inferred, and is never also paper-stated.
# ---------------------------------------------------------------------------
def test_a2_a_reconstructed_shell_is_inferred_and_never_also_paper_stated() -> None:
    outcome = settle_stage_one(_multi_bucket(), chat_fn=_Provider())

    compounds = outcome.payload["entities"]["compounds"]
    assert [row["name"] for row in compounds] == ["ATP", "ADP"]
    shell = compounds[1]
    assert shell["provenance"] == "reconstructed_from_process_participant"

    assert _pairs(shell) == [("normalization", "inferred")]
    # The disjointness both ways: the shell is not paper-stated, and the row the
    # caller actually supplied is not reported as reconstructed.
    assert ("paper_extraction", "paper_stated") not in _pairs(shell)
    assert ("normalization", "inferred") not in _pairs(compounds[0])

    entry = _entries(shell)[0]
    assert entry["support"] == "derived"          # a pointer demanded it; nothing states it
    assert entry["paper_explicit"] == "not_evaluated"   # never "not_explicit"
    assert entry["review_required"] is True and entry["uncertainty"]
    assert entry["sources"] == []


def test_a2_no_row_anywhere_carries_both_attributions() -> None:
    outcome = settle_stage_one(_multi_bucket(), chat_fn=_Provider())

    for row in _all_rows(outcome.payload):
        pairs = set(_pairs(row))
        assert not (
            {("paper_extraction", "paper_stated"), ("normalization", "inferred")} <= pairs
        ), row


# ---------------------------------------------------------------------------
# A3 -- repair adds to the origin, it does not replace it.
# ---------------------------------------------------------------------------
def _repair_provider() -> _Provider:
    return _Provider(
        _Reply(
            json.dumps(
                {
                    "repaired_rows": [
                        {
                            "pointer": "/entities/proteins/0",
                            "row": {
                                "name": "LpxA",
                                "evidence": "LpxA acetylates UDP-GlcNAc.",
                            },
                        }
                    ]
                }
            )
        )
    )


def test_a3_a_repaired_row_keeps_its_origin_and_gains_audit_repair_in_stage_order() -> None:
    provider = _repair_provider()

    outcome = settle_stage_one(deepcopy(_NAMELESS), chat_fn=provider)

    assert outcome.ok
    assert outcome.repair.repaired_pointers == ["/entities/proteins/0"]
    repaired = outcome.payload["entities"]["proteins"][0]
    assert repaired["name"] == "LpxA"

    pairs = _pairs(repaired)
    assert ("paper_extraction", "paper_stated") in pairs   # origin survived the rewrite
    assert ("audit_repair", "audit_modified") in pairs
    # STAGES-canonical order, and it is the STORED order: lineage.as_list sorts.
    assert pairs == sorted(pairs, key=lambda p: STAGES.index(p[0]))
    assert [p[0] for p in pairs] == ["paper_extraction", "audit_repair"]

    audit = [e for e in _entries(repaired) if e["stage"] == "audit_repair"][0]
    assert audit["support"] == "unsupported"       # a model produced it; nothing derived it
    assert audit["review_required"] is True and audit["uncertainty"]
    assert audit["sources"] == []

    # A row repair never touched carries no audit entry.
    reaction = outcome.payload["processes"]["reactions"][0]
    assert [p[0] for p in _pairs(reaction)] == ["paper_extraction"]


def test_a3_the_repair_prompt_never_carried_lineage() -> None:
    """Lineage is written after the decisions, so it cannot steer one.

    If it were written before repair, the row in the repair request would carry
    ``provenance_lineage`` -- and ``preserves_original_values`` would then refuse
    every returned row that did not echo it back, turning repairs that used to
    succeed into rejections.
    """

    provider = _repair_provider()

    settle_stage_one(deepcopy(_NAMELESS), chat_fn=provider)

    assert provider.prompts
    assert all(LINEAGE_KEY not in prompt for prompt in provider.prompts)


# ---------------------------------------------------------------------------
# A4 (in-process half) -- the boundary diagnostic describes the payload it
# judged. The behavioural base proof is the golden capture; see the module
# docstring.
# ---------------------------------------------------------------------------
def test_a4_the_boundary_record_hashes_the_payload_without_lineage(tmp_path: Path) -> None:
    recorder = activate(run_id="c034", artifact_dir=tmp_path)

    outcome = settle_stage_one(_multi_bucket(), chat_fn=_Provider())

    boundaries = recorder.boundaries
    assert len(boundaries) == 1
    assert boundaries[0]["outcome"] == OUTCOME_OK
    assert boundaries[0]["response_hash"] == payload_hash(_strip_lineage(outcome.payload))
    assert boundaries[0]["response_hash"] != payload_hash(outcome.payload)


def test_a4_the_outcome_summary_hash_covers_lineage_deliberately() -> None:
    """The one observable value C-034 moves, pinned so it cannot move by accident.

    ``to_summary()["payload_hash"]`` fingerprints the whole returned payload, so
    a payload that gained lineage gets a new one. That is ``PRODUCT_CONTRACT``
    line 178 -- "lineage changes must remain detectable" -- and not a decision
    change: it describes the artifact, it is not an input to any judgement, and
    the fingerprint the boundary DECIDED on is the one above, taken before these
    writes. The golden capture reports this delta by name rather than absorbing
    it; this test is what keeps the two statements honest with each other.
    """

    outcome = settle_stage_one(_multi_bucket(), chat_fn=_Provider())

    assert outcome.to_summary()["payload_hash"] == payload_hash(outcome.payload)
    assert outcome.to_summary()["payload_hash"] != payload_hash(
        _strip_lineage(outcome.payload)
    )
    # Counts are what the summary reports about content, and lineage is not
    # content: reconstruction added one shell, and nothing else moved.
    assert outcome.to_summary()["entity_counts"]["total"] == 3
    assert outcome.to_summary()["process_counts"]["total"] == 1


# ---------------------------------------------------------------------------
# A5 -- lineage cannot be a way to talk the gate into a pass.
# ---------------------------------------------------------------------------
def test_a5_an_incomplete_outcome_stays_incomplete_with_lineage_present() -> None:
    payload = deepcopy(_NAMELESS)

    outcome = settle_stage_one(payload, repair_rows=False)

    assert outcome.ok is False
    assert outcome.outcome == OUTCOME_CONTRACT_FAILED
    assert outcome.incomplete_reason
    assert outcome.failure is not None
    # The payload came back as far as it got, attributed -- and no larger.
    assert len(outcome.payload["processes"]["reactions"]) == 1
    assert outcome.payload["entities"]["proteins"][0]["name"] == ""
    assert ("paper_extraction", "paper_stated") in _pairs(
        outcome.payload["entities"]["proteins"][0]
    )
    # Nothing was invented to satisfy the contract: every added row is a declared
    # shell, and every shell says so.
    for row in _all_rows(outcome.payload):
        if ("normalization", "inferred") in _pairs(row):
            assert row.get("resolution_status") == "unresolved"


# ---------------------------------------------------------------------------
# A6 -- settling twice records the same facts once.
# ---------------------------------------------------------------------------
def test_a6_settling_the_same_payload_twice_does_not_duplicate_any_entry() -> None:
    first = settle_stage_one(_multi_bucket(), chat_fn=_Provider())
    second = settle_stage_one(deepcopy(first.payload), chat_fn=_Provider())

    assert second.ok and second.outcome == first.outcome
    assert json.dumps(second.payload, sort_keys=True) == json.dumps(
        first.payload, sort_keys=True
    )
    for row in _all_rows(second.payload):
        entries = _entries(row)
        assert len(entries) == len({json.dumps(e, sort_keys=True) for e in entries}), row

    # The shell stays inferred on the second pass: on that pass it IS a row
    # "present in the input", and calling it paper-stated would be false.
    assert _pairs(second.payload["entities"]["compounds"][1]) == [
        ("normalization", "inferred")
    ]


# ---------------------------------------------------------------------------
# A7 -- lineage another stage wrote is kept, never overwritten.
# ---------------------------------------------------------------------------
_INBOUND_ENTRY = {
    "stage": "rag_retrieval",
    "origin": "rag_literature",
    "support": "direct",
    "paper_explicit": "not_explicit",
    "reason": "admitted from retrieved literature before this run resumed",
    "review_required": False,
    "uncertainty": "",
    "sources": [
        {"source_id": "PMC123456", "source_type": "article", "uri": "", "locator": "results"}
    ],
}


def test_a7_a_row_arriving_with_lineage_keeps_it_and_gains_this_stages_entry() -> None:
    payload = _multi_bucket()
    payload["entities"]["proteins"][0][LINEAGE_KEY] = [dict(_INBOUND_ENTRY)]

    outcome = settle_stage_one(payload, chat_fn=_Provider())

    row = outcome.payload["entities"]["proteins"][0]
    entries = _entries(row)
    assert _INBOUND_ENTRY in entries          # byte-for-byte, not re-derived
    assert len(entries) == 2
    assert ("paper_extraction", "paper_stated") in _pairs(row)


def test_a7_a_row_whose_stored_lineage_is_malformed_does_not_abort_the_boundary() -> None:
    """A payload defect must not turn a recoverable boundary into a crash."""

    payload = _multi_bucket()
    payload["entities"]["proteins"][0][LINEAGE_KEY] = [{"not": "an entry"}]

    outcome = settle_stage_one(payload, chat_fn=_Provider())

    assert outcome.ok and outcome.outcome == OUTCOME_OK
    assert outcome.payload["entities"]["proteins"][0][LINEAGE_KEY] == [{"not": "an entry"}]
    # Every other row is still attributed.
    assert ("paper_extraction", "paper_stated") in _pairs(
        outcome.payload["entities"]["compounds"][0]
    )


# ---------------------------------------------------------------------------
# A8 -- lineage moves the payload hash and never the graph hash.
# ---------------------------------------------------------------------------
def test_a8_lineage_is_outside_the_graph_hash_and_inside_the_payload_hash() -> None:
    outcome = settle_stage_one(_multi_bucket(), chat_fn=_Provider())
    settled = outcome.payload
    stripped = _strip_lineage(settled)

    assert any(_entries(row) for row in _all_rows(settled))  # there IS lineage to detect
    assert canonical_graph_sha256(settled) == canonical_graph_sha256(stripped)
    assert canonical_payload_sha256(settled) != canonical_payload_sha256(stripped)


# ---------------------------------------------------------------------------
# A9 -- the writes land only on the payload the boundary owns.
# ---------------------------------------------------------------------------
def test_a9_the_callers_payload_and_its_row_objects_are_never_written_to() -> None:
    payload = _multi_bucket()
    held_row = payload["entities"]["compounds"][0]
    before = json.dumps(payload, sort_keys=True)

    outcome = settle_stage_one(payload, chat_fn=_Provider())

    assert json.dumps(payload, sort_keys=True) == before
    assert LINEAGE_KEY not in held_row
    assert outcome.payload is not payload
    assert outcome.payload["entities"]["compounds"][0] is not held_row
    assert LINEAGE_KEY in outcome.payload["entities"]["compounds"][0]


def test_a9_the_repair_path_also_leaves_the_callers_payload_alone() -> None:
    payload = deepcopy(_NAMELESS)
    held_row = payload["entities"]["proteins"][0]
    before = json.dumps(payload, sort_keys=True)

    outcome = settle_stage_one(payload, chat_fn=_repair_provider())

    assert json.dumps(payload, sort_keys=True) == before
    assert LINEAGE_KEY not in held_row
    assert LINEAGE_KEY in outcome.payload["entities"]["proteins"][0]


# ---------------------------------------------------------------------------
# Shapes that must not be annotated into something else.
# ---------------------------------------------------------------------------
def test_a_bare_string_row_is_skipped_rather_than_converted_to_an_object() -> None:
    payload = _multi_bucket()
    payload["entities"]["compounds"] = ["ATP"]

    outcome = settle_stage_one(payload, chat_fn=_Provider())

    assert outcome.payload["entities"]["compounds"][0] == "ATP"


def test_a_payload_that_is_not_a_dict_settles_without_attribution() -> None:
    outcome = settle_stage_one([], chat_fn=_Provider())  # type: ignore[arg-type]

    assert outcome.payload == {} or isinstance(outcome.payload, dict)
