"""C-037: lineage records emitted at the gap-resolution stage.

A gap-filled row is not paper-stated content and must never be able to read as such.
R-004 could not tell whether gap-fill independently re-derived three reactions or
whether they leaked through a merge, and the two need different fixes; these records
are what makes the question decidable. The tests are labelled:

* **new acceptance** -- emission that did not exist before this card. No base failure is
  claimed for these beyond "the record is absent", which is what "new" means.
* **preservation** -- the instrumented stage's decisions and outputs are unchanged.
  These pass on base `b5bbf08` AND at the tip; they are the reason instrumentation is
  allowed to be invisible to everything except a lineage reader.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import Any, Dict, List


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

llm_client = types.ModuleType("t2pw.llm.client")
llm_client.chat = lambda *args, **kwargs: "{}"
llm_client.chat_with_tools = lambda *args, **kwargs: "{}"
sys.modules.setdefault("t2pw.llm.client", llm_client)

from t2pw.curation import gap_resolver  # noqa: E402
from t2pw.pipeline.lineage import LINEAGE_KEY  # noqa: E402

ORGANISM = "Pseudomonas putida"
STATE = "Pseudomonas cytosol"

#: The shape C-036 writes when an audit patch rewrites a row, as `lineage.record`
#: serializes it. Seeded directly so this file does not depend on that card's code.
AUDIT_ENTRY: Dict[str, Any] = {
    "stage": "audit_repair",
    "origin": "audit_modified",
    "support": "unsupported",
    "paper_explicit": "not_evaluated",
    "reason": "an audit patch rewrote this row",
    "review_required": False,
    "uncertainty": "",
    "sources": [],
}


def _payload() -> Dict[str, Any]:
    """One protein with a species and a placed location (nothing for this stage to do),
    one without either, a complex with neither, and two compounds -- one unplaced, one
    with a location row whose biological state is blank."""
    return {
        "entities": {
            "species": [
                {
                    "name": ORGANISM,
                    "taxonomy_id": "303",
                    "pathbank_species_id": 7,
                    "domain": "Bacteria",
                }
            ],
            "proteins": [
                {"name": "NdmC", "species": ORGANISM, "mapped_ids": {"uniprot": "Q88FY2"}},
                {"name": "NdmD", "mapped_ids": {"uniprot": "Q88FY1"}},
            ],
            "protein_complexes": [
                {
                    "name": "NdmCDE",
                    "components": ["NdmC", {"name": "NdmD", "stoichiometry": 2}],
                }
            ],
            "compounds": [
                {"name": "theobromine", "class": "compound", "mapped_ids": {"hmdb": "HMDB0002825"}},
                {"name": "caffeine", "class": "compound", "mapped_ids": {"hmdb": "HMDB0001847"}},
            ],
        },
        "biological_states": [
            {"name": STATE, "species": ORGANISM, "subcellular_location": "cytosol"}
        ],
        "element_locations": {
            "compound_locations": [{"compound": "caffeine"}],
            "protein_locations": [{"protein": "NdmC", "biological_state": STATE}],
        },
        "processes": {
            "reactions": [
                {"name": "caffeine demethylation", "inputs": ["caffeine"], "outputs": ["theobromine"]}
            ]
        },
    }


def _pathbank_locations(*args: Any, **kwargs: Any) -> List[Dict[str, Any]]:
    return [
        {
            "location": "cytosol",
            "score": 9,
            "source": "pathbank_db",
            "evidence": "location_frequency=9",
        }
    ]


def _resolve(payload: Dict[str, Any], monkeypatch: Any, **kwargs: Any) -> Any:
    monkeypatch.setattr(gap_resolver, "_db_location_candidates", _pathbank_locations)
    options: Dict[str, Any] = {
        "id_source": "api",
        "use_llm": False,
        "enable_id_resolution": False,
    }
    options.update(kwargs)
    return gap_resolver.resolve_gaps(payload, **options)


def _entries(row: Dict[str, Any]) -> List[Dict[str, Any]]:
    return list(row.get(LINEAGE_KEY) or [])


def _gap_entries(row: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [entry for entry in _entries(row) if entry.get("stage") == "gap_resolution"]


def _every_row(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for bucket in gap_resolver._safe_dict(payload.get("entities")).values():
        rows.extend(row for row in gap_resolver._safe_list(bucket) if isinstance(row, dict))
    rows.extend(
        row for row in gap_resolver._safe_list(payload.get("biological_states"))
        if isinstance(row, dict)
    )
    for bucket in gap_resolver._safe_dict(payload.get("element_locations")).values():
        rows.extend(row for row in gap_resolver._safe_list(bucket) if isinstance(row, dict))
    for bucket in gap_resolver._safe_dict(payload.get("processes")).values():
        rows.extend(row for row in gap_resolver._safe_list(bucket) if isinstance(row, dict))
    return rows


def _stripped(value: Any) -> Any:
    """``value`` with every lineage key removed, at any depth. A no-op on base."""
    if isinstance(value, dict):
        return {k: _stripped(v) for k, v in value.items() if k != LINEAGE_KEY}
    if isinstance(value, list):
        return [_stripped(item) for item in value]
    return value


def _named(rows: List[Dict[str, Any]], field: str, name: str) -> Dict[str, Any]:
    return next(row for row in rows if row.get(field) == name)


# --------------------------------------------------------------------------- #
# new acceptance
# --------------------------------------------------------------------------- #

def test_a_species_filled_from_the_global_organism_is_recorded_as_inferred(monkeypatch: Any) -> None:
    """new acceptance. The paper did not state this protein's organism -- the stage
    carried the pathway's over. `inferred`/`derived` says exactly that, and
    `paper_stated` is never reachable from here."""
    resolved, _ = _resolve(_payload(), monkeypatch)

    filled = _named(resolved["entities"]["proteins"], "name", "NdmD")
    assert filled["organism"] == ORGANISM
    entry = _gap_entries(filled)[0]
    assert entry["origin"] == "inferred"
    assert entry["support"] == "derived"
    assert entry["paper_explicit"] == "not_evaluated"
    assert entry["sources"] == []
    assert "global organism" in entry["reason"]


def test_a_row_this_stage_did_not_change_carries_no_record(monkeypatch: Any) -> None:
    """new acceptance -- an over-attribution guard, so it is vacuous on base (nothing is
    written there) and load-bearing at the tip. F-004: attribution is positional, so it
    may only land on the row that owns the change. NdmC already had a species and a
    placed location; a record on it would be an assertion about a row never touched."""
    resolved, _ = _resolve(_payload(), monkeypatch)

    untouched = _named(resolved["entities"]["proteins"], "name", "NdmC")
    assert LINEAGE_KEY not in untouched
    assert _entries(_named(resolved["entities"]["compounds"], "name", "theobromine")) == []
    assert _entries(_named(resolved["biological_states"], "name", STATE)) == []
    assert _entries(resolved["processes"]["reactions"][0]) == []


def test_a_created_location_row_names_the_database_record_that_placed_it(monkeypatch: Any) -> None:
    """new acceptance. `database_grounded` requires a named source, and the PathBank
    location record is one. Its support is `indirect`: the record reports where the
    compound is seen elsewhere, which backs this placement only by implication."""
    resolved, report = _resolve(_payload(), monkeypatch)

    created = _named(resolved["element_locations"]["compound_locations"], "compound", "theobromine")
    assert created["biological_state"] == STATE
    entry = _gap_entries(created)[0]
    assert (entry["origin"], entry["support"]) == ("database_grounded", "indirect")
    assert entry["review_required"] is False
    assert entry["sources"] == [
        {
            "source_id": "cytosol",
            "source_type": "pathbank_db",
            "uri": "",
            "locator": "compound=theobromine; location_frequency=9",
        }
    ]
    assert report["summary"]["locations_added"] >= 1


def test_a_filled_biological_state_is_recorded_on_the_location_row_it_filled(monkeypatch: Any) -> None:
    """new acceptance. The caffeine location row existed with a blank state; the record
    goes on that row, not on the compound entity, which this stage did not change."""
    resolved, _ = _resolve(_payload(), monkeypatch)

    filled = _named(resolved["element_locations"]["compound_locations"], "compound", "caffeine")
    assert filled["biological_state"] == STATE
    assert "left blank" in _gap_entries(filled)[0]["reason"]
    assert LINEAGE_KEY not in _named(resolved["entities"]["compounds"], "name", "caffeine")


def test_a_defaulted_compartment_is_unsupported_and_asks_for_review(monkeypatch: Any) -> None:
    """new acceptance. No location candidate was retrieved for the complex, so the
    whole-cell default is this stage's own guess. It may not borrow the evidence of a
    candidate that was never selected, and it says it needs review."""
    resolved, _ = _resolve(_payload(), monkeypatch)

    placed = _named(resolved["element_locations"]["protein_locations"], "protein", "NdmCDE")
    entry = _gap_entries(placed)[0]
    assert (entry["origin"], entry["support"]) == ("inferred", "unsupported")
    assert entry["sources"] == []
    assert entry["review_required"] is True
    assert entry["uncertainty"]


def test_a_synthesized_biological_state_is_attributed_to_this_stage(monkeypatch: Any) -> None:
    """new acceptance. An AutoState row is content the payload did not have; without a
    record it is indistinguishable from a state the extraction produced."""
    resolved, _ = _resolve(_payload(), monkeypatch)

    created = [
        state for state in resolved["biological_states"] if state["name"].startswith("AutoState_")
    ]
    assert created, "the complex had no matching state, so one is synthesized"
    entry = _gap_entries(created[0])[0]
    assert (entry["origin"], entry["support"]) == ("inferred", "derived")
    assert entry["review_required"] is True  # inherits the defaulted compartment's doubt


def test_restructured_complex_components_are_recorded_on_the_complex_row(monkeypatch: Any) -> None:
    """new acceptance. The complex row owns its component list. A record on a component
    dict would be a sibling misattribution AND would make the restructuring look like a
    content change to the resolver's own comparison."""
    resolved, _ = _resolve(_payload(), monkeypatch)

    complex_row = resolved["entities"]["protein_complexes"][0]
    reasons = [entry["reason"] for entry in _gap_entries(complex_row)]
    assert any("restructured this complex" in reason for reason in reasons)
    assert all(LINEAGE_KEY not in component for component in complex_row["components"])


def test_identifier_mapping_names_the_records_it_committed(monkeypatch: Any) -> None:
    """new acceptance. A PathBank row IS this entity's record and states the identifier,
    so `direct` is honest and every committed identifier is named as a source."""
    monkeypatch.setattr(gap_resolver, "lookup_compound_api_background", lambda *a, **k: {})
    monkeypatch.setattr(gap_resolver, "lookup_protein_api_background", lambda *a, **k: {})
    monkeypatch.setattr(gap_resolver, "lookup_hmdb_background", lambda *a, **k: {})
    monkeypatch.setattr(
        gap_resolver,
        "_run_id_strategy",
        lambda **kwargs: {
            "attempts": [
                {
                    "status": "mapped",
                    "source": "db",
                    "provider": "PathBankDB",
                    "confidence": 0.93,
                    "mapped_ids": {"pathbank_compound_id": "PW_C123"},
                }
            ]
        },
    )
    payload = _payload()
    payload["entities"]["compounds"].append({"name": "theacrine", "class": "compound"})

    resolved, _ = _resolve(payload, monkeypatch, enable_id_resolution=True)

    mapped = _named(resolved["entities"]["compounds"], "name", "theacrine")
    assert mapped["mapped_ids"] == {"pathbank_compound_id": "PW_C123"}
    entry = next(e for e in _gap_entries(mapped) if e["origin"] == "database_grounded")
    assert entry["support"] == "direct"
    assert entry["sources"] == [
        {
            "source_id": "PW_C123",
            "source_type": "pathbank_compound_id",
            "uri": "",
            "locator": "PathBankDB",
        }
    ]


def test_no_entry_this_stage_writes_can_read_as_paper_stated(monkeypatch: Any) -> None:
    """new acceptance. The card's central prohibition, checked over every row of a real
    resolution rather than at one call site."""
    resolved, _ = _resolve(_payload(), monkeypatch)

    written = [entry for row in _every_row(resolved) for entry in _entries(row)]
    assert written, "the fixture has gaps, so something must have been attributed"
    for entry in written:
        assert entry["stage"] == "gap_resolution"
        assert entry["origin"] != "paper_stated"
        assert entry["paper_explicit"] == "not_evaluated"
        assert entry["origin"] in {"inferred", "database_grounded"}


def test_an_earlier_stages_record_survives_and_is_not_restated(monkeypatch: Any) -> None:
    """new acceptance. C-036 attributes rows an audit patch rewrote. Lineage is
    append-only: that entry must come through untouched, and this stage adds only the
    fact that only it knows -- that a detected gap is why the row now has a species."""
    payload = _payload()
    payload["entities"]["proteins"][1][LINEAGE_KEY] = [dict(AUDIT_ENTRY)]

    resolved, _ = _resolve(payload, monkeypatch)

    entries = _entries(_named(resolved["entities"]["proteins"], "name", "NdmD"))
    assert AUDIT_ENTRY in entries
    assert len(entries) == 2
    assert [entry["stage"] for entry in entries] == ["gap_resolution", "audit_repair"]
    assert len(_gap_entries(_named(resolved["entities"]["proteins"], "name", "NdmD"))) == 1


def test_rows_only_an_enrichment_patch_changed_are_not_attributed_here(monkeypatch: Any) -> None:
    """new acceptance, no double-attribution -- likewise a guard, vacuous on base and
    load-bearing at the tip. A patch names a JSON pointer, not a row. This stage cannot
    honestly say which row a pointer owns, so it says nothing and leaves the patch
    applier -- the writer that does know -- to attribute what it rewrote."""
    def fake_agent(*, tools: Any, tool_executor: Any, **kwargs: Any) -> str:
        tool_executor(
            "propose_patch",
            {
                "op": "add",
                "path": "/entities/species/0/domain",
                "value": "Bacteria (verified)",
                "evidence": "lookup_species confirmed the domain for taxonomy 303.",
                "confidence": 0.97,
            },
        )
        return "{}"

    monkeypatch.setattr(gap_resolver, "chat", lambda *a, **k: "{}")
    monkeypatch.setattr(gap_resolver, "chat_with_tools", fake_agent)
    payload = _payload()
    payload["entities"]["species"][0].pop("pathbank_species_id")

    resolved, report = _resolve(payload, monkeypatch, use_llm=True)

    assert report["enrichment"]["patches_proposed"] == 1
    species_row = resolved["entities"]["species"][0]
    assert species_row["domain"] == "Bacteria (verified)"
    assert LINEAGE_KEY not in species_row


def test_a_malformed_inbound_lineage_is_reported_not_raised(monkeypatch: Any) -> None:
    """new acceptance. Recording provenance must never change which gaps are resolved,
    so a bad inbound record cannot be allowed to kill the stage -- and it cannot be
    dropped in silence either, because a missing entry reads as "this was original"."""
    payload = _payload()
    payload["entities"]["proteins"][1][LINEAGE_KEY] = [dict(AUDIT_ENTRY, stage="pwml_export")]

    resolved, report = _resolve(payload, monkeypatch)

    filled = _named(resolved["entities"]["proteins"], "name", "NdmD")
    assert filled["organism"] == ORGANISM, "the gap is still resolved"
    assert filled[LINEAGE_KEY] == [dict(AUDIT_ENTRY, stage="pwml_export")], "left as found"
    assert report["lineage_errors"]
    assert "stage" in report["lineage_errors"][0]


# --------------------------------------------------------------------------- #
# preservation -- passes on base b5bbf08 and at the tip
# --------------------------------------------------------------------------- #

def test_which_gaps_are_detected_and_how_they_resolve_is_unchanged(monkeypatch: Any) -> None:
    """preservation. Every decision this stage makes, pinned: the issues it found, the
    counts it reports, the rows it wrote and the compartments it chose. With lineage
    stripped the resolved payload is byte-identical to what base produces."""
    payload = _payload()
    resolved, report = _resolve(payload, monkeypatch)

    assert [issue["issue_key"] for issue in report["stage3"]["issues"]] == [
        "protein:ndmd",
        "compound:theobromine",
        "compound:caffeine",
        "protein_complex:ndmcde",
    ]
    assert report["summary"] == {
        "mapped_ids_added": 0,
        "organisms_added": 2,
        "locations_added": 3,
        "location_states_filled": 1,
        "complex_components_structured": 1,
        "complex_components_resolved": 2,
        "complex_stoichiometry_issues_deferred": 1,
        "items_considered": 4,
        "enrichment_patches_accepted": 0,
        "enrichment_patches_rejected": 0,
    }
    assert _stripped(resolved["element_locations"]) == {
        "compound_locations": [
            {"compound": "caffeine", "biological_state": STATE},
            {"compound": "theobromine", "biological_state": STATE},
        ],
        "protein_locations": [
            {"protein": "NdmC", "biological_state": STATE},
            {"protein": "NdmD", "biological_state": STATE},
            {"protein": "NdmCDE", "biological_state": "AutoState_pseudomonas_putidacell"},
        ],
    }
    assert _stripped(resolved["biological_states"]) == [
        {"name": STATE, "species": ORGANISM, "subcellular_location": "cytosol"},
        {
            "name": "AutoState_pseudomonas_putidacell",
            "subcellular_location": "cell",
            "species": ORGANISM,
        },
    ]
    assert _stripped(resolved["entities"]["protein_complexes"][0]["components"]) == [
        {"name": "NdmC", "mapped_ids": {"uniprot": "Q88FY2"}, "mapping_status": "mapped"},
        {
            "name": "NdmD",
            "stoichiometry": 2,
            "mapped_ids": {"uniprot": "Q88FY1"},
            "mapping_status": "mapped",
        },
    ]
    assert resolved["processes"] == _payload()["processes"]


def test_a_payload_with_nothing_to_resolve_is_returned_untouched(monkeypatch: Any) -> None:
    """preservation. Nothing was gap-filled, so nothing is attributed and nothing is
    added -- instrumentation must not manufacture either rows or provenance."""
    payload = _payload()
    payload["entities"]["proteins"] = [payload["entities"]["proteins"][0]]
    payload["entities"]["protein_complexes"] = []
    payload["entities"]["compounds"] = []
    payload["element_locations"]["compound_locations"] = []

    resolved, report = _resolve(payload, monkeypatch)

    assert report["stage3"]["issues"] == []
    assert all(LINEAGE_KEY not in row for row in _every_row(resolved))
    assert "lineage_errors" not in report
    assert _stripped(resolved) == resolved
    assert resolved["biological_states"] == payload["biological_states"]
