"""Identifier-mapping provenance lineage (C-044).

Two kinds of test live here and they are labelled, because G9 treats them
differently:

``test_new_*`` are **NEW ACCEPTANCE** tests for a capability that did not exist
before this card. They fail at ``bcc0bfe`` for the plain reason that Stage 2
wrote no ``provenance_lineage`` at all -- not because anything regressed. No
base failure is fabricated for them and none is claimed.

``test_preservation_*`` is the **G9 behavioural preservation proof** for A5: the
complete observable output of ``map_payload`` with ``provenance_lineage``
stripped, over a battery of deliberately non-canonical fixtures, byte-identical
to the golden captured by running the same battery at ``bcc0bfe``. That golden
is committed next to this file and was produced from base *first*.

Every assertion is about what **this stage** recorded. None of them asserts that
nobody else attributed a row: an attribution from another stage is exactly what
a concurrent card is expected to add, so "nobody did X" would be green here and
red at the merge.
"""

from __future__ import annotations

import dataclasses
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import patch

import pytest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# Only base-available symbols are imported at module level: the G9 golden
# generator imports ``preservation_fixtures`` / ``preservation_projection`` from
# this file while pointing ``t2pw`` at the base export, and an import of a
# symbol this card introduces would make that impossible. This card's two
# helpers are reached through the module (``map_ids._record_mapping_lineage``).
from t2pw.mapping import map_ids  # noqa: E402
from t2pw.mapping.map_ids import map_payload  # noqa: E402
from t2pw.pipeline.canonical_hash import (  # noqa: E402
    canonical_graph_sha256,
    canonical_payload_sha256,
)
from t2pw.pipeline.lineage import (  # noqa: E402
    LINEAGE_KEY,
    LineageEntry,
    LineageError,
    read as read_lineage,
)

GOLDEN = Path(__file__).with_name("test_map_ids_mapping_lineage_golden.json")

_STAGE = "identifier_mapping"


# ── helpers ──────────────────────────────────────────────────────────────────


def _mapping_entries(row: Dict[str, Any]) -> List[LineageEntry]:
    """Only what THIS stage recorded. Never 'the row has exactly one entry'."""
    return [e for e in read_lineage(row).canonical() if e.stage == _STAGE]


def _only(row: Dict[str, Any]) -> LineageEntry:
    entries = _mapping_entries(row)
    assert len(entries) == 1, f"expected one {_STAGE} entry, got {len(entries)}"
    return entries[0]


def _entity_rows(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for rows in (payload.get("entities") or {}).values():
        if isinstance(rows, list):
            out.extend(r for r in rows if isinstance(r, dict))
    return out


def _strip_lineage(value: Any) -> Any:
    """``value`` with every ``provenance_lineage`` key removed, at any depth."""
    if isinstance(value, dict):
        return {k: _strip_lineage(v) for k, v in value.items() if k != LINEAGE_KEY}
    if isinstance(value, list):
        return [_strip_lineage(v) for v in value]
    return value


def _protein_row(**overrides: Any) -> Dict[str, Any]:
    row = {"name": "AcpP", "mapping_meta": {}}
    row.update(overrides)
    return row


# ── the offline runner shared by the preservation battery ────────────────────


class _NoNetwork(RuntimeError):
    """Raised if anything under test reaches for the network."""


def run_offline(payload: Dict[str, Any], cache_path: Path) -> Dict[str, Any]:
    """``map_payload`` with every external door shut.

    ``id_source="db"`` with no resolver, no LLM, no NCBI and an ``HttpClient``
    that raises: whatever comes out is a function of the input alone, which is
    what makes a byte-comparison against a golden meaningful. The synonym
    lookup is shut separately because it reaches a model rather than
    ``HttpClient``, and a sampled answer would make the golden a coin toss.
    """
    env = {"T2PW_SPECIES_LLM": "0", "T2PW_SPECIES_NCBI": "0"}
    with patch.dict(os.environ, env), patch.object(
        map_ids.PathBankDbResolver, "from_env", classmethod(lambda cls, overrides=None: None)
    ), patch.object(
        map_ids, "_ai_protein_synonym_lookup", return_value=[]
    ), patch.object(
        map_ids.HttpClient, "get", side_effect=_NoNetwork("network call during an offline run")
    ):
        return map_payload(
            payload,
            cache_path=cache_path,
            id_source="db",
            use_cache=False,
        )


def preservation_fixtures() -> Dict[str, Dict[str, Any]]:
    """Deliberately NON-canonical documents: unsorted keys, padded and
    unicode-bearing names, nulls, an empty bucket, a bucket this mapper does not
    resolve, mixed-type identifiers, and rows carrying identifiers an *earlier*
    pass already wrote. Round-tripping the mapper's own output would hide a
    re-serialization; none of these is the mapper's output."""
    return {
        "carried_ids": {
            "topic": "enterobactin — biosynthesis",
            "entities": {
                "proteins": [
                    {
                        "mapped_ids": {"uniprot": "P0A9A9"},
                        "name": "  Fur  ",
                        "organism": "Escherichia coli",
                    },
                    {"name": "EntB", "mapped_ids": {}, "notes": None},
                    {
                        "name": "Unknown",
                        "mapped_ids": {"uniprot": "Unknown", "pathbank_protein_id": 0},
                    },
                ],
                "compounds": [
                    {"name": "PhoP", "mapped_ids": {"kegg": "C00003", "chebi": "CHEBI:15846"}},
                    {"synonyms": ["ent"], "name": "enterobactin"},
                ],
                "protein_complexes": [],
                "cell_types": [{"name": "hepatocyte"}],
            },
            "processes": {"reactions": [], "transports": [], "interactions": []},
        },
        "bare_names": {
            "entities": {
                "compounds": [{"name": "2,3-dihydroxybenzoate"}, {"name": "  ATP "}],
                "proteins": [{"name": "EntA", "species": "Escherichia coli"}],
            },
            "processes": {"reactions": []},
            "topic": None,
        },
        "empty": {"entities": {}, "processes": {}},
    }


def preservation_projection(cache_dir: Path) -> str:
    """The battery's complete observable output, lineage stripped, as bytes."""
    captured: Dict[str, Any] = {}
    for name, payload in sorted(preservation_fixtures().items()):
        captured[name] = _strip_lineage(
            run_offline(payload, cache_dir / f"{name}_cache.json")
        )
    return json.dumps(captured, indent=2, ensure_ascii=False, sort_keys=True)


# ── NEW ACCEPTANCE — A1: grounded identities are attributed ──────────────────


def test_new_matched_row_is_database_grounded_naming_the_actual_record() -> None:
    row = _protein_row(
        name="MCR-1",
        mapped_ids={"uniprot": "A0A0R6L508", "pathbank_protein_id": 4821},
        mapping_meta={
            "identity_status": "verified",
            "resolution": {"status": "matched", "order_step": "exact_protein_name_species"},
        },
    )
    assert map_ids._record_mapping_lineage({"proteins": [row]})["attributed"] == 1
    entry = _only(row)

    assert (entry.origin, entry.support) == ("database_grounded", "direct")
    assert entry.paper_explicit == "not_evaluated"  # never "not_explicit"
    assert not entry.review_required
    named = {(s.source_type, s.source_id) for s in entry.sources}
    assert named == {("uniprot", "A0A0R6L508"), ("pathbank_protein_id", "4821")}


def test_new_support_is_indirect_when_the_record_was_reached_by_synonym() -> None:
    """A record found through a synonym supports the identity by implication."""
    row = {
        "name": "acyl carrier protein",
        "mapped_ids": {"pathbank_compound_id": 721},
        "mapping_meta": {
            "resolution": {"status": "matched", "order_step": "synonym"},
        },
    }
    map_ids._record_mapping_lineage({"compounds": [row]})
    assert _only(row).support == "indirect"


def test_new_buckets_this_stage_does_not_resolve_are_not_attributed() -> None:
    """``not_applicable`` claims nothing rather than claiming an attribution."""
    row = {"name": "hepatocyte", "mapping_meta": {"resolution": {"status": "not_applicable"}}}
    counts = map_ids._record_mapping_lineage({"cell_types": [row]})
    assert counts == {"attributed": 0, "unchanged": 0, "skipped": 1,
                      "inbound_lineage_errors": 0}
    assert LINEAGE_KEY not in row


# ── NEW ACCEPTANCE — A2: failures are attributed as failures ─────────────────


@pytest.mark.parametrize("status", ["unavailable", "not_evaluated"])
def test_new_failed_or_unrun_lookup_is_unresolved_not_database_grounded(status: str) -> None:
    row = _protein_row(
        name="Fur",
        mapped_ids={},
        mapping_meta={
            "identity_status": "unresolved",
            "identity_verdict": {"verification_status": status},
            "rejected_mapped_ids": {"uniprot": "P0A9A9"},
            "unverified_identity_claim": {
                "identifiers": {"uniprot": "P0A9A9"},
                "verification_status": status,
            },
            "resolution": {"status": "novel", "issue": "implausible_name_match"},
        },
    )
    map_ids._record_mapping_lineage({"proteins": [row]})
    entry = _only(row)

    assert entry.origin == "unresolved"
    assert entry.review_required and entry.uncertainty
    assert "P0A9A9" in entry.uncertainty
    assert entry.sources == ()  # unresolved is not a SOURCED origin


def test_new_a_sourceless_database_grounded_is_rejected_by_the_c015_validator() -> None:
    """This stage does not police itself: the origin is SOURCED, so C-015 raises.

    Named here so the guarantee is visible at this seam -- if a future edit lets
    ``_mapping_lineage_facts`` return ``database_grounded`` with no source, the
    entry cannot be constructed and the mapper fails loudly rather than shipping
    an unbacked claim.
    """
    with pytest.raises(LineageError, match="requires a named source"):
        LineageEntry(
            stage=_STAGE, origin="database_grounded", support="direct",
            paper_explicit="not_evaluated", reason="unbacked", review_required=False,
        )


def test_new_a_matched_row_with_no_retrieved_record_is_unresolved() -> None:
    """The sentinel-only row cannot name a record, so it does not claim one."""
    row = _protein_row(
        name="mystery enzyme",
        mapped_ids={"uniprot": "Unknown", "drugbank": "-"},
        mapping_meta={"resolution": {"status": "matched", "order_step": "uniprot"}},
    )
    map_ids._record_mapping_lineage({"proteins": [row]})
    entry = _only(row)
    assert (entry.origin, entry.sources) == ("unresolved", ())


# ── NEW ACCEPTANCE — A3: rejection is the only strip ─────────────────────────


def test_new_a_rejected_verdict_is_excluded_and_names_the_contradicting_record() -> None:
    row = _protein_row(
        name="mcr genes",
        mapped_ids={},
        mapping_meta={
            "identity_status": "unresolved",
            "identity_verdict": {"verification_status": "rejected"},
            "rejected_mapped_ids": {"uniprot": "P08235"},
            "resolution": {"status": "novel", "issue": "implausible_name_match"},
        },
    )
    map_ids._record_mapping_lineage({"proteins": [row]})
    entry = _only(row)

    assert (entry.origin, entry.support) == ("excluded", "direct")
    assert [(s.source_type, s.source_id) for s in entry.sources] == [("uniprot", "P08235")]


@pytest.mark.parametrize("status", ["verified", "unavailable", "not_evaluated", ""])
def test_new_no_verification_status_other_than_rejected_produces_excluded(status: str) -> None:
    """§ 8: ``rejected`` is the ONLY case an identifier may be stripped in, so it
    is the only one that may read as ``excluded`` -- even when the row carries
    exactly the same stripped-identifier metadata."""
    meta = {
        "identity_verdict": {"verification_status": status},
        "rejected_mapped_ids": {"uniprot": "P08235"},
        "name_gate": {"verdict": "reject"},
        "resolution": {"status": "novel"},
    }
    facts = map_ids._mapping_lineage_facts(_protein_row(mapping_meta=meta))
    assert facts is not None and facts["origin"] != "excluded"


def test_new_a_row_that_never_ran_the_ladder_is_judged_on_its_name_gate() -> None:
    """A compound has no ``identity_verdict``; the gate is what refutes it."""
    row = {
        "name": "PhoP",
        "mapped_ids": {},
        "mapping_meta": {
            "name_gate": {"verdict": "reject"},
            "rejected_mapped_ids": {"kegg": "C00003"},
            "resolution": {"status": "novel", "issue": "implausible_name_match"},
        },
    }
    map_ids._record_mapping_lineage({"compounds": [row]})
    assert _only(row).origin == "excluded"

    keeps = dict(row["mapping_meta"], name_gate={"verdict": "keep"})
    assert map_ids._mapping_lineage_facts({"name": "PhoP", "mapping_meta": keeps})["origin"] != "excluded"


# ── NEW ACCEPTANCE — A4: the Fur shape stays fixed ──────────────────────────


def test_new_the_fur_shape_preserves_the_accession_as_unresolved_not_excluded() -> None:
    """``runs_verify/2026-08-04_1306`` PMC12452463: ``Fur`` shipped
    ``uniprot:P0A9A9`` with ``identifier_resolution: "ok"`` and
    ``candidate_evidence: "no_candidate_describes_the_shipped_identifier"``. The
    pool held ten iron-sulfur proteins and P0A9A9 was never among them, so
    nothing refuted it -- the ladder simply had nothing to judge on. C-033 made
    that preserve the accession; this asserts the lineage says so too, and does
    not paper the strip over by calling the outcome grounded either."""
    row = _protein_row(
        name="Fur",
        mapped_ids={},
        mapping_meta={
            "identity_status": "unresolved",
            "identity_verdict": {
                "verified": False,
                "reason": "identity_evidence_missing",
                "verification_status": "not_evaluated",
                "checks": {
                    "identifier_resolution": "ok",
                    "candidate_evidence": "no_candidate_describes_the_shipped_identifier",
                },
            },
            "unverified_identity_claim": {
                "identifiers": {"uniprot": "P0A9A9"},
                "verification_status": "not_evaluated",
                "failed_check": "identity_evidence_missing",
            },
            "rejected_mapped_ids": {"uniprot": "P0A9A9"},
            "resolution": {"status": "novel", "issue": "implausible_name_match"},
        },
    )
    map_ids._record_mapping_lineage({"proteins": [row]})
    entry = _only(row)

    assert entry.origin == "unresolved"
    assert entry.origin != "excluded"
    assert entry.review_required
    assert "P0A9A9" in entry.uncertainty
    assert "not evidence that it is false" in entry.uncertainty
    # the preserved claim is untouched: attribution describes, it does not strip
    assert row["mapping_meta"]["unverified_identity_claim"]["identifiers"] == {
        "uniprot": "P0A9A9"
    }


def test_new_a_placeholder_backed_protein_is_attributed_not_repaired() -> None:
    """TRAP-3 / § 13: ``placeholder_backed_proteins`` is a standing policy
    disagreement, not a defect. The row is described truthfully -- the record it
    names is a placeholder, so it is ``unresolved`` and asks for review -- and
    nothing here changes whether it ships."""
    row = _protein_row(
        name="chorismate synthase",
        mapped_ids={"uniprot": "Unknown", "pathbank_protein_id": 5061},
        mapping_meta={
            "identity_status": "placeholder",
            "resolution": {"status": "matched", "order_step": "pathbank_protein_id"},
        },
    )
    before = json.dumps(row, sort_keys=True)
    map_ids._record_mapping_lineage({"proteins": [row]})
    entry = _only(row)

    assert (entry.origin, entry.review_required) == ("unresolved", True)
    assert entry.sources == ()
    assert json.dumps(_strip_lineage(row), sort_keys=True) == before


# ── NEW ACCEPTANCE — A6: no double attribution, inbound lineage preserved ────


def test_new_inbound_lineage_is_kept_and_a_second_pass_adds_nothing() -> None:
    inbound = {
        "stage": "paper_extraction", "origin": "paper_stated", "support": "direct",
        "paper_explicit": "explicit", "reason": "stated in the results section",
        "review_required": False,
        "sources": [{"source_id": "PMC12452463", "source_type": "pmcid"}],
    }
    row = _protein_row(
        name="EntB",
        mapped_ids={"uniprot": "P0ADI4"},
        mapping_meta={"resolution": {"status": "matched", "order_step": "uniprot"}},
        provenance_lineage=[dict(inbound)],
    )
    entities = {"proteins": [row]}

    assert map_ids._record_mapping_lineage(entities)["attributed"] == 1
    first = list(row[LINEAGE_KEY])
    counts = map_ids._record_mapping_lineage(entities)

    assert counts["attributed"] == 0 and counts["unchanged"] == 1
    assert row[LINEAGE_KEY] == first
    stages = [e.stage for e in read_lineage(row).canonical()]
    assert stages.count("paper_extraction") == 1 and stages.count(_STAGE) == 1


def test_new_a_genuinely_different_second_attribution_is_not_suppressed() -> None:
    """The guard suppresses an exactly equivalent repeat and nothing else.

    ``map_payload`` runs more than once over the same payload with gap
    resolution in between, and the two passes may reach different verdicts. A
    second pass that concludes something else must still be recorded, or the
    lineage would claim this stage said one thing when it said two.
    """
    row = _protein_row(
        name="PhoP",
        mapped_ids={"uniprot": "P0ADI4"},
        mapping_meta={"resolution": {"status": "matched", "order_step": "uniprot"}},
    )
    entities = {"proteins": [row]}
    assert map_ids._record_mapping_lineage(entities)["attributed"] == 1

    # the second pass strips the identifier and reaches a different verdict
    row["mapped_ids"] = {}
    row["mapping_meta"] = {
        "identity_verdict": {"verification_status": "rejected"},
        "rejected_mapped_ids": {"uniprot": "P0ADI4"},
        "resolution": {"status": "novel", "issue": "implausible_name_match"},
    }
    assert map_ids._record_mapping_lineage(entities)["attributed"] == 1

    origins = sorted(e.origin for e in _mapping_entries(row))
    assert origins == ["database_grounded", "excluded"]


def test_new_the_idempotence_guard_uses_c015s_own_content_identity() -> None:
    """Equivalence is decided by :class:`LineageEntry`'s own equality, which
    covers exactly the fields ``_entry_key`` enumerates -- no private key, no
    nonce, no counter, no timestamp. Two entries built from the same content are
    equal even when their sources were supplied in a different order."""
    fields = {
        "stage": _STAGE, "origin": "database_grounded", "support": "direct",
        "paper_explicit": "not_evaluated", "reason": "r", "review_required": False,
    }
    a = LineageEntry(**fields, sources=[{"source_id": "P1"}, {"source_id": "P2"}])
    b = LineageEntry(**fields, sources=[{"source_id": "P2"}, {"source_id": "P1"}])
    assert a == b
    assert {f.name for f in dataclasses.fields(LineageEntry)} == {
        "stage", "origin", "support", "paper_explicit", "reason", "review_required",
        "uncertainty", "sources",
    }
    assert LineageEntry(**dict(fields, reason="different"), sources=a.sources) != a


def test_new_a_malformed_inbound_lineage_is_reported_not_raised() -> None:
    """Recording provenance may not change which identifiers this stage ships."""
    row = _protein_row(
        mapped_ids={"uniprot": "P0ADI4"},
        mapping_meta={"resolution": {"status": "matched"}},
        provenance_lineage=[{"stage": "not_a_stage"}],
    )
    counts = map_ids._record_mapping_lineage({"proteins": [row]})
    assert counts["inbound_lineage_errors"] == 1 and counts["attributed"] == 0
    assert row["mapped_ids"] == {"uniprot": "P0ADI4"}


# ── NEW ACCEPTANCE — A7: no aliasing writes ─────────────────────────────────


def test_new_lineage_lands_only_on_map_payloads_own_copy_never_the_caller_input(
    tmp_path: Path,
) -> None:
    payload = preservation_fixtures()["carried_ids"]
    before = json.dumps(payload, sort_keys=True)

    result = run_offline(payload, tmp_path / "cache.json")

    assert json.dumps(payload, sort_keys=True) == before
    assert LINEAGE_KEY not in json.dumps(payload)
    assert any(_mapping_entries(r) for r in _entity_rows(result["payload"]))


def test_new_no_lineage_reaches_the_mapping_cache_file(tmp_path: Path) -> None:
    """``map_ids`` reaches a disk cache; a lineage write landing in a cached
    record would poison every later run. The cache stores resolver results and
    never an entity row, and the attribution pass runs after ``cache.save()``."""
    cache_path = tmp_path / "id_mapping_cache.json"
    cache_path.write_text(
        json.dumps({"proteins": {}, "compounds": {}, "complexes": {}}), encoding="utf-8"
    )
    payload = preservation_fixtures()["carried_ids"]

    env = {"T2PW_SPECIES_LLM": "0", "T2PW_SPECIES_NCBI": "0"}
    with patch.dict(os.environ, env), patch.object(
        map_ids.PathBankDbResolver, "from_env", classmethod(lambda cls, overrides=None: None)
    ), patch.object(
        map_ids.HttpClient, "get", side_effect=_NoNetwork("network")
    ):
        result = map_payload(payload, cache_path=cache_path, id_source="db", use_cache=True)

    assert LINEAGE_KEY not in cache_path.read_text(encoding="utf-8")
    assert any(_mapping_entries(r) for r in _entity_rows(result["payload"]))


# ── NEW ACCEPTANCE — A8: hashing ────────────────────────────────────────────


def test_new_lineage_moves_the_payload_hash_and_not_the_graph_hash() -> None:
    """§ 6: lineage must not change graph equivalence, and must stay detectable."""
    row = _protein_row(
        name="EntB",
        mapped_ids={"uniprot": "P0ADI4"},
        mapping_meta={"resolution": {"status": "matched", "order_step": "uniprot"}},
    )
    payload = {"entities": {"proteins": [row]}, "processes": {"reactions": []}}
    graph_before = canonical_graph_sha256(payload)
    payload_before = canonical_payload_sha256(payload)

    map_ids._record_mapping_lineage(payload["entities"])

    assert row[LINEAGE_KEY]
    assert canonical_graph_sha256(payload) == graph_before
    assert canonical_payload_sha256(payload) != payload_before


# ── NEW ACCEPTANCE — A9: no network, no new lookups ─────────────────────────


def test_new_the_attribution_pass_opens_no_socket_and_no_connection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """This card adds zero database and zero network calls. The pass is given a
    row that would otherwise be worth looking up and must still touch nothing."""
    import socket

    def _boom(*args: Any, **kwargs: Any) -> Any:
        raise _NoNetwork("the attribution pass performed I/O")

    monkeypatch.setattr(socket.socket, "connect", _boom)
    monkeypatch.setattr(map_ids.HttpClient, "get", _boom)
    monkeypatch.setattr(map_ids.PathBankDbResolver, "_ensure_connection", _boom)
    monkeypatch.setattr(map_ids.MappingCache, "get", _boom)
    monkeypatch.setattr(map_ids.MappingCache, "set", _boom)

    rows = [
        _protein_row(name="Fur", mapped_ids={"uniprot": "P0A9A9"},
                     mapping_meta={"resolution": {"status": "matched"}}),
        {"name": "ATP", "mapped_ids": {"chebi": "CHEBI:15422"},
         "mapping_meta": {"resolution": {"status": "matched", "order_step": "synonym"}}},
    ]
    assert map_ids._record_mapping_lineage({"proteins": rows[:1], "compounds": rows[1:]})[
        "attributed"
    ] == 2


# ── G9 PRESERVATION — A5: no decision change ────────────────────────────────


def test_preservation_map_payload_output_without_lineage_matches_the_base_golden(
    tmp_path: Path,
) -> None:
    """THE LOAD-BEARING CLAUSE. The golden was produced by running this exact
    battery against ``bcc0bfe`` -- before any of this card's code existed -- and
    is committed unchanged. Same identifiers, same ``mapping_meta``, same
    ``identity_status``, same report, same gate outcomes: everything this stage
    observably emits other than ``provenance_lineage`` itself."""
    assert GOLDEN.exists(), f"missing base golden: {GOLDEN}"
    assert preservation_projection(tmp_path) == GOLDEN.read_text(encoding="utf-8")


def test_preservation_the_battery_actually_produces_lineage(tmp_path: Path) -> None:
    """Guards the golden against being vacuously green: if the battery stopped
    attributing anything, the comparison above would pass for the wrong reason."""
    result = run_offline(preservation_fixtures()["carried_ids"], tmp_path / "c.json")
    attributed = [r for r in _entity_rows(result["payload"]) if _mapping_entries(r)]
    assert len(attributed) >= 4
    assert {e.origin for r in attributed for e in _mapping_entries(r)} <= set(
        ["database_grounded", "unresolved", "excluded"]
    )
