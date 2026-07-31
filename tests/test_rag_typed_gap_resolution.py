"""Typed gaps are closed by the PAYLOAD, not by a candidate having been accepted.

The failure this file pins
--------------------------
An unmapped-enzyme gap asks "which protein IS this?". A missing-compartment gap
asks "where does this happen?". A missing-precursor gap asks "which participant
is this reaction missing?". None of those is answered by adding a reaction row —
and yet a reaction whose evidence span happened to contain "UniProt P12345" used
to be admitted *as the fill*, closing the gap in the report while the protein
still carried no ``mapped_ids`` at all and the Unknown-protein fallback was the
only thing actually standing behind it.

So the two are now separate things:

* a REACTION candidate offered to a typed gap is always refused
  (``candidate_type_cannot_fill_gap``);
* the typed evidence in its span is lifted out as a structured
  :class:`~t2pw.rag.admission.TypedResolution`, which
  ``synthesize_with_report`` then either APPLIES to the payload — in the schema's
  own representation — or does not;
* ``unresolved_gaps`` is derived from the resulting payload, so a gap is reported
  closed only when the payload really expresses the resolution.

Every assertion below is on the emitted payload and, where relevant, on the
strict post-extraction contract that payload has to satisfy.

Offline / deterministic: no chromadb, no network, no live LLM.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

if "openai" not in sys.modules:
    openai_stub = types.ModuleType("openai")

    class _OpenAI:
        def __init__(self, *_: object, **__: object) -> None:
            self.chat = types.SimpleNamespace(
                completions=types.SimpleNamespace(create=lambda **__: None)
            )

    openai_stub.OpenAI = _OpenAI
    openai_stub.RateLimitError = RuntimeError
    openai_stub.APIError = RuntimeError
    openai_stub.APITimeoutError = RuntimeError
    openai_stub.AuthenticationError = RuntimeError
    openai_stub.BadRequestError = RuntimeError
    sys.modules["openai"] = openai_stub

from t2pw.pipeline.entity_identity import (  # noqa: E402
    PATHBANK_UNKNOWN_FALLBACK_RULE,
    PATHBANK_UNKNOWN_PROTEIN_ID,
    PATHBANK_UNKNOWN_PROTEIN_NAME,
    PATHBANK_UNKNOWN_PROTEIN_UNIPROT,
    has_protein_external_identity,
    is_pathbank_unknown_protein,
)
from t2pw.pipeline.stage_contracts import validate_post_extraction  # noqa: E402
from t2pw.pwml.ir import (  # noqa: E402
    build_pwml_ir,
    validate_required_pwml_contract,
)
from t2pw.rag.admission import AdmissionPolicy  # noqa: E402
from t2pw.rag.identity import (  # noqa: E402
    build_identity_verifier,
    fail_closed_verifier,
)
from t2pw.rag.retrieve import EvidenceBundle, Gap, query_for_gap  # noqa: E402
from t2pw.rag.store import Chunk, Retrieved  # noqa: E402
from t2pw.rag.synthesize import (  # noqa: E402
    _open_metabolites,
    _payload_closes_gap,
    _payload_participants,
    synthesize_with_report,
)


REQUESTED_PATHWAY = "caffeine degradation"
REQUESTED_ORGANISM = "Pseudomonas putida"


def _seed() -> dict:
    """``A -> B`` catalysed by EnzX. B is the open end; EnzX is unmapped."""
    return {
        "entities": {
            "species": [{"name": "Pseudomonas putida"}],
            "compounds": [{"name": "A"}, {"name": "B"}],
            "proteins": [{"name": "EnzX"}],
        },
        "processes": {
            "reactions": [
                {
                    "name": "R0 seed step",
                    "inputs": ["A"],
                    "outputs": ["B"],
                    "enzymes": [{"protein": "EnzX"}],
                    "evidence": "EnzX converts A to B.",
                    "scope_membership": "core",
                }
            ]
        },
    }


def _seed_context() -> dict:
    return {
        "text": "caffeine degradation in Pseudomonas putida",
        "source": {
            "source_id": "PMID:SEED",
            "source_title": "seed paper",
            "source_type": "paper",
            "organism": "Pseudomonas putida",
        },
    }


def _typed_gap(kind: str, label: str, **kwargs) -> Gap:  # noqa: D401
    return Gap(
        kind=kind,
        label=label,
        symbols=[label],
        source="gate",
        requested_pathway=REQUESTED_PATHWAY,
        requested_organism=REQUESTED_ORGANISM,
        **kwargs,
    )


def _bundle(gap: Gap, text: str) -> EvidenceBundle:
    chunk = Chunk(
        id="c1",
        text=text,
        source_id="PMID:B",
        source_title="downstream paper",
        source_type="paper",
        source_uri="https://example.org/PMID:B",
        organism="Pseudomonas putida",
        section="results",
    )
    return EvidenceBundle(
        gap=gap, query=query_for_gap(gap), hits=[Retrieved(chunk=chunk, score=0.9)]
    )


def _run(gap: Gap, text: str, **kwargs):
    return synthesize_with_report(
        _seed(),
        [_bundle(gap, text)],
        _seed_context(),
        gaps=[gap],
        requested_pathway=REQUESTED_PATHWAY,
        requested_organism=REQUESTED_ORGANISM,
        **kwargs,
    )


def _unresolved_ids(result) -> set:
    return {row["gap_id"] for row in result.unresolved_gaps if row.get("gap_id")}


def _protein(payload: dict, name: str) -> dict:
    return next(
        r for r in payload["entities"]["proteins"] if r["name"] == name
    )


# ---------------------------------------------------------------------------
# (a) UniProt evidence: verified mapped_ids, or the gap stays open.
# ---------------------------------------------------------------------------
_IDENTITY_SPAN = "EnzX (UniProt P12345) converts A to B."


def _resolver_candidates(*, uniprot="P12345", name="EnzX", organism=REQUESTED_ORGANISM,
                         score=0.95, rival_score=0.10):
    """Local resolver evidence in the shape ``verify_real_protein_identity`` reads.

    Two rows on purpose: the ladder's last rung is a MARGIN over the best rival
    naming a different accession, so a single row would leave that rung untested.
    """
    rows = [
        {
            "uniprot": uniprot,
            "name": name,
            "protein_name": name,
            "gene": name,
            "organism": organism,
            "species": organism,
            "score": score,
            "source": "uniprot",
            "entity_type": "protein",
        },
        {
            "uniprot": "Q99999",
            "name": "EnzOther",
            "protein_name": "EnzOther",
            "organism": organism,
            "species": organism,
            "score": rival_score,
            "source": "uniprot",
            "entity_type": "protein",
        },
    ]

    def _provider(_name):
        return rows

    return _provider


def test_an_unverified_accession_leaves_the_gap_open_and_writes_no_mapped_ids() -> None:
    """Production default: fail-closed. The accession stays a claim.

    The verifier wired in production is ``identity.fail_closed_verifier`` because
    the RAG chain produces passages, not resolver candidates. The gap staying open
    is what keeps the existing Unknown-protein fallback in play.
    """
    gap = _typed_gap("unmapped_enzyme", "EnzX")
    attempts: list = []
    result = _run(
        gap, _IDENTITY_SPAN, identity_verifier=fail_closed_verifier(attempts)
    )

    proposals = result.admission["resolutions"]
    assert [p["kind"] for p in proposals] == ["identifier"]
    assert proposals[0]["value"] == {"uniprot": "P12345"}
    assert proposals[0]["applied"] is False

    # The ladder was ASKED and declined for a stated reason -- not skipped.
    assert [a["reason"] for a in attempts] == ["no_local_resolver_evidence"]
    assert attempts[0]["verified"] is False

    assert "mapped_ids" not in _protein(result.payload, "EnzX")
    assert gap.gap_id in _unresolved_ids(result)
    unresolved = next(
        row for row in result.unresolved_gaps if row.get("gap_id") == gap.gap_id
    )
    assert unresolved["recommended_route"] == "identity_resolver"
    assert validate_post_extraction(result.payload)["ok"] is True


def test_a_verified_accession_writes_mapped_ids_and_closes_the_gap() -> None:
    """The REAL ladder, over real candidate evidence, confirming an identity."""
    gap = _typed_gap("unmapped_enzyme", "EnzX")
    attempts: list = []
    verifier = build_identity_verifier(
        _resolver_candidates(), organism=REQUESTED_ORGANISM, record=attempts
    )
    result = _run(gap, _IDENTITY_SPAN, identity_verifier=verifier)

    assert attempts and attempts[0]["verified"] is True, attempts
    assert attempts[0]["score"] >= 0.5
    # The ladder's rungs, as the ladder itself reports them. ``margin`` is -1 with
    # ``no_competing_candidate``: no rival passed rungs 2-4 while naming a
    # different accession, which is the ladder's own way of saying the margin rung
    # had nothing to compare against.
    assert attempts[0]["checks"]["identifier_resolution"] == "ok"
    assert attempts[0]["checks"]["candidate_evidence"] == "ok"
    assert attempts[0]["checks"]["species"] == "ok"
    assert attempts[0]["checks"]["name"] == "keep"
    assert attempts[0]["checks"]["score"] == "ok"
    assert attempts[0]["checks"]["margin"] in {"ok", "no_competing_candidate"}
    assert result.admission["resolutions"][0]["applied"] is True
    assert _protein(result.payload, "EnzX")["mapped_ids"] == {"uniprot": "P12345"}
    assert gap.gap_id not in _unresolved_ids(result)
    assert validate_post_extraction(result.payload)["ok"] is True


def test_the_ladder_refuses_a_wellformed_accession_with_no_supporting_candidate() -> None:
    """Well-formed is not verified. The grammar says nothing about ownership."""
    gap = _typed_gap("unmapped_enzyme", "EnzX")
    attempts: list = []
    # Candidates exist, but none of them describes P12345.
    verifier = build_identity_verifier(
        _resolver_candidates(uniprot="Q11111"),
        organism=REQUESTED_ORGANISM,
        record=attempts,
    )
    result = _run(gap, _IDENTITY_SPAN, identity_verifier=verifier)

    assert attempts and attempts[0]["verified"] is False
    assert "mapped_ids" not in _protein(result.payload, "EnzX")
    assert gap.gap_id in _unresolved_ids(result)


def test_an_ec_number_alone_never_closes_an_unmapped_enzyme_gap() -> None:
    """Strict export identity is UniProt/DrugBank. An EC number is annotation.

    Closure reuses ``has_protein_external_identity`` — the predicate the Stage-3
    gate and the PathWhiz export apply — so a protein carrying only an EC number
    is still unmapped, and still on the Unknown-fallback path that would carry it
    honestly.
    """
    gap = _typed_gap("unmapped_enzyme", "EnzX")
    result = _run(
        gap,
        "EnzX (EC 1.2.3.4) converts A to B.",
        identity_verifier=lambda _n, _v: True,  # even a permissive caller
    )

    proposals = result.admission["resolutions"]
    assert proposals and proposals[0]["value"] == {"ec": "1.2.3.4"}
    assert proposals[0]["applied"] is True  # the annotation IS written...
    assert _protein(result.payload, "EnzX")["mapped_ids"] == {"ec": "1.2.3.4"}
    assert not has_protein_external_identity(_protein(result.payload, "EnzX"))
    # ...and the gap is STILL open, because EC is not export identity.
    assert gap.gap_id in _unresolved_ids(result)


def test_the_unknown_sentinel_does_not_count_as_a_resolved_identity() -> None:
    """The honest fallback stays honest: it exports, it does not claim a mapping."""
    gap = _typed_gap("unmapped_enzyme", "EnzX")
    result = _run(gap, "EnzX was studied.")

    payload = result.payload
    row = _protein(payload, "EnzX")
    # The real sentinel shape, from ``t2pw.pipeline.entity_identity``.
    row["name"] = PATHBANK_UNKNOWN_PROTEIN_NAME
    row["mapped_ids"] = {
        "uniprot": PATHBANK_UNKNOWN_PROTEIN_UNIPROT,
        "pathbank_protein_id": PATHBANK_UNKNOWN_PROTEIN_ID,
    }
    row["mapping_meta"] = {"chosen_rule": PATHBANK_UNKNOWN_FALLBACK_RULE}
    assert is_pathbank_unknown_protein(row)

    unknown_gap = _typed_gap("unmapped_enzyme", PATHBANK_UNKNOWN_PROTEIN_NAME)
    assert not _payload_closes_gap(
        payload, unknown_gap, _open_metabolites(payload), _payload_participants(payload)
    )


def test_an_accession_attached_to_another_protein_is_not_this_ones_identity() -> None:
    """Locality: "EnzX interacts with EnzY (UniProt P12345)" identifies EnzY."""
    gap = _typed_gap("unmapped_enzyme", "EnzX")
    result = _run(gap, "EnzX interacts with EnzY (UniProt P12345).")
    assert result.admission["resolutions"] == []
    assert gap.gap_id in _unresolved_ids(result)


def test_no_reaction_row_is_ever_added_for_a_typed_gap() -> None:
    """The separation: the gap is closed by an identity, not by a new reaction."""
    gap = _typed_gap("unmapped_enzyme", "EnzX")
    result = _run(
        gap,
        _IDENTITY_SPAN,
        identity_verifier=build_identity_verifier(
            _resolver_candidates(), organism=REQUESTED_ORGANISM
        ),
    )

    names = [r["name"] for r in result.payload["processes"]["reactions"]]
    assert names == ["R0 seed step"], "no reaction row was added for a typed gap"
    assert result.admission["counts"]["accepted"] == 0
    assert result.admission["resolutions"][0]["applied"] is True
    assert gap.gap_id not in _unresolved_ids(result)


def test_a_reaction_candidate_offered_to_a_typed_gap_is_refused_by_kind() -> None:
    """Even an arrow-parseable reaction in the same span cannot fill a typed gap."""
    gap = _typed_gap("unmapped_enzyme", "EnzX")
    result = _run(gap, "EnzX (UniProt P12345) | A -> C | enzyme: EnzX")

    assert result.admission["counts"]["accepted"] == 0
    assert result.admission["rejected"], "the reaction candidate must be recorded"
    assert result.admission["rejected"][0]["reasons"][-1].startswith(
        "candidate_type_cannot_fill_gap"
    )
    assert [r["name"] for r in result.payload["processes"]["reactions"]] == [
        "R0 seed step"
    ]


# ---------------------------------------------------------------------------
# (b) Compartment evidence: a real localization, real schema fields, or nothing.
# ---------------------------------------------------------------------------
def test_a_localization_assertion_builds_a_complete_biological_state() -> None:
    """Location, state and element_location — the whole resolvable structure."""
    gap = _typed_gap("missing_compartment", "B")
    result = _run(gap, "B occurs in the periplasm.")

    proposals = result.admission["resolutions"]
    assert [p["kind"] for p in proposals] == ["compartment"]
    assert proposals[0]["value"] == {"location": "periplasm"}
    assert proposals[0]["applied"] is True

    payload = result.payload
    # (a) the subcellular location entity
    assert any(
        row["name"] == "periplasm"
        for row in payload["entities"]["subcellular_locations"]
    )
    # (b) a biological state built by the repository's own helper, carrying the
    #     species and the canonical compartment
    states = payload["biological_states"]
    state = next(st for st in states if st.get("subcellular_location") == "periplasm")
    assert state["name"].startswith("AutoState_")
    assert state["species"] == REQUESTED_ORGANISM
    assert state["source_refs"] == ["PMID:B"]
    assert state["evidence"] == "B occurs in the periplasm."
    # (c) an element_locations row REFERENCING that exact state
    location_row = next(
        row
        for row in payload["element_locations"]["compound_locations"]
        if row["compound"] == "B"
    )
    assert location_row["biological_state"] == state["name"]
    assert location_row["source_refs"] == ["PMID:B"]

    assert gap.gap_id not in _unresolved_ids(result)
    assert validate_post_extraction(payload)["ok"] is True


def test_a_canonicalizable_compartment_carries_compartment_canonical() -> None:
    """``compartment_canonical`` is filled where the repository's vocab covers it.

    That vocabulary (``gap_resolver._CANONICAL_COMPARTMENT_VOCAB``) is eukaryote-
    leaning and has no term for ``periplasm``, so the field is legitimately absent
    there — the state still carries the ``subcellular_location`` the IR resolves
    on. ``nucleus`` IS in the vocab, which is what pins that this resolution goes
    through the shared helper rather than writing its own state shape.
    """
    gap = _typed_gap("missing_compartment", "B")
    result = _run(gap, "B is localized in the nucleus.")

    state = next(
        st
        for st in result.payload["biological_states"]
        if st.get("subcellular_location") == "nucleus"
    )
    assert state["compartment_canonical"] == "nucleus"
    assert gap.gap_id not in _unresolved_ids(result)


def test_the_biological_state_reference_resolves_in_the_pwml_ir() -> None:
    """The IR is the arbiter: a dangling state reference is not a closed gap."""
    gap = _typed_gap("missing_compartment", "B")
    result = _run(gap, "B occurs in the periplasm.")

    _ir, report = build_pwml_ir(result.payload, strict_db=False)
    assert report["unresolved"]["biological_state_references"] == []

    contract = validate_required_pwml_contract(result.payload, strict_db=False)
    location_errors = [
        issue
        for issue in contract["errors"]
        if "biological_state" in str(issue.get("code", ""))
        or "biological_state" in str(issue.get("pointer", ""))
    ]
    assert location_errors == [], location_errors


def test_a_dangling_biological_state_reference_is_not_closure() -> None:
    """A non-empty string is not a resolvable reference."""
    gap = _typed_gap("missing_compartment", "B")
    result = _run(gap, "B occurs in the periplasm.")
    payload = result.payload

    payload["element_locations"]["compound_locations"][0]["biological_state"] = (
        "NoSuchState"
    )
    assert not _payload_closes_gap(
        payload, gap, _open_metabolites(payload), _payload_participants(payload)
    )

    _ir, report = build_pwml_ir(payload, strict_db=False)
    assert report["unresolved"]["biological_state_references"], (
        "the IR must see the dangling reference this predicate now refuses"
    )


def test_a_compartment_word_merely_co_occurring_locates_nothing() -> None:
    """"B and nuclear extracts were analyzed" is not a localization of B."""
    gap = _typed_gap("missing_compartment", "B")
    result = _run(gap, "B and nuclear extracts were analyzed.")

    assert result.admission["resolutions"] == []
    assert "element_locations" not in result.payload
    assert gap.gap_id in _unresolved_ids(result)


def test_a_reaction_compartment_is_not_a_compound_localization() -> None:
    """"A is converted to B in the periplasm" locates the REACTION, not B."""
    gap = _typed_gap("missing_compartment", "B")
    result = _run(gap, "A is converted to B in the periplasm.")

    assert result.admission["resolutions"] == []
    assert gap.gap_id in _unresolved_ids(result)


def test_a_compartment_for_an_entity_not_in_the_payload_is_not_applied() -> None:
    """Nothing to attach it to means nothing was resolved."""
    gap = _typed_gap("missing_compartment", "Zed")
    result = _run(gap, "Zed is localized in the periplasm.")

    proposals = result.admission["resolutions"]
    assert proposals and proposals[0]["applied"] is False
    assert "not an entity of this payload" in proposals[0]["reasons"][0]
    assert "element_locations" not in result.payload
    assert gap.gap_id in _unresolved_ids(result)


def test_a_location_the_schema_cannot_express_leaves_the_gap_open() -> None:
    gap = _typed_gap("missing_compartment", "B")
    result = _run(gap, "B is localized in the nucleoid-associated fraction.")

    assert result.admission["resolutions"] == []
    assert "element_locations" not in result.payload
    assert gap.gap_id in _unresolved_ids(result)


# ---------------------------------------------------------------------------
# (c) Precursor repair: only a genuinely incomplete reaction, only its empty side.
# ---------------------------------------------------------------------------
def _incomplete_seed() -> dict:
    """``R0`` has NO inputs. That is what "missing precursor" actually means."""
    return {
        "entities": {
            "species": [{"name": "Pseudomonas putida"}],
            "compounds": [{"name": "B"}],
            "proteins": [{"name": "EnzX"}],
        },
        "processes": {
            "reactions": [
                {
                    "name": "R0 seed step",
                    "inputs": [],
                    "outputs": ["B"],
                    "enzymes": [{"protein": "EnzX"}],
                    "evidence": "B is produced by EnzX.",
                    "scope_membership": "core",
                }
            ]
        },
    }


def _run_incomplete(gap, text, **kwargs):
    return synthesize_with_report(
        _incomplete_seed(),
        [_bundle(gap, text)],
        _seed_context(),
        gaps=[gap],
        requested_pathway=REQUESTED_PATHWAY,
        requested_organism=REQUESTED_ORGANISM,
        **kwargs,
    )


def test_a_precursor_resolution_patches_the_genuinely_incomplete_reaction() -> None:
    """``R0`` really has no inputs; ``Q -> B`` supplies one."""
    gap = _typed_gap(
        "missing_precursor", "R0 seed step", target_symbols=["B"], missing_side="inputs"
    )
    result = _run_incomplete(gap, "Q is converted to B by EnzX.")

    proposals = result.admission["resolutions"]
    assert [p["kind"] for p in proposals] == ["precursor"]
    assert proposals[0]["value"] == {
        # ``participants`` is the complete evidence-stated set the applier adds
        # atomically; ``participant`` is kept for readers of a single-participant
        # repair.
        "participants": ["Q"],
        "participant": "Q",
        "side": "inputs",
        "reaction": "R0 seed step",
    }
    assert proposals[0]["applied"] is True

    reactions = result.payload["processes"]["reactions"]
    assert [r["name"] for r in reactions] == ["R0 seed step"], "no new reaction added"
    repaired = reactions[0]
    assert [
        p if isinstance(p, str) else p.get("name") for p in repaired["inputs"]
    ] == ["Q"]
    assert repaired["source_refs"]
    assert repaired["rag_provenance"]["source_id"]

    # The gap predicate is re-run against the RESULTING payload.
    assert gap.gap_id not in _unresolved_ids(result)
    assert validate_post_extraction(result.payload)["ok"] is True

    # ...and the payload is still referentially intact.
    registered = {
        row["name"].casefold()
        for bucket in ("compounds", "proteins", "protein_complexes")
        for row in result.payload["entities"].get(bucket, [])
    }
    for reaction in reactions:
        names = [
            p if isinstance(p, str) else p.get("name")
            for p in reaction.get("inputs", []) + reaction.get("outputs", [])
        ]
        names += [a.get("entity") for a in reaction.get("enzymes", [])]
        for name in names:
            assert name and name.casefold() in registered, name


def test_a_complete_reaction_is_never_patched() -> None:
    """``R0`` with both sides populated is not missing a precursor."""
    gap = _typed_gap(
        "missing_precursor", "R0 seed step", target_symbols=["A", "B"]
    )
    result = _run(gap, "Q is converted to B by EnzX.")

    assert result.admission["resolutions"] == []
    inputs = [
        p if isinstance(p, str) else p.get("name")
        for p in result.payload["processes"]["reactions"][0]["inputs"]
    ]
    assert inputs == ["A"], "a populated side must not gain a participant"
    assert gap.gap_id in _unresolved_ids(result)


def test_evidence_about_unrelated_chemistry_does_not_patch_the_reaction() -> None:
    """"X is converted to Q by EnzY" shares nothing with R0's known side."""
    gap = _typed_gap(
        "missing_precursor", "R0 seed step", target_symbols=["B"], missing_side="inputs"
    )
    result = _run_incomplete(gap, "X is converted to Q by EnzY.")

    assert result.admission["resolutions"] == []
    assert result.payload["processes"]["reactions"][0]["inputs"] == []
    assert gap.gap_id in _unresolved_ids(result)


def test_a_precursor_resolution_for_an_absent_reaction_is_not_proposed() -> None:
    gap = _typed_gap(
        "missing_precursor", "R9 not here", target_symbols=["B"], missing_side="inputs"
    )
    result = _run_incomplete(gap, "Q is converted to B by EnzX.")

    assert result.admission["resolutions"] == []
    assert gap.gap_id in _unresolved_ids(result)


def test_evidence_in_the_wrong_direction_does_not_patch_the_reaction() -> None:
    """``B -> Q`` puts B on the substrate side; R0 produces B."""
    gap = _typed_gap(
        "missing_precursor", "R0 seed step", target_symbols=["B"], missing_side="inputs"
    )
    result = _run_incomplete(gap, "B is converted to Q by EnzX.")

    assert result.admission["resolutions"] == []
    assert result.payload["processes"]["reactions"][0]["inputs"] == []


# ---------------------------------------------------------------------------
# 1. Scope admission applies to typed proposals too.
# ---------------------------------------------------------------------------
def _foreign_chunk_run(gap, text):
    """A Homo sapiens / cholesterol passage attacking an E. coli / caffeine element."""
    chunk = Chunk(
        id="c-foreign",
        text=text,
        source_id="PMC_FOREIGN",
        source_title="Cholesterol biosynthesis in human cells",
        source_type="paper",
        source_uri="https://example.org/PMC_FOREIGN",
        organism="Homo sapiens",
        section="results",
        observed_organisms=["Homo sapiens"],
        observed_pathways=["cholesterol biosynthesis"],
    )
    bundle = EvidenceBundle(
        gap=gap, query=query_for_gap(gap), hits=[Retrieved(chunk=chunk, score=0.9)]
    )
    return synthesize_with_report(
        _seed(),
        [bundle],
        _seed_context(),
        gaps=[gap],
        requested_pathway=REQUESTED_PATHWAY,
        requested_organism=REQUESTED_ORGANISM,
        identity_verifier=lambda _n, _v: True,
    )


def test_a_foreign_species_and_pathway_proposal_is_refused_before_it_mutates() -> None:
    """Nothing from a Homo sapiens cholesterol chunk may touch this payload."""
    for kind, target, text in (
        (
            "unmapped_enzyme",
            "EnzX",
            "In Homo sapiens, cholesterol biosynthesis requires EnzX "
            "(UniProt P12345).",
        ),
        (
            "missing_compartment",
            "B",
            "In Homo sapiens cholesterol biosynthesis, B is localized in the "
            "nucleus.",
        ),
    ):
        gap = _typed_gap(kind, target)
        result = _foreign_chunk_run(gap, text)

        proposals = result.admission["resolutions"]
        assert proposals, f"{kind}: the proposal must be recorded, not silently lost"
        row = proposals[0]
        assert row["applied"] is False, kind
        assert row["status"] == "rejected", kind
        assert row["organism_match"] == "mismatch", kind
        assert row["requested_pathway_match"] == "mismatch", kind
        assert row["span_observed_organisms"] == ["Homo sapiens"], kind
        assert row["paper_observed_pathways"] == ["cholesterol biosynthesis"], kind
        assert row["requested_organism"] == REQUESTED_ORGANISM, kind

        payload = result.payload
        assert "mapped_ids" not in _protein(payload, "EnzX"), kind
        assert "element_locations" not in payload, kind
        assert gap.gap_id in _unresolved_ids(result), kind


def test_a_typed_proposal_carries_the_full_scope_picture() -> None:
    """Requested, span-observed and paper-observed are three separate fields."""
    gap = _typed_gap("unmapped_enzyme", "EnzX")
    result = _run(gap, _IDENTITY_SPAN)

    row = result.admission["resolutions"][0]
    assert row["requested_pathway"] == REQUESTED_PATHWAY
    assert row["requested_organism"] == REQUESTED_ORGANISM
    # The span names no organism, so the paper-level reading is the fallback --
    # and the requested value is never promoted into either observed field.
    assert row["span_observed_organisms"] == []
    assert REQUESTED_ORGANISM not in row["span_observed_pathways"]
    assert row["organism_match"] in {"match", "unknown"}


def test_a_mixed_local_pathway_stays_mixed_and_obeys_the_policy() -> None:
    gap = _typed_gap("unmapped_enzyme", "EnzX")
    text = (
        "Linking caffeine degradation to cholesterol biosynthesis, EnzX "
        "(UniProt P12345) was characterised."
    )
    lenient = _run(gap, text, identity_verifier=lambda _n, _v: True)
    assert lenient.admission["resolutions"][0]["requested_pathway_match"] == "mixed"
    assert lenient.admission["resolutions"][0]["applied"] is True

    strict = _run(
        gap,
        text,
        identity_verifier=lambda _n, _v: True,
        admission_policy=AdmissionPolicy(require_pathway_match=True),
    )
    assert strict.admission["resolutions"][0]["status"] == "rejected"
    assert strict.admission["resolutions"][0]["applied"] is False
    assert "mapped_ids" not in _protein(strict.payload, "EnzX")
