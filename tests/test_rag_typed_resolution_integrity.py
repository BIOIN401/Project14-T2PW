"""Typed resolutions: right shape of row, exact repairs, order-independent.

Three defects, all of the same family — a typed resolution that *looked* applied
because nothing checked what it was applied TO.

1. **Entity type.** ``expected_type`` is a per-KIND constant: every
   ``unmapped_enzyme`` gap "expects a protein". The row being repaired may be a
   protein COMPLEX, which takes a PathBank/PathWhiz complex id and never a UniProt
   accession. Writing ``mapped_ids.uniprot`` onto one made ``identity_status``
   report ``verified`` for a complex whose identity was still unknown, and quietly
   removed it from the Unknown-protein-component fallback that was carrying it
   honestly. Synthesis compounded it by rebuilding the complex as a bare
   ``proteins`` row — components and all their provenance gone.

2. **Precursor exactness.** "Some novel participant is somewhere on this
   reaction" was accepted as closure. A participant appended to the side that was
   already populated leaves the empty side exactly as empty as it was. And
   evidence naming two substrates had its first one taken and its second dropped,
   writing a reaction no paper states.

3. **Order dependence.** Proposals were collapsed on ``(gap_id, kind, value)``,
   so whichever passage was retrieved first decided the outcome: a foreign
   paper's rejected proposal suppressed a local paper's valid one, and an
   unknown-scope proposal hid a later explicit mismatch.

Everything is asserted on the emitted payload and, where a structure is written,
on the PWML IR and the strict contracts that payload has to satisfy.

Offline / deterministic: no chromadb, no network, no live LLM.
"""

from __future__ import annotations

import copy
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

# C-051 / D-015 (LOCKED): the exporter no longer resolves compound identity
# after the canonical freeze, so a raw extraction payload is taken through the
# pre-freeze stage that now does that work. helpers_prefreeze asserts the stage
# actually ruled on every compound row.
from helpers_prefreeze import prefrozen_when_compounded  # noqa: E402
from t2pw.pipeline.entity_identity import (  # noqa: E402
    PATHBANK_UNKNOWN_PROTEIN_ID,
    PATHBANK_UNKNOWN_PROTEIN_NAME,
    PATHBANK_UNKNOWN_PROTEIN_UNIPROT,
    has_protein_external_identity,
    identity_status,
)
from t2pw.pipeline.stage_contracts import validate_post_extraction  # noqa: E402
from t2pw.pwml.ir import build_pwml_ir, validate_required_pwml_contract  # noqa: E402
from t2pw.rag.admission import (  # noqa: E402
    REASON_CONFLICTING_RESOLUTION,
    REASON_MULTI_PARTICIPANT_REPAIR,
    REASON_SIDE_NO_LONGER_MISSING,
    REASON_UNSUPPORTED_TARGET_TYPE,
    RESOLUTION_IDENTIFIER,
    TypedResolution,
)
from t2pw.rag.identity import (  # noqa: E402
    build_identity_verifier,
    fail_closed_verifier,
)
from t2pw.rag.retrieve import (  # noqa: E402
    EvidenceBundle,
    Gap,
    detect_gaps,
    entity_bucket_of,
    query_for_gap,
)
from t2pw.rag.store import Chunk, Retrieved  # noqa: E402
from t2pw.rag.synthesize import (  # noqa: E402
    _apply_typed_resolutions,
    _payload_closes_gap,
    _open_metabolites,
    _payload_participants,
    synthesize_with_report,
)

REQUESTED_PATHWAY = "caffeine degradation"
REQUESTED_ORGANISM = "Pseudomonas putida"


def _context() -> dict:
    return {
        "text": "caffeine degradation in Pseudomonas putida",
        "source": {
            "source_id": "PMID:SEED",
            "source_title": "seed paper",
            "source_type": "paper",
            "organism": REQUESTED_ORGANISM,
        },
    }


def _chunk(text: str, *, cid="c1", source="PMID:B", organism=REQUESTED_ORGANISM,
           observed_organisms=None, observed_pathways=None, title="downstream paper"):
    return Chunk(
        id=cid,
        text=text,
        source_id=source,
        source_title=title,
        source_type="paper",
        source_uri=f"https://example.org/{source}",
        organism=organism,
        section="results",
        observed_organisms=list(observed_organisms or []),
        observed_pathways=list(observed_pathways or []),
    )


def _bundle(gap, chunk) -> EvidenceBundle:
    return EvidenceBundle(
        gap=gap, query=query_for_gap(gap), hits=[Retrieved(chunk=chunk, score=0.9)]
    )


def _unresolved_ids(result) -> set:
    return {row["gap_id"] for row in result.unresolved_gaps if row.get("gap_id")}


# ===========================================================================
# 2. Entity type is preserved, and a complex never takes a protein identity.
# ===========================================================================
def _complex_seed() -> dict:
    """``ComplexX`` catalyses ``A -> B`` and is backed by the Unknown fallback."""
    return {
        "entities": {
            "species": [{"name": REQUESTED_ORGANISM}],
            "compounds": [{"name": "A"}, {"name": "B"}],
            "protein_complexes": [
                {
                    "name": "ComplexX",
                    "components": [
                        {
                            "name": PATHBANK_UNKNOWN_PROTEIN_NAME,
                            "pathbank_protein_id": PATHBANK_UNKNOWN_PROTEIN_ID,
                            "uniprot": PATHBANK_UNKNOWN_PROTEIN_UNIPROT,
                        }
                    ],
                    "mapping_meta": {
                        "chosen_rule": "pathbank_unknown_protein_fallback"
                    },
                }
            ],
        },
        "processes": {
            "reactions": [
                {
                    "name": "R0 seed step",
                    "inputs": ["A"],
                    "outputs": ["B"],
                    "enzymes": [{"protein": "ComplexX"}],
                    "evidence": "ComplexX converts A to B.",
                    "scope_membership": "core",
                }
            ]
        },
    }


COMPLEX_IDENTITY_SPAN = "ComplexX (UniProt P12345) converts A to B."


def _complex_gap(seed: dict) -> Gap:
    """The gap as ``detect_gaps`` really produces it, type and all."""
    gaps = detect_gaps(
        seed,
        {"gate": {"errors": [{"reason": "Protein 'ComplexX' is missing a UniProt identifier"}]}},
        requested_pathway=REQUESTED_PATHWAY,
        requested_organism=REQUESTED_ORGANISM,
    )
    return next(g for g in gaps if g.kind == "unmapped_enzyme")


def test_the_gap_contract_carries_the_targets_real_bucket_not_the_kind_default() -> None:
    """``expected_type`` says "protein"; the target is a protein COMPLEX."""
    seed = _complex_seed()
    gap = _complex_gap(seed)

    assert gap.expected_type == "protein"  # the per-kind constant, unchanged
    assert gap.target_entity_bucket == "protein_complexes"
    assert gap.target_entity_type == "protein_complex"
    assert gap.to_dict()["target_entity_type"] == "protein_complex"
    assert "Target entity type: protein_complex" in gap.describe()

    # And the same lookup used directly.
    assert entity_bucket_of(seed, "ComplexX") == ("protein_complexes", "protein_complex")
    assert entity_bucket_of(seed, "A") == ("compounds", "compound")
    assert entity_bucket_of(seed, "nothing here") == ("", "")


def test_a_complex_cannot_acquire_a_uniprot_identity_end_to_end() -> None:
    """The full regression: ComplexX + "UniProt P12345" resolves nothing.

    Driven with a verifier that confirms EVERYTHING, so the refusal cannot be
    mistaken for the production fail-closed default. What stops it is the type of
    the row, not the strength of the evidence.
    """
    seed = _complex_seed()
    gap = _complex_gap(seed)
    result = synthesize_with_report(
        seed,
        [_bundle(gap, _chunk(COMPLEX_IDENTITY_SPAN))],
        _context(),
        gaps=[gap],
        requested_pathway=REQUESTED_PATHWAY,
        requested_organism=REQUESTED_ORGANISM,
        identity_verifier=lambda _name, _value: True,
    )

    # (a) the proposal is REPORTED, and reported as refused for the stated reason.
    proposals = result.admission["resolutions"]
    assert [p["kind"] for p in proposals] == [RESOLUTION_IDENTIFIER]
    row = proposals[0]
    assert row["applied"] is False
    assert row["status"] == "rejected"
    assert row["target_entity_type"] == "protein_complex"
    assert any(r.startswith(REASON_UNSUPPORTED_TARGET_TYPE) for r in row["reasons"]), row

    entities = result.payload["entities"]
    # (b) no protein row was invented, and the complex kept no uniprot.
    complexes = {r["name"]: r for r in entities.get("protein_complexes", [])}
    assert "ComplexX" in complexes, entities
    assert "ComplexX" not in {r["name"] for r in entities.get("proteins", [])}
    assert "uniprot" not in (complexes["ComplexX"].get("mapped_ids") or {})
    assert not has_protein_external_identity(complexes["ComplexX"])

    # (c) the complex name is still the graph actor.
    enzymes = result.payload["processes"]["reactions"][0]["enzymes"]
    assert [e.get("entity") for e in enzymes] == ["ComplexX"]

    # (d) its internal Unknown-protein fallback survived synthesis.
    components = complexes["ComplexX"]["components"]
    assert [c["name"] for c in components] == [PATHBANK_UNKNOWN_PROTEIN_NAME]
    assert components[0]["pathbank_protein_id"] == PATHBANK_UNKNOWN_PROTEIN_ID
    assert (
        complexes["ComplexX"]["mapping_meta"]["chosen_rule"]
        == "pathbank_unknown_protein_fallback"
    )

    # (e) the gap is NOT reported as a verified identity.
    assert gap.gap_id in _unresolved_ids(result)
    assert identity_status(complexes["ComplexX"]) != "verified"
    assert validate_post_extraction(result.payload)["ok"] is True


def test_the_write_itself_refuses_a_complex_even_with_the_gap_contract_blank() -> None:
    """Belt and braces: the applier checks the BUCKET, not just the contract.

    A proposal built without a gap contract (``target_entity_type=""``) passes the
    scope screen. The write must still refuse it, because the rule being enforced
    is "a UniProt identity may only be written to ``entities.proteins``" — a
    property of the payload, not of the proposal.
    """
    payload = {
        "entities": {
            "species": [{"name": REQUESTED_ORGANISM}],
            "protein_complexes": [{"name": "ComplexX", "components": []}],
        },
        "processes": {"reactions": []},
    }
    proposal = TypedResolution(
        gap_id="gap-unmapped_enzyme-deadbeef",
        gap_kind="unmapped_enzyme",
        target="ComplexX",
        kind=RESOLUTION_IDENTIFIER,
        value={"uniprot": "P12345"},
        target_entity_type="",  # deliberately unset
        status="proposed",
    )
    records = _apply_typed_resolutions(
        payload, [proposal], identity_verifier=lambda _n, _v: True
    )

    assert records[0]["applied"] is False
    assert any(
        r.startswith(REASON_UNSUPPORTED_TARGET_TYPE) for r in records[0]["reasons"]
    ), records
    assert "mapped_ids" not in payload["entities"]["protein_complexes"][0]


def test_a_complex_compartment_is_refused_rather_than_written_to_protein_locations() -> None:
    """``element_locations`` has no bucket a complex resolves in.

    ``protein_locations.protein`` is resolved by the PWML IR against
    ``entities.proteins`` (``_entity_type_from_location_bucket``), so a complex
    placed there is a reference nothing can follow. The honest outcome is an
    explicit refusal and an open gap.
    """
    payload = {
        "entities": {
            "species": [{"name": REQUESTED_ORGANISM}],
            "protein_complexes": [{"name": "ComplexX", "components": []}],
        },
        "processes": {"reactions": []},
    }
    proposal = TypedResolution(
        gap_id="gap-missing_compartment-deadbeef",
        gap_kind="missing_compartment",
        target="ComplexX",
        kind="compartment",
        value={"location": "periplasm"},
        target_entity_type="",
        status="proposed",
    )
    records = _apply_typed_resolutions(payload, [proposal])

    assert records[0]["applied"] is False
    assert any(
        r.startswith(REASON_UNSUPPORTED_TARGET_TYPE) for r in records[0]["reasons"]
    ), records
    assert "element_locations" not in payload


def test_a_complex_identity_gap_is_only_closed_by_a_complex_level_identifier() -> None:
    """``has_protein_external_identity`` is not the predicate for a complex.

    Applied to a complex row it reads ``mapped_ids.uniprot`` — a field nothing may
    write there — so a complex "closed" by it would be closed by a value that can
    never legitimately exist. The complex's own identity is a PathBank/PathWhiz
    id, which is what closure requires.
    """
    gap = Gap(
        kind="unmapped_enzyme",
        label="ComplexX",
        symbols=["ComplexX"],
        source="gate",
        target_entity_bucket="protein_complexes",
        target_entity_type="protein_complex",
    )

    def _payload_with(row: dict) -> dict:
        return {"entities": {"protein_complexes": [row]}, "processes": {"reactions": []}}

    bare = _payload_with({"name": "ComplexX", "components": []})
    assert _payload_closes_gap(bare, gap, set(), set()) is False

    # A smuggled UniProt does NOT close it, even though the protein predicate
    # would say yes about the very same dict.
    smuggled = {"name": "ComplexX", "mapped_ids": {"uniprot": "P12345"}}
    assert has_protein_external_identity(smuggled) is True
    assert _payload_closes_gap(_payload_with(smuggled), gap, set(), set()) is False

    # A real complex-level identity does.
    real = _payload_with({"name": "ComplexX", "pathbank_complex_id": 12345})
    assert _payload_closes_gap(real, gap, set(), set()) is True


# ===========================================================================
# 3. Precursor application and closure are exact.
# ===========================================================================
def _incomplete_seed() -> dict:
    """``R0`` has NO inputs. That is what "missing precursor" actually means."""
    return {
        "entities": {
            "species": [{"name": REQUESTED_ORGANISM}],
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


def _precursor_gap(**kwargs) -> Gap:
    return Gap(
        kind="missing_precursor",
        label="R0 seed step",
        symbols=["R0 seed step"],
        source="qa_graph",
        target_symbols=["B"],
        requested_pathway=REQUESTED_PATHWAY,
        requested_organism=REQUESTED_ORGANISM,
        **kwargs,
    )


def test_the_repair_re_checks_the_missing_side_against_the_delivered_payload() -> None:
    """A reaction that is no longer missing that side is never appended to.

    The proposal is built against the seed and applied against the payload being
    DELIVERED. Between the two, synthesis rebuilds reactions and other resolutions
    run, so the row's shape at proposal time is not evidence about its shape now.
    """
    payload = {
        "entities": {"species": [{"name": REQUESTED_ORGANISM}], "compounds": [{"name": "B"}]},
        "processes": {
            "reactions": [
                {"name": "R0 seed step", "inputs": ["P"], "outputs": ["B"]}
            ]
        },
    }
    proposal = TypedResolution(
        gap_id="gap-missing_precursor-deadbeef",
        gap_kind="missing_precursor",
        target="R0 seed step",
        kind="precursor",
        value={
            "participants": ["Q"],
            "participant": "Q",
            "side": "inputs",
            "reaction": "R0 seed step",
        },
        status="proposed",
    )
    records = _apply_typed_resolutions(payload, [proposal])

    assert records[0]["applied"] is False
    assert any(
        r.startswith(REASON_SIDE_NO_LONGER_MISSING) for r in records[0]["reasons"]
    ), records
    assert payload["processes"]["reactions"][0]["inputs"] == ["P"]


def test_a_reaction_with_both_sides_empty_is_not_a_precursor_repair_either() -> None:
    """Both empty is not "missing inputs" — it is a reaction with no chemistry."""
    payload = {
        "entities": {"species": [{"name": REQUESTED_ORGANISM}]},
        "processes": {"reactions": [{"name": "R0 seed step", "inputs": [], "outputs": []}]},
    }
    proposal = TypedResolution(
        gap_id="g",
        gap_kind="missing_precursor",
        target="R0 seed step",
        kind="precursor",
        value={"participants": ["Q"], "side": "inputs", "reaction": "R0 seed step"},
        status="proposed",
    )
    records = _apply_typed_resolutions(payload, [proposal])
    assert records[0]["applied"] is False
    assert any(
        r.startswith(REASON_SIDE_NO_LONGER_MISSING) for r in records[0]["reasons"]
    )
    assert payload["processes"]["reactions"][0]["inputs"] == []


def test_a_participant_added_to_the_wrong_side_leaves_the_gap_open() -> None:
    """THE closure regression: novel-participant-anywhere is not a repair.

    ``R0`` is missing its INPUTS. Appending ``Q`` to the outputs adds a
    participant the gap did not know about — the old predicate's entire test — and
    leaves the inputs exactly as empty as they were.
    """
    gap = _precursor_gap(missing_side="inputs")
    wrong = {
        "entities": {"species": [{"name": REQUESTED_ORGANISM}]},
        "processes": {
            "reactions": [{"name": "R0 seed step", "inputs": [], "outputs": ["B", "Q"]}]
        },
    }
    assert (
        _payload_closes_gap(wrong, gap, _open_metabolites(wrong), _payload_participants(wrong))
        is False
    )

    right = copy.deepcopy(wrong)
    right["processes"]["reactions"][0] = {
        "name": "R0 seed step",
        "inputs": ["Q"],
        "outputs": ["B"],
    }
    assert (
        _payload_closes_gap(right, gap, _open_metabolites(right), _payload_participants(right))
        is True
    )


def test_closure_requires_the_original_anchors_to_stay_on_their_own_side() -> None:
    """A "repair" that moved ``B`` to the inputs rewrote the reaction."""
    gap = _precursor_gap(missing_side="inputs")
    moved = {
        "entities": {"species": [{"name": REQUESTED_ORGANISM}]},
        "processes": {
            "reactions": [{"name": "R0 seed step", "inputs": ["B", "Q"], "outputs": []}]
        },
    }
    assert (
        _payload_closes_gap(moved, gap, _open_metabolites(moved), _payload_participants(moved))
        is False
    )


def test_closure_needs_a_participant_the_gap_did_not_already_know() -> None:
    """Re-stating ``B`` on the empty side identifies nothing."""
    gap = _precursor_gap(missing_side="inputs")
    restated = {
        "entities": {"species": [{"name": REQUESTED_ORGANISM}]},
        "processes": {
            "reactions": [{"name": "R0 seed step", "inputs": ["B"], "outputs": ["B"]}]
        },
    }
    assert (
        _payload_closes_gap(
            restated, gap, _open_metabolites(restated), _payload_participants(restated)
        )
        is False
    )


def test_a_two_substrate_repair_is_applied_whole_and_closes_the_gap() -> None:
    """"glycine and succinyl-CoA" is two substrates, and both go in.

    The shape that used to lose one: the evidence names two participants for the
    missing side and the applier took ``fresh[0]``. What lands now is the complete
    evidence-stated set, the payload stays referentially intact, and the PWML
    contract still holds.
    """
    gap = _precursor_gap(missing_side="inputs")
    result = synthesize_with_report(
        _incomplete_seed(),
        [_bundle(gap, _chunk("P and Q are converted to B by EnzX."))],
        _context(),
        gaps=[gap],
        requested_pathway=REQUESTED_PATHWAY,
        requested_organism=REQUESTED_ORGANISM,
    )

    proposals = result.admission["resolutions"]
    assert [p["kind"] for p in proposals] == ["precursor"]
    assert proposals[0]["value"]["participants"] == ["P", "Q"]
    assert proposals[0]["applied"] is True

    reaction = result.payload["processes"]["reactions"][0]
    assert sorted(reaction["inputs"]) == ["P", "Q"]
    assert reaction["outputs"] == ["B"]

    # Referential integrity: both new participants are registered entities.
    compounds = {r["name"] for r in result.payload["entities"]["compounds"]}
    assert {"P", "Q"} <= compounds

    assert gap.gap_id not in _unresolved_ids(result)
    assert validate_post_extraction(result.payload)["ok"] is True
    _ir, report = build_pwml_ir(prefrozen_when_compounded(result.payload), strict_db=False)
    assert report["unresolved"]["biological_state_references"] == []
    validate_required_pwml_contract(result.payload)


def test_a_participant_set_that_cannot_be_added_in_full_is_refused_not_narrowed() -> None:
    """One of the two already sits on the other side; neither goes in.

    ``R0`` is ``? -> B``. "B and Q are converted to B" states a substrate set the
    repair cannot add whole — putting ``B`` on both sides is a self-loop nobody
    claimed — so the proposal is reported as
    ``unsupported_multi_participant_repair`` rather than silently reduced to
    ``Q``.
    """
    gap = _precursor_gap(missing_side="inputs")
    result = synthesize_with_report(
        _incomplete_seed(),
        [_bundle(gap, _chunk("B and Q are converted to B by EnzX."))],
        _context(),
        gaps=[gap],
        requested_pathway=REQUESTED_PATHWAY,
        requested_organism=REQUESTED_ORGANISM,
    )

    proposals = result.admission["resolutions"]
    assert [p["kind"] for p in proposals] == ["precursor"]
    row = proposals[0]
    assert row["applied"] is False
    assert row["status"] == "rejected"
    assert any(r.startswith(REASON_MULTI_PARTICIPANT_REPAIR) for r in row["reasons"]), row

    assert result.payload["processes"]["reactions"][0]["inputs"] == []
    assert gap.gap_id in _unresolved_ids(result)


def test_the_applier_refuses_a_partial_set_at_the_write_too() -> None:
    """Atomicity enforced where the mutation happens, not only where it is planned."""
    payload = {
        "entities": {"species": [{"name": REQUESTED_ORGANISM}], "compounds": [{"name": "B"}]},
        "processes": {"reactions": [{"name": "R0", "inputs": [], "outputs": ["B"]}]},
    }
    proposal = TypedResolution(
        gap_id="g",
        gap_kind="missing_precursor",
        target="R0",
        kind="precursor",
        value={"participants": ["Q", "B"], "side": "inputs", "reaction": "R0"},
        status="proposed",
    )
    records = _apply_typed_resolutions(payload, [proposal])

    assert records[0]["applied"] is False
    assert any(
        r.startswith(REASON_MULTI_PARTICIPANT_REPAIR) for r in records[0]["reasons"]
    ), records
    assert payload["processes"]["reactions"][0]["inputs"] == []


# ===========================================================================
# 4. No first-hit / retrieval-order dependence.
# ===========================================================================
def _identity_seed() -> dict:
    return {
        "entities": {
            "species": [{"name": REQUESTED_ORGANISM}],
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


LOCAL_CHUNK = _chunk(
    "In Pseudomonas putida caffeine degradation, EnzX (UniProt P12345) converts A to B.",
    cid="c-local",
    source="PMC_LOCAL",
    organism=REQUESTED_ORGANISM,
    observed_organisms=[REQUESTED_ORGANISM],
    observed_pathways=[REQUESTED_PATHWAY],
    title="Caffeine degradation in Pseudomonas putida",
)
FOREIGN_CHUNK = _chunk(
    "In Homo sapiens cholesterol biosynthesis, EnzX (UniProt P12345) converts A to B.",
    cid="c-foreign",
    source="PMC_FOREIGN",
    organism="Homo sapiens",
    observed_organisms=["Homo sapiens"],
    observed_pathways=["cholesterol biosynthesis"],
    title="Cholesterol biosynthesis in human cells",
)


def _identity_run(chunks):
    gap = Gap(
        kind="unmapped_enzyme",
        label="EnzX",
        symbols=["EnzX"],
        source="gate",
        requested_pathway=REQUESTED_PATHWAY,
        requested_organism=REQUESTED_ORGANISM,
    )
    bundles = [_bundle(gap, chunk) for chunk in chunks]
    result = synthesize_with_report(
        _identity_seed(),
        bundles,
        _context(),
        gaps=[gap],
        requested_pathway=REQUESTED_PATHWAY,
        requested_organism=REQUESTED_ORGANISM,
        identity_verifier=lambda _n, _v: True,
    )
    return gap, result


def test_a_rejected_foreign_proposal_does_not_suppress_the_valid_local_one() -> None:
    """Same accession, two papers, one out of scope. The local one still applies.

    Under the old ``(gap_id, kind, value)`` collapse this was a coin flip decided
    by retrieval order: with the foreign passage first, its rejection was the only
    surviving record and a resolvable gap stayed open.
    """
    gap, result = _identity_run([FOREIGN_CHUNK, LOCAL_CHUNK])

    proposals = result.admission["resolutions"]
    assert len(proposals) == 2, proposals
    by_source = {p["evidence"]["source_id"]: p for p in proposals}
    assert by_source["PMC_LOCAL"]["applied"] is True
    assert by_source["PMC_FOREIGN"]["applied"] is False
    assert by_source["PMC_FOREIGN"]["organism_match"] == "mismatch"

    proteins = {r["name"]: r for r in result.payload["entities"]["proteins"]}
    assert proteins["EnzX"]["mapped_ids"] == {"uniprot": "P12345"}
    assert gap.gap_id not in _unresolved_ids(result)


def test_an_applied_proposal_does_not_hide_the_contradicting_mismatch() -> None:
    """The foreign paper's disagreement stays visible in the report."""
    _gap, result = _identity_run([LOCAL_CHUNK, FOREIGN_CHUNK])

    statuses = {
        p["evidence"]["source_id"]: (p["status"], p["organism_match"])
        for p in result.admission["resolutions"]
    }
    assert statuses["PMC_FOREIGN"][0] == "rejected"
    assert statuses["PMC_FOREIGN"][1] == "mismatch"
    assert statuses["PMC_LOCAL"][0] == "applied"


def test_both_retrieval_orders_produce_identical_payloads_and_reports() -> None:
    """The invariant: permuting the bundles changes nothing at all."""
    gap_a, forward = _identity_run([LOCAL_CHUNK, FOREIGN_CHUNK])
    gap_b, reverse = _identity_run([FOREIGN_CHUNK, LOCAL_CHUNK])

    assert gap_a.gap_id == gap_b.gap_id
    assert forward.payload == reverse.payload
    assert forward.admission["resolutions"] == reverse.admission["resolutions"]
    assert forward.unresolved_gaps == reverse.unresolved_gaps
    # Not a vacuous comparison: both really did see both papers.
    assert len(forward.admission["resolutions"]) == 2


def test_identical_evidence_from_the_same_chunk_still_collapses() -> None:
    """Preserving scope and provenance is not "keep every duplicate"."""
    _gap, result = _identity_run([LOCAL_CHUNK, LOCAL_CHUNK])
    assert len(result.admission["resolutions"]) == 1


# ===========================================================================
# 5. Production identity reporting stays honest.
# ===========================================================================
def test_the_reported_identity_outcomes_are_four_separate_counts() -> None:
    """Verified, EC-only annotation, rejected, and Unknown fallback are distinct.

    Rolling them together is how "the gap was addressed" comes to mean four
    different things at once. Each is exercised here and each lands in its own
    place in the payload.
    """
    seed = _identity_seed()
    gap = Gap(
        kind="unmapped_enzyme",
        label="EnzX",
        symbols=["EnzX"],
        source="gate",
        requested_pathway=REQUESTED_PATHWAY,
        requested_organism=REQUESTED_ORGANISM,
    )

    def _run(text, **kwargs):
        return synthesize_with_report(
            copy.deepcopy(seed),
            [_bundle(gap, _chunk(text))],
            _context(),
            gaps=[gap],
            requested_pathway=REQUESTED_PATHWAY,
            requested_organism=REQUESTED_ORGANISM,
            **kwargs,
        )

    # (a) verified -> mapped_ids, gap closed.
    verified = _run("EnzX (UniProt P12345) converts A to B.", identity_verifier=lambda *_: True)
    assert gap.gap_id not in _unresolved_ids(verified)

    # (b) EC only -> annotation, gap STILL open (EC is not export identity).
    ec_only = _run("EnzX (EC 1.2.3.4) converts A to B.", identity_verifier=lambda *_: True)
    ec_row = next(r for r in ec_only.payload["entities"]["proteins"] if r["name"] == "EnzX")
    assert ec_row["mapped_ids"] == {"ec": "1.2.3.4"}
    assert not has_protein_external_identity(ec_row)
    assert gap.gap_id in _unresolved_ids(ec_only)

    # (c) no verifier wired (production) -> nothing written, routed to the resolver.
    fallback = _run("EnzX (UniProt P12345) converts A to B.")
    assert "mapped_ids" not in next(
        r for r in fallback.payload["entities"]["proteins"] if r["name"] == "EnzX"
    )
    routed = next(
        row for row in fallback.unresolved_gaps if row.get("gap_id") == gap.gap_id
    )
    assert routed["recommended_route"] == "identity_resolver"

    # (d) rejected on scope -> reported, nothing written.
    rejected = synthesize_with_report(
        copy.deepcopy(seed),
        [_bundle(gap, FOREIGN_CHUNK)],
        _context(),
        gaps=[gap],
        requested_pathway=REQUESTED_PATHWAY,
        requested_organism=REQUESTED_ORGANISM,
        identity_verifier=lambda *_: True,
    )
    assert rejected.admission["resolutions"][0]["status"] == "rejected"
    assert gap.gap_id in _unresolved_ids(rejected)

    # The four outcomes really are four.
    assert len(
        {
            gap.gap_id not in _unresolved_ids(verified),
            gap.gap_id not in _unresolved_ids(ec_only),
        }
    ) == 2


# ===========================================================================
# 6-8. Reconciliation: one decision per (gap_id, kind, target), never a
#      deterministic arbitrary write.
# ===========================================================================
# Sorting the proposals made the ORDER deterministic; it did not make the
# OUTCOME evidence-based. Applying a group one member at a time still let the
# first member write and a contradicting second member overwrite it, or be
# reported applied on top of it. The winner was whichever value sorted first,
# which is not evidence about biology.
def _json(value) -> str:
    import json

    return json.dumps(value, sort_keys=True)


def _identity_gap() -> Gap:
    return Gap(
        kind="unmapped_enzyme",
        label="EnzX",
        symbols=["EnzX"],
        source="gate",
        requested_pathway=REQUESTED_PATHWAY,
        requested_organism=REQUESTED_ORGANISM,
    )


def _run_group(seed: dict, gap: Gap, chunks, **kwargs):
    return synthesize_with_report(
        copy.deepcopy(seed),
        [_bundle(gap, chunk) for chunk in chunks],
        _context(),
        gaps=[gap],
        requested_pathway=REQUESTED_PATHWAY,
        requested_organism=REQUESTED_ORGANISM,
        **kwargs,
    )


ACC_A = _chunk("EnzX (UniProt P12345) converts A to B.", cid="cA", source="PMC_A")
ACC_B = _chunk("EnzX (UniProt Q99999) converts A to B.", cid="cB", source="PMC_B")
ACC_A2 = _chunk("EnzX (UniProt P12345) also converts A to B.", cid="cA2", source="PMC_A2")
ACC_EC = _chunk("EnzX (EC 1.2.3.4) converts A to B.", cid="cE", source="PMC_E")


# --- probe 1: identity conflict ------------------------------------------
def test_two_accessions_for_one_protein_write_neither() -> None:
    """THE identity conflict probe. A permissive verifier confirms both — and
    confirming both is confirming nothing.

    ``P12345`` (paper A) and ``Q99999`` (paper B) are distinct values of a
    SINGULAR identity field. Applied sequentially, one would have overwritten the
    other and both would have been reported applied; applied with a "first wins"
    rule, the answer would be alphabetical. Neither is written, the gap stays
    open, and both values are reported with their sources.
    """
    gap = _identity_gap()
    result = _run_group(
        _identity_seed(), gap, [ACC_A, ACC_B], identity_verifier=lambda *_: True
    )

    rows = result.admission["resolutions"]
    assert len(rows) == 2
    assert [r["applied"] for r in rows] == [False, False]
    assert {r["status"] for r in rows} == {"rejected"}

    reason = rows[0]["reasons"][0]
    assert reason.startswith(REASON_CONFLICTING_RESOLUTION)
    # Both values AND both source ids are named.
    for token in ("P12345", "Q99999", "PMC_A", "PMC_B"):
        assert token in reason, reason

    proteins = {r["name"]: r for r in result.payload["entities"]["proteins"]}
    assert "mapped_ids" not in proteins["EnzX"]
    assert not has_protein_external_identity(proteins["EnzX"])
    assert gap.gap_id in _unresolved_ids(result)


def test_the_identity_conflict_is_byte_identical_under_reversed_paper_order() -> None:
    """Reversing the papers changes nothing, down to the serialized bytes."""
    gap_f = _identity_gap()
    gap_r = _identity_gap()
    forward = _run_group(
        _identity_seed(), gap_f, [ACC_A, ACC_B], identity_verifier=lambda *_: True
    )
    reverse = _run_group(
        _identity_seed(), gap_r, [ACC_B, ACC_A], identity_verifier=lambda *_: True
    )

    assert _json(forward.payload) == _json(reverse.payload)
    assert _json(forward.admission["resolutions"]) == _json(
        reverse.admission["resolutions"]
    )
    assert _json(forward.unresolved_gaps) == _json(reverse.unresolved_gaps)
    assert _json(forward.admission["identity_outcomes"]) == _json(
        reverse.admission["identity_outcomes"]
    )


def test_a_resolver_that_names_exactly_one_winner_may_break_the_tie() -> None:
    """Fail-closed is not "never resolvable" — it is "not without an adjudicator".

    A real ladder judges an accession against resolver candidates with score and
    margin, so it confirms at most one of two competing values. That single
    winner is applied; the loser is still reported as the conflict it was.
    """
    gap = _identity_gap()
    result = _run_group(
        _identity_seed(),
        gap,
        [ACC_A, ACC_B],
        identity_verifier=lambda _n, value: value.get("uniprot") == "P12345",
    )

    by_source = {r["evidence"]["source_id"]: r for r in result.admission["resolutions"]}
    assert by_source["PMC_A"]["applied"] is True
    assert by_source["PMC_B"]["applied"] is False
    assert by_source["PMC_B"]["reasons"][0].startswith(REASON_CONFLICTING_RESOLUTION)

    proteins = {r["name"]: r for r in result.payload["entities"]["proteins"]}
    assert proteins["EnzX"]["mapped_ids"] == {"uniprot": "P12345"}
    assert gap.gap_id not in _unresolved_ids(result)


# --- probe 1b: corroboration is not conflict ------------------------------
def test_the_same_accession_from_two_papers_writes_once_and_keeps_both_refs() -> None:
    """Corroboration raises confidence, not the number of mutations."""
    gap = _identity_gap()
    result = _run_group(
        _identity_seed(), gap, [ACC_A, ACC_A2], identity_verifier=lambda *_: True
    )

    rows = result.admission["resolutions"]
    # Both evidence records are retained in the diagnostic report...
    assert len(rows) == 2
    assert {r["evidence"]["source_id"] for r in rows} == {"PMC_A", "PMC_A2"}
    assert [r["applied"] for r in rows] == [True, True]

    # ...but there is ONE mutation, with merged provenance.
    proteins = {r["name"]: r for r in result.payload["entities"]["proteins"]}
    assert proteins["EnzX"]["mapped_ids"] == {"uniprot": "P12345"}
    assert {"PMC_A", "PMC_A2"} <= set(proteins["EnzX"]["source_refs"])
    assert gap.gap_id not in _unresolved_ids(result)

    # And the OUTCOME is counted per target, not per corroborating proposal.
    outcomes = result.admission["identity_outcomes"]
    assert outcomes["proposals"]["verified"] == 2
    assert outcomes["targets"]["verified"] == 1


def test_an_ec_annotation_and_a_verified_accession_coexist() -> None:
    """Complementary values: they say different things and do not contradict."""
    gap = _identity_gap()
    result = _run_group(
        _identity_seed(), gap, [ACC_A, ACC_EC], identity_verifier=lambda *_: True
    )

    assert [r["applied"] for r in result.admission["resolutions"]] == [True, True]
    proteins = {r["name"]: r for r in result.payload["entities"]["proteins"]}
    assert proteins["EnzX"]["mapped_ids"] == {"uniprot": "P12345", "ec": "1.2.3.4"}
    assert has_protein_external_identity(proteins["EnzX"])
    assert gap.gap_id not in _unresolved_ids(result)


# --- probe 2: precursor conflict ------------------------------------------
PREC_P = _chunk("P is converted to B by EnzX.", cid="cP", source="PMC_P")
PREC_Q = _chunk("Q is converted to B by EnzX.", cid="cQ", source="PMC_Q")
PREC_P2 = _chunk("P is converted to B by EnzX.", cid="cP2", source="PMC_P2")


def test_two_different_precursor_sets_apply_neither() -> None:
    """THE precursor conflict probe. ``P`` does not win for sorting first.

    Two papers state different chemistry for the same empty side. The proposals
    are sorted, so ``P``'s would have been applied first and ``Q``'s then refused
    as ``precursor_side_no_longer_missing`` — a lexical accident reported as a
    repair. Neither is applied and both sets are named with their sources.
    """
    gap = _precursor_gap(missing_side="inputs")
    result = _run_group(_incomplete_seed(), gap, [PREC_P, PREC_Q])

    rows = result.admission["resolutions"]
    assert len(rows) == 2
    assert [r["applied"] for r in rows] == [False, False]
    reason = rows[0]["reasons"][0]
    assert reason.startswith(REASON_CONFLICTING_RESOLUTION)
    for token in ("'P'", "'Q'", "PMC_P", "PMC_Q"):
        assert token in reason, reason

    assert result.payload["processes"]["reactions"][0]["inputs"] == []
    assert gap.gap_id in _unresolved_ids(result)


def test_the_precursor_conflict_is_identical_under_reversed_order() -> None:
    """Same three-way outcome whichever paper arrived first."""
    gap_f = _precursor_gap(missing_side="inputs")
    gap_r = _precursor_gap(missing_side="inputs")
    forward = _run_group(_incomplete_seed(), gap_f, [PREC_P, PREC_Q])
    reverse = _run_group(_incomplete_seed(), gap_r, [PREC_Q, PREC_P])

    assert _json(forward.payload) == _json(reverse.payload)
    assert _json(forward.admission["resolutions"]) == _json(
        reverse.admission["resolutions"]
    )


def test_the_same_precursor_set_from_two_papers_is_applied_once() -> None:
    """Identical chemistry from two sources: one repair, provenance merged."""
    gap = _precursor_gap(missing_side="inputs")
    result = _run_group(_incomplete_seed(), gap, [PREC_P, PREC_P2])

    assert [r["applied"] for r in result.admission["resolutions"]] == [True, True]
    reaction = result.payload["processes"]["reactions"][0]
    assert reaction["inputs"] == ["P"]  # once, not twice
    assert {"PMC_P", "PMC_P2"} <= set(reaction["source_refs"])
    assert gap.gap_id not in _unresolved_ids(result)


# --- probe 3: compartment truthfulness ------------------------------------
LOC_PERI = _chunk("B is localized in the periplasm.", cid="cPeri", source="PMC_PERI")
LOC_CYTO = _chunk("B is localized in the cytosol.", cid="cCyto", source="PMC_CYTO")
LOC_PERI2 = _chunk("B occurs in the periplasm.", cid="cPeri2", source="PMC_PERI2")


def _compartment_gap() -> Gap:
    return Gap(
        kind="missing_compartment",
        label="B",
        symbols=["B"],
        source="gate",
        requested_pathway=REQUESTED_PATHWAY,
        requested_organism=REQUESTED_ORGANISM,
    )


def test_two_locations_are_both_written_never_one_written_and_two_claimed() -> None:
    """THE compartment probe: periplasm AND cytosol, both resolvable.

    The schema supports it — ``t2pw.pwml.ir`` keys a location by
    ``(entity_type, entity_key, biological_state_key)``, so an element in two
    states resolves to two distinct locations. What it never supported was the old
    behaviour: the first proposal wrote a row, the second created an orphan state
    and a subcellular-location row, found "a row for this target" already there,
    wrote nothing, and reported applied anyway.
    """
    gap = _compartment_gap()
    result = _run_group(_identity_seed(), gap, [LOC_PERI, LOC_CYTO])

    rows = result.admission["resolutions"]
    assert [r["applied"] for r in rows] == [True, True]

    located = result.payload["element_locations"]["compound_locations"]
    assert len(located) == 2
    states = {r["biological_state"] for r in located}
    assert len(states) == 2
    assert all(r["compound"] == "B" for r in located)

    # Every claimed state EXISTS, and every declared state is USED.
    declared = {s["name"] for s in result.payload["biological_states"]}
    assert states <= declared
    referenced = states | {
        r.get("biological_state")
        for r in result.payload["processes"]["reactions"]
        if r.get("biological_state")
    }
    assert declared <= referenced, declared - referenced

    # Each location kept its own source.
    by_state = {r["biological_state"]: r for r in located}
    assert sorted(
        ref for r in by_state.values() for ref in r["source_refs"]
    ) == ["PMC_CYTO", "PMC_PERI"]

    # Both subcellular_locations rows are used by a state.
    named = {r["name"] for r in result.payload["entities"]["subcellular_locations"]}
    assert {"periplasm", "cytosol"} <= named
    used = {
        s.get("subcellular_location") for s in result.payload["biological_states"]
    }
    assert {"periplasm", "cytosol"} <= used

    assert gap.gap_id not in _unresolved_ids(result)
    assert validate_post_extraction(result.payload)["ok"] is True
    _ir, report = build_pwml_ir(prefrozen_when_compounded(result.payload), strict_db=False)
    assert report["unresolved"]["biological_state_references"] == []
    validate_required_pwml_contract(result.payload)


def test_the_two_location_outcome_is_identical_under_reversed_order() -> None:
    """Retrieval order does not decide which compartment a metabolite is in."""
    gap_f = _compartment_gap()
    gap_r = _compartment_gap()
    forward = _run_group(_identity_seed(), gap_f, [LOC_PERI, LOC_CYTO])
    reverse = _run_group(_identity_seed(), gap_r, [LOC_CYTO, LOC_PERI])

    assert _json(forward.payload) == _json(reverse.payload)
    assert _json(forward.admission["resolutions"]) == _json(
        reverse.admission["resolutions"]
    )


def test_the_same_location_from_two_papers_merges_onto_one_row() -> None:
    """One location row, both sources on it — not two rows for one claim."""
    gap = _compartment_gap()
    result = _run_group(_identity_seed(), gap, [LOC_PERI, LOC_PERI2])

    assert [r["applied"] for r in result.admission["resolutions"]] == [True, True]
    located = result.payload["element_locations"]["compound_locations"]
    assert len(located) == 1
    assert sorted(located[0]["source_refs"]) == ["PMC_PERI", "PMC_PERI2"]
    assert len(result.payload["biological_states"]) == 1


def test_a_refused_compartment_leaves_no_orphan_state_or_location_row() -> None:
    """Nothing is created until every refusal has had its chance.

    The payload declares no species, so no biological state can be built. The old
    order created the ``subcellular_locations`` row first and left it behind.
    """
    payload = {
        "entities": {"compounds": [{"name": "B"}]},  # no species
        "processes": {"reactions": []},
    }
    proposal = TypedResolution(
        gap_id="gap-missing_compartment-deadbeef",
        gap_kind="missing_compartment",
        target="B",
        kind="compartment",
        value={"location": "periplasm"},
        status="proposed",
    )
    records = _apply_typed_resolutions(payload, [proposal])

    assert records[0]["applied"] is False
    assert "subcellular_locations" not in payload["entities"]
    assert "biological_states" not in payload
    assert "element_locations" not in payload


# --- identity reporting units ---------------------------------------------
def test_identity_outcomes_never_mixes_proposal_counts_with_gap_counts() -> None:
    """Three blocks, three units, and an honest name for the fallback count."""
    gap = _identity_gap()
    result = _run_group(
        _identity_seed(), gap, [ACC_A, ACC_B], identity_verifier=lambda *_: True
    )
    outcomes = result.admission["identity_outcomes"]

    assert set(outcomes) == {
        "proposals",
        "targets",
        "unresolved_identity_gaps",
        "unresolved_for_unknown_fallback",
        "unknown_fallback_present",
    }
    schema = {
        "verified",
        "annotation_only",
        "rejected_scope_or_type",
        "unverified_no_resolver",
        "rejected_by_identity_resolver",
        "conflicting",
    }
    # Proposal and target blocks carry the SAME buckets, so the two are readable
    # against each other; only the population differs.
    assert set(outcomes["proposals"]) == schema
    assert set(outcomes["targets"]) == schema
    # Two conflicting PROPOSALS, one conflicting TARGET, one open GAP.
    assert outcomes["proposals"]["conflicting"] == 2
    assert outcomes["targets"]["conflicting"] == 1
    assert outcomes["unresolved_identity_gaps"] == 1

    # The payload carries no Unknown sentinel: this stage inserts none, so the
    # gap is AWAITING the fallback, not carried by it.
    assert outcomes["unresolved_for_unknown_fallback"] == 1
    assert outcomes["unknown_fallback_present"] == 0


def test_a_gap_already_backed_by_the_unknown_sentinel_is_counted_as_such() -> None:
    """The other side of the same rule: a real sentinel IS reported present."""
    seed = _complex_seed()
    gap = _complex_gap(seed)
    result = synthesize_with_report(
        seed,
        [_bundle(gap, _chunk(COMPLEX_IDENTITY_SPAN))],
        _context(),
        gaps=[gap],
        requested_pathway=REQUESTED_PATHWAY,
        requested_organism=REQUESTED_ORGANISM,
        identity_verifier=lambda *_: True,
    )
    outcomes = result.admission["identity_outcomes"]

    assert outcomes["proposals"]["rejected_scope_or_type"] == 1
    assert outcomes["unresolved_identity_gaps"] == 1
    assert outcomes["unknown_fallback_present"] == 1
    assert outcomes["unresolved_for_unknown_fallback"] == 0


# ---------------------------------------------------------------------------
# The tie-break, driven by the REAL identity ladder.
# ---------------------------------------------------------------------------
# "Apply neither unless a resolver returns one explicit winning verified identity
# with adequate score and margin" is a claim about score and margin, and a lambda
# adjudicator tests neither. These two drive the adjudication through
# ``t2pw.rag.identity.build_identity_verifier`` -> ``verify_real_protein_identity``,
# whose last two rungs ARE the score floor (0.5) and the margin floor (0.1) over
# the best competing candidate.
def _resolver(rows):
    return lambda _name: rows


_DECISIVE_CANDIDATES = [
    {
        "uniprot": "P12345",
        "name": "EnzX",
        "protein_name": "EnzX",
        "gene": "EnzX",
        "organism": REQUESTED_ORGANISM,
        "species": REQUESTED_ORGANISM,
        "score": 0.95,
        "source": "uniprot",
        "entity_type": "protein",
    },
    {
        "uniprot": "Q99999",
        "name": "EnzOther",
        "protein_name": "EnzOther",
        "organism": REQUESTED_ORGANISM,
        "species": REQUESTED_ORGANISM,
        "score": 0.10,
        "source": "uniprot",
        "entity_type": "protein",
    },
]

#: Both candidates claim to BE ``EnzX`` and score within 0.02 of each other, so
#: the ladder's margin rung refuses both. This is the case a "highest score wins"
#: adjudicator would have resolved, wrongly and confidently.
_AMBIGUOUS_CANDIDATES = [
    dict(_DECISIVE_CANDIDATES[0], score=0.92),
    {
        "uniprot": "Q99999",
        "name": "EnzX",
        "protein_name": "EnzX",
        "gene": "EnzX",
        "organism": REQUESTED_ORGANISM,
        "species": REQUESTED_ORGANISM,
        "score": 0.90,
        "source": "uniprot",
        "entity_type": "protein",
    },
]


def test_the_real_ladder_breaks_the_tie_when_the_evidence_is_decisive() -> None:
    """One accession has resolver evidence, the other does not. That one wins."""
    attempts: list = []
    gap = _identity_gap()
    result = _run_group(
        _identity_seed(),
        gap,
        [ACC_A, ACC_B],
        identity_verifier=build_identity_verifier(
            _resolver(_DECISIVE_CANDIDATES),
            organism=REQUESTED_ORGANISM,
            record=attempts,
        ),
    )

    by_source = {r["evidence"]["source_id"]: r for r in result.admission["resolutions"]}
    assert by_source["PMC_A"]["applied"] is True
    assert by_source["PMC_B"]["applied"] is False
    assert by_source["PMC_B"]["reasons"][0].startswith(REASON_CONFLICTING_RESOLUTION)

    # The winner passed the ladder's own rungs, by the ladder's own report.
    winner = next(a for a in attempts if a["verified"])
    assert winner["score"] >= 0.5
    assert winner["checks"]["species"] == "ok"
    assert winner["checks"]["name"] == "keep"
    assert winner["checks"]["score"] == "ok"
    # Only one candidate names this protein, so there is no rival to have a
    # margin over -- the ladder says exactly that rather than inventing a number.
    assert winner["checks"]["margin"] == "no_competing_candidate"

    proteins = {r["name"]: r for r in result.payload["entities"]["proteins"]}
    assert proteins["EnzX"]["mapped_ids"] == {"uniprot": "P12345"}
    assert gap.gap_id not in _unresolved_ids(result)


def test_an_insufficient_margin_leaves_both_accessions_unwritten() -> None:
    """Two plausible candidates 0.02 apart: the ladder confirms NEITHER.

    Both accessions clear the score floor, so a "best score wins" tie-break would
    have picked ``P12345`` and reported a verified identity. The margin rung is
    what makes this a refusal, and the gap stays open for the Unknown fallback.
    """
    attempts: list = []
    gap = _identity_gap()
    result = _run_group(
        _identity_seed(),
        gap,
        [ACC_A, ACC_B],
        identity_verifier=build_identity_verifier(
            _resolver(_AMBIGUOUS_CANDIDATES),
            organism=REQUESTED_ORGANISM,
            record=attempts,
        ),
    )

    assert [r["applied"] for r in result.admission["resolutions"]] == [False, False]
    assert all(not a["verified"] for a in attempts)
    assert {a["reason"] for a in attempts} == {"ambiguous_insufficient_margin"}
    assert all(
        a["checks"]["margin"].startswith("insufficient:") for a in attempts
    ), attempts

    proteins = {r["name"]: r for r in result.payload["entities"]["proteins"]}
    assert "mapped_ids" not in proteins["EnzX"]
    assert gap.gap_id in _unresolved_ids(result)

    outcomes = result.admission["identity_outcomes"]
    assert outcomes["proposals"]["conflicting"] == 2
    assert outcomes["targets"]["conflicting"] == 1
    assert outcomes["proposals"]["verified"] == 0


# ---------------------------------------------------------------------------
# "Nothing judged this claim" is not "the ladder judged it and refused".
# ---------------------------------------------------------------------------
# Both branches used to report the prose "identity not verified: ..." and both
# were counted as ``unverified_no_resolver``, which made a wired resolver's
# considered rejection indistinguishable from never having asked one. The branch
# is now recorded as a machine-readable code by the component that knows it — the
# verifier reports whether it had candidate evidence, ``_apply_identity_decision``
# stamps the code, ``_identity_outcomes`` reads it — and no consumer matches on
# reason text anywhere along that path.
_LOW_SCORING_CANDIDATE = [
    {
        # A GENUINE candidate for this protein and this accession: right name,
        # right species, right entity type. The only thing wrong with it is that
        # the resolver is not confident, and that is a judgement, not an absence.
        "uniprot": "P12345",
        "name": "EnzX",
        "protein_name": "EnzX",
        "gene": "EnzX",
        "organism": REQUESTED_ORGANISM,
        "species": REQUESTED_ORGANISM,
        "score": 0.10,
        "source": "uniprot",
        "entity_type": "protein",
    }
]


def test_a_wired_resolver_rejecting_on_score_is_not_counted_as_no_resolver() -> None:
    """The requested regression: score 0.10 for a real P12345/EnzX candidate.

    The ladder RAN, found the candidate, accepted its entity type, species and
    name, and refused on the score floor. That is ``rejected_by_identity_resolver``
    — the claim was judged and found wanting — and it must not be pooled with the
    production case, where no candidate evidence existed to judge at all.
    """
    attempts: list = []
    gap = _identity_gap()
    result = _run_group(
        _identity_seed(),
        gap,
        [ACC_A],
        identity_verifier=build_identity_verifier(
            _resolver(_LOW_SCORING_CANDIDATE),
            organism=REQUESTED_ORGANISM,
            record=attempts,
        ),
    )

    # The ladder's own verdict: it got as far as the score rung and stopped there.
    assert len(attempts) == 1
    assert attempts[0]["verified"] is False
    assert attempts[0]["reason"] == "score_below_minimum"
    assert attempts[0]["outcome"] == "rejected_by_identity_resolver"
    assert attempts[0]["candidates_considered"] == 1
    assert attempts[0]["checks"]["candidate_evidence"] == "ok"

    outcomes = result.admission["identity_outcomes"]
    assert outcomes["proposals"]["rejected_by_identity_resolver"] == 1
    assert outcomes["proposals"]["unverified_no_resolver"] == 0
    assert outcomes["proposals"]["verified"] == 0
    assert outcomes["targets"]["rejected_by_identity_resolver"] == 1

    proteins = {r["name"]: r for r in result.payload["entities"]["proteins"]}
    assert "mapped_ids" not in proteins["EnzX"]
    assert not has_protein_external_identity(proteins["EnzX"])
    assert gap.gap_id in _unresolved_ids(result)


def test_the_absent_resolver_case_stays_unverified_no_resolver() -> None:
    """The other side, at the same seam: no candidate provider, nothing judged.

    This is what production wires. The verifier is NOT ``None`` — it is the real
    fail-closed adapter — so "was a verifier passed?" is the wrong question and
    the answer has to come from whether the ladder had evidence.
    """
    attempts: list = []
    gap = _identity_gap()
    result = _run_group(
        _identity_seed(),
        gap,
        [ACC_A],
        identity_verifier=fail_closed_verifier(attempts),
    )

    assert attempts[0]["reason"] == "no_local_resolver_evidence"
    assert attempts[0]["outcome"] == "unverified_no_resolver"

    outcomes = result.admission["identity_outcomes"]
    assert outcomes["proposals"]["unverified_no_resolver"] == 1
    assert outcomes["proposals"]["rejected_by_identity_resolver"] == 0
    assert outcomes["proposals"]["verified"] == 0
    assert gap.gap_id in _unresolved_ids(result)


def test_a_provider_with_no_candidates_for_this_protein_is_also_no_resolver() -> None:
    """A provider that knows nothing about EnzX judged nothing about EnzX."""
    attempts: list = []
    result = _run_group(
        _identity_seed(),
        _identity_gap(),
        [ACC_A],
        identity_verifier=build_identity_verifier(
            _resolver([]), organism=REQUESTED_ORGANISM, record=attempts
        ),
    )
    assert attempts[0]["outcome"] == "unverified_no_resolver"
    assert result.admission["identity_outcomes"]["proposals"][
        "unverified_no_resolver"
    ] == 1


def test_the_classification_survives_rewording_the_reason_text() -> None:
    """No consumer recovers the branch from prose.

    A stub verifier that hands back the code and a refusal message deliberately
    phrased like the OTHER branch is still classified by its code. Under the old
    string-prefix rule this reported ``unverified_no_resolver``.
    """
    from t2pw.rag.identity import IdentityCheck

    misleading = IdentityCheck(
        verified=False, identity_outcome="rejected_by_identity_resolver"
    )
    result = _run_group(
        _identity_seed(),
        _identity_gap(),
        [ACC_A],
        identity_verifier=lambda *_: misleading,
    )
    outcomes = result.admission["identity_outcomes"]
    assert outcomes["proposals"]["rejected_by_identity_resolver"] == 1
    assert outcomes["proposals"]["unverified_no_resolver"] == 0


def test_a_bare_boolean_verifier_that_refuses_counts_as_a_resolver_rejection() -> None:
    """A stub with no code to give HAS still judged: it was wired and said no."""
    result = _run_group(
        _identity_seed(), _identity_gap(), [ACC_A], identity_verifier=lambda *_: False
    )
    outcomes = result.admission["identity_outcomes"]
    assert outcomes["proposals"]["rejected_by_identity_resolver"] == 1
    assert outcomes["proposals"]["unverified_no_resolver"] == 0
