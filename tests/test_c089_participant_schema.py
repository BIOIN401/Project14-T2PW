"""C-089 / F-119 / F-125 -- one participant schema, two readers reconciled to it.

Ruling 7 asked for *one* canonical participant-key source consumed by both
readers. Two defects motivated it, and they are different halves of the same
shape -- **a reader narrower than the schema it reads**:

* **F-119** (`mapping/identity_admission.py`). ``_PARTICIPANT_NAME_KEYS`` read
  ``("entity", "name", "ref", "id")``. ``payload_models.py`` declares eight
  name-bearing keys, and ``protein`` -- unreadable -- is the *dominant* actor
  shape in the corpus (1,820 occurrences against 615 for ``entity``).
  ``PARTICIPANT_FIELDS`` additionally omitted ``elements_with_states``, the only
  participant slot ``TransportModel`` and ``ReactionCoupledTransportModel``
  actually have, while listing five fields neither model declares. The failure
  direction is **stripping a correct external accession** from a cofactor a
  reaction genuinely uses.
* **F-125** (`bench/semantic.py`). ``_orphaned_references`` -- acceptance
  priority 3, an ABSOLUTE gate -- read ``inputs``, ``outputs`` and the enzyme
  slots and **never** ``cargo``, ``transporters`` or ``elements_with_states``.
  It counts 3 orphans on the committed corpus where a schema-complete reader
  counts 6.

G9 -- BOTH HALVES ARE CORRECTIONS, NOT NEW CAPABILITY
=====================================================
F-119's measured *corpus* exposure is 0, and calling it new capability on that
basis would be exactly the mislabelling G9 rejects. It is a correction, and a
base-failing behavioural proof exists:

* :func:`test_f119_a_used_cofactor_keeps_its_identity` runs **12 schema-legal
  payloads** through today's production ``map_ids._admit_identities``. In every
  one a reaction or transport **genuinely uses** the cofactor, and at the base
  SHA production refuses it anyway and empties ``mapped_ids`` -- contradicting
  PASS C's own stated rule (``map_ids.py:8334-8338``). 12 of 12 fail at base.
  **The input is SYNTHETIC and schema-legal, and is labelled as such.** G9
  requires the proof to fail *behaviourally* on the base SHA; it does not
  require the input to come from a committed artifact. Two controls
  (:func:`test_f119_control_entity_keyed_actor_is_read`,
  :func:`test_f119_control_bare_string_participant_is_read`) pass at base and at
  tip.
* **Corpus exposure is 0 and this fix changes no committed artifact's outcome.**
  Replaying all 92 committed artifacts with a schema-complete reader rescues
  **0 of the 18** PASS C refusals -- independently reproducing the C-081
  reviewer's result. :func:`test_the_corpus_still_refuses_exactly_eighteen`
  pins that, and it is a control, not a proof: it passes at base too. Do not
  look for a corpus delta here; there is none, by measurement.
* F-125's proof is from **REAL COMMITTED DATA**:
  :func:`test_f125_the_three_invisible_orphans_are_found` and
  :func:`test_f125_paired_ente_caught_in_one_slot_missed_in_the_other`.
  ``elements_with_states`` contributes **0** orphans on this corpus -- its fix
  is latent, like F-119's, and nothing measurable was closed there.

BASE-RUNNABILITY. Every base-failing test above names **no symbol this card
introduces** and builds its input from literals or committed artifacts, so it
runs unchanged on the base SHA and fails on an assertion, never on an import.
The tests that inspect the new module (``test_participant_slots_*``,
``test_*_name_keys_*``, ``test_interactions_*``) are **explicitly labelled NEW
acceptance tests for a new module**; they import it inside their own bodies and
are NOT offered as G9 proofs.
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from t2pw.bench import semantic as S  # noqa: E402
from t2pw.mapping import identity_admission as ia  # noqa: E402
from t2pw.mapping import map_ids  # noqa: E402

#: Spelled out rather than imported, so this file states the contract instead of
#: asserting it against itself.
RULE_COFACTOR_ROLE_UNUSED = "cofactor_role_used_by_no_reaction"

#: PMC12856317's PLP row, trimmed to three accessions. A real molecule with real
#: identifiers; the PAYLOADS around it below are synthetic.
COFACTOR = "Pyridoxal 5'-phosphate"
COFACTOR_IDS = {"chebi": "CHEBI:18405", "kegg": "C00018", "hmdb": "HMDB0001491"}


def _cofactor_payload(processes: Dict[str, Any]) -> Dict[str, Any]:
    """SYNTHETIC, schema-legal. One cofactor-role row, one process using it."""

    return {
        "processes": copy.deepcopy(processes),
        "entities": {"compounds": [{"name": COFACTOR, "class": "cofactor",
                                    "mapped_ids": dict(COFACTOR_IDS)}]},
    }


#: The twelve shapes REV-086 probed. Each is a slot/key pair ``payload_models.py``
#: declares, in which the cofactor is a genuine participant: an input, an output,
#: an enzyme, a modifier, a transporter or an element-with-state.
USED_SHAPES: List[Any] = [
    pytest.param({"reactions": [{"name": "R", "modifiers": [{"protein": COFACTOR,
                                                             "role": "cofactor"}]}]},
                 id="reactions.modifiers-protein"),
    pytest.param({"reactions": [{"name": "R",
                                 "modifiers": [{"protein_complex": COFACTOR}]}]},
                 id="reactions.modifiers-protein_complex"),
    pytest.param({"reactions": [{"name": "R", "enzymes": [{"protein": COFACTOR}]}]},
                 id="reactions.enzymes-protein"),
    pytest.param({"reactions": [{"name": "R", "inputs": [{"compound": COFACTOR,
                                                          "stoichiometry": 1}]}]},
                 id="reactions.inputs-compound"),
    pytest.param({"reactions": [{"name": "R", "outputs": [{"compound": COFACTOR}]}]},
                 id="reactions.outputs-compound"),
    pytest.param({"reactions": [{"name": "R", "inputs": [{"element": COFACTOR}]}]},
                 id="reactions.inputs-element"),
    pytest.param({"reactions": [{"name": "R", "inputs": [{"nucleic_acid": COFACTOR}]}]},
                 id="reactions.inputs-nucleic_acid"),
    pytest.param({"reactions": [{"name": "R",
                                 "inputs": [{"element_collection": COFACTOR}]}]},
                 id="reactions.inputs-element_collection"),
    pytest.param({"transports": [{"name": "T", "cargo": "iron",
                                  "elements_with_states": [{"side": "left",
                                                            "element": COFACTOR}]}]},
                 id="transports.elements_with_states-element"),
    pytest.param({"transports": [{"name": "T", "cargo": "iron",
                                  "transporters": [{"protein": COFACTOR}]}]},
                 id="transports.transporters-protein"),
    pytest.param({"reaction_coupled_transports": [
        {"name": "RCT", "reaction": "R", "transport": "T",
         "elements_with_states": [{"side": "left", "element": COFACTOR}]}]},
        id="rct.elements_with_states-element"),
    pytest.param({"reaction_coupled_transports": [
        {"name": "RCT", "reaction": "R", "transport": "T",
         "enzymes": [{"protein": COFACTOR}]}]},
        id="rct.enzymes-protein"),
]


def _refusals(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    report = map_ids._admit_identities(payload, payload["entities"])
    return [e for e in report["withheld"] if e.get("rule") == RULE_COFACTOR_ROLE_UNUSED]


# ---------------------------------------------------------------------------
# F-119 -- G9 BASE-FAILING PROOF. Synthetic, schema-legal input; corpus delta 0.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("processes", USED_SHAPES)
def test_f119_a_used_cofactor_keeps_its_identity(processes: Dict[str, Any]) -> None:
    """A cofactor a process GENUINELY uses may not have its accessions stripped.

    PASS C's own rule (``map_ids.py:8334-8338``) refuses only a row that "no
    reaction and no transport uses as an input, output, enzyme, modifier, cargo
    or transporter". Every payload here uses it as exactly one of those, written
    in a shape ``payload_models.py`` declares.

    BASE SHA: all twelve refuse and empty ``mapped_ids``. That is the behavioural
    failure G9 requires -- not a missing symbol.
    """

    payload = _cofactor_payload(processes)
    refused = _refusals(payload)
    shipping = payload["entities"]["compounds"][0].get("mapped_ids") or {}
    assert refused == [], (
        f"a cofactor this payload genuinely uses was refused: {refused}")
    assert shipping == COFACTOR_IDS, (
        f"accessions were stripped from a used cofactor: {shipping}")


def test_f119_control_entity_keyed_actor_is_read() -> None:
    """CONTROL -- the one dict key the base reader could already see. Green at base."""

    payload = _cofactor_payload({"reactions": [{"name": "R",
                                                "modifiers": [{"entity": COFACTOR}]}]})
    assert _refusals(payload) == []
    assert payload["entities"]["compounds"][0]["mapped_ids"] == COFACTOR_IDS


def test_f119_control_bare_string_participant_is_read() -> None:
    """CONTROL -- a bare-string participant. Green at base."""

    payload = _cofactor_payload({"reactions": [{"name": "R", "inputs": [COFACTOR]}]})
    assert _refusals(payload) == []
    assert payload["entities"]["compounds"][0]["mapped_ids"] == COFACTOR_IDS


def test_f119_non_vacuity_an_unused_cofactor_is_still_refused() -> None:
    """NON-VACUITY. The widened reader did not disable the rule.

    Same twelve slots, but every process names a DIFFERENT molecule. The
    cofactor row is genuinely unused, and PASS C must still strip it. Without
    this, "no refusals" above would be satisfiable by deleting the rule.
    """

    for processes in (p.values[0] for p in USED_SHAPES):
        other = json.loads(json.dumps(processes).replace(COFACTOR, "succinyl-CoA"))
        payload = _cofactor_payload(other)
        assert _refusals(payload), (
            f"an UNUSED cofactor was not refused; the rule is not firing: {other}")
        assert not payload["entities"]["compounds"][0].get("mapped_ids")


# ---------------------------------------------------------------------------
# F-125 -- G9 BASE-FAILING PROOF from REAL COMMITTED DATA.
# ---------------------------------------------------------------------------
#: The three orphans priority 3 could not see, lifted from committed artifacts.
INVISIBLE_ORPHANS = [
    ("runs/2026-08-02_2130/papers/PMC12096016/strict/final_mapped.json",
     "/processes/transports/0/transporters", "EntE"),
    ("runs/2026-08-02_2130/papers/PMC12096016/strict/final_mapped.json",
     "/processes/transports/1/transporters", "EntE"),
    ("runs_verify/2026-08-24_1402/papers/PMC12180156/research/final_mapped.json",
     "/processes/transports/0/transporters", "/entities/proteins/0"),
]

#: The artifact that carries the paired EntE fixture.
PMC12096016_STRICT = "runs/2026-08-02_2130/papers/PMC12096016/strict/final_mapped.json"


def _artifact(rel: str) -> Dict[str, Any]:
    path = ROOT / rel
    if not path.is_file():
        pytest.skip(f"committed run artifact not present: {rel}")
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.mark.parametrize("rel,pointer,name", INVISIBLE_ORPHANS,
                         ids=[f"{p.rsplit('/', 3)[-1]}-{n}" for _, p, n in INVISIBLE_ORPHANS])
def test_f125_the_three_invisible_orphans_are_found(rel: str, pointer: str,
                                                    name: str) -> None:
    """Priority 3 must see every entity reference a transport actually makes.

    The third is a leaked JSON pointer, ``/entities/proteins/0``, sitting in a
    transporter NAME slot in a committed artifact -- an unambiguous
    referential-integrity violation the absolute gate for referential integrity
    could not see.

    BASE SHA: none of the three is reported, because ``transporters`` is not
    read at all.
    """

    findings = S._orphaned_references(_artifact(rel))
    hit = [f for f in findings if f["pointer"] == pointer and f["name"] == name]
    assert hit, (
        f"{pointer} -> {name!r} is an orphan and priority 3 did not report it; "
        f"reported instead: {[(f['pointer'], f['name']) for f in findings]}")


def test_f125_paired_ente_caught_in_one_slot_missed_in_the_other() -> None:
    """THE SAME NAME, THE SAME ARTIFACT, TWO SLOTS -- so the gap is the slot list.

    ``EntE`` is undeclared in ``PMC12096016/strict``. Priority 3 reports it from
    ``/processes/reactions/3/enzymes`` at base **and** at tip. It does not report
    it from ``/processes/transports/0/transporters`` at base, though the
    transporter entry spells the name under ``entity`` *and* ``protein`` -- keys
    the base reader can already read. Nothing but the slot list explains the
    difference.
    """

    payload = _artifact(PMC12096016_STRICT)
    found = {(f["pointer"], f["name"]) for f in S._orphaned_references(payload)}
    assert ("/processes/reactions/3/enzymes", "EntE") in found, (
        "the enzyme-slot half of the pair regressed; this half is green at base")
    assert ("/processes/transports/0/transporters", "EntE") in found, (
        "the same undeclared name, in the same artifact, is invisible in the "
        "transporter slot -- the slot list is the only difference")


#: ``/processes/transports/0`` of ``PMC12096016/strict``, and the one declared
#: protein around it, lifted verbatim. Kept so the proof survives a checkout
#: without ``runs/``.
LIFTED_TRANSPORT_ORPHAN = {
    "entities": {"proteins": [{"name": "TolC"}], "compounds": [{"name": "enterobactin"}]},
    "processes": {"transports": [{
        "name": "Enterobactin secretion via TolC",
        "cargo": "enterobactin",
        "transporters": [{"entity": "EntE", "entity_type": "protein", "protein": "EntE"}],
        "elements_with_states": [
            {"element": "enterobactin", "side": "left",
             "biological_state": "cytoplasmic state"},
            {"element": "enterobactin", "side": "right",
             "biological_state": "extracellular state"}],
    }]},
}


def test_f125_lifted_transporter_orphan_is_found() -> None:
    """The lifted fixture, corpus-independent. Base: 0 findings."""

    findings = S._orphaned_references(copy.deepcopy(LIFTED_TRANSPORT_ORPHAN))
    assert [(f["pointer"], f["name"]) for f in findings] == [
        ("/processes/transports/0/transporters", "EntE")]


def test_f125_non_vacuity_a_declared_transporter_is_not_an_orphan() -> None:
    """NON-VACUITY. The widened reader reports references, not every string.

    Declare ``EntE`` and the finding must vanish; ``cargo`` and both
    ``elements_with_states`` entries name ``enterobactin``, which IS declared,
    and must stay silent. Without this, the assertion above would be satisfiable
    by reporting everything a transport row contains.
    """

    payload = copy.deepcopy(LIFTED_TRANSPORT_ORPHAN)
    payload["entities"]["proteins"].append({"name": "EntE"})
    assert S._orphaned_references(payload) == []


def test_f125_a_process_name_reference_is_not_an_entity_orphan() -> None:
    """``reaction`` / ``transport`` on a coupled transport name PROCESSES.

    They are real string references and a dangling one is a real defect, but of
    a different class. Resolving them against the ENTITY registry would report
    every well-formed coupled transport as an orphan. Green at base and at tip;
    it pins a boundary this card deliberately did not cross.
    """

    payload = {
        "entities": {"proteins": [{"name": "MenD"}]},
        "processes": {"reaction_coupled_transports": [{
            "name": "RCT",
            "reaction": "menaquinone biosynthesis step 1",
            "transport": "menaquinone export",
            "enzymes": [{"protein": "MenD"}],
        }]},
    }
    assert S._orphaned_references(payload) == []


# ---------------------------------------------------------------------------
# Anti-widening controls. Green at base AND at tip -- they are not G9 proofs.
# ---------------------------------------------------------------------------
def _committed_artifacts() -> List[Path]:
    out = []
    for path in ROOT.rglob("final_mapped.json"):
        parts = path.parts
        if ".git" in parts or "worktrees" in parts:
            continue
        if any(p.startswith(".pytest") or p.startswith("temp_pytest") for p in parts):
            continue
        out.append(path)
    return sorted(out)


def test_the_corpus_still_refuses_exactly_eighteen() -> None:
    """ANTI-WIDENING. This card must not move acceptance priority 1.

    A schema-complete participant reader rescues **0 of the 18** PASS C
    refusals, so no committed artifact's outcome changes and no accession is
    restored. That reproduces the C-081 reviewer's result independently, and it
    is why F-119's corpus exposure is 0 while the defect is still real.
    """

    artifacts = _committed_artifacts()
    if len(artifacts) < 20:
        pytest.skip(f"committed run corpus not present (found {len(artifacts)})")

    refused: List[str] = []
    for path in artifacts:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not isinstance(payload.get("entities"), dict):
            continue
        report = map_ids._admit_identities(payload, payload["entities"])
        refused.extend(e["name"] for e in report["withheld"]
                       if e.get("rule") == RULE_COFACTOR_ROLE_UNUSED)

    assert len(refused) == 18, (
        f"PASS C refused {len(refused)} rows, not the pinned 18: {sorted(refused)}")


def test_the_corpus_enzyme_slot_orphans_are_unchanged() -> None:
    """ANTI-REGRESSION. Widening the slot list did not disturb the old reader.

    The three orphans priority 3 already counted are enzyme-slot findings and
    must survive byte-identical -- same pointer, same name, same count. Green at
    base and at tip.
    """

    artifacts = _committed_artifacts()
    if len(artifacts) < 20:
        pytest.skip(f"committed run corpus not present (found {len(artifacts)})")

    enzyme_findings: List[Any] = []
    for path in artifacts:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not isinstance(payload.get("entities"), dict):
            continue
        rel = path.relative_to(ROOT).as_posix()
        for finding in S._orphaned_references(payload):
            if finding["pointer"].endswith("/enzymes"):
                enzyme_findings.append((rel, finding["pointer"], finding["name"]))

    assert sorted(enzyme_findings) == [
        ("runs/2026-08-02_2130/papers/PMC12096016/strict/final_mapped.json",
         "/processes/reactions/3/enzymes", "EntE"),
        ("runs/2026-08-02_2130/papers/PMC12096016/strict/final_mapped.json",
         "/processes/reactions/3/enzymes", "EntE"),
        ("runs_verify/2026-08-04_1207/papers/PMC12452463/strict/final_mapped.json",
         "/processes/reactions/1/enzymes", "EntD"),
    ]


# ---------------------------------------------------------------------------
# NEW ACCEPTANCE TESTS for the new module. Explicitly labelled: these are NOT
# G9 proofs. They error at the base SHA because the module does not exist there,
# and symbol absence is not proof of anything.
# ---------------------------------------------------------------------------
def _declared_fields(model: Any) -> set:
    return set(model.model_fields)


def _bucket_models() -> Dict[str, Any]:
    from t2pw.pipeline import payload_models as pm

    return {
        "reactions": pm.ReactionModel,
        "transports": pm.TransportModel,
        "reaction_coupled_transports": pm.ReactionCoupledTransportModel,
    }


def test_participant_slots_name_only_declared_model_fields() -> None:
    """NEW. Schema conformance -- so § 4's four corrections cannot regress."""

    from t2pw.pipeline.participant_schema import PARTICIPANT_SLOTS

    models = _bucket_models()
    assert set(PARTICIPANT_SLOTS) == set(models)
    for bucket, slots in PARTICIPANT_SLOTS.items():
        assert slots, f"{bucket} declares no participant slot"
        missing = [s for s in slots if s not in _declared_fields(models[bucket])]
        assert missing == [], (
            f"PARTICIPANT_SLOTS[{bucket!r}] names {missing}, absent from "
            f"{models[bucket].__name__}")


def test_legacy_slots_are_absent_from_every_bucket_model() -> None:
    """NEW. The complement, which is what pins the four corrections.

    ``modifiers`` is NOT a ``TransportModel`` field, and ``inputs``, ``outputs``,
    ``cargo`` and ``transporters`` are NOT ``ReactionCoupledTransportModel``
    fields, though ``PARTICIPANT_FIELDS`` listed all five as if they were. They
    are retained -- ``extra="allow"`` means they can still arrive at runtime, and
    dropping a slot strips identities and hides orphans -- but they are retained
    where a test can see them for what they are.
    """

    from t2pw.pipeline.participant_schema import (
        PARTICIPANT_LEGACY_SLOTS,
        PARTICIPANT_SLOTS,
    )

    models = _bucket_models()
    for bucket, slots in PARTICIPANT_LEGACY_SLOTS.items():
        declared = _declared_fields(models[bucket])
        wrong = [s for s in slots if s in declared]
        assert wrong == [], (
            f"{wrong} are declared fields on {models[bucket].__name__}; they "
            f"belong in PARTICIPANT_SLOTS, not the legacy tail")
        assert not (set(slots) & set(PARTICIPANT_SLOTS[bucket]))

    assert "modifiers" in PARTICIPANT_LEGACY_SLOTS["transports"]
    for field in ("inputs", "outputs", "cargo", "transporters"):
        assert field in PARTICIPANT_LEGACY_SLOTS["reaction_coupled_transports"]
    assert "elements_with_states" in PARTICIPANT_SLOTS["transports"]
    assert "elements_with_states" in PARTICIPANT_SLOTS["reaction_coupled_transports"]


def test_participant_name_keys_are_the_payload_models_union() -> None:
    """NEW. The eight-key union, and the legacy tail kept separate from it.

    ``canonical.py:330`` is NOT the reconciliation target: it omits ``element``,
    and 416 dict-shaped participant entries in the committed corpus carry
    ``element`` and nothing else.
    """

    from t2pw.pipeline import payload_models as pm
    from t2pw.pipeline.participant_schema import (
        PARTICIPANT_LEGACY_NAME_KEYS,
        PARTICIPANT_NAME_KEYS,
        PARTICIPANT_SCHEMA_NAME_KEYS,
    )

    participant_like = (pm.ProcessParticipantModel, pm.ActorModel,
                        pm.ElementWithStateModel)
    declared = set()
    for model in participant_like:
        declared |= _declared_fields(model)

    # Every key in the union is really a field on one of the three models...
    assert set(PARTICIPANT_SCHEMA_NAME_KEYS) <= declared
    assert len(PARTICIPANT_SCHEMA_NAME_KEYS) == 8
    assert "element" in PARTICIPANT_SCHEMA_NAME_KEYS, (
        "canonical.py:330 omits `element`; 416 corpus entries carry it alone")
    # ...and every field on those models that is NOT an entity name is excluded.
    assert declared - set(PARTICIPANT_SCHEMA_NAME_KEYS) == {
        "biological_state", "stoichiometry", "coefficient", "evidence",
        "entity_type", "role", "confidence", "provenance", "source_refs", "side",
    }

    every_field = set()
    for name in dir(pm):
        model = getattr(pm, name)
        if isinstance(model, type) and hasattr(model, "model_fields"):
            every_field |= set(model.model_fields)
    for key in PARTICIPANT_LEGACY_NAME_KEYS:
        assert key not in every_field, f"{key!r} is a model field; promote it"

    assert PARTICIPANT_NAME_KEYS == (
        PARTICIPANT_SCHEMA_NAME_KEYS + PARTICIPANT_LEGACY_NAME_KEYS)
    assert PARTICIPANT_NAME_KEYS[:5] == (
        "name", "entity", "compound", "protein", "protein_complex"), (
        "first-match readers depend on this prefix order")


def test_interactions_is_not_a_participant_bucket() -> None:
    """NEW. SEQUENCING GUARD -- C-089 must not implement any part of D-069.

    Whether an interaction endpoint confers a participant role is C-091's
    ruling. This card defines what a key and a slot ARE. If ``interactions``
    drifts in here as a side effect of the reconciliation, this fails.
    """

    from t2pw.pipeline.participant_schema import (
        PARTICIPANT_LEGACY_SLOTS,
        PARTICIPANT_SLOTS,
        participant_slots,
    )

    assert "interactions" not in PARTICIPANT_SLOTS
    assert "interactions" not in PARTICIPANT_LEGACY_SLOTS
    assert participant_slots("interactions") == ()
    assert "interactions" not in ia.PARTICIPANT_FIELDS
    assert "interactions" not in S._REACTION_BUCKETS
    for slots in list(PARTICIPANT_SLOTS.values()) + list(PARTICIPANT_LEGACY_SLOTS.values()):
        assert not ({"entity_1", "entity_2", "participants"} & set(slots))


def test_process_name_slots_are_never_participant_slots() -> None:
    """NEW. ``reaction`` / ``transport`` reference processes, not entities."""

    from t2pw.pipeline.participant_schema import (
        PARTICIPANT_LEGACY_SLOTS,
        PARTICIPANT_SLOTS,
        PROCESS_NAME_SLOTS,
    )

    models = _bucket_models()
    for bucket, slots in PROCESS_NAME_SLOTS.items():
        for slot in slots:
            assert slot in _declared_fields(models[bucket])
            assert slot not in PARTICIPANT_SLOTS[bucket]
            assert slot not in PARTICIPANT_LEGACY_SLOTS[bucket]


def test_both_readers_are_derived_views_not_copies() -> None:
    """NEW. Ruling 7's actual requirement: ONE source, both readers consume it."""

    from t2pw.pipeline.participant_schema import (
        ENZYME_ROLE_SLOTS,
        PARTICIPANT_NAME_KEYS,
        PARTICIPANT_SLOTS,
        participant_slots,
    )

    assert ia._PARTICIPANT_NAME_KEYS == PARTICIPANT_NAME_KEYS
    assert ia.PARTICIPANT_FIELDS == {b: participant_slots(b) for b in PARTICIPANT_SLOTS}
    assert S._REACTION_BUCKETS == tuple(PARTICIPANT_SLOTS)
    assert S._ENZYME_ROLE_SLOT_SET == frozenset(ENZYME_ROLE_SLOTS)

    # Strictly additive over what identity_admission read before C-089: a slot
    # this reader stops seeing strips a real accession.
    for bucket, before in {
        "reactions": ("inputs", "outputs", "enzymes", "modifiers"),
        "reaction_coupled_transports": ("inputs", "outputs", "enzymes", "modifiers",
                                        "cargo", "transporters"),
        "transports": ("cargo", "transporters", "modifiers"),
    }.items():
        assert set(before) <= set(ia.PARTICIPANT_FIELDS[bucket]), (
            f"{bucket} lost a slot it read before C-089")


def test_schema_conformance_non_vacuity() -> None:
    """NON-VACUITY for the two conformance tests above.

    The helper they rest on must actually discriminate: a field that IS declared
    is seen, and an invented one is not. Without this, both would pass against a
    helper that returned every string.
    """

    models = _bucket_models()
    assert "cargo" in _declared_fields(models["transports"])
    assert "cargo" not in _declared_fields(models["reaction_coupled_transports"])
    assert "modifiers" not in _declared_fields(models["transports"])
    assert "c089_not_a_field" not in _declared_fields(models["reactions"])
