"""Quarantine agrees with the schemas and contracts downstream of it.

Two failure directions, both silent, both covered here:

* **False negatives.** A rule stricter than production quarantines a process that
  would have exported. The first version read only ``cargo`` on a transport and
  only ``entity_1``/``entity_2`` on an interaction, so every transport written
  with ``transport_elements`` and every interaction written ``left``/``right``
  was recorded as having no participants at all. The positive controls below
  exist so that class of bug cannot come back quietly.
* **False accepts.** A rule looser than the downstream contract admits a row and
  the export fails on it three stages later, which is the failure quarantine was
  built to prevent. Those assertions run the *real* gates -- the Stage 3 gate and
  ``validate_required_pwml_contract`` -- over the reduced payload rather than
  checking internal state.
"""

from __future__ import annotations

import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# C-051 / D-015 (LOCKED): the exporter no longer resolves compound identity
# after the canonical freeze, so a raw extraction payload is taken through the
# pre-freeze stage that now does that work. helpers_prefreeze asserts the stage
# actually ruled on every compound row.
from helpers_prefreeze import prefrozen_when_compounded  # noqa: E402
from t2pw.pipeline.process_normalizer import (  # noqa: E402
    GateValidationError,
    run_strict_post_normalization_gates,
)
from t2pw.pipeline.strict_quarantine import (  # noqa: E402
    ACCEPTED_STATES,
    QUARANTINED_BROKEN_REFERENCE,
    QUARANTINED_UNMAPPED_ENTITY,
    RCT_NOT_EXPORTABLE_REASON,
    quarantine_and_close,
)
from t2pw.pwml.ir import (  # noqa: E402
    build_pwml_ir,
    validate_pwml_ir,
    validate_required_pwml_contract,
)


def _base() -> Dict[str, Any]:
    """A pathway that passes both strict gates on its own.

    Every case below adds exactly one thing, so any gate failure afterwards
    belongs to that addition.
    """

    payload: Dict[str, Any] = {
        "metadata": {
            "pw_id": "PW000000",
            "pathway_name": "Glutathione biosynthesis",
            "name": "Glutathione biosynthesis",
            "pathway_subject": "Metabolic",
            "subject": "Metabolic",
            "organism": "Homo sapiens",
            "width": 3200,
            "height": 1400,
            "key_compounds": ["L-glutamate", "glutathione"],
        },
        "entities": {
            "species": [
                {"name": "Homo sapiens", "taxonomy_id": "9606", "classification": "Eukaryote"}
            ],
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
            "protein_locations": [],
        },
        "processes": {
            "reactions": [
                {
                    "name": "gamma-glutamylcysteine synthesis",
                    "inputs": ["L-glutamate", "L-cysteine"],
                    "outputs": ["gamma-glutamylcysteine"],
                    "enzymes": [],
                    "biological_state": "cytosol",
                },
                {
                    "name": "glutathione synthesis",
                    "inputs": ["gamma-glutamylcysteine", "glycine"],
                    "outputs": ["glutathione"],
                    "enzymes": [],
                    "biological_state": "cytosol",
                },
            ],
            "transports": [],
            "interactions": [],
        },
    }
    for reaction, protein in zip(
        payload["processes"]["reactions"],
        ["glutamate-cysteine ligase", "glutathione synthetase"],
    ):
        wrapper = f"{protein} complex"
        payload["entities"]["protein_complexes"].append(
            {
                "name": wrapper,
                "class": "protein_complex",
                "generated": True,
                "generation_reason": "single_protein_pathwhiz_wrapper",
                "species": "Homo sapiens",
                "components": [{"name": protein, "stoichiometry": 1}],
            }
        )
        reaction["enzymes"] = [
            {"entity": wrapper, "entity_type": "protein_complex", "role": "catalyst"}
        ]
    return payload


def _strict_failures(payload: Dict[str, Any]) -> List[str]:
    """Both gate stacks the exporter runs, as one list of reasons."""

    reasons: List[str] = []
    try:
        run_strict_post_normalization_gates(deepcopy(payload), enforce_all_proteins_connected=True)
    except GateValidationError as exc:
        reasons.extend(str(row.get("reason", "")) for row in (exc.details.get("errors") or []))
    contract = validate_required_pwml_contract(deepcopy(payload), strict_db=True)
    if not contract.get("ok"):
        reasons.extend(str(issue.get("code", "")) for issue in contract.get("errors", []))
    return reasons


def _states(result) -> Dict[str, str]:
    return result.states()


def test_the_base_pathway_passes_both_strict_gates() -> None:
    assert _strict_failures(_base()) == []


# ── Transports: all three cargo shapes the IR reads ─────────────────────────


def _transport(**fields: Any) -> Dict[str, Any]:
    return {
        "name": "glutathione efflux",
        "from_biological_state": "cytosol",
        "to_biological_state": "cytosol",
        "transporters": [],
        **fields,
    }


@pytest.mark.parametrize(
    "transport",
    [
        pytest.param(
            _transport(transport_elements=[{"element": "glutathione", "stoichiometry": 1}]),
            id="transport_elements_rows",
        ),
        pytest.param(
            _transport(transport_elements=["glutathione"]),
            id="transport_elements_bare_strings",
        ),
        pytest.param(_transport(cargo="glutathione"), id="cargo"),
        pytest.param(
            _transport(cargo_complex="glutamate-cysteine ligase complex", cargo=""),
            id="cargo_complex",
        ),
    ],
)
def test_every_supported_transport_shape_survives_unchanged(transport: Dict[str, Any]) -> None:
    """``pwml/ir.py``:1780-1789 accepts all of these; so must admission.

    ``transport_elements`` is the primary shape and the bare ``cargo`` keys are
    the fallback. Quarantining a transport written in the primary shape as
    "missing cargo" is a false negative on a completely valid row.
    """

    payload = _base()
    payload["processes"]["transports"] = [deepcopy(transport)]

    result = quarantine_and_close(payload, strict_db=True)

    assert _states(result)["glutathione efflux"] in ACCEPTED_STATES
    assert result.payload["processes"]["transports"] == [transport]


def test_a_transport_with_no_cargo_in_any_shape_is_still_quarantined() -> None:
    payload = _base()
    payload["processes"]["transports"] = [_transport()]

    result = quarantine_and_close(payload, strict_db=True)
    record = next(
        row for row in result.quarantine_report["quarantined"] if row["name"] == "glutathione efflux"
    )

    assert record["state"] == QUARANTINED_BROKEN_REFERENCE
    assert record["reason"] == "missing_cargo"


def test_a_transport_naming_an_undeclared_cargo_is_quarantined() -> None:
    payload = _base()
    payload["processes"]["transports"] = [
        _transport(transport_elements=[{"element": "never declared"}])
    ]

    result = quarantine_and_close(payload, strict_db=True)
    record = next(
        row for row in result.quarantine_report["quarantined"] if row["name"] == "glutathione efflux"
    )

    assert record["state"] == QUARANTINED_UNMAPPED_ENTITY
    assert record["essential_participant"] == "never declared"


# ── Interactions: all three endpoint spellings production resolves ──────────


@pytest.mark.parametrize(
    "endpoints",
    [
        pytest.param({"left": "glutathione", "right": "glutathione synthetase"}, id="left_right"),
        pytest.param(
            {"entity_1": "glutathione", "entity_2": "glutathione synthetase"}, id="entity_1_2"
        ),
        pytest.param(
            {"source": "glutathione", "target": "glutathione synthetase"}, id="source_target"
        ),
    ],
)
def test_every_interaction_endpoint_alias_resolves(endpoints: Dict[str, str]) -> None:
    """``validate_registry_references``:4188 reads all three; so must admission."""

    payload = _base()
    interaction = {"name": "product inhibition", "relationship": "inhibits", **endpoints}
    payload["processes"]["interactions"] = [interaction]

    result = quarantine_and_close(payload, strict_db=True)

    assert _states(result)["product inhibition"] in ACCEPTED_STATES
    assert result.payload["processes"]["interactions"] == [interaction]


def test_an_interaction_missing_one_endpoint_is_quarantined() -> None:
    payload = _base()
    payload["processes"]["interactions"] = [
        {"name": "half an interaction", "left": "glutathione", "relationship": "inhibits"}
    ]

    result = quarantine_and_close(payload, strict_db=True)
    record = next(
        row for row in result.quarantine_report["quarantined"] if row["name"] == "half an interaction"
    )

    assert record["state"] == QUARANTINED_BROKEN_REFERENCE
    assert record["reason"] == "missing_entity_2"


# ── Reaction-coupled transports have no exportable representation ───────────


def test_a_coupled_transport_is_refused_explicitly_rather_than_accepted() -> None:
    """``build_pwml_ir`` cannot represent one, so admitting it is a false accept.

    :1934-1946 builds every RCT with ``left``, ``right`` and ``enzymes`` hard-coded
    to ``[]`` and never fills them from the raw row; :3018-3022 then raises three
    errors for exactly that. A reviewer told "accepted" here would meet those
    errors two stages later with no explanation of where they came from.
    """

    payload = _base()
    payload["processes"]["transports"] = [_transport(cargo="glutathione")]
    payload["processes"]["reaction_coupled_transports"] = [
        {
            "name": "efflux coupling",
            "reaction": "glutathione synthesis",
            "transport": "glutathione efflux",
            "enzymes": [{"entity": "glutathione synthetase complex"}],
        }
    ]

    result = quarantine_and_close(payload, strict_db=True)
    record = next(
        row for row in result.quarantine_report["quarantined"] if row["name"] == "efflux coupling"
    )

    assert record["state"] == QUARANTINED_BROKEN_REFERENCE
    assert record["reason"] == RCT_NOT_EXPORTABLE_REASON
    assert result.payload["processes"]["reaction_coupled_transports"] == []
    # And the reduced payload really is exportable, which is the claim the
    # refusal is making.
    assert result.ok is True
    ir, _report = build_pwml_ir(prefrozen_when_compounded(deepcopy(result.payload)), strict_db=True)
    assert validate_pwml_ir(ir)["ok"] is True


def test_the_ir_still_cannot_represent_a_coupled_transport() -> None:
    """The premise of the refusal above, asserted against the IR itself.

    If this ever fails, the RCT rule is obsolete and must be replaced by real
    end-to-end support rather than left as a refusal of something that works.
    """

    payload = _base()
    payload["processes"]["transports"] = [_transport(cargo="glutathione")]
    payload["processes"]["reaction_coupled_transports"] = [
        {"name": "efflux coupling", "reaction": "glutathione synthesis", "transport": "glutathione efflux"}
    ]

    ir, _report = build_pwml_ir(prefrozen_when_compounded(payload), strict_db=True)
    rct = ir["processes"]["reaction_coupled_transports"][0]

    assert rct["left"] == [] and rct["right"] == [] and rct["enzymes"] == []
    codes = {issue.get("code") for issue in validate_pwml_ir(ir).get("errors", [])}
    assert {"rct_missing_left", "rct_missing_right", "rct_missing_enzyme"} <= codes


# ── Entity representability, measured through the real gates ────────────────


def test_a_protein_without_species_quarantines_the_reaction_and_clears_the_gates() -> None:
    """``protein_missing_species`` is an error for every declared protein row."""

    payload = _base()
    payload["entities"]["proteins"].append({"name": "GPX1", "mapped_ids": {"uniprot": "P07203"}})
    payload["entities"]["compounds"].append({"name": "glutathione disulfide"})
    payload["element_locations"]["compound_locations"].append(
        {"compound": "glutathione disulfide", "biological_state": "cytosol"}
    )
    payload["processes"]["reactions"].append(
        {
            "name": "glutathione oxidation",
            "inputs": ["glutathione"],
            "outputs": ["glutathione disulfide"],
            "modifiers": [{"entity": "GPX1"}],
            "biological_state": "cytosol",
        }
    )
    assert _strict_failures(payload), "the defect must be real before it can be recovered"

    result = quarantine_and_close(payload, strict_db=True)

    record = next(
        row for row in result.quarantine_report["quarantined"] if row["name"] == "glutathione oxidation"
    )
    assert record["reason"] == "protein_missing_species"
    assert result.ok is True
    assert _strict_failures(result.payload) == []


def test_a_nameless_entity_row_is_removed_rather_than_left_to_fail_the_contract() -> None:
    """``compound_missing_name`` fires on unreferenced rows too.

    The contract walks every declared row, so a nameless one it cannot export is
    a failure quarantine has already looked at and waved through unless closure
    drops it. ``ok=True`` followed by a predictable contract failure is the exact
    outcome this forbids.
    """

    payload = _base()
    payload["entities"]["compounds"].append({"mapped_ids": {"hmdb": "HMDB0000123"}})
    payload["entities"]["proteins"].append({"species": "Homo sapiens"})
    assert _strict_failures(payload), "the nameless rows must really fail the contract"

    result = quarantine_and_close(payload, strict_db=True)

    assert result.ok is True
    assert _strict_failures(result.payload) == []
    removed = [
        row
        for row in result.removed_entity_report["removed_entities"]
        if row["reason"] == "malformed_entity_row"
    ]
    assert {row["bucket"] for row in removed} == {"compounds", "proteins"}


def test_a_complex_missing_species_quarantines_the_reaction_that_uses_it() -> None:
    payload = _base()
    payload["entities"]["proteins"].append(
        {"name": "GPX1", "species": "Homo sapiens", "mapped_ids": {"uniprot": "P07203"}}
    )
    payload["entities"]["protein_complexes"].append(
        {
            "name": "glutathione peroxidase complex",
            "class": "protein_complex",
            "generated": True,
            "generation_reason": "single_protein_pathwhiz_wrapper",
            "components": [{"name": "GPX1", "stoichiometry": 1}],
        }
    )
    payload["entities"]["compounds"].append({"name": "glutathione disulfide"})
    payload["element_locations"]["compound_locations"].append(
        {"compound": "glutathione disulfide", "biological_state": "cytosol"}
    )
    payload["processes"]["reactions"].append(
        {
            "name": "glutathione oxidation",
            "inputs": ["glutathione"],
            "outputs": ["glutathione disulfide"],
            "enzymes": [{"entity": "glutathione peroxidase complex"}],
            "biological_state": "cytosol",
        }
    )
    assert _strict_failures(payload)

    result = quarantine_and_close(payload, strict_db=True)

    record = next(
        row for row in result.quarantine_report["quarantined"] if row["name"] == "glutathione oxidation"
    )
    assert record["reason"] == "protein_complex_missing_species"
    assert _strict_failures(result.payload) == []


def test_a_generated_wrapper_component_without_species_is_caught() -> None:
    """``generated_complex_component_missing_species`` is an error for wrappers."""

    payload = _base()
    payload["entities"]["proteins"].append({"name": "GPX1", "mapped_ids": {"uniprot": "P07203"}})
    payload["entities"]["protein_complexes"].append(
        {
            "name": "glutathione peroxidase complex",
            "class": "protein_complex",
            "generated": True,
            "generation_reason": "single_protein_pathwhiz_wrapper",
            "species": "Homo sapiens",
            "components": [{"name": "GPX1", "stoichiometry": 1}],
        }
    )
    payload["entities"]["compounds"].append({"name": "glutathione disulfide"})
    payload["element_locations"]["compound_locations"].append(
        {"compound": "glutathione disulfide", "biological_state": "cytosol"}
    )
    payload["processes"]["reactions"].append(
        {
            "name": "glutathione oxidation",
            "inputs": ["glutathione"],
            "outputs": ["glutathione disulfide"],
            "enzymes": [{"entity": "glutathione peroxidase complex"}],
            "biological_state": "cytosol",
        }
    )
    assert _strict_failures(payload)

    result = quarantine_and_close(payload, strict_db=True)

    record = next(
        row for row in result.quarantine_report["quarantined"] if row["name"] == "glutathione oxidation"
    )
    assert "protein_missing_species" in record["reason"]
    assert _strict_failures(result.payload) == []


def test_a_hand_declared_complex_with_unlisted_subunits_is_not_quarantined() -> None:
    """The contract only *warns* for a non-generated complex, so neither blocks.

    A multi-subunit complex a curator declared legitimately names subunits the
    payload does not carry as separate protein rows. Treating that as blocking
    would quarantine every reaction catalysed by a real complex -- a false
    negative on the most ordinary shape in the schema.
    """

    payload = _base()
    payload["entities"]["protein_complexes"].append(
        {
            "name": "glutathione peroxidase",
            "class": "protein_complex",
            "species": "Homo sapiens",
            "components": [{"name": "GPX subunit alpha"}, {"name": "GPX subunit beta"}],
        }
    )
    payload["entities"]["compounds"].append({"name": "glutathione disulfide"})
    payload["element_locations"]["compound_locations"].append(
        {"compound": "glutathione disulfide", "biological_state": "cytosol"}
    )
    payload["processes"]["reactions"].append(
        {
            "name": "glutathione oxidation",
            "inputs": ["glutathione"],
            "outputs": ["glutathione disulfide"],
            "enzymes": [{"entity": "glutathione peroxidase"}],
            "biological_state": "cytosol",
        }
    )

    result = quarantine_and_close(payload, strict_db=True)

    assert _states(result)["glutathione oxidation"] in ACCEPTED_STATES
    assert result.ok is True
    assert _strict_failures(result.payload) == []


def test_an_unknown_backed_complex_is_preserved_exactly() -> None:
    """Byte-for-byte. Requirement 4 is about preservation, not mere survival."""

    payload = _base()
    unknown_protein = {
        "name": "Unknown",
        "species": "Homo sapiens",
        "organism": "Homo sapiens",
        "pathbank_protein_id": 9659,
        "mapped_ids": {"uniprot": "Unknown", "pathbank_protein_id": 9659},
        "mapping_meta": {
            "chosen_rule": "pathbank_unknown_protein_fallback",
            "fallback_used": True,
            "pathbank_protein_id": 9659,
            "resolution": {"status": "fallback", "issue": "pathbank_unknown_sentinel"},
        },
    }
    unknown_complex = {
        "name": "glutathione hydrolase",
        "class": "protein_complex",
        "generated": True,
        "species": "Homo sapiens",
        "components": [{"name": "Unknown", "stoichiometry": 1}],
        "mapping_meta": {"chosen_rule": "pathbank_unknown_protein_fallback"},
    }
    payload["entities"]["proteins"].append(deepcopy(unknown_protein))
    payload["entities"]["protein_complexes"].append(deepcopy(unknown_complex))
    payload["entities"]["compounds"].append({"name": "cysteinylglycine"})
    payload["element_locations"]["compound_locations"].append(
        {"compound": "cysteinylglycine", "biological_state": "cytosol"}
    )
    payload["processes"]["reactions"].append(
        {
            "name": "glutathione hydrolysis",
            "inputs": ["glutathione"],
            "outputs": ["cysteinylglycine"],
            "enzymes": [{"entity": "glutathione hydrolase"}],
            "biological_state": "cytosol",
        }
    )

    result = quarantine_and_close(payload, strict_db=True)

    assert result.ok is True
    surviving_proteins = result.payload["entities"]["proteins"]
    surviving_complexes = result.payload["entities"]["protein_complexes"]
    assert unknown_protein in surviving_proteins
    assert unknown_complex in surviving_complexes
    assert _strict_failures(result.payload) == []
