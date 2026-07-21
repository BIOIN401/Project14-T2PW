"""WP6 provenance & gates tests — offline, deterministic, self-contained.

The point of this module is to be the **tripwire** for the separation invariant
(docs/rag/03_separation_invariant.md): it proves a synthesized payload survives
the *existing, unmodified* Stage 3 and Stage 8 gates by calling the **real** gate
functions directly — never a RAG-specific variant, never a weakened copy. If any
of these tests start failing, either WP5 synthesis regressed or someone loosened a
gate; the fix is in RAG, never in the gate.

No chromadb, no live network, no live LLM. The guarded ``openai`` stub mirrors
tests/test_audit_json_llm_payload.py because importing the core pipeline modules
(and ``t2pw.rag.synthesize``) transitively reaches ``t2pw.llm.client`` ->
``openai`` at import time. Kept at the top so the module passes run alone.
"""

from __future__ import annotations

import sys
import types
from copy import deepcopy
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

from t2pw.config import rag_config  # noqa: E402
from t2pw.pipeline.process_normalizer import (  # noqa: E402
    GateValidationError,
    run_strict_post_normalization_gates,
)
from t2pw.pwml.ir import validate_required_pwml_contract  # noqa: E402
from t2pw.rag.provenance import (  # noqa: E402
    RAG_ADDITIVE_KEYS,
    strip_provenance,
    validate_provenance,
)
from t2pw.rag.synthesize import synthesize  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures.
# ---------------------------------------------------------------------------
def _prov(source_id: str, uri: str) -> dict:
    """The four WP0 additive keys (RAG_ADDITIVE_KEYS) + core-typed source_refs.

    Exactly the shape WP5 ``synthesize`` attaches to every reaction / non-cofactor
    entity: a primary ``rag_provenance`` pointer (namespaced so it never collides
    with the core ``provenance`` string field), an ``evidence`` passage, the
    ``source_papers`` it came from, a ``rag_confidence``, and the core-owned
    ``source_refs`` list.
    """
    return {
        "rag_provenance": {"source_id": source_id, "source_uri": uri, "source_type": "paper"},
        "evidence": [{"text": "evidence passage", "source_id": source_id, "source_uri": uri}],
        "source_papers": [{"source_id": source_id, "title": "t", "uri": uri, "source_type": "paper"}],
        "source_refs": [source_id],
        "rag_confidence": 0.9,
    }


def _synthesized_mapped_payload() -> dict:
    """A synthesized payload after the core mapping stage has completed.

    WP5 ``synthesize`` emits, at seam S3, a standard ``Payload`` (core
    ``entities`` / ``processes`` buckets) with the additive provenance keys on
    every reaction and non-cofactor entity — but it does **not** carry the
    external DB identities (UniProt / ChEBI / pathbank ids), the ``biological_states``,
    or the generated-complex wrappers that the core mapping/normalization stage
    adds *after* S3. Those steps need the offline-unavailable reference DB, so we
    supply their result here to reach the closest **honest** fixture the Stage 3/8
    gates can accept (they legitimately require a mapped payload). Nothing about
    the gate is stubbed or weakened; the additive provenance keys ride on top and
    are ignored by both gates. The un-mapped ``synthesize`` output is exercised
    separately in :func:`test_real_synthesize_output_is_provenance_complete`.
    """
    return {
        "metadata": {
            "name": "Glycolysis (synthesized)",
            "pathway_name": "Glycolysis (synthesized)",
            "subject": "Metabolic",
            "pathway_subject": "Metabolic",
        },
        "entities": {
            "species": [{"name": "Homo sapiens", "pathwhiz_id": 1}],
            "subcellular_locations": [{"name": "cytosol", "pathwhiz_id": 2}],
            "compounds": [
                {
                    "name": "Glucose",
                    "pathbank_compound_id": 101,
                    "mapped_ids": {"chebi": "CHEBI:17234"},
                    **_prov("PMID:0001", "https://example.org/a"),
                },
                {
                    "name": "Glucose 6-phosphate",
                    "pathbank_compound_id": 102,
                    "mapped_ids": {"chebi": "CHEBI:4170"},
                    **_prov("PMID:0001", "https://example.org/a"),
                },
            ],
            "proteins": [
                {
                    "name": "Hexokinase",
                    "species": "Homo sapiens",
                    "pathbank_protein_id": 201,
                    "mapped_ids": {"uniprot": "P19367"},
                    **_prov("PMID:0002", "https://example.org/b"),
                },
            ],
            "protein_complexes": [
                {
                    "name": "Hexokinase complex",
                    "species": "Homo sapiens",
                    "components": ["Hexokinase"],
                    **_prov("PMID:0002", "https://example.org/b"),
                },
            ],
        },
        "biological_states": [
            {"name": "cytosol", "species": "Homo sapiens", "subcellular_location": "cytosol"},
        ],
        "processes": {
            "reactions": [
                {
                    "name": "Glucose phosphorylation",
                    "inputs": ["Glucose"],
                    "outputs": ["Glucose 6-phosphate"],
                    "biological_state": "cytosol",
                    "enzymes": [
                        {
                            "entity": "Hexokinase complex",
                            "entity_type": "protein_complex",
                            **_prov("PMID:0002", "https://example.org/b"),
                        }
                    ],
                    **_prov("PMID:0001", "https://example.org/a"),
                }
            ],
            "transports": [],
            "interactions": [],
        },
    }


def _stage3_passes(payload: dict) -> bool:
    """Call the REAL Stage 3 gate; passing == no GateValidationError, no errors."""
    try:
        report = run_strict_post_normalization_gates(
            payload, enforce_all_proteins_connected=False
        )
    except GateValidationError:
        return False
    return report.get("errors") == []


# ---------------------------------------------------------------------------
# Test 1 — the synthesized payload survives the UNMODIFIED Stage 3 + Stage 8 gates.
# ---------------------------------------------------------------------------
def test_synthesized_payload_passes_real_stage3_and_stage8_gates() -> None:
    payload = _synthesized_mapped_payload()
    before = deepcopy(payload)

    # Stage 3 — the real strict post-normalization gate.
    assert _stage3_passes(payload) is True

    # Stage 8 — the real PWML pre-export contract.
    contract = validate_required_pwml_contract(payload, strict_db=True)
    assert contract["ok"] is True, [e["code"] for e in contract["errors"]]

    # The additive provenance keys really are present (so this is a genuine test
    # of "provenance survives the gate", not of a plain payload).
    reaction = payload["processes"]["reactions"][0]
    assert reaction["rag_provenance"]["source_id"] == "PMID:0001"
    # The core ``provenance`` string field is never shadowed by the RAG dict.
    assert not isinstance(reaction.get("provenance"), dict)
    assert set(RAG_ADDITIVE_KEYS) <= set(reaction)

    # The gates are read-only w.r.t. the payload identity we care about.
    assert payload == before


# ---------------------------------------------------------------------------
# Test 2 — the SAME gates pass on the provenance-stripped payload (additive proof).
# ---------------------------------------------------------------------------
def test_strip_provenance_payload_also_passes_both_gates() -> None:
    payload = _synthesized_mapped_payload()
    plain = strip_provenance(payload)

    # Not one additive key survives anywhere in the stripped copy...
    def _rag_keys_present(value: object) -> bool:
        if isinstance(value, dict):
            if any(k in value for k in RAG_ADDITIVE_KEYS):
                return True
            return any(_rag_keys_present(v) for v in value.values())
        if isinstance(value, list):
            return any(_rag_keys_present(v) for v in value)
        return False

    assert _rag_keys_present(plain) is False
    # ...and the original is untouched (strip returns a copy).
    assert any(k in payload["processes"]["reactions"][0] for k in RAG_ADDITIVE_KEYS)

    # Both real gates pass on the plain payload too — provenance is purely additive.
    assert _stage3_passes(plain) is True
    assert validate_required_pwml_contract(plain, strict_db=True)["ok"] is True


# ---------------------------------------------------------------------------
# Test 3 — validate_provenance accepts a fully-sourced payload.
# ---------------------------------------------------------------------------
def test_validate_provenance_accepts_fully_sourced_payload() -> None:
    report = validate_provenance(_synthesized_mapped_payload())

    assert report.ok is True
    assert report.issues == []
    assert report.checked_reactions == 1
    # Glucose, Glucose 6-phosphate, Hexokinase, Hexokinase complex.
    assert report.checked_entities == 4


# ---------------------------------------------------------------------------
# Test 4 — validate_provenance FAILS on an unsourced reaction (the core tripwire).
# ---------------------------------------------------------------------------
def test_validate_provenance_flags_unsourced_reaction() -> None:
    payload = _synthesized_mapped_payload()
    payload["processes"]["reactions"].append(
        {
            "name": "invented step",
            "inputs": ["Glucose 6-phosphate"],
            "outputs": ["Fructose 6-phosphate"],
            # No provenance / evidence / source_papers / source_refs at all.
        }
    )

    report = validate_provenance(payload)

    assert report.ok is False
    offenders = {issue.label for issue in report.issues if issue.kind == "reaction"}
    assert "invented step" in offenders
    assert all(issue.reason for issue in report.issues)


def test_validate_provenance_flags_unsourced_noncofactor_entity() -> None:
    payload = _synthesized_mapped_payload()
    # A brand-new non-cofactor compound with no source pointer of any kind.
    payload["entities"]["compounds"].append({"name": "Fructose 6-phosphate"})

    report = validate_provenance(payload)

    assert report.ok is False
    assert any(
        issue.kind == "compound" and issue.label == "Fructose 6-phosphate"
        for issue in report.issues
    )


def test_validate_provenance_exempts_cofactors() -> None:
    payload = _synthesized_mapped_payload()
    # ATP is a ubiquitous cofactor -> exempt from the provenance requirement even
    # with no source pointer. The report must still be ok.
    payload["entities"]["compounds"].append({"name": "ATP"})

    report = validate_provenance(payload)

    assert report.ok is True
    assert all(issue.label != "ATP" for issue in report.issues)


# ---------------------------------------------------------------------------
# Test 5 — RAG_ENABLED=false: provenance keys are absent/ignored.
# ---------------------------------------------------------------------------
def test_provenance_absent_and_gates_pass_when_rag_disabled(monkeypatch) -> None:
    monkeypatch.setenv("RAG_ENABLED", "false")
    assert rag_config()["enabled"] is False

    # With RAG off, the payload the core pipeline sees carries no provenance keys.
    # strip_provenance models that plain view; it must contain none of the
    # additive keys and must still pass both real gates unchanged.
    plain = strip_provenance(_synthesized_mapped_payload())

    for bucket in ("compounds", "proteins", "protein_complexes"):
        for row in plain["entities"][bucket]:
            assert not any(k in row for k in RAG_ADDITIVE_KEYS)
    for reaction in plain["processes"]["reactions"]:
        assert not any(k in reaction for k in RAG_ADDITIVE_KEYS)

    assert _stage3_passes(plain) is True
    assert validate_required_pwml_contract(plain, strict_db=True)["ok"] is True


# ---------------------------------------------------------------------------
# Test 6 — the REAL WP5 synthesizer emits provenance-complete output.
# ---------------------------------------------------------------------------
def test_real_synthesize_output_is_provenance_complete() -> None:
    """Tie the canned fixture back to reality: genuine ``synthesize`` output (the
    un-mapped S3 payload) already satisfies ``validate_provenance`` — every
    reaction and non-cofactor entity is evidence-bound by construction.
    """
    seed_payload = {
        "processes": {
            "reactions": [
                {
                    "name": "caffeine demethylation",
                    "inputs": ["caffeine"],
                    "outputs": ["theobromine"],
                    "enzymes": [{"protein": "NdmA"}],
                    "source_papers": [
                        {"source_id": "PMID:seed", "title": "A", "uri": "u", "source_type": "paper"}
                    ],
                }
            ]
        }
    }
    seed_context = {
        "source": {
            "source_id": "PMID:seed",
            "source_title": "A",
            "source_type": "paper",
            "source_uri": "u",
        }
    }

    payload = synthesize(seed_payload, [], seed_context)
    report = validate_provenance(payload)

    assert report.ok is True, [issue.__dict__ for issue in report.issues]
    assert report.checked_reactions >= 1
    # Every emitted reaction carries a resolvable source_ref.
    for reaction in payload["processes"]["reactions"]:
        assert reaction.get("source_refs")
