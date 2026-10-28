"""ORCH-786: the ORCH-784 wnt_ca regression. A family-complex member whose accession
identity admission withholds stopped the whole pathway at post_remap.

G9 labels
- G9 BEHAVIOURAL PROOFS (correction; each fails on base 22538dee, passes at tip):
  * ``test_g9_slash_list_supports_each_member`` (D-118 § 1): the paper writes
    "STIM1/2", so STIM2 keeps its accession. On base, admission withholds it and
    ``validate_post_remap`` raises ``generated_wrapper_component_missing_external_identity``.
  * ``test_g9_unsupported_member_is_dropped`` and
    ``test_g9_no_identified_member_left_falls_back_to_unknown`` (D-117 § 2 step 4):
    an id-less member is dropped with a KI-001 note; with none left the complex gets the
    Unknown member. On base, post_remap raises.
- NEW ACCEPTANCE GUARD (passes on base too, no base failure claimed):
  ``test_new_acceptance_slash_list_does_not_support_unwritten_member`` -- "STIM1/2"
  never supports STIM3.

Offline: tests/orch762_harness.Doors (no PathBank DB, no LLM, every HTTP request 404).
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _entry in (ROOT / "src", ROOT / "tests"):
    if str(_entry) not in sys.path:
        sys.path.insert(0, str(_entry))

import orch762_harness as H  # noqa: E402
from t2pw.mapping import identity_admission  # noqa: E402
from t2pw.pipeline.entity_identity import has_protein_external_identity, is_pathbank_unknown_protein  # noqa: E402
from t2pw.pipeline.stage_contracts import validate_post_remap  # noqa: E402

_IDS = {"STIM1": "Q13586", "STIM2": "Q9P246", "STIM3": "Q00000", "Orai1": "Q96D31"}


def _resolution(name, acc):
    candidate = {"name": name, "gene_name": name, "uniprot": acc, "organism": "Homo sapiens",
                 "taxonomy_id": "9606", "score": 1.0}
    return {"status": "mapped", "mapped_ids": {"uniprot": acc}, "provider": "UniProt", "source": "api",
            "confidence": 1.0, "chosen_rule": "uniprot_exact_gene", "candidates": [candidate],
            "resolution": {"status": "matched"}, "identity_verdict": None, "resolved_name": None}


def _map(source_text, tmp_path, extra=()):
    names = ("STIM1/2", "STIM1", "STIM2", "Orai1", *extra)
    payload = {
        "entities": {
            "species": [{"name": "Homo sapiens", "taxonomy_id": "9606"}],
            "proteins": [{"name": n, "organism": "Homo sapiens"} for n in names],
            "compounds": [],
            "protein_complexes": [],
        },
        "processes": {"reactions": [], "transports": [], "interactions": [
            {"entity_1": "STIM1/2", "entity_2": "Orai1", "relationship": "activates", "name": "STIM1/2 activates Orai1"},
            *({"entity_1": n, "entity_2": "Orai1", "relationship": "activates", "name": f"{n} activates Orai1"}
              for n in extra),
        ]},
        identity_admission.SOURCE_INDEX_KEY: identity_admission.build_source_index(source_text),
    }
    with H.Doors(H.ReplayHttp({}), resolutions={n: _resolution(n, a) for n, a in _IDS.items()}):
        return H.stage6(payload, tmp_path)["payload"]


def _family(mapped):
    return next(c for c in mapped["entities"]["protein_complexes"] if c["name"] == "STIM1/2")


def _protein(mapped, name):
    return next(p for p in mapped["entities"]["proteins"] if p["name"] == name)


def test_g9_slash_list_supports_each_member(tmp_path):
    # The ORCH-784 shape: the paper writes "STIM1/2", never "STIM2".
    mapped = _map("STIM1/2 detects the decrease in Ca2+ and activates Orai1.", tmp_path)
    validate_post_remap(mapped)  # base: StageContractError (STIM2's accession withheld)
    assert [c["name"] for c in _family(mapped)["components"]] == ["STIM1", "STIM2"]
    assert _protein(mapped, "STIM2")["mapped_ids"]["uniprot"] == "Q9P246"


def test_g9_unsupported_member_is_dropped(tmp_path):
    # The paper names STIM1 alone; STIM2 is not written, singly or in a list.
    mapped = _map("STIM1 detects the decrease in Ca2+ and activates Orai1.", tmp_path)
    validate_post_remap(mapped)
    family = _family(mapped)
    assert [c["name"] for c in family["components"]] == ["STIM1"]
    note = family["mapping_meta"]["family_complex"]["note"]
    assert "KI-001" in note and "STIM2" in note


def test_g9_no_identified_member_left_falls_back_to_unknown(tmp_path):
    mapped = _map("Stromal interaction molecules activate Orai1.", tmp_path)
    validate_post_remap(mapped)
    [component] = _family(mapped)["components"]
    assert is_pathbank_unknown_protein(_protein(mapped, component["name"]))


def test_new_acceptance_slash_list_does_not_support_unwritten_member(tmp_path):
    mapped = _map("STIM1/2 detects the decrease in Ca2+ and activates Orai1.", tmp_path, extra=("STIM3",))
    assert not has_protein_external_identity(_protein(mapped, "STIM3"))
    assert has_protein_external_identity(_protein(mapped, "STIM2"))
