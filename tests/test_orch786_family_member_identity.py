"""ORCH-786: a family-complex member whose UniProt id identity admission withholds
must not stop the whole pathway at post_remap (ORCH-784 wnt_ca regression).

G9 labels -- both tests are G9 BEHAVIOURAL PROOFS (a correction of ORCH-783 behaviour):
on base 22538dee ``validate_post_remap`` raises
``generated_wrapper_component_missing_external_identity`` for the ``STIM1/2`` complex;
at tip the id-less member is dropped (KI-001 note) and, with no member left, the complex
falls back to the Unknown-backed member (D-070, D-117 § 2 step 4).

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
from t2pw.mapping import identity_admission, map_ids  # noqa: E402
from t2pw.pipeline.entity_identity import is_pathbank_unknown_protein  # noqa: E402
from t2pw.pipeline.stage_contracts import validate_post_remap  # noqa: E402

_IDS = {"STIM1": "Q13586", "STIM2": "Q9P246", "Orai1": "Q96D31"}


def _resolution(acc):
    return {"status": "mapped", "mapped_ids": {"uniprot": acc}, "provider": "UniProt", "source": "api",
            "confidence": 1.0, "chosen_rule": "uniprot_exact_gene", "candidates": [],
            "resolution": {"status": "matched"}, "identity_verdict": None, "resolved_name": None}


def _map(source_text, tmp_path):
    payload = {
        "entities": {
            "species": [{"name": "Homo sapiens", "taxonomy_id": "9606"}],
            "proteins": [{"name": n, "organism": "Homo sapiens"} for n in ("STIM1/2", "STIM1", "STIM2", "Orai1")],
            "compounds": [],
            "protein_complexes": [],
        },
        "processes": {"reactions": [], "transports": [], "interactions": [
            {"entity_1": "STIM1/2", "entity_2": "Orai1", "relationship": "activates", "name": "STIM1/2 activates Orai1"},
        ]},
        identity_admission.SOURCE_INDEX_KEY: identity_admission.build_source_index(source_text),
    }
    with H.Doors(H.ReplayHttp({}), resolutions={n: _resolution(a) for n, a in _IDS.items()}):
        return H.stage6(payload, tmp_path)


def _family(mapped):
    return next(c for c in mapped["payload"]["entities"]["protein_complexes"] if c["name"] == "STIM1/2")


def test_g9_withheld_member_is_dropped_and_post_remap_passes(tmp_path):
    # The ORCH-784 shape: the paper writes "STIM1/2", never "STIM2", so admission
    # withholds the reused STIM2 row's accession after the complex was built.
    mapped = _map("STIM1/2 detects the decrease in Ca2+ and activates Orai1.", tmp_path)
    validate_post_remap(mapped["payload"])  # base: StageContractError (STIM2 has no id)
    family = _family(mapped)
    assert [c["name"] for c in family["components"]] == ["STIM1"]
    note = family["mapping_meta"]["family_complex"]["note"]
    assert "KI-001" in note and "STIM2" in note


def test_g9_no_identified_member_left_falls_back_to_unknown(tmp_path):
    # Neither member name is locatable, so both accessions are withheld.
    mapped = _map("Stromal interaction molecules activate Orai1.", tmp_path)
    validate_post_remap(mapped["payload"])
    [component] = _family(mapped)["components"]
    proteins = mapped["payload"]["entities"]["proteins"]
    assert is_pathbank_unknown_protein(next(p for p in proteins if p["name"] == component["name"]))
