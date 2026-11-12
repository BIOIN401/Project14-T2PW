"""Final payload writer fixes (bound-visualizations empty, spontaneous nil, transport compound locations).

Every fix is gated by T2PW_FINAL_PAYLOAD; with the flag off the writer output is unchanged.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from test_pwml_writer import build_pwml_ir  # noqa: E402  (Stage-6 + pre-freeze wrapper)
from t2pw.pwml.validate import discover_structure_signature  # noqa: E402
from t2pw.pwml.writer import DeterministicPwmlBuilder  # noqa: E402


def _payload(reaction_spontaneous: object = None, reaction_uses_glucose: bool = False) -> dict:
    reactions = []
    if reaction_uses_glucose:
        reactions.append({"name": "Glucose to G6P", "inputs": ["Glucose"], "outputs": ["G6P"],
                          "biological_state": "cytosol"})
    elif reaction_spontaneous is not None:
        reactions.append({"name": "A to B", "inputs": ["A"], "outputs": ["B"],
                          "biological_state": "cytosol", "spontaneous": reaction_spontaneous})
    if reaction_uses_glucose and reaction_spontaneous is not None:
        reactions[0]["spontaneous"] = reaction_spontaneous
    return {
        "entities": {
            "species": [{"name": "Homo sapiens", "pathwhiz_id": 1}],
            "subcellular_locations": [{"name": "extracellular", "pathwhiz_id": 2}, {"name": "cytosol", "pathwhiz_id": 3}],
            "compounds": [{"name": "Glucose", "pathbank_compound_id": 101}, {"name": "G6P", "pathbank_compound_id": 102},
                          {"name": "A", "pathbank_compound_id": 103}, {"name": "B", "pathbank_compound_id": 104}],
            "proteins": [{"name": "GLUT1", "species": "Homo sapiens", "uniprot": "P11166", "pathbank_protein_id": 301}],
        },
        "biological_states": [
            {"name": n, "species": "Homo sapiens", "subcellular_location": n, "compartment_canonical": n}
            for n in ("extracellular", "cytosol")
        ],
        "processes": {
            "reactions": reactions,
            "transports": [{"name": "Glucose import", "cargo": "Glucose", "from_biological_state": "extracellular",
                            "to_biological_state": "cytosol", "transporters": [{"protein": "GLUT1"}]}],
            "interactions": [],
        },
    }


def _build(payload: dict, with_bound: bool = False, spontaneous: object = None):
    ir, report = build_pwml_ir(payload, strict_db=True)
    assert report["errors"] == []
    if spontaneous is not None:  # the IR hard-codes False today (ir.py), so set what a future IR would carry
        ir["processes"]["reactions"][0]["spontaneous"] = spontaneous
    if with_bound:
        ir["bound_visualizations"] = [{"key": "bv_1", "biological_state_key": ir["biological_states"][0]["key"],
                                       "x": 0, "y": 0}]
    args = SimpleNamespace(name="Generated Pathway", description="", subject="Metabolic", pw_id="PW000000",
                           height=1400, width=3200, background_color="#FFFFFF", ref=str(ROOT / "reference" / "PW000001.pwml"))
    signature = discover_structure_signature(ROOT / "reference" / "PW000001.pwml")
    builder = DeterministicPwmlBuilder(extraction=ir, signature=signature, args=args)
    return builder, builder.build().root


@pytest.fixture
def flag_on(monkeypatch):
    monkeypatch.setenv("T2PW_FINAL_PAYLOAD", "1")


@pytest.fixture
def flag_off(monkeypatch):
    monkeypatch.delenv("T2PW_FINAL_PAYLOAD", raising=False)


def test_bound_visualizations_written_when_flag_off(flag_off) -> None:
    builder, root = _build(_payload(), with_bound=True)
    assert len(builder.section_items["bound-visualizations"]) == 1
    assert len(root.findall(".//bound-visualizations/bound-visualization")) == 1


def test_bound_visualizations_empty_when_flag_on(flag_on) -> None:
    builder, root = _build(_payload(), with_bound=True)
    assert builder.section_items["bound-visualizations"] == []
    assert root.findall(".//bound-visualizations/bound-visualization") == []


def _spont(root):
    return [r.find("spontaneous") for r in root.findall(".//reactions/reaction")]


@pytest.mark.parametrize("value", [None, False, True])
def test_spontaneous_flag_off_is_plain_bool(flag_off, value) -> None:
    _, root = _build(_payload(reaction_spontaneous=False), spontaneous=value)
    nodes = _spont(root)
    assert len(nodes) == 1 and nodes[0].get("nil") is None
    assert nodes[0].text == ("true" if value else "false")


def test_spontaneous_flag_on_true_or_nil(flag_on) -> None:
    _, root = _build(_payload(reaction_spontaneous=False), spontaneous=True)
    node = _spont(root)[0]
    assert node.get("nil") is None and node.text == "true"
    for not_stated in (False, None):  # the IR default is False today: written as unknown, not as "enzyme-catalysed"
        _, root = _build(_payload(reaction_spontaneous=False), spontaneous=not_stated)
        node = _spont(root)[0]
        assert node.get("nil") == "true" and node.get("type") == "boolean"


def _transport_loc_refs(root):
    return [int(n.text) for n in root.findall(".//transport-visualizations//compound-location-id")]


def _loc_ids(root):
    return [int(n.text) for n in root.findall(".//compound-locations/compound-location/id")]


def test_transport_location_kept_when_compound_also_in_reaction_flag_on(flag_on) -> None:
    _, root = _build(_payload(reaction_uses_glucose=True))
    refs, ids = _transport_loc_refs(root), _loc_ids(root)
    assert refs
    assert set(refs) <= set(ids)
    assert len(ids) == len(set(ids))


def test_transport_location_dangles_when_flag_off(flag_off) -> None:
    _, root = _build(_payload(reaction_uses_glucose=True))
    refs, ids = _transport_loc_refs(root), _loc_ids(root)
    assert refs and not set(refs) <= set(ids)  # the documented defect, unchanged with the flag off


def test_transport_only_compound_unchanged_both_modes(monkeypatch) -> None:
    monkeypatch.delenv("T2PW_FINAL_PAYLOAD", raising=False)
    _, off = _build(_payload())
    monkeypatch.setenv("T2PW_FINAL_PAYLOAD", "1")
    _, on = _build(_payload())
    for root in (off, on):
        assert set(_transport_loc_refs(root)) <= set(_loc_ids(root))


def _subtree_xml(root, tag):
    from lxml import etree
    return [etree.tostring(n) for n in root.iter(tag)]


def test_transport_only_compound_subtrees_identical_both_modes(monkeypatch) -> None:
    def payload():  # no transporter: a plain-protein transporter is wrapped in a complex when the flag is on (B2)
        p = _payload()
        p["processes"]["transports"][0]["transporters"] = []
        return p

    monkeypatch.delenv("T2PW_FINAL_PAYLOAD", raising=False)
    _, off = _build(payload())
    monkeypatch.setenv("T2PW_FINAL_PAYLOAD", "1")
    _, on = _build(payload())
    for tag in ("compound-locations", "transport-visualizations"):
        assert _subtree_xml(off, tag) == _subtree_xml(on, tag) != []


def test_transported_compound_used_by_reaction_in_third_state(monkeypatch) -> None:
    payload = _payload(reaction_uses_glucose=True)
    payload["entities"]["subcellular_locations"].append({"name": "mitochondrion", "pathwhiz_id": 4})
    payload["biological_states"].append({"name": "mitochondrion", "species": "Homo sapiens",
                                         "subcellular_location": "mitochondrion", "compartment_canonical": "mitochondrion"})
    payload["processes"]["reactions"][0]["biological_state"] = "mitochondrion"
    monkeypatch.delenv("T2PW_FINAL_PAYLOAD", raising=False)
    _, off = _build(payload)
    monkeypatch.setenv("T2PW_FINAL_PAYLOAD", "1")
    _, on = _build(payload)
    refs, ids = set(_transport_loc_refs(on)), _loc_ids(on)
    assert refs and refs <= set(ids) and len(ids) == len(set(ids))
    # only the locations a transport references are added back; state C's base row stays skipped
    assert len(ids) - len(_loc_ids(off)) == len(refs)


def test_interaction_compound_member_has_no_dangling_location_flag_on(flag_on) -> None:
    from test_pwml_interaction_export import _payload as ipayload
    payload = ipayload()
    payload["entities"]["compounds"] = [
        {"name": "L-arabinose", "pathbank_compound_id": 101, "mapped_ids": {"chebi": "CHEBI:17535"}},
        {"name": "Y", "pathbank_compound_id": 105},
    ]
    payload["processes"]["reactions"] = [{"name": "ara to Y", "inputs": ["L-arabinose"], "outputs": ["Y"],
                                          "biological_state": "cytosol"}]
    payload["processes"]["interactions"] = [{"name": "arabinose activates AraC", "entity_1": "L-arabinose",
                                             "relationship": "activation", "entity_2": "AraC complex",
                                             "biological_state": "cytosol", "evidence": "q"}]
    _, root = _build(payload)
    refs = [int(n.text) for n in root.findall(".//interaction_compound_visualizations//compound-location-id")]
    assert refs and set(refs) <= set(_loc_ids(root))


def test_no_processes_flag_on_equals_flag_off_minus_bound(monkeypatch) -> None:
    from lxml import etree
    payload = _payload()
    payload["processes"] = {"reactions": [], "transports": [], "interactions": []}

    def dump(flag):
        if flag:
            monkeypatch.setenv("T2PW_FINAL_PAYLOAD", "1")
        else:
            monkeypatch.delenv("T2PW_FINAL_PAYLOAD", raising=False)
        _, root = _build(payload, with_bound=True)
        for n in root.findall(".//bound-visualizations"):
            n.clear()
        # B2 (vocabulary map) renames the location "extracellular" to the PathWhiz row name when the flag is on; undo that here
        out = etree.tostring(root)
        return out.replace(b"Extracellular Space", b"extracellular") if flag else out

    assert dump(False) == dump(True)


# --- B2: plain-protein transporters are written as a one-member protein-complex (PathBank shape) -----------------------

def _ids(root, path):
    return [int(n.text) for n in root.findall(path)]


def _check_transporter_complex(root, n_transporters=1):
    tts = root.findall(".//transport-transporters/transport-transporter")
    assert len(tts) == n_transporters
    complexes = {int(c.findtext("id")): c for c in root.findall(".//protein-complexes/protein-complex")}
    protein_ids = set(_ids(root, ".//proteins/protein/id"))
    pcvs = {int(v.findtext("id")): v for v in root.findall(".//protein-complex-visualizations/protein-complex-visualization")}
    ploc = {int(l.findtext("id")): l for l in root.findall(".//protein-locations/protein-location")}
    for tt in tts:
        assert tt.find("protein-id") is None
        cid = int(tt.findtext("protein-complex-id"))
        assert cid in complexes
        members = complexes[cid].findall("protein_complex-proteins/protein-complex-protein")
        assert len(members) == 1 and int(members[0].findtext("protein-id")) in protein_ids
        assert members[0].find("stoichiometry").get("nil") == "true"
    for tv in root.findall(".//transport_transporter_visualizations/transport-transporter-visualization"):
        assert tv.find("protein-location-id") is None
        assert int(tv.findtext("transport-transporter-id")) in _ids(root, ".//transport-transporters/transport-transporter/id")
        pcv = pcvs[int(tv.findtext("protein-complex-visualization-id"))]
        assert int(pcv.findtext("protein-complex-id")) in complexes
        locs = [ploc[int(n.text)] for n in pcv.findall(".//protein-complex-protein-visualization/protein-location-id")]
        assert len(locs) == 1
        assert int(locs[0].findtext("protein-id")) == int(
            complexes[int(pcv.findtext("protein-complex-id"))].find("protein_complex-proteins/protein-complex-protein/protein-id").text)
    for tag in ("protein-complexes/protein-complex", "protein-complex-visualizations/protein-complex-visualization"):
        ids = _ids(root, f".//{tag}/id")
        assert len(ids) == len(set(ids))
    return complexes


def test_plain_protein_transporter_becomes_complex_flag_on(flag_on) -> None:
    _, root = _build(_payload())
    complexes = _check_transporter_complex(root)
    assert len(complexes) == 1 and next(iter(complexes.values())).findtext("name") == "GLUT1 complex"
    assert len(root.findall(".//protein-complex-visualizations/protein-complex-visualization")) == 1


def test_two_transports_same_protein_share_one_wrapper_flag_on(flag_on) -> None:
    payload = _payload()
    second = dict(payload["processes"]["transports"][0], name="Glucose export",
                  from_biological_state="cytosol", to_biological_state="extracellular")
    payload["processes"]["transports"].append(second)
    _, root = _build(payload)
    complexes = _check_transporter_complex(root, n_transporters=2)
    assert len(complexes) == 1
    assert len(root.findall(".//transport-visualizations/transport-visualization")) == 2


def test_plain_protein_transporter_unchanged_flag_off(flag_off) -> None:
    _, root = _build(_payload())
    tt = root.find(".//transport-transporters/transport-transporter")
    assert tt.find("protein-id") is not None and tt.find("protein-complex-id") is None
    assert root.findall(".//protein-complexes/protein-complex") == []
    tv = root.find(".//transport_transporter_visualizations/transport-transporter-visualization")
    assert tv.find("protein-location-id") is not None and tv.find("protein-complex-visualization-id") is None


# --- B1 minors, B3 nucleic-acid type ---------------------------------------------------------------------------------

def _build_mutated(mutate):
    import t2pw.pwml.writer as w
    orig = w.DeterministicPwmlBuilder.__init__

    def init(self, extraction, *a, **k):
        mutate(extraction)
        orig(self, extraction, *a, **k)
    return init


def test_nameless_transporter_complex_falls_back_to_entity_key_flag_on(flag_on, monkeypatch) -> None:
    def blank(ir):
        for p in ir["entities"]["proteins"]:
            p["name"] = ""
    monkeypatch.setattr(DeterministicPwmlBuilder, "__init__", _build_mutated(blank))
    _, root = _build(_payload())
    name = root.find(".//protein-complexes/protein-complex/name").text
    assert name.strip() != "complex" and name.endswith(" complex") and name != " complex"


def test_synth_complex_id_avoids_explicit_complex_id_flag_on(flag_on) -> None:
    payload = _payload()
    payload["entities"]["protein_complexes"] = [
        {"name": "Cx", "pathbank_protein_complex_id": 40000, "components": [{"name": "GLUT1"}]}]
    _, root = _build(payload)
    ids = _ids(root, ".//protein-complexes/protein-complex/id")
    assert len(ids) == 2 and len(set(ids)) == 2 and 40000 in ids


def _na_payload(with_ids: bool = False) -> dict:
    p = _payload()
    names = ["ent mRNA", "RyhB", "ALAD1/HEMB1", "DNA polymerase", "tRNA-Leu", "rpoS gene"]
    p["entities"]["nucleic_acids"] = [{"name": n, **({"pathbank_nucleic_acid_id": 500 + i} if with_ids else {})}
                                      for i, n in enumerate(names)]
    return p


def _na_types(root) -> dict:
    return {n.findtext("name"): (n.findtext("nucleotide-type") or "") for n in root.findall(".//nucleic-acids/nucleic-acid")}


def test_nucleotide_type_set_only_when_unambiguous_flag_on(flag_on) -> None:
    builder, root = _build(_na_payload())
    types = _na_types(root)
    assert types["ent mRNA"] == "RNA" and types["tRNA-Leu"] == "RNA" and types["rpoS gene"] == "DNA"
    assert types["RyhB"] == "" and types["ALAD1/HEMB1"] == "" and types["DNA polymerase"] == ""
    assert builder.layout_debug_counts["nucleic_acids_type_unset"] == 3


def test_nucleotide_type_untouched_flag_off(flag_off) -> None:
    builder, root = _build(_na_payload())
    assert set(_na_types(root).values()) == {""}
    assert "nucleic_acids_type_unset" not in builder.layout_debug_counts


def test_nucleotide_type_skipped_for_existing_pathwhiz_rows_flag_on(flag_on) -> None:
    _, root = _build(_na_payload(with_ids=True))
    assert set(_na_types(root).values()) == {""}


def test_nucleotide_type_protein_names_stay_untyped() -> None:
    from t2pw.pwml.writer import nucleotide_type_for
    for name in ("RNA Pol II", "RNA chaperone Hfq", "DNA mismatch repair MutS", "DNA-PK", "RNA polymerase", "DNA kinase"):
        assert nucleotide_type_for({"name": name}) is None, name
    assert nucleotide_type_for({"name": "ent mRNA"}) == "RNA"
