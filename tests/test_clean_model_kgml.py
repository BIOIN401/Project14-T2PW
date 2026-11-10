"""KGML exporter: synthetic model + real runs when present. Pure stdlib structural checker (no KEGG call is ever made)."""

import json
import os
import xml.etree.ElementTree as ET

import pytest

from t2pw.clean_model import Field, build_from_payload
from t2pw.clean_model.export import kgml as K
from t2pw.clean_model.model import Prov, frozen

REQ = {"pathway": {"name", "org", "number"}, "entry": {"id", "name", "type"}, "reaction": {"id", "name", "type"},
       "relation": {"entry1", "entry2", "type"}, "subtype": {"name", "value"}, "substrate": {"id", "name"},
       "product": {"id", "name"}, "component": {"id"}}
ETYPES = {"ortholog", "enzyme", "reaction", "gene", "group", "compound", "map", "brite", "other"}
RTYPES = {"ECrel", "PPrel", "GErel", "PCrel", "maplink"}


def check(text):
    """Required attributes per element, unique ids, references resolve, DTD child order. Returns the root."""
    root = ET.fromstring(text.encode("utf-8"))
    assert root.tag == "pathway"
    for el in root.iter():
        assert REQ.get(el.tag, set()) <= set(el.attrib), (el.tag, el.attrib)
    entries = root.findall("entry")
    ids = [e.get("id") for e in entries]
    assert len(ids) == len(set(ids)) and all(i.isdigit() for i in ids)
    assert all(e.get("type") in ETYPES for e in entries)
    allids = ids + [r.get("id") for r in root.findall("reaction")]
    assert len(allids) == len(set(allids))
    for c in root.iter("component"):
        assert c.get("id") in ids
    for r in root.findall("relation"):
        assert r.get("entry1") in ids and r.get("entry2") in ids and r.get("type") in RTYPES
    for r in root.findall("reaction"):
        assert r.findall("substrate") and r.findall("product") and r.get("type") in ("reversible", "irreversible")
        for s in r.findall("substrate") + r.findall("product"):
            assert s.get("id") in ids
    tags = [c.tag for c in root]
    assert tags == sorted(tags, key=["entry", "relation", "reaction"].index)
    return root


def payload():
    return {
        "biological_states": [{"name": "cyto", "species": "E. coli", "subcellular_location": "cytoplasm"}],
        "entities": {
            "species": [{"name": "E. coli", "taxonomy_id": "562"}],
            "subcellular_locations": [{"name": "cytoplasm"}, {"name": "periplasm"}],
            "compounds": [{"name": "A", "chebi_id": "CHEBI:15377", "provenance": "extracted"}, {"name": "B"}, {"name": "C"},
                          {"name": "D", "chebi_id": "CHEBI:17", "kegg_id": "C00002"}],
            "proteins": [{"name": "EntX", "uniprot": "P0A6A3", "provenance": "extracted"}, {"name": "EntY", "provenance": "extracted"},
                         {"name": "Unknown", "pathbank_protein_id": 9659}],
            "protein_complexes": [{"name": "Mystery complex", "components": ["Unknown", "Unknown"]},
                                  {"name": "XY complex", "components": [{"name": "EntX"}, {"name": "EntY"}]}],
            "nucleic_acids": [{"name": "geneG"}],
        },
        "processes": {
            "reactions": [
                {"name": "A to B", "inputs": ["A"], "outputs": ["B"], "evidence": "q", "enzymes": [{"entity": "EntX", "role": "catalyst"}]},
                {"name": "B to D", "inputs": ["B"], "outputs": ["D", "Ghost"]},
                {"name": "ghost rxn", "inputs": ["Ghost"], "outputs": ["B"]}],
            "transports": [{"name": "A export", "cargo": "A", "from_biological_state": "cyto", "evidence": "q"}],
            "interactions": [
                {"entity_1": "A", "entity_2": "EntX", "relationship": "inhibits"},
                {"entity_1": "EntX", "entity_2": "EntY", "relationship": "regulates"},
                {"entity_1": "EntX", "entity_2": "EntY", "relationship": "phosphorylates"},
                {"entity_1": "EntX", "entity_2": "geneG", "relationship": "regulates expression"},
                {"entity_1": "EntX", "entity_2": "Nobody", "relationship": "binds"}],
        },
    }


def model():
    m = build_from_payload(payload(), {"pmcid": "PMC1", "title": "A test pathway"})
    m.reactions[0].reversible = frozen(True, "paper")
    m.reactions[1].kegg_reaction_id = frozen("R00001", "db")
    m.db_layer = {"items": [{"kind": "compound", "entity": "B", "field": "chebi_id", "value": "CHEBI:16",
                             "prov": {"label": "db", "source": "ChEBI", "accession": "CHEBI:16"}},
                            {"kind": "reaction", "entity": "A to B", "field": "rhea_id", "value": "RHEA:18985",
                             "prov": {"label": "db", "source": "Rhea", "accession": "RHEA:18985"}}],
                  "suggestions": [], "conflicts": [], "refused": []}
    m.pathway.licences.append(Field("ChEBI: CC BY 4.0", Prov("db", source="chebi")))
    return m


def entry(root, label):
    return [e for e in root.findall("entry") if e.find("graphics").get("name") == label][0]


def test_placeholders_and_structure():
    text, om = K.kgml_text(model())
    root = check(text)
    assert (root.get("name"), root.get("org"), root.get("number")) == ("undefined", "undefined", "00000")
    assert root.get("title") == "A test pathway" and "not a KEGG map" in text and om["sources_used"] == ["ChEBI: CC BY 4.0"]


def test_ids_only_when_in_model():
    text, om = K.kgml_text(model())
    root = check(text)
    assert entry(root, "D").get("name") == "cpd:C00002"          # KEGG id held, wins over chebi
    assert entry(root, "A").get("name") == "chebi:15377"         # no KEGG id: the non-KEGG id the model holds
    assert entry(root, "B").get("name") == "chebi:16"            # carried by the db layer
    assert entry(root, "C").get("name") == "undefined"           # nothing held: nothing invented
    assert entry(root, "EntX").get("name") == "uniprot:P0A6A3" and entry(root, "EntY").get("name") == "undefined"
    assert entry(root, "geneG").get("name") == "undefined" and entry(root, "geneG").get("type") == "gene"
    assert "C0000" not in text.replace("C00002", "")
    assert {u["name"] for u in om["unmapped_entities"]} == {"C", "EntY", "geneG", "Mystery complex", "XY complex"}
    assert om["non_kegg_prefixes_used"] == ["chebi", "rhea", "uniprot"] and "Non-KEGG ids used" in text


def test_strict_mode_only_kegg_ids():
    text, om = K.kgml_text(model(), strict_kegg=True)
    root = check(text)
    names = {e.find("graphics").get("name"): e.get("name") for e in root.findall("entry")}
    assert names["D"] == "cpd:C00002" and names["A"] == names["B"] == names["EntX"] == "undefined"
    assert om["non_kegg_prefixes_used"] == [] and "chebi:" not in text and "Strict mode" in text


def test_groups_and_components():
    root = check(K.kgml_text(model())[0])
    mystery, xy = entry(root, "Mystery complex"), entry(root, "XY complex")
    assert mystery.get("type") == xy.get("type") == "group" and mystery.get("name") == xy.get("name") == "undefined"
    assert mystery.findall("component") == []
    assert [c.get("id") for c in xy.findall("component")] == [entry(root, "EntX").get("id"), entry(root, "EntY").get("id")]
    assert "Unknown" not in {e.find("graphics").get("name") for e in root.findall("entry")}


def test_relations_signed_unsigned_and_skipped():
    text, om = K.kgml_text(model())
    root = check(text)
    rels = root.findall("relation")
    sub = lambda r: [(s.get("name"), s.get("value")) for s in r.findall("subtype")]
    by = {(r.get("type"), tuple(sub(r))) for r in rels}
    assert ("PCrel", (("inhibition", "--|"),)) in by
    assert ("PPrel", (("phosphorylation", "+p"),)) in by
    unsigned = [r for r in rels if r.get("type") == "PPrel" and not r.findall("subtype")]
    assert len(unsigned) == 1 and "unsigned" in text          # regulates: no subtype, never activation or inhibition
    assert ("GErel", ()) in by                                 # expression_regulation is unsigned here: no subtype
    assert om["skipped_interactions"][0]["target"] == "Nobody" and len(rels) == 4
    assert om["stats"]["relations_signed"] == 1 and om["stats"]["relations_unsigned"] == 3


def test_reactions_names_direction_transport_and_skips():
    text, om = K.kgml_text(model())
    root = check(text)
    rxs = root.findall("reaction")
    assert [r.get("type") for r in rxs] == ["reversible", "irreversible", "irreversible"]
    assert rxs[0].get("name") == "rhea:18985" and rxs[1].get("name") == "rn:R00001"
    t = rxs[2]
    assert t.get("name") == "undefined" and t.find("substrate").get("id") == t.find("product").get("id")
    assert om["direction_placeholder"] == {"directionStated": False, "count": 2, "names": ["B to D", "A export"]}
    assert "directionStated=false" in text
    assert om["no_slot_in_format"]["transport_locations"] == [{"name": "A export", "from": "cytoplasm", "to": None}]
    assert om["no_slot_in_format"]["compartments"] == 2 and om["no_slot_in_format"]["quotes"] == 2
    assert om["no_slot_in_format"]["catalysts_and_modifiers"] == 1
    reasons = {s["name"]: s["reason"] for s in om["skipped_reactions"]}
    assert "ghost rxn" in reasons and "Ghost" in reasons["B to D"]        # Ghost is undeclared: dropped, reported
    assert len(rxs) == 3 and len(root.findall("reaction/substrate")) == 3


def test_deterministic_bytes_and_sidecar(tmp_path):
    assert K.kgml_text(model()) == K.kgml_text(model())
    p = str(tmp_path / "x.kgml")
    om = K.write_kgml(model(), p)
    assert open(p, "rb").read() == K.kgml_text(model())[0].encode("utf-8")
    assert json.load(open(p + ".omissions.json", encoding="utf-8")) == json.loads(json.dumps(om))


def test_no_kegg_request_anywhere():
    src = open(K.__file__, encoding="utf-8").read()
    assert "kegg.jp" not in src and "genome.jp" not in src
    for bad in ("urllib", "socket", "requests", "http.client", "import http"):
        assert bad not in src
    assert "kegg.jp" not in K.kgml_text(model())[0]


def _runs():
    rl = "C:/t/ev-cm/runlist.json"
    if not os.path.exists(rl):
        return []
    return [r["dir"] for r in json.load(open(rl, encoding="utf-8")) if r.get("q") is True and os.path.exists(os.path.join(r["dir"], "final_mapped.json"))][:3]


@pytest.mark.skipif(not _runs(), reason="no runlist runs available")
def test_real_runs_pass_structural_check():
    for d in _runs():
        text, om = K.kgml_for_run(d)
        check(text)
        assert text == K.kgml_for_run(d)[0] and om["stats"]["entries"] > 0
