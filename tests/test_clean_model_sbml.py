"""SBML exporter: synthetic model, stdlib XML parsing only (libsbml validation is evidence, not a test dependency)."""

import re
import xml.etree.ElementTree as ET

from t2pw.clean_model import Field, build_from_payload
from t2pw.clean_model.export import sbml as X
from t2pw.clean_model.export.sbo import SBO
from t2pw.clean_model.model import Prov, frozen

SB = "{%s}" % X.SBML_NS
TW = "{%s}" % X.T2PW_NS
NS = {"s": X.SBML_NS, "t": X.T2PW_NS}
SID = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def payload():
    return {
        "biological_states": [
            {"name": "cyto", "species": "Escherichia coli", "subcellular_location": "cytoplasm"},
            {"name": "peri", "species": "Escherichia coli", "subcellular_location": "periplasm"}],
        "entities": {
            "species": [{"name": "Escherichia coli", "taxonomy_id": "562"}],
            "subcellular_locations": [{"name": "cytoplasm"}, {"name": "periplasm"}],
            "compounds": [{"name": "A", "chebi_id": "CHEBI:15377", "provenance": "extracted"}, {"name": "B"}],
            "proteins": [{"name": "EntX", "uniprot": "P0A6A3", "gene_name": "entX", "provenance": "extracted"},
                         {"name": "EntY", "provenance": "extracted"}, {"name": "Unknown", "pathbank_protein_id": 9659}],
            "protein_complexes": [{"name": "Mystery complex", "components": ["Unknown", "Unknown"]},
                                  {"name": "XY complex", "components": [{"name": "EntX"}, {"name": "EntY"}]}],
        },
        "element_locations": {"compound_locations": [{"compound": "A", "biological_state": "cyto"}]},
        "processes": {
            "reactions": [
                {"name": "A to B", "inputs": ["A"], "outputs": ["B"], "biological_state": "cyto",
                 "evidence": "EntX converts A to B in the cytoplasm",
                 "enzymes": [{"entity": "EntX", "entity_type": "protein", "role": "catalyst"}]},
                {"name": "B to A", "inputs": ["B"], "outputs": ["A"], "biological_state": "cyto", "evidence": "B converts back"}],
            "transports": [
                {"name": "A export", "cargo": "A", "from_biological_state": "cyto", "evidence": "A is exported to somewhere",
                 "transporters": [{"entity": "EntY", "entity_type": "protein"}]}],
            "interactions": [
                {"entity_1": "A", "entity_2": "EntX", "relationship": "inhibits", "evidence": "A inhibits EntX"},
                {"entity_1": "EntX", "entity_2": "EntY", "relationship": "regulates", "evidence": "EntX regulates EntY"}],
        },
    }


def model():
    m = build_from_payload(payload(), {"pmcid": "PMC1", "pmid": "123", "title": "A test pathway"})
    rx = m.reactions[1]  # paper-stated stoichiometry and direction
    rx.reactants[0].stoichiometry = frozen(2, "paper")
    rx.reversible = frozen(True, "paper")
    m.compounds[1].kegg_id = frozen("C00001", "db", source="payload")
    m.locations[0].go_id = Field("GO:0005737", Prov("db", source="GO", accession="GO:0005737", match_type="exact_label", retrieved="2026-11-10"))
    m.db_layer = {
        "items": [{"kind": "compound", "entity": "B", "field": "chebi_id", "value": "CHEBI:16",
                   "prov": {"label": "db", "source": "ChEBI", "accession": "CHEBI:16", "match_type": "exact_label", "retrieved": "2026-11-10"}}],
        "suggestions": [{"kind": "protein", "entity": "EntY", "field": "uniprot", "candidates": [{"id": "P12345"}],
                         "prov": {"label": "inferred", "source": "UniProt", "match_type": "gene_symbol_taxon"}}],
        "conflicts": [], "refused": []}
    m.pathway.licences.append(Field("ChEBI: CC BY 4.0", Prov("db", source="chebi")))
    return m


def parsed():
    text = X.sbml_text(model())
    return text, ET.fromstring(text.encode("utf-8"))


def species(root, name):
    return [s for s in root.iter(SB + "species") if s.get("name") == name][0]


def reactions(root):
    return {r.get("name"): r for r in root.iter(SB + "reaction")}


def test_wellformed_l3v2_and_deterministic():
    text, root = parsed()
    assert root.tag == SB + "sbml" and root.get("level") == "3" and root.get("version") == "2"
    assert text == X.sbml_text(model())


def test_no_initial_amount_no_unknown_species():
    text, root = parsed()
    assert "initialAmount" not in text and "initialConcentration" not in text
    for sp in root.iter(SB + "species"):
        assert sp.get("name").casefold() != "unknown"
        assert sp.get("hasOnlySubstanceUnits") == "false" and sp.get("boundaryCondition") == "false" and sp.get("constant") == "false"
    assert any(o.get("name") == "Unknown" for o in root.iter(TW + "omission"))


def test_all_unknown_complex_is_a_complex_without_members_or_accession():
    text, root = parsed()
    sp = species(root, "Mystery complex")
    assert sp.get("sboTerm") == "SBO:0000253"
    flat = "".join(sp.itertext()).lower()
    assert "unknown" in flat
    assert "identifiers.org" not in ET.tostring(sp, encoding="unicode")
    xy = ET.tostring(species(root, "XY complex"), encoding="unicode")
    assert "EntX" in xy and "identifiers.org/uniprot" not in xy


def test_stoichiometry_only_when_paper_stated():
    _, root = parsed()
    rxs = reactions(root)
    a = rxs["A to B"].find("s:listOfReactants/s:speciesReference", NS)
    assert a.get("stoichiometry") is None and a.find("s:annotation/t:stoichiometry", NS).get("state") == "unstated"
    b = rxs["B to A"].find("s:listOfReactants/s:speciesReference", NS)
    assert b.get("stoichiometry") == "2.0"
    assert all(e.get("constant") == "true" for e in root.iter(SB + "speciesReference"))


def test_reversible_placeholder_and_stated():
    _, root = parsed()
    rxs = reactions(root)
    r = rxs["A to B"]
    assert r.get("reversible") == "false" and r.find("s:annotation/t:info/t:directionStated", NS).text == "false"
    assert r.find("s:annotation/t:info/t:spontaneous", NS).text == "unstated"
    s = rxs["B to A"]
    assert s.get("reversible") == "true" and s.find("s:annotation/t:info/t:directionStated", NS).text == "true"
    assert r.find("s:listOfModifiers/s:modifierSpeciesReference", NS).get("sboTerm") == "SBO:0000013"


def test_quotes_in_notes_and_transport_with_missing_side():
    _, root = parsed()
    rxs = reactions(root)
    assert "EntX converts A to B in the cytoplasm" in "".join(rxs["A to B"].find("s:notes", NS).itertext())
    t = rxs["A export"]
    notes = "".join(t.find("s:notes", NS).itertext())
    assert t.get("sboTerm") == "SBO:0000655" and "to-location" in notes and "A is exported to somewhere" in notes
    assert t.find("s:listOfModifiers", NS) is not None
    assert X.UNSPEC_ID in {c.get("id") for c in root.iter(SB + "compartment")}


def test_interactions_unsigned_never_activation_or_inhibition():
    _, root = parsed()
    by = {i.get("relationship"): i for i in root.iter(TW + "interaction")}
    assert len(by) == 2
    assert by["inhibits"].get("sign") == "negative" and by["inhibits"].get("sbo") == "SBO:0000169"
    r = by["regulates"]
    assert r.get("sign") == "unsigned" and r.get("sbo") not in ("SBO:0000169", "SBO:0000170")
    assert "EntX regulates EntY" in "".join(r.itertext())
    assert "inhibits" not in ET.tostring(r, encoding="unicode").replace('relationship="regulates"', "")


def test_ids_carry_provenance_and_suggestions_never_bqbiol_is():
    text, root = parsed()
    chebi = [i for i in root.iter(TW + "id") if i.get("id") == "CHEBI:16"][0]
    assert chebi.get("label") == "db" and chebi.get("source") == "ChEBI"
    assert chebi.get("matchType") == "exact_label" and chebi.get("retrieved") == "2026-11-10"
    assert "https://identifiers.org/chebi/CHEBI:16" in text and "identifiers.org/chebi/CHEBI:15377" in text
    assert "P12345" not in "".join(e.get("{http://www.w3.org/1999/02/22-rdf-syntax-ns#}resource", "") for e in root.iter())  # suggestion stays out of rdf
    assert "identifiers.org/uniprot/P0A6A3" in text and "identifiers.org/uniprot/P12345" not in text
    assert list(root.iter(TW + "suggestion"))
    assert "identifiers.org/go/GO:0005737" in text and "kegg.compound/C00001" in text
    assert "ChEBI: CC BY 4.0" in text


def test_sids_unique_and_sbo_from_verified_table():
    text, root = parsed()
    ids = []
    for el in root.iter():
        if el.tag.startswith(SB):
            for a in ("id", "species", "compartment"):
                if el.get(a) is not None:
                    assert SID.match(el.get(a)), el.get(a)
            if el.get("id"):
                ids.append(el.get("id"))
        if el.get("sboTerm"):
            assert el.get("sboTerm") in SBO
    assert len(ids) == len(set(ids))
    assert root.find("s:model", NS).get("name") == "A test pathway"


RDF = "{http://www.w3.org/1999/02/22-rdf-syntax-ns#}"
BQ = "{http://biomodels.net/biology-qualifiers/}"


def _qualified(sp):
    """{qualifier local name: [resource, ...]} of one element's rdf annotation."""
    out = {}
    for q in sp.iter():
        if q.tag.startswith(BQ):
            out[q.tag[len(BQ):]] = [li.get(RDF + "resource") for li in q.iter(RDF + "li")]
    return out


def _doc(p, meta=None):
    return ET.fromstring(X.sbml_text(build_from_payload(p, meta or {})).encode("utf-8"))


def test_unknown_placeholder_never_a_species_in_any_reference_type():
    p = payload()
    p["entities"]["proteins"].append({"name": "Ghost", "pathbank_protein_id": 9659})  # flagged under another name
    p["processes"]["reactions"] = [{"name": "u", "inputs": ["Unknown", "A"], "outputs": ["B", "Ghost"],
                                    "enzymes": [{"entity": "Unknown"}, {"entity": "Ghost"}], "modifiers": [{"entity": "Unknown"}]}]
    p["processes"]["transports"] = [{"name": "t1", "cargo": "Unknown"},
                                    {"name": "t2", "cargo": "A", "transporters": [{"entity": "Unknown"}, {"entity": "EntY"}]}]
    root = _doc(p)
    names = {s.get("name").casefold() for s in root.iter(SB + "species")}
    assert "unknown" not in names and "ghost" not in names
    rxs = reactions(root)
    assert len(rxs["u"].findall("s:listOfReactants/s:speciesReference", NS)) == 1
    assert rxs["u"].find("s:listOfModifiers", NS) is None and "t1" not in rxs
    assert len(rxs["t2"].findall("s:listOfModifiers/s:modifierSpeciesReference", NS)) == 1
    om = {o.get("name"): o for o in root.iter(TW + "omission")}
    assert "Unknown" in om and "Ghost" in om and om["Unknown"].get("kind") in ("reference", "protein", "transport")
    assert any(o.get("kind") == "reference" and o.get("name") == "Unknown" for o in root.iter(TW + "omission"))


def test_plain_modifier_is_sbo_modifier_catalyst_stays_catalyst():
    p = payload()
    p["processes"]["reactions"] = [{"name": "m", "inputs": ["A"], "outputs": ["B"], "enzymes": [{"entity": "EntX"}],
                                    "modifiers": [{"entity": "EntY"}]}]
    mods = {m.get("species"): m.get("sboTerm") for m in reactions(_doc(p))["m"].iter(SB + "modifierSpeciesReference")}
    assert sorted(mods.values()) == ["SBO:0000013", "SBO:0000019"] and SBO["SBO:0000019"] == "modifier"


def test_ensembl_protein_is_gene_and_transcript_ids_are_encoded_by():
    m = model()
    prov = {"label": "db", "source": "UniProt", "accession": "P0A6A3", "match_type": "cross_reference", "retrieved": "2026-11-10"}
    m.db_layer["items"] += [{"kind": "protein", "entity": "EntX", "field": f, "value": v, "prov": prov} for f, v in (
        ("ensembl", "ENSP00000269305"), ("ensembl_gene", "ENST00000269305"), ("ensembl_gene", "ENSG00000141510"),
        ("kegg_gene", "hsa:7157"))]
    root = ET.fromstring(X.sbml_text(m).encode("utf-8"))
    q = _qualified(species(root, "EntX"))
    assert "https://identifiers.org/ensembl/ENSP00000269305" in q["is"]
    assert set(q["isEncodedBy"]) == {"https://identifiers.org/ensembl/ENST00000269305", "https://identifiers.org/ensembl/ENSG00000141510",
                                     "https://identifiers.org/kegg.genes/hsa:7157"}
    assert not any("ENST" in u or "kegg.genes" in u for u in q["is"])


def test_drugbank_from_mapped_ids_written_as_db_labelled_annotation():
    p = payload()
    p["entities"]["compounds"].append({"name": "DrugX", "mapped_ids": {"drugbank": "DB03246"}, "class": "cofactor"})
    root = _doc(p)
    sp = species(root, "DrugX")
    assert "https://identifiers.org/drugbank/DB03246" in _qualified(sp)["is"]
    i = [x for x in sp.iter(TW + "id") if x.get("db") == "drugbank"][0]
    assert i.get("id") == "DB03246" and i.get("label") == "db"


def test_guessed_pmid_not_written_as_identifier_but_paper_pmid_is():
    for label, expected in (("inferred", False), ("paper", True)):
        m = model()
        m.pathway.pmid = frozen("123", label)
        text = X.sbml_text(m)
        assert ("identifiers.org/pubmed/123" in text) is expected
        assert 'key="pmid"' in text
