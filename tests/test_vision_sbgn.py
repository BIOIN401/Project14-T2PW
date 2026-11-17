"""SBGN-ML exporter: structure, counts, determinism, XSD + libSBGN PD schematron validation (dev only)."""
from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest

from t2pw.vision.export_sbgn import sbgn_export, sbgn_text, write_sbgn

etree = pytest.importorskip("lxml.etree")
isoschematron = pytest.importorskip("lxml.isoschematron")

DATA = Path(__file__).parent / "data" / "vision" / "sbgn"
NS = {"s": "http://sbgn.org/libsbgn/0.3"}
SVRL = {"svrl": "http://purl.oclc.org/dsdl/svrl"}


ISO = "http://purl.oclc.org/dsdl/schematron"


def _schematron():
    """libSBGN sbgn_pd.sch, 'basic' phase, loaded by lxml after four in-memory adjustments (file on disk untouched):
    title moved before ns and digit-led pattern ids prefixed (lxml checks ISO RelaxNG), assert attributes lxml's
    schema does not know dropped (the assert id is kept as the message prefix), and pattern pd10133 removed
    (uses XPath 2.0 distinct-values; lxml is XSLT 1.0). pd10133 (unique arc ends) is re-checked in problems()."""
    doc = etree.parse(str(DATA / "sbgn_pd.sch"))
    root = doc.getroot()
    title = root.find("{%s}title" % ISO)
    root.remove(title)
    root.find("{%s}ns" % ISO).addprevious(title)
    for p in root.iter("{%s}pattern" % ISO):
        p.set("id", "p" + p.get("id"))
    for a in list(root.iter("{%s}active" % ISO)):
        if a.get("pattern") == "pd10133":
            a.getparent().remove(a)
        else:
            a.set("pattern", "p" + a.get("pattern"))
    for a in root.iter("{%s}assert" % ISO):
        a.text = (a.get("id") or "") + ": " + (a.text or "").strip()
        for k in ("id", "name", "see", "diagnostics", "role"):
            a.attrib.pop(k, None)
    return isoschematron.Schematron(doc, store_report=True)


XSD, SCH = etree.XMLSchema(etree.parse(str(DATA / "SBGN.xsd"))), _schematron()


def problems(text: str) -> list[str]:
    doc = etree.fromstring(text.encode("utf-8"))
    out = [str(e) for e in XSD.error_log] if not XSD.validate(doc) else []
    SCH.validate(doc)
    out += [f.text.strip()[:80] for f in SCH.validation_report.xpath("//svrl:failed-assert/svrl:text", namespaces=SVRL)]
    ends = Counter((a.get("class"), a.get("source"), a.get("target")) for a in doc.iterfind(".//s:arc", NS))
    return out + ["duplicate arc %s" % (k,) for k, n in ends.items() if n > 1]


def glyphs(text: str) -> Counter:
    return Counter(g.get("class") for g in etree.fromstring(text.encode()).iterfind(".//s:glyph", NS))


def arcs(text: str) -> Counter:
    return Counter(a.get("class") for a in etree.fromstring(text.encode()).iterfind(".//s:arc", NS))


def payload(**proc) -> dict:
    return {
        "entities": {
            "compounds": [{"name": "A", "mapped_ids": {"chebi": "15378", "kegg": ""}}, {"name": "B"}, {"name": "C"}],
            "proteins": [{"name": "E1", "mapped_ids": {"uniprot": "P12345"}}, {"name": "E2"}, {"name": "T1"}],
            "protein_complexes": [{"name": "Cx", "components": [{"name": "E1", "mapped_ids": {"uniprot": "P12345"}}, "E2"]}],
            "nucleic_acids": [{"name": "mRNA1"}],
        },
        "biological_states": [
            {"name": "cyt state", "subcellular_location": "cytoplasm"},
            {"name": "ext state", "subcellular_location": "extracellular"},
        ],
        "processes": proc,
    }


RX = {"name": "A to B", "inputs": ["A"], "outputs": ["B", "C"], "biological_state": "cyt state",
      "enzymes": [{"entity": "E1", "entity_type": "protein", "role": "catalyst"}],
      "modifiers": [{"entity": "E1", "entity_type": "protein", "role": "catalyst"},
                    {"entity": "mRNA1", "entity_type": "nucleic_acid", "role": "inhibitor"}]}
TR = {"name": "B export", "cargo": "B", "from_biological_state": "cyt state", "to_biological_state": "ext state",
      "transporters": [{"entity": "T1", "entity_type": "protein"}]}


def test_reaction_with_enzyme():
    text, rep = sbgn_export(payload(reactions=[RX]))
    assert problems(text) == []
    assert glyphs(text) == Counter({"compartment": 1, "simple chemical": 3, "macromolecule": 1, "nucleic acid feature": 1, "process": 1})
    # 1 consumption, 2 production, catalysis (enzyme + modifier merged), inhibition from the nucleic acid
    assert arcs(text) == Counter({"consumption": 1, "production": 2, "catalysis": 1, "inhibition": 1})
    assert rep["exported"]["reactions"] == 1 and rep["exported"]["process_glyphs"] == 1 and rep["omitted"] == {}
    assert "chebi=" in text and "kegg" not in text  # empty ids are not written


def test_transport_across_two_compartments():
    text, rep = sbgn_export(payload(transports=[TR]))
    assert problems(text) == []
    g = glyphs(text)
    assert g["compartment"] == 2 and g["simple chemical"] == 2 and g["process"] == 1 and g["macromolecule"] == 1
    assert arcs(text) == Counter({"consumption": 1, "production": 1, "catalysis": 1})
    assert rep["exported"]["transports"] == 1


def test_interactions_resolved_and_counted():
    ix = [
        {"entity_1": "A", "entity_2": "E1", "relationship": "Activation", "biological_state": "cyt state"},
        {"entity_1": "B", "entity_2": "A to B", "relationship": "inhibits"},
        {"entity_1": "C", "entity_2": "E1", "relationship": "binds"},
        {"entity_1": "A", "entity_2": "nothing", "relationship": "Inhibition"},  # unresolved target
        {"entity_1": "A to B", "entity_2": "E1", "relationship": "activates"},  # source is a process
        {"entity_1": "A", "entity_2": "E2", "relationship": "activates"},  # E2 catalyses nothing
    ]
    text, rep = sbgn_export(payload(reactions=[RX], interactions=ix))
    assert problems(text) == []
    a = arcs(text)
    assert a["stimulation"] == 1 and a["modulation"] == 1 and a["inhibition"] == 2  # +1 inhibition from the reaction modifier
    assert rep["exported"]["interactions"] == 3
    assert rep["omitted"]["interactions"] == {"unresolved_endpoint": 1, "source_is_process": 1, "target_entity_catalyses_no_process": 1}


def test_interaction_reaches_process_through_complex_membership():
    rx = {"name": "r", "inputs": ["A"], "outputs": ["B"], "biological_state": "cyt state",
          "enzymes": [{"entity": "Cx", "entity_type": "protein_complex"}]}
    ix = [{"entity_1": "C", "entity_2": "E2", "relationship": "downregulates"}]  # E2 is a member of Cx, Cx catalyses r
    text, rep = sbgn_export(payload(reactions=[rx], interactions=ix))
    assert problems(text) == []
    assert arcs(text)["inhibition"] == 1 and rep["exported"]["interactions"] == 1 and rep["omitted"] == {}


def test_complex_members_nested():
    text = sbgn_text(payload(reactions=[{"name": "r", "inputs": ["A"], "outputs": ["B"], "biological_state": "cyt state",
                                         "enzymes": [{"entity": "Cx", "entity_type": "protein_complex"}]}]))
    assert problems(text) == []
    doc = etree.fromstring(text.encode())
    (cx,) = doc.xpath("//s:glyph[@class='complex']", namespaces=NS)
    assert [g.find("s:label", NS).get("text") for g in cx.iterfind("s:glyph", NS)] == ["E1", "E2"]


def test_unresolved_members_and_no_compartments():
    p = payload(reactions=[{"name": "r", "inputs": ["A", "ghost"], "outputs": ["B"], "enzymes": []}])
    text, rep = sbgn_export(p)
    assert problems(text) == []  # no biological state -> one "unspecified" compartment, every glyph references it
    assert rep["unspecified_compartment"] and rep["omitted"] == {"reaction_arcs": {"unresolved_entity": 1}}
    assert glyphs(text)["compartment"] == 1


def test_deterministic_and_file(tmp_path):
    p = payload(reactions=[RX], transports=[TR], interactions=[{"entity_1": "A", "entity_2": "E1", "relationship": "Activation"}])
    assert sbgn_text(p).encode() == sbgn_text(p).encode()
    f = tmp_path / "x.sbgn"
    write_sbgn(p, f)
    assert f.read_bytes() == sbgn_text(p).encode("utf-8") and problems(f.read_text(encoding="utf-8")) == []


def test_ir_coordinates_used():
    ir = {
        "entities": {"compounds": [{"key": "cmp_1", "name": "A"}, {"key": "cmp_2", "name": "B"}, {"key": "cmp_3", "name": "C"}],
                     "proteins": [{"key": "prot_1", "name": "E1"}]},
        "biological_states": [{"key": "bs_1", "name": "cyt state"}],
        "locations": [{"entity_type": "compound", "entity_key": "cmp_1", "biological_state_key": "bs_1", "x": 500, "y": 600},
                      {"entity_type": "compound", "entity_key": "cmp_2", "biological_state_key": "bs_1", "x": 900, "y": 600}],
        "bound_visualizations": [{"biological_state_key": "bs_1", "x": 0, "y": 400, "width": 3200, "height": 700}],
        "processes": {"reactions": [{"key": "rxn_1", "name": "A to B", "biological_state_key": "bs_1"}]},
        "process_visualizations": [{"type": "reaction_visualization", "process_key": "rxn_1", "x": 700, "y": 610}],
    }
    text, rep = sbgn_export(payload(reactions=[RX]), ir=ir)
    assert problems(text) == []
    doc = etree.fromstring(text.encode())
    pos = {g.find("s:label", NS).get("text"): g.find("s:bbox", NS) for g in doc.xpath("//s:glyph[s:label]", namespaces=NS)}
    assert (pos["A"].get("x"), pos["A"].get("y")) == ("500", "600") and (pos["B"].get("x"), pos["B"].get("y")) == ("900", "600")
    (proc,) = doc.xpath("//s:glyph[@class='process']/s:bbox", namespaces=NS)
    assert (proc.get("x"), proc.get("y")) == ("700", "610")
    assert (pos["cytoplasm"].get("y"), pos["cytoplasm"].get("w"), pos["cytoplasm"].get("h")) == ("400", "3200", "700")
    assert rep["layout"] == {"ir_positioned": 3, "grid_positioned": 3}  # C, E1, mRNA1 fall back to the grid
    assert sbgn_text(payload(reactions=[RX]), ir=ir) == text


def test_validator_has_teeth():
    text = sbgn_text(payload(reactions=[RX]))
    broken = text.replace(' compartmentRef="c1"', "", 1)
    assert any(p.startswith("pd10112") for p in problems(broken))
    assert problems(text.replace('class="process"', 'class="processs"'))  # XSD enum
