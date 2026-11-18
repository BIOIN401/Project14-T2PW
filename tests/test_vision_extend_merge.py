"""V4 database extension: fake fetchers, no network, no database."""

import copy
import hashlib
import json
from pathlib import Path

from t2pw.vision import extend
from t2pw.vision.extend import extend_big_payload, extend_enabled, run_if_enabled
from t2pw.vision.extend.reactome import reaction_step
from t2pw.vision.extend.wikipathways import parse_gpml, to_steps
from t2pw.vision.payload import build_big_payload, keyed_rows
from t2pw.vision.view import DEFAULT, FULL, unresolved_refs, view

SID = "pathbank:PW000001"
FIX = Path(__file__).parent / "data" / "vision" / "extend"


def c(name, kegg, **kw):
    return {"name": name, "mapped_ids": {"kegg": kegg}, "evidence": f"{name} is made", **kw}


def base():
    return {
        "entities": {
            "species": [{"name": "Escherichia coli", "taxonomy_id": "562", "evidence": "E. coli"}],
            "subcellular_locations": [{"name": "cytoplasm", "evidence": "cytoplasm"}],
            "compounds": [c("A", "C90001"), c("B", "C90002"), c("C", "C90003"),
                          c("water", "C00001")],
            "proteins": [{"name": "E1", "mapped_ids": {"uniprot": "P11111"}, "species": "Escherichia coli",
                          "evidence": "E1"}],
            "protein_complexes": [{"name": "E1 complex", "species": "Escherichia coli",
                                   "components": [{"name": "E1", "mapped_ids": {"uniprot": "P11111"}}]}],
        },
        "biological_states": [{"name": "cell state", "species": "Escherichia coli",
                               "subcellular_location": "cytoplasm", "evidence": "in the cell"}],
        "processes": {"reactions": [
            {"name": "A to B", "inputs": ["A"], "outputs": ["B"], "biological_state": "cell state",
             "enzymes": [{"entity": "E1 complex", "entity_type": "protein_complex", "role": "catalyst"}],
             "evidence": "A is converted to B"},
            {"name": "B to C", "inputs": ["B", "water"], "outputs": ["C"], "biological_state": "cell state",
             "evidence": "B is hydrated to C"}]},
    }


def part(name, kegg=None, t="compound", **ids):
    if kegg:
        ids["kegg"] = kegg
    return {"type": t, "name": name, "synonyms": [], "ids": ids}


def rxn(acc, ins, outs, enzymes=(), location="cytoplasm"):
    return {"kind": "reaction", "accession": acc, "inputs": list(ins), "outputs": list(outs),
            "enzymes": list(enzymes), "location": location, "direction": "=>"}


def pathway(steps, sid=SID, db="pathbank"):
    return {"source_id": sid, "db": db, "accession": sid.split(":")[1], "name": "Test pathway",
            "url": "https://example.org", "updated_at": "2024-01-01", "content_sha256": "x",
            "species": "Escherichia coli", "taxonomy_id": "562", "steps": steps, "notes": []}


def five_steps():
    enz = {"type": "complex", "name": "Alpha synthase", "synonyms": [],
           "components": [part("alpha", t="protein", uniprot="P11111")]}
    A, B, C = part("Alpha-thing", "C90001"), part("Beta", "C90002"), part("Gamma", "C90003")
    D, E, G = part("Delta", "C90004"), part("Epsilon", "C90005"), part("Gee", "C90007")
    water = part("Water", "C00001")
    return [rxn("R1", [A], [B], [enz]),            # = paper step 1 (ids, member-joined enzyme)
            rxn("R2", [B, water], [C]),            # = paper step 2 (Water folds to H2O)
            rxn("R3", [C], [D]),                   # hop 1
            rxn("R4", [D], [E]),                   # hop 2
            rxn("R5", [A], [G])]                   # hop 1


def fake(pathways):
    def f(ctx, **kw):
        f.calls += 1
        return copy.deepcopy(pathways), ["fake"]
    f.calls = 0
    return f


def run(steps, **kw):
    big0 = build_big_payload(base())
    return big0, *extend_big_payload(big0, sources=("pathbank",), fetchers={"pathbank": fake([pathway(steps)])}, **kw)


def keyed(big):
    return {k: (s, r, b) for s, k, r, b in keyed_rows(big)}


def test_shared_steps_merge_and_new_steps_get_chain_hops():
    big0, big, report = run(five_steps())
    assert big["base"] == big0["base"]
    rows = keyed(big)
    base_rx = [k for k, (s, r, b) in rows.items() if s == "processes.reactions" and b]
    for k in base_rx:   # both paper steps carry the database record, once
        assert [r["source"] for r in big["provenance"][k] if r["kind"] == "database"] == [SID]
    ext = big["extension"]
    # participant names are the paper's where the ids matched (Alpha-thing is A, Gamma is C)
    assert [r["name"] for r in ext["processes"]["reactions"]] == ["C -> Delta", "Delta -> Epsilon", "A -> Gee"]
    assert {r["name"] for r in ext["entities"]["compounds"]} == {"Delta", "Epsilon", "Gee"}  # A, B, C, water merged
    assert "protein_complexes" not in ext["entities"]     # same members as "E1 complex": one enzyme
    hops = {rows[k][1]["name"]: big["hops"][k] for k in rows if rows[k][0] == "processes.reactions" and not rows[k][2]}
    assert hops == {"C -> Delta": 1, "Delta -> Epsilon": 2, "A -> Gee": 1}
    assert report["placement"][SID] == {"merged_into_base": 2, "hop_1": 2, "hop_2_plus": 1}
    assert report["sources"]["pathbank"]["counts"]["currency_renamed"] == 1


def test_default_view_is_base_plus_hop1_and_reference_closed():
    _, big, _ = run(five_steps())
    d, f = view(big, DEFAULT), view(big, FULL)
    assert [r["name"] for r in d["processes"]["reactions"]] == ["A to B", "B to C", "C -> Delta", "A -> Gee"]
    assert len(f["processes"]["reactions"]) == 5
    assert unresolved_refs(d) == [] and unresolved_refs(f) == []
    assert "Epsilon" not in {r["name"] for r in d["entities"]["compounds"]}


def test_currency_metabolites_do_not_create_hops():
    _, big, report = run([rxn("R9", [part("Q", "C90008"), part("Water", "C00001")], [part("R", "C90009")])])
    k = next(k for k, (s, r, b) in keyed(big).items() if s == "processes.reactions" and not b)
    assert big["hops"][k] is None
    assert report["placement"][SID] == {"hop_none": 1}
    assert view(big, DEFAULT)["processes"]["reactions"] == base()["processes"]["reactions"]


def test_same_name_conflicting_ids_is_renamed_not_joined():
    other_b = part("B", "C99999")     # same name as the paper's B, different KEGG id
    _, big, report = run([rxn("R6", [other_b], [part("Zed", "C90010")])])
    names = {r["name"] for r in big["extension"]["entities"]["compounds"]}
    assert "B [pathbank]" in names and "B" not in names
    step = big["extension"]["processes"]["reactions"][0]
    assert step["inputs"] == ["B [pathbank]"]
    k = next(k for k, (s, r, b) in keyed(big).items() if s == "processes.reactions" and not b)
    assert big["hops"][k] is None     # not wired to the paper's B
    assert report["sources"]["pathbank"]["counts"]["renamed_name_conflict"] == 1


def test_one_source_failing_leaves_the_others():
    def boom(ctx, **kw):
        raise ConnectionError("db down at host-x")
    rc = pathway([rxn("R-1", [part("C", "C90003")], [part("Delta", "C90004")])], sid="reactome:R-1", db="reactome")
    big0 = build_big_payload(base())
    big, report = extend_big_payload(big0, sources=("pathbank", "reactome"),
                                     fetchers={"pathbank": boom, "reactome": fake([rc])})
    assert report["failed"] == ["pathbank"]
    assert report["sources"]["pathbank"] == {"status": "failed", "pathways": [], "notes": [], "counts": {},
                                             "error": "ConnectionError"}   # no driver text (could hold a host)
    assert report["placement"] == {"reactome:R-1": {"hop_1": 1}}


def test_flag_parsing_and_off_means_nothing_runs(monkeypatch):
    for v in ("", "0", "false", "No", "off", " OFF "):
        assert not extend_enabled({"T2PW_EXTEND": v})
    assert not extend_enabled({})
    for v in ("1", "true", "yes", "on"):
        assert extend_enabled({"T2PW_EXTEND": v})
    f = fake([pathway(five_steps())])
    big0 = build_big_payload(base())
    monkeypatch.delenv("T2PW_EXTEND", raising=False)
    big, report = run_if_enabled(big0, fetchers={"pathbank": f, "reactome": f, "wikipathways": f})
    assert big is big0 and report is None and f.calls == 0
    monkeypatch.setenv("T2PW_EXTEND", "1")
    big, report = run_if_enabled(big0, sources=("pathbank",), fetchers={"pathbank": f})
    assert f.calls == 1 and big["extension"]


def test_empty_overlay_view_is_base():
    big, report = extend_big_payload(build_big_payload(base()), sources=("pathbank",),
                                     fetchers={"pathbank": fake([])})
    assert report["sources"]["pathbank"]["status"] == "no_candidates"
    assert view(big, DEFAULT) == base()


def test_currency_only_twins_counted():
    # paper step 2 is B + water -> C; the database adds H+ only: VD-11 keeps them apart
    steps = [rxn("R2", [part("Beta", "C90002"), part("Water", "C00001"), part("Hydrogen Ion", "C00080")],
                 [part("Gamma", "C90003")])]
    _, big, report = run(steps)
    assert report["currency_only_twins"]["count"] == 1


def test_reactome_reaction_from_recorded_responses(tmp_path):
    for url, body in json.loads((FIX / "reactome_R-HSA-70467.json").read_text(encoding="utf-8")).items():
        (tmp_path / (hashlib.sha256(url.encode()).hexdigest()[:32] + ".txt")).write_text(json.dumps(body), encoding="utf-8")
    step, why = reaction_step("R-HSA-70467", deadline=0, cache_dir=tmp_path)   # cache only: deadline passed
    assert why is None and step["kind"] == "reaction" and step["location"] == "cytosol"
    assert [p["name"] for p in step["inputs"]] == ["Fru(6)P", "ATP"]
    assert [p["name"] for p in step["outputs"]] == ["F1,6PP", "ADP", "H+"]
    assert step["inputs"][1]["ids"] == {"chebi": "30616"}
    assert {e["ids"]["uniprot"] for e in step["enzymes"]} == {"P17858", "P08237", "Q01813"}   # set members


GPML = """<Pathway xmlns="http://pathvisio.org/GPML/2013a" Name="t">
 <DataNode TextLabel="A" GraphId="a" Type="Metabolite"><Xref Database="ChEBI" ID="CHEBI:1"/></DataNode>
 <DataNode TextLabel="B" GraphId="b" Type="Metabolite"><Xref Database="KEGG Compound" ID="C2"/></DataNode>
 <DataNode TextLabel="ATP" GraphId="atp" Type="Metabolite"><Xref Database="ChEBI" ID="CHEBI:30616"/></DataNode>
 <DataNode TextLabel="ADP" GraphId="adp" Type="Metabolite"><Xref Database="ChEBI" ID="CHEBI:456216"/></DataNode>
 <DataNode TextLabel="Kin" GraphId="k" Type="GeneProduct"><Xref Database="Uniprot-TrEMBL" ID="P1"/></DataNode>
 <DataNode TextLabel="Inh" GraphId="i" Type="Metabolite"><Xref Database="ChEBI" ID="CHEBI:3"/></DataNode>
 <Interaction GraphId="r1"><Graphics><Point GraphRef="a"/><Point GraphRef="b" ArrowHead="mim-conversion"/>
  <Anchor GraphId="an1" Position="0.5"/></Graphics></Interaction>
 <Interaction GraphId="l2"><Graphics><Point GraphRef="k"/><Point GraphRef="an1" ArrowHead="mim-catalysis"/></Graphics></Interaction>
 <Interaction GraphId="l3"><Graphics><Point GraphRef="atp"/><Point GraphRef="an1"/></Graphics></Interaction>
 <Interaction GraphId="l4"><Graphics><Point GraphRef="an1"/><Point GraphRef="adp" ArrowHead="Arrow"/></Graphics></Interaction>
 <Interaction GraphId="l5"><Graphics><Point GraphRef="i"/><Point GraphRef="k" ArrowHead="mim-inhibition"/></Graphics></Interaction>
 <Interaction GraphId="l6"><Graphics><Point GraphRef="k"/><Point GraphRef="i" ArrowHead="Arrow"/></Graphics></Interaction>
</Pathway>"""


def test_gpml_reads_only_what_is_drawn():
    steps, notes = to_steps(parse_gpml(GPML))
    r, i = steps
    assert [p["name"] for p in r["inputs"]] == ["A", "ATP"] and [p["name"] for p in r["outputs"]] == ["B", "ADP"]
    assert [e["name"] for e in r["enzymes"]] == ["Kin"] and r["location"] is None
    assert (i["relationship"], i["entity_1"]["name"], i["entity_2"]["name"]) == ("inhibition", "Inh", "Kin")
    assert notes == ["line l6 (Arrow) not converted"]   # protein -> metabolite arrow: meaning unknown


def test_pathbank_extract_to_neutral():
    from t2pw.vision.extend.pathbank import to_neutral
    cmp = lambda side, i, n, k: {"side": side, "element_type": "Compound", "element_id": i, "name": n,
                                 "synonyms": [], "identifiers": {"kegg_id": k}}
    cx = {"element_type": "ProteinComplex", "element_id": 5, "name": "MenF",
          "components": [{"protein_id": 9, "name": "MenF", "uniprot": "P38051", "stoichiometry": 1}]}
    ex = {"accession": "PW000005", "name": "x", "species": {"name": "Escherichia coli", "taxonomy_id": 562},
          "curator": {"updated_at": "2020-01-01"}, "content_sha256": "h", "extracted_at": "2026-11-17",
          "processes": [
              {"process_type": "Reaction", "accession": "PW_R1", "direction": "=>", "spontaneous": False,
               "biological_state": {"location": "Cytoplasm"},
               "participants": [cmp("left", 7748, "Chorismate", "C00251"), cmp("right", 10, "Isochorismate", "C00885")],
               "catalysts": [cx]},
              {"process_type": "Interaction", "accession": "PW_I1", "interaction_type": "Inhibition",
               "participants": [cmp("left", 11, "DHNA", "C03657"), {**cx, "side": "right"}], "catalysts": []},
              {"process_type": "ReactionCoupledTransport", "accession": "PW_C1", "participants": [], "catalysts": []}]}
    pw = to_neutral(ex)
    r, i = pw["steps"]
    assert r["inputs"][0]["ids"] == {"kegg": "C00251", "pathbank_compound_id": 7748}
    assert r["enzymes"][0]["components"][0]["ids"]["uniprot"] == "P38051" and r["location"] == "Cytoplasm"
    assert (i["relationship"], i["entity_1"]["name"], i["entity_2"]["type"]) == ("inhibition", "DHNA", "complex")
    assert pw["notes"] == ["ReactionCoupledTransport PW_C1: not converted"]
    assert pw["source_id"] == "pathbank:PW000005" and pw["updated_at"] == "2020-01-01"


def test_report_keys():
    _, big, report = run(five_steps())
    assert set(report) >= {"sources", "failed", "placement", "currency_only_twins", "species"}
    assert extend.SOURCES == ("pathbank", "reactome", "wikipathways")
