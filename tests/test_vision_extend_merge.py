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
    assert report["sources"]["pathbank"]["counts"].get("kept_apart_compartment", 0) == 0


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
    assert report["overlay_entities_hop_none"] == 2      # Q and R (F10)
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
    entry = dict(report["sources"]["pathbank"])
    entry.pop("budget_s")
    assert entry == {"status": "failed", "pathways": [], "notes": [], "counts": {},
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
    ewas = {"schemaClass": "EntityWithAccessionedSequence"}
    cache_put(tmp_path, "https://reactome.org/ContentService/data/query/179517", {"hasMember": [ewas, ewas, ewas]})
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
    assert set(report) >= {"sources", "failed", "placement", "currency_only_twins", "species",
                           "overlay_entities_hop_none"}
    assert extend.SOURCES == ("pathbank", "reactome", "wikipathways")


# ---------------------------------------------------------------- review round 1 (F1-F10)

def run_on(b, steps, sid=SID, db="pathbank"):
    big0 = build_big_payload(b)
    return extend_big_payload(big0, sources=(db,), fetchers={db: fake([pathway(steps, sid=sid, db=db)])})


def ext_names(big, section="compounds"):
    return [r["name"] for r in big["extension"].get("entities", {}).get(section, [])]


def test_f1_charged_currency_forms_join_the_family_row_never_a_hub():
    b = base()
    b["entities"]["compounds"].append({"name": "ATP", "mapped_ids": {"chebi": "15422"}, "evidence": "ATP"})
    atp4 = {"type": "compound", "name": "ATP(4-)", "synonyms": [], "ids": {"chebi": "30616"}}
    adp3 = {"type": "compound", "name": "ADP(3-)", "synonyms": [], "ids": {"chebi": "456216"}}
    steps = [rxn("R-1", [part("C", "C90003"), atp4], [part("Delta", "C90004"), adp3]),
             rxn("R-2", [part("Ex", "C90011"), atp4], [part("Why", "C90012"), adp3])]
    big, report = run_on(b, steps, sid="reactome:R-X", db="reactome")
    names = ext_names(big)
    assert not any("[reactome]" in n for n in names)
    assert "ATP" not in names and "ADP" in names          # joined the paper's ATP; bare family name
    assert {r["name"] for r in big["extension"]["processes"]["reactions"]} == {"C + ATP -> Delta + ADP",
                                                                                "Ex + ATP -> Why + ADP"}
    assert report["placement"]["reactome:R-X"] == {"hop_1": 1, "hop_none": 1}   # ATP is no hop edge


def two_state_base():
    b = base()
    b["biological_states"].append({"name": "peri state", "species": "Escherichia coli",
                                   "subcellular_location": "periplasm", "evidence": "periplasm"})
    b["entities"]["subcellular_locations"].append({"name": "periplasm", "evidence": "periplasm"})
    b["processes"]["reactions"].append({"name": "A to B peri", "inputs": ["A"], "outputs": ["B"],
                                        "biological_state": "peri state", "evidence": "also in the periplasm"})
    b["processes"]["interactions"] = [{"name": "A inhibits B", "entity_1": "A", "entity_2": "B",
                                       "relationship": "inhibition", "biological_state": "cell state",
                                       "evidence": "A inhibits B"}]
    return b


def db_records(big, name):
    k = next(k for k, (s, r, b) in keyed(big).items() if b and isinstance(r, dict) and r.get("name") == name)
    return [r for r in big["provenance"][k] if r["kind"] == "database"]


def test_f2_compartment_fold_decides_the_merge_and_picks_the_right_row():
    A, B = part("A", "C90001"), part("B", "C90002")
    inh = {"kind": "interaction", "accession": "I1", "relationship": "inhibition", "entity_1": A, "entity_2": B}
    # GO id for cytosol folds with the paper's "cytoplasm": joins "A to B" (cell state) only
    big, report = run_on(two_state_base(), [dict(rxn("R1", [A], [B], location="cytosol"), location_go="GO:0005829"),
                                            dict(inh, location="cytosol")])
    assert len(db_records(big, "A to B")) == 1 and db_records(big, "A to B peri") == []
    assert len(db_records(big, "A inhibits B")) == 1
    assert "processes" not in big["extension"]
    # periplasm: joins the periplasm row, not the first candidate
    big, _ = run_on(two_state_base(), [rxn("R1", [A], [B], location="periplasm")])
    assert db_records(big, "A to B") == [] and len(db_records(big, "A to B peri")) == 1
    # mitochondrial matrix matches neither: written apart with its own state, counted
    big, report = run_on(two_state_base(), [rxn("R1", [A], [B], location="mitochondrial matrix"),
                                            dict(inh, location="mitochondrial matrix")])
    assert db_records(big, "A to B") == [] and db_records(big, "A to B peri") == []
    assert [r["biological_state"] for r in big["extension"]["processes"]["reactions"]] == \
        ["mitochondrial matrix (Escherichia coli)"]
    assert len(big["extension"]["processes"]["interactions"]) == 1
    assert report["sources"]["pathbank"]["counts"]["kept_apart_compartment"] == 2
    # no compartment on the database side: joins the first candidate
    big, _ = run_on(two_state_base(), [rxn("R1", [A], [B], location=None)])
    assert len(db_records(big, "A to B")) == 1


def cache_put(d, url, body):
    (d / (hashlib.sha256(url.encode()).hexdigest()[:32] + ".txt")).write_text(json.dumps(body), encoding="utf-8")


def test_f3_reactome_complex_with_a_nested_set_has_no_components(tmp_path):
    base_url = "https://reactome.org/ContentService"
    ref = lambda u, n: {"identifier": u, "displayName": f"UniProt:{u} {n}", "schemaClass": "ReferenceGeneProduct"}
    cache_put(tmp_path, f"{base_url}/data/query/R-T-1", {
        "displayName": "t", "input": [{"dbId": 1}], "output": [{"dbId": 2}], "catalystActivity": [{"dbId": 3}],
        "compartment": [{"displayName": "cytosol", "accession": "0005829", "databaseName": "GO"}]})
    cache_put(tmp_path, f"{base_url}/data/participants/R-T-1", [
        {"peDbId": 1, "displayName": "A [cytosol]", "schemaClass": "SimpleEntity",
         "refEntities": [{"identifier": "1", "displayName": "a [ChEBI:1]", "schemaClass": "ReferenceMolecule"}]},
        {"peDbId": 2, "displayName": "B [cytosol]", "schemaClass": "SimpleEntity",
         "refEntities": [{"identifier": "2", "displayName": "b [ChEBI:2]", "schemaClass": "ReferenceMolecule"}]},
        {"peDbId": 10, "displayName": "Enz [cytosol]", "schemaClass": "Complex",
         "refEntities": [ref("P00001", "X"), ref("P00002", "Y")]}])
    cache_put(tmp_path, f"{base_url}/data/query/3", {"physicalEntity": {"dbId": 10}})
    cache_put(tmp_path, f"{base_url}/data/complex/10/subunits",
              [{"dbId": 11, "schemaClass": "DefinedSet"}, {"dbId": 12, "schemaClass": "EntityWithAccessionedSequence"}])
    step, _ = reaction_step("R-T-1", deadline=0, cache_dir=tmp_path)
    assert step["enzymes"][0]["components"] == [] and step["location_go"] == "GO:0005829"
    cache_put(tmp_path, f"{base_url}/data/complex/10/subunits",
              [{"dbId": 12, "schemaClass": "EntityWithAccessionedSequence"},
               {"dbId": 13, "schemaClass": "EntityWithAccessionedSequence"}])
    step, _ = reaction_step("R-T-1", deadline=0, cache_dir=tmp_path)
    assert [c["ids"]["uniprot"] for c in step["enzymes"][0]["components"]] == ["P00001", "P00002"]


def test_f4_member_join_refused_with_an_ungrounded_component():
    enz = {"type": "complex", "name": "Alpha synthase", "synonyms": [],
           "components": [part("alpha", t="protein", uniprot="P11111"), part("beta", t="protein")]}
    big, report = run_on(base(), [rxn("R3", [part("C", "C90003")], [part("Delta", "C90004")], [enz])])
    assert ext_names(big, "protein_complexes") == ["Alpha synthase"]
    assert report["sources"]["pathbank"]["counts"].get("complex_joined_by_members", 0) == 0


def test_f6_incoming_synonym_naming_another_row_is_dropped():
    delta = dict(part("Delta", "C90004"), synonyms=["B", "delta-thing"])
    big, report = run_on(base(), [rxn("R3", [part("C", "C90003")], [delta])])
    row = next(r for r in big["extension"]["entities"]["compounds"] if r["name"] == "Delta")
    assert row["synonyms"] == ["delta-thing"]
    assert report["sources"]["pathbank"]["counts"]["synonyms_dropped_collision"] == 1


def test_f7_one_bad_pathway_does_not_stop_the_source():
    bad = pathway([rxn("RB", [{"name": "no type"}], [part("Delta", "C90004")])], sid="pathbank:PWBAD")
    good = pathway([rxn("R3", [part("C", "C90003")], [part("Delta", "C90004")])])
    big, report = extend_big_payload(build_big_payload(base()), sources=("pathbank",),
                                     fetchers={"pathbank": fake([bad, good])})
    entry = report["sources"]["pathbank"]
    assert entry["status"] == "ok" and entry["counts"]["pathways_failed"] == 1
    assert report["placement"] == {SID: {"hop_1": 1}}


def test_f9_gpml_arrowhead_not_on_last_point_is_skipped():
    g = GPML.replace('<Point GraphRef="a"/><Point GraphRef="b" ArrowHead="mim-conversion"/>',
                     '<Point GraphRef="a" ArrowHead="mim-conversion"/><Point GraphRef="b"/>')
    steps, notes = to_steps(parse_gpml(g))
    assert [s["kind"] for s in steps] == ["interaction"]
    assert "line r1: arrowhead not on the last point, skipped" in notes


# ---------------------------------------------------------------- review round 2 (R1-R4)

def test_r1_currency_never_joins_a_grounded_non_currency_row():
    b = base()
    b["entities"]["compounds"].append({"name": "Adenosine triphosphate", "mapped_ids": {"chebi": "15422"},
                                       "evidence": "ATP"})
    atp = {"type": "compound", "name": "ATP(4-)", "synonyms": [], "ids": {"chebi": "30616"}}
    big, report = run_on(b, [rxn("R-2", [part("Ex", "C90011"), atp], [part("Why", "C90012")])])
    assert "ATP" in ext_names(big)                       # bare family row, not the grounded paper row
    assert big["extension"]["processes"]["reactions"][0]["inputs"] == ["Ex", "ATP"]
    assert report["placement"][SID] == {"hop_none": 1}   # no hop through the paper's ATP row


def test_r2_catalyst_set_of_complexes_is_one_name_only_enzyme(tmp_path):
    test_reactome_reaction_from_recorded_responses(tmp_path)   # writes the recorded responses
    cache_put(tmp_path, "https://reactome.org/ContentService/data/query/179517",
              {"hasMember": [{"schemaClass": "Complex"}, {"schemaClass": "EntityWithAccessionedSequence"}]})
    step, _ = reaction_step("R-HSA-70467", deadline=0, cache_dir=tmp_path)
    assert step["enzymes"] == [{"type": "complex", "name": "PFK tetramer", "synonyms": [], "components": []}]


def test_r3_kept_apart_not_counted_when_a_stateless_row_merges():
    b = two_state_base()
    b["processes"]["reactions"] = [b["processes"]["reactions"][0],
                                   {"name": "A to B any", "inputs": ["A"], "outputs": ["B"], "evidence": "A to B"}]
    big, report = run_on(b, [rxn("R1", [part("A", "C90001")], [part("B", "C90002")], location="periplasm")])
    assert len(db_records(big, "A to B any")) == 1 and db_records(big, "A to B") == []
    assert "kept_apart_compartment" not in report["sources"]["pathbank"]["counts"]


def test_r4_intracellular_is_not_cytoplasm():
    from t2pw.vision.extend.convert import fold
    assert fold("cytosol") == fold("Cytoplasm") == fold(None, "GO:0005829") == fold(None, "0005737")
    assert fold("intracellular") != fold("cytosol") and fold(None, "GO:0005622") != fold("cytosol")


# ---------------------------------------------------------------- round 3 (live hand check: S1-S5, S7)

def test_s1_unused_species_row_never_chooses_the_organism():
    from t2pw.vision.extend import paper_context
    b = base()
    b["entities"]["species"].append({"name": "Arabidopsis thaliana", "taxonomy_id": "3702", "evidence": "x"})
    b["entities"]["proteins"].append({"name": "E9", "species": "Lactococcus lactis", "evidence": "E9"})
    ctx = paper_context(build_big_payload(b))
    assert ctx["species"] == [{"name": "Escherichia coli", "taxonomy_id": "562"},
                              {"name": "Lactococcus lactis", "taxonomy_id": None}]   # used, no species row


class FakeReader:
    """Answers pathbank.candidates' SELECTs by table (no database)."""

    def rows(self, sql, params=()):
        if "FROM species" in sql:
            return [{"id": 1, "name": "Mus musculus", "taxonomy_id": 10090}]
        if "FROM reaction_elements" in sql:
            return [{"pid": 10, "n": 5}, {"pid": 11, "n": 3}]
        if "FROM pathways p JOIN super_pathway_visualizations" in sql:
            return [{"pid": 10, "name": "Generated Pathway 1790128106", "species_id": 1, "pw_id": "PW767504"},
                    {"pid": 11, "name": "\tAndrogen and Estrogen Metabolism ", "species_id": 1, "pw_id": "PW064569"}]
        return []


def test_s2_generated_pathway_records_are_excluded_and_s7_names_cleaned():
    from t2pw.vision.extend.pathbank import candidates
    cands, notes = candidates(FakeReader(), {"species": [{"name": "Mus musculus", "taxonomy_id": "10090"}],
                                             "ids": {"pathbank_compound": {"37", "157"}}})
    assert [(c["pw_id"], c["name"]) for c in cands] == [("PW064569", "Androgen and Estrogen Metabolism")]
    assert any("PW767504" in n and "not curated" in n for n in notes)


def test_s7_pathbank_participant_names_are_stripped():
    from t2pw.vision.extend.pathbank import to_neutral
    p = {"side": "left", "element_type": "Compound", "element_id": 1, "name": "\t Chlorophyllide ",
         "synonyms": [" chlide\t"], "identifiers": {}}
    q = dict(p, side="right", name="lipid A disaccharide \t", synonyms=[])
    pw = to_neutral({"accession": "PW1", "name": " x\t", "processes": [
        {"process_type": "Reaction", "accession": "R", "participants": [p, q], "catalysts": []}]})
    r = pw["steps"][0]
    assert (r["inputs"][0]["name"], r["inputs"][0]["synonyms"], r["outputs"][0]["name"], pw["name"]) == \
        ("Chlorophyllide", ["chlide"], "lipid A disaccharide", "x")


def test_s3_reactome_species_without_taxid_skips_the_source(tmp_path):
    from t2pw.vision.extend.reactome import fetch
    cache_put(tmp_path, "https://reactome.org/ContentService/data/species/all",
              [{"dbId": 170905, "displayName": "Arabidopsis thaliana", "schemaClass": "Species"}])
    pws, notes = fetch({"species": [{"name": "Arabidopsis thaliana", "taxonomy_id": "3702"}], "ids": {}},
                       deadline=0, cache_dir=tmp_path)
    assert pws == [] and "without a taxId" in notes[0]


def test_s4_each_source_gets_its_share_of_what_is_left():
    import time
    got = {}

    def rec(name):
        def f(ctx, *, deadline, **kw):
            got[name] = deadline - time.monotonic()
            return [], []
        return f
    extend_big_payload(build_big_payload(base()), budget_s=100,
                       fetchers={s: rec(s) for s in ("pathbank", "reactome", "wikipathways")})
    assert 48 < got["pathbank"] <= 50          # 2 of 4 shares
    assert 48 < got["reactome"] <= 50          # 1 of 2 shares of the ~100 s still left
    assert 97 < got["wikipathways"] <= 100     # everything left


def test_s5_grounded_rows_never_join_on_a_synonym_alone():
    b = base()
    b["entities"]["compounds"].append({"name": "coproporphyrinogen III", "mapped_ids": {"kegg": "C03263"},
                                       "synonyms": ["Coproporphyrin III"], "evidence": "x"})
    b["processes"]["reactions"].append({"name": "C to copro", "inputs": ["C"], "outputs": ["coproporphyrinogen III"],
                                        "biological_state": "cell state", "evidence": "y"})
    copro = {"type": "compound", "name": "Coproporphyrin III", "synonyms": [], "ids": {"pathbank_compound_id": 999}}
    big, report = run_on(b, [rxn("PW_R213418", [part("coproporphyrinogen III", "C03263"), part("O2", "C00007")],
                                 [copro, part("H2O2", "C00027")])])
    step = big["extension"]["processes"]["reactions"][0]
    assert step["inputs"] == ["coproporphyrinogen III", "O2"]
    assert step["outputs"] == ["Coproporphyrin III [pathbank]", "H2O2"]     # not a self-loop
    assert report["sources"]["pathbank"]["counts"]["name_join_refused_no_shared_id"] == 1


def test_source_provenance_names_the_pathway_and_step():
    # KDSR case: a source-data error stays traceable to its record (PathBank PW000148 PW_R001169)
    big, _ = run_on(base(), [rxn("PW_R001169", [part("C", "C90003"), part("NADPH", "C00005")],
                                 [part("Delta", "C90004"), part("NADP+", "C00006")])], sid="pathbank:PW000148")
    k = next(k for k, (s, r, b) in keyed(big).items() if s == "processes.reactions" and not b)
    rec = big["provenance"][k][0]
    assert rec["kind"] == "database" and rec["source"] == "pathbank:PW000148"
    assert "PW000148" in rec["note"] and "PW_R001169" in rec["note"]
    assert big["sources"]["pathbank:PW000148"]["accession"] == "PW000148"
