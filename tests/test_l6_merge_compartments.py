"""L6 (D-W24 5): near-synonym biological states merge, their duplicate chains merge, exact duplicate reactions are written once.

Zero loss: merged-away names stay as aliases, merged-away rows stay whole in the ``final_payload_merge`` record and in
``quarantined_locked_reactions`` (so every lock stays accounted for), enzymes of every copy are unioned on the kept row, and
every reference to a merged state is re-pointed. Flag off (``T2PW_FINAL_PAYLOAD=0``) leaves the payload untouched."""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from t2pw.service.final_payload_merge import MERGE_FIELD, location_key, merge_states_and_duplicates  # noqa: E402


def _cx(name, species="Lactococcus lactis"):
    return {"name": name, "species": species}


def _enz(name, role="catalyst"):
    return {"entity": name, "entity_type": "protein_complex", "role": role, "evidence": f"{name} catalyses it"}


def _rx(name, ins, outs, state, enzymes, lock=None, **kw):
    row = {"name": name, "inputs": list(ins), "outputs": list(outs), "biological_state": state,
           "enzymes": [_enz(e) for e in enzymes], "evidence": f"paper says {name}"}
    if lock:
        row["locked_reaction_id"] = lock
    row.update(kw)
    return row


def _payload(states, reactions, complexes, species=(("Lactococcus lactis", "Prokaryote", "1358"),), **extra):
    p = {"biological_states": states,
         "entities": {"species": [{"name": n, "classification": c, "taxonomy_id": t} for n, c, t in species],
                      "protein_complexes": [_cx(c, species[0][0]) for c in complexes]},
         "processes": {"reactions": reactions, "transports": [], "interactions": []},
         "element_locations": {"compound_locations": []}}
    p.update(extra)
    return p


def _st(name, loc, species="Lactococcus lactis", **kw):
    return {"name": name, "species": species, "subcellular_location": loc, **kw}


# ---- the normalisation table -----------------------------------------------------------------------------------------
@pytest.mark.parametrize("value,prokaryote,expected", [
    ("cell", True, "cytoplasm"), ("intracellular", True, "cytoplasm"), ("cytosol", True, "cytoplasm"),
    ("Cytoplasm", True, "cytoplasm"), ("cytoplasmic state", True, "cytoplasm"),
    ("cell", False, "cell"), ("intracellular", False, "cell"), ("cytosol", False, "cytosol"), ("cytoplasm", False, "cytoplasm"),
    ("extracellular", False, "extracellular space"), ("Extracellular space", True, "extracellular space"),
    ("plastid state", False, "plastid"), ("plastid", False, "plastid"),
    ("plasma_membrane", False, "plasma membrane"), ("Plasma membrane state", False, "plasma membrane"),
    ("mitochondria", False, "mitochondrion"), ("ER", False, "endoplasmic reticulum"),
    ("Golgi apparatus", False, "golgi apparatus"), ("medial Golgi", False, "medial golgi"),
    ("synaptic vesicles", False, "synaptic vesicle"),
])
def test_l6_location_table(value, prokaryote, expected) -> None:
    assert location_key(value, prokaryote) == expected


def test_l6_different_organelles_never_share_a_key() -> None:
    places = ["nucleus", "mitochondrion", "endoplasmic reticulum", "golgi apparatus", "plastid", "chloroplast", "peroxisome",
              "lysosome", "vacuole", "cytosol", "plasma membrane", "extracellular space", "thylakoid membrane", "etioplast"]
    assert len({location_key(p, False) for p in places}) == len(places)


# ---- states and their duplicate chains -------------------------------------------------------------------------------
def test_l6_prokaryote_cell_and_intracellular_merge_and_the_duplicate_chain_is_written_once() -> None:
    """idx 9 shape: the same chain written once in 'L. lactis KF147 state' (cell) and once in the species-level
    'Lactococcus lactis intracellular state' (intracellular)."""
    species = (("Lactococcus lactis subsp. lactis KF147", "Prokaryote", "684738"), ("Lactococcus lactis", "Prokaryote", "1358"))
    states = [_st("L. lactis KF147 state", "cell", "Lactococcus lactis subsp. lactis KF147"),
              _st("extracellular state", "extracellular", "Lactococcus lactis subsp. lactis KF147"),
              _st("Lactococcus lactis intracellular state", "intracellular", "Lactococcus lactis"),
              _st("Lactococcus lactis extracellular state", "extracellular space", "Lactococcus lactis")]
    rx = [_rx("menF", ["chorismate"], ["isochorismate"], "L. lactis KF147 state", ["MenF complex"]),
          _rx("menD", ["isochorismate"], ["SEPHCHC"], "L. lactis KF147 state", ["MenD complex"]),
          _rx("MenF step", ["chorismate"], ["isochorismate"], "Lactococcus lactis intracellular state", ["MenF complex"]),
          _rx("MenD step", ["isochorismate"], ["SEPHCHC"], "Lactococcus lactis intracellular state", ["MenD complex"])]
    p = _payload(states, rx, ["MenF complex", "MenD complex"], species)
    p["element_locations"]["compound_locations"] = [{"compound": "chorismate", "biological_state": "Lactococcus lactis intracellular state"}]
    p["processes"]["transports"] = [{"name": "export", "cargo": "x", "from_biological_state": "Lactococcus lactis intracellular state",
                                     "to_biological_state": "Lactococcus lactis extracellular state"}]
    rep = merge_states_and_duplicates(p)
    assert [s["name"] for s in p["biological_states"]] == ["L. lactis KF147 state", "extracellular state"]
    assert p["biological_states"][0]["aliases"] == ["Lactococcus lactis intracellular state"]
    assert p["biological_states"][1]["aliases"] == ["Lactococcus lactis extracellular state"]
    assert [r["name"] for r in p["processes"]["reactions"]] == ["menF", "menD"]
    t = p["processes"]["transports"][0]
    assert (t["from_biological_state"], t["to_biological_state"]) == ("L. lactis KF147 state", "extracellular state")
    assert p["element_locations"]["compound_locations"][0]["biological_state"] == "L. lactis KF147 state"
    # nothing lost: the merged rows are kept whole, before re-pointing
    assert {s["original"]["name"] for s in rep["states"]} == {"Lactococcus lactis intracellular state", "Lactococcus lactis extracellular state"}
    records = p["quarantined_locked_reactions"]
    assert [r["reaction_name"] for r in records] == ["MenF step", "MenD step"]
    assert records[0]["original_reaction"]["biological_state"] == "Lactococcus lactis intracellular state"
    assert records[0]["original_reaction"]["evidence"] == "paper says MenF step"
    assert records[0]["merged_into"]["reaction_name"] == "menF"
    assert p[MERGE_FIELD]["counts"] == {"states_before": 4, "states_after": 2, "reactions_before": 4, "reactions_after": 2}


def test_l6_plastid_and_plastid_state_merge_but_organelles_do_not() -> None:
    """idx 7 shape: two boxes both labelled plastid; the nucleus and the thylakoid membrane stay apart."""
    sp = (("Arabidopsis thaliana", "Eukaryote", "3702"),)
    a = "Arabidopsis thaliana"
    states = [_st("plastid", "plastid", a), _st("nucleus", "nucleus", a), _st("thylakoid membrane", "thylakoid membrane", a),
              _st("plastid state", "plastid", a), _st("nucleus state", "nucleus", a), _st("plastid state", "plastid", a)]
    p = _payload(states, [_rx("r1", ["a"], ["b"], "plastid state", [])], [], sp)
    merge_states_and_duplicates(p)
    assert [s["name"] for s in p["biological_states"]] == ["plastid", "nucleus", "thylakoid membrane"]
    assert p["processes"]["reactions"][0]["biological_state"] == "plastid"


def test_l6_eukaryote_cell_stays_apart_from_cytosol_when_organelles_exist() -> None:
    sp = (("Homo sapiens", "Eukaryote", "9606"),)
    h = "Homo sapiens"
    p = _payload([_st("mitochondrial state", "mitochondria", h), _st("cytosolic state", "cytosol", h),
                  _st("human carnitine biosynthesis", "cell", h), _st("cell state", "intracellular", h)], [], [], sp)
    rep = merge_states_and_duplicates(p)
    # neither "cell" state becomes the cytosol (a mitochondrion exists), and with no cell type or tissue they do not merge
    # with each other either: only their names say which cell they are
    assert len(p["biological_states"]) == 4
    assert {d["reason"] for d in rep["deferred"]} == {"eukaryote_cell_without_cell_type_or_tissue"}


def test_l6_eukaryote_placeholder_does_not_take_a_named_cell_state() -> None:
    """idx 150: '__auto_state__' and 'afferent nerve fiber state' are both 'cell' with no cell type or tissue."""
    sp = (("Mus musculus", "Eukaryote", "10090"),)
    mm = "Mus musculus"
    p = _payload([_st("Type III cell state", "cell", mm, cell_type="Type III cells", tissue="taste buds"),
                  _st("afferent nerve fiber state", "cell", mm), _st("__auto_state__", "cell", mm)], [], [], sp)
    merge_states_and_duplicates(p)
    assert len(p["biological_states"]) == 3


def test_l6_eukaryote_cell_is_the_cytosol_when_nothing_else_distinguishes_them() -> None:
    sp = (("Homo sapiens", "Eukaryote", "9606"),)
    h = "Homo sapiens"
    p = _payload([_st("cytosolic state", "cytosol", h), _st("cell state", "cell", h), _st("outside", "extracellular", h)], [], [], sp)
    rep = merge_states_and_duplicates(p)
    assert [s["name"] for s in p["biological_states"]] == ["cytosolic state", "outside"]
    assert rep["eukaryote_cell_as_cytosol"] == ["cell state"]


def test_l6_different_cell_type_tissue_or_organism_never_merge() -> None:
    sp = (("Mus musculus", "Eukaryote", "10090"), ("Homo sapiens", "Eukaryote", "9606"))
    m = "Mus musculus"
    p = _payload([_st("Leydig cells of testis", "cell", m, cell_type="Leydig cells", tissue="testis"),
                  _st("adult Leydig cells", "cell", m, cell_type="Leydig cell", tissue="testis"),
                  _st("testis", "cell", m, tissue="testis"),
                  _st("human Leydig", "cell", "Homo sapiens", cell_type="Leydig cells", tissue="testis")], [], [], sp)
    merge_states_and_duplicates(p)
    assert [s["name"] for s in p["biological_states"]] == ["Leydig cells of testis", "testis", "human Leydig"]


def test_l6_positional_qualifiers_in_the_name_keep_states_apart() -> None:
    """Equivalence fixture (sour taste): 'apical extracellular' and 'synaptic extracellular' are both 'extracellular' in
    the taste bud, but the taste pore and the synaptic cleft are different places."""
    sp = (("Mus musculus", "Eukaryote", "10090"),)
    mm = "Mus musculus"
    p = _payload([_st("apical extracellular", "extracellular", mm, tissue="taste bud"),
                  _st("synaptic extracellular", "extracellular", mm, tissue="taste bud"),
                  _st("extracellular space", "extracellular", mm),
                  _st("extracellular space (taste bud lumen)", "extracellular", mm),
                  _st("Type III cell apical membrane", "apical membrane", mm, cell_type="Type III cell"),
                  _st("apical membrane of Type III cells", "apical membrane", mm, cell_type="Type III cells")], [], [], sp)
    merge_states_and_duplicates(p)
    assert [s["name"] for s in p["biological_states"]] == ["apical extracellular", "synaptic extracellular", "extracellular space",
                                                         "extracellular space (taste bud lumen)", "Type III cell apical membrane"]


_HS = (("Homo sapiens", "Eukaryote", "9606"),)
_AT = (("Arabidopsis thaliana", "Eukaryote", "3702"),)


@pytest.mark.parametrize("states,species", [
    # review F1: each pair is two different places that the location field alone would have merged
    ([("hepatocyte cytosol", "cytosol"), ("neuron cytosol", "cytosol")], _HS),
    ([("liver mitochondrion", "mitochondrion"), ("kidney mitochondrion", "mitochondrion")], _HS),
    ([("chloroplast", "chloroplast"), ("thylakoid membrane", "chloroplast")], _AT),
    ([("chloroplast (thylakoid membrane)", "chloroplast"), ("chloroplast (envelope)", "chloroplast")], _AT),
    ([("chloroplast (thylakoid membrane) state", "chloroplast (thylakoid membrane)"),
      ("chloroplast (envelope) state", "chloroplast (envelope)")], _AT),
    ([("nucleus", "nucleus"), ("nucleolus", "nucleus")], _HS),
    ([("lysosome", "lysosome"), ("lysosome membrane", "lysosome")], _HS),
    ([("blood", "extracellular"), ("urine", "extracellular")], _HS),
    ([("extracellular space", "extracellular"), ("blood", "extracellular")], _HS),
])
def test_l6_states_whose_names_disagree_never_merge(states, species) -> None:
    p = _payload([_st(n, loc, species[0][0]) for n, loc in states], [], [], species)
    rep = merge_states_and_duplicates(p)
    assert [s["name"] for s in p["biological_states"]] == [n for n, _ in states]
    assert MERGE_FIELD not in p
    assert any(d["reason"] == "names_disagree" for d in rep["deferred"]) or states[0][1] != states[1][1]


@pytest.mark.parametrize("order", [
    ["cytosol", "hepatocyte cytosol", "neuron cytosol"],
    ["neuron cytosol", "hepatocyte cytosol", "cytosol"],  # review round 2, R1: the reversed order
    ["hepatocyte cytosol", "cytosol", "neuron cytosol"],
])
def test_l6_a_generic_state_does_not_join_two_specific_ones_that_disagree(order) -> None:
    """R1: 'cytosol' agrees with both 'hepatocyte cytosol' and 'neuron cytosol', which disagree with each other: it is the
    same place as neither for certain, so all three stay apart in any row order."""
    p = _payload([_st(n, "cytosol", "Homo sapiens") for n in order], [], [], _HS)
    rep = merge_states_and_duplicates(p)
    assert [s["name"] for s in p["biological_states"]] == order
    assert MERGE_FIELD not in p
    assert [d["state"] for d in rep["deferred"] if d["reason"] == "agrees_with_states_that_disagree"] == ["cytosol"]


def test_l6_a_placeholder_at_row_0_is_not_merged_with_a_stated_state() -> None:
    """R2: the stated state must survive, but row 0 (the default state) never moves: defer."""
    p = _payload([_st("__auto_state__", "cell"), _st("cytoplasmic state", "cytoplasm"), _st("outside", "extracellular")], [], [])
    rep = merge_states_and_duplicates(p)
    assert [s["name"] for s in p["biological_states"]] == ["__auto_state__", "cytoplasmic state", "outside"]
    assert {d["reason"] for d in rep["deferred"]} == {"default_state_is_a_placeholder"}


def test_l6_states_without_a_species_never_merge() -> None:
    p = _payload([{"name": "cytoplasm", "subcellular_location": "cytoplasm"},
                  {"name": "cytoplasm state", "subcellular_location": "cytoplasm"}], [], [])
    merge_states_and_duplicates(p)
    assert len(p["biological_states"]) == 2


@pytest.mark.parametrize("states,species", [
    # the true synonyms the merge exists for keep working
    ([("Lactococcus lactis cells", "cell"), ("intracellular state", "intracellular"), ("Lactococcus lactis cytoplasm", "cytoplasm"),
      ("L. lactis KF147 state", "cell"), ("bacterial cell state", "cell"), ("cytosolic state", "cytosol")],
     (("Lactococcus lactis", "Prokaryote", "1358"),)),
    ([("plastid", "plastid"), ("plastid state", "plastid")], _AT),
    ([("extracellular", "extracellular"), ("extracellular space state", "extracellular space")], _HS),
    ([("ER state", "endoplasmic reticulum"), ("Homo sapiens endoplasmic reticulum", "ER")], _HS),
    ([("lysosome state", "lysosome"), ("Lysosome state", "lysosomal")], _HS),
])
def test_l6_true_synonyms_still_merge(states, species) -> None:
    sp = species[0][0]
    p = _payload([_st(n, loc, sp) for n, loc in states], [], [], species)
    merge_states_and_duplicates(p)
    assert [s["name"] for s in p["biological_states"]] == [states[0][0]]


def test_l6_states_without_a_location_are_keyed_by_name() -> None:
    sp = (("Homo sapiens", "Eukaryote", "9606"),)
    p = _payload([{"name": "cytoplasm", "species": "Homo sapiens"}, {"name": "nucleus", "species": "Homo sapiens"},
                  {"name": "", "species": "Homo sapiens"}, {"species": "Homo sapiens"}], [], [], sp)
    merge_states_and_duplicates(p)
    assert len(p["biological_states"]) == 4


def test_l6_a_transport_whose_ends_would_become_one_state_blocks_the_merge() -> None:
    p = _payload([_st("cell", "cell"), _st("intracellular state", "intracellular")], [], [])
    p["processes"]["transports"] = [{"name": "t", "cargo": "x", "from_biological_state": "cell", "to_biological_state": "intracellular state"}]
    before = copy.deepcopy(p)
    rep = merge_states_and_duplicates(p)
    assert {d["reason"] for d in rep["deferred"]} == {"transport_ends_would_merge"}
    assert p == before  # nothing merged: the payload is untouched, the deferral is only in the returned report


# ---- duplicate reactions ---------------------------------------------------------------------------------------------
def test_l6_isozymes_written_as_four_locked_rows_become_one_reaction_with_all_enzymes_and_every_lock_accounted() -> None:
    """idx 12 shape: one glucosylation written four times, one row per UGT, each with its own lock."""
    from t2pw.pipeline.strict_quarantine import _reconcile_locked_reactions

    ugts = ["UGT91BP2", "UGT703R1", "UGT703R2", "UGT703R3"]
    rx = [_rx(f"{u} glucosylation", ["diosgenin", "UDP-glucose"], ["trillin"], None if i < 3 else "cytoplasmic state", [u],
              lock=f"rxn_lock_00{i + 1}") for i, u in enumerate(ugts)]
    rx.append(_rx("pennogenin", ["pennogenin", "UDP-glucose"], ["pennogenin glucoside"], "cytoplasmic state", ["UGT703R1"], lock="rxn_lock_005"))
    sp = (("Paris polyphylla", "Eukaryote", "1"),)
    p = _payload([_st("cytoplasmic state", "cytoplasm", "Paris polyphylla")], rx, ugts, sp,
                 locked_reaction_filter_report={"locked_reactions_found": 5, "exported_locked_reactions": 5,
                                                "quarantined_locked_reactions": 0, "unaccounted_locked_reactions": 0})
    rep = merge_states_and_duplicates(p)
    kept = p["processes"]["reactions"]
    assert [r["name"] for r in kept] == ["UGT91BP2 glucosylation", "pennogenin"]
    assert [e["entity"] for e in kept[0]["enzymes"]] == ugts
    assert rep["isozyme_groups"] == 1
    assert [r["locked_reaction_id"] for r in p["quarantined_locked_reactions"]] == ["rxn_lock_002", "rxn_lock_003", "rxn_lock_004"]
    acct = _reconcile_locked_reactions(p, [], {})
    assert acct["unaccounted_locked_reactions"] == 0
    assert acct["locked_reactions_found"] == 5


def test_l6_rows_that_are_not_exact_duplicates_stay() -> None:
    rx = [_rx("a", ["x"], ["y"], "cell", ["E"]),
          _rx("stoich", [{"name": "x", "stoichiometry": 2}], ["y"], "cell", ["E"]),
          _rx("reverse", ["x"], ["y"], "cell", ["E"], direction="<"),
          _rx("other product", ["x"], ["z"], "cell", ["E"]),
          _rx("same", ["x"], ["y"], "cell", ["E"])]
    p = _payload([_st("cell", "cell")], rx, ["E"])
    merge_states_and_duplicates(p)
    assert [r["name"] for r in p["processes"]["reactions"]] == ["a", "stoich", "reverse", "other product"]


def test_l6_an_uncatalysed_copy_is_not_folded_into_a_catalysed_one_both_are_flagged() -> None:
    """idx 20 (review F3): 'Spontaneous elimination of SEPHCHC' and 'MenH-catalyzed conversion of SEPHCHC' are conflicting
    claims about the same step; both stay, both are flagged."""
    rx = [_rx("Spontaneous elimination", ["SEPHCHC"], ["SHCHC", "pyruvate"], "cell", [], lock="rxn_lock_005"),
          _rx("MenH-catalyzed conversion", ["SEPHCHC"], ["pyruvate", "SHCHC"], "cell", ["MenH complex"], lock="rxn_lock_004"),
          _rx("MenH again", ["SEPHCHC"], ["pyruvate", "SHCHC"], "cell", ["MenH complex"])]
    p = _payload([_st("cell", "cell")], rx, ["MenH complex"])
    rep = merge_states_and_duplicates(p)
    names = [r["name"] for r in p["processes"]["reactions"]]
    assert names == ["Spontaneous elimination", "MenH-catalyzed conversion"]  # the two catalysed copies still merge
    assert all(r["review_reasons"][0].startswith("same_step_with_and_without_enzyme") for r in p["processes"]["reactions"])
    assert {f["reason"] for f in rep["flagged"]} == {"same_step_with_and_without_enzyme"}


def test_l6_reactions_with_an_empty_side_never_merge() -> None:
    rx = [_rx("a", [], ["y"], "cell", ["E"]), _rx("b", [], ["y"], "cell", ["E"])]
    p = _payload([_st("cell", "cell")], rx, ["E"])
    rep = merge_states_and_duplicates(p)
    assert len(p["processes"]["reactions"]) == 2 and {d["reason"] for d in rep["deferred"]} == {"empty_side"}


def test_l6_a_merged_pair_leaves_the_text_record_coverage_unchanged() -> None:
    """Review F2: a merged-away copy is a KI-014 'merged duplicate, not counted', not an unclassified omission."""
    from t2pw.app.reaction_evidence import _omissions

    unmerged = _dup_payload()
    ir2 = {"processes": {"reactions": [{"name": "a"}, {"name": "b"}], "transports": [], "interactions": []}}
    base = _omissions(unmerged, None, None, ir2, "pwml_ir.json", [], "")["coverage"]
    p = _dup_payload()
    merge_states_and_duplicates(p)
    ir1 = {"processes": {"reactions": [{"name": "a"}], "transports": [], "interactions": []}}
    om = _omissions(p, None, None, ir1, "pwml_ir.json", [], "")
    cov = om["coverage"]
    assert (base["drawn"], base["total"]) == (2, 2)
    assert (cov["drawn"], cov["total"]) == (1, 1)  # 100% before, 100% after: nothing counted as lost
    assert cov["merged_duplicates"] == 1 and cov["not_counted"] == 1
    assert om["omitted"][0]["ki"] == "KI-014"
    assert p["quarantined_locked_reactions"][0]["merged_into"]["note"] == "written once as 'a' (D-W24 5)"


def test_l6_merged_away_names_leave_reaction_order() -> None:
    p = _dup_payload()
    p["reaction_order"] = ["a", "b", "other"]
    rep = merge_states_and_duplicates(p)
    assert p["reaction_order"] == ["a", "other"] and rep["reaction_order_removed"] == 1


def test_l6_a_placeholder_merged_into_the_cytoplasm_marks_its_references_estimated() -> None:
    """Review F7: '__auto_state__' (no stated compartment) merged into the prokaryote cytoplasm."""
    p = _payload([_st("cytoplasmic state", "cytoplasm"), _st("__auto_state__", "cell")], [], [])
    p["element_locations"]["compound_locations"] = [{"compound": "x", "biological_state": "__auto_state__"},
                                                   {"compound": "y", "biological_state": "cytoplasmic state"}]
    merge_states_and_duplicates(p)
    a, b = p["element_locations"]["compound_locations"]
    assert a["biological_state"] == "cytoplasmic state" and a["compartment_source"] == "estimated"
    assert "no stated compartment" in a["compartment_estimate"][0]["reason"]
    assert "compartment_source" not in b
    assert [s["name"] for s in p["biological_states"]] == ["cytoplasmic state"]


def test_l6_the_kept_state_is_the_stated_one_not_the_placeholder() -> None:
    p = _payload([_st("extracellular", "extracellular"), _st("AutoState_lactococcus_lactiscytoplasm", "cytoplasm"),
                  _st("Lactococcus lactis cytoplasm", "cytoplasm")], [], [])
    merge_states_and_duplicates(p)
    assert [s["name"] for s in p["biological_states"]] == ["extracellular", "Lactococcus lactis cytoplasm"]


def test_l6_a_merge_module_that_fails_to_import_does_not_break_the_seam(monkeypatch) -> None:
    import t2pw.service.final_payload as fp

    monkeypatch.setenv("T2PW_FINAL_PAYLOAD", "1")
    monkeypatch.setattr("t2pw.clean_model.fill.cache.Cache._path", lambda self, key: None)
    monkeypatch.setitem(sys.modules, "t2pw.service.final_payload_merge", None)
    p = _dup_payload()
    rep = fp.maybe_apply_final_id_fill(p, None, {}, fetcher=_NoNet())
    assert "error" in rep["merge"] and len(p["processes"]["reactions"]) == 2


def test_l6_a_copy_that_would_add_a_non_complex_enzyme_is_not_merged() -> None:
    rx = [_rx("a", ["x"], ["y"], "cell", ["E complex", "F complex"]), _rx("b", ["x"], ["y"], "cell", ["E complex"])]
    rx[1]["enzymes"].append({"entity": "bare E2", "entity_type": "protein", "role": "catalyst"})
    p = _payload([_st("cell", "cell")], rx, ["E complex", "F complex"])
    p["entities"]["proteins"] = [{"name": "bare E2", "species": "Lactococcus lactis", "uniprot_id": "P00001"}]
    rep = merge_states_and_duplicates(p)
    assert len(p["processes"]["reactions"]) == 2
    assert rep["deferred"][0]["reason"] == "actor_not_a_protein_complex:bare E2"


def test_l6_a_copy_with_a_bare_protein_enzyme_is_never_the_kept_row() -> None:
    """idx 113: 'C4H hydroxylation' (bare protein AtCYP73A5, quarantined at export) and 'C4H reaction' (the complex) are the
    same step. Keeping the first would have folded the drawn copy into the quarantined one and lost the step."""
    rx = [_rx("C4H hydroxylation", ["cinnamic acid"], ["p-coumaric acid"], "cell", [], lock="rxn_lock_002"),
          _rx("C4H reaction", ["cinnamic acid"], ["p-coumaric acid"], "cell", ["Cinnamate 4-hydroxylase"], lock="rxn_lock_004")]
    rx[0]["enzymes"] = [{"entity": "AtCYP73A5", "entity_type": "protein", "role": "catalyst"}]
    p = _payload([_st("cell", "cell")], rx, ["Cinnamate 4-hydroxylase"])
    p["entities"]["proteins"] = [{"name": "AtCYP73A5", "species": "Lactococcus lactis", "uniprot_id": "P92994"}]
    before = copy.deepcopy(p["processes"])
    merge_states_and_duplicates(p)
    assert p["processes"] == before


# ---- the seam --------------------------------------------------------------------------------------------------------
class _NoNet:
    def get_json(self, url, params=None):
        return None

    get_text = get_json


def _dup_payload():
    return _payload([_st("cell", "cell"), _st("intracellular state", "intracellular")],
                    [_rx("a", ["x"], ["y"], "cell", ["E"]), _rx("b", ["x"], ["y"], "intracellular state", ["E"])], ["E"])


def test_l6_the_final_payload_seam_runs_the_merge_flag_on(monkeypatch) -> None:
    import t2pw.service.final_payload as fp

    monkeypatch.setenv("T2PW_FINAL_PAYLOAD", "1")
    monkeypatch.setattr("t2pw.clean_model.fill.cache.Cache._path", lambda self, key: None)
    p = _dup_payload()
    rep = fp.maybe_apply_final_id_fill(p, None, {}, fetcher=_NoNet())
    assert rep["merge"]["counts"] == {"states_before": 2, "states_after": 1, "reactions_before": 2, "reactions_after": 1}
    assert len(p["processes"]["reactions"]) == 1 and MERGE_FIELD in p


def test_l6_flag_off_leaves_the_payload_untouched(monkeypatch) -> None:
    import t2pw.service.final_payload as fp

    monkeypatch.setenv("T2PW_FINAL_PAYLOAD", "0")
    p = _dup_payload()
    before = copy.deepcopy(p)
    assert fp.maybe_apply_final_id_fill(p, None, {}, fetcher=_NoNet()) == {}
    assert p == before


def test_l6_runs_once_and_fails_open(monkeypatch) -> None:
    p = _dup_payload()
    merge_states_and_duplicates(p)
    once = copy.deepcopy(p)
    assert merge_states_and_duplicates(p)["skipped"] == "already_merged" and p == once

    import t2pw.service.final_payload_merge as m

    def boom(*a, **k):
        raise RuntimeError("x")

    monkeypatch.setattr(m, "_merge_reactions", boom)
    q = _dup_payload()
    before = copy.deepcopy(q)
    assert m.merge_states_and_duplicates(q)["error"] == "RuntimeError: x"
    assert q == before
