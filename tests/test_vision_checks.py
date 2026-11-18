"""Development checks (src/t2pw/vision/checks): validators have teeth, cross-format comparison classes, faithfulness quotes.
Offline: the stored run fixtures of tests/data/vision/generate (no DB, no network, no LLM)."""
from __future__ import annotations

import copy
import json
import re
from pathlib import Path

import pytest

pytest.importorskip("lxml.etree")
pytest.importorskip("libsbml")

from t2pw.vision import generate as G  # noqa: E402
from t2pw.vision.checks import consistency, faithfulness, validate  # noqa: E402
from t2pw.vision.payload import build_big_payload, merge_extension  # noqa: E402
from t2pw.vision.view import DEFAULT  # noqa: E402

DATA = Path(__file__).parent / "data" / "vision" / "generate"
FIX = DATA / "run39_final_mapped.json"
OPTS = {"strict_db": True, "export_mode": "pathwhiz", "pathway_context": {"key_compounds": [
    "alas2 aminolevulinic acid synthase 2", "aminolevulinic acid ala", "glycine", "heme", "pyridoxal 5phosphate plp", "succinylcoa"]}}
QUOTE = "condensation of glycine and succinyl-CoA to produce aminolevulinic acid"


@pytest.fixture(scope="module")
def run():
    """(big payload, {fmt: bytes}, PWML IR) of the run39 fixture at the default scope."""
    G._MEMO.clear()
    G._KEPT.clear()
    big = build_big_payload(json.loads(FIX.read_text(encoding="utf-8")))
    outs = {f: G.generate(big, DEFAULT, f, export=OPTS) for f in ("pwml", "sbml", "sbgn", "kgml")}
    from t2pw.vision.view import view
    ir = G._export_pwml(view(big, DEFAULT), dict(OPTS))[1]
    G._MEMO.clear()
    G._KEPT.clear()
    return big, outs, ir


def classes(rep):
    return {f["class"] for f in rep["failures"]}


# ---------------------------------------------------------------- validators

def test_valid_outputs_pass(run):
    _, outs, _ = run
    assert validate.validate_sbml(outs["sbml"])["ok"]
    assert validate.validate_sbgn(outs["sbgn"])["ok"]
    assert validate.validate_pwml(outs["pwml"])["ok"]


def test_sbml_validator_catches_a_broken_file(run):
    sbml = run[1]["sbml"].decode("utf-8")
    broken = sbml.replace('speciesReference species="s_2_mitochondrial_matrix"', 'speciesReference species="s_nowhere"', 1)
    assert broken != sbml
    v = validate.validate_sbml(broken.encode("utf-8"))
    assert not v["ok"] and v["errors"]
    assert not validate.validate_sbml(b"<sbml><oops></sbml>")["ok"]  # not even XML


def test_sbgn_validator_catches_a_dropped_compartment_ref(run):
    sbgn = run[1]["sbgn"].decode("utf-8")
    broken = sbgn.replace(' compartmentRef="c1"', "", 1)
    assert broken != sbgn
    v = validate.validate_sbgn(broken.encode("utf-8"))
    assert not v["ok"] and any(f.startswith("pd10112") for f in v["schematron_failures"])
    assert not validate.validate_sbgn(sbgn.replace('class="process"', 'class="processs"').encode("utf-8"))["ok"]  # XSD enum


def test_pwml_validator_catches_a_broken_file(run):
    pwml = run[1]["pwml"]
    assert not validate.validate_pwml(pwml[: len(pwml) // 2])["ok"]  # truncated: not well-formed
    no_section = re.sub(rb"<compounds>.*?</compounds>", b"", pwml, count=1, flags=re.S)
    assert no_section != pwml and not validate.validate_pwml(no_section)["ok"]  # structure signature
    assert "pwc-id" in validate.validate_pwml(pwml)["db_identity_absent"]  # offline: counted, not an error


# ---------------------------------------------------------------- consistency

def test_clean_run_is_consistent(run):
    _, outs, ir = run
    rep = consistency.cross_format(outs, pwml_ir=ir)
    assert rep["ok"], rep["failures"]
    assert rep["counts"]["pwml"]["processes"] == rep["counts"]["sbml"]["processes"] == rep["counts"]["kgml"]["processes"] == 2
    # the two interactions SBML and KGML carry, PWML withholds (bare protein endpoint): explained by its own IR record
    assert rep["known"]["e_pwml_withheld_interactions"]["n"] == 2
    k = rep["known"]  # the fixture's known counts (one single-member complex; no Unknown, no modifier interaction, no extra placement)
    assert k["d_pathwhiz_complex_wrapper"]["n"] == 1
    assert k["f_unknown_sentinel"]["n"] == 0 and set(k["f_unknown_sentinel"]["per_format"].values()) == {0}
    assert k["g_pwml_modifier_interactions"]["n"] == 0 and k["h_pwml_extra_placements"]["n"] == 0
    assert "interaction_missing_in_pwml" in classes(consistency.cross_format(outs))  # without the IR it is a failure


def test_injected_missing_reaction_is_flagged(run):
    _, outs, ir = run
    sbml = run[1]["sbml"].decode("utf-8")
    cut = re.sub(r'<reaction id="r_2".*?</reaction>', "", sbml, count=1, flags=re.S)
    assert cut != sbml
    rep = consistency.cross_format(dict(outs, sbml=cut.encode("utf-8")), pwml_ir=ir)
    assert not rep["ok"] and "process_missing_in_sbml" in classes(rep)
    kgml = outs["kgml"].decode("utf-8")
    cut = re.sub(r'<reaction id="8".*?</reaction>', "", kgml, count=1, flags=re.S)
    assert cut != kgml and "process_missing_in_kgml" in classes(consistency.cross_format(dict(outs, kgml=cut.encode("utf-8")), pwml_ir=ir))


def test_canonical_name_difference_is_known_not_a_failure(run):
    _, outs, ir = run
    pwml = outs["pwml"].decode("utf-8")
    renamed = pwml.replace("<name>Glycine</name>", "<name>Glycine (canonical)</name>", 1)
    assert renamed != pwml
    rep = consistency.cross_format(dict(outs, pwml=renamed.encode("utf-8")), pwml_ir=ir)
    assert rep["ok"], rep["failures"]  # same entity by database id
    assert rep["known"]["a_canonical_names"]["n"] == 1 and "Glycine (canonical)" in rep["known"]["a_canonical_names"]["examples"][0]


def test_payload_formats_that_disagree_on_a_name_fail(run):
    _, outs, ir = run
    sbgn = outs["sbgn"].decode("utf-8")
    renamed = re.sub(r'(<(?:\w+:)?label text=)"Glycine"', r'\1"Glycin"', sbgn, count=1)  # with or without a namespace prefix
    assert renamed != sbgn
    # SBGN keeps the Glycine ids, so it is still the same entity, but its name now differs from SBML and KGML
    assert "name_differs_between_payload_formats" in classes(consistency.cross_format(dict(outs, sbgn=renamed.encode("utf-8")), pwml_ir=ir))


def test_missing_entity_in_one_format_is_flagged(run):
    _, outs, ir = run
    sbml = outs["sbml"].decode("utf-8")
    cut = re.sub(r'<species id="s_3_mitochondrial_matrix".*?</species>', "", sbml, count=1, flags=re.S)  # heme, only in an interaction
    assert cut != sbml
    rep = consistency.cross_format(dict(outs, sbml=cut.encode("utf-8")), pwml_ir=ir)
    assert "entity_missing_in_sbml" in classes(rep)
    # SBGN does not draw heme (only in an interaction): known class b, not a failure
    assert "entity_missing_in_sbgn" not in classes(consistency.cross_format(outs, pwml_ir=ir))
    assert consistency.cross_format(outs, pwml_ir=ir)["known"]["b_sbgn_entities_only_in_interactions"]["n"] == 1


def test_injected_compartment_grouping_mismatch_is_flagged(run):
    _, outs, ir = run
    sbml = outs["sbml"].decode("utf-8")
    moved = sbml.replace('name="Glycine" compartment="c_mitochondrial_matrix"', 'name="Glycine" compartment="c_cell"', 1)
    assert moved != sbml
    rep = consistency.cross_format(dict(outs, sbml=moved.encode("utf-8")), pwml_ir=ir)
    assert "compartment_placement_differs:sbml_vs_sbgn" in classes(rep)  # one placement each, and they differ
    # the exporter's own compartment_assumed notes bound what is allowed: told of one assumption, the difference is class j
    told = consistency.cross_format(dict(outs, sbml=moved.encode("utf-8")), pwml_ir=ir,
                                    sbgn_report={"notes": {"compartment_assumed:transporter_in_from_compartment": 1}})
    assert "compartment_placement_differs:sbml_vs_sbgn" not in classes(told) and told["known"]["j_sbgn_assumed_placements"]["n"] == 1


def test_sbml_only_placement_is_a_known_class_not_a_failure(run):
    _, outs, ir = run
    sbml = outs["sbml"].decode("utf-8")
    extra = sbml.replace("</listOfSpecies>", '<species id="s_99_cell" name="Glycine" compartment="c_cell"/></listOfSpecies>', 1)
    assert extra != sbml  # SBML lists Glycine in a second compartment no process uses it in
    rep = consistency.cross_format(dict(outs, sbml=extra.encode("utf-8")), pwml_ir=ir)
    assert rep["ok"], rep["failures"]
    assert rep["known"]["k_sbml_only_placements"]["n"] >= 1


def test_entity_placed_in_a_format_but_not_in_pwml_is_flagged(run):
    _, outs, ir = run
    pwml = outs["pwml"].decode("utf-8")
    cut = re.sub(r"<compound-locations>.*?</compound-locations>", "<compound-locations/>", pwml, count=1, flags=re.S)
    assert cut != pwml
    rep = consistency.cross_format(dict(outs, pwml=cut.encode("utf-8")), pwml_ir=ir)
    assert "compartment_placement_missing_in_pwml" in classes(rep)


def test_injected_catalyst_mismatch_is_flagged(run):
    _, outs, ir = run
    sbml = outs["sbml"].decode("utf-8")
    changed = sbml.replace('<modifierSpeciesReference species="s_7_mitochondrial_matrix" sboTerm="SBO:0000013"/>',
                           '<modifierSpeciesReference species="s_1_mitochondrial_matrix" sboTerm="SBO:0000013"/>', 1)
    assert changed != sbml
    assert "catalyst_mismatch:pwml_vs_sbml" in classes(consistency.cross_format(dict(outs, sbml=changed.encode("utf-8")), pwml_ir=ir))


def test_sbgn_regulatory_arcs_are_compared_with_sbml(run):
    _, outs, ir = run
    sbgn = outs["sbgn"].decode("utf-8")
    # ElementTree's namespace registry is process-wide: once an SBML module registers the default namespace after export_sbgn
    # did, the SBGN bytes carry an ns0: prefix, so the surgery takes the prefix the file uses
    pre = (re.search(r"<(\w+:)?map[ >]", sbgn).group(1)) or ""
    extra = sbgn.replace("</%smap>" % pre, '<%(p)sarc class="inhibition" id="a99" source="g1" target="p1"><%(p)sstart x="0" y="0" /><%(p)send x="0" y="0" /></%(p)sarc></%(p)smap>' % {"p": pre}, 1)
    assert extra != sbgn  # an SBGN inhibition arc that nothing in SBML or PWML explains
    assert "sbgn_regulatory_arc_extra" in classes(consistency.cross_format(dict(outs, sbgn=extra.encode("utf-8")), pwml_ir=ir))
    sbml = outs["sbml"].decode("utf-8")
    plus = sbml.replace('<modifierSpeciesReference species="s_7_mitochondrial_matrix" sboTerm="SBO:0000013"/>',
                        '<modifierSpeciesReference species="s_7_mitochondrial_matrix" sboTerm="SBO:0000013"/>'
                        '<modifierSpeciesReference species="s_3_mitochondrial_matrix" sboTerm="SBO:0000459"/>', 1)
    assert plus != sbml  # an SBML regulator SBGN has no arc for
    bad = consistency.cross_format(dict(outs, sbml=plus.encode("utf-8")), pwml_ir=ir)
    assert "sbgn_regulatory_arc_missing" in classes(bad)
    # class b only as far as the exporter's own report lists omitted reaction arcs
    told = consistency.cross_format(dict(outs, sbml=plus.encode("utf-8")), pwml_ir=ir, sbgn_report={"omitted": {"reaction_arcs": {"role_not_regulatory": 1}}})
    assert "sbgn_regulatory_arc_missing" not in classes(told) and told["known"]["b_sbgn_omitted_regulatory_arcs"]["n"] == 1


def test_unknown_sentinel_is_only_a_protein_in_pwml_and_never_in_the_clean_model(run):
    _, outs, ir = run
    sbml = outs["sbml"].decode("utf-8")
    renamed = sbml.replace('name="Glycine"', 'name="Unknown"', 1)
    assert renamed != sbml
    rep = consistency.cross_format(dict(outs, sbml=renamed.encode("utf-8")), pwml_ir=ir)
    assert "unknown_sentinel_in_sbml" in classes(rep) and rep["known"]["f_unknown_sentinel"]["per_format"]["sbml"] == 1


# ---------------------------------------------------------------- faithfulness

def test_short_quotes_are_counted_apart_and_ids_match_whole_tokens():
    text = "ALAS: " + QUOTE + " in the matrix."  # holds the default quote, so the only change below is the short one
    res = faithfulness.paper_view(big_with("glycine"), text)  # 7 characters: found in any text, so not judged
    assert res["short"] >= 1 and res["not_found"] == faithfulness.paper_view(big_with(), text)["not_found"]
    assert faithfulness._has("glycine", "free glycine here") and not faithfulness._has("glycine", "polyglycine here")
    assert faithfulness._has("1234", "id 1234.") and not faithfulness._has("1234", "id 912345")


def test_fold_finds_spaced_charges_and_primes_and_reports_before_and_after():
    quote = "permeation of H+ through the 5'-phosphate channel"
    text = "We saw permeation of H + through the 5′-phosphate channel in cells."
    assert faithfulness.quote_in(quote, faithfulness.norm_text(text)) == "no"  # before the fold
    assert faithfulness.quote_in(quote, faithfulness.fold_text(text), faithfulness.fold_text) == "verbatim"
    res = faithfulness.paper_view(big_with(quote), text)
    assert res["verbatim"] == res["before_fold"]["verbatim"] + 1 and res["before_fold"]["not_found"] == res["not_found"] + 1


def big_with(quote=QUOTE, audit_quote=None):
    base = json.loads(FIX.read_text(encoding="utf-8"))
    base["processes"]["reactions"][0]["evidence"] = quote
    if audit_quote:
        rx = base["processes"]["reactions"][1]
        rx["evidence"] = audit_quote
        rx["provenance_lineage"] = [{"stage": "audit", "origin": "audit_modified", "reason": "x", "sources": []}]
    return build_big_payload(base)


def test_faithfulness_finds_a_planted_quote_and_misses_a_planted_non_quote():
    text = "Results.  ALAS  catalyses the\n  Condensation of glycine and Succinyl-CoA  to produce aminolevulinic acid (ALA), in the matrix."
    ok = faithfulness.paper_view(big_with(), text)  # whitespace and case are normalised
    assert faithfulness.quote_in(QUOTE, faithfulness.norm_text(text)) == "verbatim" and ok["verbatim"] >= 1
    bad = faithfulness.paper_view(big_with("heme is exported to the nucleus by a novel pump"), text)
    assert bad["not_found"] == ok["not_found"] + 1 and bad["verbatim"] == ok["verbatim"] - 1  # only the planted quote changed
    assert any(e["quote"].startswith("heme is exported") for e in bad["not_found_examples"])
    assert faithfulness.paper_view(big_with(), None)["no_text"] >= 1  # no text is reported, not guessed
    frag = faithfulness.paper_view(big_with("condensation of glycine and succinyl-CoA ... in the matrix"), text)
    assert frag["fragments"] == 1


def test_audit_modified_only_rows_are_counted_as_promotion_candidates():
    text = "a PLP-dependent homodimer enzyme that mediates the condensation of glycine and succinyl-CoA."
    res = faithfulness.paper_view(big_with(audit_quote="a PLP-dependent homodimer enzyme"), text)["audit_modified_only"]
    assert res["elements"] == 1 and res["with_quote"] == 1 and res["promotion_candidates"] == 1
    res = faithfulness.paper_view(big_with(audit_quote="never written in the paper"), text)["audit_modified_only"]
    assert res["promotion_candidates"] == 0 and res["quote_not_found"] == 1


def test_extended_view_offline_cache(tmp_path):
    big = big_with()
    assert faithfulness.extended_view(big)["status"] == "no extension"
    ext = merge_extension(big, {"processes.reactions": [{"name": "ext step", "inputs": ["Glycine"], "outputs": ["heme"], "evidence": "ext"}]},
                          {"processes.reactions": [[{"kind": "database", "source": "reactome:R-1", "quote": None, "stage": "extend", "note": ""},
                                                    {"kind": "other_paper", "source": "PMC1", "quote": "heme is made from glycine", "stage": "extend", "note": ""}]]},
                          {"reactome:R-1": {"kind": "database"}, "PMC1": {"kind": "other_paper"}})
    assert faithfulness.extended_view(ext)["status"] == "extension"
    none = faithfulness.extended_view(ext)  # nothing cached: counted, never fetched
    assert none["database"]["no_cache"] >= 1 and none["other_paper"]["no_cache"] >= 1
    cache = {"reactome:R-1": "Glycine + succinyl-CoA -> heme", "PMC1": "We show that Heme is made from glycine in mitochondria."}
    res = faithfulness.extended_view(ext, cache)
    assert res["other_paper"]["verbatim"] >= 1 and res["database"]["found"] >= 1 and res["database"]["not_found"] == 0
    (tmp_path / "reactome_R_1.txt").write_text("an unrelated record", encoding="utf-8")  # directory form of the cache
    (tmp_path / "PMC1.txt").write_text("a different paper", encoding="utf-8")
    res = faithfulness.extended_view(ext, tmp_path)
    assert res["database"]["not_found"] >= 1 and res["other_paper"]["not_found"] == 1


def test_run_checks_on_a_payload_dir(tmp_path):
    """The CLI path on a minimal run folder: final_mapped.json + the source text one level up."""
    from t2pw.vision.checks import run_checks
    paper = tmp_path / "paper"
    run_dir = paper / "strict"
    run_dir.mkdir(parents=True)
    base = json.loads(FIX.read_text(encoding="utf-8"))
    (run_dir / "final_mapped.json").write_text(json.dumps(copy.deepcopy(base)), encoding="utf-8")
    (paper / "01_source_text.txt").write_text("ALAS: " + QUOTE + ".", encoding="utf-8")
    (run_dir / "quarantine_report.json").write_text(json.dumps({"strict_db": True, "decision_inputs": {
        "export_mode": "pathwhiz", "requested_core": OPTS["pathway_context"]["key_compounds"]}}), encoding="utf-8")
    out = tmp_path / "report.json"
    assert run_checks.main([str(run_dir), "--scopes", "paper,default", "--out", str(out)]) == 0
    rep = json.loads(out.read_text(encoding="utf-8"))
    t = rep["totals"]["scopes"]["default"]
    assert t["n"] == 1 and t["sbml_valid"] == 1 and t["sbgn_valid"] == 1 and t["pwml_ok"] == 1
    assert rep["totals"]["faithfulness"]["paper_quotes"]["verbatim"] >= 1
