"""C-018: verdicts, context, uncertainty, determinism, offline-ness, and a pin per
reconciled disagreement between ``admission._cofactors``/``._hubs`` and
``semantic._cofactor_names``. Fails on the base SHA: the module is new."""

import ast
import pathlib

import pytest

from t2pw.pipeline import cofactor_policy as cp
from t2pw.pipeline.cofactor_policy import PathwayContext as Ctx

GLY, TCA = Ctx("glycolysis", "Escherichia coli"), Ctx("citric acid cycle", "Escherichia coli")
OXP = Ctx("oxidative phosphorylation (ATP synthesis)", "Escherichia coli")
CUR, HB, RPT, UNK, PAR = "cofactor_or_currency", "hub", "assay_reporter", "unknown", "participant"
R_CUR, R_HUB = "ubiquitous_currency_metabolite", "connectivity_hub_not_pathway_evidence"
R_RECON, R_SUBJ = "currency_spelling_reconciled_from_hub_list", "subject_of_the_requested_pathway"
R_RPT, R_VOC = "assay_reporter_not_pathway_member", "not_in_curated_vocabulary"

CASES = (
    # (a) one clear case per verdict, each carrying a reason
    ("ATP", GLY, CUR, R_CUR, "ATP"),
    ("acetyl-CoA", GLY, HB, R_HUB, "acetyl-CoA"),
    ("MTT", GLY, RPT, R_RPT, "viability dye"),
    ("2-oxoglutarate", TCA, PAR, R_SUBJ, "2-oxoglutarate"),
    # (c) uncertainty answers unknown; whole-name matching, never a guess
    ("lipid IV A", GLY, UNK, R_VOC, ""),
    ("cAMP", GLY, UNK, R_VOC, ""),
    ("", GLY, UNK, "no_entity_name", ""),
    ("ATP", Ctx(""), UNK, "requested_pathway_not_stated", ""),
    # (f) D2: coa-sh was the ONE name in both lists -> hub, the weaker claim
    ("CoA-SH", GLY, HB, R_HUB, "CoA"),
    ("coenzyme A", GLY, HB, R_HUB, "CoA"),
    # (f) D3/D4/D5: currency/hub spelling splits collapse onto one family
    ("orthophosphate", GLY, CUR, R_RECON, "Pi"),
    ("Pi", GLY, UNK, R_VOC, ""),  # bare "pi" dropped: also phosphatidylinositol
    ("PPi", GLY, CUR, R_RECON, "PPi"),
    ("diphosphate", GLY, CUR, R_RECON, "PPi"),
    ("NH4+", GLY, CUR, R_RECON, "NH3"),
    ("ammonia", GLY, CUR, R_RECON, "NH3"),
    # (f) D6 electron was hub-only -> currency; D7 glutamate family stays hub
    ("electron", GLY, CUR, R_RECON, "electron"),
    ("L-glutamate", GLY, HB, R_HUB, "L-glutamate"),
    ("glutamine", GLY, HB, R_HUB, "L-glutamine"),
    # (f) D8/D9/D10: unicode, separator and abbreviation spellings unify
    ("α-ketoglutarate", GLY, HB, R_HUB, "2-oxoglutarate"),
    ("acetyl CoA", GLY, HB, R_HUB, "acetyl-CoA"),
    ("THF", GLY, HB, R_HUB, "tetrahydrofolate"),
    # (f) D13: reporters, which no pre-existing list contained at all
    ("GFP", GLY, RPT, R_RPT, "fluorescent protein"),
    ("X-gal", GLY, RPT, R_RPT, "detection dye or probe"),
)


@pytest.mark.parametrize("name,ctx,verdict,reason,family", CASES)
def test_verdict_reason_family_and_closed_vocabularies(name, ctx, verdict, reason, family):
    got = cp.classify_entity(name, ctx)
    assert (got.verdict, got.reason, got.family) == (verdict, reason, family)
    assert got.verdict in cp.VERDICTS and got.reason in cp.REASONS
    assert got.confidence in cp.CONFIDENCES


def test_context_is_a_parameter_one_molecule_two_requested_pathways():
    gly, oxp = cp.classify_entity("ATP", GLY), cp.classify_entity("ATP", OXP)
    assert (gly.verdict, oxp.verdict) == (CUR, PAR)
    assert (gly.matched_context, oxp.matched_context) == ("", "atp synthesis")


def test_organism_is_a_parameter_for_a_reporter_with_a_native_host():
    lipid = "lipid A biosynthesis"
    foreign = cp.classify_entity("LacZ", Ctx(lipid, "Homo sapiens"))
    native_org = cp.classify_entity("LacZ", Ctx(lipid, "Escherichia coli"))
    native_pw = cp.classify_entity("LacZ", Ctx("lactose degradation", "Escherichia coli"))
    assert (foreign.verdict, native_org.verdict, native_pw.verdict) == (RPT, UNK, PAR)
    assert native_org.reason == "reporter_native_to_organism_pathway_unresolved"


def test_determinism_same_input_same_verdict_across_repeated_calls():
    seen = {cp.classify_entity(n, c) for _ in range(5)
            for n, c in (("ATP", GLY), ("GFP", GLY), ("Pi", OXP))}
    assert len(seen) == 3


def test_leaf_module_has_no_network_db_or_llm_path_and_is_dead_code():
    mods = set()
    for node in ast.walk(ast.parse(pathlib.Path(cp.__file__).read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            mods.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            mods.add((node.module or "").split(".")[0])
    assert mods == {"__future__", "re", "dataclasses"}  # no requests/urllib/sqlite3/openai/t2pw
    src = pathlib.Path(cp.__file__).resolve().parents[2]
    assert [p.name for p in src.rglob("*.py") if p.name != "cofactor_policy.py"
            and "cofactor_policy" in p.read_text(encoding="utf-8", errors="ignore")] == []
