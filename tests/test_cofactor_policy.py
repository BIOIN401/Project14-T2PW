"""C-018: verdicts, context, uncertainty, determinism, offline-ness, and a pin per
reconciled disagreement between ``admission._cofactors``/``._hubs`` and
``semantic._cofactor_names``. Fails on the base SHA: the module is new."""

import ast
import json
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
GOLD = pathlib.Path(cp.__file__).resolve().parents[1] / "bench/gold/pinned_v1.json"  # the REAL gold

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
    # (f) D13 reporters (no list had any) + D15: the EC 3.2.1.23 activity name is NOT one (GLB1)
    ("GFP", GLY, RPT, R_RPT, "fluorescent protein"),
    ("X-gal", GLY, RPT, R_RPT, "detection dye or probe"),
    ("beta-galactosidase", Ctx("glycosphingolipid degradation", "Homo sapiens"), UNK, R_VOC, ""),
)


@pytest.mark.parametrize("name,ctx,verdict,reason,family", CASES)
def test_verdict_reason_family_and_closed_vocabularies(name, ctx, verdict, reason, family):
    got = cp.classify_entity(name, ctx)
    assert (got.verdict, got.reason, got.family) == (verdict, reason, family)
    assert got.verdict in cp.VERDICTS and got.reason in cp.REASONS
    assert got.confidence in cp.CONFIDENCES


def test_context_escapes_govern_both_verdict_and_confidence():
    gly, oxp = cp.classify_entity("ATP", GLY), cp.classify_entity("ATP", OXP)
    assert (gly.verdict, oxp.verdict, gly.confidence) == (CUR, PAR, "moderate")  # ATP HAS an escape
    assert (gly.matched_context, oxp.matched_context) == ("", "atp synthesis")
    p = Ctx("pyrimidine nucleotide biosynthesis", "Escherichia coli")  # CTP/GMP synthase products
    assert {cp.classify_entity(n, p).confidence for n in ("UTP","CTP","GMP","AMP")} == {"moderate"}


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
    assert src.parent == pathlib.Path(__file__).resolve().parents[1]  # same checkout as this test
    assert [p.name for p in src.rglob("*.py") if p.name != "cofactor_policy.py"
            and "cofactor_policy" in p.read_text(encoding="utf-8", errors="ignore")] == []


def test_reporter_native_org_keys_must_be_reporter_families():
    src = pathlib.Path(cp.__file__).read_text(encoding="utf-8")  # stray key -> () -> mis-call
    typo = src.replace('"LacZ": ("escherichia coli",)', '"LacZee": ("escherichia coli",)', 1)
    assert typo != src and set(cp._REPORTER_NATIVE_ORG) <= set(cp._REPORTER)
    with pytest.raises(ValueError, match="_REPORTER_NATIVE_ORG"):  # tables + guard, no dataclasses
        exec(compile(typo.split("#: Families moved")[0], cp.__file__, "exec"), {"__name__": "t"})


def test_marker_matching_shares_the_entity_name_normalization():
    assert cp._normalize("NAD+") == cp._normalize("NAD")  # ONE fold, so "+" is not a match hazard
    assert cp._has("β-oxidation of fatty acids", ("beta-oxidation",)) == "beta-oxidation"
    nad = cp.classify_entity("NAD+", Ctx("NAD+ biosynthesis I", "Escherichia coli"))  # gold organism
    assert (nad.verdict, nad.matched_context, nad.confidence) == (PAR, "nad biosynthesis", "high")


def test_gold_expected_substrates_and_products_are_never_confidently_disposable():
    """REAL gold: no expected substrate/product may be a reporter or HIGH-confidence currency."""
    for case in json.loads(GOLD.read_text("utf-8"))["cases"]:
        ctx = Ctx(case["requested_pathway"], case["requested_organism"])
        for e in case["expected_substrates"] + case["expected_products"]:
            got = cp.classify_entity(e["name"], ctx)
            assert (got.verdict, got.confidence) != (CUR, "high") and got.verdict != RPT, got
