"""NEW ACCEPTANCE TESTS (ORCH-742c, evaluation v2) -- gold supplement and strict matcher.

These cover a genuinely NEW capability (a separately versioned gold supplement and a
new matcher). They claim no correction of pre-existing behaviour, so per G9 they carry
no base-SHA failure: at the base these modules do not exist.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

from t2pw.bench.eval_v2 import matcher as M
from t2pw.bench.eval_v2.gold import (SUPPLEMENT_PATH, GoldV2Error, load_fulltext, load_supplement,
                                     verify_quotes)
from t2pw.bench.goldset import load_gold_set, pinned_gold_set_path

ROOT = Path(__file__).resolve().parents[1]
# D-103 (ORCH-744): PMC12657337 stays labelled but is NOT certified complete (bio-audit
# B-R3-1), so it is in LABELLED and not in COMPLETE. Its quotes are still verified.
COMPLETE = {"PMC12096016", "PMC12312563", "PMC12421875"}
LABELLED = COMPLETE | {"PMC12657337"}


def _committed(pattern: str):
    out = subprocess.run(["git", "ls-files", pattern], cwd=ROOT, capture_output=True, text=True)
    return out.stdout.split()


def test_supplement_is_invisible_to_every_existing_gold_reader():
    """pinned_v1 stays the only file in bench/gold/, so no glob over gold/*.json and
    no default loader can pick the supplement up."""
    gold_dir = pinned_gold_set_path().parent
    assert sorted(p.name for p in gold_dir.iterdir() if p.is_file()) == ["pinned_v1.json"]
    assert SUPPLEMENT_PATH.parent != gold_dir
    assert load_gold_set().source_path == str(pinned_gold_set_path())


def test_supplement_loads_three_complete_cases_all_keyed_to_pinned_v1():
    cases = load_supplement()
    pinned = {c.paper_id for c in load_gold_set().cases}
    assert {pid for pid, c in cases.items() if c.complete} == COMPLETE
    assert set(cases) == LABELLED and set(cases) <= pinned
    for c in cases.values():
        assert c.reactions and c.source["xml_sha256"] and c.source["text_sha256"]
        assert c.source["source_url"].startswith("https://www.ebi.ac.uk/europepmc/")


@pytest.mark.parametrize("pid", sorted(LABELLED))
def test_every_in_scope_quote_is_verbatim_in_the_committed_paper_text(pid):
    """Reproducibility from the repository alone: the scored reactions' quotes are in
    the git-tracked 01_source_text.txt of the most recent committed run."""
    case = load_supplement()[pid]
    texts = sorted(f for f in _committed("*01_source_text.txt") if f"/{pid}" in f)
    assert texts, pid
    missing = verify_quotes(case, (ROOT / texts[-1]).read_text(encoding="utf-8", errors="replace"))
    in_scope = {r.id for r in case.reactions}
    assert not [m for m in missing if m[0] in in_scope], missing


def test_every_quote_is_verbatim_in_the_pinned_fetched_full_text():
    """Runs only where the EuropePMC texts were fetched (kept OUTSIDE the repo)."""
    ft_dir = Path(os.environ.get("T2PW_EVAL_FULLTEXT_DIR", "C:/t/eval-fulltext"))
    cases = load_supplement()
    if not all((ft_dir / f"{pid}.txt").is_file() for pid in LABELLED):
        pytest.skip(f"fetched full texts not present in {ft_dir}")
    for pid in LABELLED:
        text = load_fulltext(cases[pid], ft_dir)  # raises on a sha256 mismatch
        assert verify_quotes(cases[pid], text) == [], pid


@pytest.mark.parametrize("claim,gold,old_credits", [
    ("menaquinone-7", "menaquinone", True),
    ("Kdo2-lipid A", "lipid A", True),
    ("acetyl coenzyme A", "coenzyme A", True),
    ("DHNA-CoA", "DHNA", True),
    ("acetyl-CoA", "CoA", False),
    ("PGE1", "PGE2", False),
    ("2,5-DHB", "2,3-DHB", False),
    ("beta-ketoglutarate", "alpha-ketoglutarate", False),
    ("demethylmenaquinone-8", "demethylmenaquinone-7", False),
])
def test_strict_matcher_never_credits_a_congener(claim, gold, old_credits):
    assert not M.matches(claim, gold)
    assert M.lenient_matches(claim, gold) is old_credits  # the old column, same inputs


@pytest.mark.parametrize("a,b", [
    ("2,3-dihydroxybenzoic acid", "2,3-dihydroxybenzoate"),
    ("α-ketoglutarate", "alpha-ketoglutarate"),
    ("farnesyl pyrophosphate", "farnesyl diphosphate"),
    ("NAD+", "NAD"),
    ("  Chorismate ", "chorismate"),
    ("2-oxoglutaric acid", "2-oxoglutarate"),
])
def test_strict_matcher_documented_equivalences(a, b):
    assert M.matches(a, b)


def test_parenthetical_and_complex_rules():
    # The inside of a parenthetical corroborates, never credits on its own (repair B2).
    assert not M.matches("oxidoreductase (entA)", "EntA")
    assert M.name_verdict("oxidoreductase (entA)", M.term_keys("EntA")) == "conflict"
    assert M.matches("2,3-dihydroxybenzoic acid (2,3-DHB)", "2,3-DHB", ("2,3-dihydroxybenzoate",))
    # Without the declared alias the outer name is not known to be 2,3-DHB: hand review.
    assert M.name_verdict("2,3-dihydroxybenzoic acid (2,3-DHB)", M.term_keys("2,3-DHB")) == "conflict"
    complexes = {M.strict_key("Isochorismate synthase"): {"components": [{"name": "EntC", "gene_name": "entC"}]}}
    assert M.term_keys("EntC") & M.enzyme_keys("Isochorismate synthase", complexes)
    assert "ente" in M.enzyme_keys("EntE complex")
    # The complex rule is ENZYME-only: participants never lose 'complex'.
    assert not M.matches("EntE complex", "EntE")


@pytest.mark.parametrize("candidate,verdict", [
    ("heptanoyl pyrophosphate (HepPP)", "conflict"),   # C7 acyl diphosphate: a different molecule
    ("heptaprenyl diphosphate (HepPP)", "match"),      # outer matches; inside corroborates
    ("heptaprenyl pyrophosphate", "match"),
    ("(HepPP)", "match"),                              # no outer name to contradict
    ("farnesyl diphosphate (FPP)", ""),                # neither part names the term
])
def test_parenthetical_inside_never_overrides_a_contradicting_outer_name(candidate, verdict):
    """NEW ACCEPTANCE TEST, regression fixture for review finding B2 (ORCH-742c repair 1):
    the gold R1 product of PMC12657337 is heptaprenyl pyrophosphate (alias HepPP)."""
    wanted = M.term_keys("heptaprenyl pyrophosphate", ("HepPP", "heptaprenyl diphosphate"))
    assert M.name_verdict(candidate, wanted) == verdict


def test_mentions_is_token_bounded():
    text = M.text_key("glucose bisphosphate and OSB-CoA were measured")
    assert M.mentions("OSB-CoA", text)
    assert not M.mentions("OSB", M.text_key("glucosebisphosphate"))


def _write(tmp_path, cases):
    p = tmp_path / "s.json"
    p.write_text(json.dumps({"cases": cases}), encoding="utf-8")
    return p


SRC = {"pmcid": "PMC1", "source_url": "u", "fetch_date": "d", "xml_sha256": "x", "text_sha256": "t"}
RX = {"id": "R1", "inputs": [{"name": "a"}], "outputs": [{"name": "b"}], "quote": "a to b"}


def test_loader_refuses_a_reaction_without_a_quote(tmp_path):
    with pytest.raises(GoldV2Error, match="quote"):
        load_supplement(_write(tmp_path, {"PMC1": {"supported_reactions_complete": True, "source": SRC,
                                                    "reactions": [{**RX, "quote": ""}]}}))


def test_loader_refuses_an_implicit_completeness_flag(tmp_path):
    with pytest.raises(GoldV2Error, match="explicit boolean"):
        load_supplement(_write(tmp_path, {"PMC1": {"source": SRC, "reactions": [RX]}}))


def test_loader_refuses_a_complete_case_with_no_reaction(tmp_path):
    with pytest.raises(GoldV2Error, match="negative"):
        load_supplement(_write(tmp_path, {"PMC1": {"supported_reactions_complete": True, "source": SRC,
                                                    "reactions": []}}))
