"""NEW ACCEPTANCE TESTS (ORCH-747) -- draft to gold assembly, and the blind/hinted compare.

A genuinely NEW capability: before this commit no code turned a drafter output into a
loadable gold supplement, and no code measured whether handing a drafter the lab's curated
record contaminates what it reports finding in the paper. Labelled new tests under G9; no
base-SHA failure is claimed.

Database-free. Cases needing the pinned paper text skip when it is absent, because that
text lives outside the repository by design (docs/evaluation_v2/README.md section 1).
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from t2pw.bench.eval_v2.gold import load_fulltext, load_supplement, verify_quotes
from t2pw.bench.eval_v2.paper_gold import (DRAFT_SCHEMA, SUPPLEMENT_SCHEMA, _same_statement,
                                           assemble, compare, verify)

ROOT = Path(__file__).resolve().parents[1]
GOLD_DIR = ROOT / "docs/evaluation_v2/pathbank_gold"
BLIND = GOLD_DIR / "PMC319880__draft_blind_final.json"
SOURCE = GOLD_DIR / "PMC319880__source.json"
FULLTEXT = Path("C:/t/eval-fulltext")


def _load(p: Path) -> dict:
    if not p.is_file():
        pytest.skip(f"{p} not committed yet")
    return json.loads(p.read_text(encoding="utf-8"))


@pytest.fixture
def draft() -> dict:
    return _load(BLIND)


@pytest.fixture
def manifest() -> dict:
    return _load(SOURCE)


def test_assemble_emits_the_schema_the_loader_reads(draft, manifest):
    out = assemble(draft, manifest)
    assert out["schema"] == SUPPLEMENT_SCHEMA
    assert draft["schema"] == DRAFT_SCHEMA
    assert list(out["cases"]) == ["PMC319880"]
    assert "never by merging it into the pinned file" in out["description"]


def test_a_model_can_never_certify_its_own_exhaustiveness(draft, manifest):
    """The reason the human step exists. Even a draft claiming completeness is written out
    as not complete, so no precision, recall or F1 can be computed from it."""
    lying = copy.deepcopy(draft)
    lying["case"]["supported_reactions_complete"] = True
    case = assemble(lying, manifest)["cases"]["PMC319880"]
    assert case["supported_reactions_complete"] is False
    assert "NOT CERTIFIED" in case["completeness_status"]


def test_assemble_refuses_an_incomplete_source_manifest(draft, manifest):
    for field in ("pmcid", "text_sha256", "fetch_date"):
        broken = {k: v for k, v in manifest.items() if k != field}
        with pytest.raises(ValueError) as err:
            assemble(draft, broken)
        assert field in str(err.value)


def test_assemble_refuses_anything_that_is_not_a_drafter_output(manifest):
    with pytest.raises(ValueError):
        assemble({"schema": "something/1", "case_id": "PMC1", "case": {}}, manifest)


def test_a_binding_row_in_a_reaction_bucket_is_rejected_by_the_loader(draft, manifest, tmp_path):
    """The defect this module caught in the drafter prompt. A binding has no product side,
    so it belongs in interactions. Put back into a two-sided bucket, the file must fail to
    load rather than quietly score as a reaction."""
    bad = copy.deepcopy(draft)
    bad["case"]["off_pathway_reactions"].append(
        {"id": "OX", "label": "P2 binds the B site", "inputs": [{"name": "P2"}],
         "outputs": [], "quote": "binds"})
    p = tmp_path / "bad.json"
    p.write_text(json.dumps(assemble(bad, manifest)), encoding="utf-8")
    with pytest.raises(Exception) as err:
        load_supplement(p)
    assert "defining participant" in str(err.value)


def test_the_assembled_blind_gold_loads_and_every_quote_verifies(draft, manifest, tmp_path):
    p = tmp_path / "gold.json"
    p.write_text(json.dumps(assemble(draft, manifest)), encoding="utf-8")
    case = load_supplement(p)["PMC319880"]
    assert case.complete is False and case.reactions
    if not (FULLTEXT / "PMC319880.txt").is_file():
        pytest.skip("pinned paper text is not on this machine")
    res = verify(p, FULLTEXT)          # raises on a text sha256 mismatch
    assert res["ok"] is True, res
    assert verify_quotes(case, load_fulltext(case, FULLTEXT)) == []


def test_a_text_that_is_not_the_pinned_one_is_refused(draft, manifest, tmp_path):
    """The hash pin is what makes a quote-verified gold reproducible. This paper went
    through three extraction revisions, so scoring against the wrong one must be
    impossible rather than merely discouraged."""
    p = tmp_path / "gold.json"
    p.write_text(json.dumps(assemble(draft, manifest)), encoding="utf-8")
    (tmp_path / "PMC319880.txt").write_text("not the paper", encoding="utf-8")
    with pytest.raises(Exception) as err:
        verify(p, tmp_path)
    assert "not the text the labels were written against" in str(err.value)


# -- compare ---------------------------------------------------------------

def test_compare_against_itself_matches_every_row_and_reports_clean(draft):
    res = compare(draft, draft)
    for bucket, got in res["buckets"].items():
        assert got["matched"] == got["blind"] == got["hinted"], (bucket, got)
        assert got["blind_only"] == [] and got["hinted_only"] == []
    assert res["contamination_check"]["verdict"].startswith("clean")


def test_compare_flags_a_reaction_only_the_hinted_pass_claims(draft):
    """The contamination this measure exists to catch: a drafter handed the curated record
    starts finding the curator's processes in a paper that does not state them."""
    hinted = copy.deepcopy(draft)
    hinted["case"]["reactions"].append(
        {"id": "RX", "label": "invented step", "inputs": [{"name": "alpha"}],
         "outputs": [{"name": "omega"}], "quote": "nowhere in the paper"})
    res = compare(draft, hinted)
    assert res["buckets"]["reactions"]["hinted_only"] == ["invented step"]
    assert res["contamination_check"]["verdict"].startswith("REVIEW")


@pytest.mark.parametrize("a,b,bucket,expected", [
    # One sentence is the strongest evidence that two rows are one claim, even when the
    # two passes name the participants differently.
    ({"quote": "X binds Y.", "source": [{"name": "polymerase"}], "target": [{"name": "Pc"}]},
     {"quote": "x  binds   y", "source": [{"name": "RNA polymerase"}],
      "target": [{"name": "promoter"}]},
     "interactions", True),
    # Opposite sign on the same pair is a disagreement, never a match. This is the shape of
    # the curated edge the paper contradicts, so it must never collapse into agreement.
    ({"effect": "activation", "source": [{"name": "araO1"}], "target": [{"name": "Pc"}]},
     {"effect": "inhibition", "source": [{"name": "araO1"}], "target": [{"name": "Pc"}]},
     "interactions", False),
    # Overlapping names on each populated side, extra cofactor on one.
    ({"inputs": [{"name": "chorismate"}], "outputs": [{"name": "isochorismate"}]},
     {"inputs": [{"name": "chorismate"}, {"name": "magnesium"}],
      "outputs": [{"name": "isochorismate"}]},
     "reactions", True),
    # No shared participant at all.
    ({"inputs": [{"name": "alpha"}], "outputs": [{"name": "beta"}]},
     {"inputs": [{"name": "gamma"}], "outputs": [{"name": "delta"}]},
     "reactions", False),
])
def test_statement_matching_tolerates_naming_but_not_disagreement(a, b, bucket, expected):
    assert _same_statement(a, b, bucket) is expected
