"""ORCH-803: the paper's own abbreviation definition unifies an entity (PO ruling 2026-11-04).

G9 labelling:
- CORRECTIONS (each fails behaviourally on base 563d39db, passes at tip). These are the three
  ORCH-801 defects, each on the paper's own sentence:
  ``test_g9_*``.
- NEGATIVE TWINS (pass on base and tip; they pin what must never pair or be admitted):
  ``test_twin_*``.
No model, network or database is called.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from t2pw.mapping.map_ids import _name_gate_verdict  # noqa: E402
from t2pw.pipeline import extraction_diagnostics as diag  # noqa: E402
from t2pw.pipeline.name_keys import strict_key  # noqa: E402
from t2pw.pipeline.paper_lexicon import PaperLexicon, abbreviation_pairs  # noqa: E402
from t2pw.pipeline.stage_one_ensemble import merge_draws, run_ensemble  # noqa: E402

# The three papers' own sentences, verbatim from each ORCH-801 leg's 01_source_text.txt.
TB = ("However, similar levels of uridine (Urd) ( Fig 4B and S5B Fig ) or thymidine (dThd)"
      "( Fig 4C and S5C Fig ) did not restore growth. Thymidine kinase (TK) phosphorylates dThd.")
CDADC1 = ("enzyme, thymidylate synthase (TYMS), which converts deoxyuridine monophosphate (dUMP) "
          "to deoxythymidine monophosphate, subsequently")
SALICYLIC = ("cinnamoyl-CoA hydratase/dehydrogenase (CHD) and 3-ketoacyl-CoA thiolase (KAT), has been "
             "shown to convert trans -CA to benzoyl-CoA (BA-CoA) and subsequently to BA in several "
             "plant species")

DUMP_IDS = {"chebi": "CHEBI:17622", "pathbank_compound_id": "1087"}


def _pairs(text: str) -> Dict[str, str]:
    return {s: l for s, l, _p in abbreviation_pairs(text)}


def _draw(reactions: List[Dict[str, Any]], compounds: List[str], **processes: Any) -> Dict[str, Any]:
    return {"entities": {"compounds": [{"name": c} for c in compounds]},
            "processes": {"reactions": reactions, **processes}}


def _compound_names(payload: Dict[str, Any]) -> List[str]:
    return sorted(strict_key(r["name"]) for r in payload["entities"]["compounds"])


@pytest.fixture
def run_scope():
    """A fresh run recorder before and after, as ``activate()`` gives each real run."""
    diag.activate()
    yield
    diag.activate()


def _pool(paper: str) -> None:
    """Run a pooled Stage 1 over ``paper`` (two trivial draws), as production does."""
    run_ensemble(lambda: (_draw([], []), []), 2, failure=RuntimeError, paper_text=paper)


def _gate(entity: str, record_name: str) -> Dict[str, Any]:
    candidate = {"name": record_name, "short_name": record_name, "pathbank_compound_id": "1087",
                 "mapped_ids": dict(DUMP_IDS)}
    return _name_gate_verdict(entity, candidates=[candidate], mapped_ids=dict(DUMP_IDS), kind="compound")


# ------------------------------------------------------------------------- G9 corrections


def test_g9_tb_dthd_is_the_papers_thymidine() -> None:
    assert _pairs(TB).get("dThd") == "thymidine"  # base: no dThd pair at all


def test_g9_tb_payload_one_node_and_the_duplicate_reaction_merged() -> None:
    d1 = _draw([{"name": "HD 5'-NT", "inputs": ["dTMP"], "outputs": ["dThd"], "enzymes": ["HD-domain 5'-nucleotidase"],
                 "evidence": "dephosphorylation of dTMP"}], ["dTMP", "dThd"])
    d2 = _draw([{"name": "5'-NT", "inputs": ["dTMP"], "outputs": ["thymidine"], "enzymes": ["HD-domain 5'-nucleotidase"],
                 "evidence": "dephosphorylation of dTMP"}], ["dTMP", "thymidine"])
    merged, _report = merge_draws([d1, d2], paper_text=TB)
    reactions = merged["processes"]["reactions"]
    assert len(reactions) == 1, reactions  # base: 2 (dTMP -> dThd beside dTMP -> thymidine)
    assert [strict_key(n) for n in reactions[0]["outputs"]] == ["thymidine"]
    assert _compound_names(merged) == ["dtmp", "thymidine"]


def test_g9_cdadc1_gate_keeps_the_papers_dump_record_both_directions(run_scope) -> None:
    _pool(CDADC1)
    long_side = _gate("deoxyuridine monophosphate", "dUMP")  # the ORCH-801 refusal
    assert long_side["verdict"] == "keep", long_side  # base: reject, no_shared_meaningful_token
    assert long_side["reason"] == "paper_abbreviation_identity"
    assert long_side["paper_pair"] == ["dUMP", "deoxyuridine monophosphate"]
    assert _gate("dUMP", "deoxyuridine monophosphate")["verdict"] == "keep"


def test_g9_salicylic_ba_coa_is_the_papers_benzoyl_coa() -> None:
    # base: 'been shown to convert trans -CA to benzoyl-CoA'
    assert _pairs(SALICYLIC).get("BA-CoA") == "benzoyl-CoA"


def test_g9_salicylic_cargo_only_ba_coa_joins_benzoyl_coa() -> None:
    d1 = _draw([{"name": "KAT", "inputs": ["benzoylacetyl-CoA"], "outputs": ["benzoyl-CoA"], "evidence": "KAT"}],
               ["benzoylacetyl-CoA", "benzoyl-CoA"])
    d2 = _draw([], ["BA-CoA"], transports=[{"name": "export", "cargo": ["BA-CoA"], "from": "peroxisome",
                                             "to": "cytosol", "evidence": "BA-CoA may be exported"}])
    merged, _report = merge_draws([d1, d2], paper_text=SALICYLIC)
    assert _compound_names(merged) == ["benzoyl coa", "benzoylacetyl coa"]  # base: + 'ba coa'
    assert [strict_key(c) for c in merged["processes"]["transports"][0]["cargo"]] == ["benzoyl coa"]


# ------------------------------------------------------------------------- negative twins


@pytest.mark.parametrize("text", [
    "thymidine (Fig. 2) was added",
    "thymidine (n = 3) was added",
    "benzoyl-CoA (Supplementary Table 2) was measured",
    "the dCMP deaminase structure (PDB 1ABC) shows",
    "dephosphorylates thymidine (dTMP-dependent) slowly",
])
def test_twin_non_definitions_pair_nothing(text: str) -> None:
    assert abbreviation_pairs(text) == []


def test_twin_protein_abbreviation_never_names_the_compound(run_scope) -> None:
    assert _pairs(TB)["TK"] == "Thymidine kinase"
    assert PaperLexicon(TB).long_form("TK") == ("Thymidine kinase", "long (SF)")
    _pool(TB)
    assert _gate("thymidine", "TK")["verdict"] == "reject"


@pytest.mark.parametrize("text", [
    "thymidine (deoxycytidine) levels rose",           # two molecules, not an abbreviation
    "the product BA-CoA (CA-CoA) was not detected",    # two abbreviations of two molecules
    "benzoyl-CoA (cinnamoyl-CoA) ratios",
    "L-Glu (D-Glu) was inactive",
])
def test_twin_two_molecules_written_side_by_side_pair_nothing(text: str) -> None:
    assert abbreviation_pairs(text) == []


def test_twin_gate_needs_this_runs_paper(run_scope) -> None:
    assert _gate("deoxyuridine monophosphate", "dUMP")["verdict"] == "reject"  # N=1: no dictionary
    _pool(CDADC1)
    diag.activate()  # the next paper's run
    assert _gate("deoxyuridine monophosphate", "dUMP")["verdict"] == "reject"


def test_twin_deoxy_fallback_never_overrides_a_full_match() -> None:
    assert _pairs("cells fed deoxyuridine (dUrd) grew")["dUrd"] == "deoxyuridine"
    assert "dCas9" not in _pairs("a catalytically inactive Cas9 (dCas9) was used")
