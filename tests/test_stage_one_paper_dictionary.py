"""NEW ACCEPTANCE TESTS for ORCH-754a: the paper as the dictionary of a pooled extraction.

G9 labelling: everything here is a NEW capability ruled by the product owner on 2026-10-19
(D-104) -- Schwartz-Hearst abbreviation pairs, paper-relative typo snapping, the long-form
rewrite, the per-row support count, the support line in the text summary, and the
post-mapping database-identity merge -- so these are explicitly labelled new acceptance
tests and claim no base-commit failure. The CORRECTIONS of delivered behaviour (the
chlorophyll inflation, finding 1's chimeric rows) live in ``tests/test_stage_one_ensemble.py``
with their base-module proofs.

No model, network or database is called anywhere here.
"""

from __future__ import annotations

import json
import os
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from t2pw.pipeline import reaction_summary  # noqa: E402
from t2pw.pipeline.draft_graph import build_draft_graph  # noqa: E402
from t2pw.pipeline.paper_lexicon import PaperLexicon, abbreviation_pairs  # noqa: E402
from t2pw.pipeline.pipeline import clean_stage_one_with_report  # noqa: E402
from t2pw.pipeline.stage_one_ensemble import (  # noqa: E402
    ENV_VAR,
    IDENTITY_REPORT_KEY,
    SUPPORT_KEY,
    merge_draws,
    merge_on_mapped_identity,
)

PAPER = (
    "Glu-tRNA is reduced to Glu 1-semialdehyde (GSA) by Glu-tRNA reductase (GluTR). "
    "GSA is converted to 5-aminolevulinic acid (ALA). Two ALA form porphobilinogen (PBG). "
    "HMB is cyclised into the first macrocycle, uroporphyrinogen (Urogen) III, by a synthase. "
    "The plastid enzyme GSAT (glutamate semialdehyde aminotransferase) is PLP dependent. "
    "Pchlide oxidoreductase (POR) and later protochlorophyllide oxidoreductase (POR) act. "
    "Energy comes from adenosine triphosphate (ATP). A protein-protein interaction (PPI) "
    "screen. Heme oxygenase releases biliverdin from heme. "
    "Hexokinase gives glucose-6-phosphate; alpha-ketoglutarate, sulfate, maltose, ferrocyanide "
    "and the synthetic pair glucuronate and glacuronate also occur."
)


def _lexicon() -> PaperLexicon:
    return PaperLexicon(PAPER)


def _reactions(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    return payload["processes"]["reactions"]


# --------------------------------------------------------------------------- rule 1


def test_schwartz_hearst_finds_long_short_swapped_and_shared_locant_pairs() -> None:
    pairs = {(short, long, pattern) for short, long, pattern in abbreviation_pairs(PAPER)}
    assert ("GSA", "Glu 1-semialdehyde", "long (SF)") in pairs
    assert ("ALA", "5-aminolevulinic acid", "long (SF)") in pairs
    assert ("GSAT", "glutamate semialdehyde aminotransferase", "SF (long)") in pairs
    assert ("Urogen III", "uroporphyrinogen III", "shared_locant") in pairs


def test_an_abbreviation_with_two_long_forms_bridges_nothing() -> None:
    lexicon = _lexicon()
    assert "POR" not in lexicon.pairs
    assert lexicon.ambiguous["POR"] == ["Pchlide oxidoreductase", "protochlorophyllide oxidoreductase"]
    assert lexicon.canonical("POR") == ("POR", [], "")
    assert lexicon.canonical("Pchlide oxidoreductase (POR)")[1] == []


def test_short_forms_match_case_sensitively() -> None:
    lexicon = _lexicon()
    assert lexicon.canonical("PPI")[0] == "protein-protein interaction"
    assert lexicon.canonical("PPi") == ("PPi", [], "")  # pyrophosphate is not the screen


@pytest.mark.parametrize("name", ["GSA", "Glu 1-semialdehyde (GSA)", "GSA (Glu 1-semialdehyde)"])
def test_every_paper_defined_form_becomes_the_long_form(name: str) -> None:
    assert _lexicon().canonical(name) == ("Glu 1-semialdehyde", ["abbreviation"], "long (SF)")


def test_a_draw_written_pair_the_paper_does_not_define_bridges_nothing() -> None:
    assert _lexicon().canonical("protoheme (heme b)") == ("protoheme (heme b)", [], "")
    assert _lexicon().canonical("Glu 2-semialdehyde (GSA)")[1] == []


# --------------------------------------------------------------------------- rule 2


@pytest.mark.parametrize("typo, fixed", [("porphobilogen", "porphobilinogen"), ("bilverdin", "biliverdin")])
def test_a_misspelling_snaps_to_the_papers_spelling(typo: str, fixed: str) -> None:
    assert _lexicon().snap(typo) == fixed


@pytest.mark.parametrize("name, why", [
    ("glucose-1-phosphate", "a locant differs"),
    ("ADP", "under 6 characters (ATP/ADP)"),
    ("beta-ketoglutarate", "alpha/beta"),
    ("sulfite", "suffix class (-ate/-ite)"),
    ("maltase", "suffix class (-ose/-ase)"),
    ("ferricyanide", "oxidation state (ferro/ferri)"),
    ("biliverdin", "already in the paper"),
    ("glicuronate", "two candidates"),
    ("D-glucose-6-phosphate", "a word the paper lacks that cannot snap (D)"),
])
def test_unsafe_snaps_never_happen(name: str, why: str) -> None:
    assert _lexicon().snap(name) is None, why


# --------------------------------------------------------------------------- rules 4, 5, 6 in the merge


def _draw(reactions: List[Dict[str, Any]], compounds: List[str] = (), **extra: Any) -> Dict[str, Any]:
    return {"entities": {"compounds": [{"name": c} for c in compounds]},
            "processes": {"reactions": reactions}, **extra}


def test_abbreviation_spellings_pool_into_one_long_form_row() -> None:
    d1 = _draw([{"name": "r1", "inputs": ["GSA"], "outputs": ["ALA"], "evidence": "GSA is converted to ALA"}],
               ["GSA", "ALA"],
               element_locations={"compound_locations": [{"compound": "GSA", "biological_state": "plastid"}]})
    d2 = _draw([{"name": "r2", "inputs": ["Glu 1-semialdehyde (GSA)"], "outputs": ["5-aminolevulinic acid (ALA)"],
                 "evidence": "Glu 1-semialdehyde (GSA) gives 5-aminolevulinic acid (ALA)"}],
               ["Glu 1-semialdehyde (GSA)", "5-aminolevulinic acid (ALA)"])
    d2["processes"]["interactions"] = [{"entity_1": "GSA", "relationship": "inhibits", "entity_2": "GluTR"}]
    merged, report = merge_draws([d1, d2], paper_text=PAPER)

    [row] = _reactions(merged)
    assert (row["inputs"], row["outputs"]) == (["Glu 1-semialdehyde"], ["5-aminolevulinic acid"])
    assert row["evidence"] == "GSA is converted to ALA"  # verbatim, never rewritten
    assert row[SUPPORT_KEY] == {"draws": [1, 2], "of": 2}
    assert [c["name"] for c in merged["entities"]["compounds"]] == ["Glu 1-semialdehyde", "5-aminolevulinic acid"]
    assert merged["element_locations"]["compound_locations"][0]["compound"] == "Glu 1-semialdehyde"
    assert merged["processes"]["interactions"][0]["entity_1"] == "Glu 1-semialdehyde"
    [rec] = report["buckets"]["/processes/reactions"]["rows"]
    assert rec["support"] == "2/2" and rec["merges"] == [{"draw": 2, "name": "r2", "bridged_by": ["abbreviation"]}]
    spellings = {(r["draw"], r["from"]) for r in report["names_rewritten"]}
    assert {(1, "GSA"), (1, "ALA"), (2, "Glu 1-semialdehyde (GSA)"), (2, "5-aminolevulinic acid (ALA)")} <= spellings
    assert d1["processes"]["reactions"][0]["inputs"] == ["GSA"]  # the draw itself is never touched


def test_a_typo_bridge_is_recorded_as_typo() -> None:
    d1 = _draw([{"name": "a", "inputs": ["porphobilinogen"], "outputs": ["X1"], "evidence": "q"}])
    d2 = _draw([{"name": "b", "inputs": ["porphobilogen"], "outputs": ["X1"], "evidence": "q"}])
    _merged, report = merge_draws([d1, d2], paper_text=PAPER)
    [rec] = report["buckets"]["/processes/reactions"]["rows"]
    assert rec["merges"][0]["bridged_by"] == ["typo"]


def test_currency_is_never_rewritten_even_when_the_paper_defines_it() -> None:
    merged, _ = merge_draws([_draw([{"name": "a", "inputs": ["X1", "ATP"], "outputs": ["X2"], "evidence": "q"}]),
                             _draw([])], paper_text=PAPER)
    assert _reactions(merged)[0]["inputs"] == ["X1", "ATP"]


def test_a_one_draw_row_is_kept_and_flagged_for_review() -> None:
    merged, report = merge_draws([_draw([{"name": "only", "inputs": ["X1"], "outputs": ["X2"], "evidence": "q"}]),
                                  _draw([]), _draw([])], paper_text=PAPER)
    assert _reactions(merged)[0][SUPPORT_KEY] == {"draws": [1], "of": 3}
    [rec] = report["buckets"]["/processes/reactions"]["rows"]
    assert rec["support"] == "1/3" and rec["review_required"] == "found by one draw only"


def test_without_paper_text_no_name_is_rewritten() -> None:
    merged, report = merge_draws([_draw([{"name": "a", "inputs": ["GSA"], "outputs": ["ALA"], "evidence": "q"}]),
                                  _draw([])])
    assert _reactions(merged)[0]["inputs"] == ["GSA"]
    assert report["paper_lexicon"]["status"] == "not_evaluated" and report["names_rewritten"] == []


def test_a_withheld_donation_keeps_the_donors_quote_in_the_side_file() -> None:
    first = {"name": "r", "inputs": ["A"], "outputs": ["B"], "evidence": "A becomes B"}
    later = {"name": "r again", "inputs": ["A", "NADPH"], "outputs": ["B"], "evidence": "NADPH reduces A to B"}
    _merged, report = merge_draws([_draw([first]), _draw([later])])
    [rec] = report["buckets"]["/processes/reactions"]["rows"]
    assert rec["cofactors_withheld"] == [{"cofactor": "NADPH", "side": "inputs", "donor_evidence": "NADPH reduces A to B"}]


def test_a_row_kept_apart_records_why() -> None:
    a = {"name": "a", "inputs": ["X1", "NAD+"], "outputs": ["X2"], "evidence": "q"}
    b = {"name": "b", "inputs": ["X1", "NADP+"], "outputs": ["X2"], "evidence": "q"}
    _merged, report = merge_draws([_draw([a]), _draw([b])])
    rows = report["buckets"]["/processes/reactions"]["rows"]
    assert [r["name"] for r in rows] == ["a", "b"]
    assert "different nicotinamide cofactors" in rows[1]["kept_apart_from"][0]["reason"]


def test_a_row_compatible_with_two_conflicting_rows_cannot_join_them() -> None:
    plain = {"name": "plain", "inputs": ["X1"], "outputs": ["X2"], "evidence": "q"}
    nad = {"name": "nad", "inputs": ["X1", "NAD+"], "outputs": ["X2"], "evidence": "q"}
    nadp = {"name": "nadp", "inputs": ["X1", "NADP+"], "outputs": ["X2"], "evidence": "q"}
    merged, _ = merge_draws([_draw([plain]), _draw([nad]), _draw([nadp])])
    assert [r["name"] for r in _reactions(merged)] == ["plain", "nadp"]  # nad joined plain; nadp may not


def test_an_undefined_state_cannot_be_compared_and_fails_closed() -> None:
    states = [{"name": "plastid state", "subcellular_location": "plastid"},
              {"name": "plastidic state", "subcellular_location": "plastids"}]
    a = {"name": "a", "inputs": ["X1"], "outputs": ["X2"], "biological_state": "plastid state", "evidence": "q"}
    b = {"name": "b", "inputs": ["X1"], "outputs": ["X2"], "biological_state": "plastidic state", "evidence": "q"}
    merged, _ = merge_draws([_draw([a], biological_states=states), _draw([b], biological_states=states)])
    assert [r["name"] for r in _reactions(merged)] == ["a"]  # same compartment, plural folded
    merged, _ = merge_draws([_draw([a], biological_states=states), _draw([b])])
    assert [r["name"] for r in _reactions(merged)] == ["a", "b"]  # draw 2 never defines its state


# --------------------------------------------------------------------------- rule 5 downstream


def test_cleaning_carries_the_support_count_and_only_its_exact_shape() -> None:
    row = {"name": "r", "inputs": ["A"], "outputs": ["B"], "evidence": "q"}
    payload = {"entities": {}, "processes": {"reactions": [
        {**row, SUPPORT_KEY: {"draws": [1, 3], "of": 3}},
        {**row, "name": "r2", SUPPORT_KEY: {"draws": [1], "of": 1}},      # N=1 can never carry it
        {**row, "name": "r3", SUPPORT_KEY: "2/3"},
    ]}}
    cleaned, _ = clean_stage_one_with_report(payload)
    assert [r.get(SUPPORT_KEY) for r in _reactions(cleaned)] == [{"draws": [1, 3], "of": 3}, None, None]


def test_the_text_summary_shows_support_and_flags_one_draw_rows() -> None:
    payload = {"entities": {}, "processes": {"reactions": [
        {"name": "two", "inputs": ["A"], "outputs": ["B"], SUPPORT_KEY: {"draws": [1, 2], "of": 3}},
        {"name": "one", "inputs": ["B"], "outputs": ["C"], SUPPORT_KEY: {"draws": [3], "of": 3}},
    ]}}
    graph = build_draft_graph(payload)
    text = reaction_summary.generate_reaction_summary(graph, {}, payload=payload)
    assert "  Support:   2/3 draws\n" in text
    assert "  Support:   1/3 draws -- REVIEW: only one draw produced this reaction" in text
    # Without the key -- every N=1 run -- not one line changes.
    bare = deepcopy(payload)
    for r in _reactions(bare):
        r.pop(SUPPORT_KEY)
    assert reaction_summary.generate_reaction_summary(build_draft_graph(bare), {}, payload=bare) == \
        reaction_summary.generate_reaction_summary(build_draft_graph(bare), {})
    assert "Support:" not in reaction_summary.generate_reaction_summary(build_draft_graph(bare), {})


# --------------------------------------------------------------------------- rule 3


def _row(name: str, ids: Any = None, **meta: Any) -> Dict[str, Any]:
    row: Dict[str, Any] = {"name": name}
    if ids is not None:
        row["mapped_ids"] = dict(ids)
        row["mapping_meta"] = {"confidence": 1.0, "identity_status": "verified", **meta}
    return row


def _mapped(reactions: List[Dict[str, Any]], compounds: List[Dict[str, Any]], **extra: Any) -> Dict[str, Any]:
    """A mapped payload a pooled Stage 1 produced: every reaction carries a support count."""

    rows = deepcopy(reactions)
    for i, row in enumerate(rows):
        row.setdefault(SUPPORT_KEY, {"draws": [i % 3 + 1], "of": 3})
    return {"entities": {"compounds": compounds}, "processes": {"reactions": rows}, **extra}


# Rule 3 reads N from the payload's support counts (``_mapped``), never from the environment.


HEME = {"chebi": "CHEBI:17627"}
FERRO = [{"name": "ferro A", "inputs": ["protoporphyrin IX", "Fe2+"], "outputs": ["protoheme"], "evidence": "q1",
          SUPPORT_KEY: {"draws": [1], "of": 3}},
         {"name": "ferro B", "inputs": ["protoporphyrin IX", "Fe2+"], "outputs": ["heme b"], "evidence": "q3",
          SUPPORT_KEY: {"draws": [3], "of": 3}}]


def test_rule3_synonyms_with_one_admitted_id_merge() -> None:
    payload = _mapped(FERRO, [_row("protoporphyrin IX", {"chebi": "CHEBI:15430"}), _row("Fe2+", {"chebi": "CHEBI:29033"}),
                              _row("protoheme", HEME), _row("heme b", HEME)])
    report = merge_on_mapped_identity(payload)[IDENTITY_REPORT_KEY]
    [row] = _reactions(payload)
    assert (row["name"], row["evidence"], row[SUPPORT_KEY]) == ("ferro A", "q1", {"draws": [1, 3], "of": 3})
    assert [m["bridged_by"] for m in report["merges"]] == ["database-id"]
    assert (report["reactions_before"], report["reactions_after"]) == (2, 1)


def test_rule3_same_name_with_different_ids_never_merges() -> None:
    rows = [{"name": "a", "inputs": ["Heme"], "outputs": ["X1"], "evidence": "q"},
            {"name": "b", "inputs": ["heme"], "outputs": ["X1"], "evidence": "q"}]
    payload = _mapped(rows, [_row("Heme", {"chebi": "CHEBI:1"}), _row("heme", {"chebi": "CHEBI:2"})])
    merge_on_mapped_identity(payload)
    assert [r["name"] for r in _reactions(payload)] == ["a", "b"]


def test_rule3_without_an_id_the_identity_is_the_long_form_stage1_wrote() -> None:
    pooled, _ = merge_draws([_draw([{"name": "a", "inputs": ["GSA"], "outputs": ["X1"], "evidence": "q"}]),
                             _draw([{"name": "b", "inputs": ["Glu 1-semialdehyde (GSA)"], "outputs": ["X1"],
                                     "enzymes": [{"entity": "E9"}], "evidence": "q"}],
                                   biological_states=[{"name": "s", "subcellular_location": "cytosol"}])],
                            paper_text=PAPER)
    assert [r["inputs"] for r in _reactions(pooled)] == [["Glu 1-semialdehyde"]]  # one row, long form
    abbreviated = {"name": "c", "inputs": ["GSA"], "outputs": ["X1"], "evidence": "q"}  # no ID, no paper here
    payload = _mapped(_reactions(pooled) + [abbreviated], [])
    merge_on_mapped_identity(payload)
    assert [r["inputs"] for r in _reactions(payload)] == [["Glu 1-semialdehyde"], ["GSA"]]


@pytest.mark.parametrize("weak", [
    {"mapping_meta": {"confidence": 1.0, "candidates": [HEME]}},                                   # a candidate only
    {"mapped_ids": HEME, "mapping_meta": {"confidence": 1.0, "identity_status": "unresolved"}},
    {"mapped_ids": HEME, "mapping_meta": {"confidence": 1.0, "identity_status": "placeholder"}},
    {"mapped_ids": HEME, "mapping_meta": {"confidence": 1.0, "best_effort": True}},
    {"mapped_ids": HEME, "mapping_meta": {"confidence": 0.5}},
    {"mapped_ids": HEME},                                                                          # no mapping record
    {"rejected_mapped_ids": HEME, "mapping_meta": {"confidence": 1.0}},                            # withheld
], ids=["candidate", "unresolved", "placeholder", "best_effort", "low_confidence", "unrecorded", "withheld"])
def test_rule3_provisional_ids_never_merge(weak: Dict[str, Any]) -> None:
    payload = _mapped(FERRO, [_row("protoheme", HEME), {"name": "heme b", **deepcopy(weak)}])
    report = merge_on_mapped_identity(payload)[IDENTITY_REPORT_KEY]
    assert len(_reactions(payload)) == 2 and report["merges"] == []


def test_rule3_the_chelatase_metal_still_separates_fe_from_mg() -> None:
    compounds = [_row("Proto IX", {"chebi": "CHEBI:15430"}), _row("protoporphyrin IX", {"chebi": "CHEBI:15430"}),
                 _row("Fe2+", {"chebi": "CHEBI:29033"}), _row("Mg2+", {"chebi": "CHEBI:18420"}),
                 _row("magnesium", {"chebi": "CHEBI:18420"}), _row("ATP", {"chebi": "CHEBI:30616"}),
                 _row("Mg-Proto IX", {"chebi": "CHEBI:50152"}), _row("Mg-protoporphyrin IX", {"chebi": "CHEBI:50152"}),
                 _row("protoheme", HEME)]
    rows = [{"name": "Fe-chelatase", "inputs": ["Proto IX", "Fe2+"], "outputs": ["protoheme"], "evidence": "q"},
            {"name": "Mg-chelatase", "inputs": ["Proto IX", "Mg2+", "ATP"], "outputs": ["Mg-Proto IX"], "evidence": "q"},
            {"name": "Mg-chelatase again", "inputs": ["protoporphyrin IX", "magnesium", "ATP"],
             "outputs": ["Mg-protoporphyrin IX"], "evidence": "q"},
            {"name": "metal only Fe", "inputs": ["Fe2+"], "outputs": ["X1"], "evidence": "q"},
            {"name": "metal only Mg", "inputs": ["Mg2+"], "outputs": ["X1"], "evidence": "q"}]
    payload = _mapped(rows, compounds)
    merge_on_mapped_identity(payload)
    # "magnesium" is currency because it shares Mg2+'s identity, so the two Mg rows are one;
    # Fe never meets Mg, even on a side made only of the metal.
    assert [r["name"] for r in _reactions(payload)] == ["Fe-chelatase", "Mg-chelatase", "metal only Fe", "metal only Mg"]


def test_rule3_applies_the_same_compatibility_test() -> None:
    rows = [{"name": "IDH3", "inputs": ["isocitrate", "NAD+"], "outputs": ["2-oxoglutarate", "NADH"], "evidence": "q"},
            {"name": "IDH1", "inputs": ["isocitric acid", "NADP+"], "outputs": ["2-oxoglutarate", "NADPH"], "evidence": "q"}]
    payload = _mapped(rows, [_row("isocitrate", {"chebi": "CHEBI:16087"}), _row("isocitric acid", {"chebi": "CHEBI:16087"})])
    report = merge_on_mapped_identity(payload)[IDENTITY_REPORT_KEY]
    assert [r["name"] for r in _reactions(payload)] == ["IDH3", "IDH1"]
    assert "nicotinamide" in report["refused"][0]["reason"]


def test_rule3_files_a_merged_away_lock_and_keeps_every_lock_accounted() -> None:
    rows = deepcopy(FERRO)
    rows[0]["locked_reaction_id"], rows[1]["locked_reaction_id"] = "L1", "L2"
    payload = _mapped(rows, [_row("protoheme", HEME), _row("heme b", HEME)],
                      locked_reaction_filter_report={"locked_reactions_found": 2})
    merge_on_mapped_identity(payload)
    [record] = payload["quarantined_locked_reactions"]
    assert (record["locked_reaction_id"], record["reason"], record["merged_into"]) == ("L2", "ensemble_identity_duplicate", "L1")
    assert payload["locked_reaction_filter_report"]["unaccounted_locked_reactions"] == 0


def test_rule3_never_merges_away_a_row_a_coupled_transport_names() -> None:
    payload = _mapped(FERRO, [_row("protoheme", HEME), _row("heme b", HEME)])
    payload["processes"]["reaction_coupled_transports"] = [{"reaction": "ferro B", "transport": "t"}]
    report = merge_on_mapped_identity(payload)[IDENTITY_REPORT_KEY]
    assert len(_reactions(payload)) == 2 and report["refused"][0]["reason"] == "named by a reaction_coupled_transport"


def test_rule3_never_touches_a_single_draw_payload_whatever_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = _mapped(FERRO, [_row("protoheme", HEME), _row("heme b", HEME)])
    for row in _reactions(payload):
        row.pop(SUPPORT_KEY)  # an N=1 payload: Stage 1 never stamps one
    single_of_one = deepcopy(payload)
    for row in _reactions(single_of_one):
        row[SUPPORT_KEY] = {"draws": [1], "of": 1}
    for candidate in (payload, single_of_one):
        before = json.dumps(candidate, sort_keys=True)
        for value in (None, "1", "3"):  # a .env saying 3 must not move a saved N=1 payload
            if value is None:
                monkeypatch.delenv(ENV_VAR, raising=False)
            else:
                monkeypatch.setenv(ENV_VAR, value)
            assert merge_on_mapped_identity(candidate) == {}
            assert json.dumps(candidate, sort_keys=True) == before


def _map_offline(payload: Dict[str, Any], tmp_path: Path) -> Dict[str, Any]:
    """``map_payload`` with every external door shut (the ``test_c073`` runner's patches)."""

    from t2pw.mapping import map_ids

    with patch.dict(os.environ, {"T2PW_SPECIES_LLM": "0", "T2PW_SPECIES_NCBI": "0"}), patch.object(
        map_ids.PathBankDbResolver, "from_env", classmethod(lambda cls, overrides=None: None)
    ), patch.object(map_ids, "_ai_protein_synonym_lookup", return_value=[]), patch.object(
        map_ids.HttpClient, "get", side_effect=RuntimeError("network call during an offline run")
    ):
        return map_ids.map_payload(payload, cache_path=tmp_path / "cache.json", id_source="db", use_cache=False)


def test_map_payload_runs_rule3_last_and_only_on_a_pooled_payload(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    rows = [{"name": "a", "inputs": ["X1"], "outputs": ["X2"], "evidence": "q"},
            {"name": "b", "inputs": ["x1"], "outputs": ["X2"], "evidence": "q"}]
    monkeypatch.setenv(ENV_VAR, "3")
    single = _map_offline({"entities": {"compounds": [{"name": "X1"}, {"name": "X2"}]},
                           "processes": {"reactions": deepcopy(rows)}}, tmp_path)
    assert [r["name"] for r in _reactions(single["payload"])] == ["a", "b"]
    assert IDENTITY_REPORT_KEY not in single["report"]
    pooled = _map_offline(_mapped(rows, [{"name": "X1"}, {"name": "X2"}]), tmp_path)
    assert [r["name"] for r in _reactions(pooled["payload"])] == ["a"]
    assert pooled["report"][IDENTITY_REPORT_KEY]["merges"][0]["bridged_by"] == "exact"


# --------------------------------------------------------------------------- finding 4: the suite ignores .env's draw count


def test_a_mid_test_dotenv_load_cannot_switch_ensembling_on(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """``tests/conftest.py``: client.py's ``load_dotenv(override=True)`` re-running inside a
    test leaves the draw count as the test set it, and still loads every other variable."""

    import dotenv

    from t2pw.pipeline.stage_one_ensemble import extraction_draws

    env = tmp_path / ".env"
    env.write_text(f"{ENV_VAR}=3\nORCH754A_PROBE=loaded\n", encoding="utf-8")
    monkeypatch.delenv("ORCH754A_PROBE", raising=False)
    dotenv.load_dotenv(dotenv_path=env, override=True)
    assert os.environ.get(ENV_VAR) is None and extraction_draws() == 1
    assert os.environ["ORCH754A_PROBE"] == "loaded"
    monkeypatch.setenv(ENV_VAR, "2")  # a test that asks for an ensemble keeps it
    env.write_text(f"{ENV_VAR}=5\n", encoding="utf-8")
    dotenv.load_dotenv(dotenv_path=env, override=True)
    assert extraction_draws() == 2
