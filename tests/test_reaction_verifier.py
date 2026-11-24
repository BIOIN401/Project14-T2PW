"""T2PW_REACTION_VERIFIER / T2PW_INTERACTION_TYPING: deterministic half, with a scripted model.

Self-contained (review R2 N3): the Stage-1 hook tests build their own minimal stand-ins for the callees
of ``service.extraction.run_stage_one`` instead of importing another test module's fixtures."""

import hashlib
import json
import types

import pytest

from t2pw.service import reaction_verifier as V

TEXT = (
    "Isochorismate is converted to 2,3-dihydro-2,3-dihydroxybenzoate and pyruvate by EntB. "
    "2,3-DHB is adenylated by EntE to 2,3-dihydroxybenzoyl-AMP in an ATP-dependent reaction. "
    "MenD condenses 2-oxoglutarate with isochorismate to form SEPHCHC and carbon dioxide. "
    "Chorismate is converted to DHNA by MenF, MenD, MenH, MenC, MenE and MenB. "
    "Chorismate is isomerised to isochorismate by MenF. Isochorismate is converted to DHNA by later enzymes. "
    "Enterobactin binds ferric iron outside the cell. "
    "Lactate dehydrogenase reduces pyruvate to lactate in the coupled assay used to follow the reaction. "
    "Loss of function of PIF1 caused increased expression of CHLH in seedlings grown in the dark. "
    "EntC and EntB interact to form a dynamic channeling system. "
    "DHNA inhibits MenD allosterically, lowering its activity in vitro."
)


def rx(i, o, ev="", lock=None, name="r"):
    r = {"name": name, "inputs": list(i), "outputs": list(o), "evidence": ev, "scope_membership": "core",
         "enzymes": [{"protein": name}]}
    if lock:
        r.update(locked_reaction_id=lock, locked=True)
    return r


def payload(rows, ix=()):
    return {"entities": {"compounds": [{"name": n} for r in rows for n in r["inputs"] + r["outputs"]]},
            "processes": {"reactions": rows, "interactions": list(ix)}}


def run(rows, answers, policy="merge"):
    p = payload(rows)
    out, rep = V.apply_verdicts(p, TEXT, answers, policy)
    assert p == payload(rows)  # input untouched
    return out["processes"]["reactions"], rep, out


def live(rows):
    return [r for r in rows if r.get("scope_membership") != "out_of_scope"]


# -- repairs ---------------------------------------------------------------------------------------
def test_incomplete_adds_only_names_in_the_paper():
    rows, rep, out = run([rx(["isochorismate"], ["2,3-dihydro-2,3-dihydroxybenzoate"], name="EntB")],
                         {0: {"verdict": "incomplete", "add_outputs": ["pyruvate", "lactate x"]}})
    assert rows[0]["outputs"] == ["2,3-dihydro-2,3-dihydroxybenzoate", "pyruvate"]
    assert rep["repaired"] == [0]
    assert any(c["name"] == "pyruvate" for c in out["entities"]["compounds"])
    assert rows[0]["provenance_lineage"][-1]["origin"] == "audit_modified"


def test_incomplete_with_a_string_adds_the_name_not_its_letters():
    rows, _, _ = run([rx(["isochorismate"], ["2,3-dihydro-2,3-dihydroxybenzoate"])],
                     {0: {"verdict": "incomplete", "add_outputs": "pyruvate"}})
    assert rows[0]["outputs"] == ["2,3-dihydro-2,3-dihydroxybenzoate", "pyruvate"]
    rows, rep, _ = run([rx(["isochorismate"], ["x"])], {0: {"verdict": "incomplete", "add_outputs": "a"}})
    assert rows[0]["outputs"] == ["x"] and rep["repaired"] == []


def test_incomplete_may_correct_a_wrong_participant():
    """R1 F7c: a wrong name is replaced (rename), not kept next to the right one, but since F-G3 only onto a
    name no other entity carries (and, R4 B1, with no abbreviation token such as AMP). The live case R1 F7c
    came from (DHB-ate renamed onto the existing 2,3-dihydroxybenzoyl-AMP) is now REFUSED: that row stays
    as Stage 1 wrote it (action "kept"; the report records the model's verdict and rename)."""
    rows, rep, _ = run([rx(["isochorismate"], ["pyruvate"], name="EntB")],
                       {0: {"verdict": "incomplete", "rename": [{"old": "pyruvate", "new": "2,3-dihydro-2,3-dihydroxybenzoate"}]}})
    assert rows[0]["outputs"] == ["2,3-dihydro-2,3-dihydroxybenzoate"] and rep["repaired"] == [0]


def test_a_rename_onto_another_entitys_name_is_refused():
    """Gemma F-G3, the vt3-PMC12096016 shape: the registry and other rows already carry
    "2,3-dihydroxybenzoyl-AMP"; renaming "activated 2,3-DHB" onto it ended the run downstream with
    PWML_IR_DUPLICATE_NAMED_ROW. Written to fail on 4eb9cd22 (which renamed and folded the row)."""
    rows = [rx(["2,3-dihydroxybenzoic acid", "ATP"], ["activated 2,3-DHB"], name="2,3-DHB adenylation"),
            rx(["activated 2,3-DHB", "L-serine"], ["enterobactin"], name="enterobactin synthesis"),
            rx(["2,3-dihydroxybenzoic acid", "ATP"], ["2,3-dihydroxybenzoyl-AMP"], name="EntE reaction"),
            rx(["2,3-dihydroxybenzoyl-AMP", "L-serine"], ["enterobactin"], name="EntF reaction")]
    for new in ("2,3-dihydroxybenzoyl-AMP", "2,3-Dihydroxybenzoyl AMP"):  # exact, and the same reference key
        out, rep = V.apply_verdicts(payload(rows), TEXT + " " + new + ".", {0: {"verdict": "vague", "rename": [
            {"old": "activated 2,3-DHB", "new": new}]}}, "merge")
        assert out["processes"]["reactions"][0]["outputs"] == ["activated 2,3-DHB"] and rep["repaired"] == [], new
        assert rep["merged"] == []
    # an added name that only differs by the reference key never becomes a second entity row
    from t2pw.pipeline.name_keys import reference_key

    out, rep = V.apply_verdicts(payload(rows), TEXT + " 2,3-Dihydroxybenzoyl AMP.", {1: {
        "verdict": "incomplete", "add_inputs": ["2,3-Dihydroxybenzoyl AMP"]}}, "merge")
    before = {reference_key(e["name"]) for e in payload(rows)["entities"]["compounds"]}
    assert {reference_key(e["name"]) for e in out["entities"]["compounds"]} == before
    assert len(out["entities"]["compounds"]) == len(payload(rows)["entities"]["compounds"])


def test_rename_to_a_name_not_in_the_paper_is_refused():
    rows, rep, _ = run([rx(["2,3-DHB"], ["activated 2,3-DHB"])],
                       {0: {"verdict": "vague", "rename": [{"old": "activated 2,3-DHB", "new": "DHB-adenylate"}]}})
    assert rows[0]["outputs"] == ["activated 2,3-DHB"] and rep["repaired"] == []


def test_reversed_needs_a_verbatim_quote():
    r = [rx(["isochorismate"], ["chorismate"])]
    rows, _, _ = run(r, {0: {"verdict": "reversed", "quote": "Chorismate is isomerised to isochorismate by MenF"}})
    assert rows[0]["inputs"] == ["chorismate"]
    rows, _, _ = run(r, {0: {"verdict": "reversed", "quote": "invented sentence about these compounds"}})
    assert rows[0]["inputs"] == ["isochorismate"]


# -- off pathway (R1 F4) ------------------------------------------------------------------------------
CHAIN = [rx(["chorismate"], ["isochorismate"], name="EntC"),
         rx(["isochorismate"], ["2,3-dihydro-2,3-dihydroxybenzoate", "pyruvate"], name="EntB"),
         rx(["2,3-DHB", "ATP"], ["2,3-dihydroxybenzoyl-AMP"], name="EntE"),
         rx(["2,3-dihydroxybenzoyl-AMP", "L-serine"], ["enterobactin"], name="EntF")]
Q_IRON = "Enterobactin binds ferric iron outside the cell"


@pytest.mark.parametrize("row,quote", [
    (rx(["ferric iron", "siderophore X"], ["iron complex"], lock="rxn_lock_009"), Q_IRON),  # isolated, verified quote
    (rx(["pyruvate", "NADH"], ["lactate", "NAD+"]), "Lactate dehydrogenase reduces pyruvate to lactate in the coupled assay"),
    (rx(["ATP"], ["ADP", "glucose 6-phosphate"]), Q_IRON),
])
def test_off_pathway_only_flags_never_drops(row, quote):
    """R3 P6: even an isolated row with a verified quote may be a gold step Stage 1 drew disconnected."""
    rows, rep, _ = run(CHAIN + [row], {4: {"verdict": "off_pathway", "quote": quote}})
    assert len(live(rows)) == 5 and rep["flagged"] == [4] and rep["omitted"] == []
    assert rows[4]["verifier"]["action"] == "flagged"
    assert rows[4].get("locked_reaction_id") == row.get("locked_reaction_id")


@pytest.mark.parametrize("idx", [0, 3])
def test_off_pathway_never_drops_the_first_or_last_step(idx):
    rows, rep, _ = run(CHAIN, {idx: {"verdict": "off_pathway", "quote": Q_IRON}})
    assert len(live(rows)) == 4 and rep["flagged"] == [idx]


def test_not_stated_only_flags():
    q = "MenD condenses 2-oxoglutarate with isochorismate to form SEPHCHC"
    rows, rep, _ = run([rx(["2-oxoglutarate", "isochorismate"], ["SEPHCHC"], ev=q)], {0: {"verdict": "not_stated"}})
    assert rows[0]["scope_membership"] == "core" and rep["flagged"] == [0]
    rows, rep, _ = run([rx(["x"], ["y"], ev="made up words that are nowhere")], {0: {"verdict": "not_stated"}})
    assert rows[0]["scope_membership"] == "core" and rep["flagged"] == [0]


# -- split / merge (R1 F7) --------------------------------------------------------------------------
SPLIT = [rx(["2-oxoglutarate", "isochorismate"], ["SEPHCHC", "carbon dioxide"], name="MenD"),
         rx(["2-oxoglutarate"], ["intermediate I"]),
         rx(["intermediate I", "isochorismate"], ["SEPHCHC"])]
MERGED = {"inputs": ["2-oxoglutarate", "isochorismate"], "outputs": ["SEPHCHC", "carbon dioxide"],
          "quote": "MenD condenses 2-oxoglutarate with isochorismate to form SEPHCHC and carbon dioxide"}


def test_split_chain_is_joined_from_its_own_rows_and_folded_into_the_paper_row():
    for policy, n_live in (("merge", 1), ("combo", 1), ("omit", 1), ("keep", 3)):
        rows, rep, _ = run(SPLIT, {1: {"verdict": "split", "merged_step": MERGED}}, policy)
        assert len(live(rows)) == n_live, policy
    rows = live(run(SPLIT, {}, "merge")[0])
    assert rows[0]["name"] == "MenD" and rows[0]["outputs"] == ["SEPHCHC", "carbon dioxide"]


def test_generic_chain_without_the_paper_row():
    chain = [rx(["2-oxoglutarate"], ["intermediate I"]),
             rx(["intermediate I", "isochorismate"], ["intermediate II"]),
             rx(["intermediate II"], ["SEPHCHC", "carbon dioxide"])]
    rows, _, _ = run(chain, {}, "merge")
    joined = live(rows)
    assert len(joined) == 1
    assert joined[0]["inputs"] == ["2-oxoglutarate", "isochorismate"]
    assert joined[0]["outputs"] == ["SEPHCHC", "carbon dioxide"]
    assert len(live(run(chain, {}, "keep")[0])) == 3


def test_two_unrelated_mechanisms_saying_intermediate_I_are_never_joined():
    a = [rx(["2-oxoglutarate"], ["intermediate I"], name="MenD"), rx(["intermediate I"], ["SEPHCHC"], name="MenD")]
    b = [rx(["glycine"], ["intermediate I"], name="ALAS"), rx(["intermediate I"], ["ALA"], name="ALAS")]
    rows, _, _ = run(a + b, {}, "merge")
    assert len(live(rows)) == 4  # nothing joined, nothing invented: all kept and flagged
    assert not any(set(r["inputs"]) >= {"2-oxoglutarate", "glycine"} for r in live(rows))


def test_model_whole_step_must_keep_the_rows_real_names():
    row = [rx(["intermediate I", "isochorismate"], ["SEPHCHC"])]
    other = dict(MERGED, inputs=["2-oxoglutarate", "chorismate"])  # drops isochorismate: a different step
    rows, rep, _ = run(row, {0: {"verdict": "split", "merged_step": other}}, "merge")
    assert rep["flagged"] == [0] and len(live(rows)) == 1


def test_model_split_on_a_real_row_never_replaces_it():
    real = [rx(["chorismate"], ["isochorismate"], name="EntC")]
    rows, rep, _ = run(real, {0: {"verdict": "split", "merged_step": MERGED}}, "merge")
    assert rows[0]["inputs"] == ["chorismate"] and rep["flagged"] == [0] and rep["appended"] == 0


# -- summary (R1 F3) ---------------------------------------------------------------------------------
def test_summary_dropped_only_when_kept_steps_make_its_product():
    steps = [rx(["chorismate"], ["isochorismate"]), rx(["isochorismate"], ["DHNA"])]
    summ = rx(["chorismate"], ["DHNA"])
    rows, _, _ = run(steps + [summ], {2: {"verdict": "summary"}}, "merge")
    assert rows[2]["scope_membership"] == "out_of_scope"
    rows, rep, _ = run([steps[0], summ], {1: {"verdict": "summary"}}, "merge")
    assert rows[1]["scope_membership"] == "core" and rep["flagged"] == [1]


def test_summary_is_not_covered_through_a_co_product():
    """R1 F3 scenario: chorismate -> isochorismate -> CO2 is a path, but nothing makes DHNA."""
    kept = [rx(["chorismate"], ["isochorismate"], name="MenF"),
            rx(["isochorismate", "2-oxoglutarate"], ["SEPHCHC", "carbon dioxide"], name="MenD")]
    summ = rx(["chorismate", "2-oxoglutarate"], ["DHNA", "carbon dioxide"])
    rows, rep, _ = run(kept + [summ], {2: {"verdict": "summary"}}, "merge")
    assert rows[2]["scope_membership"] == "core" and rep["flagged"] == [2]


def test_a_mislabelled_one_enzyme_step_is_never_dropped():
    """The EntF-type false summary: its product is made by no kept row, so it stays (whatever co-products)."""
    kept = [rx(["L-serine", "ATP"], ["seryl-AMP"]), rx(["seryl-AMP"], ["seryl-EntF", "AMP"])]
    entf = rx(["2,3-dihydroxybenzoyl-AMP", "L-serine"], ["enterobactin", "AMP"], name="EntF")
    rows, _, _ = run(CHAIN[:3] + kept + [entf], {5: {"verdict": "summary"}}, "merge")
    assert rows[5]["scope_membership"] == "core"


def test_a_co_substrate_or_a_wrong_row_never_covers_a_summary():
    """Gemma run 3 (vt3-PMC12096016): the real EntF step was called a summary and dropped because L-serine
    reached enterobactin through a wrong row (DHB + L-serine -> enterobactin) the model itself called not stated."""
    adenyl = rx(["2,3-DHB", "ATP"], ["2,3-dihydroxybenzoyl-AMP"], name="EntE")
    wrong = rx(["2,3-DHB", "L-serine"], ["enterobactin"], name="EntF?")
    entf = rx(["2,3-dihydroxybenzoyl-AMP", "L-serine"], ["enterobactin"], name="EntF")
    for wrong_verdict in ("not_stated", "ok"):
        rows, rep, _ = run([adenyl, wrong, entf], {1: {"verdict": wrong_verdict}, 2: {"verdict": "summary"}}, "merge")
        assert rows[2]["scope_membership"] == "core" and 2 in rep["flagged"], wrong_verdict


# -- interaction typing (R1 F5, F6) --------------------------------------------------------------------
Q_PIF = "Loss of function of PIF1 caused increased expression of CHLH"


def _types(rel, ans):
    p = payload([], [{"entity_1": "A", "entity_2": "B", "relationship": rel}])
    out, rep = V.apply_types(p, TEXT, {0: ans})
    return out["processes"]["interactions"][0]["relationship"], rep


def test_unsigned_row_gets_the_sign_its_after_effect_states():
    rel, rep = _types("binding", {"type": "inhibition", "quote": Q_PIF})
    assert rel == "inhibition" and rep["changed"] == [0]


@pytest.mark.parametrize("word", ["inhibits", "activation", "allosteric_inhibitor_of", "inhibited_by",
                                  "activated_by", "negatively_regulates", "positive regulator of", "represses"])
def test_a_signed_word_is_never_flipped(word):
    rel, rep = _types(word, {"type": "activation", "quote": Q_PIF})
    assert rel == word and rep["changed"] == []


def test_a_quote_that_drops_a_leading_negation_is_refused():
    rel, _ = _types("binding", {"type": "activation", "quote": "PIF1 caused increased expression of CHLH in seedlings grown in the dark"})
    assert rel == "binding"


def test_a_quote_that_states_no_effect_does_not_type_the_row():
    """R1 F6, the live false edge: EntC/EntB 'interact' became 'activation'."""
    rel, _ = _types("interaction", {"type": "activation", "quote": "EntC and EntB interact to form a dynamic channeling system"})
    assert rel == "interaction"


def test_quote_may_drop_a_citation_or_a_clause_but_not_words_or_a_negation():
    tw = V.text_words("MenD can be allosterically inhibited by DHNA in Staphylococcus aureus ( 19 ) and "
                      "Mycobacterium tuberculosis (Smith et al., 2009), suggesting feedback.")
    assert V.quote_in_paper("inhibited by DHNA in Staphylococcus aureus and Mycobacterium tuberculosis", tw)
    assert not V.quote_in_paper("inhibited by DHNA in Staphylococcus aureus and in Mycobacterium tuberculosis", tw)
    assert not V.quote_in_paper("aureus and Mycobacterium", tw)
    tw = V.text_words("FHY3 and FAR1 induce the expression of ALAD1/HEMB1 encoding an ALAD by binding to its "
                      "promoter region. PIF1 does not induce the expression of HEMA1 in the dark.")
    assert V.quote_in_paper("FHY3 and FAR1 induce the expression of ALAD1/HEMB1 by binding to its promoter region", tw)
    assert not V.quote_in_paper("PIF1 does induce the expression of HEMA1 in the dark", tw)
    assert not V.quote_in_paper("FHY3 and FAR1 strongly induce the expression of ALAD1/HEMB1 by binding", tw)


# -- round 3 (review R3 P1-P7) --------------------------------------------------------------------------
@pytest.mark.parametrize("quote", ["HY5 binds to the promoter of CHLH in the light",
                                   "GUN4 is a reductase subunit bound to CHLH",
                                   "an inducible promoter drove CHLH in the line",
                                   "antibodies raised against CHLH detected the protein",
                                   "accumulation of CHLH protein was measured in leaves"])
def test_contact_or_naming_words_are_not_effects(quote):
    assert not V.states_effect(quote)


@pytest.mark.parametrize("quote", ["2,3-DHB is a competitive inhibitor of apo-EntB isochorismatase activity",
                                   "FLU is a negative regulator and an inhibitory protein of GluTR",
                                   "HY5 is an activator of CHLH", "Fur acts as a repressor of entC",
                                   "HY5 promotes CHLH expression", "FLU inhibits GluTR", "loss of HY5 reduced CHLH",
                                   "GBP prevents degradation of GluTR1", "PIFs down-regulate CHLH", "expression was upregulated"])
def test_effect_words_are_effects(quote):
    assert V.states_effect(quote)


def test_binding_to_a_promoter_stays_unsigned():
    text = "HY5 binds to the promoter of CHLH in the light."
    p = payload([], [{"entity_1": "HY5", "entity_2": "CHLH", "relationship": "binding"}])
    out, rep = V.apply_types(p, text, {0: {"type": "activation", "quote": "HY5 binds to the promoter of CHLH in the light"}})
    assert out["processes"]["interactions"][0]["relationship"] == "binding" and rep["changed"] == []


def test_a_genus_initial_does_not_hide_the_negation_before_it():
    text = "Deletion of fur in E. coli increased expression of entC in iron-replete medium. Fig. 2 shows it."
    assert len(V.split_sentences(text)) == 2
    assert not V.quote_unqualified("increased expression of entC in iron-replete medium", text)
    assert not V.quote_unqualified("coli increased expression of entC in iron-replete medium", text)
    assert V.quote_unqualified("Deletion of fur in E. coli increased expression of entC", text)


def test_a_dropped_leading_loss_is_refused_even_when_the_quote_has_its_own_negation():
    text = "Loss of PIF1 did not reduce expression of CHLH in seedlings."
    assert not V.quote_unqualified("PIF1 did not reduce expression of CHLH", text)
    assert V.quote_unqualified("Loss of PIF1 did not reduce expression of CHLH", text)


def test_the_typing_word_is_read_as_the_exporter_reads_it():
    p = payload([], [{"entity_1": "A", "entity_2": "B", "interaction_type": "inhibits", "relationship": ""},
                     {"entity_1": "A", "entity_2": "B", "interaction_type": "binding", "relationship": ""}])
    out, rep = V.apply_types(p, TEXT, {0: {"type": "activation", "quote": Q_PIF}, 1: {"type": "inhibition", "quote": Q_PIF}})
    ix = out["processes"]["interactions"]
    assert ix[0]["interaction_type"] == "inhibits" and ix[0]["relationship"] == ""  # signed: untouched
    assert ix[1]["interaction_type"] == "inhibition" and ix[1]["relationship"] == "inhibition"  # text and PWML agree


def test_a_rename_never_leaves_two_spellings_of_one_compound():
    """R3 P4 (the AMBIGUOUS_RENAME_TARGET shape): other rows keep '2,3-dihydroxybenzoic acid'."""
    text = TEXT + " Then 2,3-dihydroxybenzoate is adenylated."
    rows = [rx(["2,3-dihydroxybenzoic acid", "ATP"], ["2,3-dihydroxybenzoyl-AMP"]),
            rx(["2,3-dihydroxybenzoic acid", "L-serine"], ["enterobactin"])]
    out, rep = V.apply_verdicts(payload(rows), text, {1: {"verdict": "vague", "rename": [
        {"old": "2,3-dihydroxybenzoic acid", "new": "2,3-dihydroxybenzoate"}]}}, "merge")
    names = {n for r in out["processes"]["reactions"] for n in r["inputs"] + r["outputs"]}
    assert "2,3-dihydroxybenzoate" not in names  # refused: the old spelling lives on in row 0
    out, _ = V.apply_verdicts(payload([rx(["isochorismate"], ["2,3-dihydro-2,3-dihydroxybenzoate"]),
                                       rx(["pyruvic acid", "NADH"], ["lactate"])]), TEXT,
                              {0: {"verdict": "incomplete", "add_outputs": ["pyruvate"]}}, "merge")
    assert out["processes"]["reactions"][0]["outputs"][-1] == "pyruvic acid"  # the payload's own spelling


def test_rows_differing_in_cofactor_are_not_folded():
    rows = [rx(["malate", "NAD+"], ["oxaloacetate", "NADH"]), rx(["malate", "NADP+"], ["oxaloacetate", "NADPH"])]
    out, rep = V.apply_verdicts(payload(rows), TEXT + " A hydrogen ion is released.", {1: {
        "verdict": "incomplete", "add_outputs": ["hydrogen ion"]}}, "merge")
    assert rep["repaired"] == [1] and rep["merged"] == []


def test_an_abbreviated_synonym_is_never_written_next_to_the_full_name():
    """R4 B1: "DHB-AMP" is in the paper, but other rows say 2,3-dihydroxybenzoyl-AMP. Old code (4d623f2a)
    renamed the row (the old name occurs in no other row and "dhbamp" folds to nothing in the payload)."""
    text = TEXT + " DHB-AMP and DHB then react."
    rows = [rx(["2,3-DHB", "ATP"], ["2,3-dihydroxybenzoyl-AMP"]),
            rx(["2,3-DHB", "ATP"], ["activated 2,3-DHB"]),
            rx(["2,3-dihydroxybenzoic acid"], ["x"])]
    out, rep = V.apply_verdicts(payload(rows), text, {1: {"verdict": "vague", "rename": [
        {"old": "activated 2,3-DHB", "new": "DHB-AMP"}]}}, "merge")
    names = {n for r in out["processes"]["reactions"] for n in r["inputs"] + r["outputs"]}
    assert "DHB-AMP" not in names and rep["repaired"] == []
    out, rep = V.apply_verdicts(payload(rows), text, {2: {"verdict": "incomplete", "add_inputs": ["DHB"]}}, "merge")
    assert "DHB" not in out["processes"]["reactions"][2]["inputs"]
    # F-G3: not even the payload's own exact spelling (another entity's name)
    out, rep = V.apply_verdicts(payload(rows), text, {1: {"verdict": "vague", "rename": [
        {"old": "activated 2,3-DHB", "new": "2,3-dihydroxybenzoyl-AMP"}]}}, "merge")
    assert rep["merged"] == [] and rep["repaired"] == []


def test_budget_is_a_wall_clock_bound_across_retries():
    from t2pw.llm.client import worst_case_call_seconds

    t, attempts = V.Budget(seconds=700).take()
    assert worst_case_call_seconds(timeout=t, max_retries=attempts) <= 700
    assert V.Budget(seconds=40).take() is None
    seen = {}
    V._ask("s", "u", lambda *a, **k: seen.update(k) or '{"rows": []}', "stage", (120.0, 2))
    assert seen["timeout"] == 120.0 and seen["max_retries"] == 2


# -- run-level: budget, failure, lock manifest, report (R1 F8, F9) ----------------------------------
def test_budget_caps_calls_and_rows_past_it_stay_untouched(monkeypatch):
    calls = []

    def fake_ask(system, user, chat_fn, stage, limit=None):
        calls.append(stage)
        return {"rows": []}
    monkeypatch.setattr(V, "_ask", fake_ask)
    monkeypatch.setattr(V, "MAX_CALLS", 2)
    rows = [rx(["a%d" % i], ["b%d" % i]) for i in range(95)]  # 4 chunks of 30
    out, rep = V.verify_stage_one(payload(rows), TEXT, {}, interactions=True)
    assert len(calls) == 2 and rep["calls_skipped_over_budget"] >= 1 and rep["error"] is None
    assert rep["calls"] == 2 and rep["seconds"] >= 0 and "model" in rep


def test_model_error_fails_open_and_flags_default_off(monkeypatch):
    for f in (V.ENV_FLAG, V.IX_FLAG, V.GRANULARITY_ENV):
        monkeypatch.delenv(f, raising=False)
    assert not V.enabled() and not V.typing_enabled() and V.granularity() == V.DEFAULT_GRANULARITY

    def boom(*a, **k):
        raise RuntimeError("provider down")
    p = payload([rx(["a"], ["b"])])
    out, rep = V.verify_stage_one(p, TEXT, {}, chat_fn=boom)
    assert out is p and "provider down" in rep["error"]


def test_failed_typing_call_keeps_the_finished_reaction_step_and_its_lock_prune(tmp_path, monkeypatch):
    """Gemma run 3 (vt3-PMC12312563): the typing reply was bad JSON; the whole payload was discarded but the
    omitted rows' locks were still pruned. Now the reaction step stands and manifest and payload agree."""
    def fake_ask(system, user, chat_fn, stage, limit=None):
        if "interaction" in stage:
            raise ValueError("Invalid \\uXXXX escape")
        return {"rows": [{"id": 4, "verdict": "summary"}]}
    monkeypatch.setattr(V, "_ask", fake_ask)
    monkeypatch.setenv(V.ENV_FLAG, "1")
    monkeypatch.setenv(V.IX_FLAG, "1")
    rows = [dict(r, locked_reaction_id="rxn_lock_00%d" % i) for i, r in enumerate(CHAIN)]
    rows.append(rx(["chorismate"], ["2,3-dihydro-2,3-dihydroxybenzoate"], lock="rxn_lock_009"))
    (tmp_path / "locked_reaction_manifest.json").write_text(
        json.dumps([{"locked_reaction_id": r["locked_reaction_id"]} for r in rows]), encoding="utf-8")
    state = {}
    out = V.apply_to_stage_one(payload(rows, [{"entity_1": "a", "entity_2": "b", "relationship": ""}]), TEXT, state,
                               tmp_path / "diag", {}, "task", manifest_dir=tmp_path)
    rep = state[V.REPORT_STATE_KEY]
    assert "Invalid" in rep["error"] and rep["reactions"]["omitted"] == [4]
    assert out["processes"]["reactions"][4]["scope_membership"] == "out_of_scope"
    left = json.loads((tmp_path / "locked_reaction_manifest.json").read_text(encoding="utf-8"))
    assert len(left) == 4 and rep["lock_manifest_entries_removed"] == 1


def test_hook_prunes_the_lock_manifest_and_hands_the_report_to_the_run(tmp_path, monkeypatch):
    def fake_ask(system, user, chat_fn, stage, limit=None):
        return {"rows": [{"id": 4, "verdict": "summary"}]}
    monkeypatch.setattr(V, "_ask", fake_ask)
    monkeypatch.setenv(V.ENV_FLAG, "1")
    monkeypatch.delenv(V.IX_FLAG, raising=False)
    rows = [dict(r, locked_reaction_id="rxn_lock_00%d" % i) for i, r in enumerate(CHAIN)]
    rows.append(rx(["chorismate"], ["2,3-dihydro-2,3-dihydroxybenzoate"], lock="rxn_lock_009"))  # covered summary
    manifest = [{"locked_reaction_id": r["locked_reaction_id"]} for r in rows]
    (tmp_path / "locked_reaction_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    state = {}
    out = V.apply_to_stage_one(payload(rows), TEXT, state, tmp_path / "diag", {"pathway_name": "enterobactin"},
                               "task", manifest_dir=tmp_path)
    left = json.loads((tmp_path / "locked_reaction_manifest.json").read_text(encoding="utf-8"))
    assert [e["locked_reaction_id"] for e in left] == ["rxn_lock_000", "rxn_lock_001", "rxn_lock_002", "rxn_lock_003"]
    assert state[V.REPORT_STATE_KEY]["lock_manifest_entries_removed"] == 1
    assert state[V.REPORT_STATE_KEY]["reactions"]["omitted"] == [4]
    assert out["processes"]["reactions"][4]["scope_membership"] == "out_of_scope"


def test_batch_driver_writes_the_report_only_when_present():
    from t2pw.batch import driver as D

    at = types.SimpleNamespace(session_state={"reaction_verifier_report": {"flags": {}, "error": None}})
    out = {}
    D._add_common_artifacts(at, {}, out)
    assert "reaction_verifier_report.json" in out
    out = {}
    D._add_common_artifacts(types.SimpleNamespace(session_state={}), {}, out)
    assert "reaction_verifier_report.json" not in out


# -- flags off = main, on the two code paths this branch touches (no network) ----------------------
ALL_FLAGS = (V.ENV_FLAG, V.IX_FLAG, V.GRANULARITY_ENV, "T2PW_VD18_SPAN_TEMPLATES")

#: sha256 of [(name, pattern, flags)] of rag.admission._ALL_PROSE_PATTERNS at main fd814b8e (14 templates),
#: computed from C:/t/acc-base by C:/t/acc-runs/scripts/main_pattern_hash.py. Frozen: an independent copy of
#: main's list, not the branch's own tuple (review R2 N1).
MAIN_PROSE_PATTERNS_SHA256 = "4aa0034162fcab79dafa4cd877c5f08bdcc81a9d4573bf573e61b48848017c4b"


class _Reporter:
    def __getattr__(self, name):
        class _Ctx:
            def __enter__(self):
                return None

            def __exit__(self, *a):
                return False
        return lambda *a, **k: _Ctx()


@pytest.fixture
def stage_one_world(monkeypatch, tmp_path):
    """Minimal stand-ins for the callees of ``service.extraction.run_stage_one`` (no network, no model)."""
    import t2pw.config as config
    import t2pw.paths as paths
    import t2pw.pipeline.extraction_diagnostics as diag
    import t2pw.pipeline.pipeline as pipeline
    import t2pw.pipeline.stage_one_boundary as boundary

    w = types.SimpleNamespace(stage1={"entities": {}, "processes": {
        "reactions": [{"name": "EntB", "inputs": ["isochorismate"], "outputs": ["2,3-diDHB"]}],
        "interactions": [{"entity_1": "Fur", "entity_2": "entC", "relationship": "binding"}]}})
    monkeypatch.setattr(paths, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(pipeline, "run_stage_one_with_chunking", lambda text, **kw: (json.loads(json.dumps(w.stage1)), []))
    monkeypatch.setattr(boundary, "settle_stage_one", lambda p, **kw: types.SimpleNamespace(
        payload=dict(p, settled=True), ok=True, failure=None, reconstruction={}, incomplete_reason="",
        to_summary=lambda: {"ok": True}))
    monkeypatch.setattr(diag, "current", lambda: types.SimpleNamespace(artifacts=lambda: {}))
    monkeypatch.setattr(config, "rag_config", lambda *a, **k: {"enabled": False})
    return w


def _run_stage_one(state):
    from t2pw.service import extraction as ex

    return ex.run_stage_one("the paper", pathway_context={"pathway_name": "Menaquinone"}, user_task_context="focus",
                            diagnostics_dir="C:/diag/run-1", enable_chunking=True, chunk_size=1200.0, chunk_overlap=200.0,
                            extract_attempts=3.0, temperature=0.25, extract_tokens=8000.0, rag_incomplete_flag=0,
                            state=state, reporter=_Reporter())


def test_flags_off_stage_one_hook_is_main(stage_one_world, monkeypatch):
    """service.extraction.run_stage_one with every new flag unset: the verifier is never entered and
    Stage 1 returns what main returns (the settled boundary payload, untouched)."""
    for f in ALL_FLAGS:
        monkeypatch.delenv(f, raising=False)
    entered = []
    monkeypatch.setattr(V, "apply_to_stage_one", lambda *a, **k: entered.append(1) or a[0])
    monkeypatch.setattr(V, "_ask", lambda *a, **k: entered.append(2) or {})
    state = {}
    res = _run_stage_one(state)
    assert entered == [] and "reaction_verifier" not in state and V.REPORT_STATE_KEY not in state
    assert res.stage_one == dict(stage_one_world.stage1, settled=True)
    monkeypatch.setenv(V.ENV_FLAG, "1")  # sensitivity: the same fixture with the flag on enters the verifier
    _run_stage_one({})
    assert entered == [1]


def test_flags_off_admission_is_main(monkeypatch):
    """rag.admission with T2PW_VD18_SPAN_TEMPLATES unset walks a template list identical, pattern by
    pattern, to main fd814b8e's (frozen hash), so its verdicts are main's."""
    from t2pw.rag import admission as AD

    for f in ALL_FLAGS:
        monkeypatch.delenv(f, raising=False)
    rows = [[n, p.pattern, int(p.flags)] for n, p in AD._prose_patterns()]
    assert len(rows) == 14
    assert hashlib.sha256(json.dumps(rows, ensure_ascii=False).encode("utf-8")).hexdigest() == MAIN_PROSE_PATTERNS_SHA256
