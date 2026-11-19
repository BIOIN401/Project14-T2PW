"""T2PW_REACTION_VERIFIER / T2PW_INTERACTION_TYPING: deterministic half, with a scripted model."""

import json

from t2pw.service import reaction_verifier as V

TEXT = (
    "Isochorismate is converted to 2,3-dihydro-2,3-dihydroxybenzoate and pyruvate by EntB. "
    "2,3-DHB is adenylated by EntE to 2,3-dihydroxybenzoyl-AMP in an ATP-dependent reaction. "
    "MenD condenses 2-oxoglutarate with isochorismate to form SEPHCHC and carbon dioxide. "
    "Chorismate is converted to DHNA by MenF, MenD, MenH, MenC, MenE and MenB. "
    "Chorismate is isomerised to isochorismate by MenF. Isochorismate is converted to DHNA by later enzymes. "
    "Enterobactin binds ferric iron outside the cell. "
    "Loss of function of PIF1 caused increased expression of CHLH in seedlings grown in the dark."
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


def test_incomplete_adds_only_names_in_the_paper():
    rows, rep, out = run([rx(["isochorismate"], ["2,3-dihydro-2,3-dihydroxybenzoate"], name="EntB")],
                         {0: {"verdict": "incomplete", "add_outputs": ["pyruvate", "lactate"]}})
    assert rows[0]["outputs"] == ["2,3-dihydro-2,3-dihydroxybenzoate", "pyruvate"]
    assert rep["repaired"] == [0]
    assert any(c["name"] == "pyruvate" for c in out["entities"]["compounds"])
    assert rows[0]["provenance_lineage"][-1]["origin"] == "audit_modified"


def test_vague_rename_then_fold_into_the_identical_row():
    rows, rep, _ = run([rx(["2,3-DHB", "ATP"], ["2,3-dihydroxybenzoyl-AMP"], name="EntE"),
                        rx(["2,3-DHB", "ATP"], ["activated 2,3-DHB"], name="EntE2")],
                       {1: {"verdict": "vague", "rename": [{"old": "activated 2,3-DHB", "new": "2,3-dihydroxybenzoyl-AMP"}]}})
    assert len(live(rows)) == 1 and rep["merged"] == [{"index": 1, "into": 0, "name": "EntE2"}]
    assert {e["protein"] for e in rows[0]["enzymes"]} == {"EntE", "EntE2"}


def test_rename_to_a_name_not_in_the_paper_is_refused():
    rows, rep, _ = run([rx(["2,3-DHB"], ["activated 2,3-DHB"])],
                       {0: {"verdict": "vague", "rename": [{"old": "activated 2,3-DHB", "new": "DHB-adenylate"}]}})
    assert rows[0]["outputs"] == ["activated 2,3-DHB"] and rep["repaired"] == []


def test_off_pathway_is_omitted_and_reported_with_its_quote():
    rows, rep, _ = run([rx(["enterobactin", "ferric iron"], ["ferric-enterobactin"], lock="rxn_lock_001")],
                       {0: {"verdict": "off_pathway", "quote": "Enterobactin binds ferric iron outside the cell"}})
    r = rows[0]
    assert r["scope_membership"] == "out_of_scope" and "locked_reaction_id" not in r and "locked" not in r
    assert r["verifier"]["action"] == "omitted" and r["verifier"]["model_quote"].startswith("Enterobactin binds")
    assert rep["omitted"] == [0]


def test_not_stated_with_a_verbatim_quote_is_only_flagged():
    q = "MenD condenses 2-oxoglutarate with isochorismate to form SEPHCHC"
    rows, rep, _ = run([rx(["2-oxoglutarate", "isochorismate"], ["SEPHCHC"], ev=q)], {0: {"verdict": "not_stated"}})
    assert rows[0]["scope_membership"] == "core" and rep["flagged"] == [0]
    # even with no verbatim quote anywhere: a model's "not stated" never deletes (replay: 5 certified steps)
    rows, rep, _ = run([rx(["x"], ["y"], ev="made up words that are nowhere")], {0: {"verdict": "not_stated"}})
    assert rows[0]["scope_membership"] == "core" and rep["flagged"] == [0]


def test_off_pathway_link_of_the_chain_is_kept():
    chain = [rx(["a"], ["b"]), rx(["b"], ["c"]), rx(["c"], ["d"])]
    rows, rep, _ = run(chain, {1: {"verdict": "off_pathway"}})
    assert rows[1]["scope_membership"] == "core" and rep["flagged"] == [1]
    rows, rep, _ = run(chain, {2: {"verdict": "off_pathway"}})  # a sink: omitted
    assert rows[2]["scope_membership"] == "out_of_scope"


SPLIT = [rx(["2-oxoglutarate", "isochorismate"], ["SEPHCHC", "carbon dioxide"], name="MenD"),
         rx(["2-oxoglutarate"], ["intermediate I"]),
         rx(["intermediate I", "isochorismate"], ["SEPHCHC"])]
MERGED = {"inputs": ["2-oxoglutarate", "isochorismate"], "outputs": ["SEPHCHC", "carbon dioxide"],
          "quote": "MenD condenses 2-oxoglutarate with isochorismate to form SEPHCHC and carbon dioxide"}


def test_split_policies():
    ans = {1: {"verdict": "split", "merged_step": MERGED}, 2: {"verdict": "split", "merged_step": MERGED}}
    for policy, n_live in (("merge", 1), ("combo", 1), ("omit", 1), ("keep", 3)):
        rows, rep, _ = run(SPLIT, ans, policy)
        assert len(live(rows)) == n_live, policy
    bad = {1: {"verdict": "split", "merged_step": dict(MERGED, quote="not in the paper at all ok")}}
    # the model's whole step fails the paper check: the chain's own rows are joined instead
    rows = live(run(SPLIT, bad, "merge")[0])
    assert len(rows) == 2 and rows[1]["inputs"] == ["2-oxoglutarate", "isochorismate"] and rows[1]["outputs"] == ["SEPHCHC"]
    plain_split = [rx(["isochorismate"], ["X1"]), rx(["X1"], ["SEPHCHC"])]  # no generic name: nothing to join
    assert len(live(run(plain_split, {0: bad[1]}, "merge")[0])) == 2  # kept
    assert len(live(run(plain_split, {0: bad[1]}, "combo")[0])) == 1  # combo omits it anyway


def test_generic_intermediates_are_split_even_when_the_model_says_ok():
    chain = [rx(["2-oxoglutarate"], ["intermediate I"]),
             rx(["intermediate I", "isochorismate"], ["intermediate II"]),
             rx(["intermediate II"], ["SEPHCHC", "carbon dioxide"])]
    rows, rep, _ = run(chain, {}, "merge")  # no merged_step from the model: the chain's own rows are joined
    joined = live(rows)
    assert len(joined) == 1
    assert joined[0]["inputs"] == ["2-oxoglutarate", "isochorismate"]
    assert joined[0]["outputs"] == ["SEPHCHC", "carbon dioxide"]
    rows, _, _ = run(SPLIT[:1] + chain, {}, "merge")  # joined step equals the paper's row: folded into it
    assert len(live(rows)) == 1
    assert len(live(run(chain, {}, "keep")[0])) == 3


def test_summary_under_merge_needs_its_steps():
    steps = [rx(["chorismate"], ["isochorismate"]), rx(["isochorismate"], ["DHNA"])]
    summ = rx(["chorismate"], ["DHNA"])
    rows, _, _ = run(steps + [summ], {2: {"verdict": "summary"}}, "merge")
    assert rows[2]["scope_membership"] == "out_of_scope"
    rows, rep, _ = run([steps[0], summ], {1: {"verdict": "summary"}}, "merge")
    assert rows[1]["scope_membership"] == "core" and rep["flagged"] == [1]
    rows, _, _ = run([steps[0], summ], {1: {"verdict": "summary"}}, "combo")
    assert rows[1]["scope_membership"] == "out_of_scope"


def test_reversed_needs_a_verbatim_quote():
    r = [rx(["isochorismate"], ["chorismate"])]
    rows, _, _ = run(r, {0: {"verdict": "reversed", "quote": "Chorismate is isomerised to isochorismate by MenF"}})
    assert rows[0]["inputs"] == ["chorismate"]
    rows, _, _ = run(r, {0: {"verdict": "reversed", "quote": "invented sentence about these compounds"}})
    assert rows[0]["inputs"] == ["isochorismate"]


def test_interaction_type_from_after_effect_needs_a_verbatim_quote():
    p = payload([], [{"entity_1": "PIF1", "entity_2": "CHLH", "relationship": "binding"},
                     {"entity_1": "HY5", "entity_2": "CHLH", "relationship": "binding"},
                     {"entity_1": "PIF1", "entity_2": "GUN4", "relationship": "inhibits"}])
    ans = {0: {"type": "inhibition", "quote": "Loss of function of PIF1 caused increased expression of CHLH"},
           1: {"type": "activation", "quote": "HY5 strongly activates CHLH in the light"},
           2: {"type": "inhibition", "quote": "Loss of function of PIF1 caused increased expression of CHLH"}}
    out, rep = V.apply_types(p, TEXT, ans)
    ix = out["processes"]["interactions"]
    assert [x["relationship"] for x in ix] == ["inhibition", "binding", "inhibits"]
    assert rep["changed"] == [0]


def test_quote_may_drop_a_citation_but_not_words():
    tw = V.text_words("MenD can be allosterically inhibited by DHNA in Staphylococcus aureus ( 19 ) and "
                      "Mycobacterium tuberculosis (Smith et al., 2009), suggesting feedback.")
    assert V.quote_in_paper("inhibited by DHNA in Staphylococcus aureus and Mycobacterium tuberculosis", tw)
    assert not V.quote_in_paper("inhibited by DHNA in Staphylococcus aureus and in Mycobacterium tuberculosis", tw)
    assert not V.quote_in_paper("aureus and Mycobacterium", tw)  # too short to be a quote


def test_a_signed_interaction_is_never_flipped():
    p = payload([], [{"entity_1": "PIF1", "entity_2": "HEMA1", "relationship": "inhibits"}])
    ans = {0: {"type": "activation", "quote": "Loss of function of PIF1 caused increased expression of CHLH"}}
    out, rep = V.apply_types(p, TEXT, ans)
    assert out["processes"]["interactions"][0]["relationship"] == "inhibits" and rep["changed"] == []


def test_model_error_fails_open_and_flags_default_off(monkeypatch):
    monkeypatch.delenv(V.ENV_FLAG, raising=False)
    monkeypatch.delenv(V.IX_FLAG, raising=False)
    assert not V.enabled() and not V.typing_enabled() and V.granularity() == V.DEFAULT_GRANULARITY

    def boom(*a, **k):
        raise RuntimeError("provider down")
    p = payload([rx(["a"], ["b"])])
    out, rep = V.verify_stage_one(p, TEXT, {}, chat_fn=boom)
    assert out is p and "provider down" in rep["error"]


def test_hook_end_to_end_with_a_scripted_model(tmp_path, monkeypatch):
    seen = []

    def fake_ask(system, user, chat_fn, stage):
        seen.append(json.loads(user))
        return {"rows": [{"id": 0, "verdict": "incomplete", "add_outputs": ["pyruvate"]}]}
    monkeypatch.setattr(V, "_ask", fake_ask)
    monkeypatch.setenv(V.ENV_FLAG, "1")
    monkeypatch.delenv(V.IX_FLAG, raising=False)
    p = payload([rx(["isochorismate"], ["2,3-dihydro-2,3-dihydroxybenzoate"])])
    state = {}
    out = V.apply_to_stage_one(p, TEXT, state, tmp_path, {"pathway_name": "enterobactin biosynthesis"}, "task line")
    assert "pyruvate" in out["processes"]["reactions"][0]["outputs"]
    assert seen[0]["requested_pathway"] == "enterobactin biosynthesis" and seen[0]["task"] == "task line"
    assert state["reaction_verifier"]["error"] is None
    assert json.loads((tmp_path / V.REPORT_FILE).read_text(encoding="utf-8"))["reactions"]["repaired"] == [0]
