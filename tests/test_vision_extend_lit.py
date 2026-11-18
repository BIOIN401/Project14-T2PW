"""V8 literature extension: recorded Europe PMC fixtures and a fake LLM, no network."""

import json
from pathlib import Path

import pytest

from t2pw.vision.extend import literature
from t2pw.vision.extend.literature import extend_literature, quote_in, run_if_enabled, screen
from t2pw.vision.payload import build_big_payload, keyed_rows

FIX = Path(__file__).parent / "data" / "vision" / "extend_lit"
PAPER = {"pmcid": "PMC1111111", "title": "Siderophore biosynthesis in Escherichia coli (the input paper)"}


def c(name, kegg):
    return {"name": name, "mapped_ids": {"kegg": kegg}, "evidence": f"{name} is made"}


def base():
    return {
        "entities": {
            "species": [{"name": "Escherichia coli", "taxonomy_id": "562", "evidence": "E. coli"}],
            "subcellular_locations": [{"name": "cytoplasm", "evidence": "cytoplasm"}],
            "compounds": [c("chorismate", "C00251"), c("isochorismate", "C00885"), c("dihydrodiol", "C04171"),
                          c("water", "C00001")],
            "proteins": [{"name": "EntC", "mapped_ids": {"uniprot": "P0AEJ2"}, "species": "Escherichia coli",
                          "evidence": "EntC"},
                         {"name": "EntB", "mapped_ids": {"uniprot": "P0ADI4"}, "species": "Escherichia coli",
                          "evidence": "EntB"}],
        },
        "biological_states": [{"name": "cell state", "species": "Escherichia coli",
                               "subcellular_location": "cytoplasm", "evidence": "in the cell"}],
        "processes": {"reactions": [
            {"name": "chorismate to isochorismate", "inputs": ["chorismate"], "outputs": ["isochorismate"],
             "biological_state": "cell state", "evidence": "EntC makes isochorismate",
             "enzymes": [{"entity": "EntC", "entity_type": "protein", "role": "catalyst"}]},
            {"name": "isochorismate to dihydrodiol", "inputs": ["isochorismate", "water"], "outputs": ["dihydrodiol"],
             "biological_state": "cell state", "evidence": "EntB makes the diol",
             "enzymes": [{"entity": "EntB", "entity_type": "protein", "role": "catalyst"}]}]},
    }


def big0():
    return build_big_payload(base(), paper_meta=PAPER)


def fake_get(calls=None):
    def get(url, *, deadline, cache_dir):
        if calls is not None:
            calls.append(url)
        if "fullTextXML" in url:
            return (FIX / "fulltext_PMC9000001.xml").read_text(encoding="utf-8") if "PMC9000001" in url else None
        for term in ("chorismate", "dihydrodiol"):
            if term in url:
                return (FIX / f"search_{term}.json").read_text(encoding="utf-8")
        return json.dumps({"resultList": {"result": []}})
    return get


def rx(ins, outs, enzymes, quote, name=""):
    return {"name": name, "inputs": [{"name": n, "stoichiometry": None} for n in ins],
            "outputs": [{"name": n, "stoichiometry": None} for n in outs], "enzymes": enzymes,
            "reversible": False, "quote": quote}


# What a model would answer for each passage (a passage that names a trigger gets that reply).
REPLIES = [
    ("AroC converts", rx(["enolpyruvylshikimate phosphate"], ["chorismate"], ["AroC"],
                         "AroC converts enolpyruvylshikimate phosphate to chorismate")),
    ("EntC converts", rx(["chorismate"], ["isochorismate"], ["EntC"], "EntC converts chorismate to isochorismate")),
    ("XyzA converts fooate to barate", rx(["fooate"], ["barate"], ["XyzA"], "XyzA converts fooate to barate")),
    ("EntA converts", rx(["dihydrodiol"], ["dihydroxybenzoate"], ["EntA"],
                         "EntA converts dihydrodiol to dihydroxybenzoate")),
    # a paraphrase: the passage never says this
    ("The diol is then", rx(["dihydrodiol"], ["dihydroxybenzoate"], [],
                            "the diol is converted into dihydroxybenzoate rapidly")),
]


def fake_chat(calls=None):
    def chat(system, user):
        if calls is not None:
            calls.append(user)
        return json.dumps({"reactions": [r for trigger, r in REPLIES if trigger in user]})
    return chat


def run(**kw):
    kw.setdefault("get", fake_get())
    kw.setdefault("chat_fn", fake_chat())
    return extend_literature(big0(), **kw)


def ext_rows(big, section):
    return [r for s, _, r, is_base in keyed_rows(big) if s == section and not is_base]


def test_quote_check_rejects_paraphrase_and_accepts_verbatim():
    text = "Here we show that EntA converts  dihydrodiol\nto dihydroxybenzoate in E. coli."
    assert quote_in("EntA converts dihydrodiol to dihydroxybenzoate", text)
    assert quote_in("entA CONVERTS dihydrodiol to dihydroxybenzoate", text)        # case / spacing normalised
    assert not quote_in("EntA turns dihydrodiol into dihydroxybenzoate", text)     # paraphrase
    assert not quote_in("EntA converts", text)                                     # under 20 characters
    big, report = run()
    by_id = {p["source_id"]: p for p in report["papers"]}
    assert by_id["PMID:30000002"]["rejected"] == {"quote_not_verbatim": 1}


def test_xml_spacing_artefacts_and_escaped_titles():
    from t2pw.vision.extend.literature import clean_text

    # live: itertext puts a space where a tag ended ('in <i>B. subtilis</i>.' -> 'subtilis .')
    assert quote_in("catalyzed by HepT ( HepS ) in B. subtilis.", "is catalyzed by HepT (HepS) in B. subtilis . Next")
    assert not quote_in("catalyzed by HepS in B. subtilis.", "is catalyzed by HepT (HepS) in B. subtilis . Next")
    assert clean_text("Menaquinone in &lt;i&gt;Lactococcus lactis&lt;/i&gt;: advances") == \
        "Menaquinone in Lactococcus lactis : advances"


def test_full_text_failure_is_recorded():
    real = fake_get()

    def get(url, *, deadline, cache_dir):
        if "fullTextXML" in url:
            raise TimeoutError("read timeout")
        return real(url, deadline=deadline, cache_dir=cache_dir)

    big, report = run(get=get)
    by_id = {p["source_id"]: p for p in report["papers"]}
    assert by_id["PMC9000001"]["full_text_chars"] == 0
    assert by_id["PMC9000001"]["full_text_errors"] == ["TimeoutError"]
    assert by_id["PMC9000001"]["passages"] == 1          # the abstract only


def test_input_paper_preprint_and_no_journal_are_dropped():
    big, report = run()
    why = {x["id"]: (x["decision"], x["reason"]) for x in report["candidates"]}
    assert why["PMC1111111"] == ("dropped", "input_paper")
    assert why["DOI:10.1101/2024.01.01.500001"] == ("dropped", "preprint")
    assert why["PMID:30000003"] == ("dropped", "abstract_names_other_organism")
    assert {x["reason"] for x in report["candidates"] if x["decision"] == "dropped"} >= {"no_identifier"}
    assert [p["source_id"] for p in report["papers"]] == ["PMC9000001", "PMID:30000002"]
    assert set(big["sources"]) == {"paper", "PMC9000001", "PMID:30000002"}
    # also matched by DOI or PMID alone
    item = {"pmid": "123", "doi": "10.1/X", "journalInfo": {"journal": {"title": "J"}}}
    assert screen(item, {"doi": "10.1/x"}) == "input_paper"
    assert screen(item, {"pmid": "123"}) == "input_paper"
    assert screen(item, {"pmcid": "PMC5"}) is None
    assert screen({"pmid": "1", "journalInfo": {"journal": {"title": "bioRxiv"}}}, {}) == "preprint"
    assert screen({"pmid": "1", "pubTypeList": {"pubType": ["Retracted Publication"]},
                   "journalInfo": {"journal": {"title": "J"}}}, {}) == "retracted"
    assert screen({"pmid": "1"}, {}) == "no_journal_record"


def test_steps_provenance_and_sources_complete():
    big, report = run()
    rows = ext_rows(big, "processes.reactions")
    assert sorted((r["inputs"], r["outputs"]) for r in rows) == [
        (["dihydrodiol"], ["dihydroxybenzoate"]), (["enolpyruvylshikimate phosphate"], ["chorismate"])]
    for s, k, r, is_base in keyed_rows(big):
        if is_base:
            continue
        assert r.get("provenance") in (None, "extension_lit")
        recs = big["provenance"][k]
        assert recs and all(x["kind"] == "other_paper" and x["stage"] == "extension_lit" and x["quote"]
                            and x["source"] in big["sources"] for x in recs), (k, recs)
        if s == "processes.reactions":
            assert big["hops"][k] == 1
            text = (FIX / "fulltext_PMC9000001.xml").read_text(encoding="utf-8") + json.dumps(
                json.loads((FIX / "search_dihydrodiol.json").read_text(encoding="utf-8")))
            assert all(x["quote"] in text for x in recs)
    for sid in ("PMC9000001", "PMID:30000002"):
        src = big["sources"][sid]
        assert src["kind"] == "other_paper"
        for f in ("title", "journal", "year", "url", "retrieved"):
            assert src[f], (sid, f)
    assert big["sources"]["PMC9000001"]["pmid"] == "30000001"
    assert big["sources"]["PMC9000001"]["doi"] == "10.9999/jtb.00001-19"
    # no compartment invented: no state, no location row from literature
    assert not ext_rows(big, "biological_states") and not ext_rows(big, "entities.subcellular_locations")
    assert all("biological_state" not in r for r in rows)
    # the catalyst each quote names, nothing else
    assert sorted(e["entity"] for r in rows for e in r["enzymes"]) == ["AroC", "EntA"]
    by_id = {p["source_id"]: p for p in report["papers"]}
    assert by_id["PMC9000001"]["rejected"] == {"not_connected": 1}
    assert report["llm"]["calls"] == 3      # OA abstract + 1 full-text passage; abstract-only paper


def test_merge_into_base_row_adds_record_and_no_duplicate():
    big, report = run()
    base_rx = [k for s, k, r, is_base in keyed_rows(big) if is_base and s == "processes.reactions"
               and r["inputs"] == ["chorismate"]]
    assert len(base_rx) == 1
    recs = big["provenance"][base_rx[0]]
    lit = [x for x in recs if x["kind"] == "other_paper"]
    assert len(lit) == 1 and lit[0]["source"] == "PMC9000001"
    assert lit[0]["quote"] == "EntC converts chorismate to isochorismate"
    assert not any(r["inputs"] == ["chorismate"] for r in ext_rows(big, "processes.reactions"))
    names = [r["name"] for r in ext_rows(big, "entities.compounds")]
    assert "chorismate" not in names and "dihydrodiol" not in names            # joined to base rows
    assert sorted(names) == ["dihydroxybenzoate", "enolpyruvylshikimate phosphate"]
    assert report["placement"]["PMC9000001"] == {"merged_into_base": 1, "hop_1": 1}
    assert report["placement"]["PMID:30000002"] == {"hop_1": 1}
    assert big["base"] == base()                                               # base never edited


def test_unsupported_enzyme_and_unstated_currency_are_dropped():
    from t2pw.vision.extend.literature import check_reaction
    from t2pw.vision.payload import _Index

    text = "Here we show that EntA converts dihydrodiol to dihydroxybenzoate in Escherichia coli."
    step, why, drops = check_reaction(rx(["dihydrodiol", "NAD+"], ["dihydroxybenzoate", "NADH"], ["EntB"],
                                         "EntA converts dihydrodiol to dihydroxybenzoate"), text, "Escherichia coli",
                                      _Index(big0()))
    assert why is None and step["enzymes"] == [] and step["direction"] is None and step["location"] is None
    assert [p["name"] for p in step["inputs"]] == ["dihydrodiol"]
    assert drops == {"currency_not_in_quote": 2, "enzyme_not_in_quote": 1}


def test_stored_quote_is_the_source_text():   # F3
    from t2pw.vision.extend.literature import check_reaction
    from t2pw.vision.payload import _Index

    text = "At the branch point, EntC converts chorismate\nto  isochorismate ( see Fig. 2 ) ."
    step, why, _ = check_reaction(rx(["chorismate"], ["isochorismate"], ["EntC"],
                                     "entc converts chorismate to ISOCHORISMATE"), text,
                                  "Escherichia coli", _Index(big0()), "match")
    assert why is None and step["quote"] == "EntC converts chorismate\nto  isochorismate"
    from t2pw.vision.extend.literature import source_span
    assert source_span("isochorismate (see fig. 2).", text) == "isochorismate ( see Fig. 2 ) ."
    assert source_span("EntC converts chorismate to MenF", text) is None


def test_swapped_direction_and_other_organism_quotes_are_refused():   # F7
    from t2pw.vision.extend.literature import check_reaction
    from t2pw.vision.payload import _Index

    text = "EntC converts chorismate to isochorismate. In Bacillus subtilis, MenF converts chorismate to isochorismate."
    _, why, _ = check_reaction(rx(["isochorismate"], ["chorismate"], [], "EntC converts chorismate to isochorismate"),
                               text, "Escherichia coli", _Index(big0()), "match")
    assert why == "admission: evidence_direction_disagrees_with_claim"
    _, why, _ = check_reaction(rx(["chorismate"], ["isochorismate"], ["MenF"],
                                  "In Bacillus subtilis, MenF converts chorismate to isochorismate"),
                               text, "Escherichia coli", _Index(big0()), "match")
    assert why == "quote_organism_mismatch"


def test_vd19_organism_verdicts():
    from t2pw.vision.extend.literature import check_reaction
    from t2pw.vision.payload import _Index

    text = "EntC converts chorismate to isochorismate. In Escherichia coli, EntC converts chorismate to isochorismate."
    plain = rx(["chorismate"], ["isochorismate"], ["EntC"], "EntC converts chorismate to isochorismate")
    named = rx(["chorismate"], ["isochorismate"], ["EntC"],
               "In Escherichia coli, EntC converts chorismate to isochorismate")
    for claim, paper, related in ((plain, "match", False), (plain, "unknown", True), (plain, "genus_level", True),
                                  (named, "unknown", False)):    # the quote's own organism comes first
        step, why, _ = check_reaction(claim, text, "Escherichia coli", _Index(big0()), paper)
        assert why is None and step["related_organism"] is related, (paper, claim["quote"])


def test_vd19_related_organism_rows_are_full_view_only():
    from t2pw.vision.view import DEFAULT, FULL, PAPER_ONLY, unresolved_refs, view

    real = fake_get()

    def get(url, *, deadline, cache_dir):   # the abstract-only paper no longer names its organism
        body = real(url, deadline=deadline, cache_dir=cache_dir)
        return body.replace(" in Escherichia coli", "") if body and "dihydrodiol" in url else body

    big, report = run(get=get)
    by_id = {p["source_id"]: p for p in report["papers"]}
    assert by_id["PMID:30000002"]["organism_match"] == "unknown" and by_id["PMID:30000002"]["accepted"] == 1
    keys = [k for s, k, r, b in keyed_rows(big) if not b and s == "processes.reactions" and r["inputs"] == ["dihydrodiol"]]
    recs = big["provenance"][keys[0]]
    assert all(x["related_organism"] is True and "related_organism" in x["note"] for x in recs)
    names = lambda v: [r["name"] for r in v["processes"]["reactions"]]  # noqa: E731
    step = ext_rows(big, "processes.reactions")
    rel_name = next(r["name"] for r in step if r["inputs"] == ["dihydrodiol"])
    assert rel_name in names(view(big, FULL))
    assert rel_name not in names(view(big, DEFAULT)) and rel_name not in names(view(big, PAPER_ONLY))
    d = view(big, DEFAULT)
    assert "dihydroxybenzoate" not in [r["name"] for r in d["entities"]["compounds"]]
    for scope in (DEFAULT, FULL, PAPER_ONLY):
        assert unresolved_refs(view(big, scope)) == []


def test_title_only_and_prefixed_doi_input_paper_match():   # F1
    item = {"pmid": "9", "title": "Siderophore biosynthesis in <i>Escherichia coli</i>",
            "doi": "10.5/abc", "journalInfo": {"journal": {"title": "J"}}}
    assert screen(item, {"title": "Siderophore biosynthesis in Escherichia coli"}) == "input_paper"
    assert screen(item, {"title": "Siderophore biosynthesis in &lt;i&gt;Escherichia coli&lt;/i&gt;"}) == "input_paper"
    assert screen(item, {"doi": "https://doi.org/10.5/ABC"}) == "input_paper"
    assert screen(item, {"doi": "doi:10.5/abc"}) == "input_paper"
    assert screen(item, {"title": "Another paper"}) is None
    big, report = run(paper={"title": PAPER["title"]})          # no ids at all: the title alone excludes it
    assert ("PMC1111111", "input_paper") in {(x["id"], x["reason"]) for x in report["candidates"]}


def test_top_level_failure_is_fail_open(monkeypatch):   # F5
    def boom(big):
        raise RuntimeError("frontier broke")

    monkeypatch.setattr(literature, "frontier", boom)
    big = big0()
    out, report = extend_literature(big, get=fake_get(), chat_fn=fake_chat())
    assert out is big and report["status"] == "failed" and "frontier broke" in report["error"]


def test_step_counted_only_when_to_rows_keeps_it(monkeypatch):   # F6
    from collections import Counter

    def drop_all(big, pw):
        return {}, {}, {}, Counter(skipped_self_loop=1)

    monkeypatch.setattr(literature, "to_rows", drop_all)
    big, report = run()
    by_id = {p["source_id"]: p for p in report["papers"]}
    assert by_id["PMID:30000002"]["accepted"] == 0
    assert by_id["PMID:30000002"]["rejected"]["to_rows: skipped_self_loop"] == 1
    assert big["extension"] == {}


def test_llm_budget_exhaustion_stops_cleanly():
    chats = []
    big, report = run(max_llm_calls=1, chat_fn=fake_chat(chats))
    assert len(chats) == 1 and report["llm"]["calls"] == 1
    assert report["status"] == "llm_budget_exhausted"
    assert [p["status"] for p in report["papers"]] == ["ok", "llm_budget_exhausted"]
    assert not ext_rows(big, "processes.reactions")     # only the (empty) abstract was read


def test_time_budget_exhaustion_stops_cleanly():
    chats = []
    big, report = run(budget_s=0, chat_fn=fake_chat(chats))
    assert report["status"] == "budget_exhausted" and not chats
    assert big["extension"] == {}


def test_max_papers_caps_and_records():
    big, report = run(max_papers=1)
    assert [p["source_id"] for p in report["papers"]] == ["PMC9000001"]
    assert ("PMID:30000002", "over_max_papers") in {(x["id"], x["reason"]) for x in report["candidates"]}


def test_failure_mid_paper_keeps_nothing_from_that_paper(monkeypatch):
    real = literature._rows_for

    def flaky(big, step, sid, n, title, note):
        if sid == "PMC9000001" and n == 2:
            raise RuntimeError("boom")
        return real(big, step, sid, n, title, note)

    monkeypatch.setattr(literature, "_rows_for", flaky)
    big, report = run()
    by_id = {p["source_id"]: p for p in report["papers"]}
    assert by_id["PMC9000001"]["status"] == "failed" and "boom" in by_id["PMC9000001"]["error"]
    assert "PMC9000001" not in big["sources"]
    assert not any(x["source"] == "PMC9000001" for recs in big["provenance"].values() for x in recs)
    assert by_id["PMID:30000002"]["accepted"] == 1 and "PMID:30000002" in big["sources"]


def test_search_failure_is_fail_open():
    def bad_get(url, *, deadline, cache_dir):
        raise OSError("offline")

    big, report = run(get=bad_get)
    assert report["status"] == "no_candidates" and all(q["status"] == "failed" for q in report["queries"])
    assert big["extension"] == {}


def test_queries_use_frontier_and_organism():
    calls = []
    run(get=fake_get(calls))
    searches = [u for u in calls if "search?" in u]
    assert len(searches) == 2
    assert all("Escherichia+coli" in u for u in searches)
    assert any("chorismate" in u for u in searches) and any("dihydrodiol" in u for u in searches)


@pytest.mark.parametrize("env,sources,runs", [({}, ("literature",), False), ({"T2PW_EXTEND": "1"}, (), False),
                                              ({"T2PW_EXTEND": "1"}, ("literature",), True)])
def test_runs_only_when_flag_on_and_requested(monkeypatch, env, sources, runs):
    monkeypatch.delenv("T2PW_EXTEND", raising=False)
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    big = big0()
    out, report = run_if_enabled(big, sources=sources, get=fake_get(), chat_fn=fake_chat())
    assert (report is not None) == runs
    if not runs:
        assert out is big
