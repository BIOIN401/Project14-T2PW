"""V5 scope layer: LLM answer validation, keyword fallback table, rescope steps. No network."""

import json

import pytest

from t2pw.vision import scope as sc
from t2pw.vision.view import DEFAULT, FULL, PAPER_ONLY, ScopeSpec

H2 = ScopeSpec(False, 2)


def fake(paper_only, max_hops, reason="r"):
    return lambda system, user: json.dumps({"paper_only": paper_only, "max_hops": max_hops, "reason": reason})


def boom(system, user):
    raise RuntimeError("provider down")


# --- LLM path -------------------------------------------------------------------------------------------

def test_valid_llm_json_wins_over_keywords():
    assert sc.choose_scope("only what the paper shows", llm=fake(False, 3, "deep")) == (ScopeSpec(False, 3), "deep")
    assert sc.choose_scope("hello", llm=fake(True, None))[0] == PAPER_ONLY
    assert sc.rescope(DEFAULT, "full pathway", llm=fake(False, None))[0] == FULL


def test_code_fenced_json_is_accepted():
    llm = lambda s, u: '```json\n{"paper_only": false, "max_hops": 2, "reason": "x"}\n```'
    assert sc.choose_scope("x", llm=llm)[0] == H2


def test_prompt_carries_mode_and_current_scope():
    seen = {}
    sc.rescope(H2, "more", llm=lambda s, u: seen.update(system=s, user=u) or fake(False, 3)(s, u))
    assert "MODE: rescope" in seen["user"] and '"max_hops": 2' in seen["user"] and "strict JSON" in seen["system"]


@pytest.mark.parametrize("bad", [
    "not json",
    "[]",
    '{"paper_only": false}',
    '{"paper_only": false, "max_hops": 11}',
    '{"paper_only": false, "max_hops": -1}',
    '{"paper_only": false, "max_hops": 1.5}',
    '{"paper_only": false, "max_hops": true}',
    '{"paper_only": false, "max_hops": "2"}',
    '{"paper_only": "yes", "max_hops": 1}',
    '{"paper_only": 1, "max_hops": 1}',
    '{"paper_only": false, "max_hops": 1, "reason": 5}',
    "",
])
def test_invalid_llm_answer_uses_fallback(bad):
    spec, reason = sc.choose_scope("the complete pathway please", llm=lambda s, u: bad)
    assert spec == FULL and reason.startswith("keyword")
    assert sc.rescope(H2, "unrelated chatter", llm=lambda s, u: bad) == (H2, "keyword: unsure, unchanged")


def test_llm_exception_uses_fallback():
    assert sc.choose_scope("only the paper", llm=boom)[0] == PAPER_ONLY
    assert sc.rescope(DEFAULT, "more detailed", llm=boom)[0] == H2


def test_llm_timeout_uses_fallback(monkeypatch):
    import threading
    monkeypatch.setattr(sc, "LLM_TIMEOUT_S", 0.05)
    gate = threading.Event()
    assert sc.choose_scope("whole pathway", llm=lambda s, u: gate.wait(5) or "{}")[0] == FULL
    gate.set()


def test_llm_none_offline_never_touches_client(monkeypatch):
    monkeypatch.setenv("T2PW_OFFLINE_CURATOR", "1")
    assert sc._default_llm() is None
    assert sc.choose_scope("everything known")[0] == FULL
    assert sc.rescope(DEFAULT, "simpler")[0] == PAPER_ONLY


# --- choose fallback table ------------------------------------------------------------------------------

@pytest.mark.parametrize("text,want", [
    ("Draw the glycolysis pathway from this paper", DEFAULT),
    ("", DEFAULT),
    ("only what the paper shows", PAPER_ONLY),
    ("Strictly from this paper, please", PAPER_ONLY),
    ("just the paper, nothing else", PAPER_ONLY),
    ("paper only", PAPER_ONLY),
    ("the whole pathway", FULL),
    ("I want the complete pathway", FULL),
    ("show everything known about it", FULL),
    ("the entire pathway with all that is known", FULL),
    ("two steps around the paper", H2),
    ("include 3 hops of context", ScopeSpec(False, 3)),
    ("one step beyond what the paper says", ScopeSpec(False, 1)),
    ("not the whole pathway", DEFAULT),
    ("I don't want the entire pathway, just a normal view", DEFAULT),
    ("not only the paper but the whole pathway", FULL),
    ("never show the complete pathway", DEFAULT),
    ("fifteen steps around", DEFAULT),
    ("not just the paper", DEFAULT),
    ("not the whole pathway, only the paper", PAPER_ONLY),
    ("no problem, show the whole pathway", FULL),
    ("use the full text of the paper", DEFAULT),
    ("the whole paper, please", DEFAULT),
    ("everything the paper says", DEFAULT),
])
def test_choose_fallback_table(text, want):
    assert sc.fallback_choose(text)[0] == want
    assert sc.choose_scope(text, llm=boom)[0] == want


# --- rescope --------------------------------------------------------------------------------------------

@pytest.mark.parametrize("current,text,want", [
    (DEFAULT, "can you make this more detailed?", H2),
    (DEFAULT, "expand it", H2),
    (DEFAULT, "show more context", H2),
    (DEFAULT, "this is not detailed enough", H2),
    (DEFAULT, "too simple", H2),
    (H2, "more detail please", ScopeSpec(False, 3)),
    (PAPER_ONLY, "more detailed", DEFAULT),
    (FULL, "more detailed", FULL),
    (FULL, "less detailed", DEFAULT),
    (FULL, "simpler please", DEFAULT),
    (H2, "simplify", DEFAULT),
    (DEFAULT, "simpler", PAPER_ONLY),
    (DEFAULT, "less", PAPER_ONLY),
    (DEFAULT, "this is too busy", PAPER_ONLY),
    (DEFAULT, "too detailed", PAPER_ONLY),
    (DEFAULT, "only the paper", PAPER_ONLY),
    (H2, "only what the paper shows", PAPER_ONLY),
    (DEFAULT, "show me the full pathway", FULL),
    (PAPER_ONLY, "everything known", FULL),
    (PAPER_ONLY, "three steps around", ScopeSpec(False, 3)),
    (H2, "what time is it?", H2),
    (DEFAULT, "thanks, looks good", DEFAULT),
    (H2, "not the whole pathway", H2),
    (H2, "simpler but more detailed", H2),
    (H2, "not just the paper", H2),
    (DEFAULT, "no problem, show the full pathway", FULL),
    (DEFAULT, "use the full text of the paper", DEFAULT),
    (DEFAULT, "reduce the noise in the figure", DEFAULT),
    (DEFAULT, "I will prepare the figure and compare it", DEFAULT),
    (H2, "reduce the detail", DEFAULT),
    (H2, "pare it down", DEFAULT),
])
def test_rescope_fallback(current, text, want):
    assert sc.rescope(current, text, llm=boom)[0] == want


def test_rescope_steps_chain():
    s = DEFAULT
    s = sc.rescope(s, "more detailed", llm=boom)[0]
    assert s == H2
    assert sc.rescope(PAPER_ONLY, "more detailed", llm=boom)[0] == DEFAULT
    assert sc.rescope(FULL, "less detailed", llm=boom)[0] == DEFAULT  # FULL -> DEFAULT (view.less_detailed)
