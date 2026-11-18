"""generate(): every format from one scoped view. Offline: a stored 1-reaction run payload (no DB, no network, no LLM)."""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from t2pw.vision import generate as G
from t2pw.vision.payload import build_big_payload, merge_extension
from t2pw.vision.view import DEFAULT, FULL, PAPER_ONLY, view

FIX = Path(__file__).parent / "data" / "vision" / "generate" / "run56_final_mapped.json"
OPTS = {"strict_db": True, "export_mode": "pathwhiz"}


def big():
    b = build_big_payload(json.loads(FIX.read_text(encoding="utf-8")))
    # one extension reaction between two base compounds: one hop out, not from the paper
    return merge_extension(b, {"processes.reactions": [
        {"name": "ext step", "inputs": ["sulfacetamide"], "outputs": ["meropenem"], "evidence": "ext"}]})


def gen(b, scope, fmt, **kw):
    return G.generate(b, scope, fmt, export=OPTS, **kw)


@pytest.fixture(autouse=True)
def _fresh_memo():
    G._MEMO.clear()
    yield
    G._MEMO.clear()


def test_scope_parse():
    assert G.parse_scope("paper") is PAPER_ONLY and G.parse_scope("default") == DEFAULT and G.parse_scope("full") == FULL
    assert G.parse_scope("hops:2").max_hops == 2 and not G.parse_scope("hops:2").paper_only
    with pytest.raises(ValueError):
        G.parse_scope("everything")


def test_every_format_and_unknown_format():
    b = big()
    for fmt in G.FORMATS:
        out = gen(b, DEFAULT, fmt)
        assert isinstance(out, bytes) and out
    assert gen(b, DEFAULT, "pwml").lstrip().startswith(b"<?xml")
    assert gen(b, DEFAULT, "svg").lstrip().startswith(b"<svg")
    assert b"<html" in gen(b, DEFAULT, "html")
    assert json.loads(gen(b, DEFAULT, "json")) == view(b, DEFAULT)
    with pytest.raises(ValueError):
        gen(b, DEFAULT, "docx")


def test_same_view_same_bytes_and_cache(tmp_path, monkeypatch):
    b = big()
    first = {f: gen(b, DEFAULT, f, cache_dir=tmp_path) for f in G.FORMATS}
    assert len(list(tmp_path.iterdir())) == len(G.FORMATS)
    G._MEMO.clear()
    assert {f: gen(b, DEFAULT, f) for f in G.FORMATS} == first  # recomputed, still the same bytes

    def boom(*a, **k):
        raise AssertionError("cache hit must not recompute")
    monkeypatch.setattr(G, "_export_pwml", boom)
    monkeypatch.setattr(G, "_sbml", boom)
    G._MEMO.clear()
    assert {f: gen(b, DEFAULT, f, cache_dir=tmp_path) for f in ("pwml", "sbml", "svg")} == {
        f: first[f] for f in ("pwml", "sbml", "svg")}


def test_pwml_export_runs_once_per_view(monkeypatch):
    b, calls, real = big(), [], G._export_pwml
    monkeypatch.setattr(G, "_export_pwml", lambda v, o: calls.append(1) or real(v, o))
    for f in ("pwml", "svg", "html", "sbgn"):
        gen(b, DEFAULT, f)
    assert len(calls) == 1


def test_default_view_is_the_stored_payload():
    raw = json.loads(FIX.read_text(encoding="utf-8"))
    assert view(build_big_payload(raw), DEFAULT) == raw


def n_pwml(x):
    return len(re.findall(rb"<reaction>", x))


def test_scope_reaches_every_format():
    b = big()
    assert [r["name"] for r in view(b, PAPER_ONLY)["processes"]["reactions"]] == ["PSA conversion to sulfacetamide"]
    assert len(view(b, DEFAULT)["processes"]["reactions"]) == 2
    import libsbml
    sb = {}
    for tag, sc in (("paper", PAPER_ONLY), ("default", DEFAULT)):
        pw = n_pwml(gen(b, sc, "pwml"))
        sbml = libsbml.readSBMLFromString(gen(b, sc, "sbml").decode()).getModel().getNumReactions()
        sbgn = len(re.findall(rb'class="process"', gen(b, sc, "sbgn")))
        kgml = len(re.findall(rb"<reaction\b", gen(b, sc, "kgml")))
        sb[tag] = (pw, sbml, sbgn, kgml)
        assert pw == sbml == sbgn == kgml, (tag, pw, sbml, sbgn, kgml)
    assert sb["paper"] == (1, 1, 1, 1) and sb["default"] == (2, 2, 2, 2)
    for f in ("svg", "html"):
        assert gen(b, PAPER_ONLY, f) != gen(b, DEFAULT, f)


def test_cache_key_includes_env_switches(tmp_path, monkeypatch):
    b = big()
    gen(b, DEFAULT, "sbml", cache_dir=tmp_path)
    monkeypatch.setenv("T2PW_PWML_LAYOUT", "v1")
    gen(b, DEFAULT, "sbml", cache_dir=tmp_path)
    monkeypatch.setenv("T2PW_FINAL_PAYLOAD", "0")
    gen(b, DEFAULT, "sbml", cache_dir=tmp_path)
    assert len(list(tmp_path.glob("*.sbml"))) == 3 and not list(tmp_path.glob("*.tmp"))


def test_refused_export_raises(monkeypatch):
    monkeypatch.setattr(G, "_export_pwml", lambda v, o: (None, {}, "refused"))
    with pytest.raises(G.GenerateError):
        gen(big(), DEFAULT, "pwml")
    assert gen(big(), DEFAULT, "sbgn")  # sbgn falls back to its own grid
    G._MEMO.clear()
    with pytest.raises(G.GenerateError):
        gen(big(), DEFAULT, "svg")


def test_cli(tmp_path):
    src = tmp_path / "big.json"
    src.write_text(json.dumps(big()), encoding="utf-8")
    out = tmp_path / "o.sbml"
    assert G.main([str(src), "--scope", "paper", "--fmt", "sbml", "--out", str(out)]) == 0
    assert out.read_bytes() == gen(big(), PAPER_ONLY, "sbml")
