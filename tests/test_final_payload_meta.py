"""Final payload, title and reference (D-W12): ``final_payload_pathway``, its use by the IR and the PWML writer.
No network: the fill layer's FakeFetcher answers the NCBI esummary call."""

from __future__ import annotations

import copy
import sys
from pathlib import Path
from types import SimpleNamespace

from lxml import etree

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from clean_model_fill_fakes import FakeFetcher  # noqa: E402
from helpers_prefreeze import prefrozen  # noqa: E402
from test_final_payload_fill import _pipeline_payload  # noqa: E402
from t2pw.clean_model.fill.cache import Cache  # noqa: E402
from t2pw.clean_model.fill.http import FetchError  # noqa: E402
from t2pw.pwml.compound_resolution import NO_DB_RESOLVER  # noqa: E402
from t2pw.pwml.ir import build_pwml_ir, validate_pwml_ir  # noqa: E402
from t2pw.pwml.validate import discover_structure_signature, repair_tree  # noqa: E402
from t2pw.pwml.writer import DeterministicPwmlBuilder  # noqa: E402
from t2pw.service.final_payload import (  # noqa: E402
    FINAL_PAYLOAD_PATHWAY_FIELD as KEY, apply_final_pathway_meta, effective_pathway_name, maybe_apply_final_id_fill,
    maybe_apply_final_pathway_meta, paper_meta_from_paper,
)

TODAY = "2026-11-11"
TITLE = "Caffeine degradation by NdmA in Pseudomonas putida"
PUBMED = {"result": {"uids": ["555"], "555": {
    "title": TITLE, "pubdate": "2019 Mar 4", "authors": [{"name": "Doe J"}, {"name": "Roe R"}],
    "articleids": [{"idtype": "pubmed", "value": "555"}, {"idtype": "doi", "value": "10.1000/xyz"}, {"idtype": "pmc", "value": "PMC777"}]}}}


def _p():
    return prefrozen(_pipeline_payload(), db_resolver=NO_DB_RESOLVER)


def _apply(payload, meta, routes=None, **kw):
    f = FakeFetcher(routes if routes is not None else [("esummary", {"db": "pubmed"}, PUBMED)])
    return apply_final_pathway_meta(payload, meta, fetcher=f, cache=Cache(None), today=TODAY, **kw)


def _pwml(payload, name="Generated Pathway"):
    ir, rep = build_pwml_ir(copy.deepcopy(payload), strict_db=False)
    assert rep["errors"] == [] and validate_pwml_ir(ir)["ok"]
    ref = ROOT / "reference" / "PW000001.pwml"
    sig = discover_structure_signature(ref)
    args = SimpleNamespace(name=effective_pathway_name(payload, name), description="", subject="Metabolic", pw_id="PW000000",
                           height=1400, width=3200, background_color="#FFFFFF", ref=str(ref))
    root = repair_tree(etree.ElementTree(DeterministicPwmlBuilder(extraction=ir, signature=sig, args=args).build().root), sig).getroot()
    return ir, root


def test_paper_mode_title_and_reference_from_the_lookup_with_provenance():
    p = _p()
    rep = _apply(p, {"pmid": "555", "title": "ignored fallback"})
    assert rep["error"] is None and rep["written"] and rep["title_source"] == "NCBI"
    b = p[KEY]
    assert b["title"]["value"] == TITLE and b["title"]["match_type"] == "accession" and b["title"]["retrieved"] == TODAY
    assert (b["pmid"]["value"], b["doi"]["value"], b["year"]["value"]) == ("555", "10.1000/xyz", 2019)
    assert b["authors"]["value"] == ["Doe J", "Roe R"]
    assert b["reference"]["pubmed_id"] == "555"
    assert b["reference"]["text"] == f"Doe J, Roe R: {TITLE}. 2019. doi: 10.1000/xyz. PMID: 555."


def test_name_mode_uses_the_pathway_name_and_has_no_reference():
    p = _p()
    f = FakeFetcher([])
    rep = apply_final_pathway_meta(p, {"organism": "Pseudomonas putida"}, fetcher=f, cache=Cache(None), today=TODAY,
                                   pathway_name="  Caffeine   degradation ")
    assert rep["title_source"] == "pathway name" and not f.calls  # no id, so no lookup
    assert p[KEY]["title"]["value"] == "Caffeine degradation" and "reference" not in p[KEY] and "pmid" not in p[KEY]


def test_lookup_failure_falls_back_to_meta_title_then_pathway_name_then_nothing():
    p = _p()
    _apply(p, {"pmid": "555", "title": "Given title"}, routes=[])  # esummary answers nothing
    assert p[KEY]["title"]["value"] == "Given title" and p[KEY]["title"]["source"] == "paper metadata"
    assert p[KEY]["reference"]["pubmed_id"] == "555"
    p = _p()
    _apply(p, {"pmid": "555"}, routes=[("esummary", None, FetchError("down"))], pathway_name="The pathway")
    assert p[KEY]["title"]["source"] == "pathway name"
    p, before = _p(), copy.deepcopy(_p())
    rep = _apply(p, {"pmid": "555"}, routes=[("esummary", None, FetchError("down"))])
    assert rep["written"] is False and rep["error"] is None and p == before  # nothing known: no text invented


def test_fail_open_restores_the_payload_on_an_exception(monkeypatch):
    p, before = _p(), copy.deepcopy(_p())
    monkeypatch.setattr("t2pw.clean_model.fill.fill_model", lambda *a, **k: 1 / 0)
    rep = _apply(p, {"pmid": "555"})
    assert rep["written"] is False and "ZeroDivisionError" in rep["error"] and p == before
    assert effective_pathway_name(p, "Generated Pathway") == "Generated Pathway"


def test_guessed_pmid_is_not_looked_up():
    p = _p()
    f = FakeFetcher([("esummary", None, PUBMED)])
    apply_final_pathway_meta(p, {"pmid": "555", "pmid_source": "PMC text header", "title": "T"}, fetcher=f, cache=Cache(None), today=TODAY)
    assert not f.calls and "pmid" not in p[KEY] and "reference" not in p[KEY]


def test_effective_name_only_for_the_default_name_and_only_with_the_key():
    p = _p()
    assert effective_pathway_name(p, "Generated Pathway") == "Generated Pathway"
    p[KEY] = {"title": {"value": "  Real   title "}}
    assert effective_pathway_name(p, "Generated Pathway") == "Real title"
    assert effective_pathway_name(p, "User chosen") == "User chosen"
    assert effective_pathway_name(None, "Generated Pathway") == "Generated Pathway"


def test_ir_and_pwml_without_the_key_are_unchanged():
    ir, root = _pwml(_p())
    assert ir["pathway"]["name"] == "x" and "references" not in ir["pathway"]  # the payload's metadata name, as before
    pw = root.find(".//pathway")
    assert pw.findtext("name") == "Generated Pathway" and root.findtext("cached-name") == "Generated Pathway"
    assert len(pw.find("references")) == 0


def test_ir_and_pwml_with_the_key_carry_title_and_one_reference():
    p = _p()
    p["metadata"].pop("pathway_name")
    _apply(p, {"pmid": "555"})
    ir, root = _pwml(p)
    assert ir["pathway"]["name"] == TITLE
    assert ir["pathway"]["references"] == [{"pubmed-id": "555", "text": p[KEY]["reference"]["text"]}]
    pw = root.find(".//pathway")
    assert pw.findtext("name") == TITLE and root.findtext("cached-name") == TITLE
    refs = pw.findall("references/reference")
    assert len(refs) == 1
    r = refs[0]
    assert [c.tag for c in r] == ["id", "pubmed-id", "text", "element-id", "element-type"]
    assert r.findtext("pubmed-id") == "555" and r.findtext("text").startswith("Doe J, Roe R: " + TITLE)
    assert r.findtext("element-type") == "Pathway" and r.findtext("element-id") == pw.findtext("id")
    assert r.find("id").get("type") == "integer" and r.find("element-id").get("type") == "integer"
    # everything else in the PWML is what it was without the key
    p0 = _p()
    p0["metadata"].pop("pathway_name")
    _, root0 = _pwml(p0)
    for el in (root, root0):
        for t in ("cached-name",):
            el.find(t).text = ""
        n = el.find(".//pathway/name"); n.text = ""
        ref_c = el.find(".//pathway/references")
        for c in list(ref_c):
            ref_c.remove(c)
    assert etree.tostring(root) == etree.tostring(root0)


def test_reference_without_a_paper_id_and_a_user_chosen_name_are_respected():
    p = _p()
    _apply(p, {}, pathway_name="Only a name")
    ir, _ = _pwml(p, name="Mine")
    assert "references" not in ir["pathway"]
    assert _pwml(p, name="Mine")[1].find(".//pathway").findtext("name") == "Mine"


def test_wiring_helpers(monkeypatch):
    p, before = _p(), copy.deepcopy(_p())
    monkeypatch.delenv("T2PW_FINAL_PAYLOAD", raising=False)
    assert maybe_apply_final_pathway_meta(p, {"pathway_name": "n"}, {}) == {} and p == before
    monkeypatch.setenv("T2PW_FINAL_PAYLOAD", "1")
    state = {"final_payload_paper": {"pmid": "555", "title": "T"}}
    rep = maybe_apply_final_pathway_meta(p, {"pathway_name": "n", "likely_organism": "Pseudomonas putida"}, state,
                                         fetcher=FakeFetcher([("esummary", {"db": "pubmed"}, PUBMED)]), cache=Cache(None), today=TODAY)
    assert rep["written"] and p[KEY]["title"]["value"] == TITLE


def test_seam_runs_the_pathway_meta_after_the_id_fill(monkeypatch):
    monkeypatch.setenv("T2PW_FINAL_PAYLOAD", "1")
    seen = []
    monkeypatch.setattr("t2pw.service.final_payload.apply_final_id_fill", lambda *a, **k: seen.append("ids") or {"written": 0})
    monkeypatch.setattr("t2pw.service.final_payload.maybe_apply_final_pathway_meta", lambda *a, **k: seen.append("meta") or {"written": True})
    rep = maybe_apply_final_id_fill(_p(), {}, {})
    assert seen == ["ids", "meta"] and rep["pathway_meta"] == {"written": True}


def test_paper_meta_from_paper():
    assert paper_meta_from_paper({"paper_id": "PMC123", "title": "T", "year": 2020}) == {"pmcid": "PMC123", "title": "T", "year": "2020"}
    assert paper_meta_from_paper(SimpleNamespace(paper_id="31415", pmcid="")) == {"pmid": "31415"}
    assert paper_meta_from_paper(SimpleNamespace(paper_id="some-slug", title="")) == {}
    assert paper_meta_from_paper(None) == {}


def test_key_survives_freeze_and_quarantine(tmp_path):
    from t2pw.service.postpipeline import freeze_canonical_payload
    p = _p()
    _apply(p, {"pmid": "555"})
    r = freeze_canonical_payload(p, strict_db=False, outputs_dir=tmp_path, pathway_context={}, export_mode="pathwhiz",
                                 research_mode=False, tmp=tmp_path, state={})
    if r["quarantine_ok"]:
        assert r["payload"][KEY] == p[KEY]
