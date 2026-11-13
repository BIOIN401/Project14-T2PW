"""D-W20c: DNA/RNA type of nucleic acids read from the paper text (no network, no LLM, fakes only)."""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from test_final_payload_writer_fixes import _build, _na_types, _payload  # noqa: E402
from t2pw.pwml.writer import nucleotide_type_for  # noqa: E402
from t2pw.service import final_payload as fp  # noqa: E402
from t2pw.service import final_payload_nucleic as fn  # noqa: E402


def _p(*rows: dict) -> dict:
    return {"entities": {"nucleic_acids": [dict(r) for r in rows]}}


def _run(text: str, *rows: dict, **kw):
    p = _p(*rows)
    rep = fn.apply_paper_nucleic_types(p, text, today="2026-11-12", **kw)
    return p, rep


def test_rna_cue_resolves_with_quoted_provenance() -> None:
    text = "Intro sentence. Under high iron the RyhB sRNA represses enterobactin synthesis. Other text."
    p, rep = _run(text, {"name": "RyhB"})
    row = p["entities"]["nucleic_acids"][0]
    assert row["nucleotide_type_paper"] == "RNA" and rep["resolved"] == 1
    e = p["final_payload_fill"]["entries"][0]
    assert e["source"] == "paper" and e["cue"] == "srna" and e["offset"] == text.index("RyhB")
    assert e["quote"] == "Under high iron the RyhB sRNA represses enterobactin synthesis."[: len(e["quote"])] or "RyhB sRNA" in e["quote"]
    assert p["final_payload_fill"]["written"] == 1


def test_dna_cue_and_whole_token_case_insensitive() -> None:
    p, _ = _run("The pora promoter binds RVE1. The PORAX gene is unrelated.", {"name": "PORA"})
    assert p["entities"]["nucleic_acids"][0]["nucleotide_type_paper"] == "DNA"
    p, rep = _run("The PORAX gene is unrelated.", {"name": "PORA"})  # PORAX is not PORA
    assert rep["not_mentioned"] == 1 and "nucleotide_type_paper" not in p["entities"]["nucleic_acids"][0]


def test_synonym_is_a_mention() -> None:
    p, _ = _run("The HY5 locus was deleted.", {"name": "Elongated hypocotyl", "synonyms": ["HY5"]})
    assert p["entities"]["nucleic_acids"][0]["nucleotide_type_paper"] == "DNA"


def test_conflicting_mentions_are_not_voted() -> None:
    text = "The Xa gene is active. The Xa mRNA is stable. The Xa gene again. The Xa locus too."
    p, rep = _run(text, {"name": "Xa"})
    assert rep["conflicting"] == 1 and rep["resolved"] == 0 and "final_payload_fill" not in p


def test_conflict_inside_one_window_and_far_cue_ignored() -> None:
    p, rep = _run("The Xa gene mRNA level rose.", {"name": "Xa"})
    assert rep["conflicting"] == 1
    p, rep = _run("The Xa was measured in many conditions over several weeks, and the gene list grew.", {"name": "Xa"})
    assert rep["no_cue"] == 1  # 'gene' is far outside the two-token window


def test_whole_sentence_window_is_available_and_stricter_on_neighbours() -> None:
    text = "The CHLH promoter and the miR171c transcript were measured."
    assert _run(text, {"name": "miR171c"})[1]["resolved"] == 1  # 'transcript' adjacent
    assert _run(text, {"name": "miR171c"}, window=None)[1]["conflicting"] == 1


def test_no_mention_no_cue_no_text() -> None:
    assert _run("Nothing here.", {"name": "Xa"})[1]["not_mentioned"] == 1
    assert _run("Xa was measured.", {"name": "Xa"})[1]["no_cue"] == 1
    p, rep = _run("", {"name": "Xa"})
    assert rep["no_paper_text"] == 1 and p == _p({"name": "Xa"})
    assert fn.apply_paper_nucleic_types(_p({"name": "Xa"}), None)["no_paper_text"] == 1


def test_guards_ids_name_settled_and_protein_like() -> None:
    text = "The Xa gene. The Ya gene. The Za mRNA. DNA polymerase gene. The Wa gene."
    p, rep = _run(text, {"name": "Xa", "pathbank_nucleic_acid_id": 5}, {"name": "Za mRNA"}, {"name": "DNA polymerase"},
                  {"name": "Wa", "pathwhiz_id": 9}, {"name": "Ya"})
    assert rep["candidates"] == 2 and rep["protein_like"] == 1 and rep["resolved"] == 1  # only Y resolved
    rows = p["entities"]["nucleic_acids"]
    assert "nucleotide_type_paper" not in rows[0] and "nucleotide_type_paper" not in rows[3]
    assert "nucleotide_type_paper" not in rows[1] and rows[4]["nucleotide_type_paper"] == "DNA"


def test_long_sentence_quote_is_truncated_around_the_mention() -> None:
    text = ("filler word " * 80) + "see the Qq promoter here " + ("tail word " * 80) + "."
    p, _ = _run(text, {"name": "Qq"})
    q = p["final_payload_fill"]["entries"][0]["quote"]
    assert "Qq promoter" in q and len(q) <= fn.QUOTE_MAX + 6


def test_existing_fill_block_is_extended_not_replaced() -> None:
    p = _p({"name": "Xa"})
    p["final_payload_fill"] = {"schema": 1, "date": "d", "written": 2, "by_field": {"compound.chebi_id": 2}, "entries": [{"a": 1}, {"a": 2}]}
    fn.apply_paper_nucleic_types(p, "The Xa gene.", today="2026-11-12")
    blk = p["final_payload_fill"]
    assert blk["written"] == 3 and len(blk["entries"]) == 3 and blk["by_field"]["compound.chebi_id"] == 2


def test_failure_leaves_payload_untouched(monkeypatch) -> None:
    p = _p({"name": "Xa"})
    before = copy.deepcopy(p)
    monkeypatch.setattr(fn, "classify", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    rep = fn.apply_paper_nucleic_types(p, "The Xa gene.")
    assert p == before and rep["error"].startswith("RuntimeError") and rep["resolved"] == 0


def test_seam_flag_off_does_nothing_and_flag_on_reads_state(monkeypatch) -> None:
    p = _p({"name": "Xa"})
    monkeypatch.delenv("T2PW_FINAL_PAYLOAD", raising=False)
    assert fp.maybe_apply_final_id_fill(p, {}, {"final_payload_paper_text": "The Xa gene."}) == {} and p == _p({"name": "Xa"})
    monkeypatch.setenv("T2PW_FINAL_PAYLOAD", "1")
    rep = fp.maybe_apply_paper_nucleic_types(p, {"final_payload_paper_text": "The Xa gene.",
                                                 "final_payload_paper_text_source": "01_source_text.txt"})
    assert rep["resolved"] == 1 and p["final_payload_fill"]["entries"][0]["source_file"] == "01_source_text.txt"
    assert fp.maybe_apply_paper_nucleic_types(_p({"name": "Xa"}), {})["no_paper_text"] == 1


def test_writer_accepts_paper_type_only_with_flag_and_no_id(monkeypatch) -> None:
    assert nucleotide_type_for({"name": "CHLH promoter"}) is None
    assert nucleotide_type_for({"name": "CHLH promoter", "nucleotide_type_paper": "DNA"}) == "DNA"
    assert nucleotide_type_for({"name": "RNA chaperone", "nucleotide_type_paper": "bogus"}) is None

    def payload(**extra):
        pl = _payload()
        pl["entities"]["nucleic_acids"] = [{"name": "CHLH promoter", "nucleotide_type_paper": "DNA", **extra}]
        return pl

    monkeypatch.setenv("T2PW_FINAL_PAYLOAD", "1")
    builder, root = _build(payload())
    assert _na_types(root) == {"CHLH promoter": "DNA"}
    assert "nucleic_acids_type_unset" not in builder.layout_debug_counts
    _, root = _build(payload(pathbank_nucleic_acid_id=7))  # existing PathWhiz row: never typed
    assert _na_types(root) == {"CHLH promoter": ""}
    monkeypatch.delenv("T2PW_FINAL_PAYLOAD")
    _, root = _build(payload())  # flag off: byte-identical behaviour, no type
    assert _na_types(root) == {"CHLH promoter": ""}
