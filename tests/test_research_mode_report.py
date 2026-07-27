from __future__ import annotations

import csv
import io
import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from t2pw.pipeline.export_mode import RESEARCH, format_gaps, relax_report, review_flags  # noqa: E402
from t2pw.rag.research_report import (  # noqa: E402
    UNSOURCED_MARKER,
    build_citation_report,
)
from t2pw.rag.tiers import assign_tiers  # noqa: E402


#: Any page-style citation. Nothing in this pipeline may ever produce one.
_PAGE_PATTERNS = (
    re.compile(r"\bpp?\.\s*\d", re.IGNORECASE),
    re.compile(r"\bpages?\s+\d", re.IGNORECASE),
)


# ---------------------------------------------------------------------------
# Payload builders.
# ---------------------------------------------------------------------------
def _prov(source_id: str, *, chunk_id: str = "", title: str = "", section: str = "results") -> dict:
    return {
        "source_id": source_id,
        "source_title": title or f"{source_id} study",
        "source_type": "paper",
        "source_uri": f"https://example.org/{source_id}",
        "section": section,
        "organism": "lactococcus lactis",
        "chunk_id": chunk_id,
    }


def _evidence(source_id: str, *, chunk_id: str, text: str, score: float = 0.83) -> dict:
    return {
        "text": text,
        "source_id": source_id,
        "source_uri": f"https://example.org/{source_id}",
        "section": "results",
        "score": score,
        "chunk_id": chunk_id,
    }


def _paper(source_id: str) -> dict:
    return {
        "source_id": source_id,
        "title": f"{source_id} study",
        "uri": f"https://example.org/{source_id}",
        "source_type": "paper",
        "organism": "lactococcus lactis",
    }


def _sourced_reaction() -> dict:
    return {
        "name": "menaquinone demethylation",
        "inputs": ["menaquinone", {"name": "SAM", "stoichiometry": "2"}],
        "outputs": ["demethylmenaquinone"],
        "enzymes": [
            {
                "entity": "MenG",
                "entity_type": "protein",
                "role": "catalyst",
                "mapped_ids": {"uniprot": "P0A6C0"},
                "rag_provenance": _prov("PMC1", chunk_id="a" * 32),
                "evidence": [_evidence("PMC1", chunk_id="a" * 32, text="MenG converts menaquinone.")],
                "source_papers": [_paper("PMC1")],
                "source_refs": ["PMC1"],
                "rag_confidence": 0.83,
            }
        ],
        "rag_provenance": _prov("PMC1", chunk_id="a" * 32),
        "evidence": [
            _evidence("PMC1", chunk_id="a" * 32, text="MenG converts menaquinone."),
            _evidence("PMC2", chunk_id="b" * 32, text="Demethylation was confirmed.", score=0.71),
        ],
        "source_papers": [_paper("PMC1"), _paper("PMC2")],
        "source_refs": ["PMC1", "PMC2"],
        "rag_confidence": 0.83,
    }


def _unsourced_reaction() -> dict:
    return {"name": "mystery step", "inputs": ["X"], "outputs": ["Y"]}


def _payload(*, reactions=None, compounds=None, proteins=None) -> dict:
    return {
        "entities": {
            "species": [{"name": "Lactococcus lactis"}],
            "compounds": list(compounds or []),
            "proteins": list(proteins or []),
        },
        "processes": {"reactions": list(reactions or [])},
    }


def _scored(document_type: str = ""):
    return SimpleNamespace(
        selection=SimpleNamespace(
            scores={
                "PMC1": SimpleNamespace(document_type=document_type),
                "PMC2": SimpleNamespace(document_type=document_type),
            }
        ),
        candidates=[],
    )


def _relaxed_report() -> dict:
    strict = {
        "stage": "post_extraction",
        "ok": False,
        "errors": [
            {"code": "generated_wrapper_missing_components", "pointer": "/entities/protein_complexes/0",
             "message": "wrapper has no components"},
            {"code": "reaction_enzyme_unresolved", "pointer": "/processes/reactions/0/enzymes/0",
             "message": "enzyme has no external identity"},
        ],
        "warnings": [],
    }
    return relax_report(strict)


def _build(payload, rag_result=None):
    relaxed = _relaxed_report()
    return build_citation_report(
        payload,
        tier_report=assign_tiers(payload, rag_result=rag_result, mode=RESEARCH),
        review_flags=review_flags(relaxed),
        format_gaps=format_gaps(relaxed),
        rag_result=rag_result,
    )


# ---------------------------------------------------------------------------
# Tests.
# ---------------------------------------------------------------------------
def test_reaction_line_renders_name_equation_tier_and_a_source():
    report = _build(_payload(reactions=[_sourced_reaction()]), _scored())

    assert len(report.lines) == 1
    line = report.lines[0]
    assert line.startswith("menaquinone demethylation: ")
    assert "menaquinone + 2 SAM -> demethylmenaquinone" in line
    assert re.search(r"\[Tier [ABCD]\]", line)
    assert "sources: " in line and "PMC1" in line
    assert line.split("sources: ", 1)[1] != "NONE"


def test_tier_d_element_renders_loud_unsourced_marker():
    report = _build(_payload(reactions=[_unsourced_reaction()]))

    element = report.reactions[0]
    assert element["tier"] == "D"
    assert UNSOURCED_MARKER in element["line"]
    markdown = report.to_markdown()
    assert UNSOURCED_MARKER in markdown
    assert "no supporting passage of any kind" in markdown


def test_markdown_surfaces_review_flags_and_skipped_format_rules():
    report = _build(_payload(reactions=[_sourced_reaction()]), _scored())
    markdown = report.to_markdown()

    assert markdown.startswith("# Research-mode citation report")
    banner = markdown.split("## Reactions", 1)[0]
    assert "Review flags (non-blocking, UNRESOLVED) | 1" in banner
    assert "PathWhiz FORMAT rules SKIPPED | 1" in banner

    assert "## Review flags" in markdown
    assert "reaction_enzyme_unresolved" in markdown
    assert "## Skipped PathWhiz FORMAT rules" in markdown
    assert "generated_wrapper_missing_components" in markdown
    # The honest limitation must be stated, not buried in code comments.
    assert "cannot be distinguished from an independent" in markdown
    assert "Reference sections are discarded at ingest" in markdown


def test_json_round_trips_with_expected_top_level_keys():
    report = _build(_payload(reactions=[_sourced_reaction()]), _scored())
    data = json.loads(report.to_json())

    assert set(data) == {
        "citations",
        "format_gaps",
        "limitations",
        "pathway",
        "review_flags",
        "sources",
        "summary",
        "tiers",
    }
    assert data["citations"]["reactions"][0]["name"] == "menaquinone demethylation"
    assert data["summary"]["review_flag_count"] == 1
    assert data["summary"]["skipped_format_rule_count"] == 1
    assert data["summary"]["page_numbers_available"] is False
    assert data["tiers"]["counts"]
    # Stable key order: serializing twice is byte-identical.
    assert report.to_json() == json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False)


def test_csv_has_one_header_row_and_one_row_per_element():
    payload = _payload(
        reactions=[_sourced_reaction(), _unsourced_reaction()],
        compounds=[{"name": "menaquinone", "rag_provenance": _prov("PMC1", chunk_id="a" * 32)}],
    )
    report = _build(payload, _scored())

    rows = list(csv.reader(io.StringIO(report.to_csv())))
    assert rows[0][0] == "kind"
    assert len(rows) == 1 + len(report.elements)
    assert len(report.elements) == len(report.reactions) + len(report.entities)
    kinds = [row[0] for row in rows[1:]]
    assert kinds.count("reaction") == 2
    assert set(kinds) <= {"reaction", "entity"}


def test_rag_confidence_is_never_presented_as_a_probability():
    report = _build(_payload(reactions=[_sourced_reaction()]), _scored())
    markdown = report.to_markdown()

    assert "rag_confidence: 0.83 (heuristic, not a probability)" in markdown
    assert "0.83%" not in markdown
    assert "83%" not in markdown
    assert "NOT a probability and NOT a" in markdown


def test_no_page_numbers_are_ever_fabricated():
    payload = _payload(
        reactions=[_sourced_reaction(), _unsourced_reaction()],
        proteins=[{"name": "MenG", "rag_provenance": _prov("PMC1", chunk_id="a" * 32)}],
    )
    report = _build(payload, _scored())

    for text in (report.to_markdown(), report.to_json(), report.to_csv(), *report.lines):
        for pattern in _PAGE_PATTERNS:
            assert not pattern.search(text), f"fabricated page citation: {pattern.pattern}"

    # Sections are cited instead, and a missing section says so rather than guessing.
    citation = report.reactions[0]["sources"][0]["citation"]
    assert "results section" in citation
    assert "retrieval score" in report.to_markdown()


def test_missing_section_says_section_unknown_and_never_invents_one():
    reaction = _sourced_reaction()
    reaction["rag_provenance"] = _prov("PMC9", chunk_id="c" * 32, section="")
    reaction["evidence"] = [dict(_evidence("PMC9", chunk_id="c" * 32, text="claim"), section="")]
    reaction["source_papers"] = [_paper("PMC9")]
    reaction["source_refs"] = ["PMC9"]
    reaction["enzymes"] = []
    report = _build(_payload(reactions=[reaction]))

    citations = report.reactions[0]["sources"]
    assert any(c["citation"].endswith("(section unknown)") for c in citations)


def test_degenerate_payload_still_produces_a_usable_report():
    report = build_citation_report(None)

    assert report.lines == []
    assert report.summary["element_count"] == 0
    markdown = report.to_markdown()
    assert "(no reactions)" in markdown and "(no entities)" in markdown
    assert "None. No FORMAT rule was skipped." in markdown
    rows = list(csv.reader(io.StringIO(report.to_csv())))
    assert len(rows) == 1 and rows[0][0] == "kind"


def test_review_source_is_labelled_and_flags_attach_to_their_element():
    report = _build(_payload(reactions=[_sourced_reaction()]), _scored("multi_example_review"))

    enzyme = next(e for e in report.entities if e["pointer"].endswith("/enzymes/0"))
    assert any("review" in c["kind"] for c in enzyme["sources"])
    assert any(f["code"] == "reaction_enzyme_unresolved" for f in enzyme["flags"])


# ---------------------------------------------------------------------------
# The readable .txt report -- the format an actual reviewer opens.
# ---------------------------------------------------------------------------
def _readable(**kwargs) -> str:
    payload = _payload(
        reactions=[_sourced_reaction(), _unsourced_reaction()],
        proteins=[{"name": "MenG", "rag_provenance": _prov("PMC1", chunk_id="a" * 32)}],
    )
    return build_citation_report(
        payload, rag_result=_scored("primary_research"), **kwargs
    ).to_text()


def test_text_report_numbers_every_reaction_with_substrates_and_products():
    text = _readable()

    assert "REACTION 1 of 2" in text
    assert "REACTION 2 of 2" in text
    assert "  IN " in text and "  OUT " in text
    assert "menaquinone demethylation" in text
    assert "demethylmenaquinone" in text
    # Stoichiometry is preserved in the participant list.
    assert "2 SAM" in text


def test_text_report_gives_each_reaction_its_own_evidence_block():
    text = _readable()
    first = text.split("REACTION 2 of 2")[0]

    assert "EVIDENCE" in first
    assert "MenG converts menaquinone." in first
    assert "section: results" in first


def test_text_report_shows_the_catalyst_with_its_own_tier():
    text = _readable(identity_source={
        "entities": {"proteins": [{"name": "MenG", "mapped_ids": {"uniprot": "P0A6C0"}}]}
    })

    assert "ENZYME  MenG" in text
    assert "uniprot: P0A6C0" in text


def test_text_report_makes_an_unsourced_reaction_impossible_to_miss():
    text = _readable()

    assert UNSOURCED_MARKER in text
    assert "WHAT TO CHECK FIRST" in text
    assert "mystery step" in text.split("WHAT TO CHECK FIRST")[1]


def test_text_report_never_fabricates_a_page_number():
    text = _readable()

    for pattern in _PAGE_PATTERNS:
        assert not pattern.search(text), f"fabricated page citation: {pattern.pattern}"


def test_text_report_does_not_list_a_quotation_as_a_paper():
    """A verbatim quote in source_refs must never be numbered like a citation."""

    reaction = _sourced_reaction()
    reaction["source_refs"] = ["PMC1", "MenG converts menaquinone to demethylmenaquinone."]
    text = build_citation_report(
        _payload(reactions=[reaction]), rag_result=_scored("primary_research")
    ).to_text()

    body = text.split("EVIDENCE")[1].split("=" * 80)[0]
    assert "NOT additional papers" in body
    # The quotation must not appear as a numbered source entry.
    assert "[2] MenG converts menaquinone to demethylmenaquinone" not in body


def test_text_report_does_not_repeat_catalysts_in_the_entity_appendix():
    text = _readable()
    appendix = text.split("ENTITIES BY TIER")[1]

    assert "Catalysts are listed with their reaction" in appendix
    # MenG is tiered twice -- once as a declared protein, once as the enzyme
    # actor of its reaction. The actor is rendered under the reaction, so the
    # appendix must list the protein exactly once rather than showing a
    # duplicate the reviewer would read as two different entities.
    assert appendix.count("  - MenG") == 1


def test_text_report_states_the_limitations():
    text = _readable()

    assert "KNOWN LIMITATIONS" in text
    assert "page numbers" in text.lower()
    assert "review" in text.lower()
