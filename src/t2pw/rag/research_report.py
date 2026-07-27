"""Citation report + export bundle for research mode — presentation only.

Research mode produces no PWML. What it produces is a *candidate* pathway that a
biologist has to judge, and judging it needs three things strict mode never has
to render: which paper each step came from, how strongly it is grounded, and
what the pipeline chose not to enforce. This module turns
``(payload, TierReport, review flags, skipped FORMAT rules)`` into a citation
report and three downloadable serializations (JSON / Markdown / CSV).

Two rules shape every string here:

* **Never fabricate a locator.** No page number exists anywhere in this
  codebase — ``ingest`` hashes the window ordinal into the chunk id and keeps no
  offset — so a citation is ``"<title> (<source_id>) -- <section> section"`` plus
  the verbatim quote and the retrieval score, and ``"(section unknown)"`` when
  even the section is missing. There is no DOI field on ``CandidatePaper``
  either, so the URI is used where one is wanted.
* **Never look clean when it is not.** A relaxed run that skipped fourteen
  FORMAT rules and carries six review flags must say so in the first screenful,
  not in an appendix. The Markdown opens with a counts banner and dedicates a
  headed section to every flag, with code and JSON pointer.

Nothing is mutated: the payload is read, the tier side-car is read, and new
strings come out. The provenance walk deliberately reuses the flattener idiom
from ``streamlit_app._rag_provenance_rows`` rather than inventing a second
provenance vocabulary — the per-source facts come from
:class:`~t2pw.rag.tiers.TierAssignment.sources`, and this module only adds the
retrieval score, which tiering has no use for.
"""

from __future__ import annotations

import csv
import io
import json
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from t2pw.pipeline.export_mode import DEFAULT_EXPORT_MODE
from t2pw.rag.tiers import (
    TIER_A,
    TIER_B,
    TIER_C,
    TIER_D,
    TIER_LABELS,
    TierReport,
    assign_tiers,
)

#: Loud, non-negotiable markers. Tier D is the one that must be impossible to
#: skim past: it means the element has no provenance of any kind.
UNSOURCED_MARKER = "!! UNSOURCED - REVIEW REQUIRED !!"
SINGLE_SOURCE_MARKER = "! SINGLE-SOURCE - REVIEW REQUIRED"
TIER_MARKERS: Dict[str, str] = {TIER_C: SINGLE_SOURCE_MARKER, TIER_D: UNSOURCED_MARKER}

#: Shown verbatim in the Markdown. Honesty about what this pipeline cannot do.
LIMITATIONS: Tuple[str, ...] = (
    "A review that CITES the seed paper cannot be distinguished from an independent "
    "corroboration. Reference sections are discarded at ingest and CandidatePaper carries "
    "no DOI, so there is no citation graph to check. When a review corroborates a step, "
    "treat it as 'possibly the same claim restated', never as a second independent finding.",
    "No page numbers exist in this pipeline. Chunk positions are hashed away at ingest, so "
    "every citation below gives the section plus the verbatim quoted passage and nothing "
    "more. Any page reference you need must be looked up in the source paper by hand.",
    "rag_confidence is an internal heuristic (the top retrieval score, or 0.5 + 0.1 per "
    "provenance pointer when no score exists). It is NOT a probability and NOT a "
    "percentage; 0.6 usually just means 'one pointer, no retrieval score'.",
    "A tier is a claim about EVIDENCE, not about IDENTITY. A Tier B element is corroborated "
    "but still has no external identifier, and a Tier A identifier was accepted only when it "
    "was a genuine match (best-effort guesses and the PathBank Unknown sentinel are refused).",
    "Research mode relaxed the PathWhiz FORMAT rules listed above and emitted no PWML. This "
    "report is a review artifact, not an import-ready pathway.",
)

_CSV_COLUMNS: Tuple[str, ...] = (
    "kind",
    "pointer",
    "name",
    "tier",
    "tier_label",
    "equation",
    "identifier",
    "identifier_source",
    "distinct_paper_count",
    "is_review_only",
    "rag_confidence",
    "sources",
    "quotes",
    "flags",
    "notes",
)


def _as_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_list(value: Any) -> List[Any]:
    return value if isinstance(value, list) else []


def _text(value: Any) -> str:
    return str(value if value is not None else "").strip()


def _score(value: Any) -> Optional[float]:
    try:
        return round(float(value), 4)
    except (TypeError, ValueError):
        return None


def _resolve_pointer(payload: Dict[str, Any], pointer: str) -> Dict[str, Any]:
    """Walk a JSON pointer back into the payload (``{}`` when it does not resolve)."""

    node: Any = payload
    for token in pointer.split("/")[1:]:
        token = token.replace("~1", "/").replace("~0", "~")
        if isinstance(node, dict):
            node = node.get(token)
        elif isinstance(node, list):
            try:
                node = node[int(token)]
            except (ValueError, IndexError):
                return {}
        else:
            return {}
    return _as_dict(node)


def _participant(entry: Any) -> str:
    """Render one reaction participant (a bare string, or ``{name, stoichiometry}``)."""

    if not isinstance(entry, dict):
        return _text(entry)
    name = _text(entry.get("name"))
    stoich = _text(entry.get("stoichiometry"))
    return f"{stoich} {name}".strip() if stoich and stoich not in {"1", "1.0"} else name


def _equation(row: Dict[str, Any]) -> str:
    inputs = [p for p in (_participant(i) for i in _as_list(row.get("inputs"))) if p]
    outputs = [p for p in (_participant(o) for o in _as_list(row.get("outputs"))) if p]
    return f"{' + '.join(inputs) or '(no substrates)'} -> {' + '.join(outputs) or '(no products)'}"


def _score_index(payload: Dict[str, Any]) -> Dict[str, float]:
    """``source_id -> best retrieval score`` seen anywhere in this payload.

    Scores live only on ``evidence`` records and tiering discards them, so they
    are re-collected here. Reported as "best score recorded for this source in
    this run" — deliberately not per-passage precise, because an evidence record
    carries no stable locator to key a per-passage score to.
    """

    best: Dict[str, float] = {}
    rows: List[Dict[str, Any]] = []
    for reaction in _as_list(_as_dict(payload.get("processes")).get("reactions")):
        rows.append(_as_dict(reaction))
        rows.extend(_as_dict(a) for a in _as_list(_as_dict(reaction).get("enzymes")))
    for bucket in _as_dict(payload.get("entities")).values():
        rows.extend(_as_dict(row) for row in _as_list(bucket))
    for row in rows:
        for record in _as_list(row.get("evidence")):
            source_id = _text(_as_dict(record).get("source_id"))
            value = _score(_as_dict(record).get("score"))
            if source_id and value is not None and value > best.get(source_id, -1.0):
                best[source_id] = value
    return best


def _title_index(rag_result: Any) -> Dict[str, str]:
    """``source_id -> title`` from the fetched candidates, as a display fallback."""

    return {
        _text(getattr(c, "id", "")): _text(getattr(c, "title", ""))
        for c in (getattr(rag_result, "candidates", None) or [])
        if _text(getattr(c, "id", ""))
    }


def _citation(source: Dict[str, Any], scores: Dict[str, float], titles: Dict[str, str]) -> Dict[str, Any]:
    """One renderable citation. NEVER contains a page number — none exists."""

    source_id = _text(source.get("source_id"))
    title = _text(source.get("title")) or titles.get(source_id, "") or "(untitled source)"
    section = _text(source.get("section"))
    return {
        "source_id": source_id,
        "title": title,
        "uri": _text(source.get("uri")),
        "section": section or "(section unknown)",
        "quote": _text(source.get("quote")),
        "retrieval_score": scores.get(source_id),
        "document_type": _text(source.get("document_type")),
        "kind": _text(source.get("label")),
        "is_review": bool(source.get("is_review")),
        "retrieved": bool(source.get("retrieved")),
        "citation": (
            f"{title} ({source_id or 'no source id'}) -- "
            f"{section + ' section' if section else '(section unknown)'}"
        ),
    }


def _flag(issue: Any) -> Dict[str, str]:
    data = _as_dict(issue)
    return {
        "code": _text(data.get("code")) or "(no code)",
        "pointer": _text(data.get("pointer") or data.get("path")) or "(no pointer)",
        "message": _text(data.get("message") or data.get("detail") or data.get("reason")),
        "stage": _text(data.get("stage")),
        "category": _text(data.get("research_category")),
    }


def _element_flags(pointer: str, flags: Sequence[Dict[str, str]]) -> List[Dict[str, str]]:
    return [f for f in flags if pointer and f["pointer"].startswith(pointer)]


@dataclass
class ResearchReport:
    """The rendered citation report plus its three export serializations."""

    pathway: str = ""
    mode: str = DEFAULT_EXPORT_MODE
    summary: Dict[str, Any] = field(default_factory=dict)
    lines: List[str] = field(default_factory=list)
    reactions: List[Dict[str, Any]] = field(default_factory=list)
    entities: List[Dict[str, Any]] = field(default_factory=list)
    sources: List[Dict[str, Any]] = field(default_factory=list)
    review_flags: List[Dict[str, str]] = field(default_factory=list)
    format_gaps: List[Dict[str, str]] = field(default_factory=list)
    tiers: Dict[str, Any] = field(default_factory=dict)
    limitations: List[str] = field(default_factory=lambda: list(LIMITATIONS))

    @property
    def elements(self) -> List[Dict[str, Any]]:
        """Every tiered element, reactions first — the CSV's row set."""

        return [*self.reactions, *self.entities]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "citations": {"reactions": self.reactions, "entities": self.entities},
            "format_gaps": self.format_gaps,
            "limitations": self.limitations,
            "pathway": {"name": self.pathway, "export_mode": self.mode},
            "review_flags": self.review_flags,
            "sources": self.sources,
            "summary": self.summary,
            "tiers": self.tiers,
        }

    def to_json(self) -> str:
        """Machine-readable, stable key order (sorted) so diffs stay reviewable."""

        return json.dumps(self.to_dict(), indent=2, sort_keys=True, ensure_ascii=False)

    def to_markdown(self) -> str:
        return "\n".join(_markdown_blocks(self))

    def to_csv(self) -> str:
        """One header row plus exactly one row per tiered element."""

        buffer = io.StringIO()
        writer = csv.DictWriter(buffer, fieldnames=list(_CSV_COLUMNS), lineterminator="\n")
        writer.writeheader()
        for element in self.elements:
            citations = element["sources"]
            row = {column: element.get(column, "") for column in _CSV_COLUMNS}
            row.update(
                rag_confidence=_confidence_text(element["rag_confidence"]),
                sources=" | ".join(c["citation"] for c in citations) or "(none)",
                quotes=" | ".join(c["quote"] for c in citations if c["quote"]),
                flags=" | ".join(f"{f['code']}@{f['pointer']}" for f in element["flags"]),
                notes=" | ".join(element["notes"]),
            )
            writer.writerow(row)
        return buffer.getvalue()


def _confidence_text(value: Optional[float]) -> str:
    """Confidence is a heuristic; it is never rendered bare or as a percentage."""

    return "" if value is None else f"{value} (heuristic, not a probability)"


def build_citation_report(
    payload: Any,
    *,
    tier_report: Optional[TierReport] = None,
    review_flags: Iterable[Any] = (),
    format_gaps: Iterable[Any] = (),
    rag_result: Any = None,
) -> ResearchReport:
    """Render a **pre-merge** synthesized payload into a reviewable citation report.

    ``payload`` must be the pre-merge payload (``rag_result.payload``); the merged
    one has had every RAG carrier stripped by ``pipeline._clean_processes``. When
    ``tier_report`` is omitted, tiers are assigned offline (no network). Both flag
    lists come from ``export_mode.review_flags`` / ``export_mode.format_gaps``.
    """

    payload_dict = _as_dict(payload)
    report = tier_report if tier_report is not None else assign_tiers(payload_dict, rag_result=rag_result)
    scores = _score_index(payload_dict)
    titles = _title_index(rag_result)
    flags = [_flag(item) for item in review_flags]
    gaps = [_flag(item) for item in format_gaps]

    reactions: List[Dict[str, Any]] = []
    entities: List[Dict[str, Any]] = []
    all_sources: Dict[str, Dict[str, Any]] = {}

    for assignment in report.assignments:
        row = _resolve_pointer(payload_dict, assignment.pointer)
        citations = [_citation(s, scores, titles) for s in assignment.sources]
        for citation in citations:
            all_sources.setdefault(citation["source_id"] or citation["citation"], citation)
        element = {
            "pointer": assignment.pointer,
            "kind": assignment.kind,
            "name": assignment.name,
            "equation": _equation(row) if assignment.kind == "reaction" else "",
            "tier": assignment.tier,
            "tier_label": assignment.label,
            "marker": TIER_MARKERS.get(assignment.tier, ""),
            "reason": assignment.reason,
            "identifier": assignment.identifier,
            "identifier_source": assignment.identifier_source,
            "distinct_paper_count": assignment.distinct_paper_count,
            "is_review_only": assignment.is_review_only,
            "rag_confidence": _score(row.get("rag_confidence")),
            "sources": citations,
            "notes": list(assignment.notes),
            "flags": _element_flags(assignment.pointer, flags),
        }
        element["line"] = _element_line(element)
        (reactions if assignment.kind == "reaction" else entities).append(element)

    counts = report.counts
    summary = {
        "element_count": len(reactions) + len(entities),
        "reaction_count": len(reactions),
        "entity_count": len(entities),
        "tier_counts": counts,
        "tier_labels": dict(TIER_LABELS),
        "unsourced_count": counts.get(TIER_D, 0),
        "review_required_count": counts.get(TIER_C, 0) + counts.get(TIER_D, 0),
        "review_flag_count": len(flags),
        "skipped_format_rule_count": len(gaps),
        "distinct_sources_cited": len(all_sources),
        "page_numbers_available": False,
        "pwml_emitted": False,
    }

    return ResearchReport(
        pathway=_pathway_name(payload_dict),
        mode=report.mode,
        summary=summary,
        lines=[element["line"] for element in reactions],
        reactions=reactions,
        entities=entities,
        sources=sorted(all_sources.values(), key=lambda c: c["source_id"]),
        review_flags=flags,
        format_gaps=gaps,
        tiers=report.to_dict(),
    )


def _pathway_name(payload: Dict[str, Any]) -> str:
    for candidate in (_as_dict(payload.get("pathway")).get("name"), payload.get("name"), payload.get("pathway_name")):
        if _text(candidate):
            return _text(candidate)
    return "(unnamed research pathway)"


def _element_line(element: Dict[str, Any]) -> str:
    """The one-line-per-element rendering the reviewer scans."""

    subject = element["name"] or "(unnamed)"
    body = f"{subject}: {element['equation']}" if element["equation"] else subject
    marker = f"   {element['marker']}" if element["marker"] else ""
    citations = "; ".join(c["citation"] for c in element["sources"]) or "NONE"
    return f"{body}   [Tier {element['tier']}]{marker}   sources: {citations}"


def _source_bullets(element: Dict[str, Any], indent: str = "  ") -> List[str]:
    lines: List[str] = []
    for citation in element["sources"]:
        detail = [citation["citation"]]
        if citation["uri"]:
            detail.append(f"<{citation['uri']}>")
        if citation["retrieval_score"] is not None:
            detail.append(f"[retrieval score {citation['retrieval_score']}]")
        detail.append(f"[{citation['kind']}]")
        lines.append(f"{indent}- {' '.join(detail)}")
        if citation["quote"]:
            lines.append(f"{indent}  > \"{citation['quote']}\"")
        else:
            lines.append(f"{indent}  > (no verbatim passage recorded for this source)")
    if not lines:
        lines.append(f"{indent}- {UNSOURCED_MARKER} no supporting passage of any kind")
    return lines


def _element_section(heading: str, elements: Sequence[Dict[str, Any]], empty: str) -> List[str]:
    """One Markdown section: headline, why-this-tier, supporting passages, flags."""

    out: List[str] = [f"## {heading}", ""]
    if not elements:
        return out + [empty, ""]
    for element in elements:
        identity = (
            f" -- identifier `{element['identifier']}` ({element['identifier_source']})"
            if element["identifier"]
            else ""
        )
        out.append(f"- {element['line']}{identity}")
        out.append(f"  - why: Tier {element['tier']} = {element['tier_label']} -- {element['reason']}")
        out.append("  - supporting passages:")
        out += _source_bullets(element, indent="    ")
        out += [f"  - note: {note}" for note in element["notes"]]
        out += [f"  - FLAG `{f['code']}` at `{f['pointer']}`: {f['message']}" for f in element["flags"]]
        if element["rag_confidence"] is not None:
            out.append(f"  - rag_confidence: {_confidence_text(element['rag_confidence'])}")
    return out + [""]


def _flag_table(flags: Sequence[Dict[str, str]], empty: str) -> List[str]:
    if not flags:
        return [empty, ""]
    rows = ["| Code | JSON pointer | Message |", "| --- | --- | --- |"]
    rows += [
        f"| `{f['code']}` | `{f['pointer']}` | {f['message'].replace('|', '/') or '(no message)'} |"
        for f in flags
    ]
    return rows + [""]


def _markdown_blocks(report: ResearchReport) -> List[str]:
    summary = report.summary
    counts = summary["tier_counts"]
    needs_review = summary["review_required_count"] or summary["review_flag_count"]
    out: List[str] = [
        f"# Research-mode citation report -- {report.pathway}",
        "",
        "## READ THIS FIRST",
        "",
        (
            f"**{'REVIEW REQUIRED' if needs_review else 'NO TIER C/D ELEMENTS'}** -- "
            f"export mode `{report.mode}`. Research mode relaxes PathWhiz FORMAT rules and turns "
            "the remaining checks into non-blocking annotations. **This report passing is not the "
            "same as the pathway being validated.** No PWML was emitted."
        ),
        "",
        "| Signal | Count |",
        "| --- | --- |",
        f"| Tier A -- {TIER_LABELS[TIER_A]} | {counts.get(TIER_A, 0)} |",
        f"| Tier B -- {TIER_LABELS[TIER_B]} | {counts.get(TIER_B, 0)} |",
        f"| Tier C -- {TIER_LABELS[TIER_C]} | {counts.get(TIER_C, 0)} |",
        f"| Tier D -- {TIER_LABELS[TIER_D]} | {counts.get(TIER_D, 0)} |",
        f"| Review flags (non-blocking, UNRESOLVED) | {summary['review_flag_count']} |",
        f"| PathWhiz FORMAT rules SKIPPED | {summary['skipped_format_rule_count']} |",
        f"| Reactions / entities | {summary['reaction_count']} / {summary['entity_count']} |",
        f"| Distinct sources cited | {summary['distinct_sources_cited']} |",
        "",
    ]
    if summary["unsourced_count"]:
        out += [f"> {UNSOURCED_MARKER} {summary['unsourced_count']} element(s) have no provenance at all.", ""]

    out += _element_section("Reactions", report.reactions, "(no reactions)")
    out += _element_section("Entities", report.entities, "(no entities)")

    out += [
        f"## Review flags -- {summary['review_flag_count']} BIOLOGY/PROVENANCE finding(s), NOT resolved",
        "",
        "These checks ran and failed. Research mode recorded them instead of aborting.",
        "",
    ]
    out += _flag_table(report.review_flags, "None. Every BIOLOGY/PROVENANCE check passed.")

    out += [
        f"## Skipped PathWhiz FORMAT rules -- {summary['skipped_format_rule_count']}",
        "",
        "These rules exist only so the PathWhiz importer accepts a file. They were NOT "
        "enforced, and are listed so the skip is never invisible.",
        "",
    ]
    out += _flag_table(report.format_gaps, "None. No FORMAT rule was skipped.")

    out += ["## Known limitations -- read before trusting a tier", ""]
    out += [f"{index}. {text}" for index, text in enumerate(report.limitations, start=1)]
    out.append("")
    return out


__all__ = [
    "LIMITATIONS",
    "SINGLE_SOURCE_MARKER",
    "TIER_MARKERS",
    "UNSOURCED_MARKER",
    "ResearchReport",
    "build_citation_report",
]
