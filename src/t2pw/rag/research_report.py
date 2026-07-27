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
import textwrap
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


def _participants(row: Dict[str, Any], side: str) -> List[str]:
    """The substrate or product names of a reaction, as a list.

    Kept as a list rather than re-split from the rendered equation: compound
    names legitimately contain ``+`` (``Ca2+``, ``NAD+``), so splitting the
    equation string back apart would silently invent extra participants.
    """

    return [p for p in (_participant(i) for i in _as_list(row.get(side))) if p]


def _equation(row: Dict[str, Any]) -> str:
    inputs = _participants(row, "inputs")
    outputs = _participants(row, "outputs")
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
    # Boundary-aware: a bare startswith makes "/processes/reactions/1" swallow
    # every flag on "/processes/reactions/10".."/19", attributing a biology
    # problem to an element that has none.
    return [
        f
        for f in flags
        if pointer
        and (f["pointer"] == pointer or f["pointer"].startswith(pointer + "/"))
    ]


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

    def to_text(self) -> str:
        """A plain-text report meant to be read top to bottom by a human.

        Numbered steps with substrates, products and catalyst, each followed by
        its own evidence block. This is the format for a reviewer who is not
        going to open JSON — the other three serializations are for machines,
        spreadsheets, and rendering respectively.
        """

        return "\n".join(_text_blocks(self))

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
    mode: Any = DEFAULT_EXPORT_MODE,
    identity_source: Any = None,
) -> ResearchReport:
    """Render a **pre-merge** synthesized payload into a reviewable citation report.

    ``payload`` must be the pre-merge payload (``rag_result.payload``); the merged
    one has had every RAG carrier stripped by ``pipeline._clean_processes``. When
    ``tier_report`` is omitted, tiers are assigned offline (no network). Both flag
    lists come from ``export_mode.review_flags`` / ``export_mode.format_gaps``.
    """

    payload_dict = _as_dict(payload)
    report = (
        tier_report
        if tier_report is not None
        else assign_tiers(
            payload_dict,
            rag_result=rag_result,
            mode=mode,
            identity_source=identity_source,
        )
    )
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
            "inputs": _participants(row, "inputs") if assignment.kind == "reaction" else [],
            "outputs": _participants(row, "outputs") if assignment.kind == "reaction" else [],
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
        pathway=_pathway_name(payload_dict, sorted(all_sources.values(), key=lambda c: c["source_id"])),
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


def _pathway_name(payload: Dict[str, Any], sources: Sequence[Dict[str, Any]] = ()) -> str:
    """The pathway's display name, falling back to the seed paper's title.

    ``metadata.pathway_name`` is where the pipeline actually stores it; the
    other keys are older shapes. When none is set, the seed paper's title is a
    far better heading than "(unnamed)" -- it is what the user uploaded.
    """

    for candidate in (
        _as_dict(payload.get("metadata")).get("pathway_name"),
        _as_dict(payload.get("pathway")).get("name"),
        payload.get("name"),
        payload.get("pathway_name"),
    ):
        if _text(candidate):
            return _text(candidate)
    for source in sources:
        source = _as_dict(source)
        if str(source.get("source_id")) == "seed_paper" and _text(source.get("title")):
            return _text(source.get("title"))
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


WIDTH = 80
_RULE = "=" * WIDTH
_THIN = "-" * WIDTH

#: What each tier means, in a reviewer's words rather than the pipeline's.
_TIER_GUIDE: Tuple[Tuple[str, str], ...] = (
    (TIER_A, "an external database ID was found (UniProt / DrugBank / PathBank)"),
    (TIER_B, "no ID, but two independent sources state it, each with its own quote"),
    (TIER_C, "only ONE source states this. Read the quote and decide."),
    (TIER_D, "NOTHING backs this. Check these first."),
)


def _wrap(text: str, indent: str, width: int = WIDTH) -> List[str]:
    """Wrap a paragraph to ``width``, prefixing every line with ``indent``."""

    body = " ".join(str(text or "").split())
    if not body:
        return []
    return textwrap.wrap(
        body, width=width, initial_indent=indent, subsequent_indent=indent
    ) or [indent + body]


def _enzymes_by_reaction(entities: Sequence[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """Group tiered enzyme actors under their parent reaction pointer.

    Enzyme actors are tiered as their own elements (pointer
    ``/processes/reactions/3/enzymes/0``) because their grounding is independent
    of the reaction's. For a readable report they belong next to the step they
    catalyse, so they are folded back here.
    """

    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for element in entities:
        pointer = str(element.get("pointer") or "")
        marker = "/enzymes/"
        if marker in pointer:
            grouped.setdefault(pointer.split(marker)[0], []).append(element)
    return grouped


def _tier_tag(element: Dict[str, Any]) -> str:
    """``Tier C - single-source`` plus the identifier when there is one."""

    tier = str(element.get("tier") or "?")
    tag = f"Tier {tier} - {element.get('tier_label') or ''}".rstrip(" -")
    identifier = str(element.get("identifier") or "")
    if identifier:
        source = str(element.get("identifier_source") or "id")
        tag = f"{tag} [{source}: {identifier}]"
    return tag


def _evidence_lines(element: Dict[str, Any]) -> List[str]:
    """The per-element evidence block: who says so, where, and in what words."""

    out: List[str] = []
    sources = [_as_dict(s) for s in _as_list(element.get("sources"))]
    if not sources:
        out.append("    (none — nothing in the evidence store supports this)")
        return out

    # A pointer that is neither a retrieved paper nor the seed sentinel is a
    # verbatim quotation that Stage 1 wrote into source_refs. Numbering it
    # alongside real papers reads as a citation to a paper that does not exist,
    # so it is reported separately and never counted.
    def _identified(source: Dict[str, Any]) -> bool:
        return bool(source.get("retrieved")) or str(source.get("source_id")) == "seed_paper"

    papers = [s for s in sources if _identified(s)]
    quotations = [s for s in sources if not _identified(s)]

    if not papers:
        out.append("    (no identifiable source — see the note below)")

    for idx, source in enumerate(papers, start=1):
        title = str(source.get("title") or "(untitled source)")
        source_id = str(source.get("source_id") or "?")
        seed = source_id == "seed_paper"
        label = "seed paper" if seed else source_id
        heading = _wrap(f"{title}  ({label})", "        ")
        if heading:
            heading[0] = f"    [{idx}] " + heading[0].lstrip()
        out.extend(heading)

        section = str(source.get("section") or "")
        detail = "section: not recorded" if not section or "unknown" in section else f"section: {section}"
        score = source.get("retrieval_score")
        if score is not None:
            detail += f"   ·   best retrieval score for this source: {score}"
        out.append(f"        {detail}")

        if source.get("is_review"):
            out.append(
                "        NOTE: this source is a review — it may be restating the seed"
            )
            out.append("              rather than reporting an independent finding.")

        quote = str(source.get("quote") or "")
        if quote:
            wrapped = _wrap(quote, "         ", WIDTH - 1)
            if wrapped:
                wrapped[0] = '        "' + wrapped[0].lstrip()
                wrapped[-1] = wrapped[-1] + '"'
            out.extend(wrapped)
        elif seed:
            out.append(
                "        (no quoted passage: the seed paper is not chunked into the"
            )
            out.append("         evidence store, so only the reaction claim is recorded)")
        else:
            out.append("        (no quoted passage recorded for this source)")

        uri = str(source.get("uri") or "")
        if uri:
            out.append(f"        {uri}")
        out.append("")

    if quotations:
        out.append(
            f"    Also recorded: {len(quotations)} quoted passage(s) carried as free text"
        )
        out.append("    rather than a paper id. These are NOT additional papers and were")
        out.append("    not counted toward corroboration:")
        for source in quotations[:3]:
            snippet = " ".join(str(source.get("source_id") or "").split())
            out.extend(_wrap(f'"{snippet[:160]}"', "        "))
        if len(quotations) > 3:
            out.append(f"        ... and {len(quotations) - 3} more")
        out.append("")
    return out


def _reaction_block(
    element: Dict[str, Any],
    number: int,
    total: int,
    enzymes: Sequence[Dict[str, Any]],
) -> List[str]:
    """One reaction, rendered the way a biologist reads a pathway step."""

    out: List[str] = [_THIN, f"REACTION {number} of {total}   ·   {_tier_tag(element)}"]
    marker = str(element.get("marker") or "")
    if marker:
        out.append(f"    {marker}")
    out.append(_THIN)
    out.append(f"  {element.get('name') or '(unnamed reaction)'}")
    out.append("")

    inputs = [str(v) for v in _as_list(element.get("inputs"))]
    outputs = [str(v) for v in _as_list(element.get("outputs"))]
    if not inputs and not outputs and element.get("equation"):
        # Older reports carry only the rendered equation. Show each side whole
        # rather than splitting on " + " -- compound names contain it (Ca2+,
        # NAD+), so splitting would invent participants that were never there.
        halves = str(element["equation"]).split(" -> ", 1)
        inputs = [halves[0]] if halves[0] else []
        outputs = [halves[1]] if len(halves) > 1 and halves[1] else []
    inputs = inputs or ["(no substrates recorded)"]
    outputs = outputs or ["(no products recorded)"]
    for label, values in (("IN ", inputs), ("OUT", outputs)):
        for pos, value in enumerate(values):
            lead = f"  {label}   " if pos == 0 else "      + "
            out.extend(_wrap(value, lead, WIDTH) or [lead + value])
        out.append("")

    if enzymes:
        for enzyme in enzymes:
            out.append(f"  ENZYME  {enzyme.get('name') or '(unnamed)'}")
            out.append(f"          {_tier_tag(enzyme)}")
    else:
        out.append("  ENZYME  (none recorded)")
    out.append("")

    out.append("  WHY THIS TIER")
    out.extend(_wrap(element.get("reason"), "    "))
    for note in _as_list(element.get("notes")):
        wrapped = _wrap(str(note), "      ")
        if wrapped:
            # Hanging indent: the bullet only on the first line.
            wrapped[0] = "    - " + wrapped[0].lstrip()
            out.extend(wrapped)
    out.append("")

    flags = _as_list(element.get("flags"))
    if flags:
        out.append("  ** REVIEW FLAGS ON THIS REACTION **")
        for flag in flags:
            flag = _as_dict(flag)
            out.append(f"    - [{flag.get('code','')}] {flag.get('message','')}")
        out.append("")

    out.append("  EVIDENCE")
    out.extend(_evidence_lines(element))
    return out


def _text_blocks(report: "ResearchReport") -> List[str]:
    """The whole plain-text report, in reading order."""

    summary = _as_dict(report.summary)
    counts = _as_dict(summary.get("tier_counts"))
    reactions = list(report.reactions)
    entities = list(report.entities)
    by_reaction = _enzymes_by_reaction(entities)

    out: List[str] = [_RULE]
    out.extend(_wrap((report.pathway or "candidate pathway").upper(), ""))
    out.append(f"Candidate pathway · export mode: {report.mode} · NOT an importable PWML file")
    out.append(_RULE)
    out.append("")

    out.append("HOW TO READ THIS")
    for tier, meaning in _TIER_GUIDE:
        out.append(f"  Tier {tier}  {TIER_LABELS.get(tier, '')}")
        out.extend(_wrap(meaning, "          "))
    out.append("")
    out.append("  Every claim below is traced to a source. Nothing here is invented:")
    out.append("  if a source or a page is missing, it says so rather than guessing.")
    out.append("")

    out.append("AT A GLANCE")
    out.append(f"  Reactions            {summary.get('reaction_count', len(reactions))}")
    out.append(f"  Entities             {summary.get('entity_count', len(entities))}")
    out.append(f"  Distinct sources     {summary.get('distinct_sources_cited', 0)}")
    out.append(
        "  Tiers                "
        + "   ".join(f"{t}={counts.get(t, 0)}" for t, _ in _TIER_GUIDE)
    )
    needs = int(summary.get("review_required_count", 0) or 0)
    out.append(f"  NEEDS REVIEW         {needs} element(s) are Tier C or D")
    unsourced = int(summary.get("unsourced_count", 0) or 0)
    if unsourced:
        out.append(f"  UNSOURCED            {unsourced} element(s) have no provenance at all")
    out.append(f"  Review flags         {summary.get('review_flag_count', 0)}")
    out.append(f"  Format rules skipped {summary.get('skipped_format_rule_count', 0)}")
    out.append("")

    out.append(_RULE)
    out.append("THE PATHWAY, STEP BY STEP")
    out.append(_RULE)
    out.append("")
    if reactions:
        for number, element in enumerate(reactions, start=1):
            out.extend(
                _reaction_block(
                    element, number, len(reactions), by_reaction.get(element["pointer"], [])
                )
            )
            out.append("")
    else:
        out.append("  (no reactions in this pathway)")
        out.append("")

    out.append(_RULE)
    out.append("WHAT TO CHECK FIRST")
    out.append(_RULE)
    out.append("")
    ranked = [e for e in (*reactions, *entities) if e.get("tier") in {TIER_D, TIER_C}]
    ranked.sort(key=lambda e: (e.get("tier") != TIER_D, str(e.get("name") or "")))
    if ranked:
        for element in ranked:
            kind = "reaction" if element.get("kind") == "reaction" else "entity  "
            out.append(f"  [Tier {element.get('tier')}] {kind}  {element.get('name')}")
            out.extend(_wrap(element.get("reason"), "              "))
    else:
        out.append("  Nothing is single-source or unsourced. Every element has")
        out.append("  either a database identifier or two independent sources.")
    out.append("")

    out.append(_RULE)
    out.append("ENTITIES BY TIER")
    out.append(_RULE)
    out.append("")
    out.append("Catalysts are listed with their reaction above and not repeated here.")
    out.append("")
    # Enzyme actors are tiered as their own elements but were already rendered
    # under the step they catalyse; listing them again reads as a duplicate.
    standalone = [e for e in entities if "/enzymes/" not in str(e.get("pointer") or "")]
    for tier, _ in _TIER_GUIDE:
        members = [e for e in standalone if e.get("tier") == tier]
        out.append(f"Tier {tier} — {TIER_LABELS.get(tier, '')}   ({len(members)})")
        if not members:
            out.append("  (none)")
        for element in members:
            identifier = str(element.get("identifier") or "")
            suffix = f"   [{element.get('identifier_source')}: {identifier}]" if identifier else ""
            out.append(f"  - {element.get('name')}{suffix}")
        out.append("")

    out.append(_RULE)
    out.append("SOURCES CITED")
    out.append(_RULE)
    out.append("")
    for source in report.sources:
        source = _as_dict(source)
        source_id = str(source.get("source_id") or "?")
        label = "seed paper (uploaded)" if source_id == "seed_paper" else source_id
        out.append(f"  {label}")
        out.extend(_wrap(str(source.get("title") or "(untitled)"), "    "))
        if source.get("is_review"):
            out.append("    type: REVIEW — may restate the seed rather than corroborate it")
        if source.get("uri"):
            out.append(f"    {source.get('uri')}")
        out.append("")

    if report.review_flags or report.format_gaps:
        out.append(_RULE)
        out.append("WHAT THE PIPELINE DID NOT ENFORCE")
        out.append(_RULE)
        out.append("")
        if report.review_flags:
            out.append(f"Biology / provenance flags ({len(report.review_flags)})")
            out.append("  These would have STOPPED a strict run. They did not stop this one.")
            for flag in report.review_flags:
                flag = _as_dict(flag)
                out.append(f"  - [{flag.get('code','')}] {flag.get('pointer','')}")
                out.extend(_wrap(flag.get("message"), "      "))
            out.append("")
        if report.format_gaps:
            out.append(f"PathWhiz format rules skipped ({len(report.format_gaps)})")
            out.append("  Importer requirements only — they say nothing about the biology.")
            for gap in report.format_gaps:
                gap = _as_dict(gap)
                out.append(f"  - [{gap.get('code','')}] {gap.get('pointer','')}")
            out.append("")

    out.append(_RULE)
    out.append("KNOWN LIMITATIONS — READ BEFORE CITING THIS")
    out.append(_RULE)
    out.append("")
    for number, limitation in enumerate(report.limitations, start=1):
        out.append(f"  {number}.")
        out.extend(_wrap(limitation, "     "))
        out.append("")
    return out


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
