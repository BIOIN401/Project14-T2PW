"""Offline eligibility dry-run over a stored batch plan.

What this does
--------------
Reads a run directory's ``plan.json``, re-screens every paper it lists with
``t2pw.rag.eligibility``, and prints which papers *would* be accepted or rejected
and why. Nothing is fetched, no LLM is called, and the run directory is never
written to -- the report goes to stdout and, with ``--out``, to a JSON file of
your choosing.

Where the abstract comes from, in priority order
------------------------------------------------
1. ``--abstracts`` -- an explicit ``{paper_id: abstract}`` JSON map.
2. **The plan or skipped record itself.** A run written on or after 2026-07-29
   persists the exact
   text each decision was taken on under
   ``eligibility.screening_input``. Both accepted plan records and screened
   rejected records in ``skipped.json`` are replayed. This is the normal path.
3. **The offline acquisition cache.** Legacy plans predate persisted screening
   inputs, but their publisher abstracts may still exist under
   ``data/rag_index/acquire_cache``.
4. **Derived from the stored full text.** If neither persisted source has an
   abstract, the run directory may have ``papers/<slug>/01_source_text.txt``.
   :func:`derive_abstract` recovers a bounded lead window from it -- see its
   docstring for exactly how honest that is.
5. Title only.

Every decision records which of those it used in
``screening_input.abstract_source``, and the report says so, because a verdict
from a real abstract and a verdict from a title are not the same claim. Anything
short of a real abstract stays marked ``provisional``.

Usage::

    python scripts/eligibility_dry_run.py runs/2026-07-28_2122
    python scripts/eligibility_dry_run.py runs/2026-07-28_2122 --title-only
    python scripts/eligibility_dry_run.py runs/2026-07-28_2122 --out report.json
    python scripts/eligibility_dry_run.py runs/2026-07-28_2122 --abstracts abs.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from t2pw.rag.eligibility import (  # noqa: E402
    OUTCOME_ELIGIBLE,
    OUTCOME_PINNED_OVERRIDE,
    EligibilityDecision,
    EligibilityThresholds,
    RequestedScope,
    screen_paper,
    thresholds_from_config,
)

WIDTH = 100

#: Where a screening abstract came from. Recorded on every decision.
SOURCE_SUPPLIED = "supplied_map"
SOURCE_PLAN = "plan_screening_input"
SOURCE_ACQUISITION_CACHE = "acquisition_cache"
SOURCE_DERIVED = "derived_from_stored_full_text"
SOURCE_NONE = "title_only"

# ---------------------------------------------------------------------------
# Abstract recovery from a stored full text (legacy plans only).
# ---------------------------------------------------------------------------
#: How much lead text to keep. An abstract runs 1-3k characters; this window is
#: the abstract plus, usually, the first sentences of the introduction.
DERIVED_ABSTRACT_CHARS = 1800

#: Sentences that are licence/authorship boilerplate rather than science.
_BOILERPLATE = re.compile(
    r"creative commons|permits unrestricted|permits use|permits non|properly cited"
    r"|properly attributed|open[- ]?access article|all rights reserved|creativecommons"
    r"|you do not have permission|full version of this article|for correspondence"
    r"|this article is available from|conceptualization|methodology|data curation"
    r"|formal analysis|orcid\.org|@|ror\.org|is licensed|under a licen"
    r"|terms of the|reproduction in any medium|issue-copyright",
    re.I,
)

#: Function words. A window rich in these is running prose; a window with none is
#: the journal/author/affiliation front matter that a PMC XML-to-text dump starts
#: with. This is what locates the abstract in a file that has no "Abstract" tag.
_FUNCTION_WORDS = frozenset(
    "the of and in is are to a for with that by as from this we which on be been "
    "has have was were it an these their our but not or can may than when where "
    "after before during between both such at into via using".split()
)

#: Tokens that mark a window as front matter even when it reads like prose.
_NON_PROSE = frozenset(
    "department university universidad institute institut hospital laboratory "
    "laboratoire faculty school college centre center clinic orcid "
    "conceptualization methodology supervision validation visualization "
    "affiliation email correspondence".split()
)

_PROSE_WINDOW = 20
_PROSE_FUNCTION_RATIO = 0.20


def _strip(token: str) -> str:
    return token.lower().strip(".,;:()[]")


def _prose_start(tokens: List[str]) -> int:
    """Index of the first token of running prose in ``tokens`` (0 if never)."""
    for index in range(0, max(1, len(tokens) - _PROSE_WINDOW)):
        window = tokens[index : index + _PROSE_WINDOW]
        if sum(1 for t in window if any(c.isdigit() for c in t)) > 1:
            continue
        if any(_strip(t) in _NON_PROSE for t in window):
            continue
        ratio = sum(1 for t in window if _strip(t) in _FUNCTION_WORDS) / len(window)
        if ratio >= _PROSE_FUNCTION_RATIO:
            return index
    return 0


def derive_abstract(full_text: str, *, limit: int = DERIVED_ABSTRACT_CHARS) -> str:
    """Recover a bounded abstract-like lead window from a stored full text.

    **This is a proxy, not a parsed abstract.** ``01_source_text.txt`` is a flat
    XML-to-text dump with no abstract markup, so this: jumps past an ``Abstract`` /
    ``Summary`` heading when one exists; otherwise skips the journal/author/
    affiliation front matter by finding where running prose starts; then drops
    licence and contributor-role sentences and keeps ``limit`` characters.

    Measured on the 28 papers of ``runs/2026-07-28_2122`` it recovers clean lead
    text for 26; the other two leak an affiliation or contribution-statement
    fragment, which adds institution names but no pathway or reaction vocabulary,
    so it cannot manufacture a false accept. Decisions taken on a derived abstract
    stay marked ``provisional`` for exactly this reason.

    Deterministic: a pure function of the input string.
    """
    text = re.sub(r"\s+", " ", str(full_text or "")).strip()
    if not text:
        return ""
    head = text[:30000]
    marker = re.search(r"\b(ABSTRACT|Abstract|Summary|SUMMARY)\b", head)
    if marker:
        head = head[marker.end() :]
    tokens = head.split(" ")
    head = " ".join(tokens[_prose_start(tokens) :])

    kept: List[str] = []
    total = 0
    for sentence in re.split(r"(?<=[.!?])\s+", head):
        if total >= limit:
            break
        if len(sentence) < 20 or _BOILERPLATE.search(sentence):
            continue
        digits = sum(c.isdigit() for c in sentence) / max(1, len(sentence))
        if digits > 0.15:
            continue
        kept.append(sentence)
        total += len(sentence) + 1
    return " ".join(kept)[:limit]


# ---------------------------------------------------------------------------
# Plan reading.
# ---------------------------------------------------------------------------
def _load_plan(run_dir: Path) -> Dict[str, Any]:
    path = run_dir / "plan.json"
    if not path.exists():
        raise SystemExit(f"no plan.json in {run_dir}")
    return json.loads(path.read_text(encoding="utf-8"))


def _load_screened_records(
    run_dir: Path, plan: Dict[str, Any]
) -> List[Tuple[Dict[str, Any], str]]:
    """Accepted plan records plus reproducible screened records in skipped.json.

    Old ``skipped.json`` entries that have no persisted eligibility input are
    intentionally ignored: they may be acquisition errors or duplicates, and
    inventing a new eligibility decision for them would not reproduce history.
    """
    records: List[Tuple[Dict[str, Any], str]] = [
        (record, "plan")
        for record in (plan.get("papers") or [])
        if isinstance(record, dict)
    ]
    skipped_path = run_dir / "skipped.json"
    try:
        skipped = json.loads(skipped_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        skipped = []
    for record in skipped if isinstance(skipped, list) else []:
        if not isinstance(record, dict):
            continue
        eligibility = record.get("eligibility")
        screening = (
            eligibility.get("screening_input")
            if isinstance(eligibility, dict)
            else None
        )
        if isinstance(screening, dict):
            records.append((record, "skipped"))
    return records


def _load_acquisition_abstracts() -> Dict[str, str]:
    """Best locally cached abstract for each paper id; never performs I/O off disk."""
    cache_dir = ROOT / "data" / "rag_index" / "acquire_cache"
    abstracts: Dict[str, str] = {}
    if not cache_dir.is_dir():
        return abstracts
    for path in sorted(cache_dir.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        for row in payload.get("candidates") or [] if isinstance(payload, dict) else []:
            if not isinstance(row, dict):
                continue
            paper_id = str(row.get("id") or "").strip()
            abstract = str(row.get("abstract") or "").strip()
            if paper_id and abstract and len(abstract) > len(abstracts.get(paper_id, "")):
                abstracts[paper_id] = abstract
    return abstracts


def _requested_scope(record: Dict[str, Any]) -> RequestedScope:
    """The requested scope for one plan record.

    ``requested_pathway`` / ``requested_organism`` are the current spelling;
    ``topic`` / ``organism`` are what plans written before this change carry (and
    in those, ``organism`` WAS the requested organism -- it was stamped onto every
    paper regardless of what the paper reported, which is the bug the split fixes).
    """
    return RequestedScope(
        requested_pathway=str(
            record.get("requested_pathway") or record.get("topic") or ""
        ),
        requested_organism=str(
            record.get("requested_organism") or record.get("organism") or ""
        ),
        pinned=not str(record.get("topic") or record.get("requested_pathway") or ""),
    )


def _abstract_for(
    record: Dict[str, Any],
    run_dir: Path,
    supplied: Dict[str, str],
    cached: Dict[str, str],
    *,
    title_only: bool,
) -> Tuple[str, str, bool]:
    """``(abstract, source, authoritative)`` in deterministic priority order."""
    paper_id = str(record.get("paper_id") or "")
    if title_only:
        return "", SOURCE_NONE, False
    if supplied.get(paper_id):
        return str(supplied[paper_id]), SOURCE_SUPPLIED, True

    stored = record.get("eligibility")
    if isinstance(stored, dict):
        screening = stored.get("screening_input")
        if isinstance(screening, dict) and str(screening.get("abstract") or "").strip():
            if "abstract_authoritative" in screening:
                authoritative = bool(screening["abstract_authoritative"])
            else:
                original_source = str(screening.get("abstract_source") or "")
                authoritative = original_source not in {SOURCE_DERIVED, SOURCE_NONE}
            return str(screening["abstract"]), SOURCE_PLAN, authoritative

    if cached.get(paper_id):
        return str(cached[paper_id]), SOURCE_ACQUISITION_CACHE, True

    slug = str(record.get("slug") or "")
    if slug:
        source_text = run_dir / "papers" / slug / "01_source_text.txt"
        if source_text.exists():
            try:
                raw = source_text.read_text(encoding="utf-8", errors="replace")
            except OSError:
                raw = ""
            derived = derive_abstract(raw)
            if derived:
                return derived, SOURCE_DERIVED, False
    return "", SOURCE_NONE, False


def dry_run(
    run_dir: Path,
    *,
    thresholds: Optional[EligibilityThresholds] = None,
    abstracts: Optional[Dict[str, str]] = None,
    title_only: bool = False,
) -> Dict[str, Any]:
    """Screen every paper in ``run_dir``'s plan; return a JSON-able report."""
    plan = _load_plan(run_dir)
    limits = thresholds or thresholds_from_config()
    supplied = abstracts or {}
    cached = {} if title_only else _load_acquisition_abstracts()
    decisions: List[EligibilityDecision] = []
    record_sources: List[str] = []
    sources: Dict[str, int] = {}
    screened_records = _load_screened_records(run_dir, plan)
    for record, record_source in screened_records:
        abstract, source, authoritative = _abstract_for(
            record, run_dir, supplied, cached, title_only=title_only
        )
        sources[source] = sources.get(source, 0) + 1
        record_sources.append(record_source)
        decisions.append(
            screen_paper(
                paper_id=str(record.get("paper_id") or ""),
                title=str(record.get("title") or ""),
                abstract=abstract,
                scope=_requested_scope(record),
                thresholds=limits,
                abstract_source=source,
                # A derived abstract is a proxy for the real one, so the verdict is
                # provisional and the review-flag rules for provisional verdicts
                # apply -- which is why this is threaded in rather than stamped on
                # afterwards.
                abstract_is_authoritative=authoritative,
            )
        )

    by_outcome: Dict[str, int] = {}
    by_class: Dict[str, int] = {}
    for decision in decisions:
        by_outcome[decision.outcome] = by_outcome.get(decision.outcome, 0) + 1
        by_class[decision.classification] = by_class.get(decision.classification, 0) + 1

    return {
        "run_dir": str(run_dir),
        "plan_created_at": plan.get("created_at"),
        "papers_in_plan": len(plan.get("papers") or []),
        "screened_records": len(screened_records),
        "screened_records_from_skipped": record_sources.count("skipped"),
        "abstract_sources": dict(sorted(sources.items())),
        "title_only": sources.get(SOURCE_NONE, 0) == len(decisions) and bool(decisions),
        "provisional": any(d.provisional for d in decisions),
        "thresholds": limits.to_dict(),
        "outcomes": dict(sorted(by_outcome.items())),
        "classifications": dict(sorted(by_class.items())),
        "accepted": [d.paper_id for d in decisions if d.eligible],
        "rejected": [d.paper_id for d in decisions if not d.eligible],
        "needs_manual_review": [d.paper_id for d in decisions if d.needs_manual_review],
        "decisions": [
            {**d.to_dict(), "title": d.title, "record_source": record_source}
            for d, record_source in zip(decisions, record_sources)
        ],
    }


def render(report: Dict[str, Any]) -> str:
    """Human-readable form of :func:`dry_run`'s report."""
    out: List[str] = []
    out.append("=" * WIDTH)
    out.append("ELIGIBILITY DRY-RUN (offline; nothing fetched, no LLM, run dir untouched)")
    out.append("=" * WIDTH)
    out.append(f"run          : {report['run_dir']}")
    out.append(f"plan created : {report.get('plan_created_at') or '(unknown)'}")
    out.append(f"papers       : {report['papers_in_plan']}")
    out.append("abstracts    : " + (
        ", ".join(f"{k}={v}" for k, v in report["abstract_sources"].items()) or "(none)"
    ))
    if report["provisional"]:
        out.append(
            "               PROVISIONAL -- some or all verdicts were taken without a "
            "publisher abstract"
        )
        out.append(
            f"               ('{SOURCE_DERIVED}' is a bounded lead window recovered "
            "from the stored full text)."
        )
    out.append("")
    out.append("thresholds")
    out.append("-" * WIDTH)
    for key, value in report["thresholds"].items():
        out.append(f"  {key:36} : {value}")
    out.append("")
    out.append("outcomes")
    out.append("-" * WIDTH)
    for outcome, count in report["outcomes"].items():
        out.append(f"  {outcome:26} : {count}")
    out.append("")
    out.append("classifications")
    out.append("-" * WIDTH)
    for name, count in report["classifications"].items():
        out.append(f"  {name:26} : {count}")
    out.append("")

    accepting = {OUTCOME_ELIGIBLE, OUTCOME_PINNED_OVERRIDE}
    for heading, wanted in (("ACCEPTED", True), ("REJECTED", False)):
        rows = [d for d in report["decisions"] if (d["outcome"] in accepting) is wanted]
        out.append(heading + f"  ({len(rows)})")
        out.append("-" * WIDTH)
        for d in rows:
            flag = "  [REVIEW]" if d["needs_manual_review"] else ""
            out.append(
                f"  {d['paper_id']}  score {d['score']:>6}  {d['outcome']}"
                f"  [{d['classification']}]{flag}"
            )
            out.append(f"      title    : {d['title'][:WIDTH - 18]}")
            out.append(
                f"      requested: {d['requested_pathway']} / "
                f"{d['requested_organism'] or '(none)'}"
            )
            out.append(
                f"      observed : organisms={', '.join(d['observed_organisms']) or '(none)'}"
                f"  match={d['organism_match']}"
            )
            out.append(
                "      input    : "
                f"{d['screening_input']['abstract_source']}"
                f" ({d['screening_input']['abstract_chars']} abstract chars"
                + (
                    f", sha {d['screening_input']['abstract_sha256'][:12]}"
                    if d["screening_input"]["abstract_sha256"]
                    else ""
                )
                + ")"
            )
            if d["matched_positive"]:
                out.append(f"      positive : {', '.join(d['matched_positive'][:8])}")
            if d["matched_negative"]:
                out.append(f"      negative : {', '.join(d['matched_negative'][:8])}")
            out.append(f"      reason   : {d['reason']}")
            out.append("")
        out.append("")

    review = report["needs_manual_review"]
    out.append(f"NEEDS MANUAL INSPECTION  ({len(review)})")
    out.append("-" * WIDTH)
    if review:
        out.append(
            "  Screening found some signal but could not decide. Check each abstract "
            "before"
        )
        out.append("  trusting the verdict above.")
        for paper_id in review:
            match = next(
                (d for d in report["decisions"] if d["paper_id"] == paper_id), None
            )
            title = (match or {}).get("title", "")
            outcome = (match or {}).get("outcome", "")
            out.append(f"  - {paper_id}  ({outcome})  {title[:WIDTH - 40]}")
    else:
        out.append("  (none)")
    return "\n".join(out) + "\n"


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", help="a run directory containing plan.json")
    parser.add_argument("--out", default="", help="also write the JSON report here")
    parser.add_argument(
        "--abstracts",
        default="",
        help="JSON file mapping paper_id -> abstract (highest priority source)",
    )
    parser.add_argument(
        "--title-only",
        action="store_true",
        help="ignore every abstract source and screen titles alone",
    )
    args = parser.parse_args(argv)

    abstracts: Dict[str, str] = {}
    if args.abstracts:
        raw = json.loads(Path(args.abstracts).read_text(encoding="utf-8"))
        if isinstance(raw, dict):
            abstracts = {str(k): str(v or "") for k, v in raw.items()}

    report = dry_run(
        Path(args.run_dir),
        abstracts=abstracts or None,
        title_only=bool(args.title_only),
    )
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001 - older/redirected stdio
        pass
    print(render(report))
    if args.out:
        Path(args.out).write_text(
            json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
