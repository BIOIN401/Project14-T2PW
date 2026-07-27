"""Human-readable reports over a batch run manifest — pure formatting.

An overnight run produces one JSON-Lines manifest row per *(paper, mode)* run.
That file is machine-shaped and useless at 8am: what the human needs is (a) a
verdict per paper, (b) a per-paper breakdown naming every artifact that got
written, and (c) a ranked fix-list saying which defect to attack first. This
module is the only place that turns rows into those strings.

The design rests on one asymmetry that makes triage cheap. Every paper is run
twice, so the *pair* of outcomes is diagnostic:

* strict PASS + research PASS -> ``ok``
* strict FAIL + research PASS -> ``format-blocked``: a PathWhiz FORMAT rule
  stopped strict export while the relaxed run still produced a candidate. This
  is expected wear, not a crash, and it is grouped by issue code so the fix is
  one rule at a time.
* strict FAIL + research FAIL -> ``broken``: the failure is upstream of export —
  a crash, an LLM refusal, a network fault, or biology the extractor cannot
  read.
* research FAIL -> **research-mode defect**. Research mode is *fail-open* by
  construction (it relaxes FORMAT rules and reports flags instead of stopping),
  so a research failure can only mean the code itself is wrong. Those are
  printed first and loudest, and a ``broken`` paper is reported in that section
  too, because its research run also failed.

Two hard constraints, both learned the hard way:

* **Never raise on bad input.** This report is often the only surviving evidence
  of a run that died at 3am. Any field may be missing, any line may be truncated
  mid-write; malformed lines are skipped and counted as ``unreadable`` so their
  absence is visible rather than silent.
* **Plain text, <= 100 columns, no dependencies.** It has to be readable in
  Notepad over a remote session, so every wrap goes through :mod:`textwrap`.

I/O is confined to :func:`load_manifest`; every other function takes rows and
returns a string.
"""

from __future__ import annotations

import json
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

#: Report width. Notepad-friendly, and the same number the pipeline reports use.
WIDTH = 100

#: Canonical mode names. The manifest may carry the app's radio labels
#: ("Strict PWML" / "Research (relaxed)") or the short slugs; both normalize here.
MODE_STRICT = "strict"
MODE_RESEARCH = "research"

#: Canonical run statuses.
STATUS_PASS = "pass"
STATUS_FAIL = "fail"
STATUS_TIMEOUT = "timeout"
STATUS_SKIPPED = "skipped"
STATUS_UNREADABLE = "unreadable"
STATUS_UNKNOWN = "unknown"

#: Triage classes, worst first. ``research-defect`` is the pure case (strict
#: passed, research did not); ``broken`` also counts as a research-mode defect
#: and is reported in that section as well.
CLASS_RESEARCH_DEFECT = "research-defect"
CLASS_BROKEN = "broken"
CLASS_FORMAT_BLOCKED = "format-blocked"
CLASS_INCOMPLETE = "incomplete"
CLASS_OK = "ok"

CLASS_ORDER: Tuple[str, ...] = (
    CLASS_RESEARCH_DEFECT,
    CLASS_BROKEN,
    CLASS_FORMAT_BLOCKED,
    CLASS_INCOMPLETE,
    CLASS_OK,
)

CLASS_BLURB: Dict[str, str] = {
    CLASS_RESEARCH_DEFECT: "strict PASS + research FAIL -- RESEARCH-MODE DEFECT",
    CLASS_BROKEN: "strict FAIL + research FAIL -- real bug, also a research defect",
    CLASS_FORMAT_BLOCKED: "strict FAIL + research PASS -- a PathWhiz FORMAT rule",
    CLASS_INCOMPLETE: "a mode is missing, skipped or unrecognised",
    CLASS_OK: "strict PASS + research PASS",
}

#: Shouted so it cannot be skimmed past.
DEFECT_MARKER = "!! RESEARCH-MODE DEFECT !!"

#: Shouted for the same reason: a run that passed and produced nothing is the one
#: outcome a status column cannot express.
WARNING_MARKER = "!! PASSED BUT PRODUCED NO DELIVERABLE !!"

#: Counts worth putting first when a run reports many of them.
_PREFERRED_COUNT_KEYS: Tuple[str, ...] = (
    "reactions",
    "entities",
    "proteins",
    "compounds",
    "tier_a",
    "tier_b",
    "tier_c",
    "tier_d",
    "review_flags",
    "flags",
    "skipped_format_rules",
    "normalization_actions",
    "citations",
)

_PASS_WORDS = frozenset({"pass", "passed", "ok", "success", "succeeded", "green", "true"})
_FAIL_WORDS = frozenset({"fail", "failed", "failure", "error", "errored", "crash", "red", "false"})
_TIMEOUT_WORDS = frozenset({"timeout", "timed_out", "timedout", "time-out", "hung"})
_SKIP_WORDS = frozenset({"skip", "skipped", "not_run", "notrun", "pending"})


# ---------------------------------------------------------------------------
# Normalisation. Everything below assumes rows are hostile: wrong types,
# missing keys, half-written lines.
# ---------------------------------------------------------------------------
def _text(value: Any, default: str = "") -> str:
    """Coerce anything to a single-line string without raising."""
    if value is None:
        return default
    if isinstance(value, str):
        out = value.strip()
    else:
        try:
            out = str(value).strip()
        except Exception:  # pragma: no cover - defensive
            return default
    out = " ".join(out.split())
    return out or default


def _norm_status(value: Any) -> str:
    raw = _text(value).lower().replace(" ", "_")
    if not raw:
        return STATUS_UNKNOWN
    if raw in _PASS_WORDS:
        return STATUS_PASS
    if raw in _TIMEOUT_WORDS or "timeout" in raw:
        return STATUS_TIMEOUT
    if raw in _SKIP_WORDS or raw.startswith("skip"):
        return STATUS_SKIPPED
    if raw in _FAIL_WORDS or "fail" in raw or "error" in raw:
        return STATUS_FAIL
    return STATUS_UNKNOWN


def _norm_mode(value: Any) -> str:
    raw = _text(value).lower()
    if not raw:
        return ""
    if "research" in raw or "relax" in raw:
        return MODE_RESEARCH
    if "strict" in raw or "pwml" in raw:
        return MODE_STRICT
    return raw


def _is_failure(status: str) -> bool:
    """A run that did not produce its artifacts. Timeouts count as failures."""
    return status in (STATUS_FAIL, STATUS_TIMEOUT, STATUS_UNKNOWN)


def _issue_codes(value: Any) -> List[str]:
    """Accept a list of strings, a list of dicts, or a single string."""
    if value is None:
        return []
    items: Sequence[Any]
    if isinstance(value, str):
        items = [value]
    elif isinstance(value, dict):
        items = [value]
    elif isinstance(value, (list, tuple, set)):
        items = list(value)
    else:
        items = [value]
    codes: List[str] = []
    for item in items:
        if isinstance(item, dict):
            for key in ("code", "issue_code", "id", "name", "rule"):
                candidate = _text(item.get(key))
                if candidate:
                    codes.append(candidate)
                    break
        else:
            candidate = _text(item)
            if candidate:
                codes.append(candidate)
    return _dedup(codes)


def _dedup(values: Iterable[str]) -> List[str]:
    seen: Dict[str, None] = {}
    for value in values:
        if value not in seen:
            seen[value] = None
    return list(seen)


def _norm_warnings(value: Any) -> List[str]:
    """A run's warnings: it passed, but a deliverable is missing.

    Kept separate from ``issue_codes`` on purpose. A warning does not make the run
    a failure, so it must not enter the ranked fix-list -- but it must also never
    be invisible, because a "pass" with no report, no citations and no tiers looks
    identical to a real pass in every other column.
    """
    if value is None:
        return []
    items: Sequence[Any] = value if isinstance(value, (list, tuple, set)) else [value]
    out: List[str] = []
    for item in items:
        if isinstance(item, dict):
            rendered = _text(item.get("message")) or _text(item.get("code")) or _text(item)
        else:
            rendered = _text(item)
        if rendered:
            out.append(rendered)
    return _dedup(out)


def _norm_file_errors(value: Any) -> List[Tuple[str, str]]:
    """``[(path, error), ...]`` for artifacts whose write FAILED.

    ``write_artifacts`` records ``{"bytes": 0, "error": ...}`` for a file it could
    not write. Those entries are listed among the files, so without this they read
    as a zero-byte artifact -- which is exactly how a run with no ``.pwml`` on
    disk passes for a clean one.
    """
    if value is None:
        return []
    if isinstance(value, (str, dict)):
        value = [value]
    if not isinstance(value, (list, tuple)):
        return []
    out: List[Tuple[str, str]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        error = _text(item.get("error"))
        if not error:
            continue
        path = ""
        for key in ("path", "file", "name", "relpath", "filename"):
            path = _text(item.get(key))
            if path:
                break
        out.append((path or "(unnamed artifact)", error))
    return out


def _norm_files(value: Any) -> List[Tuple[str, Optional[int]]]:
    """Return ``[(path, size_or_None), ...]`` from strings and/or dicts."""
    if value is None:
        return []
    if isinstance(value, (str, dict)):
        value = [value]
    if not isinstance(value, (list, tuple)):
        return []
    out: List[Tuple[str, Optional[int]]] = []
    for item in value:
        if isinstance(item, dict):
            path = ""
            for key in ("path", "file", "name", "relpath", "filename"):
                path = _text(item.get(key))
                if path:
                    break
            size: Optional[int] = None
            for key in ("bytes", "size", "size_bytes", "nbytes"):
                if key in item:
                    try:
                        size = int(item[key])
                    except (TypeError, ValueError):
                        size = None
                    if size is not None:
                        break
            if path:
                out.append((path, size))
        else:
            path = _text(item)
            if path:
                out.append((path, None))
    return out


def _norm_counts(value: Any) -> List[Tuple[str, str]]:
    """Flatten a counts mapping one level deep, preferred keys first."""
    if not isinstance(value, dict):
        return []
    flat: List[Tuple[str, str]] = []
    for key, raw in value.items():
        name = _text(key)
        if not name:
            continue
        if isinstance(raw, dict):
            for sub_key, sub_raw in raw.items():
                sub_name = _text(sub_key)
                if sub_name:
                    flat.append((f"{name}.{sub_name}", _count_text(sub_raw)))
        else:
            flat.append((name, _count_text(raw)))

    def _rank(pair: Tuple[str, str]) -> Tuple[int, str]:
        low = pair[0].lower()
        for index, preferred in enumerate(_PREFERRED_COUNT_KEYS):
            if low == preferred or low.replace(".", "_") == preferred:
                return (index, low)
        return (len(_PREFERRED_COUNT_KEYS), low)

    return sorted(flat, key=_rank)


def _count_text(raw: Any) -> str:
    if isinstance(raw, (list, tuple, set)):
        return str(len(raw))
    if isinstance(raw, bool):
        return "yes" if raw else "no"
    return _text(raw, default="?")


def _fmt_seconds(value: Any) -> str:
    try:
        seconds = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if seconds < 0:
        return "n/a"
    if seconds < 90:
        return f"{seconds:.1f}s"
    minutes, rest = divmod(seconds, 60.0)
    return f"{seconds:.1f}s ({int(minutes)}m{int(rest):02d}s)"


def _fmt_bytes(size: Optional[int]) -> str:
    if size is None:
        return ""
    if size < 1024:
        return f"{size} B"
    if size < 1024 * 1024:
        return f"{size / 1024:.1f} kB"
    return f"{size / (1024 * 1024):.1f} MB"


def _wrap(text: str, *, indent: int = 0, first: str = "") -> List[str]:
    """Wrap ``text`` to :data:`WIDTH`, never returning an empty list."""
    body = _text(text)
    if not body:
        return []
    pad = " " * indent
    lines = textwrap.wrap(
        body,
        width=max(20, WIDTH),
        initial_indent=pad + first,
        subsequent_indent=pad + " " * len(first),
        break_long_words=True,
        break_on_hyphens=False,
    )
    return lines or [pad + first + body[: WIDTH - indent]]


def _rule(char: str = "-") -> str:
    return char * WIDTH


# ---------------------------------------------------------------------------
# Row / paper model.
# ---------------------------------------------------------------------------
@dataclass
class ModeRun:
    """One manifest row, normalised."""

    mode: str = ""
    status: str = STATUS_UNKNOWN
    stage: str = ""
    seconds: Any = None
    failure_kind: str = ""
    message: str = ""
    issue_codes: List[str] = field(default_factory=list)
    counts: List[Tuple[str, str]] = field(default_factory=list)
    files: List[Tuple[str, Optional[int]]] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    file_errors: List[Tuple[str, str]] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return self.status == STATUS_PASS

    @property
    def failed(self) -> bool:
        return _is_failure(self.status)

    @property
    def skipped(self) -> bool:
        return self.status == STATUS_SKIPPED

    @property
    def warned(self) -> bool:
        """Passed, but did not produce one of the things it exists to produce."""
        return bool(self.passed and (self.warnings or self.file_errors))

    @property
    def verdict(self) -> str:
        """The status as the summary prints it -- warnings included."""
        if self.warned:
            return f"{STATUS_PASS.upper()} (no research deliverable)"
        return self.status.upper()


@dataclass
class PaperTriage:
    """Every run recorded for one paper, plus its verdict."""

    key: str = ""
    paper_id: str = ""
    slug: str = ""
    title: str = ""
    source_uri: str = ""
    topic: str = ""
    order: int = 0
    runs: Dict[str, ModeRun] = field(default_factory=dict)

    @property
    def strict(self) -> Optional[ModeRun]:
        return self.runs.get(MODE_STRICT)

    @property
    def research(self) -> Optional[ModeRun]:
        return self.runs.get(MODE_RESEARCH)

    @property
    def label(self) -> str:
        return self.paper_id or self.slug or self.key

    @property
    def research_defect(self) -> bool:
        """True whenever the fail-open mode failed at all -- ``broken`` included."""
        research = self.research
        return bool(research is not None and research.failed)

    @property
    def triage_class(self) -> str:
        strict, research = self.strict, self.research
        if strict is None or research is None:
            return CLASS_INCOMPLETE
        if strict.skipped or research.skipped:
            return CLASS_INCOMPLETE
        if strict.passed and research.passed:
            return CLASS_OK
        if strict.failed and research.passed:
            return CLASS_FORMAT_BLOCKED
        if strict.failed and research.failed:
            return CLASS_BROKEN
        if strict.passed and research.failed:
            return CLASS_RESEARCH_DEFECT
        return CLASS_INCOMPLETE


# ---------------------------------------------------------------------------
# Loading.
# ---------------------------------------------------------------------------
def load_manifest(path: Union[str, Path]) -> List[Dict[str, Any]]:
    """Read a JSON-Lines manifest, tolerating anything the writer left behind.

    A missing file yields ``[]`` (the runner may not have written a row yet, and
    :func:`status_line` is polled mid-run). A line that is not a JSON object
    becomes a synthetic ``{"status": "unreadable"}`` row so its existence is
    still counted and reported instead of vanishing.
    """
    target = Path(path)
    try:
        raw = target.read_text(encoding="utf-8", errors="replace")
    except (OSError, ValueError):
        return []

    rows: List[Dict[str, Any]] = []
    for index, line in enumerate(raw.splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            parsed = json.loads(stripped)
        except Exception:
            parsed = None
        if isinstance(parsed, dict):
            rows.append(parsed)
        else:
            rows.append(
                {
                    "status": STATUS_UNREADABLE,
                    "_unreadable": True,
                    "_manifest_line": index,
                    "_raw": stripped[:200],
                }
            )
    return rows


def _is_unreadable(row: Any) -> bool:
    if not isinstance(row, dict):
        return True
    if row.get("_unreadable"):
        return True
    return _text(row.get("status")).lower() == STATUS_UNREADABLE


def _to_run(row: Dict[str, Any]) -> ModeRun:
    return ModeRun(
        mode=_norm_mode(row.get("mode")),
        status=_norm_status(row.get("status")),
        stage=_text(row.get("stage")),
        seconds=row.get("seconds"),
        failure_kind=_text(row.get("failure_kind")),
        message=_text(row.get("message")),
        issue_codes=_issue_codes(row.get("issue_codes")),
        counts=_norm_counts(row.get("counts")),
        files=_norm_files(row.get("files")),
        warnings=_norm_warnings(row.get("warnings")),
        file_errors=_norm_file_errors(row.get("files")),
    )


def group_papers(rows: Sequence[Dict[str, Any]]) -> List[PaperTriage]:
    """Group manifest rows by paper, keeping the last row per *(paper, mode)*.

    Last-wins means a retry supersedes the attempt it retried, which is what a
    reader wants: the final state of each run, not its history.
    """
    papers: Dict[str, PaperTriage] = {}
    for index, row in enumerate(rows or []):
        if _is_unreadable(row):
            continue
        paper_id = _text(row.get("paper_id"))
        slug = _text(row.get("slug"))
        title = _text(row.get("title"))
        key = paper_id or slug or title or f"(unidentified row {index + 1})"
        paper = papers.get(key)
        if paper is None:
            paper = PaperTriage(
                key=key,
                paper_id=paper_id,
                slug=slug,
                title=title,
                source_uri=_text(row.get("source_uri")),
                topic=_text(row.get("topic")),
                order=len(papers),
            )
            papers[key] = paper
        else:
            paper.paper_id = paper.paper_id or paper_id
            paper.slug = paper.slug or slug
            paper.title = paper.title or title
            paper.source_uri = paper.source_uri or _text(row.get("source_uri"))
            paper.topic = paper.topic or _text(row.get("topic"))
        run = _to_run(row)
        paper.runs[run.mode or "(mode missing)"] = run
    return list(papers.values())


def _sorted_papers(papers: Sequence[PaperTriage]) -> List[PaperTriage]:
    """Worst first, manifest order within a class."""
    rank = {name: index for index, name in enumerate(CLASS_ORDER)}
    return sorted(papers, key=lambda p: (rank.get(p.triage_class, 99), p.order))


def _mode_order(paper: PaperTriage) -> List[str]:
    known = [m for m in (MODE_STRICT, MODE_RESEARCH) if m in paper.runs]
    extra = sorted(m for m in paper.runs if m not in (MODE_STRICT, MODE_RESEARCH))
    return known + extra


# ---------------------------------------------------------------------------
# Aggregate tallies.
# ---------------------------------------------------------------------------
def _tally(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    papers = group_papers(rows)
    unreadable = sum(1 for row in rows or [] if _is_unreadable(row))
    counts = {
        "papers": len(papers),
        "unreadable": unreadable,
        "rows": len(rows or []),
        "timeouts": 0,
        "skipped": 0,
        "warned": 0,
        "write_failures": 0,
        MODE_STRICT: {"pass": 0, "fail": 0, "warned": 0},
        MODE_RESEARCH: {"pass": 0, "fail": 0, "warned": 0},
    }
    for paper in papers:
        for mode in (MODE_STRICT, MODE_RESEARCH):
            run = paper.runs.get(mode)
            if run is None:
                continue
            if run.passed:
                counts[mode]["pass"] += 1
                if run.warned:
                    counts[mode]["warned"] += 1
            elif run.failed:
                counts[mode]["fail"] += 1
        for run in paper.runs.values():
            if run.status == STATUS_TIMEOUT:
                counts["timeouts"] += 1
            elif run.skipped:
                counts["skipped"] += 1
            if run.warned:
                counts["warned"] += 1
            counts["write_failures"] += len(run.file_errors)
    counts["by_class"] = {name: 0 for name in CLASS_ORDER}
    for paper in papers:
        counts["by_class"][paper.triage_class] = (
            counts["by_class"].get(paper.triage_class, 0) + 1
        )
    counts["papers_list"] = papers
    return counts


# ---------------------------------------------------------------------------
# SUMMARY.txt
# ---------------------------------------------------------------------------
def build_summary(
    rows: Sequence[Dict[str, Any]],
    *,
    started_at: str = "",
    finished_at: str = "",
) -> str:
    """Render SUMMARY.txt: header counts, triage matrix, per-paper, fix-order."""
    tally = _tally(rows)
    papers: List[PaperTriage] = tally["papers_list"]
    out: List[str] = []

    # 1. Header ------------------------------------------------------------
    out.append(_rule("="))
    out.append("T2PW OVERNIGHT BATCH RUN -- SUMMARY")
    out.append(_rule("="))
    if started_at:
        out.append(f"started  : {_text(started_at)}")
    if finished_at:
        out.append(f"finished : {_text(finished_at)}")
    out.append(f"papers attempted : {tally['papers']}")
    out.append(
        f"strict PWML      : {tally[MODE_STRICT]['pass']} pass / "
        f"{tally[MODE_STRICT]['fail']} fail"
    )
    out.append(
        f"research relaxed : {tally[MODE_RESEARCH]['pass']} pass / "
        f"{tally[MODE_RESEARCH]['fail']} fail"
    )
    out.append(f"timeouts         : {tally['timeouts']}")
    out.append(f"skipped          : {tally['skipped']}")
    # Counted apart from "pass" on purpose: these runs did not fail, but they did
    # not produce their deliverable either, so the night is NOT all green.
    out.append(
        f"pass, no deliv.  : {tally['warned']}  "
        f"(reported as \"{STATUS_PASS.upper()} (no research deliverable)\" below)"
    )
    out.append(f"artifact writes  : {tally['write_failures']} FAILED")
    out.append(f"manifest rows    : {tally['rows']} (unreadable: {tally['unreadable']})")
    if not papers:
        out.append("")
        out.append("No readable paper rows in the manifest -- nothing was recorded.")
        out.append("If the run was expected to produce results, the runner died before")
        out.append("writing its first row; check the runner's own stderr log.")
        out.append("")
        out.append(_rule("="))
        return "\n".join(out) + "\n"

    # 2. Triage matrix -----------------------------------------------------
    out.append("")
    out.append(_rule("="))
    out.append("TRIAGE MATRIX")
    out.append(_rule("="))
    out.extend(_triage_matrix(papers))

    # 3. Per-paper breakdown ----------------------------------------------
    out.append("")
    out.append(_rule("="))
    out.append("PER-PAPER BREAKDOWN")
    out.append(_rule("="))
    out.append("Ordered worst-first; every artifact written for the paper is named.")
    for position, paper in enumerate(_sorted_papers(papers), start=1):
        out.append("")
        out.extend(_paper_block(paper, position))

    # 4. What to fix first -------------------------------------------------
    out.append("")
    out.append(_rule("="))
    out.append("WHAT TO FIX FIRST")
    out.append(_rule("="))
    out.extend(_fix_order(papers))
    out.append("")
    out.append(_rule("="))
    return "\n".join(out) + "\n"


def _triage_matrix(papers: Sequence[PaperTriage]) -> List[str]:
    out: List[str] = []
    defects = [p for p in papers if p.research_defect]

    # Research-mode defects, first and loudest.
    out.append("")
    if defects:
        out.append(DEFECT_MARKER)
        out.extend(
            _wrap(
                "Research mode is fail-open by design: it relaxes FORMAT rules and "
                "records review flags instead of stopping. Therefore ANY research "
                "failure is a code defect, not a data problem. Fix these before "
                "anything else.",
                indent=0,
            )
        )
        out.append(f"papers affected: {len(defects)}")
        for paper in _sorted_papers(defects):
            run = paper.research
            out.append("")
            out.extend(_wrap(f"{paper.label} -- {paper.title or '(untitled)'}", indent=2))
            detail = f"class={paper.triage_class}"
            if paper.triage_class == CLASS_BROKEN:
                detail += " (strict failed too -- fix the shared cause)"
            out.append(f"    {detail}")
            if run is not None:
                out.extend(
                    _wrap(
                        f"research: {run.status.upper()} at stage="
                        f"{run.stage or '(unknown)'} after {_fmt_seconds(run.seconds)}",
                        indent=4,
                    )
                )
                if run.failure_kind:
                    out.extend(_wrap(run.failure_kind, indent=4, first="failure_kind: "))
                out.extend(_wrap(run.message, indent=6, first="message: "))
                if run.issue_codes:
                    out.extend(
                        _wrap(", ".join(run.issue_codes), indent=6, first="codes: ")
                    )
        out.append("")
        out.append(_rule("-"))
    else:
        out.append(
            f"research-mode defects: NONE -- research mode held fail-open on all "
            f"{len(papers)} paper(s)."
        )
        out.append(_rule("-"))

    # Passes that produced nothing. Second loudest, because they are invisible in
    # every pass/fail column yet leave the paper folder without its deliverable.
    warned = [
        (paper, mode, run)
        for paper in _sorted_papers(papers)
        for mode, run in paper.runs.items()
        if run.warned
    ]
    if warned:
        out.append("")
        out.append(WARNING_MARKER)
        out.extend(
            _wrap(
                "These runs did not fail -- the pipeline ran to completion -- but the "
                "folder has no deliverable in it. This is NOT a clean night. Check "
                "whether the missing output was expected before treating it as green.",
                indent=0,
            )
        )
        out.append(f"runs affected: {len(warned)}")
        for paper, mode, run in warned:
            out.append("")
            out.extend(_wrap(f"{paper.label} [{mode}] -- {paper.title or '(untitled)'}", indent=2))
            for warning in run.warnings:
                out.extend(_wrap(warning, indent=4, first="- "))
            for path, error in run.file_errors:
                out.extend(_wrap(f"{path}: {error}", indent=4, first="- WRITE FAILED "))
        out.append("")
        out.append(_rule("-"))

    # The classes themselves.
    for name in CLASS_ORDER:
        members = [p for p in papers if p.triage_class == name]
        header = f"{name:<15} {len(members):>3}  ({CLASS_BLURB.get(name, '')})"
        out.append(header)
        if members:
            labels = ", ".join(p.label for p in sorted(members, key=lambda p: p.order))
            out.extend(_wrap(labels, indent=4))
    return out


def _paper_block(paper: PaperTriage, position: int) -> List[str]:
    out: List[str] = []
    out.append(_rule("-"))
    marker = f"  {DEFECT_MARKER}" if paper.research_defect else ""
    out.extend(
        _wrap(
            f"[{position}] {paper.title or '(untitled)'}  ->  {paper.triage_class}{marker}",
            indent=0,
        )
    )
    out.extend(_wrap(paper.paper_id or "(none)", indent=4, first="id     : "))
    if paper.slug:
        out.extend(_wrap(paper.slug, indent=4, first="slug   : "))
    out.extend(_wrap(paper.source_uri or "(none)", indent=4, first="uri    : "))
    out.extend(_wrap(paper.topic or "(none)", indent=4, first="topic  : "))

    for mode in _mode_order(paper):
        run = paper.runs[mode]
        out.append("")
        out.extend(
            _wrap(
                f"{run.verdict} | stage={run.stage or '(unknown)'} "
                f"| time={_fmt_seconds(run.seconds)}",
                indent=4,
                first=f"{mode:<9}: ",
            )
        )
        for warning in run.warnings:
            out.extend(_wrap(warning, indent=8, first="warning      : "))
        for path, error in run.file_errors:
            out.extend(_wrap(f"{path}: {error}", indent=8, first="WRITE FAILED : "))
        if run.failed:
            out.extend(
                _wrap(
                    run.failure_kind or "(unclassified)",
                    indent=8,
                    first="failure_kind : ",
                )
            )
            out.extend(_wrap(run.message or "(no message recorded)", indent=8, first="message      : "))
            codes = ", ".join(run.issue_codes) if run.issue_codes else "(none reported)"
            out.extend(_wrap(codes, indent=8, first="issue_codes  : "))
        if run.counts:
            pairs = "  ".join(f"{name}={value}" for name, value in run.counts)
            out.extend(_wrap(pairs, indent=8, first="counts       : "))
        if run.files:
            broken = dict(run.file_errors)
            out.append(f"        files        : {len(run.files)}")
            for path, size in run.files:
                size_text = _fmt_bytes(size)
                suffix = f"  ({size_text})" if size_text else ""
                if path in broken:
                    # Named at the file, not only in a summary line: the reader is
                    # looking for the artifact, and it is not on disk.
                    suffix += f"  !! WRITE FAILED: {broken[path]}"
                out.extend(_wrap(f"{path}{suffix}", indent=12, first="- "))
        else:
            out.append("        files        : none recorded")
    return out


def _fix_order(papers: Sequence[PaperTriage]) -> List[str]:
    out: List[str] = []
    defects = _sorted_papers([p for p in papers if p.research_defect])
    broken = _sorted_papers([p for p in papers if p.triage_class == CLASS_BROKEN])
    blocked = [p for p in papers if p.triage_class == CLASS_FORMAT_BLOCKED]

    out.append("")
    out.append("1. RESEARCH-MODE DEFECTS -- research mode must never fail. Fix first.")
    if not defects:
        out.append("     (none)")
    for paper in defects:
        run = paper.research
        codes = ", ".join(run.issue_codes) if run and run.issue_codes else "no codes"
        stage = (run.stage if run else "") or "(unknown)"
        out.extend(_wrap(f"{paper.label}: {stage} -- {codes}", indent=5, first="- "))

    out.append("")
    out.append("2. BROKEN -- both modes failed: crash / LLM / network / biology.")
    if not broken:
        out.append("     (none)")
    for paper in broken:
        run = paper.strict or paper.research
        kind = (run.failure_kind if run else "") or "(unclassified)"
        out.extend(_wrap(f"{paper.label}: {kind}", indent=5, first="- "))

    out.append("")
    out.append("3. FORMAT-BLOCKED -- strict only, grouped by issue code (research is fine).")
    if not blocked:
        out.append("     (none)")
    else:
        grouped: Dict[str, List[str]] = {}
        for paper in blocked:
            run = paper.strict
            codes = (run.issue_codes if run else []) or [
                f"(no code; failure_kind={(run.failure_kind if run else '') or 'unknown'})"
            ]
            for code in codes:
                grouped.setdefault(code, []).append(paper.label)
        for code, labels in sorted(
            grouped.items(), key=lambda item: (-len(item[1]), item[0])
        ):
            unique = _dedup(labels)
            out.extend(
                _wrap(
                    f"{code}  ({len(unique)} paper(s)): {', '.join(unique)}",
                    indent=5,
                    first="- ",
                )
            )
    return out


# ---------------------------------------------------------------------------
# failures_by_code.txt
# ---------------------------------------------------------------------------
def build_failures_by_code(rows: Sequence[Dict[str, Any]]) -> str:
    """Render the ranked fix-list: every issue code, worst blast radius first."""
    papers = group_papers(rows)
    out: List[str] = []
    out.append(_rule("="))
    out.append("FAILURES BY ISSUE CODE -- the ranked fix-list")
    out.append(_rule("="))
    out.extend(
        _wrap(
            "Ranked by how many distinct papers hit the code, because a code that "
            "blocks six papers is worth more than one that blocks a single paper "
            "six times.",
            indent=0,
        )
    )

    # code -> {"papers": {label: None}, "modes": {mode: None}, "runs": int}
    by_code: Dict[str, Dict[str, Any]] = {}
    uncoded: Dict[str, Dict[str, Any]] = {}
    failing_runs = 0

    for paper in papers:
        for mode, run in paper.runs.items():
            if not run.failed:
                continue
            failing_runs += 1
            if run.issue_codes:
                for code in run.issue_codes:
                    slot = by_code.setdefault(
                        code, {"papers": {}, "modes": {}, "runs": 0}
                    )
                    slot["papers"][paper.label] = None
                    slot["modes"][mode] = None
                    slot["runs"] += 1
            else:
                kind = run.failure_kind or run.status or "(unclassified)"
                slot = uncoded.setdefault(kind, {"papers": {}, "modes": {}, "runs": 0})
                slot["papers"][paper.label] = None
                slot["modes"][mode] = None
                slot["runs"] += 1

    out.append("")
    out.append(
        f"failing runs: {failing_runs}   distinct issue codes: {len(by_code)}   "
        f"uncoded failure kinds: {len(uncoded)}"
    )

    out.append("")
    out.append(_rule("-"))
    out.append("ISSUE CODES")
    out.append(_rule("-"))
    if not by_code:
        out.append("(no issue codes were reported)")
    else:
        ranked = sorted(
            by_code.items(),
            key=lambda item: (-len(item[1]["papers"]), -item[1]["runs"], item[0]),
        )
        for position, (code, slot) in enumerate(ranked, start=1):
            labels = list(slot["papers"])
            modes = ", ".join(sorted(slot["modes"])) or "(unknown)"
            out.append("")
            out.extend(_wrap(f"{position:>2}. {code}", indent=0))
            out.append(
                f"     papers: {len(labels)}   runs: {slot['runs']}   modes: {modes}"
            )
            out.extend(_wrap(", ".join(labels), indent=5, first="affected: "))

    out.append("")
    out.append(_rule("-"))
    out.append("FAILURES WITH NO ISSUE CODES (grouped by failure_kind)")
    out.append(_rule("-"))
    out.extend(
        _wrap(
            "These have no code to group on, so they are listed by kind -- nothing "
            "may be invisible in this report.",
            indent=0,
        )
    )
    if not uncoded:
        out.append("(none -- every failure reported at least one issue code)")
    else:
        ranked_uncoded = sorted(
            uncoded.items(),
            key=lambda item: (-len(item[1]["papers"]), -item[1]["runs"], item[0]),
        )
        for kind, slot in ranked_uncoded:
            labels = list(slot["papers"])
            modes = ", ".join(sorted(slot["modes"])) or "(unknown)"
            out.append("")
            out.extend(_wrap(f"{kind}", indent=0, first="* "))
            out.append(
                f"     papers: {len(labels)}   runs: {slot['runs']}   modes: {modes}"
            )
            out.extend(_wrap(", ".join(labels), indent=5, first="affected: "))

    out.append("")
    out.append(_rule("="))
    return "\n".join(out) + "\n"


# ---------------------------------------------------------------------------
# One-line progress.
# ---------------------------------------------------------------------------
def status_line(
    rows: Sequence[Dict[str, Any]],
    *,
    total_expected: int = 0,
    running: str = "",
) -> str:
    """One line, safe to print every few minutes while the batch is running."""
    tally = _tally(rows)
    done = tally["papers"]
    total = total_expected if total_expected else done
    parts: List[str] = [f"{done}/{total} done"]

    for mode, name in ((MODE_STRICT, "strict"), (MODE_RESEARCH, "research")):
        passed = tally[mode]["pass"]
        failed = tally[mode]["fail"]
        segment = f"{name} {passed} pass"
        if failed:
            segment += f" {failed} fail"
        parts.append(segment)

    if tally["timeouts"]:
        parts.append(f"{tally['timeouts']} timeout")
    if tally["skipped"]:
        parts.append(f"{tally['skipped']} skipped")
    if tally["unreadable"]:
        parts.append(f"{tally['unreadable']} unreadable")
    if tally["warned"]:
        parts.append(f"{tally['warned']} NO DELIVERABLE")

    defects = sum(1 for paper in tally["papers_list"] if paper.research_defect)
    if defects:
        parts.append(f"{defects} RESEARCH DEFECT")

    running_label = _text(running)
    if running_label:
        parts.append(f"running: {running_label}")
    return " | ".join(parts)


__all__ = [
    "WIDTH",
    "MODE_STRICT",
    "MODE_RESEARCH",
    "STATUS_PASS",
    "STATUS_FAIL",
    "STATUS_TIMEOUT",
    "STATUS_SKIPPED",
    "STATUS_UNREADABLE",
    "CLASS_OK",
    "CLASS_FORMAT_BLOCKED",
    "CLASS_BROKEN",
    "CLASS_RESEARCH_DEFECT",
    "CLASS_INCOMPLETE",
    "CLASS_ORDER",
    "DEFECT_MARKER",
    "WARNING_MARKER",
    "ModeRun",
    "PaperTriage",
    "load_manifest",
    "group_papers",
    "build_summary",
    "build_failures_by_code",
    "status_line",
]
