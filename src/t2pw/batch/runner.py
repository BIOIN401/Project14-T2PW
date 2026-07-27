"""Orchestration for the unattended overnight batch run.

This is the part that has to survive a night alone: it fetches the papers once,
then walks every *(paper, mode)* pair, records what happened, and regenerates the
human reports after each one so a half-finished run is still readable at 3am.

Three decisions dominate the design.

**1. Every paper+mode runs in a CHILD PROCESS.** The parent re-invokes
``scripts/batch_run.py --single ...`` and waits with a timeout. This looks like
overkill until you try the alternatives: a Python thread cannot be killed (there
is no ``Thread.kill``, and a thread blocked in a socket ``recv`` will never look
at a stop flag), and ``signal.SIGALRM`` -- the usual in-process watchdog -- simply
does not exist on Windows. A hung LLM request, a wedged PathBank MySQL socket, or
a Streamlit rerun that never returns is therefore *unkillable* in-process. Only a
separate OS process can be terminated from outside. On Windows the child is put
in its own process group and killed with ``taskkill /F /T`` so any worker it
spawned dies with it. Please do not "simplify" this into a thread.

**2. Resume is automatic and requires no flag.** The user's words: "i shouldnt
need to resume it, itll handle itself." So on start we look for the most recent
run directory that still has pending pairs and continue it, using
``manifest.jsonl`` as the resume state and ``plan.json`` as the work list. The
paper text lives in the run directory (``01_source_text.txt``), so a resumed run
never re-fetches and never re-runs finished work -- and it never silently skips
unfinished work either. Which path was taken is printed.

**3. Strictly sequential, one pair at a time.** ``data/id_mapping_cache.json``,
``data/enrichment_cache.json`` and the RAG index under ``data/rag_index`` are
shared *mutable* state with read-modify-write access patterns and no locking.
Two concurrent runs interleave writes and corrupt them. The wall-clock cost of
being sequential is the price of a run whose caches are still usable in the
morning; those two caches are also snapshotted before the first paper so a bad
night is revertible.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from t2pw.batch import driver, report
from t2pw.batch.driver import MODE_RESEARCH, MODE_STRICT, RunOutcome
from t2pw.batch.fetch import BatchPaper, fetch_papers
from t2pw.paths import PROJECT_ROOT

# ── Run-directory layout ───────────────────────────────────────────────────
DEFAULT_OUT_DIRNAME = "runs"
PAPERS_DIRNAME = "papers"
CACHE_SNAPSHOT_DIRNAME = "cache_snapshot"
MANIFEST_NAME = "manifest.jsonl"
SUMMARY_NAME = "SUMMARY.txt"
FAILURES_NAME = "failures_by_code.txt"
LOG_NAME = "batch.log"
SKIPPED_NAME = "skipped.json"
PLAN_NAME = "plan.json"
PAPER_DESC_NAME = "00_PAPER.txt"
SOURCE_TEXT_NAME = "01_source_text.txt"
RESULT_NAME = "RESULT.txt"

#: Caches worth snapshotting. Absent files are skipped silently.
SNAPSHOT_FILES: Tuple[str, ...] = ("id_mapping_cache.json", "enrichment_cache.json")

#: ``runs/2026-07-27_2210``. Minute resolution: two runs a minute apart are
#: distinguishable, and the name sorts lexicographically in time order, which is
#: what makes "most recent" a plain ``sorted()``.
RUN_STAMP_FMT = "%Y-%m-%d_%H%M"
_RUN_DIR_RE = re.compile(r"^\d{4}-\d{2}-\d{2}_\d{4}")

#: The child's result line is prefixed so Streamlit's own chatter on stdout can
#: never be mistaken for the outcome.
OUTCOME_SENTINEL = "@@T2PW_OUTCOME@@"

#: Per paper+mode wall-clock ceiling enforced by the *parent*.
DEFAULT_PAPER_TIMEOUT = 3600.0

#: Whole-night ceiling. 10 papers x 2 modes x :data:`DEFAULT_PAPER_TIMEOUT` is 20
#: hours in the worst case, which is not "overnight" -- it is still running when
#: the user needs the machine. The loop stops cleanly past this and the run stays
#: resumable, so the remainder finishes the next night.
DEFAULT_DEADLINE_HOURS = 10.0

#: How old a run directory may be and still be *continued*. Without this the
#: newest directory that still owes work wins even if it is a month old, so every
#: night re-runs one stale paper, fetches nothing, and exits -- forever.
RESUME_MAX_AGE_HOURS = 24.0

#: The one artifact each mode exists to produce. If the driver produced it and
#: the *write* then failed, the run is a failure however clean the pipeline was:
#: a "pass" with no ``.pwml`` on disk is the worst kind of lie.
REQUIRED_ARTIFACTS: Dict[str, Tuple[str, ...]] = {
    MODE_STRICT: ("pathway.pwml",),
    MODE_RESEARCH: ("research_pathway_report.txt",),
}

#: The child is given a slightly smaller internal budget than the parent's kill
#: timeout so it has room to write its artifacts and print its row instead of
#: being killed one second before finishing.
_CHILD_GRACE = 120.0

#: How long to wait for a killed child's pipes to close before giving up on them.
_DRAIN_TIMEOUT = 30.0

ALL_MODES: Tuple[str, ...] = (MODE_STRICT, MODE_RESEARCH)

_STATUS_PASS = "pass"
_STATUS_TIMEOUT = "timeout"
_STATUS_ERROR = "error"


# ---------------------------------------------------------------------------
# Small helpers.
# ---------------------------------------------------------------------------
def _text(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ("" if value is None else str(value))


def _safe_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _safe_list(value: Any) -> List[Any]:
    return list(value) if isinstance(value, (list, tuple)) else []


def _fmt_seconds(value: Any) -> str:
    try:
        seconds = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if seconds < 90:
        return f"{seconds:.1f}s"
    minutes, rest = divmod(seconds, 60.0)
    return f"{seconds:.1f}s ({int(minutes)}m{int(rest):02d}s)"


def parse_modes(value: Any) -> List[str]:
    """``"strict,research"`` -> ``["strict", "research"]``.

    Order is preserved (the caller may want research first), duplicates are
    dropped, and anything unrecognised is ignored rather than raised on. An empty
    result falls back to both modes, because a run that silently did nothing is
    the worst possible outcome for an overnight job.
    """

    if value is None:
        return list(ALL_MODES)
    if isinstance(value, (list, tuple)):
        tokens = [_text(item) for item in value]
    else:
        tokens = [part for part in re.split(r"[,\s;]+", _text(value)) if part]
    out: List[str] = []
    for token in tokens:
        low = token.lower()
        mode = ""
        if "research" in low or "relax" in low:
            mode = MODE_RESEARCH
        elif "strict" in low or "pwml" in low:
            mode = MODE_STRICT
        if mode and mode not in out:
            out.append(mode)
    return out or list(ALL_MODES)


def resolve_out_dir(value: Any = None) -> Path:
    """Resolve ``--out`` against the project root, not the shell's cwd.

    A double-clicked ``.bat`` and a shell invocation must land in the same place.
    """

    if value in (None, ""):
        return PROJECT_ROOT / DEFAULT_OUT_DIRNAME
    candidate = Path(value)
    return candidate if candidate.is_absolute() else (PROJECT_ROOT / candidate)


def _write_text(path: Path, text: str) -> None:
    """Every write is UTF-8: entity names really do contain non-cp1252 bytes."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def _read_text_or_error(path: Path) -> Tuple[str, str]:
    """``(text, error)``. Exactly one of the two is meaningful.

    Swallowing the ``OSError`` and returning ``""`` is what turned a *missing
    input file* into "the extractor found nothing to extract" -- an
    infrastructure fault filed as a biology defect. The two cases are told apart
    here so the caller can classify them differently: an empty-but-present file
    really is a paper with no text, a missing one is a broken run directory.
    """

    try:
        return path.read_text(encoding="utf-8", errors="replace"), ""
    except OSError as exc:
        return "", f"{type(exc).__name__}: {exc}"


def _ascii_safe(text: str) -> str:
    """A rendering of ``text`` that no console encoding can refuse."""

    try:
        return str(text).encode("ascii", "backslashreplace").decode("ascii", "replace")
    except Exception:  # noqa: BLE001 - a __str__ that raises must not kill a log line
        return "(unprintable log line)"


def force_utf8_stdio() -> None:
    """Make this process's stdout/stderr UTF-8, whatever the console says.

    The child's stdout is ALWAYS a pipe, and a piped ``sys.stdout.encoding`` is
    the ANSI codepage (cp1252 here), not UTF-8. One Greek beta in an entity name
    -- ``beta-hydroxymyristoyl`` is the canonical lipid A intermediate -- then
    raises ``UnicodeEncodeError`` *after* the artifacts and RESULT.txt are
    already on disk, and the parent records a crash for a pair that passed. Since
    a row exists the pair is never retried, so the work is lost for good.

    Called first thing in the CLI, before anything can print.
    """

    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except Exception:  # noqa: BLE001 - a wrapped/replaced stream may refuse
            pass


def child_env() -> Dict[str, str]:
    """The environment for a child process: ours, plus forced UTF-8 I/O.

    ``os.environ`` is *copied*, never replaced: the child needs every var the
    ``.env`` load put there (API keys, PathBank DSN) or it would fail for a
    completely unrelated reason.
    """

    env = dict(os.environ)
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    return env


def _json_default(value: Any) -> Any:
    if isinstance(value, (bytes, bytearray)):
        return f"<{len(value)} bytes omitted>"
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (set, frozenset)):
        return sorted(str(item) for item in value)
    return str(value)


def _json_text(value: Any) -> str:
    try:
        return json.dumps(value, indent=2, ensure_ascii=False, default=_json_default)
    except Exception as exc:  # noqa: BLE001 - never lose a file to a dump bug
        return json.dumps({"_unserializable": True, "error": str(exc)}, indent=2)


# ---------------------------------------------------------------------------
# Logging. One line per event, to both the console and batch.log.
# ---------------------------------------------------------------------------
class Logger:
    """Append-only progress log.

    The file is opened and closed per line on purpose: an overnight run that is
    killed with the window's X button must still leave a complete log, and a
    long-lived buffered handle would lose the tail.

    Nothing here may raise. Logging a paper *title* verbatim is enough to kill
    the whole night otherwise: ``print`` to a cp1252 console (or, worse, to a
    redirected log file or Task Scheduler's captured stream) raises
    ``UnicodeEncodeError`` on a Greek letter -- and ``UnicodeEncodeError`` is a
    ``ValueError``, so an ``except OSError`` around the write does not catch it.
    """

    def __init__(self, run_dir: Optional[Path] = None, *, echo: bool = True) -> None:
        self.path = (Path(run_dir) / LOG_NAME) if run_dir else None
        self.echo = echo

    def __call__(self, message: str) -> None:
        stamp = datetime.now().strftime("%H:%M:%S")
        line = f"[{stamp}] {message}"
        if self.echo:
            try:
                print(line, flush=True)
            except Exception:  # noqa: BLE001 - UnicodeEncodeError is a ValueError
                try:
                    print(_ascii_safe(line), flush=True)
                except Exception:  # noqa: BLE001 - a closed/dead stdout: keep going
                    pass
        if self.path is None:
            return
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, "a", encoding="utf-8") as handle:
                handle.write(line + "\n")
        except Exception:  # noqa: BLE001 - the log is never worth the batch
            pass


# ---------------------------------------------------------------------------
# Run directories: discovery, planning, resume state.
# ---------------------------------------------------------------------------
def list_run_dirs(out_dir: Path) -> List[Path]:
    """Every timestamped run directory, most recent first."""

    base = Path(out_dir)
    if not base.is_dir():
        return []
    dirs = [
        child
        for child in base.iterdir()
        if child.is_dir() and _RUN_DIR_RE.match(child.name)
    ]
    return sorted(dirs, key=lambda p: p.name, reverse=True)


def latest_run_dir(out_dir: Path) -> Optional[Path]:
    dirs = list_run_dirs(out_dir)
    return dirs[0] if dirs else None


def new_run_dir(out_dir: Path, *, now: Optional[datetime] = None) -> Path:
    """Create ``out_dir/YYYY-MM-DD_HHMM``, suffixing on a same-minute collision."""

    base = Path(out_dir)
    stamp = (now or datetime.now()).strftime(RUN_STAMP_FMT)
    candidate = base / stamp
    suffix = 2
    while candidate.exists():
        candidate = base / f"{stamp}-{suffix}"
        suffix += 1
    (candidate / PAPERS_DIRNAME).mkdir(parents=True, exist_ok=True)
    return candidate


def load_plan(run_dir: Path) -> Dict[str, Any]:
    """The work list written when the run started. ``{}`` when unreadable."""

    raw = _read_text(Path(run_dir) / PLAN_NAME)
    if not raw.strip():
        return {}
    try:
        parsed = json.loads(raw)
    except Exception:  # noqa: BLE001 - a truncated plan is not fatal
        return {}
    return parsed if isinstance(parsed, dict) else {}


def save_plan(run_dir: Path, plan: Dict[str, Any]) -> None:
    _write_text(Path(run_dir) / PLAN_NAME, _json_text(plan) + "\n")


def plan_pairs(plan: Dict[str, Any]) -> List[Tuple[str, str]]:
    """``[(slug, mode), ...]`` in run order: every unit of work the run owes."""

    modes = parse_modes(plan.get("modes"))
    pairs: List[Tuple[str, str]] = []
    for record in _safe_list(plan.get("papers")):
        slug = _text(_safe_dict(record).get("slug"))
        if not slug:
            continue
        for mode in modes:
            pairs.append((slug, mode))
    return pairs


def completed_pairs(run_dir: Path) -> set:
    """``{(slug, mode)}`` already recorded in ``manifest.jsonl``.

    This -- not a marker file -- is the resume state, because it is written and
    flushed as each pair finishes and therefore cannot disagree with reality.
    """

    done: set = set()
    for row in report.load_manifest(Path(run_dir) / MANIFEST_NAME):
        if not isinstance(row, dict):
            continue
        slug = _text(row.get("slug")) or _text(row.get("paper_id"))
        mode = _text(row.get("mode")).lower()
        if slug and mode:
            done.add((slug, mode))
    return done


def pending_pairs(run_dir: Path, plan: Optional[Dict[str, Any]] = None) -> List[Tuple[str, str]]:
    plan = plan if plan is not None else load_plan(run_dir)
    done = completed_pairs(run_dir)
    return [pair for pair in plan_pairs(plan) if pair not in done]


def _stamp_from_name(run_dir: Path) -> Optional[float]:
    """The POSIX time encoded in ``2026-07-27_2210``, or ``None`` if unparseable."""

    match = _RUN_DIR_RE.match(Path(run_dir).name)
    if not match:
        return None
    try:
        return datetime.strptime(match.group(0), RUN_STAMP_FMT).timestamp()
    except (ValueError, OverflowError, OSError):
        return None


def run_dir_touched_at(run_dir: Path) -> float:
    """The most recent moment this run showed any sign of life, as POSIX time.

    The newest of (a) the timestamp in the directory name, (b) the directory's
    own mtime and (c) ``manifest.jsonl``'s mtime. All three are considered
    because any one of them can lie: a directory copied off a backup has a fresh
    mtime and an ancient name, and a run whose clock was wrong has the reverse.
    The *newest* evidence wins, so a directory is only called stale when nothing
    about it looks recent.
    """

    run_dir = Path(run_dir)
    stamps: List[float] = []
    named = _stamp_from_name(run_dir)
    if named is not None:
        stamps.append(named)
    for candidate in (run_dir, run_dir / MANIFEST_NAME):
        try:
            stamps.append(candidate.stat().st_mtime)
        except OSError:
            pass
    return max(stamps) if stamps else 0.0


def find_resumable(
    out_dir: Path,
    *,
    max_age_hours: float = RESUME_MAX_AGE_HOURS,
    now: Optional[float] = None,
    log: Optional[Callable[[str], None]] = None,
) -> Optional[Path]:
    """The run directory to *continue*, or ``None`` to start a fresh one.

    Only the NEWEST run directory is ever considered. Reaching back past it was a
    trap: an aborted run from weeks ago still owes work, so it beat last night's
    complete run, and every night from then on re-ran one stale paper, fetched no
    new papers, and exited. If the newest run is finished, the right answer is a
    fresh run -- not an older incomplete one.

    A directory with no readable ``plan.json`` is not resumable either: without
    the work list there is nothing to continue. And a run older than
    ``max_age_hours`` is left alone however much it owes, because "resume" is
    meant for last night's interrupted batch, not for archaeology.

    ``log`` receives one line explaining the choice; it is optional because the
    caller has no run directory (and therefore no log file) until this returns.
    """

    say = log or (lambda _message: None)
    dirs = list_run_dirs(out_dir)
    if not dirs:
        say("no previous run directories, so this is a fresh run")
        return None

    newest = dirs[0]
    others = len(dirs) - 1
    plan = load_plan(newest)
    if not plan_pairs(plan):
        say(
            f"ignoring {newest.name}: its {PLAN_NAME} is missing, unreadable or lists no "
            "papers, so there is no work list to continue -- starting a fresh run"
        )
        return None
    pending = pending_pairs(newest, plan)
    if not pending:
        say(
            f"the newest run {newest.name} is complete -- starting a fresh run"
            + (f" (not reaching back past {others} older run(s))" if others else "")
        )
        return None

    reference = time.time() if now is None else float(now)
    age_hours = max(0.0, (reference - run_dir_touched_at(newest)) / 3600.0)
    if age_hours > max(0.0, float(max_age_hours)):
        say(
            f"ignoring {newest.name}: older than {max_age_hours:.0f}h "
            f"(last touched {age_hours:.1f}h ago) even though {len(pending)} pair(s) are "
            "still pending -- starting a fresh run instead of re-running stale work"
        )
        return None

    say(
        f"resuming {newest.name}: the newest run, last touched {age_hours:.1f}h ago, "
        f"with {len(pending)} pair(s) still pending"
    )
    return newest


# ---------------------------------------------------------------------------
# Cache snapshot.
# ---------------------------------------------------------------------------
def snapshot_caches(run_dir: Path, *, data_dir: Optional[Path] = None) -> List[str]:
    """Copy the mutable caches so a bad night can be reverted. Never raises."""

    source = Path(data_dir) if data_dir else (PROJECT_ROOT / "data")
    target = Path(run_dir) / CACHE_SNAPSHOT_DIRNAME
    copied: List[str] = []
    for name in SNAPSHOT_FILES:
        origin = source / name
        if not origin.is_file():
            continue  # silently: a fresh checkout has neither cache yet
        try:
            target.mkdir(parents=True, exist_ok=True)
            shutil.copy2(origin, target / name)
            copied.append(name)
        except OSError:
            pass
    return copied


# ---------------------------------------------------------------------------
# Per-paper input files.
# ---------------------------------------------------------------------------
def paper_dir(run_dir: Path, slug: str) -> Path:
    return Path(run_dir) / PAPERS_DIRNAME / slug


def write_paper_inputs(run_dir: Path, paper: BatchPaper) -> Path:
    """Write ``00_PAPER.txt`` + ``01_source_text.txt`` and return the folder.

    The source text is written verbatim: it is the exact string handed to the
    app, so a failure can be reproduced by pasting this file into the browser.
    It is also what a resumed run reads back, which is why no re-fetch is needed.
    """

    target = paper_dir(run_dir, paper.slug)
    target.mkdir(parents=True, exist_ok=True)
    _write_text(target / PAPER_DESC_NAME, paper.describe())
    _write_text(target / SOURCE_TEXT_NAME, paper.full_text)
    return target


def _plan_record(paper: BatchPaper) -> Dict[str, Any]:
    record = paper.to_dict()
    record.pop("full_text", None)  # it lives in 01_source_text.txt, not here
    record["chars"] = len(paper.full_text)
    return record


def paper_from_plan(run_dir: Path, slug: str) -> Dict[str, Any]:
    """Rebuild the driver's input for one planned paper, text included.

    Returns a plain dict: ``driver.run_one`` accepts any object or mapping that
    can produce an id and a full text, so no ``BatchPaper`` round-trip is needed.

    A source text file that cannot be READ is reported as
    ``source_text_error`` rather than as an empty string. The distinction is the
    whole point: an empty-but-readable file is a paper with no usable text (a
    biology/extraction outcome), a missing or unreadable one is a broken run
    directory (an infrastructure fault), and filing the second as the first sent
    the morning's debugging into the extractor for no reason.
    """

    plan = load_plan(run_dir)
    record: Dict[str, Any] = {}
    for candidate in _safe_list(plan.get("papers")):
        data = _safe_dict(candidate)
        if _text(data.get("slug")) == slug:
            record = dict(data)
            break
    record.setdefault("slug", slug)
    record.setdefault("paper_id", slug)
    source_path = paper_dir(run_dir, slug) / SOURCE_TEXT_NAME
    text, error = _read_text_or_error(source_path)
    record["full_text"] = text
    record["source_text_path"] = str(source_path)
    if error:
        record["source_text_error"] = error
    return record


# ---------------------------------------------------------------------------
# Manifest + reports.
# ---------------------------------------------------------------------------
def _repair_truncated_tail(path: Path) -> bool:
    """Terminate a manifest whose last line lost its newline. Returns ``True``
    when a newline had to be added.

    A hard reboot mid-write leaves a partial final line. Appending straight onto
    it FUSES the two rows: the previous row becomes unparseable *and* the new row
    is invisible to :func:`completed_pairs`, so that pair is re-run on every
    resume, appending another mangled row each time and compounding forever.
    Terminating the tail first costs one seek and confines the damage to the one
    line the reboot actually broke.
    """

    try:
        with open(path, "r+b") as handle:
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            if size == 0:
                return False
            handle.seek(size - 1)
            if handle.read(1) == b"\n":
                return False
            handle.seek(size)
            handle.write(b"\n")
            handle.flush()
            try:
                os.fsync(handle.fileno())
            except OSError:
                pass
            return True
    except OSError:
        # No file yet (the common case) or it is unreadable; the append below
        # will either create it or fail loudly on its own terms.
        return False


def append_manifest(run_dir: Path, row: Dict[str, Any]) -> None:
    """Append one JSON-Lines row and flush it to the OS immediately.

    Flushed (and fsynced where possible) because this file *is* the resume state:
    a row still sitting in a buffer when the machine reboots means that pair is
    silently redone, which the user explicitly forbade.
    """

    path = Path(run_dir) / MANIFEST_NAME
    path.parent.mkdir(parents=True, exist_ok=True)
    _repair_truncated_tail(path)
    line = json.dumps(row, ensure_ascii=False, default=_json_default)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(line + "\n")
        handle.flush()
        try:
            os.fsync(handle.fileno())
        except OSError:
            pass


def refresh_reports(run_dir: Path, *, started_at: str = "", finished_at: str = "") -> None:
    """Regenerate SUMMARY.txt and failures_by_code.txt from the manifest.

    Called after *every* pair so progress is checkable mid-run, and wrapped so a
    formatting bug can never kill the batch that produced the data.
    """

    run_dir = Path(run_dir)
    rows = report.load_manifest(run_dir / MANIFEST_NAME)
    try:
        _write_text(
            run_dir / SUMMARY_NAME,
            report.build_summary(rows, started_at=started_at, finished_at=finished_at),
        )
        _write_text(run_dir / FAILURES_NAME, report.build_failures_by_code(rows))
    except Exception:  # noqa: BLE001
        _write_text(run_dir / SUMMARY_NAME, "report generation failed:\n" + traceback.format_exc())


# ---------------------------------------------------------------------------
# RESULT.txt -- "exactly what went wrong", in plain words.
# ---------------------------------------------------------------------------
def result_text(row: Dict[str, Any], *, paper: Optional[Dict[str, Any]] = None) -> str:
    """Render the per-paper-per-mode verdict file."""

    row = _safe_dict(row)
    paper = _safe_dict(paper)
    status = _text(row.get("status")).lower() or "unknown"
    warnings = row_warnings(row)
    verdict = "PASS" if status == _STATUS_PASS else "FAIL"
    if verdict == "PASS" and warnings:
        # A pass that produced no deliverable must not read like a clean run.
        verdict = "PASS (with warnings)"

    lines: List[str] = [
        f"RESULT: {verdict}",
        "=" * 40,
        f"paper      : {_text(row.get('paper_id')) or '(unknown)'}",
        f"title      : {_text(paper.get('title')) or _text(row.get('title')) or '(unknown)'}",
        f"slug       : {_text(row.get('slug')) or '(unknown)'}",
        f"mode       : {_text(row.get('mode')) or '(unknown)'}",
        f"status     : {status}",
        f"stage      : {_text(row.get('stage')) or '(unknown)'}",
        f"wall time  : {_fmt_seconds(row.get('seconds'))}",
    ]

    if warnings:
        lines += ["", "WARNINGS", "-" * 40]
        lines += [f"  ! {item}" for item in warnings]

    if status != _STATUS_PASS:
        codes = _safe_list(row.get("issue_codes"))
        lines += [
            "",
            "WHAT WENT WRONG",
            "-" * 40,
            f"failure_kind : {_text(row.get('failure_kind')) or '(unclassified)'}",
            f"message      : {_text(row.get('message')) or '(no message recorded)'}",
            f"issue codes  : {', '.join(str(code) for code in codes) if codes else '(none reported)'}",
            "",
            "DETAIL / TRACEBACK",
            "-" * 40,
            _text(row.get("detail")) or "(no detail was captured)",
        ]
    else:
        lines += ["", f"message    : {_text(row.get('message')) or '(none)'}"]

    counts = _safe_dict(row.get("counts"))
    if counts:
        lines += ["", "COUNTS", "-" * 40]
        for key in sorted(counts):
            lines.append(f"  {key} = {counts[key]}")

    files = _safe_list(row.get("files"))
    lines += ["", "FILES WRITTEN", "-" * 40]
    if files:
        for item in files:
            data = _safe_dict(item)
            name = _text(data.get("name")) or _text(item)
            size = data.get("bytes")
            error = _text(data.get("error"))
            lines.append(f"  - {Path(name).name}" + (f"  ({size} B)" if size else ""))
            if error:
                # Never silent: an artifact the driver produced but that never
                # reached the disk is the difference between a real pass and a lie.
                lines.append(f"      !! WRITE FAILED: {error}")
    else:
        lines.append("  (none)")

    return "\n".join(lines) + "\n"


def write_artifacts(target_dir: Path, artifacts: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Write the driver's artifacts into ``target_dir``.

    ``Path(name).name`` strips any directory part: an artifact key is a filename,
    and a key containing a separator must not be able to write outside the run.
    """

    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    written: List[Dict[str, Any]] = []
    for name, blob in sorted(_safe_dict(artifacts).items()):
        safe = Path(_text(name)).name
        if not safe:
            continue
        path = target_dir / safe
        try:
            if isinstance(blob, (bytes, bytearray)):
                path.write_bytes(bytes(blob))
                size = len(blob)
            else:
                body = blob if isinstance(blob, str) else _json_text(blob)
                path.write_text(body, encoding="utf-8")
                size = len(body.encode("utf-8"))
        except Exception as exc:  # noqa: BLE001 - a full disk, a lock, a bad name
            written.append({"name": safe, "bytes": 0, "error": f"{type(exc).__name__}: {exc}"})
            continue
        written.append({"name": safe, "bytes": size})
    return written


def failed_writes(files: Sequence[Any]) -> List[Dict[str, Any]]:
    """The entries of a ``files`` list whose write did not happen."""

    out: List[Dict[str, Any]] = []
    for item in _safe_list(files):
        data = _safe_dict(item)
        if _text(data.get("error")):
            out.append(data)
    return out


def row_warnings(row: Dict[str, Any]) -> List[str]:
    """Every reason a *passing* run must not be reported as clean.

    Two independent kinds, and both have to be here. ``warnings`` is what the
    driver reports (it ran, but produced no deliverable). A failed artifact write
    is invisible to the driver -- it handed over bytes and only the write knows --
    and when the lost artifact is not *required* for the mode the run correctly
    stays a ``pass``, so ``status`` will never carry it either. ``report`` already
    treats both as "warned" (``ModeRun.warned``); reading only ``warnings`` here
    is how a night that silently wrote nothing to disk got logged as ALL GREEN.
    """

    out = [_text(item) for item in _safe_list(row.get("warnings")) if _text(item)]
    for entry in failed_writes(row.get("files")):
        name = Path(_text(entry.get("name")) or "(unnamed artifact)").name
        out.append(f"artifact {name} could not be written: {_text(entry.get('error'))}")
    return _dedup_warnings(out)


def _dedup_warnings(items: Sequence[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def required_artifacts(mode: str, artifacts: Dict[str, Any]) -> List[str]:
    """The names this mode *must* land on disk, given what the driver produced.

    Only artifacts the driver actually produced are required: research mode with
    no RAG synthesis legitimately has no report, and strict mode that never got
    to export legitimately has no ``.pwml`` -- those are already judged upstream.
    What is checked here is narrower and absolute: if the bytes existed and the
    write failed, the run did not pass.
    """

    present = _safe_dict(artifacts)
    return [name for name in REQUIRED_ARTIFACTS.get(mode, ()) if name in present]


# ---------------------------------------------------------------------------
# Manifest rows.
# ---------------------------------------------------------------------------
def _identify(row: Dict[str, Any], paper: Dict[str, Any], slug: str, mode: str) -> Dict[str, Any]:
    """Stamp the paper identity onto a row so ``report`` can group and name it."""

    row = dict(row)
    row["slug"] = slug
    row["mode"] = _text(row.get("mode")) or mode
    row["paper_id"] = _text(row.get("paper_id")) or _text(paper.get("paper_id")) or slug
    for key in ("title", "source_uri", "topic"):
        value = _text(paper.get(key))
        if value and not _text(row.get(key)):
            row[key] = value
    row["dir"] = f"{PAPERS_DIRNAME}/{slug}/{mode}"
    return row


def _relocate_files(row: Dict[str, Any], slug: str, mode: str) -> Dict[str, Any]:
    """Make ``files`` entries run-dir-relative so SUMMARY.txt names real paths."""

    row = dict(row)
    prefix = f"{PAPERS_DIRNAME}/{slug}/{mode}/"
    relocated: List[Dict[str, Any]] = []
    for item in _safe_list(row.get("files")):
        data = _safe_dict(item)
        name = _text(data.get("name")) or _text(item)
        if not name:
            continue
        # Carry the WHOLE record forward, only rewriting ``name``. Rebuilding it
        # from scratch silently dropped the ``error`` key, which is the only
        # record that an artifact never reached the disk -- so a run whose
        # deliverable failed to write was reported as a clean pass. This runs on
        # both the child and the parent side, so a drop here is unrecoverable.
        entry: Dict[str, Any] = dict(data)
        entry["name"] = prefix + Path(name).name
        relocated.append(entry)
    row["files"] = relocated
    return row


def _timeout_row(*, slug: str, mode: str, paper: Dict[str, Any], seconds: float, timeout: float, tail: str) -> Dict[str, Any]:
    return _identify(
        {
            "status": _STATUS_TIMEOUT,
            "stage": "unknown",
            "seconds": round(seconds, 2),
            "failure_kind": _STATUS_TIMEOUT,
            "message": (
                f"the child process was still running after {timeout:.0f}s and was killed, "
                "so this paper+mode produced nothing"
            ),
            "detail": (
                "The parent kills a stuck child rather than letting it eat the night. "
                "Typical causes: an LLM request with no server-side timeout, a wedged "
                "PathBank MySQL socket, or a Streamlit rerun that never returns.\n\n"
                + (tail or "(the child printed nothing before it was killed)")
            ),
            "issue_codes": [],
            "counts": {},
            "files": [],
        },
        paper,
        slug,
        mode,
    )


def _crash_row(*, slug: str, mode: str, paper: Dict[str, Any], seconds: float, returncode: Any, tail: str) -> Dict[str, Any]:
    return _identify(
        {
            "status": _STATUS_ERROR,
            "stage": "unknown",
            "seconds": round(seconds, 2),
            "failure_kind": driver.KIND_CRASH,
            "message": (
                f"the child process exited with code {returncode} without printing a "
                "result line, so it crashed rather than hung"
            ),
            "detail": tail or "(the child printed nothing at all)",
            "issue_codes": [],
            "counts": {},
            "files": [],
        },
        paper,
        slug,
        mode,
    )


# ---------------------------------------------------------------------------
# The child half: one paper, one mode.
# ---------------------------------------------------------------------------
def run_single(
    run_dir: Path,
    slug: str,
    mode: str,
    *,
    timeout: float = DEFAULT_PAPER_TIMEOUT,
    run_fn: Optional[Callable[..., RunOutcome]] = None,
) -> Dict[str, Any]:
    """Run one paper+mode, write its folder, and return its manifest row.

    Executed in the child process. Everything is written here (not in the parent)
    so a crash after the pipeline succeeded still leaves the artifacts on disk.
    """

    run_dir = Path(run_dir)
    mode = parse_modes(mode)[0]
    paper = paper_from_plan(run_dir, slug)
    target = paper_dir(run_dir, slug) / mode
    target.mkdir(parents=True, exist_ok=True)

    read_error = _text(paper.get("source_text_error"))
    if read_error:
        # An unreadable input file is an infrastructure fault, not a biology one.
        row = _identify(
            {
                "status": _STATUS_ERROR,
                "stage": "init",
                "seconds": 0.0,
                "failure_kind": driver.KIND_CRASH,
                "message": (
                    "the paper's source text file could not be read, so the app was never "
                    f"started: {_text(paper.get('source_text_path')) or SOURCE_TEXT_NAME}"
                ),
                "detail": (
                    f"{read_error}\n\nThis is an infrastructure fault in the run directory, "
                    "not an extraction failure: the file is missing or unreadable, so nothing "
                    "was ever handed to the app."
                ),
                "issue_codes": [],
                "counts": {},
                "files": [],
                "warnings": [],
            },
            paper,
            slug,
            mode,
        )
        _write_text(target / RESULT_NAME, result_text(row, paper=paper))
        return _relocate_files(row, slug, mode)

    runner = run_fn or driver.run_one
    outcome = runner(paper, mode, timeout=max(60.0, float(timeout)))

    row = _identify(outcome.to_dict(), paper, slug, mode)
    row["files"] = write_artifacts(target, getattr(outcome, "artifacts", {}) or {})

    # A "pass" whose required artifact never reached the disk is a failure. The
    # driver cannot see this: it hands over bytes, and only the write knows.
    lost = [
        entry
        for entry in failed_writes(row["files"])
        if _text(entry.get("name")) in required_artifacts(mode, getattr(outcome, "artifacts", {}) or {})
    ]
    if lost:
        names = ", ".join(_text(entry.get("name")) for entry in lost)
        reasons = "; ".join(_text(entry.get("error")) or "(no reason recorded)" for entry in lost)
        row["status"] = "fail"
        row["failure_kind"] = driver.KIND_CRASH
        row["message"] = (
            f"the pipeline succeeded but the required artifact(s) {names} could not be "
            f"written, so this run produced nothing on disk: {reasons}"
        )
        row["detail"] = "\n".join(
            part for part in (_text(row.get("detail")), f"failed writes: {reasons}") if part
        )

    _write_text(target / RESULT_NAME, result_text(row, paper=paper))
    return _relocate_files(row, slug, mode)


def emit_outcome(row: Dict[str, Any]) -> None:
    """Print the child's single result line, sentinel-prefixed.

    Written as explicit UTF-8 *bytes* to ``sys.stdout.buffer`` rather than
    printed. The child's stdout is always a pipe, and a piped stream's encoding
    is the ANSI codepage: printing ``ensure_ascii=False`` JSON containing one
    Greek beta then raises ``UnicodeEncodeError`` after the artifacts and
    RESULT.txt are already written, the parent sees no row and records a crash,
    and because a row now exists the pair is never retried. Losing a completed
    paper that way is the single most expensive bug in this file, so the encoding
    is pinned here as well as in the environment and in ``force_utf8_stdio``.
    """

    line = OUTCOME_SENTINEL + " " + json.dumps(row, ensure_ascii=False, default=_json_default)
    buffer = getattr(sys.stdout, "buffer", None)
    if buffer is not None:
        try:
            buffer.write(line.encode("utf-8", "replace") + b"\n")
            buffer.flush()
            return
        except Exception:  # noqa: BLE001 - detached/closed buffer: fall through
            pass
    # No buffer (a captured/wrapped stdout): ASCII-only JSON cannot be refused by
    # any codec, and \uXXXX escapes are still valid JSON for the parent.
    print(
        OUTCOME_SENTINEL + " " + json.dumps(row, ensure_ascii=True, default=_json_default),
        flush=True,
    )


def parse_child_output(stdout: str) -> Optional[Dict[str, Any]]:
    """Recover the child's row from its stdout, or ``None`` if it printed none.

    The sentinel is authoritative. The bare-JSON fallback exists only so a child
    that was run by hand (or by a future caller that forgot the sentinel) is not
    reported as a crash.
    """

    fallback: Optional[Dict[str, Any]] = None
    for line in _text(stdout).splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith(OUTCOME_SENTINEL):
            payload = stripped[len(OUTCOME_SENTINEL) :].strip()
            try:
                parsed = json.loads(payload)
            except Exception:  # noqa: BLE001 - truncated line: keep looking
                continue
            if isinstance(parsed, dict):
                return parsed
            continue
        if stripped.startswith("{") and stripped.endswith("}"):
            try:
                parsed = json.loads(stripped)
            except Exception:  # noqa: BLE001
                continue
            if isinstance(parsed, dict) and "status" in parsed:
                fallback = parsed
    return fallback


# ---------------------------------------------------------------------------
# The parent half: spawning and killing children.
# ---------------------------------------------------------------------------
@dataclass
class ChildResult:
    """What the parent learned from one child process."""

    returncode: Optional[int] = None
    stdout: str = ""
    stderr: str = ""
    timed_out: bool = False


def _kill_tree(proc: Any) -> None:
    """Kill the child *and its descendants*.

    ``proc.kill()`` alone only kills the direct child; a Streamlit/LLM run may
    have spawned workers that would keep the pipes open and the CPU busy. On
    Windows ``taskkill /F /T`` walks the tree. On POSIX this degrades to killing
    the process group we asked for at spawn time, then the process itself -- the
    whole function is guarded so it is a harmless no-op wherever a step is
    unavailable.
    """

    pid = getattr(proc, "pid", None)
    if os.name == "nt" and pid:
        try:
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(pid)],
                capture_output=True,
                timeout=60,
                check=False,
            )
        except Exception:  # noqa: BLE001 - taskkill missing or already-dead pid
            pass
    elif pid:
        try:
            os.killpg(os.getpgid(pid), 9)  # type: ignore[attr-defined]
        except Exception:  # noqa: BLE001 - no killpg on this platform, or gone
            pass
    try:
        proc.kill()
    except Exception:  # noqa: BLE001
        pass


def launch_child(cmd: Sequence[str], timeout: float) -> ChildResult:
    """``subprocess.run(cmd, timeout=...)`` semantics, plus a process-tree kill.

    ``subprocess.run`` is not used directly for exactly one reason: on timeout it
    kills only the immediate child, and a hung pipeline run may have descendants
    holding the very socket that hung. So the process is started in its own group
    and :func:`_kill_tree` is used instead.
    """

    kwargs: Dict[str, Any] = {
        "stdout": subprocess.PIPE,
        "stderr": subprocess.PIPE,
        "text": True,
        "encoding": "utf-8",
        "errors": "replace",
        # The pipes are decoded as UTF-8 above, so the child must ENCODE as UTF-8.
        # Without this its piped stdout defaults to the ANSI codepage and one
        # non-cp1252 character in an entity name loses the whole result row.
        "env": child_env(),
    }
    if os.name == "nt":
        kwargs["creationflags"] = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    else:
        kwargs["start_new_session"] = True

    proc = subprocess.Popen(list(cmd), **kwargs)  # noqa: S603 - argv is ours
    try:
        stdout, stderr = proc.communicate(timeout=max(1.0, float(timeout)))
    except subprocess.TimeoutExpired:
        _kill_tree(proc)
        try:
            stdout, stderr = proc.communicate(timeout=_DRAIN_TIMEOUT)
        except Exception:  # noqa: BLE001 - pipes may never close; do not hang here
            stdout, stderr = "", ""
        return ChildResult(None, stdout or "", stderr or "", True)
    except KeyboardInterrupt:
        # Ctrl+C reaches the parent only (the child is in its own group), so the
        # child must be taken down explicitly or it would outlive the batch.
        _kill_tree(proc)
        raise
    return ChildResult(proc.returncode, stdout or "", stderr or "", False)


def _tail(result: ChildResult, *, limit: int = 4000) -> str:
    parts = []
    if result.stdout.strip():
        parts.append("--- child stdout (tail) ---\n" + result.stdout[-limit:])
    if result.stderr.strip():
        parts.append("--- child stderr (tail) ---\n" + result.stderr[-limit:])
    return "\n\n".join(parts)


def child_command(script_path: Path, run_dir: Path, slug: str, mode: str, timeout: float) -> List[str]:
    """The argv the parent uses to re-invoke this same CLI for one pair."""

    return [
        sys.executable,
        str(Path(script_path)),
        "--single",
        "--run-dir",
        str(Path(run_dir)),
        "--slug",
        slug,
        "--mode",
        mode,
        "--timeout",
        f"{max(60.0, float(timeout) - _CHILD_GRACE):.0f}",
    ]


# ---------------------------------------------------------------------------
# Status.
# ---------------------------------------------------------------------------
def print_status(out_dir: Any = None, *, echo: Callable[[str], None] = print) -> int:
    """Print one progress line for the current/most recent run. Always exits 0."""

    base = resolve_out_dir(out_dir)
    run_dir = latest_run_dir(base)
    if run_dir is None:
        echo(f"no runs found under {base}")
        return 0
    plan = load_plan(run_dir)
    rows = report.load_manifest(run_dir / MANIFEST_NAME)
    total = len(_safe_list(plan.get("papers")))
    pending = pending_pairs(run_dir, plan)
    running = f"{len(pending)} pair(s) pending" if pending else "complete"
    echo(f"{run_dir.name}  {report.status_line(rows, total_expected=total, running=running)}")
    echo(f"summary: {run_dir / SUMMARY_NAME}")
    return 0


# ---------------------------------------------------------------------------
# The batch loop.
# ---------------------------------------------------------------------------
def _plan_for_fresh_run(
    run_dir: Path,
    *,
    topics_path: Path,
    modes: List[str],
    limit: Optional[int],
    fetch_fn: Callable[..., Tuple[List[BatchPaper], List[Dict[str, Any]]]],
    log: Callable[[str], None],
) -> Dict[str, Any]:
    """Fetch the papers, lay out their folders, and write ``plan.json``."""

    topics_text = _read_text(topics_path)
    if not topics_text.strip():
        log(f"topics file is empty or missing: {topics_path}")
    log(f"fetching papers (topics={topics_path}, limit={limit if limit else 'none'}) ...")
    try:
        papers, skipped = fetch_fn(topics_text, limit=limit)
    except Exception:  # noqa: BLE001 - fetch_papers promises not to raise, but
        log("paper acquisition crashed; see plan.json for the traceback")
        papers, skipped = [], [{"reason": "fetch_crashed", "detail": traceback.format_exc()}]

    _write_text(run_dir / SKIPPED_NAME, _json_text(skipped) + "\n")
    for paper in papers:
        write_paper_inputs(run_dir, paper)

    # plan.json is written BEFORE the per-paper titles are logged, deliberately.
    # A title is arbitrary text from a journal (Greek letters, em dashes), and
    # logging it is the riskiest thing in this function. With the plan already on
    # disk, even a catastrophic log failure leaves a resumable run directory
    # instead of an orphan that find_resumable refuses and the next launch
    # replaces with another empty one, forever.
    plan = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "topics_path": str(topics_path),
        "modes": list(modes),
        "limit": limit,
        "papers": [_plan_record(paper) for paper in papers],
        "skipped": len(skipped),
    }
    save_plan(run_dir, plan)
    for paper in papers:
        log(f"  + {paper.slug}  ({len(paper.full_text)} chars)  {paper.title or '(untitled)'}")
    log(f"planned {len(papers)} paper(s) x {len(modes)} mode(s) = {len(papers) * len(modes)} run(s); "
        f"{len(skipped)} candidate(s) skipped (see {SKIPPED_NAME})")
    return plan


def _orphan_rescue(run_dir: Optional[Path], *, echo: bool, error: str) -> None:
    """Make a run directory that died mid-setup readable and resumable.

    Called only when :func:`_run_batch` raised something nobody predicted. The
    failure mode this exists for: the crash lands after ``new_run_dir`` but before
    ``save_plan``, so the directory has no ``plan.json``, :func:`find_resumable`
    refuses it, and every subsequent launch creates ANOTHER empty directory --
    zero papers run, forever, with nothing on disk saying why.
    """

    if run_dir is None:
        return
    log = Logger(run_dir, echo=echo)
    log("UNEXPECTED ERROR in run_batch -- see the traceback below")
    for line in error.splitlines():
        log("  " + line)
    if not load_plan(run_dir):
        # An empty work list is honest: there is nothing to continue, so the next
        # launch starts fresh and fetches again instead of finding an orphan.
        save_plan(
            run_dir,
            {
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "modes": list(ALL_MODES),
                "papers": [],
                "aborted": True,
                "error": error,
            },
        )
        log(f"wrote a minimal {PLAN_NAME} so this directory is not an orphan")
    try:
        refresh_reports(run_dir)
    except Exception:  # noqa: BLE001 - already on the failure path
        pass


def run_batch(
    *,
    topics_path: Any = None,
    out: Any = None,
    modes: Any = None,
    limit: Optional[int] = None,
    timeout: float = DEFAULT_PAPER_TIMEOUT,
    deadline_hours: float = DEFAULT_DEADLINE_HOURS,
    fresh: bool = False,
    script_path: Any = None,
    fetch_fn: Optional[Callable[..., Tuple[List[BatchPaper], List[Dict[str, Any]]]]] = None,
    child_fn: Optional[Callable[[Sequence[str], float], ChildResult]] = None,
    data_dir: Any = None,
    echo: bool = True,
    clock: Callable[[], float] = time.monotonic,
) -> int:
    """Run the whole batch. Returns the process exit code (0 = everything passed).

    A thin guard around :func:`_run_batch`: whatever goes wrong, the run
    directory is left with a ``plan.json`` and a readable log rather than as an
    orphan the runner would replace with a fresh empty one every night.
    """

    state: Dict[str, Any] = {"run_dir": None}
    try:
        return _run_batch(
            state,
            topics_path=topics_path,
            out=out,
            modes=modes,
            limit=limit,
            timeout=timeout,
            deadline_hours=deadline_hours,
            fresh=fresh,
            script_path=script_path,
            fetch_fn=fetch_fn,
            child_fn=child_fn,
            data_dir=data_dir,
            echo=echo,
            clock=clock,
        )
    except KeyboardInterrupt:
        raise
    except Exception:  # noqa: BLE001 - an unattended run must never orphan its dir
        _orphan_rescue(state.get("run_dir"), echo=echo, error=traceback.format_exc())
        return 1


def _run_batch(
    state: Dict[str, Any],
    *,
    topics_path: Any = None,
    out: Any = None,
    modes: Any = None,
    limit: Optional[int] = None,
    timeout: float = DEFAULT_PAPER_TIMEOUT,
    deadline_hours: float = DEFAULT_DEADLINE_HOURS,
    fresh: bool = False,
    script_path: Any = None,
    fetch_fn: Optional[Callable[..., Tuple[List[BatchPaper], List[Dict[str, Any]]]]] = None,
    child_fn: Optional[Callable[[Sequence[str], float], ChildResult]] = None,
    data_dir: Any = None,
    echo: bool = True,
    clock: Callable[[], float] = time.monotonic,
) -> int:
    """The batch loop itself.

    Resume is the default and needs no flag: the newest run directory is
    continued when it is recent and still owes work. ``fresh=True`` forces a new
    one. ``state["run_dir"]`` is published as soon as the directory exists so the
    caller can rescue it if this raises.
    """

    started_clock = clock()
    deadline_seconds = max(0.0, float(deadline_hours)) * 3600.0
    out_dir = resolve_out_dir(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    script = Path(script_path) if script_path else (PROJECT_ROOT / "scripts" / "batch_run.py")
    cli_modes = parse_modes(modes) if modes else None

    # find_resumable runs before there is a log file, so its reasoning is
    # collected and replayed once the Logger exists. Why it chose what it chose
    # is exactly what a 3am post-mortem needs.
    notes: List[str] = []
    resumed = None if fresh else find_resumable(out_dir, log=notes.append)
    if resumed is not None:
        run_dir = resumed
        state["run_dir"] = run_dir
        log = Logger(run_dir, echo=echo)
        for note in notes:
            log("  resume check     : " + note)
        plan = load_plan(run_dir)
        if cli_modes and cli_modes != parse_modes(plan.get("modes")):
            plan["modes"] = cli_modes
            save_plan(run_dir, plan)
            log(f"--modes overrides the plan; now running: {', '.join(cli_modes)}")
        log(f"CONTINUING the incomplete run {run_dir.name} (no --fresh given)")
        log(f"  already recorded : {len(completed_pairs(run_dir))} paper+mode run(s) -- these are skipped")
        log(f"  still to do      : {len(pending_pairs(run_dir, plan))} paper+mode run(s)")
    else:
        run_dir = new_run_dir(out_dir)
        state["run_dir"] = run_dir
        log = Logger(run_dir, echo=echo)
        if fresh:
            log(f"STARTING a fresh run {run_dir.name} (--fresh was given)")
        else:
            log(f"STARTING a new run {run_dir.name} (no incomplete run to continue)")
        for note in notes:
            log("  resume check     : " + note)
        copied = snapshot_caches(run_dir, data_dir=data_dir)
        log(
            f"  cache snapshot   : {', '.join(copied)}"
            if copied
            else "  cache snapshot   : no caches on disk yet, nothing to snapshot"
        )
        topics = Path(topics_path) if topics_path else (PROJECT_ROOT / "topics.txt")
        plan = _plan_for_fresh_run(
            run_dir,
            topics_path=topics,
            modes=cli_modes or list(ALL_MODES),
            limit=limit,
            fetch_fn=fetch_fn or fetch_papers,
            log=log,
        )

    started_at = _text(plan.get("created_at")) or datetime.now().isoformat(timespec="seconds")
    refresh_reports(run_dir, started_at=started_at)

    pending = pending_pairs(run_dir, plan)
    if not pending:
        log("nothing left to do in this run directory")
    else:
        log(f"running {len(pending)} paper+mode pair(s) sequentially")

    launch = child_fn or launch_child
    records = {
        _text(_safe_dict(rec).get("slug")): _safe_dict(rec)
        for rec in _safe_list(plan.get("papers"))
    }
    interrupted = False
    deadline_reached = False
    if deadline_seconds > 0:
        log(f"whole-night deadline : {deadline_hours:.1f}h from now")

    # STRICTLY SEQUENTIAL -- one child at a time, never a pool. The id-mapping
    # cache, the enrichment cache and the RAG index are shared mutable files with
    # read-modify-write access and no locking; concurrent runs corrupt them.
    for index, (slug, mode) in enumerate(pending, start=1):
        # The per-pair timeout alone does not bound the night: ten papers x two
        # modes x one hour is twenty hours, which is still running at lunchtime.
        spent = clock() - started_clock
        if deadline_seconds > 0 and spent >= deadline_seconds:
            deadline_reached = True
            not_attempted = len(pending) - index + 1
            log(
                f"deadline reached, {not_attempted} pair(s) not attempted "
                f"(after {spent / 3600.0:.1f}h of a {deadline_hours:.1f}h budget)"
            )
            break

        paper = records.get(slug, {"slug": slug, "paper_id": slug})
        label = f"[{index}/{len(pending)}] {slug} / {mode}"
        log(f"{label} -> starting (timeout {timeout:.0f}s)")
        began = clock()
        try:
            result = launch(child_command(script, run_dir, slug, mode, timeout), timeout)
        except KeyboardInterrupt:
            interrupted = True
            log("interrupted by Ctrl+C -- the manifest is valid and this run is resumable")
            break
        except Exception:  # noqa: BLE001 - a spawn failure is one bad pair, not the night
            result = ChildResult(None, "", traceback.format_exc(), False)
        elapsed = clock() - began

        # The child's own row is ground truth and is looked for FIRST, even on a
        # timeout: a child that finished its work, wrote its artifacts and printed
        # its row a moment before the parent's stopwatch expired really did pass,
        # and throwing that away would mark the pair done with a fabricated
        # failure -- never to be retried.
        parsed = parse_child_output(result.stdout)
        if parsed is not None:
            row = _relocate_files(_identify(parsed, paper, slug, mode), slug, mode)
            if result.timed_out:
                log(f"{label} -> the child was killed on timeout but had already "
                    "printed its result line; using the child's own row")
        elif result.timed_out:
            row = _timeout_row(
                slug=slug, mode=mode, paper=paper, seconds=elapsed, timeout=timeout,
                tail=_tail(result),
            )
        else:
            row = _crash_row(
                slug=slug, mode=mode, paper=paper, seconds=elapsed,
                returncode=result.returncode, tail=_tail(result),
            )

        # The child writes RESULT.txt on the paths it survives; the parent writes
        # it ONLY for the paths it does not (timeout, crash), so the file always
        # exists. Writing unconditionally clobbered the child's version, which is
        # the richer one -- it is the only place a per-artifact write error is
        # spelled out -- with a reconstruction that had already lost it.
        target = paper_dir(run_dir, slug) / mode
        result_path = target / RESULT_NAME
        if not result_path.exists():
            try:
                _write_text(result_path, result_text(row, paper=paper))
            except OSError:
                pass

        append_manifest(run_dir, row)
        refresh_reports(run_dir, started_at=started_at)
        status = _text(row.get("status")) or "unknown"
        detail = _text(row.get("failure_kind"))
        warned = row_warnings(row)
        log(
            f"{label} -> {status.upper()}"
            + (f" ({detail})" if status != _STATUS_PASS and detail else "")
            + (" WITH WARNINGS" if status == _STATUS_PASS and warned else "")
            + f" in {_fmt_seconds(row.get('seconds'))}"
        )
        for item in warned:
            log(f"    ! {item}")
        log("    " + report.status_line(report.load_manifest(run_dir / MANIFEST_NAME),
                                        total_expected=len(records)))

    finished_at = datetime.now().isoformat(timespec="seconds")
    refresh_reports(run_dir, started_at=started_at, finished_at=finished_at)

    rows = report.load_manifest(run_dir / MANIFEST_NAME)
    attempted = [row for row in rows if isinstance(row, dict) and _text(row.get("mode"))]
    failures = [row for row in attempted if _text(row.get("status")).lower() != _STATUS_PASS]
    warnings = [
        row
        for row in attempted
        if _text(row.get("status")).lower() == _STATUS_PASS and row_warnings(row)
    ]

    log(f"run directory : {run_dir}")
    log(f"summary       : {run_dir / SUMMARY_NAME}")
    log(f"fix-list      : {run_dir / FAILURES_NAME}")
    if deadline_reached:
        log(f"DEADLINE REACHED with {len(pending_pairs(run_dir))} pair(s) still pending; "
            "the run is resumable -- rerun the same command to finish it")
        return 1
    if not attempted:
        # No pair ran at all. Vacuously "nothing failed", but a night that
        # produced nothing is a failed night, so it is reported as one.
        log("NOTHING RAN: no paper+mode pair was attempted (no papers fetched?)")
        return 1
    if interrupted:
        log(f"INTERRUPTED with {len(pending_pairs(run_dir))} pair(s) still pending; "
            "rerun the same command to continue")
        return 1
    if failures:
        log(f"FAILURES: {len(failures)} of {len(attempted)} paper+mode run(s) did not pass")
        return 1
    if warnings:
        # Every run passed, but at least one produced no deliverable. Saying ALL
        # GREEN here is how a night that generated no research report at all gets
        # filed as a success and nobody looks in the folder.
        log(f"PASSED WITH WARNINGS: {len(attempted)} paper+mode run(s) passed but "
            f"{len(warnings)} produced no deliverable or lost an artifact on write "
            f"-- see {SUMMARY_NAME}")
        return 0
    log(f"ALL GREEN: {len(attempted)} paper+mode run(s) passed")
    return 0


__all__ = [
    "ALL_MODES",
    "DEFAULT_DEADLINE_HOURS",
    "DEFAULT_PAPER_TIMEOUT",
    "REQUIRED_ARTIFACTS",
    "RESUME_MAX_AGE_HOURS",
    "FAILURES_NAME",
    "LOG_NAME",
    "MANIFEST_NAME",
    "OUTCOME_SENTINEL",
    "PAPERS_DIRNAME",
    "PLAN_NAME",
    "RESULT_NAME",
    "SKIPPED_NAME",
    "SOURCE_TEXT_NAME",
    "SUMMARY_NAME",
    "ChildResult",
    "Logger",
    "append_manifest",
    "child_command",
    "child_env",
    "completed_pairs",
    "emit_outcome",
    "failed_writes",
    "find_resumable",
    "force_utf8_stdio",
    "required_artifacts",
    "row_warnings",
    "run_dir_touched_at",
    "latest_run_dir",
    "launch_child",
    "list_run_dirs",
    "load_plan",
    "new_run_dir",
    "paper_dir",
    "paper_from_plan",
    "parse_child_output",
    "parse_modes",
    "pending_pairs",
    "plan_pairs",
    "print_status",
    "refresh_reports",
    "resolve_out_dir",
    "result_text",
    "run_batch",
    "run_single",
    "save_plan",
    "snapshot_caches",
    "write_artifacts",
    "write_paper_inputs",
]
