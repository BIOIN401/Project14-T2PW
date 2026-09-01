"""What a leg that is killed, times out or crashes leaves behind. C-111 / F-148.

**The artifact needed to rule out retry amplification is exactly the artifact the
kill destroyed.**

Three T-107 legs timed out. ``batch.log`` contains zero occurrences of ``retry``,
``retrying``, ``attempt``, ``backoff``, ``rate limit`` or ``429`` -- and that does
**not** exclude retry amplification, because the three timed-out legs preserved no
attempt record of any kind. *"No evidence of retries"* is not *"evidence of no
retries"*; it is the absence of the instrument. This module is the instrument.

**It instruments. It does not fix.** Nothing here changes retry behaviour, the leg
ceiling, ``leg_timeout_override_*`` or the finalization seam, and nothing here
makes the parent guess a stage.

WHY EVERY RECORD IS FLUSHED AND fsync'ed AS IT IS MADE
-----------------------------------------------------
The C-111 probe measured the outer-kill path: on Windows the parent's
``runner._kill_tree`` is ``taskkill /F /T``, a FORCE kill, and a force-killed child
runs no ``atexit`` hook, no signal handler and no ``finally``. **A leg gets exactly
zero instructions at the moment it dies.** So a write-at-the-end checkpoint is
worth nothing on the path that actually loses payloads, and every event here is
appended, flushed and fsync'ed on its own, before the next one is attempted. An
item that exists only in memory at the moment of death is not preserved.

The file is JSON Lines for the same reason: a process killed mid-write truncates
at most the last line, and every earlier line is still readable.

THE NINE ITEMS, and where each comes from
-----------------------------------------
:data:`NINE_ITEMS` names them. Items 1, 2, 3 and 8 come from the CHILD's own
incremental trace (``LEG_TRACE.jsonl``); items 4, 5, 6, 7 and 9 come from the
PARENT, which is the only process still alive after a kill, and are written to
``LEG_TERMINAL.json`` beside it. :func:`summarize` reduces both back into the nine
off disk, which is how they are read back rather than asserted.

`stage=unknown` IS NOT A DEFECT AND IS NOT REPAIRED HERE
--------------------------------------------------------
On the outer-kill path ``runner._timeout_row`` records ``stage="unknown"`` because
**the parent genuinely does not know** -- it killed a child that never reported
back. That is honest, and F-148 § 6 records it as such. The in-process path knows
it was at ``stage="input"`` and declares its missing budget rather than guessing,
which is F-092 defect 3 closed. **Two legs, one ``stage`` field, two entirely
different epistemic situations.** The fix is not to make the parent guess -- that
would be a regression dressed as a repair -- it is to make the MECHANISM explicit,
which is item 5, :data:`TIMEOUT_SOURCES`.

SECURITY
--------
Attempt **counts, reasons, timings and hashes** are recorded. **Prompt and
response bodies are not**, and this module widens no existing policy on prompt
content: it records only what ``llm.client.CompletionDiagnostics`` already
computes -- which is counts, statuses, clipped error text and content hashes.
Every string reaching a durable record passes :func:`redact` first, whose
:data:`CREDENTIAL_PATTERNS` are the sprint's existing detectors, pinned equal to
``evidence/g11/g11_evidence.py``'s ``CRED_PATTERNS`` by test rather than re-derived.
"""

from __future__ import annotations

import json
import os
import re
import threading
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

#: The child's own incremental record. Append-only JSON Lines, one flushed and
#: fsync'ed line per event.
LEG_TRACE_NAME = "LEG_TRACE.jsonl"

#: The parent's terminal record, written after the child is gone. The parent is
#: the only process still alive on the outer-kill path, so items 4-7 and 9 can
#: only be written here.
LEG_TERMINAL_NAME = "LEG_TERMINAL.json"

#: Longest string kept on any durable record. Diagnostics are written on every
#: attempt, so nothing unbounded may enter them. Mirrors ``llm.client._DIAG_CHARS``
#: in intent; a little wider because a retry reason is the whole point here.
MAX_FIELD_CHARS = 400


# --------------------------------------------------------------------------- #
# Item 5 -- the timeout source. A CLOSED vocabulary, because the run currently
# reports two of these as one thing.
# --------------------------------------------------------------------------- #
#: The child stopped itself: its own in-process deadline fired and it said so.
#: ``PMC12444477/strict`` on T-107. The leg KNOWS where it was.
SOURCE_IN_PROCESS_DEADLINE = "in_process_deadline"

#: The batch parent killed a child that never reported back. ``PMC12444477/research``
#: and ``PMC12096016/strict`` on T-107. The parent does NOT know where the leg was,
#: and ``stage="unknown"`` on this path is that honesty, not a defect.
SOURCE_OUTER_PARENT_KILL = "outer_parent_kill"

#: Something outside the batch parent killed the whole process tree -- the G11
#: bounded wrapper's outer wall clock, or an operator. Distinguishable because the
#: child's trace has no terminal record beside it at all: the parent that would
#: have written one was killed too.
SOURCE_WRAPPER = "wrapper"

#: The provider boundary ran out of time: every attempt of one call exceeded its
#: request timeout (``llm.client.LLMOperationTimeout``). A provider casualty, and
#: F-159 is why this must be visible -- ``failure_kind = contract`` means only
#: "there were issue codes", so a provider casualty carrying any code wears a
#: ``contract`` label downstream.
SOURCE_PROVIDER = "provider"

#: No timeout happened. Recorded explicitly so "the field is empty" and "the field
#: was never written" are not the same reading.
SOURCE_NONE = "none"

TIMEOUT_SOURCES: Tuple[str, ...] = (
    SOURCE_IN_PROCESS_DEADLINE,
    SOURCE_OUTER_PARENT_KILL,
    SOURCE_WRAPPER,
    SOURCE_PROVIDER,
    SOURCE_NONE,
)


def require_timeout_source(value: Any) -> str:
    """Return ``value`` if it is one of the five sources, else raise.

    Closed for the same reason ``deadline.require_reason`` is closed: an invented
    source reaching a diagnostic is an unrecognised string a later reader buckets
    by hand, and the whole point of item 5 is that these mechanisms are genuinely
    different.
    """

    token = str(value or "").strip()
    if token not in TIMEOUT_SOURCES:
        raise ValueError(
            f"{value!r} is not one of the timeout sources: " + ", ".join(TIMEOUT_SOURCES)
        )
    return token


#: The nine items C-111 must preserve, in the charter's order. Named as data so a
#: test can assert the set rather than nine separate string literals.
NINE_ITEMS: Tuple[str, ...] = (
    "attempt_counts_by_stage",          # 1
    "retry_reasons",                    # 2
    "stage_elapsed_seconds",            # 3
    "finalization_reserve",             # 4
    "timeout_source",                   # 5
    "payload_before_cleanup",           # 6
    "cleanup_decisions",                # 7
    "total_model_calls",                # 8
    "terminal_state_before_cleanup",    # 9
)


# --------------------------------------------------------------------------- #
# Credential redaction. NOT a new detector set.
# --------------------------------------------------------------------------- #
#: The sprint's existing credential detectors, from
#: ``docs/pwml_recovery_sprint/evidence/g11/g11_evidence.py``'s ``CRED_PATTERNS``.
#:
#: **Vendored, not re-derived, and pinned byte-equal by test** -- production code
#: under ``src/`` cannot import a module that lives under ``docs/`` and is not on
#: the package path, and a divergent second detector set is exactly what the
#: charter forbids. ``tests/test_c111_timeout_observability.py`` loads
#: ``g11_evidence`` by path and asserts this list is pattern-for-pattern identical,
#: so the two cannot drift: change one and that test goes red.
#:
#: The comments the original carries about its boundaries are load-bearing and are
#: preserved with it: ``\\b`` on the OpenAI shape because ``task-``, ``risk-``,
#: ``disk-``, ``desk-`` and ``mask-`` are ordinary job vocabulary; deliberately NO
#: boundary on the Google shape.
CREDENTIAL_PATTERNS: List[Tuple[str, "re.Pattern[str]"]] = [
    ("openai_style_key", re.compile(r"\bsk-[A-Za-z0-9_\-]{16,}")),
    ("github_token", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}")),
    ("google_api_key", re.compile(r"AIza[0-9A-Za-z_\-]{20,}")),
    ("aws_access_key_id", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("bearer_token", re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._\-]{10,}")),
    ("inline_secret_assignment", re.compile(
        r"(?i)\b(api[_-]?key|apikey|access[_-]?token|auth[_-]?token|secret|"
        r"password|passwd)\b\s*[=:]\s*[^\s\"',]{6,}")),
    ("credentialed_url", re.compile(r"(?i)\b[a-z][a-z0-9+.\-]*://[^\s/@:]+:[^\s/@]+@")),
]


def scan_credentials(text: Any) -> List[str]:
    """Names of every credential shape found in ``text``. Empty is clean."""

    raw = text if isinstance(text, str) else ("" if text is None else str(text))
    return [name for name, pattern in CREDENTIAL_PATTERNS if pattern.search(raw)]


def redact(value: Any, *, limit: int = MAX_FIELD_CHARS) -> str:
    """``value`` as a bounded string with every credential shape replaced.

    Applied to every string that reaches a durable record. A diagnostic that
    leaked an API key would be worse than no diagnostic at all, and a scanner
    nobody has seen fail is not a scanner -- so the C-111 acceptance test plants
    an OpenAI-style key, a bearer token and an inline assignment in a retry reason
    and reads the artifact back off disk to watch each one get caught.
    """

    text = value if isinstance(value, str) else ("" if value is None else str(value))
    for name, pattern in CREDENTIAL_PATTERNS:
        text = pattern.sub(f"[redacted:{name}]", text)
    if len(text) > limit:
        text = f"{text[:limit]}… (+{len(text) - limit} chars elided)"
    return text


def _clean(value: Any, *, depth: int = 0) -> Any:
    """Recursively bound and redact one field value. Never raises."""

    if depth > 4:
        return "[depth limit]"
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, (int, float)):
        return value
    if isinstance(value, str):
        return redact(value)
    if isinstance(value, dict):
        return {str(k)[:64]: _clean(v, depth=depth + 1) for k, v in list(value.items())[:64]}
    if isinstance(value, (list, tuple, set)):
        return [_clean(v, depth=depth + 1) for v in list(value)[:64]]
    return redact(value)


# --------------------------------------------------------------------------- #
# The child's incremental trace
# --------------------------------------------------------------------------- #
class LegTrace:
    """One leg's append-only, flush-as-you-go record.

    Every method is guarded: instrumentation exists to make a killed leg
    reportable, so it must never be the thing that kills the leg. That is the
    same rule ``deadline.LegDeadline.checkpoint`` already obeys for ``persist``.
    """

    def __init__(
        self,
        path: Any,
        *,
        clock: Any = time.monotonic,
        child_deadline_seconds: float = 0.0,
        finalization_reserve_seconds: float = 0.0,
    ) -> None:
        self.path = Path(path)
        self._clock = clock
        self.started = float(clock())
        self.child_deadline_seconds = float(child_deadline_seconds)
        self.finalization_reserve_seconds = float(finalization_reserve_seconds)
        self._lock = threading.Lock()
        self._sequence = 0
        self._open_stages: Dict[str, float] = {}
        self.write_errors = 0
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        except Exception:  # noqa: BLE001 - a trace must never break a leg
            pass

    @property
    def elapsed(self) -> float:
        return max(0.0, float(self._clock()) - self.started)

    def event(self, kind: str, **fields: Any) -> Dict[str, Any]:
        """Append one event, flushed and fsync'ed before returning.

        Re-opened per event on purpose: a long-lived buffered handle loses
        everything not yet flushed when the process is force-killed, and the
        force kill is the case this whole module exists for.
        """

        with self._lock:
            self._sequence += 1
            record: Dict[str, Any] = {
                "seq": self._sequence,
                "kind": str(kind),
                "elapsed_seconds": round(self.elapsed, 3),
            }
            for key, value in fields.items():
                record[str(key)] = _clean(value)
            try:
                line = json.dumps(record, ensure_ascii=False, default=str)
            except Exception:  # noqa: BLE001
                line = json.dumps({"seq": self._sequence, "kind": str(kind),
                                   "unserializable": True})
            try:
                with open(self.path, "a", encoding="utf-8", newline="\n") as handle:
                    handle.write(line + "\n")
                    handle.flush()
                    os.fsync(handle.fileno())
            except Exception:  # noqa: BLE001 - see the class docstring
                self.write_errors += 1
            return record

    # -- item 3: per-stage elapsed ------------------------------------------ #
    def stage_begin(self, stage: str, **fields: Any) -> None:
        self._open_stages[str(stage)] = self.elapsed
        self.event("stage_begin", stage=str(stage), **fields)

    def stage_end(self, stage: str, **fields: Any) -> None:
        began = self._open_stages.pop(str(stage), None)
        self.event(
            "stage_end",
            stage=str(stage),
            stage_elapsed_seconds=(
                round(self.elapsed - began, 3) if began is not None else None
            ),
            **fields,
        )

    # -- items 1, 2, 8: attempts, retry reasons, model calls ----------------- #
    def model_attempt(
        self,
        *,
        stage: str = "",
        attempt: int = 0,
        status: str = "",
        model: str = "",
        reason: str = "",
        finish_reason: str = "",
        content_chars: int = 0,
        request_hash: str = "",
        response_hash: str = "",
    ) -> None:
        """One crossing of the provider boundary. Counts, reasons and hashes only.

        No prompt body and no response body reaches this record -- ``request_hash``
        and ``response_hash`` are ``llm.client``'s own content fingerprints, which
        is exactly what PRODUCT_CONTRACT § 9's *"response hashes"* asks for and no
        wider than the policy already in force.
        """

        self.event(
            "model_attempt",
            stage=str(stage or "unattributed"),
            attempt=int(attempt or 0),
            status=str(status or ""),
            model=str(model or ""),
            reason=str(reason or ""),
            finish_reason=str(finish_reason or ""),
            content_chars=int(content_chars or 0),
            request_hash=str(request_hash or ""),
            response_hash=str(response_hash or ""),
        )


# --------------------------------------------------------------------------- #
# Process-global activation. The child runs one leg at a time, by construction:
# ``_run_batch`` is STRICTLY SEQUENTIAL and each leg is its own process.
# --------------------------------------------------------------------------- #
_ACTIVE: Optional[LegTrace] = None


def activate(path: Any, **kwargs: Any) -> Optional[LegTrace]:
    """Start recording this process's leg. Returns ``None`` if it cannot."""

    global _ACTIVE
    try:
        _ACTIVE = LegTrace(path, **kwargs)
    except Exception:  # noqa: BLE001 - never break a leg to instrument it
        _ACTIVE = None
    return _ACTIVE


def active() -> Optional[LegTrace]:
    return _ACTIVE


def deactivate() -> None:
    global _ACTIVE
    _ACTIVE = None


def record_model_attempt(**kwargs: Any) -> None:
    """Publish one attempt to the active trace. A no-op when none is active.

    This is what ``llm.client.CompletionDiagnostics.note`` calls, so an attempt
    reaches disk BEFORE the next attempt begins -- which is the only ordering that
    survives a force kill.
    """

    trace = _ACTIVE
    if trace is None:
        return
    try:
        trace.model_attempt(**kwargs)
    except Exception:  # noqa: BLE001
        return


def record_event(kind: str, **fields: Any) -> None:
    """Publish one event to the active trace. A no-op when none is active."""

    trace = _ACTIVE
    if trace is None:
        return
    try:
        trace.event(kind, **fields)
    except Exception:  # noqa: BLE001
        return


# --------------------------------------------------------------------------- #
# Reading it back off disk
# --------------------------------------------------------------------------- #
def read_events(where: Any) -> List[Dict[str, Any]]:
    """Every event in a leg's trace. A truncated last line is skipped, not fatal.

    ``where`` may be the trace file or the leg directory that holds it.
    """

    path = Path(where)
    if path.is_dir():
        path = path / LEG_TRACE_NAME
    if not path.is_file():
        return []
    events: List[Dict[str, Any]] = []
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        return []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            parsed = json.loads(line)
        except Exception:  # noqa: BLE001 - a kill truncates at most the last line
            continue
        if isinstance(parsed, dict):
            events.append(parsed)
    return events


#: Files this module itself writes. They are artifacts of the INSTRUMENT, and
#: counting them as pipeline payload would make every killed leg look like it had
#: produced something.
INSTRUMENT_FILES = frozenset({LEG_TRACE_NAME, LEG_TERMINAL_NAME})


def scan_payload(leg_dir: Any) -> Dict[str, Any]:
    """Item 6: what is on disk in the leg directory, RIGHT NOW.

    Called by the parent at the moment the child dies, before anything is
    written, so it answers *"did a payload exist before cleanup"* by looking
    rather than by inference. ``runner._timeout_row`` writes ``files: []`` and
    ``counts: {}`` unconditionally; this is the reading that says whether that
    emptiness describes the disk.
    """

    directory = Path(leg_dir)
    files: List[Dict[str, Any]] = []
    try:
        entries = sorted(p for p in directory.iterdir() if p.is_file())
    except Exception:  # noqa: BLE001 - the directory may not exist at all
        entries = []
    for entry in entries:
        try:
            size = entry.stat().st_size
        except Exception:  # noqa: BLE001
            size = -1
        files.append({
            "name": entry.name,
            "bytes": size,
            "instrument": entry.name in INSTRUMENT_FILES,
        })
    payload = [f for f in files if not f["instrument"]]
    return {
        "existed": bool(payload),
        "leg_dir": str(directory),
        "files": files,
        "payload_file_count": len(payload),
        "payload_bytes": sum(max(0, int(f["bytes"])) for f in payload),
    }


def finalization_reserve_record(
    *,
    elapsed_seconds: float,
    leg_timeout_seconds: float,
    grace_seconds: float,
) -> Dict[str, Any]:
    """Item 4: how much of the finalization reserve was available, used and left.

    The reserve is the ``grace`` between the parent's leg ceiling and the child
    deadline it hands out (``deadline.child_deadline_seconds``): the window in
    which a leg that has run out of time writes down what § 9 requires it to
    preserve. ``used`` is how far past the child deadline the leg actually ran --
    on both T-107 outer-kill legs that was almost exactly the whole 120 s, which
    is F-148 § 3's sharpest finding.

    **This RECORDS the arithmetic. It changes no ceiling, no grace and no
    override** -- that is operational, not code, and out of scope for C-111.
    """

    from t2pw.pipeline.deadline import child_deadline_seconds

    ceiling = float(leg_timeout_seconds)
    grace = float(grace_seconds)
    child_deadline = child_deadline_seconds(ceiling, grace=grace)
    used = max(0.0, float(elapsed_seconds) - child_deadline)
    return {
        "available_seconds": round(grace, 2),
        "used_seconds": round(used, 2),
        "left_seconds": round(grace - used, 2),
        "child_deadline_seconds": round(child_deadline, 2),
        "leg_timeout_seconds": round(ceiling, 2),
        "elapsed_seconds": round(float(elapsed_seconds), 2),
        "exhausted": used >= grace,
    }


def classify_timeout_source(
    *,
    parent_killed: bool,
    child_reported: bool,
    termination_reason: str = "",
    provider_timeout: bool = False,
) -> str:
    """Item 5. Which mechanism ended this leg -- and they are genuinely different.

    * the parent killed a child that never reported back -> :data:`SOURCE_OUTER_PARENT_KILL`
    * the child reported its own timeout -> :data:`SOURCE_IN_PROCESS_DEADLINE`
    * every attempt of one provider call ran out -> :data:`SOURCE_PROVIDER`
    * nothing timed out -> :data:`SOURCE_NONE`

    :data:`SOURCE_WRAPPER` is deliberately NOT reachable from here: it names the
    case where the batch parent was killed too, so no parent survives to classify
    anything. It is inferred by a reader who finds a ``LEG_TRACE.jsonl`` with no
    ``LEG_TERMINAL.json`` beside it, and :func:`summarize` reports exactly that.

    **The parent is never asked to guess a STAGE here.** It is asked which
    mechanism fired, which is a fact it does hold.
    """

    if provider_timeout:
        return SOURCE_PROVIDER
    if parent_killed and not child_reported:
        return SOURCE_OUTER_PARENT_KILL
    if child_reported and str(termination_reason or "").strip():
        return SOURCE_IN_PROCESS_DEADLINE
    if parent_killed:
        return SOURCE_OUTER_PARENT_KILL
    return SOURCE_NONE


def record_terminal(
    leg_dir: Any,
    *,
    timeout_source: str,
    terminal_state: Dict[str, Any],
    finalization_reserve: Optional[Dict[str, Any]] = None,
    payload_before_cleanup: Optional[Dict[str, Any]] = None,
    cleanup_decisions: Optional[Iterable[Dict[str, Any]]] = None,
) -> Optional[Path]:
    """Write the PARENT's terminal record beside the child's trace.

    The parent is the only process still alive after an outer kill, so items 4,
    5, 6, 7 and 9 can only be written here. Guarded end to end: a diagnostic that
    can raise would turn a reportable failure into an unreportable one.
    """

    directory = Path(leg_dir)
    record = {
        "timeout_source": require_timeout_source(timeout_source),
        "terminal_state_before_cleanup": _clean(terminal_state),
        "finalization_reserve": _clean(finalization_reserve or {}),
        "payload_before_cleanup": _clean(payload_before_cleanup or {}),
        "cleanup_decisions": _clean(list(cleanup_decisions or [])),
    }
    try:
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / LEG_TERMINAL_NAME
        with open(path, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(record, indent=2, ensure_ascii=False, default=str))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        return path
    except Exception:  # noqa: BLE001
        return None


def cleanup_decision(
    *, artifact: str, decision: str, decided_by: str, detail: str = "", size_bytes: int = -1
) -> Dict[str, Any]:
    """Item 7: one cleanup decision affecting a partial artifact.

    What was discarded, and **by which decision** -- the second half being the one
    a post-mortem cannot reconstruct afterwards.
    """

    return {
        "artifact": redact(artifact),
        "decision": redact(decision),
        "decided_by": redact(decided_by),
        "detail": redact(detail),
        "bytes": int(size_bytes),
    }


def summarize(leg_dir: Any) -> Dict[str, Any]:
    """The nine items, reduced off disk. :data:`NINE_ITEMS` are always all present.

    Reading this back after a synthetic kill is REV-111 B1: an item that exists
    only in memory at the moment of death is not preserved, so the only proof that
    counts is a read from the filesystem.
    """

    directory = Path(leg_dir)
    events = read_events(directory)

    attempts_by_stage: Dict[str, int] = {}
    retry_reasons: List[Dict[str, Any]] = []
    stage_elapsed: Dict[str, float] = {}
    stage_first_last: Dict[str, List[float]] = {}
    model_calls = 0

    for event in events:
        kind = str(event.get("kind", ""))
        stage = str(event.get("stage", "") or "")
        elapsed = event.get("elapsed_seconds")
        if stage and isinstance(elapsed, (int, float)):
            span = stage_first_last.setdefault(stage, [float(elapsed), float(elapsed)])
            span[0] = min(span[0], float(elapsed))
            span[1] = max(span[1], float(elapsed))
        if kind == "model_attempt":
            model_calls += 1
            key = stage or "unattributed"
            attempts_by_stage[key] = attempts_by_stage.get(key, 0) + 1
            attempt_no = int(event.get("attempt") or 0)
            reason = str(event.get("reason") or "")
            status = str(event.get("status") or "")
            # Item 2 is "retry reason (per retry, WHERE ONE OCCURRED)", so the
            # rows kept are the attempts that did not succeed -- each one is the
            # reason the attempt after it happened. A successful third attempt is
            # counted in item 1 and item 8 and carries no retry reason, because
            # there is none: recording it with an empty ``reason`` would make a
            # clean retry look like an unexplained one.
            if reason or (status and status != "ok"):
                retry_reasons.append({
                    "stage": key,
                    "attempt": attempt_no,
                    "status": status,
                    "reason": reason,
                    "elapsed_seconds": elapsed,
                })
        elif kind == "stage_end":
            measured = event.get("stage_elapsed_seconds")
            if stage and isinstance(measured, (int, float)):
                stage_elapsed[stage] = round(float(measured), 3)

    for stage, (first, last) in stage_first_last.items():
        stage_elapsed.setdefault(stage, round(last - first, 3))

    terminal_path = directory / LEG_TERMINAL_NAME
    terminal: Dict[str, Any] = {}
    if terminal_path.is_file():
        try:
            loaded = json.loads(terminal_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                terminal = loaded
        except Exception:  # noqa: BLE001
            terminal = {}

    trace_exists = (directory / LEG_TRACE_NAME).is_file()
    ends = [e for e in events if str(e.get("kind")) == "leg_end"]
    closed = bool(ends)
    child_source = str(ends[-1].get("timeout_source") or "") if ends else ""
    if terminal:
        timeout_source = str(terminal.get("timeout_source") or SOURCE_NONE)
    elif child_source in TIMEOUT_SOURCES and child_source != SOURCE_NONE:
        # The child stopped ITSELF and said which mechanism did it. No parent
        # record is needed to know that, and the child is the only process that
        # was there.
        timeout_source = child_source
    elif trace_exists and not closed:
        # A trace that was never closed, with no parent record beside it: the
        # batch parent was killed too. That is the wrapper case, and it is the
        # only way it can be recognised.
        timeout_source = SOURCE_WRAPPER
    else:
        timeout_source = SOURCE_NONE

    summary: Dict[str, Any] = {
        "attempt_counts_by_stage": attempts_by_stage,
        "retry_reasons": retry_reasons,
        "stage_elapsed_seconds": stage_elapsed,
        "finalization_reserve": terminal.get("finalization_reserve", {}),
        "timeout_source": timeout_source,
        "payload_before_cleanup": terminal.get("payload_before_cleanup", {}),
        "cleanup_decisions": terminal.get("cleanup_decisions", []),
        "total_model_calls": model_calls,
        "terminal_state_before_cleanup": terminal.get("terminal_state_before_cleanup", {}),
    }
    summary["_trace_events"] = len(events)
    summary["_trace_present"] = trace_exists
    summary["_terminal_present"] = bool(terminal)
    summary["_trace_closed"] = closed
    return summary


__all__ = [
    "CREDENTIAL_PATTERNS",
    "INSTRUMENT_FILES",
    "LEG_TERMINAL_NAME",
    "LEG_TRACE_NAME",
    "MAX_FIELD_CHARS",
    "NINE_ITEMS",
    "SOURCE_IN_PROCESS_DEADLINE",
    "SOURCE_NONE",
    "SOURCE_OUTER_PARENT_KILL",
    "SOURCE_PROVIDER",
    "SOURCE_WRAPPER",
    "TIMEOUT_SOURCES",
    "LegTrace",
    "activate",
    "active",
    "classify_timeout_source",
    "cleanup_decision",
    "deactivate",
    "finalization_reserve_record",
    "read_events",
    "record_event",
    "record_model_attempt",
    "record_terminal",
    "redact",
    "require_timeout_source",
    "scan_credentials",
    "scan_payload",
    "summarize",
]
