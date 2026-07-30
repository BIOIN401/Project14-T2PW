"""Bounded, always-persisted diagnostics for the Stage-0/Stage-1 LLM boundaries.

WHY THIS MODULE EXISTS
----------------------
Until now a Stage-0/Stage-1 death produced one sentence and nothing on disk. The
two legs lost in ``runs/2026-07-28_0919`` are the canonical example: both wrote
``files: []`` and only ``RESULT.txt``, so the postmortem in
``t2pw/llm/client.py`` had to *infer* the cause from the shape of the failure
because nothing in the run recorded the model, the ``finish_reason``, how many
attempts were spent, or whether the provider had said anything at all.

Five different upstream faults all surface downstream as the same message
("Payload must include a processes object"):

  1. the provider returned an empty completion;
  2. it returned text that is not JSON;
  3. it returned valid JSON that declares no processes;
  4. it returned processes that *cleaning* then discarded row by row;
  5. cleaning kept rows but the stage contract rejected the result.

Those are :data:`OUTCOMES`. Telling them apart is the entire point: (1) is a
retry, (2) is a localized JSON repair, (3) is a prompt/scope problem, (4) is a
cleaning-rule problem and the discarded rows are the evidence, (5) is a contract
problem and the failing pointers are the evidence.

WHAT IS BOUNDED, AND WHY IT IS BOUNDED HERE
-------------------------------------------
Payload ``evidence`` fields reach six figures of characters and a merged payload
from a real run is 306 KB - 4.7 MB. A diagnostic summary that embeds them is
useless (nobody opens it) and expensive (it is written on every attempt). So the
rule this module enforces is: **diagnostics carry counts, identities and
hashes -- never repeated evidence blobs.** Every string is clipped through
``failure_detail.clip``, every sample list is capped at :data:`MAX_SAMPLE`, and
raw model text is reduced to a preview plus a :func:`payload_hash`. The hash is
what lets two attempts be compared ("the model returned the identical body
twice") without storing either body twice.

WHY THERE IS A MODULE-LEVEL RECORDER
------------------------------------
The facts are produced deep in the call stack -- inside ``chat()``'s retry loop,
inside ``_clean_processes``'s per-row ``continue`` -- and consumed at the top, by
the app and the batch driver. Threading a recorder through every signature in a
3,000-line pipeline would touch every caller and still miss the ones that matter
(the ones that raise). :func:`current` gives those sites a recorder without a new
parameter, and :func:`activate` lets a run scope it to a directory. Recording is
therefore never conditional, which is the property that makes "persist on
failures as well as passes" true rather than intended.

Writes are eager: :meth:`ExtractionDiagnostics.record_boundary` and friends flush
to ``artifact_dir`` as soon as they are called. A stage that raises has already
written its diagnostics by the time the exception unwinds, so no ``finally``
block anywhere has to be correct for the artifacts to exist.
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
from contextvars import ContextVar
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence
from uuid import uuid4

from t2pw.pipeline.failure_detail import clip

logger = logging.getLogger(__name__)

__all__ = [
    "STAGE0_ATTEMPTS_NAME",
    "BOUNDARY_REPORT_NAME",
    "CLEANING_REPORT_NAME",
    "MAPPED_FAILURE_SNAPSHOT_NAME",
    "AUDIT_ITERATIONS_NAME",
    "GAP_ITERATIONS_NAME",
    "ARTIFACT_NAMES",
    "BOUNDARY_STAGE0_PREPROCESS",
    "BOUNDARY_STAGE1_EXTRACTION",
    "BOUNDARY_STAGE1_JSON_REPAIR",
    "BOUNDARY_STAGE1_ROW_REPAIR",
    "BOUNDARY_STAGE2_INFERENCE",
    "OUTCOME_OK",
    "OUTCOME_EMPTY_COMPLETION",
    "OUTCOME_INVALID_JSON",
    "OUTCOME_ZERO_PROCESSES",
    "OUTCOME_DISCARDED_BY_CLEANING",
    "OUTCOME_CONTRACT_FAILED",
    "OUTCOME_SEMANTIC_GUARD_FAILED",
    "OUTCOMES",
    "MAX_SAMPLE",
    "MAX_ATTEMPTS_KEPT",
    "MAX_BOUND_DEPTH",
    "MAX_BOUND_KEYS",
    "MAX_BOUND_ITEMS",
    "BULK_KEYS",
    "PREVIEW_CHARS",
    "bound",
    "bound_mapping",
    "descriptor",
    "payload_hash",
    "count_entities",
    "count_processes",
    "DiscardLedger",
    "classify_outcome",
    "ExtractionDiagnostics",
    "current",
    "activate",
    "deactivate",
]


# ---------------------------------------------------------------------------
# Artifact names. These are the filenames an operator looks for after a failure,
# so they are constants rather than string literals scattered across the app,
# the pipeline and the batch driver.
# ---------------------------------------------------------------------------
STAGE0_ATTEMPTS_NAME = "stage0_attempts.json"
BOUNDARY_REPORT_NAME = "extraction_boundary_report.json"
CLEANING_REPORT_NAME = "cleaning_report.json"
MAPPED_FAILURE_SNAPSHOT_NAME = "mapped_failure_snapshot.json"
AUDIT_ITERATIONS_NAME = "audit_iteration_summary.json"
GAP_ITERATIONS_NAME = "gap_iteration_summary.json"

ARTIFACT_NAMES = (
    STAGE0_ATTEMPTS_NAME,
    BOUNDARY_REPORT_NAME,
    CLEANING_REPORT_NAME,
    MAPPED_FAILURE_SNAPSHOT_NAME,
    AUDIT_ITERATIONS_NAME,
    GAP_ITERATIONS_NAME,
)


# ---------------------------------------------------------------------------
# Boundary names. One per place a completion crosses from the provider into the
# pipeline; the repair boundaries are named separately from the stage that asked
# for them so "the extraction failed and then a repair also failed" is two rows,
# not one ambiguous one.
# ---------------------------------------------------------------------------
BOUNDARY_STAGE0_PREPROCESS = "stage0_preprocess"
BOUNDARY_STAGE1_EXTRACTION = "stage1_extraction"
BOUNDARY_STAGE1_JSON_REPAIR = "stage1_json_repair"
BOUNDARY_STAGE1_ROW_REPAIR = "stage1_row_repair"
BOUNDARY_STAGE2_INFERENCE = "stage2_inference"


# ---------------------------------------------------------------------------
# The five failure categories the module exists to separate, plus success.
# ---------------------------------------------------------------------------
#: The provider answered but sent no usable text at all.
OUTCOME_EMPTY_COMPLETION = "empty_completion"
#: Text arrived and is not parseable as a JSON object (salvage included).
OUTCOME_INVALID_JSON = "invalid_json"
#: A JSON object arrived and declares no process rows whatsoever.
OUTCOME_ZERO_PROCESSES = "valid_json_zero_processes"
#: Process rows arrived and cleaning dropped every one of them.
OUTCOME_DISCARDED_BY_CLEANING = "processes_discarded_by_cleaning"
#: Cleaning kept rows and the stage contract rejected the result anyway.
OUTCOME_CONTRACT_FAILED = "contract_failed_after_cleaning"
#: A repair draw came back parseable and was REFUSED because it did not preserve
#: the content it was asked to repair. Distinct from ``invalid_json`` on purpose:
#: the model produced valid JSON, so nothing about the syntax explains the
#: rejection, and an operator seeing this needs to know the guard fired rather
#: than that the model failed to close a brace.
OUTCOME_SEMANTIC_GUARD_FAILED = "semantic_guard_failed"
#: Nothing went wrong at this boundary.
OUTCOME_OK = "ok"

OUTCOMES = (
    OUTCOME_OK,
    OUTCOME_EMPTY_COMPLETION,
    OUTCOME_INVALID_JSON,
    OUTCOME_ZERO_PROCESSES,
    OUTCOME_DISCARDED_BY_CLEANING,
    OUTCOME_CONTRACT_FAILED,
    OUTCOME_SEMANTIC_GUARD_FAILED,
)


# ---------------------------------------------------------------------------
# Bounds. Every one of these is a hard cap applied at capture time.
# ---------------------------------------------------------------------------
#: Most discarded names / pointers kept per reason. Small on purpose: the count
#: is the finding, the sample only has to make the *shape* recognisable.
MAX_SAMPLE = 8

#: Most per-attempt rows kept on one boundary record. LLM_MAX_RETRIES defaults to
#: 8 and repair adds a few, so this never truncates a real run -- it only bounds
#: a pathological loop.
MAX_ATTEMPTS_KEPT = 24

#: Longest raw-text preview stored anywhere in these artifacts. A preview is for
#: recognising the shape of a reply ("it apologised", "it opened a code fence"),
#: not for reading it back.
PREVIEW_CHARS = 200

#: Longest single string kept in a sample entry.
_SAMPLE_CHARS = 120

#: How deep :func:`bound` walks before collapsing a value to a descriptor.
#:
#: Six, not four: the deepest legitimate structure in these artifacts is a
#: cleaning report's ``discarded_sample`` -> reason -> list -> entry -> value,
#: which sits at depth 4, and a boundary's ``attempt_log`` -> row -> value at
#: depth 3. A cap of 4 (``failure_detail.MAX_DEPTH``) would censure the sample
#: entries this module exists to show, so it gets its own, deliberately larger,
#: cap rather than borrowing that one.
MAX_BOUND_DEPTH = 6

#: Most keys kept from any one mapping inside a diagnostic value.
MAX_BOUND_KEYS = 24

#: Most items kept from any one sequence inside a diagnostic value.
MAX_BOUND_ITEMS = 12

#: Keys whose values are verbatim source text, model output, or provenance
#: blobs. These never survive as text: they become a
#: ``{"chars", "hash", "preview"}`` descriptor at any depth, because they are
#: the fields that reach six figures of characters and they are the fields an
#: artifact rewritten on every attempt must never carry twice.
BULK_KEYS = frozenset(
    {
        "abstract",
        "body",
        "chunk",
        "chunk_text",
        "chunks",
        "content",
        "evidence",
        "full_text",
        "message",
        "passage",
        "passages",
        "prompt",
        "quote",
        "quotes",
        "rag_provenance",
        "raw",
        "raw_response",
        "response",
        "source_papers",
        "source_refs",
        "source_text",
        "text",
    }
)


def _safe_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _safe_list(value: Any) -> List[Any]:
    return value if isinstance(value, list) else []


def payload_hash(value: Any) -> str:
    """A stable content fingerprint for a prompt, a raw reply or a payload.

    Two attempts that produced byte-identical bodies get the same hash, which is
    what makes "the retry changed nothing" visible without storing either body.
    Dicts and lists are hashed from canonical (sorted-key) JSON so key order in
    the producing code cannot change the fingerprint; anything unserializable
    falls back to ``str``.
    """

    if value is None:
        return ""
    if isinstance(value, (bytes, bytearray)):
        raw = bytes(value)
    elif isinstance(value, str):
        raw = value.encode("utf-8", "replace")
    else:
        try:
            raw = json.dumps(
                value, sort_keys=True, ensure_ascii=False, default=str
            ).encode("utf-8", "replace")
        except (TypeError, ValueError):  # pragma: no cover - default=str makes this rare
            raw = str(value).encode("utf-8", "replace")
    return "sha256:" + hashlib.sha256(raw).hexdigest()[:16]


def _preview(text: Any) -> str:
    """A bounded, single-line look at raw model text."""

    if not isinstance(text, str):
        text = "" if text is None else str(text)
    collapsed = " ".join(text.split())
    return clip(collapsed, PREVIEW_CHARS)


def descriptor(value: Any) -> Dict[str, Any]:
    """A value replaced by its size, its fingerprint and a look at its head.

    This is what a blob becomes. All three parts earn their place: ``chars``
    says how much was elided so nothing is hidden, ``hash`` lets two attempts be
    compared for equality without either body being stored, and ``preview``
    makes the *shape* recognisable ("it apologised", "it opened a code fence")
    without making the content readable.
    """

    if isinstance(value, str):
        return {"chars": len(value), "hash": payload_hash(value), "preview": _preview(value)}
    try:
        rendered = json.dumps(value, ensure_ascii=False, default=str)
    except (TypeError, ValueError):  # pragma: no cover - default=str makes this rare
        rendered = str(value)
    body: Dict[str, Any] = {
        "chars": len(rendered),
        "hash": payload_hash(value),
        "preview": _preview(rendered),
    }
    if isinstance(value, Mapping):
        body["keys"] = len(value)
    elif isinstance(value, (list, tuple, set)):
        body["items"] = len(value)
    return body


def bound(value: Any, *, depth: int = 0, key: str = "") -> Any:
    """THE recursive sanitizer. Every diagnostic value passes through this one.

    Before it existed, bounding was per-entry-point: attempt rows were clipped
    by one helper, ``record_boundary``'s ``**extra`` keys were stored verbatim,
    and ``record_cleaning`` / ``record_mapped_failure`` / ``record_iteration``
    each stored ``dict(report)`` untouched. So the bound held for exactly the
    shapes somebody had thought of, and a caller passing a nested report -- which
    is the normal case, since these ARE reports -- put an unbounded structure
    into a file rewritten on every attempt.

    Four rules, applied at every level:

    * a key in :data:`BULK_KEYS` becomes a :func:`descriptor`, at any depth;
    * a mapping keeps at most :data:`MAX_BOUND_KEYS` keys, a sequence at most
      :data:`MAX_BOUND_ITEMS` items, and both state what was dropped;
    * a string is clipped to ``failure_detail.MAX_VALUE_CHARS`` -- clipped, not
      described, so a human-readable reason stays readable;
    * anything past :data:`MAX_BOUND_DEPTH` becomes a descriptor.

    Never raises. A diagnostics sanitizer that can throw on an odd value takes
    out the diagnostics for the run it was meant to explain.
    """

    if str(key).casefold() in BULK_KEYS:
        return descriptor(value)
    if isinstance(value, bool) or value is None or isinstance(value, (int, float)):
        return value
    if isinstance(value, str):
        return clip(value)
    if depth >= MAX_BOUND_DEPTH:
        return descriptor(value)

    if isinstance(value, Mapping):
        out: Dict[str, Any] = {}
        for index, (name, item) in enumerate(value.items()):
            if index >= MAX_BOUND_KEYS:
                out["…"] = f"+{len(value) - MAX_BOUND_KEYS} more key(s) elided"
                break
            label = str(name)
            out[label] = bound(item, depth=depth + 1, key=label)
        return out

    if isinstance(value, (list, tuple, set)):
        items = list(value)
        kept: List[Any] = [bound(item, depth=depth + 1) for item in items[:MAX_BOUND_ITEMS]]
        if len(items) > MAX_BOUND_ITEMS:
            kept.append(f"…+{len(items) - MAX_BOUND_ITEMS} more item(s) elided")
        return kept

    return clip(value)


def bound_mapping(value: Any) -> Dict[str, Any]:
    """:func:`bound` for a value that must come back as a dict."""

    bounded = bound(_safe_dict(value))
    return bounded if isinstance(bounded, dict) else {}


def count_entities(payload: Any) -> Dict[str, int]:
    """Per-bucket entity row counts plus ``total``.

    Tolerant by construction: a payload whose ``entities`` is a list, a string or
    absent counts as zero rather than raising, because this runs on exactly the
    malformed payloads that caused the failure being diagnosed.
    """

    entities = _safe_dict(_safe_dict(payload).get("entities"))
    counts: Dict[str, int] = {}
    total = 0
    for bucket, rows in entities.items():
        if not isinstance(rows, list):
            continue
        counts[str(bucket)] = len(rows)
        total += len(rows)
    counts["total"] = total
    return counts


def count_processes(payload: Any) -> Dict[str, int]:
    """Per-bucket process row counts plus ``total``. Same tolerance as above."""

    processes = _safe_dict(_safe_dict(payload).get("processes"))
    counts: Dict[str, int] = {}
    total = 0
    for bucket, rows in processes.items():
        if not isinstance(rows, list):
            continue
        counts[str(bucket)] = len(rows)
        total += len(rows)
    counts["total"] = total
    return counts


class DiscardLedger:
    """Every row a cleaning pass dropped, with the rule that dropped it.

    A dropped row is invisible by construction -- ``_clean_processes`` drops rows
    with a bare ``continue`` -- so this is the only place the fact survives. It
    keeps a full count per reason (cheap, and the count is the actual finding)
    and a :data:`MAX_SAMPLE`-bounded sample of names/pointers per reason (enough
    to recognise the shape; never the row itself, which carries evidence text).
    """

    def __init__(self) -> None:
        self._counts: Dict[str, int] = {}
        self._samples: Dict[str, List[Dict[str, str]]] = {}
        self._total = 0

    def record(self, *, reason: str, pointer: str = "", name: Any = "") -> None:
        key = str(reason or "unspecified")
        self._counts[key] = self._counts.get(key, 0) + 1
        self._total += 1
        bucket = self._samples.setdefault(key, [])
        if len(bucket) >= MAX_SAMPLE:
            return
        entry: Dict[str, str] = {}
        if pointer:
            entry["pointer"] = clip(str(pointer), _SAMPLE_CHARS)
        label = name if isinstance(name, str) else ("" if name is None else str(name))
        if label.strip():
            entry["name"] = clip(label.strip(), _SAMPLE_CHARS)
        if entry:
            bucket.append(entry)

    @property
    def total(self) -> int:
        return self._total

    def counts_by_reason(self) -> Dict[str, int]:
        return dict(sorted(self._counts.items()))

    def samples(self) -> Dict[str, List[Dict[str, str]]]:
        return {reason: list(rows) for reason, rows in sorted(self._samples.items()) if rows}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "discarded_total": self._total,
            "discarded_by_reason": self.counts_by_reason(),
            "discarded_sample": self.samples(),
            "sample_cap_per_reason": MAX_SAMPLE,
        }


def classify_outcome(
    *,
    empty_completion: bool = False,
    parsed: bool = True,
    raw_process_count: Optional[int] = None,
    cleaned_process_count: Optional[int] = None,
    contract_ok: Optional[bool] = None,
) -> str:
    """Reduce what is known about one boundary to exactly one of :data:`OUTCOMES`.

    Order matters and is not arbitrary -- it walks the pipeline forwards, so the
    *earliest* thing that went wrong is the one reported. A reply that was empty
    also has zero processes and also fails the contract; calling that
    ``contract_failed_after_cleaning`` would name the symptom and hide the cause,
    which is the exact confusion this module exists to end.

    ``None`` means "not observed yet" and never decides anything: a boundary
    recorded before cleaning has run reports on what it can see and is updated in
    place when cleaning finishes.
    """

    if empty_completion:
        return OUTCOME_EMPTY_COMPLETION
    if not parsed:
        return OUTCOME_INVALID_JSON
    if raw_process_count is not None and raw_process_count <= 0:
        return OUTCOME_ZERO_PROCESSES
    if (
        raw_process_count is not None
        and raw_process_count > 0
        and cleaned_process_count is not None
        and cleaned_process_count <= 0
    ):
        return OUTCOME_DISCARDED_BY_CLEANING
    if contract_ok is False:
        return OUTCOME_CONTRACT_FAILED
    return OUTCOME_OK


class ExtractionDiagnostics:
    """The per-run collector. Thread-safe, bounded, and eager about writing.

    Every ``record_*`` method returns the stored record so a caller can attach it
    to an exception or a session-state key, and flushes to ``artifact_dir`` when
    one is set. A run therefore has its artifacts on disk *before* the stage that
    produced them returns or raises.
    """

    def __init__(
        self,
        *,
        run_id: str = "",
        artifact_dir: Optional[Path | str] = None,
    ) -> None:
        self.run_id = str(run_id or "")
        self.artifact_dir: Optional[Path] = Path(artifact_dir) if artifact_dir else None
        self._lock = threading.RLock()
        self._boundaries: List[Dict[str, Any]] = []
        self._stage0: List[Dict[str, Any]] = []
        self._cleaning: List[Dict[str, Any]] = []
        self._mapped_failure: Dict[str, Any] = {}
        self._iterations: Dict[str, List[Dict[str, Any]]] = {"audit": [], "gap": []}
        self._write_errors: List[str] = []

    # -- recording ---------------------------------------------------------
    def record_boundary(
        self,
        *,
        stage: str,
        boundary: str,
        model: str = "",
        finish_reason: str = "",
        attempts: int = 0,
        response_status: str = "",
        outcome: str = OUTCOME_OK,
        raw_entity_counts: Optional[Mapping[str, int]] = None,
        raw_process_counts: Optional[Mapping[str, int]] = None,
        request_hash: str = "",
        response_hash: str = "",
        raw_chars: Optional[int] = None,
        raw_preview: str = "",
        error: str = "",
        terminal_reason: str = "",
        attempt_log: Optional[Sequence[Mapping[str, Any]]] = None,
        **extra: Any,
    ) -> Dict[str, Any]:
        """Record one crossing of an LLM extraction boundary.

        ``raw_*_counts`` are omitted rather than zeroed when the reply could not
        be parsed: "we could not count" and "we counted zero" are two different
        findings and collapsing them re-creates the ambiguity being fixed.
        """

        record: Dict[str, Any] = {
            "stage": str(stage or ""),
            "boundary": str(boundary or ""),
            "outcome": str(outcome or OUTCOME_OK),
            "model": str(model or ""),
            "finish_reason": str(finish_reason or ""),
            "attempts": int(attempts or 0),
            "response_status": str(response_status or ""),
        }
        if terminal_reason:
            record["terminal_reason"] = str(terminal_reason)
        if raw_entity_counts is not None:
            record["raw_entity_counts"] = dict(raw_entity_counts)
        if raw_process_counts is not None:
            record["raw_process_counts"] = dict(raw_process_counts)
        if request_hash:
            record["request_hash"] = str(request_hash)
        if response_hash:
            record["response_hash"] = str(response_hash)
        if raw_chars is not None:
            record["raw_chars"] = int(raw_chars)
        if raw_preview:
            record["raw_preview"] = _preview(raw_preview)
        if error:
            record["error"] = clip(str(error))
        if attempt_log:
            record["attempt_log"] = [
                _bounded_attempt(item) for item in list(attempt_log)[:MAX_ATTEMPTS_KEPT]
            ]
        # ``**extra`` is the open door: callers pass repair samples, rejected-row
        # lists, failing codes -- arbitrary nested structures this module never
        # sees the shape of. Every one goes through the sanitizer.
        for key, value in extra.items():
            if value is None:
                continue
            label = str(key)
            record[label] = bound(value, depth=1, key=label)

        with self._lock:
            self._boundaries.append(record)
        self._flush()
        return record

    def attach_cleaning_to_boundary(
        self,
        record: Mapping[str, Any],
        *,
        cleaned_entity_counts: Optional[Mapping[str, int]] = None,
        cleaned_process_counts: Optional[Mapping[str, int]] = None,
        ledger: Optional[DiscardLedger] = None,
        contract_ok: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """Fold post-cleaning facts into a boundary record already stored.

        Cleaning happens after the boundary returns, so the record is written
        first (it has to be -- the stage can raise in between) and completed
        here. The outcome is re-derived rather than overwritten so the earliest
        cause still wins; see :func:`classify_outcome`.
        """

        with self._lock:
            target: Optional[Dict[str, Any]] = None
            for stored in self._boundaries:
                if stored is record:
                    target = stored
                    break
            if target is None:
                target = dict(record)
                self._boundaries.append(target)

            if cleaned_entity_counts is not None:
                target["cleaned_entity_counts"] = dict(cleaned_entity_counts)
            if cleaned_process_counts is not None:
                target["cleaned_process_counts"] = dict(cleaned_process_counts)
            if ledger is not None:
                target.update(ledger.to_dict())
            if contract_ok is not None:
                target["contract_ok"] = bool(contract_ok)

            raw_total = _safe_dict(target.get("raw_process_counts")).get("total")
            cleaned_total = _safe_dict(target.get("cleaned_process_counts")).get("total")
            target["outcome"] = classify_outcome(
                empty_completion=target.get("outcome") == OUTCOME_EMPTY_COMPLETION,
                parsed=target.get("outcome") != OUTCOME_INVALID_JSON,
                raw_process_count=raw_total if isinstance(raw_total, int) else None,
                cleaned_process_count=cleaned_total if isinstance(cleaned_total, int) else None,
                contract_ok=target.get("contract_ok"),
            )
            result = target
        self._flush()
        return result

    def record_stage0_attempt(self, attempt: Mapping[str, Any]) -> Dict[str, Any]:
        """Record one Stage-0 draw (the first, and the bounded-head retry)."""

        record = _bounded_attempt(attempt)
        with self._lock:
            self._stage0.append(record)
        self._flush()
        return record

    def record_cleaning(self, report: Mapping[str, Any]) -> Dict[str, Any]:
        """Record one cleaning pass's before/after counts and discard ledger."""

        record = bound_mapping(report)
        with self._lock:
            self._cleaning.append(record)
        self._flush()
        return record

    def record_mapped_failure(self, snapshot: Mapping[str, Any]) -> Dict[str, Any]:
        """Record what the payload looked like when mapping produced a failure.

        Written only once mapping has actually run: absence of this artifact is
        itself the signal that the run died before the mapper was called.
        """

        record = bound_mapping(snapshot)
        with self._lock:
            self._mapped_failure = record
        self._flush()
        return record

    def record_iteration(self, kind: str, summary: Mapping[str, Any]) -> Dict[str, Any]:
        """Record one audit or gap-resolution round summary."""

        key = "gap" if str(kind).strip().lower().startswith("gap") else "audit"
        record = bound_mapping(summary)
        with self._lock:
            self._iterations.setdefault(key, []).append(record)
        self._flush()
        return record

    # -- reading -----------------------------------------------------------
    @property
    def boundaries(self) -> List[Dict[str, Any]]:
        with self._lock:
            return [dict(item) for item in self._boundaries]

    @property
    def stage0_attempts(self) -> List[Dict[str, Any]]:
        with self._lock:
            return [dict(item) for item in self._stage0]

    def outcomes(self) -> List[str]:
        """Every distinct outcome recorded, in the order first seen."""

        seen: List[str] = []
        for record in self.boundaries:
            outcome = str(record.get("outcome") or "")
            if outcome and outcome not in seen:
                seen.append(outcome)
        return seen

    def has_content(self) -> bool:
        with self._lock:
            return bool(
                self._boundaries
                or self._stage0
                or self._cleaning
                or self._mapped_failure
                or any(self._iterations.values())
            )

    def artifacts(self) -> Dict[str, Any]:
        """The artifact name -> object map, omitting artifacts with no content.

        Omission is meaningful and deliberate: no ``mapped_failure_snapshot.json``
        means mapping never ran, which is a different diagnosis from mapping
        running and producing nothing.
        """

        with self._lock:
            out: Dict[str, Any] = {}
            if self._stage0:
                out[STAGE0_ATTEMPTS_NAME] = {
                    "run_id": self.run_id,
                    "attempt_count": len(self._stage0),
                    "attempts": self._stage0[:MAX_ATTEMPTS_KEPT],
                }
            if self._boundaries:
                out[BOUNDARY_REPORT_NAME] = {
                    "run_id": self.run_id,
                    "boundary_count": len(self._boundaries),
                    "outcomes": self.outcomes(),
                    "boundaries": self._boundaries,
                }
            if self._cleaning:
                out[CLEANING_REPORT_NAME] = {
                    "run_id": self.run_id,
                    "pass_count": len(self._cleaning),
                    "passes": self._cleaning,
                }
            if self._mapped_failure:
                out[MAPPED_FAILURE_SNAPSHOT_NAME] = dict(self._mapped_failure)
            if self._iterations.get("audit"):
                out[AUDIT_ITERATIONS_NAME] = {
                    "run_id": self.run_id,
                    "round_count": len(self._iterations["audit"]),
                    "rounds": self._iterations["audit"],
                }
            if self._iterations.get("gap"):
                out[GAP_ITERATIONS_NAME] = {
                    "run_id": self.run_id,
                    "round_count": len(self._iterations["gap"]),
                    "rounds": self._iterations["gap"],
                }
            return out

    # -- writing -----------------------------------------------------------
    def write(self, artifact_dir: Optional[Path | str] = None) -> List[str]:
        """Write every artifact with content into ``artifact_dir``.

        Never raises. A diagnostics writer that can kill the run it is diagnosing
        is worse than no diagnostics: a failed write is recorded in
        ``write_errors`` and surfaced with the artifacts instead.
        """

        target = Path(artifact_dir) if artifact_dir else self.artifact_dir
        if target is None:
            return []
        written: List[str] = []
        try:
            target.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            self._note_write_error(f"{target}: {type(exc).__name__}: {exc}")
            return []
        for name, blob in self.artifacts().items():
            path = target / name
            try:
                path.write_text(
                    json.dumps(blob, indent=2, ensure_ascii=False, default=str),
                    encoding="utf-8",
                )
            except (OSError, TypeError, ValueError) as exc:
                self._note_write_error(f"{name}: {type(exc).__name__}: {exc}")
                continue
            written.append(name)
        return written

    @property
    def write_errors(self) -> List[str]:
        with self._lock:
            return list(self._write_errors)

    def _note_write_error(self, message: str) -> None:
        with self._lock:
            if message not in self._write_errors:
                self._write_errors.append(clip(message))
        logger.warning("extraction diagnostics could not be written: %s", message)

    def _flush(self) -> None:
        if self.artifact_dir is None:
            return
        self.write(self.artifact_dir)


def _bounded_attempt(attempt: Mapping[str, Any]) -> Dict[str, Any]:
    """One attempt row, sanitized.

    Historically this had its own hand-rolled bounding rules, which is exactly
    the per-entry-point drift :func:`bound` replaced. The one thing it still does
    itself is flatten a bulk field into ``<name>_chars`` / ``<name>_hash`` /
    ``<name>_preview`` sibling keys rather than a nested descriptor, because
    ``stage0_attempts.json`` readers and tests address those flat names.
    """

    source = _safe_dict(attempt)
    out: Dict[str, Any] = {}
    for key, value in source.items():
        name = str(key)
        if name.casefold() in BULK_KEYS:
            parts = descriptor(value)
            out[f"{name}_chars"] = parts["chars"]
            out[f"{name}_hash"] = parts["hash"]
            out[f"{name}_preview"] = parts["preview"]
            continue
        out[name] = bound(value, depth=1, key=name)
    return out


# ---------------------------------------------------------------------------
# The ambient recorder.
#
# EXECUTION-LOCAL, NOT PROCESS-GLOBAL. A module-level variable was the first
# implementation and it is wrong for the only deployment that matters: Streamlit
# runs each browser session's script on its own ScriptRunner thread, in one
# process. Two sessions extracting two papers at the same time would have shared
# a single recorder -- session B's ``activate`` discarding session A's
# boundaries mid-run, and both writing the same files -- so the artifacts would
# have been a blend of two papers with nothing saying so.
#
# A ContextVar is read and written per execution context. A new thread starts
# from the default rather than inheriting, which is exactly the isolation
# wanted here: nothing a worker thread records can land in another run's
# artifacts. The cost is that a helper which offloads work to its own thread
# records into a fresh detached recorder instead of the run's -- acceptable,
# because every site that records (Stage 0, Stage 1, the repair passes, the
# audit/gap loop) runs on the script thread, and the alternative failure is
# silent cross-run contamination.
# ---------------------------------------------------------------------------
_ACTIVE: ContextVar[Optional[ExtractionDiagnostics]] = ContextVar(
    "t2pw_extraction_diagnostics", default=None
)


def current() -> ExtractionDiagnostics:
    """The recorder for the run in progress, creating a detached one if needed.

    Never returns ``None``, so every call site records unconditionally. A
    detached recorder (no ``artifact_dir``) collects in memory and writes
    nothing, which is exactly what a unit test or a library caller wants.
    """

    recorder = _ACTIVE.get()
    if recorder is None:
        recorder = ExtractionDiagnostics()
        _ACTIVE.set(recorder)
    return recorder


def activate(
    *,
    run_id: str = "",
    artifact_dir: Optional[Path | str] = None,
    unique: bool = False,
) -> ExtractionDiagnostics:
    """Start a fresh recorder for a run and make it :func:`current`.

    Called once at the top of a pipeline run. Replacing rather than clearing
    means a stale recorder can never leak diagnostics from the previous paper
    into this one's artifacts, and the ContextVar means it cannot leak into a
    concurrent one's either.

    ``unique=True`` puts this invocation's artifacts in their own subdirectory of
    ``artifact_dir``. Production passes it: two Streamlit sessions given the same
    base directory would otherwise write ``extraction_boundary_report.json`` over
    each other, and the loser would be diagnosed with the winner's evidence --
    worse than no artifact, because it looks authoritative. It defaults to
    ``False`` so a caller that wants a known, fixed path (a test, a replay) still
    gets exactly the directory it named.
    """

    target = Path(artifact_dir) if artifact_dir else None
    if target is not None and unique:
        target = target / f"run-{uuid4().hex[:12]}"
    recorder = ExtractionDiagnostics(run_id=run_id, artifact_dir=target)
    _ACTIVE.set(recorder)
    return recorder


def deactivate() -> None:
    """Drop the current recorder. The next :func:`current` builds a fresh one."""

    _ACTIVE.set(None)
