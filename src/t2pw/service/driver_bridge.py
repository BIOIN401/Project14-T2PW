"""Driver bridge (BR): the web worker's run function until A7 swaps in R5's ``run_pathway``.

:func:`run_pathway` has the W0 signature ``(PathwayRequest) -> PathwayResult`` and runs
the job through ``t2pw.batch.driver.run_one``, the AppTest path that produced every
benchmark number. Nothing here re-implements the pipeline: it turns the request into
the one "paper" the driver takes, calls the driver once, writes the driver's in-memory
artifacts into ``request.out_dir``, and maps the ``RunOutcome`` onto ``PathwayResult``.

Input rules (each refusal is a result with an explicit kind, never a dropped input):

* Several articles are combined exactly as the Streamlit app combines several text
  entries and PDFs: every stripped, non-empty part, pasted texts first, joined with
  ``"\\n\\n"`` (``streamlit_app.py``: ``text = "\\n\\n".join(text_parts)``). Only
  ``text`` is used after that line, so pasting the joined string into the driver's one
  text box gives the app the same input as N boxes would. Order: texts, PMIDs, PDFs.
* PDFs are read with the app's own defaults (references skipped, OCR fallback OFF,
  all pages) through W1's ``read_pdfs``. PMIDs go through W1's ``resolve_pmids``.
* Name mode (a pathway name and no sources) runs ONE fetched paper: a batch topic line
  ``name | organism | 1`` (W1 ``resolve_name``, ``count=1``), then the app's own RAG.
* The driver derives ``pinned`` and the extraction focus from the pathway name, so a
  request it would silently re-read (an organism with no pathway name, a pinned request
  that names a pathway, a custom ``extraction_focus``, an unknown ``export_mode``) is
  refused with ``failure_kind="contract"``, which W4 never retries.
* Line endings: the text handed to the driver is what a batch leg reads back from its
  ``01_source_text.txt`` (written in text mode, read with universal newlines), so the
  driver's input is byte-identical to a benchmark leg's.

Isolation: the result is built only from the ``RunOutcome`` in memory (artifacts are
bytes/str, the PWML is the exporter's ``xml_bytes``) and written under ``out_dir``; this
module never reads ``outputs/`` or ``tmp/``. But the app writes the fixed
``PROJECT_ROOT/outputs/`` and ``tmp/`` and clears ``outputs/`` at the start of a run, and
the driver this module calls reads the quarantine report files back from there
(``batch/driver.py`` ``_add_identity_artifacts``). So the real requirement is ONE PIPELINE
RUN PER CHECKOUT at a time. Nothing enforces it: the worker's lock is per database, not
per checkout, so it holds only while one worker and no batch run or Streamlit session run
the pipeline in the same checkout (a W9 runbook item).

Import-light: ``t2pw.batch.*`` and ``t2pw.service.inputs`` are imported inside the
functions, so importing this module loads no streamlit, no app and no driver.
"""

from __future__ import annotations

import dataclasses
import functools
import json
import os
import tempfile
import time
from dataclasses import fields
from pathlib import Path
from typing import Any, List, Optional, Tuple

from t2pw.pipeline.release_status import ReleaseStatus
from t2pw.service.pathway import (
    KIND_CONTRACT,
    KIND_CRASH,
    KIND_NETWORK,
    KIND_NO_REACTIONS,
    KIND_TIMEOUT,
    MODE_RESEARCH,
    MODE_STRICT,
    STAGE_INIT,
    STATUS_ERROR,
    STATUS_FAIL,
    STATUS_TIMEOUT,
    PathwayRequest,
    PathwayResult,
)

#: The app's joiner for several text parts (``streamlit_app.py``, ``text = ...join``).
APP_JOIN = "\n\n"
#: Papers a name-mode job runs: one batch leg (PO ruling on Q-BR-2, 2026-11-05).
NAME_MODE_COUNT = 1
#: The driver's own manifest row (``RunOutcome.to_dict()``, ``detail`` included),
#: written beside the artifacts. Not a ``report_paths`` key: those are public downloads.
OUTCOME_NAME = "run_outcome.json"
#: Room between the driver's own deadline and the worker's kill, so the driver can
#: classify its timeout and the artifacts get written. Same value as ``batch.runner``.
GRACE_S = 120.0

CODE_UNSUPPORTED = "bridge_unsupported_request"
CODE_NO_INPUT = "bridge_input_unavailable"
CODE_DEADLINE = "bridge_deadline_passed"
WARN_RELEASE_UNREADABLE = (
    f"the run's release record could not be read into the result; it is in {OUTCOME_NAME}"
)

_Problem = Tuple[str, str, str, str]  # status, failure_kind, message, issue code


def _refuse(status: str, kind: str, message: str, code: str) -> PathwayResult:
    return PathwayResult(status=status, stage=STAGE_INIT, failure_kind=kind,
                         message=message, issue_codes=[code])


def _contract(message: str) -> _Problem:
    return STATUS_ERROR, KIND_CONTRACT, message, CODE_UNSUPPORTED


def _unsupported(req: PathwayRequest) -> Optional[_Problem]:
    """A request field the driver cannot honour as given, or None."""

    sc = req.scope
    if req.export_mode not in (MODE_STRICT, MODE_RESEARCH):
        return _contract(f"export mode must be '{MODE_STRICT}' or '{MODE_RESEARCH}', "
                         f"not {req.export_mode!r}")
    if sc.requested_organism and not sc.requested_pathway:
        return _contract("the organism can only be used together with a pathway name; "
                         "add the pathway name and submit again")
    if sc.pinned and sc.requested_pathway:
        return _contract("a pinned request cannot also name a pathway: the driver treats "
                         "every named pathway as a topic request")
    if not (sc.requested_pathway or req.texts or req.pmids or req.pdf_paths):
        return _contract("nothing to run: give a pathway name, a text, PubMed IDs or PDFs")
    if req.extraction_focus:
        from t2pw.batch import driver

        derived = driver._extraction_focus(sc.to_dict())  # noqa: SLF001 - the driver's own rule
        if req.extraction_focus != derived:
            return _contract("a custom extraction focus is not supported yet; the focus is "
                             "set from the pathway name and organism")
    return None


def _worst(problems: List[_Problem]) -> Optional[_Problem]:
    """One refusal for several bad inputs: a permanent kind wins over a retryable one.

    The message keeps every input's own message, highest-ranked first.
    """

    if not problems:
        return None
    rank = (KIND_CONTRACT, KIND_NO_REACTIONS, KIND_NETWORK)
    ranked = sorted(problems, key=lambda p: rank.index(p[1]))  # stable: input order kept
    status, kind, _, code = ranked[0]
    return status, kind, "; ".join(p[2] for p in ranked), code


def _as_batch_reads(text: str) -> str:
    """``text`` as a batch leg reads it back from ``01_source_text.txt``.

    ``runner._write_text`` writes in text mode (``\\n`` -> ``os.linesep``) and
    ``runner._read_text_or_error`` reads with universal newlines (``\\r\\n`` and ``\\r``
    -> ``\\n``). Same round trip, in memory, on the same platform.
    """

    return text.replace("\n", os.linesep).replace("\r\n", "\n").replace("\r", "\n")


def _paper(req: PathwayRequest) -> Tuple[Any, Optional[_Problem]]:
    """The driver's input: a ``BatchPaper`` (name mode) or a plain dict."""

    from t2pw.service import inputs

    sc = req.scope
    if not (req.texts or req.pmids or req.pdf_paths):
        try:
            got = inputs.resolve_name(sc.requested_pathway, sc.requested_organism,
                                      count=NAME_MODE_COUNT)
        except ValueError as exc:
            return None, _contract(str(exc))
        if got.papers:
            paper = got.papers[0]
            return dataclasses.replace(paper, full_text=_as_batch_reads(paper.full_text)), None
        outage = any(isinstance(r, dict) and r.get("reason") in inputs._SEARCH_ERRORS  # noqa: SLF001
                     for r in got.skipped)
        if outage:
            return None, (STATUS_ERROR, KIND_NETWORK, got.message, CODE_NO_INPUT)
        return None, (STATUS_FAIL, KIND_NO_REACTIONS, got.message, CODE_NO_INPUT)

    parts = [t.strip() for t in req.texts if t.strip()]  # the app skips empty boxes too
    ids = [f"text{i + 1}" for i in range(len(parts))]
    problems: List[_Problem] = []
    if req.pmids:
        from t2pw.rag.acquire import CandidatePaper, fetch_full_text

        # use_cache=False: fetch_full_text caches an empty (failed) download (W1 Q3).
        uncached = lambda pmcid: fetch_full_text(  # noqa: E731
            CandidatePaper(id=pmcid, source="PMC"), use_cache=False)
        for r in inputs.resolve_pmids(req.pmids, fetch_text_fn=uncached):
            if r.ok:
                parts.append(r.text.strip())  # the app strips every part
                ids.append(f"PMID{r.pmid}")
            elif r.status == inputs.LOOKUP_FAILED:
                problems.append((STATUS_ERROR, KIND_NETWORK, r.message, CODE_NO_INPUT))
            elif r.status == inputs.INVALID:
                problems.append(_contract(r.message))
            else:  # NO_PMCID / NO_FULL_TEXT: the driver's "no full text" verdict
                problems.append((STATUS_FAIL, KIND_NO_REACTIONS, r.message, CODE_NO_INPUT))
    if req.pdf_paths:
        app_defaults = functools.partial(inputs.parse_pdf, enable_ocr_fallback=False)
        for r in inputs.read_pdfs(req.pdf_paths, parse_fn=app_defaults):
            if r.ok:
                parts.append(r.text.strip())  # the app strips every part
                ids.append(Path(r.path).name)
            elif r.status == inputs.EMPTY:
                problems.append((STATUS_FAIL, KIND_NO_REACTIONS, r.message, CODE_NO_INPUT))
            else:
                problems.append(_contract(r.message))
    worst = _worst(problems)
    if worst:
        return None, worst
    if not parts:
        return None, (STATUS_FAIL, KIND_NO_REACTIONS, "the texts given are empty", CODE_NO_INPUT)
    return {
        "paper_id": "web:" + "+".join(ids),
        "full_text": _as_batch_reads(APP_JOIN.join(parts)),
        "requested_pathway": sc.requested_pathway,
        "requested_organism": sc.requested_organism,
    }, None


def _release_status(value: Any) -> Tuple[Optional[ReleaseStatus], str]:
    """``RunOutcome.release_status`` (a ``ReleaseStatus`` or its frozen dict) as the type.

    Keys ``ReleaseStatus`` has no field for (the boundary adds ``applied``,
    ``review_reasons``, ``retrieval_attempts_source``) stay in ``attempt_log`` and
    :data:`OUTCOME_NAME`. Never invented: an unreadable record is ``None`` plus a warning.
    """

    if value is None or isinstance(value, ReleaseStatus):
        return value, ""
    try:
        d = dict(value)
        kw = {f.name: d[f.name] for f in fields(ReleaseStatus) if f.name in d}
        for name in ("semantic_failed_checks", "missing_anchors", "reasons"):
            if name in kw:
                kw[name] = tuple(kw[name])
        if "semantic_check_evaluability" in kw:
            kw["semantic_check_evaluability"] = tuple(
                (e["check"], e["applicable"], e["inapplicable_reason"])
                for e in kw["semantic_check_evaluability"])
        return ReleaseStatus(**kw), ""
    except Exception:  # noqa: BLE001 - a malformed record must not lose the run
        return None, WARN_RELEASE_UNREADABLE


def _to_result(outcome: Any, out_dir: Path) -> PathwayResult:
    from t2pw.batch import driver, runner

    artifacts = dict(outcome.artifacts or {})
    written = runner.write_artifacts(out_dir, artifacts)  # the batch's own writer
    failed = {e["name"] for e in runner.failed_writes(written)}
    paths = {e["name"]: out_dir / e["name"] for e in written if e["name"] not in failed}
    pwml_names = (driver.PWML_RELEASE_READY_NAME, driver.PWML_REVIEW_REQUIRED_NAME)
    pwml = {n: paths.pop(n) for n in pwml_names if n in paths}

    res = PathwayResult(
        status=outcome.status, stage=outcome.stage, failure_kind=outcome.failure_kind,
        message=outcome.message, warnings=list(outcome.warnings),
        issue_codes=list(outcome.issue_codes), counts=dict(outcome.counts),
        pwml_path=pwml.get(outcome.pwml_artifact), report_paths=paths)
    res.release_status, warn = _release_status(outcome.release_status)
    if warn:
        res.warnings.append(warn)

    lost = [n for n in runner.required_artifacts(outcome.mode, artifacts) if n in failed]
    if lost:  # same rule and wording as batch.runner.run_single
        res.status, res.failure_kind = STATUS_FAIL, KIND_CRASH
        res.message = (f"the pipeline succeeded but the required artifact(s) {', '.join(lost)} "
                       "could not be written, so this run produced nothing on disk")
    res.warnings += [f"report {n} could not be written" for n in sorted(failed - set(lost))]

    row = outcome.to_dict()
    row["bridge_files"] = written
    try:  # a disk fault here must not turn a final verdict into a retried crash
        (out_dir / OUTCOME_NAME).write_text(
            json.dumps(row, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    except OSError:
        res.warnings.append(f"the run record {OUTCOME_NAME} could not be written")
    entry = {k: v for k, v in row.items() if k not in ("detail", "bridge_files")}
    res.attempt_log = [{"attempt": 1, "runner": "t2pw.batch.driver.run_one", **entry}]
    return res


def run_pathway(request: PathwayRequest) -> PathwayResult:
    """Run one job through ``batch.driver.run_one``. W0 signature; never edits the request."""

    out_dir = Path(request.out_dir) if request.out_dir else Path(tempfile.mkdtemp(prefix="t2pw-job-"))
    out_dir.mkdir(parents=True, exist_ok=True)
    problem = _unsupported(request)
    paper = None
    if problem is None:
        paper, problem = _paper(request)
    if problem is not None:
        return _refuse(*problem)

    kwargs = {}
    if request.deadline is not None:
        from t2pw.pipeline.deadline import child_deadline_seconds

        left = float(request.deadline) - time.time()  # after any fetch above
        if left <= 0:
            return _refuse(STATUS_TIMEOUT, KIND_TIMEOUT,
                           "the run's deadline passed before the pipeline started", CODE_DEADLINE)
        kwargs["timeout"] = child_deadline_seconds(left, grace=GRACE_S)

    from t2pw.batch import driver, runner

    runner.force_utf8_stdio()  # the worker child's stdout is a file (cp1252 on Windows)
    return _to_result(driver.run_one(paper, request.export_mode, **kwargs), out_dir)
