"""The frozen ``run_pathway`` interface for the web server (W0).

*** ``run_pathway`` HERE IS A STUB. *** It runs no pipeline, makes no network or
LLM call, writes nothing, and always returns the same "not implemented" error
result. R5 replaces the body; the two dataclasses and the vocabulary below are
the contract every W-task builds against and must not change shape without a
new W0-style task.

Must stay cheap to import: never import ``streamlit`` or
``t2pw.app.streamlit_app`` (the app runs its UI on import), and never import
``t2pw.batch.driver`` (R6 makes the driver call this module, so that import
would become a cycle).

The vocabulary strings MIRROR ``t2pw.batch.driver`` (``MODE_*``, ``STAGE_*``,
``_STATUS_*``, ``KIND_*``) rather than importing them, for the reason above;
``tests/test_service_pathway_interface.py`` asserts the two sets are equal.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from t2pw.pipeline.release_status import ReleaseStatus
from t2pw.rag.eligibility import OUTCOME_SCOPE_CONFLICT, RequestedScope

# ── Mirrors of t2pw.batch.driver (equality is tested) ──────────────────────
MODE_STRICT = "strict"
MODE_RESEARCH = "research"

STAGE_INIT = "init"
STAGE_APP_LOAD = "app_load"
STAGE_INPUT = "input"
STAGE_STAGE1 = "stage1"
STAGE_POST_PIPELINE = "post_pipeline"
STAGE_PWML_EXPORT = "pwml_export"
STAGE_RESEARCH_REPORT = "research_report"

STATUS_PASS = "pass"
STATUS_FAIL = "fail"
STATUS_TIMEOUT = "timeout"
STATUS_ERROR = "error"
STATUS_SCOPE_CONFLICT = OUTCOME_SCOPE_CONFLICT

KIND_CONTRACT = "contract"
KIND_LLM = "llm"
KIND_NETWORK = "network"
KIND_AMBIGUOUS = "ambiguous_review_scope"
KIND_NO_REACTIONS = "no_reactions"
KIND_CRASH = "crash"
KIND_TIMEOUT = "timeout"
KIND_UNKNOWN = "unknown"


@dataclass(frozen=True)
class PathwayRequest:
    """One pathway to build. Frozen: the run may not rewrite its request.

    ``scope.requested_pathway`` is the pathway name (name mode) and
    ``scope.requested_organism`` the drop-down organism. Inputs are any mix of
    pasted ``texts``, ``pdf_paths`` and ``pmids``. ``out_dir=None`` lets the
    implementation choose (today the app writes to ``PROJECT_ROOT/outputs``).
    ``deadline`` is an absolute ``time.time()`` epoch second, ``None`` = no cap.
    ``extraction_focus`` is the user's own focus / task-context text (the app's
    "Optional extraction focus" box); empty means R5 derives it from ``scope``,
    as the batch driver does today.
    Consent, email and job ids are job-store fields, not pipeline inputs.
    """

    texts: Tuple[str, ...] = ()
    pdf_paths: Tuple[Path, ...] = ()
    pmids: Tuple[str, ...] = ()
    scope: RequestedScope = field(default_factory=RequestedScope)
    extraction_focus: str = ""
    export_mode: str = MODE_STRICT
    out_dir: Optional[Path] = None
    deadline: Optional[float] = None


@dataclass
class PathwayResult:
    """What one ``run_pathway`` call produced.

    Defaults mirror ``RunOutcome``: an unfilled result reads as an error at
    ``init``, never as a pass. ``release_status`` is the run's own release
    record (release_ready / review_required / diagnostic_only); a consumer
    must read it, never infer it from ``pwml_path``'s filename.

    ``warnings``, ``issue_codes`` and ``counts`` mirror ``RunOutcome``. A run
    can pass without producing one of its deliverables; that is a warning,
    and a result with warnings must never be shown as clean.

    ``report_paths`` maps a filename to the file written under ``out_dir``.
    Its keys are exactly the filenames ``t2pw.batch.driver`` puts in
    ``RunOutcome.artifacts`` (its diagnostic artifacts included), except the
    PWML itself (``pathway.pwml`` / ``pathway.review_required.pwml``), which
    is ``pwml_path``. Only files actually written are present; which ones
    depends on the mode and on how far the run got. The driver is the source
    of truth; today's names, by kind:

    * payloads and gates: ``stage1_payload.json``, ``merged_payload.json``,
      ``gate_fail_report.json``, ``initial_stage3_gate_report.json``,
      ``final_stage3_gate_report.json``, ``contract_reports.json``
    * identity: ``final_mapped.json``, ``rag_admission_report.json``
    * quarantine: ``quarantine_report.json``, ``removed_entity_report.json``,
      ``graph_closure_iterations.json``, ``coverage_summary.json``,
      ``coverage_diagnostics.json``
    * scope: ``scope_conflict.json``, ``scope_specialization.json``
    * diagnostics: ``stage0_attempts.json``,
      ``extraction_boundary_report.json``, ``cleaning_report.json``,
      ``mapped_failure_snapshot.json``, ``audit_iteration_summary.json``,
      ``gap_iteration_summary.json``
    * PWML export (strict mode): ``pwml_ir.json``, ``pwml_ir_report.json``,
      ``pwml_ir_validation.json``, ``pwml_validation_report.json``,
      ``pwml_qa.json``, ``pwml_required_field_gate_report.json``
    * research deliverables (research mode): ``review_flags.json``,
      ``research_pathway_report.txt``, ``research_pathway_citations.json``,
      ``research_pathway_elements.csv``

    The citation report already exists: ``research_pathway_citations.json``
    (with its ``.txt`` and ``.csv`` views), written in research mode only and
    only when RAG synthesis ran.

    ``attempt_log`` holds in-run attempts only: LLM / provider / stage retries
    inside ONE ``run_pathway`` call. Job-level retries (W4) are recorded in the
    W3 job store, not here. W4 decides a retry from ``failure_kind`` and
    ``pwml_path is None``.
    """

    status: str = STATUS_ERROR
    stage: str = STAGE_INIT
    failure_kind: str = ""
    message: str = ""
    warnings: List[str] = field(default_factory=list)
    issue_codes: List[str] = field(default_factory=list)
    counts: Dict[str, Any] = field(default_factory=dict)
    release_status: Optional[ReleaseStatus] = None
    pwml_path: Optional[Path] = None
    image_path: Optional[Path] = None
    report_paths: Dict[str, Path] = field(default_factory=dict)
    attempt_log: List[Dict[str, Any]] = field(default_factory=list)


def run_pathway(request: PathwayRequest) -> PathwayResult:
    """STUB (W0). Ignores ``request`` and returns a fixed error result. R5 implements it."""

    return PathwayResult(
        status=STATUS_ERROR,
        stage=STAGE_INIT,
        failure_kind=KIND_UNKNOWN,
        message="run_pathway is a W0 stub: no pipeline ran. R5 implements it.",
    )
