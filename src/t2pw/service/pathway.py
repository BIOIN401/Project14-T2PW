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
    Consent, email and job ids are job-store fields, not pipeline inputs.
    """

    texts: Tuple[str, ...] = ()
    pdf_paths: Tuple[Path, ...] = ()
    pmids: Tuple[str, ...] = ()
    scope: RequestedScope = field(default_factory=RequestedScope)
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
    ``report_paths`` maps a report filename to where it was written.
    """

    status: str = STATUS_ERROR
    stage: str = STAGE_INIT
    failure_kind: str = ""
    message: str = ""
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
