"""Run one paper through the *real* Streamlit app, headlessly, for one export mode.

WHY this drives the app instead of calling the pipeline functions directly
---------------------------------------------------------------------------
``src/t2pw/app/streamlit_app.py`` is not a thin view over the pipeline: it *is*
the orchestrator. Export-mode selection, the Stage-0 ambiguity refusal, the
Stage-3 pre-export revalidation, the RAG seam (S1/S2/S5), and the pre-merge
payload that the citation report has to be built from all live in that file and
nowhere else. Any batch runner that re-implemented that wiring would drift from
the app within a week, and an overnight run whose failures do not reproduce in
the browser is worse than no overnight run at all.

So this module drives the app through Streamlit's own headless harness,
``streamlit.testing.v1.AppTest``: it clicks the same widgets a human clicks and
reads the same ``st.session_state`` the app writes. The app is never imported
here, never refactored, and never duplicated -- a change to the app is picked up
automatically on the next batch run.

WHY the app's failures are read rather than caught
-------------------------------------------------
The app reports almost every problem by calling ``st.error(...)`` followed by
``st.stop()``, not by raising. ``st.stop()`` is a clean stop as far as AppTest is
concerned, so a run that failed looks exactly like a run that succeeded unless
you inspect the emitted elements. That is why :func:`run_one` checks
``at.error`` / ``at.exception`` / ``session_state["pipeline_error"]`` *before* it
looks for artifacts, and why it refuses to call anything a pass that it cannot
positively confirm.

WHY the extraction-focus box is filled in
-----------------------------------------
Topic-based fetching returns reviews, and the app refuses to extract from a
multi-example review unless a target example is named -- correctly, since the
alternative is a pathway that mixes two organisms. The app's remedy is the
"Optional extraction focus / task context" box, and the runner already knows the
topic it searched for, so :func:`_set_extraction_focus` types it in. That widget
has no ``key`` in the app, so it is addressed by *label*; if it cannot be found
the run continues with a warning rather than dying.

WHY a nested runtime_schema_report never fails a run
----------------------------------------------------
``stage_contracts.RUNTIME_SCHEMA_MODE`` is ``"report"``, which records recursive
shape findings as *warnings* on the parent contract report while the nested
report keeps its own ``ok=False``/``errors`` for information. So the parent
(``ok=True``, ``errors=[]``) is the verdict. :func:`_blocking_reports` scans only
top-level ``*_contract_report`` objects for that reason; :func:`_collect_reports`
stays exhaustive but is used only to write the archive and to surface the
runtime-schema findings as warnings.

WHY a strict Stage-3 gate error never fails a RESEARCH run
----------------------------------------------------------
The Stage-3 hard gate encodes what the PathWhiz *importer* accepts. Research mode
produces no PWML -- it produces a citation report for a human -- so those rules
are FORMAT findings there, and the project's rule for FORMAT findings in research
mode is "review flag, never failure" (``batch/report.py`` ranks a strict-PASS +
research-FAIL pair as a RESEARCH-MODE DEFECT to fix first). The gate leg of
:func:`_drive` was mode-blind and ran *before* the research branch, so
PMC12444477's research run in 2026-07-28_0919 was filed FAIL with 30 gate errors
while every contract report it produced said ``ok=true``, ``still_blocking=0``,
``research_blocked=[]``. Same class as the nested-``runtime_schema_report`` misread
below, same paper, different seam. See :func:`_research_gate_flags`.

WHY strict mode needs a third click
-----------------------------------
Verified against the app: ``pwml_generate_btn`` ("Run audit and DB mapping")
runs audit + DB mapping and stores ``post_pipeline_artifacts``, but PWML is
deliberately gated behind the human review step -- the app says "PWML generation
is paused until approval" and only ``refinement_generate_pwml`` ("Generate
PWML") calls the exporter and stores ``pwml_export_result``. A strict run that
stops after two clicks has produced no ``.pwml`` file, so the driver clicks the
third button too. Research mode intentionally has no PWML, so its absence there
is not a failure and the button is never clicked.
"""

from __future__ import annotations

import json
import re
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from t2pw.paths import PACKAGE_ROOT
from t2pw.pipeline.export_mode import STRUCTURAL_GUARD_CODES
from t2pw.pipeline.failure_detail import headline as _detail_headline
from t2pw.pipeline.gate_reports import (
    CANONICAL_PAYLOAD_KEY,
    FINAL_GATE_REPORT_KEY,
    INITIAL_GATE_REPORT_KEY,
    SOURCE_FAIL_CLOSED,
    blocking_findings,
    dedupe_findings,
    gate_verdict,
)
from t2pw.rag.eligibility import (
    OUTCOME_SCOPE_CONFLICT,
    RequestedScope,
    apply_stage0_observation,
    thresholds_from_config,
)

# ── Export-mode radio labels, exactly as the app spells them ───────────────
STRICT = "Strict PWML"
RESEARCH = "Research (relaxed)"

#: Short mode tokens used in ``RunOutcome.mode``, folder names and the manifest.
MODE_STRICT = "strict"
MODE_RESEARCH = "research"

#: The app under test. Never imported -- AppTest execs it as a script.
DEFAULT_APP_PATH = PACKAGE_ROOT / "app" / "streamlit_app.py"

#: One app *interaction* can legitimately take an hour (LLM + PathBank + RAG).
DEFAULT_APP_TIMEOUT = 3600.0
#: Whole-run wall-clock budget across all interactions for one paper+mode.
DEFAULT_TIMEOUT = 7200.0

# ── Widget keys / labels, all verified against the app ─────────────────────
KEY_EXPORT_MODE = "export_mode_radio"
KEY_INPUT_MODE = "input_mode_radio"
KEY_TEXT = "pathway_text_0"
LABEL_RUN_PIPELINE = "Run pipeline"
#: The extraction-focus box has **no key** in the app (it is a bare
#: ``st.text_area("Optional extraction focus / task context", height=100, help=...)``),
#: so it can only be addressed by label. Matched as a prefix so a trailing wording
#: tweak or an added ``:`` does not silently drop the focus text.
LABEL_EXTRACTION_FOCUS = "Optional extraction focus"
KEY_POST_PIPELINE = "pwml_generate_btn"
KEY_GENERATE_PWML = "refinement_generate_pwml"
#: Always paste: full text arrives as a ``str``, so ``st.file_uploader`` -- which
#: AppTest cannot drive with a real PDF -- is never involved.
INPUT_MODE_PASTE = "Paste text"

# ── Stage names, coarsest to furthest ──────────────────────────────────────
STAGE_INIT = "init"
STAGE_APP_LOAD = "app_load"
STAGE_INPUT = "input"
STAGE_STAGE1 = "stage1"
STAGE_POST_PIPELINE = "post_pipeline"
STAGE_PWML_EXPORT = "pwml_export"
STAGE_RESEARCH_REPORT = "research_report"

_STATUS_PASS = "pass"
_STATUS_FAIL = "fail"
_STATUS_TIMEOUT = "timeout"
_STATUS_ERROR = "error"
#: Stage 0 read a pathway/organism that contradicts what the batch asked for. NOT
#: a failure: nothing is broken, this paper is simply not the paper the topic line
#: was asking for. ``report._norm_status`` folds it out of the failure tally, so it
#: can never enter the ranked fix-list. See :func:`_reconcile_stage0_scope`.
_STATUS_SCOPE_CONFLICT = OUTCOME_SCOPE_CONFLICT

# ── failure_kind vocabulary ───────────────────────────────────────────────
KIND_CONTRACT = "contract"
KIND_LLM = "llm"
KIND_NETWORK = "network"
KIND_AMBIGUOUS = "ambiguous_review_scope"
KIND_NO_REACTIONS = "no_reactions"
KIND_CRASH = "crash"
#: Every timeout -- whichever interaction hung, and whether the *driver* or the
#: parent process noticed it -- reports this one kind. The parent's synthesized
#: timeout row uses the same string, so the ranked fix-list has ONE timeout
#: bucket instead of splitting half the hangs into "unknown".
KIND_TIMEOUT = "timeout"
KIND_UNKNOWN = "unknown"

#: ``detail`` is carried on a single stdout line, then into ``manifest.jsonl``
#: and ``RESULT.txt``. Full IR + gate reports run to 70-100 KB, which bloats all
#: three for no gain: the same reports are written as separate artifact files, so
#: the detail is a pointer, not the record.
DETAIL_LIMIT = 8000

#: Warning text for a research run that produced no citation report because RAG
#: never triggered. Not a failure -- but it must be impossible to mistake for a
#: run that produced the research deliverable.
WARN_NO_RESEARCH_REPORT = "no research report: RAG did not trigger for this paper"

#: Warning text used when a Stage-0 scope conflict is recorded but not acted on
#: (``RAG_ELIGIBILITY_STAGE0_CONFLICT_ABORTS=false``).
WARN_SCOPE_CONFLICT_PREFIX = "stage-0 scope conflict (recorded, run continued): "

#: Artifact naming the requested vs Stage-0-observed scope and the conflicts.
SCOPE_CONFLICT_NAME = "scope_conflict.json"

#: Why an aborted Stage-0 scope conflict has no PWML (D-062, C-077). Named HERE
#: and not in ``pipeline.release_status``: that module is the classifier and this
#: is one caller's reason line, exactly as ``_finalize_gate_failure`` gets its
#: ``strict_technical_gates_blocked_export`` from the classifier's own vocabulary
#: while this seam's cause has no entry there.
#:
#: The wording is deliberate on both halves. It says the run STOPPED BEFORE
#: SERIALIZATION -- which is what happened -- and it does NOT say "no defensible
#: connected core", which would be false: the payload that reached this point is
#: on disk as ``stage1_payload.json`` and ``merged_payload.json`` and, on two of
#: the six T-105 conflict legs, clears the gold set's own connected-core floor.
#: This is the same distinction ``_finalize_gate_failure`` draws in its docstring
#: -- ``diagnostic_only`` is a statement about SERIALIZATION, not about biology.
REASON_STAGE0_SCOPE_CONFLICT = "stage0_scope_conflict_stopped_the_run_before_serialization"

#: Warning text when the app rendered no extraction-focus box but the paper had a
#: topic to name in it. Not a failure -- but a topic-fetched review will then be
#: refused as ``ambiguous_review_scope``, and the reason must be on the record.
WARN_NO_FOCUS_BOX = (
    f'the app rendered no "{LABEL_EXTRACTION_FOCUS} ..." text area, '
    "so the extraction focus could not be set"
)

#: Prefix for one runtime-schema finding surfaced as a warning. See
#: :func:`_runtime_schema_warnings` for why these are never blocking.
WARN_RUNTIME_SCHEMA_PREFIX = "runtime-schema finding (informational, not blocking): "

#: Runtime-schema findings are per-row, so one bad bucket can produce dozens.
#: The warnings list rides into the manifest and RESULT.txt, so it is capped;
#: nothing is lost because every report is written to ``contract_reports.json``.
RUNTIME_SCHEMA_WARN_LIMIT = 10

#: Prefix for one strict Stage-3 gate error surfaced as a warning on a RESEARCH
#: run. See :func:`_research_gate_flags` for why these are recorded but never
#: blocking in that mode.
WARN_RESEARCH_GATE_PREFIX = "strict gate error (research mode: review flag, not blocking): "

#: Same cap and same reasoning as :data:`RUNTIME_SCHEMA_WARN_LIMIT`: PMC12444477's
#: research run produced 30 gate errors, and 30 lines of warning text in every
#: manifest row and RESULT.txt buries the one line a human needs. Every error is
#: written in full to ``gate_fail_report.json`` / ``final_stage3_gate_report.json``
#: and to the ``strict_gate_errors_not_blocking`` block of ``review_flags.json``.
RESEARCH_GATE_WARN_LIMIT = 10

#: Research-mode review-flag key for the strict gate errors that did not fail the
#: run. Deliberately NOT ``biology_provenance_flags``: those come from
#: ``export_mode.review_flags`` and mean "the biology is unverified", while these
#: mean "a PathWhiz-importer rule was not satisfied, which research mode does not
#: require". Mixing the two would make the biology flag count unreadable.
RESEARCH_GATE_FLAG_KEY = "strict_gate_errors_not_blocking"

# Network markers are checked ahead of LLM markers on purpose: an LLM call that
# cannot open a socket is a network failure, and "connection"/"uniprot"/"mysql"
# are far more specific signals than the generic "timed out" that both classes
# of failure share.
_NETWORK_MARKERS: Tuple[str, ...] = (
    "connection",
    "connectionerror",
    "connect to",
    "could not connect",
    "refused",
    "dns",
    "getaddrinfo",
    "unreachable",
    "uniprot",
    "europepmc",
    "europe pmc",
    "ncbi",
    "eutils",
    "mysql",
    "pymysql",
    "operationalerror",
    "socket",
    "ssl",
    "certificate",
    "read timed out",
    "http error",
    "httperror",
    "502",
    "503",
    "504",
    "bad gateway",
    "network",
)
_LLM_MARKERS: Tuple[str, ...] = (
    "llm",
    "openai",
    "anthropic",
    "claude",
    "json decode",
    "jsondecodeerror",
    "invalid json",
    "malformed json",
    "could not parse",
    "failed to parse",
    "unparseable",
    "truncated",
    "max tokens",
    "max_tokens",
    "context length",
    "rate limit",
    "prompt",
    "completion",
    "timed out",
    "timeout",
)
_CONTRACT_MARKERS: Tuple[str, ...] = (
    "stage boundary",
    "stagecontracterror",
    "contract",
    "hard-gate",
    "hard gate",
    "gate validation",
    "gatevalidationerror",
    "required-field gate",
    "fails the pre-export stage 3",
    # The app renders every StageContractError raised at the extraction boundary
    # as "Extraction boundary failed: <report>" -- a contract failure, even when
    # the surrounding banner text happens to mention the LLM. Checked ahead of
    # the LLM markers (see _classify) so the wording of an adjacent hint cannot
    # relabel a structural guard as a model fault.
    "boundary failed",
    "payload must be a dict",
    "payload must include",
    "rows must be objects",
)

#: Structural-guard wording -> the ``STRUCTURAL_GUARD_CODES`` code it names.
#:
#: ``t2pw.pipeline.stage_contracts`` raises these with a code, but the app shows
#: the code only inside an ``st.json(failure.report)`` block that AppTest does not
#: expose as text -- all the driver sees is ``st.error``'s sentence. Recovering the
#: code from the sentence is what lets the ranked fix-list group the failure with
#: the other contract problems instead of stranding it as "no code reported".
#:
#: Filtered against the real frozenset at import: if a code is renamed or a guard
#: is downgraded, its phrase drops out of the map rather than inventing a code
#: that ``export_mode`` no longer knows.
_STRUCTURAL_GUARD_MESSAGES: Tuple[Tuple[str, str], ...] = tuple(
    (phrase, code)
    for phrase, code in (
        ("payload must be a dict", "invalid_payload"),
        ("payload must include an entities object", "entities_required"),
        ("payload must include a processes object", "processes_required"),
        ("entity rows must be objects", "entity_not_object"),
        ("process rows must be objects", "process_not_object"),
    )
    if code in STRUCTURAL_GUARD_CODES
)
_NO_REACTION_MARKERS: Tuple[str, ...] = (
    "no reactions",
    "zero reactions",
    "empty pathway",
    "missing processes.reactions",
    "produced no reactions",
)

#: Every key in ``post_pipeline_artifacts`` whose value is a stage report worth
#: writing to disk. Discovered by suffix rather than hard-coded so a new boundary
#: added to the app is picked up without editing this file.
_CONTRACT_SUFFIXES: Tuple[str, ...] = ("_contract_report", "_runtime_schema_report")

#: The ONLY suffix whose reports may fail a run. See :func:`_blocking_reports`.
_BLOCKING_SUFFIX = "_contract_report"

#: Key of the recursive shape report nested inside a stage contract report, and
#: the suffix it is stored under when the app also keeps it at the top level.
_RUNTIME_SCHEMA_KEY = "runtime_schema_report"
_RUNTIME_SCHEMA_SUFFIX = "_runtime_schema_report"

_ENTITY_COUNT_KEYS: Tuple[str, ...] = ("proteins", "compounds")


# ---------------------------------------------------------------------------
# Small tolerant accessors. The app's own ``_safe_dict``/``_safe_list`` cannot be
# reused because importing the app module is exactly what must not happen here.
# ---------------------------------------------------------------------------
def _safe_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _safe_list(value: Any) -> List[Any]:
    return list(value) if isinstance(value, (list, tuple)) else []


def _text(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ("" if value is None else str(value))


def _ss(at: Any, key: str, default: Any = None) -> Any:
    """Read ``at.session_state[key]`` without ever raising.

    ``AppTest.session_state`` is a ``SafeSessionState``, not a ``Mapping``:
    ``.get(...)`` raises ``AttributeError`` and a missing key raises ``KeyError``
    with a docs link. Both are normal here (many keys only exist after a
    successful stage), so both degrade to ``default``.
    """

    try:
        return at.session_state[key]
    except Exception:  # noqa: BLE001 -- KeyError, AttributeError, or worse
        return default


def _element_texts(elements: Any) -> List[str]:
    """The ``.value`` of every element in one of AppTest's element lists.

    Iterated directly rather than through :func:`_safe_list`: ``at.error`` and
    friends return an ``ElementList``, which is a ``Sequence`` but neither a
    ``list`` nor a ``tuple``, so a type check on those two silently reports that
    the app said nothing -- and a run that failed loudly would look clean.
    """

    out: List[str] = []
    try:
        iterator = iter(elements)
    except TypeError:
        return out
    for element in iterator:
        try:
            value = element.value
        except Exception:  # noqa: BLE001 -- element shapes vary by element type
            value = None
        rendered = _text(value)
        if rendered:
            out.append(rendered)
    return out


def _json_default(value: Any) -> Any:
    """Bytes and other blobs are described, never inlined into a JSON report."""

    if isinstance(value, (bytes, bytearray)):
        return f"<{len(value)} bytes omitted>"
    if isinstance(value, (set, frozenset)):
        return sorted(str(item) for item in value)
    if isinstance(value, Path):
        return str(value)
    return str(value)


def _json_text(value: Any) -> str:
    """Serialize for writing. ``ensure_ascii=False`` keeps entity names readable."""

    try:
        return json.dumps(value, indent=2, ensure_ascii=False, default=_json_default)
    except Exception as exc:  # noqa: BLE001 -- never lose an artifact to a dump bug
        return json.dumps({"_unserializable": True, "error": str(exc)}, indent=2)


# ---------------------------------------------------------------------------
# Paper adapter. ``t2pw.batch.fetch.BatchPaper`` is the expected input, but a
# ``rag.acquire.CandidatePaper`` or a plain dict works too -- the driver only
# ever needs an id and the full text.
# ---------------------------------------------------------------------------
def _paper_field(paper: Any, names: Tuple[str, ...]) -> str:
    for name in names:
        if isinstance(paper, dict):
            if name in paper:
                found = _text(paper[name])
                if found:
                    return found
            continue
        found = _text(getattr(paper, name, ""))
        if found:
            return found
    return ""


def _paper_id(paper: Any) -> str:
    return _paper_field(paper, ("paper_id", "id", "pmcid", "pmid", "doi", "slug")) or "(unknown-paper)"


def _paper_text(paper: Any) -> str:
    return _paper_field(paper, ("full_text", "text", "body", "abstract"))


def _extraction_focus(paper: Any) -> str:
    """The text to type into the app's extraction-focus box, or ``""``.

    WHY this exists
    ---------------
    Topic-based fetching returns review articles ("The regulation of lipid A
    biosynthesis"), and the app deliberately refuses to extract from a
    multi-example review when no target example is named -- otherwise it would emit
    a pathway mixing E. coli LpxC regulation with A. baumannii LPS regulation. The
    app's own error says the cure: name the example in the extraction-focus box.
    A runner that leaves the box empty converts that correct refusal into a batch
    of ``ambiguous_review_scope`` failures, which is a runner gap, not an app bug.

    WHY the topic and not something cleverer
    ----------------------------------------
    The topic is the phrase the paper was *searched for*, so it is the one scope
    statement that is known to be true of this paper and was not invented here.
    The organism is appended when the paper record has one and the topic does not
    already say it -- "lipid A biosynthesis in Escherichia coli" is the shape the
    app's Stage-0 scope resolver reads best.

    A PINNED paper has ``topic == ""`` (there was no search). It gets an empty box:
    naming a scope nobody asked for would bias extraction toward it, and a pinned
    paper was chosen precisely because a human already knows what it is about.
    """

    topic = _paper_field(paper, ("requested_pathway", "topic"))
    if not topic:
        return ""
    # The REQUESTED organism, deliberately. This box states the scope the batch
    # asked for; it must not be filled from whatever organism the paper turned out
    # to mention, which would re-point extraction at the paper's own scope.
    # ``organism`` is the pre-2026-07-29 plan-record spelling, kept so an older run
    # directory still resumes.
    organism = _paper_field(paper, ("requested_organism", "organism"))
    if organism and organism.lower() not in topic.lower():
        return f"{topic} in {organism}"
    return topic


def _requested_scope(paper: Any) -> RequestedScope:
    """The batch's REQUEST for this paper, read from its plan record.

    ``requested_pathway`` / ``requested_organism`` are the current spelling;
    ``topic`` / ``organism`` are what a plan written before 2026-07-29 carries (and
    there ``organism`` WAS the requested organism -- it was stamped onto every
    paper regardless of what the paper reported). A paper with no topic is a pinned
    id: a human chose it, so there is no request to contradict.
    """
    pathway = _paper_field(paper, ("requested_pathway", "topic"))
    organism = _paper_field(paper, ("requested_organism", "organism"))
    return RequestedScope(
        requested_pathway=pathway,
        requested_organism=organism,
        pinned=not pathway,
    )


def _finalize_scope_conflict(outcome: RunOutcome, *, scope: RequestedScope) -> None:
    """Terminal path: Stage 0 read a scope that contradicts the batch request.

    Classifies the run as well as stopping it -- the same two-questions split
    :func:`_finalize_gate_failure` makes, and for the same reason. Stopping the
    run said only "this is not the requested paper"; every downstream reader then
    had to re-derive "so was anything ever attempted?" from an absence. Measured
    on the six T-105 conflict legs of ``runs_verify/2026-08-22_2147``, all six
    manifest rows carried NO ``release_status`` at all, so ``batch.report``'s only
    surviving reading of them was ``STATUS_INELIGIBLE`` -- whose own definition
    says *"nothing was attempted, so nothing failed"*. That is false on the
    evidence: Stage 1 completed, and ``stage1_payload.json`` and
    ``merged_payload.json`` were both written before this function is reached.
    D-062 reverses exactly that drop.

    WHAT IS CLASSIFIED, AND WHY IT IS NOT ``review_required``
    --------------------------------------------------------
    D-062's ruling names ``review_required``. It is NOT constructible at this
    seam, and that is a measurement, not a preference:

    * ``PRODUCT_CONTRACT`` section 4 defines ``review_required`` as "**Valid,
      useful PWML produced**, but one or more important biological uncertainties
      are explicitly identified", and :attr:`ReleaseStatus.produced_pwml` encodes
      that -- ``diagnostic_only`` is documented there as "the one state with no
      final PWML". This seam is reached BEFORE the audit, the DB mapping, the
      freeze and the export, so no PWML exists and none can.
    * The only route to ``REVIEW_REQUIRED`` through
      :func:`~t2pw.pipeline.release_status.classify_release_status` runs through
      ``strict_gates_passed=True``. The strict technical gates never ran here.
      Asserting they passed would fabricate a measurement, which is the one thing
      a classifier input must never do.

    So this path records what it can actually stand behind: the pipeline DID
    execute, and the strict gates did NOT pass. That is ``diagnostic_only``, and
    it is a statement about SERIALIZATION rather than about the biology --
    :data:`REASON_STAGE0_SCOPE_CONFLICT` says so in the reason line rather than
    letting ``strict_technical_gates_blocked_export`` stand alone and imply a gate
    verdict nobody measured. Producing a ``review_required`` PWML instead would
    require driving the run onward under an organism known to be wrong, which is a
    product decision nobody has taken; it is escalated, not assumed.

    ``strict_acceptance_eligible`` is ``False`` either way
    (``release_status.py:836`` pins it to ``status == RELEASE_READY``), so
    nothing here can raise the strict rate -- D-062's second prohibition.

    THE REQUEST IS RECORDED, NEVER REWRITTEN. ``requested_scope`` lands beside
    ``observed_context`` on the outcome and in the manifest row, in separate
    fields. Measured on the same six legs, the requested ORGANISM reached the row
    nowhere: ``topic`` carries the requested pathway alone and
    ``scope_conflicts`` carries it only inside a prose sentence, so D-062's "with
    the requested scope recorded alongside it so the mismatch stays auditable" was
    not true of the preserved record. It is now.
    """

    from t2pw.pipeline.release_status import classify_release_status

    outcome.requested_scope = scope.to_dict()
    outcome.release_status = classify_release_status(
        # Both observations, and both this path's own: Stage 1 completed (the
        # caller only reaches here after ``pipeline_ready``), and the strict
        # technical gates did not pass because they never ran.
        pipeline_executed=True,
        strict_gates_passed=False,
        extra_reasons=(REASON_STAGE0_SCOPE_CONFLICT,),
    )


def _reconcile_stage0_scope(at: Any, paper: Any, outcome: RunOutcome) -> bool:
    """Fold Stage 0's reading of this paper into the run, without rewriting the request.

    WHY this lives here
    -------------------
    ``t2pw.rag.eligibility.apply_stage0_observation`` encodes the rule that the
    batch's requested pathway/organism is immutable and that Stage 0 may only
    contribute *observations*. This is its production boundary: the app has just
    finished Stage 0 and stored its context in
    ``st.session_state["pathway_context"]``, and the plan record next to it carries
    what the batch asked for. Nowhere else do both halves exist at once.

    WHY a conflict stops the run
    ----------------------------
    The screening gate decides from a title and abstract; Stage 0 has read the
    whole paper. When it comes back saying "cholesterol biosynthesis in Homo
    sapiens" for a paper the topics file asked for as "lipid A biosynthesis in
    Escherichia coli", continuing spends the audit, the DB mapping and the export
    on a paper that is not the one requested -- and the one thing that must never
    happen is adopting the paper's apparent scope as the batch's request, which
    would silently change what the whole batch is about. So the run stops with an
    explicit :data:`_STATUS_SCOPE_CONFLICT`, which is *not* a failure: nothing is
    broken. Set ``RAG_ELIGIBILITY_STAGE0_CONFLICT_ABORTS=false`` to annotate and
    carry on instead.

    Returns ``True`` when the caller should stop driving this run.
    """
    scope = _requested_scope(paper)
    if scope.pinned:
        return False  # a pinned paper has no request to contradict
    stage0 = _ss(at, "pathway_context")
    if not isinstance(stage0, dict) or not stage0:
        return False  # Stage 0 said nothing readable; nothing to reconcile

    unchanged, observed, conflicts = apply_stage0_observation(scope, stage0)
    outcome.observed_context = observed.to_dict()
    # Recorded, and asserted: the request that comes back is the request that went
    # in. If this ever trips, something rewrote the batch's scope.
    if (unchanged.requested_pathway, unchanged.requested_organism) != (
        scope.requested_pathway,
        scope.requested_organism,
    ):  # pragma: no cover - defensive; RequestedScope is frozen
        outcome.warnings.append(
            "requested scope was altered during Stage-0 reconciliation; ignored"
        )
    outcome.counts["stage0_observed_pathways"] = len(observed.observed_pathways)
    outcome.counts["stage0_observed_organisms"] = len(observed.observed_organisms)
    if not conflicts:
        return False

    outcome.scope_conflicts = list(conflicts)
    outcome.counts["scope_conflicts"] = len(conflicts)
    if OUTCOME_SCOPE_CONFLICT not in outcome.issue_codes:
        outcome.issue_codes.append(OUTCOME_SCOPE_CONFLICT)
    outcome.artifacts[SCOPE_CONFLICT_NAME] = _json_text(
        {
            "requested": scope.to_dict(),
            "observed": observed.to_dict(),
            "conflicts": list(conflicts),
            "stage0_context": {
                key: stage0.get(key)
                for key in ("pathway_name", "likely_organism", "document_type")
            },
        }
    )
    detail = "; ".join(conflicts)
    if thresholds_from_config().stage0_conflict_aborts:
        outcome.status = _STATUS_SCOPE_CONFLICT
        outcome.failure_kind = ""
        outcome.message = (
            "Stage 0 read a scope that contradicts the batch request, so this paper "
            f"is not the paper the topic line asked for: {detail}"
        )
        outcome.detail = detail
        # D-062 / C-077. Stopping the run and CLASSIFYING it are two questions;
        # this path used to answer only the first, so the manifest row's only
        # surviving reading was ``STATUS_INELIGIBLE`` -- "nothing was attempted".
        # Something was attempted and it completed. See _finalize_scope_conflict,
        # including why the honest classification here is not ``review_required``.
        _finalize_scope_conflict(outcome, scope=scope)
        return True
    outcome.warnings.append(f"{WARN_SCOPE_CONFLICT_PREFIX}{detail}")
    return False


def _find_by_label(elements: Any, prefix: str) -> Any:
    """The first element in an AppTest element list whose label starts with ``prefix``.

    Needed because the widget this addresses has no ``key`` in the app, so
    ``at.text_area(key=...)`` cannot reach it. Returns ``None`` rather than raising:
    a widget that moved is a warning, never a crash.
    """

    try:
        iterator = iter(elements)
    except TypeError:
        return None
    for element in iterator:
        if _text(getattr(element, "label", "")).startswith(prefix):
            return element
    return None


def _set_extraction_focus(at: Any, paper: Any, outcome: RunOutcome) -> str:
    """Type the paper's scope into the unkeyed extraction-focus box.

    Returns the focus text actually delivered (``""`` when there was nothing to
    say, or when the box could not be found -- in which case a warning is
    recorded). Never raises: losing the focus box degrades the run's *scope*, and
    that has to show up as a warning on a real result rather than as a crash that
    produces nothing at all.
    """

    focus = _extraction_focus(paper)
    if not focus:
        return ""
    box = _find_by_label(getattr(at, "text_area", []), LABEL_EXTRACTION_FOCUS)
    if box is None:
        if WARN_NO_FOCUS_BOX not in outcome.warnings:
            outcome.warnings.append(WARN_NO_FOCUS_BOX)
        return ""
    try:
        box.set_value(focus)
    except Exception:  # noqa: BLE001 -- a widget-API change must not kill the batch
        if WARN_NO_FOCUS_BOX not in outcome.warnings:
            outcome.warnings.append(WARN_NO_FOCUS_BOX)
        return ""
    return focus


# ---------------------------------------------------------------------------
# Mode adapter.
# ---------------------------------------------------------------------------
def _resolve_mode(mode: Any) -> Tuple[str, str]:
    """Return ``(short_token, radio_label)`` for anything that names a mode."""

    raw = _text(mode).lower()
    if "research" in raw or "relax" in raw:
        return MODE_RESEARCH, RESEARCH
    # Default is strict, deliberately: an unrecognised mode must never silently
    # relax the gates, which is the same rule the app itself applies.
    return MODE_STRICT, STRICT


# ---------------------------------------------------------------------------
# Outcome.
# ---------------------------------------------------------------------------
def _release_status_row(status: Any) -> Optional[Dict[str, Any]]:
    """Serialize a release classification for the manifest, or answer ``None``.

    Two shapes reach this. ``_finalize_gate_failure`` attaches a
    :class:`~t2pw.pipeline.release_status.ReleaseStatus`; the strict PASS path
    attaches the boundary's already-serialized record. Both are recorded as they
    are -- nothing here re-classifies, re-orders or fills in a field, because a
    classification produced after the freeze is a merge rule 8 defect, not a
    convenience.

    ``None`` means "this run has no classification". D-038 keeps that ABSENT from
    the row rather than writing a placeholder into it: ``report.py:860-861``
    renders an absent status honestly and a fabricated one would be indistinguish-
    able from a measured one.
    """

    to_dict = getattr(status, "to_dict", None)
    if callable(to_dict):
        serialized = to_dict()
        return dict(serialized) if isinstance(serialized, dict) else None
    return dict(status) if isinstance(status, dict) and status else None


#: Why an INNER (in-process) timeout row carries no ``budget`` record, stated IN
#: the row instead of left as a hole a reader has to notice.
#:
#: PRODUCT_CONTRACT section 9 asks a terminal record for the elapsed and remaining
#: budget as well as the stop reason. The OUTER path can answer: ``runner._timeout_row``
#: is handed both ``seconds`` and the kill ``timeout``, so it publishes a full
#: ``budget`` block. The INNER path cannot. ``_finalize_timeout`` receives the
#: message, the detail and the codes -- never the ``_Budget`` -- and the leg ceiling
#: it would need is ``run_one``'s ``timeout=`` argument, which no field of
#: ``RunOutcome`` records. ``seconds`` is not a substitute: it is stamped in
#: ``run_one`` AFTER ``_drive`` returns, and elapsed without a ceiling is not a
#: budget.
#:
#: So the absence is recorded rather than filled in. Defaulting the ceiling to
#: ``deadline.LEG_TIMEOUT_SECONDS`` would put "3600s" into rows whose leg ran with a
#: different ``timeout=`` -- an invented number in the one field a reader would trust
#: to be measured. Threading the ``_Budget`` down to this seam is the honest fix and
#: it belongs to whoever owns ``_drive``'s five call sites, not to this card.
#:
#: The key is deliberately NOT named ``budget``: a consumer that finds ``budget`` on
#: an OUTER row must never find a different shape under the same name here.
INNER_BUDGET_UNRECORDED = (
    "not recorded on the in-process timeout path: this seam is handed the timeout "
    "detail only, never the leg budget, and the leg ceiling is run_one's timeout= "
    "argument, which no RunOutcome field carries -- so elapsed and remaining cannot "
    "be stated truthfully here and are not guessed. The parent's killed-child row "
    "(batch.runner._timeout_row) carries the full budget record."
)


@dataclass
class RunOutcome:
    """Everything the batch runner needs to write one paper+mode to disk.

    ``artifacts`` maps a filename to ``bytes`` or ``str`` ready to write (the
    caller owns the folder layout and must open with ``encoding="utf-8"``).
    ``counts`` and ``issue_codes`` are what the aggregator ranks on.

    ``warnings`` is for a run that genuinely passed but did *not* produce one of
    its deliverables. The status stays ``pass`` (the pipeline ran; RAG declining
    to trigger is not a code defect), but the run must never be summarised as
    clean, so the warning travels all the way into the manifest row.
    """

    paper_id: str = ""
    mode: str = MODE_STRICT
    status: str = _STATUS_ERROR
    stage: str = STAGE_INIT
    seconds: float = 0.0
    failure_kind: str = ""
    message: str = ""
    detail: str = ""
    issue_codes: List[str] = field(default_factory=list)
    artifacts: Dict[str, Any] = field(default_factory=dict)
    counts: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    #: Ways Stage 0's reading of this paper contradicts the batch's REQUEST. Set
    #: by :func:`_reconcile_stage0_scope`; the request itself is never rewritten.
    scope_conflicts: List[str] = field(default_factory=list)
    #: What Stage 0 observed (pathways / organisms / aliases / ambiguities). Kept
    #: separate from the request so neither can be mistaken for the other.
    observed_context: Dict[str, Any] = field(default_factory=dict)
    #: The batch's REQUEST for this paper, recorded when Stage 0 read something
    #: that contradicts it (D-062: "with the requested scope recorded alongside it
    #: so the mismatch stays auditable"). The OTHER half of the pair above, and
    #: kept in its own field for the same reason: the requested organism is never
    #: written into an observed one, and vice versa. Empty on every run that had
    #: nothing to reconcile, so an unaffected row is byte-identical to before.
    requested_scope: Dict[str, Any] = field(default_factory=dict)
    #: The release classification this run CARRIES -- the record the quarantine
    #: boundary froze, or the one ``_finalize_gate_failure`` built for a run the
    #: gates blocked. ``None`` means the run has none; it never means "not
    #: releasable", and it is never inferred from ``status`` or from a filename.
    release_status: Any = None
    #: The PWML filename actually written, ``""`` when none was. D-004 makes the
    #: name a function of :attr:`release_status`, so the two travel together and a
    #: reader never has to reconstruct one from the other.
    pwml_artifact: str = ""
    #: Which D-005 clock ran out, written by :func:`_finalize_timeout` on the
    #: INNER (in-process) timeout path. One of the seven termination reasons, or
    #: ``""`` when this run has no timeout classification at all. Never inferred
    #: from :attr:`status`: ``status="timeout"`` says the clock was involved,
    #: this says WHICH clock, and F-092 defect 3 is precisely what happens when
    #: only the first of those two facts survives to disk.
    termination_reason: str = ""
    #: Whether :attr:`termination_reason` is one of D-005's two OPERATIONAL
    #: outcomes. ``False`` when there is no reason at all -- absence of a
    #: classification is not a claim that the leg failed semantically.
    termination_is_operational: bool = False

    @property
    def passed(self) -> bool:
        return self.status == _STATUS_PASS

    @property
    def scope_conflicted(self) -> bool:
        return self.status == _STATUS_SCOPE_CONFLICT

    def files(self) -> List[Dict[str, Any]]:
        """``[{"name":..., "bytes": n}, ...]`` -- the manifest's ``files`` field."""

        return [
            {"name": name, "bytes": len(blob) if isinstance(blob, (bytes, bytearray, str)) else 0}
            for name, blob in self.artifacts.items()
        ]

    def capped_detail(self) -> str:
        """``detail`` trimmed to :data:`DETAIL_LIMIT`, saying what was dropped.

        Nothing is actually lost: every report that makes ``detail`` large is
        also written as its own artifact file, and the marker names them.
        """

        text = self.detail or ""
        if len(text) <= DETAIL_LIMIT:
            return text
        dropped = len(text) - DETAIL_LIMIT
        names = ", ".join(sorted(self.artifacts)) or "(no artifact files were written)"
        return (
            text[:DETAIL_LIMIT]
            + f"\n\n... [detail truncated at {DETAIL_LIMIT} chars; {dropped} more character(s) "
            f"dropped. The full reports are on disk in this folder: {names}]"
        )

    def to_dict(self) -> Dict[str, Any]:
        """A JSON-Lines manifest row. Artifact *contents* are never included."""

        row: Dict[str, Any] = {
            "paper_id": self.paper_id,
            "mode": self.mode,
            "status": self.status,
            "stage": self.stage,
            "seconds": round(float(self.seconds), 2),
            "failure_kind": self.failure_kind,
            "message": self.message,
            "detail": self.capped_detail(),
            "issue_codes": list(self.issue_codes),
            "counts": dict(self.counts),
            "files": self.files(),
            "warnings": list(self.warnings),
        }
        # Only present when there was something to say, so an unaffected run's row
        # is byte-identical to before.
        if self.scope_conflicts:
            row["scope_conflicts"] = list(self.scope_conflicts)
        if self.observed_context:
            row["observed_context"] = dict(self.observed_context)
        # The request, beside the observation, never merged into it. Measured on
        # the six T-105 conflict legs, the requested ORGANISM reached the row
        # nowhere -- ``topic`` is the requested pathway alone -- so the mismatch
        # D-062 requires to stay auditable was only recoverable by opening a
        # per-paper file. Conditional like the two above, so a run with nothing
        # to reconcile writes exactly the row it wrote before.
        if self.requested_scope:
            row["requested_scope"] = dict(self.requested_scope)
        # D-004 / D-038: the manifest records structured status INDEPENDENTLY of
        # the filename, so a consumer never has to equate ``pathway.pwml`` with
        # strict success. Exactly two keys, both conditional. D-038 struck
        # ``pipeline_status`` and ``strict_acceptance_passed`` (neither names
        # anything that exists) and struck top-level ``strict_acceptance_eligible``
        # / ``completeness``, which already live INSIDE ``release_status`` -- a
        # second copy of a benchmark-gating flag is a second source of truth.
        release = _release_status_row(self.release_status)
        if release:
            row["release_status"] = release
        if self.pwml_artifact:
            row["pwml_artifact"] = self.pwml_artifact
        # F-092 defect 3. ``_finalize_timeout`` has classified the INNER timeout
        # correctly since C-032, and this row then threw the answer away: neither
        # name was declared on the dataclass nor emitted here, so ``grep -ric
        # operation_timeout`` over runs_verify/2026-08-21_2239 and
        # runs_verify/2026-08-22_2147 returns nothing on six real timeout legs.
        # PRODUCT_CONTRACT section 9 requires the terminal record to preserve "the
        # exact stop reason"; a bare ``failure_kind="timeout"`` is not it.
        #
        # ``operational_failure`` and not ``termination_is_operational`` is the ROW
        # key on purpose: ``runner._timeout_row`` (the OUTER path) and
        # ``deadline.Admission.to_dict`` already publish this fact under that name,
        # and a manifest consumer must not need two names for one fact depending on
        # which side of the process boundary the kill came from. The ATTRIBUTE keeps
        # the name the classifier writes.
        #
        # Conditional, like the four keys above, so every row that did not time out
        # is byte-identical to what it was before this card.
        if self.termination_reason:
            row["termination_reason"] = self.termination_reason
            row["operational_failure"] = bool(self.termination_is_operational)
            row["budget_unrecorded"] = INNER_BUDGET_UNRECORDED
        return row


# ---------------------------------------------------------------------------
# Failure classification.
# ---------------------------------------------------------------------------
def _slug_code(reason: str) -> str:
    """Turn a free-text gate reason into a groupable code.

    Gate errors are ``{"path", "reason"}`` with no code, and the reason embeds the
    offending entity name (``"Forbidden complex reference detected: <token>"``).
    Cutting at the first ``:`` and dropping digits collapses every instance of
    one rule onto one code, which is what makes ranking by code meaningful.
    """

    head = reason.split(":", 1)[0]
    slug = re.sub(r"[^a-z0-9]+", "_", head.lower()).strip("_")
    slug = re.sub(r"_\d+", "", slug).strip("_")
    return f"gate.{slug[:64]}" if slug else "gate.unspecified"


def _issue_code(issue: Any) -> str:
    """A contract-report or gate error's structured code."""

    if isinstance(issue, str):
        return _slug_code(issue)
    data = _safe_dict(issue)
    code = _text(data.get("code"))
    if code:
        return code
    reason = _text(data.get("reason")) or _text(data.get("message")) or _text(data.get("detail"))
    return _slug_code(reason) if reason else ""


def _issue_pointer(issue: Any) -> str:
    data = _safe_dict(issue)
    return _text(data.get("pointer")) or _text(data.get("path"))


def _report_errors(report: Any) -> List[Any]:
    """The blocking issues of a contract/gate report, ``[]`` when it passed."""

    data = _safe_dict(report)
    if not data:
        return []
    if data.get("ok") is True and not data.get("errors"):
        return []
    return _safe_list(data.get("errors"))


def _collect_reports(artifacts: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Every stage report present, keyed by its artifact key -- for the ARCHIVE.

    This is what ``contract_reports.json`` is built from, so it is deliberately
    exhaustive: contract reports, standalone runtime-schema reports, and the
    nested ``runtime_schema_report`` of each contract report flattened alongside
    its parent so a human can grep one file.

    It is **not** the blocking scan set -- see :func:`_blocking_reports`. Feeding
    this dict to the pass/fail decision is exactly the bug that reported
    PMC12444477's research run as FAIL/``entity_missing_mapping_meta`` when every
    real contract report was ``ok=True`` with zero errors.

    Gate reports are deliberately excluded: they are read separately (they carry
    ``path``/``reason`` rather than ``code``/``pointer``), and counting them here
    too would report every gate error a second time as a contract error.
    """

    reports: Dict[str, Dict[str, Any]] = {}
    for key, value in artifacts.items():
        if not isinstance(value, dict) or not value:
            continue
        if key.endswith(_CONTRACT_SUFFIXES):
            reports[key] = value
        nested = _safe_dict(value.get(_RUNTIME_SCHEMA_KEY))
        if nested:
            reports[f"{key}.{_RUNTIME_SCHEMA_KEY}"] = nested
    return reports


def _blocking_reports(artifacts: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Only the reports whose own ``errors`` may fail the run.

    A top-level ``*_contract_report`` is the stage's verdict, and its ``errors``
    list is the blocking one. A ``runtime_schema_report`` -- nested inside a
    contract report or standing alone at the top level -- is **never** blocking:
    ``stage_contracts.RUNTIME_SCHEMA_MODE`` is ``"report"``, which by design folds
    recursive shape findings onto the parent as *warnings* while the nested report
    keeps its own ``ok=False`` / ``errors=[...]`` for information. The parent's
    ``ok=True, errors=[], warnings=[...]`` is therefore the authoritative verdict,
    and reading the nested ``ok`` as a verdict inverts it.

    Concretely (PMC12444477, research, 6 reactions / 29 entities / 0 gate errors)::

        post_normalization_contract_report            ok=True  errors=0 warnings=1
        post_normalization_contract_report.runtime_...  ok=False errors=1

    The run had succeeded. Failing it on the nested report threw the research
    deliverable away, which is why that shape is a regression test.
    """

    reports: Dict[str, Dict[str, Any]] = {}
    for key, value in artifacts.items():
        if not isinstance(value, dict) or not value:
            continue
        if key.endswith(_RUNTIME_SCHEMA_SUFFIX) or not key.endswith(_BLOCKING_SUFFIX):
            continue
        # The parent's own errors only: the nested report is dropped so it cannot
        # be mistaken for a second, failing boundary.
        reports[key] = {k: v for k, v in value.items() if k != _RUNTIME_SCHEMA_KEY}
    return reports


def _runtime_schema_warnings(reports: Dict[str, Dict[str, Any]]) -> Tuple[List[str], int]:
    """``(warning_lines, finding_count)`` for every runtime-schema report.

    These stay visible -- a recursive shape finding is real information about the
    payload -- but as warnings, never as a reason to fail. ``reports`` is the
    archive dict from :func:`_collect_reports`, so both the nested
    ``<parent>.runtime_schema_report`` entries and any standalone top-level
    ``*_runtime_schema_report`` are covered.
    """

    lines: List[str] = []
    total = 0
    dropped = 0
    for name, report in sorted(reports.items()):
        if not name.endswith(_RUNTIME_SCHEMA_SUFFIX) and not name.endswith(
            f".{_RUNTIME_SCHEMA_KEY}"
        ):
            continue
        errors = _safe_list(_safe_dict(report).get("errors"))
        total += len(errors)
        for issue in errors:
            code = _issue_code(issue) or "(no code)"
            pointer = _issue_pointer(issue) or "(no pointer)"
            message = _text(_safe_dict(issue).get("message")) or _text(issue)
            line = f"{WARN_RUNTIME_SCHEMA_PREFIX}[{name}] {code} @ {pointer}: {message}"
            if line in lines:
                # Byte-identical: the same finding on the same report, not a second
                # one, so it is collapsed rather than counted as truncated.
                continue
            if len(lines) >= RUNTIME_SCHEMA_WARN_LIMIT:
                dropped += 1
                continue
            lines.append(line)
    if dropped:
        lines.append(
            f"{WARN_RUNTIME_SCHEMA_PREFIX}and {dropped} more finding(s); "
            "all of them are in contract_reports.json"
        )
    return lines, total


def _research_gate_flags(gate_errors: List[Any]) -> Tuple[List[Dict[str, str]], List[str]]:
    """``(review_flags, warning_lines)`` for a RESEARCH run's strict gate errors.

    WHY a research run is not failed by these
    -----------------------------------------
    The Stage-3 hard gate (``process_normalizer.run_strict_post_normalization_gates``)
    encodes what the *PathWhiz importer* will accept: every protein connected,
    no forbidden complex tokens, every actor present in the registry. Research
    mode does not produce PWML at all -- it produces a citation report for a human
    reviewer -- so a rule about importer acceptance is, in that mode, a FORMAT
    finding. The project's own doctrine (``batch/report.py``: "research mode must
    never fail") is that FORMAT problems in research mode become review flags.

    THE RUN THIS EXISTS FOR (2026-07-28_0919, PMC12444477 "The regulation of
    lipid A biosynthesis", research mode): every contract report said
    ``ok=true`` / ``still_blocking=0`` / ``research_blocked=[]``, and the run was
    still filed FAIL with 30 gate errors, because the gate leg of :func:`_drive`
    read ``gate_fail_report`` / ``final_stage3_gate_report`` straight out of the
    artifacts dict and failed on them *before* the ``mode == MODE_RESEARCH``
    branch ever ran. Four of those 30 errors were self-inflicted: research mode
    skips ``drop_process_orphan_proteins`` / ``prune_disconnected_proteins`` and
    then used to judge the survivors with ``enforce_all_proteins_connected=True``
    (fixed in ``process_normalizer.normalize_process_payload``). The other 26 were
    real findings about a payload nobody was going to import.

    This is the same class of defect as the nested ``runtime_schema_report``
    misread documented on :func:`_collect_reports` -- a non-authoritative signal
    used as the verdict -- re-landed at a different seam, and on the same paper.

    Gate errors carry ``path``/``reason`` and no ``code``, so the flag rows are
    normalized to the ``code``/``pointer``/``message`` shape the rest of the
    research channel uses (``rag.research_report._flag`` reads either spelling).
    Warning lines are capped at :data:`RESEARCH_GATE_WARN_LIMIT`; the flag list
    is never capped, because ``review_flags.json`` is the record.
    """

    flags: List[Dict[str, str]] = []
    lines: List[str] = []
    dropped = 0
    for issue in gate_errors:
        data = _safe_dict(issue)
        code = _issue_code(issue) or "(no code)"
        pointer = _issue_pointer(issue) or "(no pointer)"
        # ``detail`` is a structured dict since gate errors began carrying the
        # value they rejected, so it can no longer stand in for the message --
        # _text() would splat a dict repr into review_flags.json. It is summarized
        # onto its own ``found`` key instead, and survives in full in
        # gate_fail_report.json.
        message = _text(data.get("reason")) or _text(data.get("message")) or _text(issue)
        flag: Dict[str, str] = {
            "code": code,
            "pointer": pointer,
            "message": message,
            "research_category": "format",
            "source": "strict_post_normalization_gate",
        }
        found = _detail_headline(data.get("detail"))
        if found:
            flag["found"] = found
        flags.append(flag)
        line = f"{WARN_RESEARCH_GATE_PREFIX}{code} @ {pointer}: {message}"
        if line in lines:
            # Byte-identical: the same rule on the same pointer, so it is collapsed
            # rather than counted as truncated. One forbidden-complex rule can be
            # reported by both gate reports at once.
            continue
        if len(lines) >= RESEARCH_GATE_WARN_LIMIT:
            dropped += 1
            continue
        lines.append(line)
    if dropped:
        lines.append(
            f"{WARN_RESEARCH_GATE_PREFIX}and {dropped} more gate error(s); "
            "all of them are in review_flags.json and the gate report files"
        )
    return flags, lines


def _collect_issue_codes(reports: Dict[str, Dict[str, Any]]) -> Tuple[List[str], List[str], int]:
    """``(codes, human_lines, error_count)`` across every failing report."""

    codes: List[str] = []
    lines: List[str] = []
    total = 0
    for name, report in sorted(reports.items()):
        errors = _report_errors(report)
        if not errors:
            continue
        total += len(errors)
        for issue in errors:
            code = _issue_code(issue)
            if code and code not in codes:
                codes.append(code)
            pointer = _issue_pointer(issue)
            detail = (
                _text(_safe_dict(issue).get("message"))
                or _text(_safe_dict(issue).get("reason"))
                or _text(issue)
            )
            lines.append(f"[{name}] {code or '(no code)'} @ {pointer or '(no pointer)'}: {detail}")
    return codes, lines, total


def _report_issues(reports: Dict[str, Dict[str, Any]]) -> List[Any]:
    """The failing issues themselves, not their codes.

    :func:`_collect_issue_codes` reduces each finding to a code and a display
    line, which is everything the manifest needs and not enough to *deduplicate*:
    two channels reporting one defect produce two lines whose codes differ only
    because the code is derived from the message, and the message embeds the
    entity name. The raw issues keep their ``pointer``/``path``, which is what
    :func:`t2pw.pipeline.gate_reports.finding_key` identifies them by.
    """

    issues: List[Any] = []
    for _, report in sorted(reports.items()):
        issues.extend(_report_errors(report))
    return issues


def _structural_guard_codes(text: str) -> List[str]:
    """The ``STRUCTURAL_GUARD_CODES`` named by free-text app output, if any.

    Structural guards abort in BOTH export modes on purpose -- a payload with no
    ``processes`` object has no pathway to review, so relaxing the check would only
    convert a clean ``StageContractError`` into an ``AttributeError`` two stages
    later. The refusal is correct; only the *label* was wrong. PMC13278307's
    research run was filed as ``failure_kind="llm"`` because the app's adjacent
    hint says "usually a transient LLM failure", while the actual error was
    "Extraction boundary failed: Payload must include a processes object."
    """

    low = text.lower()
    codes: List[str] = []
    for phrase, code in _STRUCTURAL_GUARD_MESSAGES:
        if phrase in low and code not in codes:
            codes.append(code)
    return codes


def _classify(
    *,
    text: str,
    issue_codes: List[str],
    contract_signal: bool,
    ambiguous: bool,
    no_reactions: bool,
    crashed: bool,
) -> str:
    """Map the evidence onto one ``failure_kind``.

    Order matters. Structured evidence beats wording, and wording beats the mere
    presence of a traceback -- the app calls ``st.exception(exc)`` in its generic
    handler, so ``at.exception`` being non-empty says nothing about *what* broke.
    """

    if ambiguous:
        return KIND_AMBIGUOUS
    if no_reactions:
        return KIND_NO_REACTIONS
    if contract_signal or issue_codes:
        return KIND_CONTRACT
    low = text.lower()
    if any(marker in low for marker in _CONTRACT_MARKERS):
        return KIND_CONTRACT
    if any(marker in low for marker in _NO_REACTION_MARKERS):
        return KIND_NO_REACTIONS
    network = any(marker in low for marker in _NETWORK_MARKERS)
    llm = any(marker in low for marker in _LLM_MARKERS)
    if network:
        return KIND_NETWORK
    if llm:
        return KIND_LLM
    if crashed:
        return KIND_CRASH
    return KIND_UNKNOWN


# ---------------------------------------------------------------------------
# Payload counting.
# ---------------------------------------------------------------------------
def _payload_counts(payload: Any) -> Dict[str, int]:
    data = _safe_dict(payload)
    processes = _safe_dict(data.get("processes"))
    entities = _safe_dict(data.get("entities"))
    counts = {
        "reactions": len(_safe_list(processes.get("reactions"))),
        "transports": len(_safe_list(processes.get("transports"))),
        "entities": sum(len(_safe_list(rows)) for rows in entities.values()),
    }
    for key in _ENTITY_COUNT_KEYS:
        counts[key] = len(_safe_list(entities.get(key)))
    return counts


# ---------------------------------------------------------------------------
# Interaction helpers.
# ---------------------------------------------------------------------------
def _looks_like_timeout(exc: BaseException) -> bool:
    """AppTest signals a run timeout as ``RuntimeError('... timed out ...')``."""

    return "timed out" in str(exc).lower()


class _Budget:
    """Wall-clock budget shared by every interaction of one paper+mode run."""

    def __init__(self, total: float, per_run: float) -> None:
        self.started = time.monotonic()
        self.total = max(1.0, float(total))
        self.per_run = max(1.0, float(per_run))

    @property
    def elapsed(self) -> float:
        return time.monotonic() - self.started

    @property
    def remaining(self) -> float:
        return self.total - self.elapsed

    def slice(self) -> float:
        """Seconds to allow the next interaction: never past the total budget."""

        return max(1.0, min(self.per_run, self.remaining))


def _collect_app_text(at: Any) -> Tuple[str, List[str], List[str]]:
    """``(joined_text, errors, exceptions)`` -- everything the app said."""

    errors = _element_texts(getattr(at, "error", []))
    exceptions = _element_texts(getattr(at, "exception", []))
    warnings = _element_texts(getattr(at, "warning", []))
    joined = "\n".join([*errors, *exceptions, *warnings])
    return joined, errors, exceptions


def _find_pwml_result(at: Any, artifacts: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    """Locate the PWML export result. Verified key first, then a real fallback.

    Verified: the app stores it in ``session_state["pwml_export_result"]``, and
    the exporter's dict carries ``ok`` / ``xml_bytes`` / ``pwml_ir``. The fallback
    scan exists because ``run_pwml_export``'s output is also folded into
    ``post_pipeline_artifacts`` on some paths, and a future refactor of the app
    must degrade to "found it anyway" rather than to a fabricated pass.
    """

    direct = _safe_dict(_ss(at, "pwml_export_result"))
    if direct:
        return "pwml_export_result", direct
    for key, value in artifacts.items():
        candidate = _safe_dict(value)
        if candidate and ("xml_bytes" in candidate or "pwml_ir" in candidate):
            return f"post_pipeline_artifacts[{key}]", candidate
    for key, value in artifacts.items():
        if "pwml" in key.lower() and isinstance(value, dict) and value:
            return f"post_pipeline_artifacts[{key}]", value
    return "", {}


def _xml_bytes(result: Dict[str, Any]) -> bytes:
    for key in ("xml_bytes", "pwml_bytes", "pwml_xml_bytes"):
        blob = result.get(key)
        if isinstance(blob, (bytes, bytearray)) and blob:
            return bytes(blob)
        if isinstance(blob, str) and blob.strip():
            return blob.encode("utf-8")
    return b""


# ---------------------------------------------------------------------------
# Artifact assembly.
# ---------------------------------------------------------------------------
def _add_common_artifacts(at: Any, artifacts: Dict[str, Any], out: Dict[str, Any]) -> None:
    """Stage-1 and merged payloads, whenever the app got far enough to store them."""

    _add_diagnostic_artifacts(at, out)
    stage_one = _ss(at, "stage_one")
    if isinstance(stage_one, dict) and stage_one:
        out["stage1_payload.json"] = _json_text(stage_one)
    merged = _ss(at, "final_payload")
    if isinstance(merged, dict) and merged:
        out["merged_payload.json"] = _json_text(merged)
    # The audit trail, unchanged: the pre-audit gate run is still written under
    # its original filename. What changed is that it is EVIDENCE -- the audit's
    # own input -- and no longer arithmetic. ``initial_stage3_gate_report.json``
    # is the same content under a name that says so, so a reader who opens the
    # run directory does not have to know the history to read it correctly.
    gate_fail = _safe_dict(artifacts.get("gate_fail_report"))
    if gate_fail:
        out["gate_fail_report.json"] = _json_text(gate_fail)
    initial_gate = _safe_dict(artifacts.get(INITIAL_GATE_REPORT_KEY))
    if initial_gate:
        out["initial_stage3_gate_report.json"] = _json_text(initial_gate)
    stage3 = _safe_dict(artifacts.get(FINAL_GATE_REPORT_KEY))
    if stage3:
        out["final_stage3_gate_report.json"] = _json_text(stage3)
    # One combined file rather than a dozen near-empty ones: the aggregator wants
    # codes (which it gets from the manifest) and a human wants one place to look.
    contracts = _collect_reports(artifacts)
    if contracts:
        out["contract_reports.json"] = _json_text(contracts)
    _add_identity_artifacts(at, artifacts, out)


def _add_identity_artifacts(at: Any, artifacts: Dict[str, Any], out: Dict[str, Any]) -> None:
    """Persist the artifacts an offline scientific audit cannot work without.

    Each used to be reachable only on paths that a failing run never takes, which
    made the most consequential questions unanswerable after the fact:

    ``final_mapped.json`` -- was written **only** by :func:`_add_strict_artifacts`,
    i.e. only when a strict run got all the way to a PWML export. Every strict run
    in ``runs/2026-07-28_2122`` died at the Stage-3 gate, so not one of them
    stored a mapped payload, and ``merged_payload.json`` is *pre-mapping*: it
    carries no accession of any kind. Auditing identity against it reports "zero
    false identifiers" for every run, which reads as a clean bill of health and
    is really "the file cannot answer the question". Writing it here -- from the
    common path, which every failure branch already calls -- is what makes "real
    IDs versus Unknown fallbacks" measurable on a run that failed.

    ``rag_admission_report.json`` -- ``AdmissionReport.rejected`` never reached
    disk at all. It is the only record of which claims the admission gate refused,
    and without it "was a rejected reaction reintroduced downstream?" is not a
    check that fails, it is a check that cannot run. Contrast
    ``quarantined_locked_reactions``, which *is* persisted into the payload, and
    which is why locked-reaction reintroduction has always been auditable.

    The QUARANTINE SET -- ``quarantine_report.json``,
    ``removed_entity_report.json``, ``graph_closure_iterations.json`` and
    ``coverage_summary.json`` -- never crossed this seam at all.
    ``write_quarantine_artifacts`` produces all four into the app's working
    ``outputs/`` directory, which the next pipeline run then clears, so
    ``find runs runs_verify -name quarantine_report.json`` returned ZERO across
    all 15 committed run directories and ``bench.acceptance``'s
    ``_QUARANTINE_FILES`` lookup never once resolved. That is refusal evidence
    destroyed at the batch boundary: a strict leg the boundary REFUSED stores its
    PRE-quarantine fallback payload and no record of the refusal, so which
    processes were dropped -- and which essential participant was undeclared --
    is not a question that gets a wrong answer after the run, it is a question
    that cannot be asked. Only two of the four are in session state at all; the
    other two exist solely as files, and the ``{name: path}`` map the producer
    returns is its designed hand-off, so the files are what is read.

    Carried VERBATIM. This is somebody else's decision record: the driver is
    writing evidence, not producing biology, and re-serializing the documents
    here would make the batch a second author of a report whose ``decision_id``
    binds it to one payload version under one set of rules.

    The name comes from an allowlist and the path must carry that same name, for
    the reason spelled out at :data:`_DIAGNOSTIC_ARTIFACT_NAMES`: a session-state
    key is not a trusted source of run-directory filenames, and an unchecked path
    would let an unrelated file's contents land under a name an auditor trusts.

    All of it is best-effort: a run that never reached mapping legitimately has
    no mapped payload, RAG is optional, and a run that never reached the boundary
    has no decision to record. Absence stays a diagnosis, not a crash.
    """

    # THE canonical payload first: enriched, quarantined, and the object the
    # final Stage-3 gate report is bound to by ``payload_sha256``. It used to be
    # ``final_mapped_db`` -- post-remap and PRE-enrichment -- so re-running the
    # gates on this file answered a question about a payload that was neither
    # gated nor exported. ``final_mapped_db`` remains the fallback for legacy
    # artifact sets and for a run that never reached the boundary.
    mapped = (
        artifacts.get(CANONICAL_PAYLOAD_KEY)
        or artifacts.get("final_mapped_db")
        or artifacts.get("final_mapped")
    )
    if isinstance(mapped, dict) and mapped:
        out["final_mapped.json"] = _json_text(mapped)

    admission = _rag_admission(_ss(at, "rag_result"))
    if admission:
        out["rag_admission_report.json"] = _json_text(admission)

    # The stamped artifact set first, because that is where every other artifact
    # on this path comes from; the session key the boundary writes itself is the
    # fallback, and it is the ONLY hand-off left for a run that took the decision
    # and then died before the set was stamped -- which is precisely the run whose
    # refusal record matters most.
    produced = _safe_dict(artifacts.get("quarantine_artifacts")) or _safe_dict(
        _ss(at, "quarantine_artifacts")
    )
    if produced:
        # Imported here, not at module scope, so a leg that never reached the
        # boundary is provably untouched by this block.
        from t2pw.pipeline.strict_quarantine import (
            CLOSURE_REPORT_FILENAME,
            COVERAGE_REPORT_FILENAME,
            QUARANTINE_REPORT_FILENAME,
            REMOVED_ENTITY_REPORT_FILENAME,
        )

        for name in (
            QUARANTINE_REPORT_FILENAME,
            REMOVED_ENTITY_REPORT_FILENAME,
            CLOSURE_REPORT_FILENAME,
            COVERAGE_REPORT_FILENAME,
        ):
            source = _text(produced.get(name))
            if not source or Path(source).name != name:
                continue
            try:
                document = Path(source).read_text(encoding="utf-8")
            except (OSError, ValueError):  # swept away, locked, or not utf-8
                continue
            if document:
                out[name] = document


def _rag_admission(rag_result: Any) -> Dict[str, Any]:
    """The admission report as PRODUCTION shapes it: ``rag_result.synthesis.admission``.

    The report hangs off ``SynthesisResult``, not off the orchestration result,
    so ``getattr(rag_result, "admission")`` is always ``None`` against a real run
    -- the artifact would simply never be written, and the reintroduction check
    would report "not evaluated" forever while looking like it was wired up. The
    access path here is copied from the app's own RAG panel
    (``streamlit_app.render_rag_panels``), which is the authoritative reader:

        synthesis = getattr(rag_result, "synthesis", None)
        admission = getattr(synthesis, "admission", None)

    ``rag_result.admission`` is still accepted as a fallback so a hand-built test
    double or a future flattening does not silently stop producing the artifact.
    Every step is ``getattr``-guarded: RAG is optional, and a run where it never
    triggered has no orchestration result at all.
    """

    if rag_result is None:
        return {}
    synthesis = getattr(rag_result, "synthesis", None)
    admission = getattr(synthesis, "admission", None) if synthesis is not None else None
    if not isinstance(admission, dict) or not admission:
        admission = getattr(rag_result, "admission", None)
    return admission if isinstance(admission, dict) and admission else {}


#: Session-state key the app publishes its Stage-0/Stage-1 diagnostics under, as
#: a ``{filename: object}`` map. Spelled out rather than imported from
#: ``t2pw.app.streamlit_app`` for the same reason ``_safe_dict`` is duplicated
#: here: importing the app module is exactly what this file must not do.
_DIAGNOSTICS_KEY = "extraction_diagnostics"

#: Filenames the app may publish. An allowlist, because these become filenames in
#: the run directory and a session-state key is not a trusted source; ``Path().
#: name`` in ``runner.write_artifacts`` is the second guard, not the first.
_DIAGNOSTIC_ARTIFACT_NAMES: Tuple[str, ...] = (
    "stage0_attempts.json",
    "extraction_boundary_report.json",
    "cleaning_report.json",
    "mapped_failure_snapshot.json",
    "audit_iteration_summary.json",
    "gap_iteration_summary.json",
)


def _add_diagnostic_artifacts(at: Any, out: Dict[str, Any]) -> None:
    """Carry the app's extraction diagnostics into the leg's artifact set.

    THE FAILURE THIS CLOSES. Every leg lost in ``runs/2026-07-28_0919`` carries
    ``files: []`` in the manifest and has nothing but ``RESULT.txt`` on disk,
    because every artifact hand-off in the app sat downstream of the ``st.stop()``
    the failure took. The diagnostics are published on the failure paths too, and
    this is called from ``_add_common_artifacts`` -- which the Stage-1 failure
    branch, the scope-conflict branch, the empty-pathway branch and the timeout
    branch all already invoke -- so a leg that dies at Stage 0 or Stage 1 now
    lands with the evidence for why.

    Only artifacts the app actually produced are copied; a missing
    ``mapped_failure_snapshot.json`` still means "mapping never ran", which is
    itself a diagnosis.
    """

    published = _safe_dict(_ss(at, _DIAGNOSTICS_KEY))
    for name in _DIAGNOSTIC_ARTIFACT_NAMES:
        blob = published.get(name)
        if isinstance(blob, (dict, list)) and blob:
            out[name] = _json_text(blob)


# ---------------------------------------------------------------------------
# Release-state artifact naming (D-004 / D-038 / PRODUCT_CONTRACT 13).
# ---------------------------------------------------------------------------
#: Reserved for ``release_ready`` and for nothing else.
PWML_RELEASE_READY_NAME = "pathway.pwml"
#: Valid, useful PWML that a reviewer must still look at.
PWML_REVIEW_REQUIRED_NAME = "pathway.review_required.pwml"

#: Recorded when a strict leg exported but carried no usable frozen record. It is
#: a WARNING, not a failure: the pipeline ran and produced bytes. It has to be
#: loud, because an unclassified run and a clean release-ready one are otherwise
#: indistinguishable in the artifact set (D-038 3).
WARN_RELEASE_STATUS_UNAVAILABLE = (
    "release classification unavailable on a passing strict leg: no usable record "
    "at pwml_result['quarantine_report']['release'], so the export was written as "
    f"{PWML_REVIEW_REQUIRED_NAME} and no release_status was recorded"
)
#: Recorded when the frozen record says ``diagnostic_only``. PRODUCT_CONTRACT 13
#: defines no final PWML for that state, so bytes that DID exist were withheld --
#: which must be stated, not left as a silent gap in the artifact set.
WARN_PWML_WITHHELD_DIAGNOSTIC_ONLY = (
    "PWML bytes were produced but withheld: the frozen release record classifies "
    "this run diagnostic_only, and PRODUCT_CONTRACT 13 defines no final PWML for "
    "that state"
)


def _frozen_release_record(pwml_result: Dict[str, Any]) -> Dict[str, Any]:
    """The classification the quarantine boundary FROZE, read from memory.

    ``pwml_result["quarantine_report"]["release"]`` and nothing else (D-038 2).
    Two sources were rejected for this and both matter:

    * ``outcome.artifacts["quarantine_report.json"]`` is a ``str`` --
      ``_add_quarantine_artifacts`` stores ``Path(...).read_text(...)`` -- so the
      subscript could not execute even if it were sound.
    * it is not sound. The batch strict export runs through the third entry point,
      which exports ``refinement_working_json``; the report on disk can therefore
      describe a DIFFERENT payload than the XML being named, while the in-memory
      record is bound to the exported graph by ``decision_matches`` on the payload
      hash and the decision-input hash.

    Re-deriving the classification here instead -- calling
    ``classify_release_status`` on the PASS path -- would be an exporter answering
    a biological question after the freeze, which merge rule 8 forbids outright.

    An unrecognised ``status`` is discarded rather than trusted: the contract
    defines exactly three states, and a fourth string is a record this code cannot
    interpret, which is the same predicament as having none.

    THE ONE THING APPLIED ON TOP (C-087, F-123, **D-068** LOCKED), and it is not a
    re-derivation. Pre-freeze canonicalization runs at the EXPORT seam, AFTER the
    boundary froze this record, so its verdict cannot reach
    ``classify_release_status``: measured across ``src``, no caller of that function
    holds a prefreeze report and no holder of a prefreeze report calls it. The
    verdict therefore reaches the FROZEN RECORD, through
    ``release_status.cap_release_for_prefreeze_declination`` -- the classifier
    module's own monotone cap, whose only transition is ``release_ready`` ->
    ``review_required``. It reads a status string and a verdict another stage
    already reached; it reads, writes and repairs no biology; it can only remove a
    strict success. Merge rule 8 forbids an exporter REPAIRING biology after the
    freeze, and this is the opposite operation.

    WHY HERE AND NOT AT ``_finalize_pwml_export``. This function is the single
    choke point: ``_add_strict_artifacts`` (``:1614``) derives the PWML FILENAME
    from it and ``_finalize_pwml_export`` (``:2049``) derives the MANIFEST ROW from
    it. Capping only the row would leave a declined leg shipping a bare
    ``pathway.pwml``, which ``PRODUCT_CONTRACT`` 13 reads as "ship it, no review
    needed" -- D-035 clause 8's *"must not become a successful export"* would then
    still be enforced only by coincidence on the other channel, which is exactly
    what D-068 forbids.

    BOTH published shapes are accepted, in the order of decreasing information. The
    whole report carries ``ok``; the ``prefreeze_review_required`` sub-mapping that
    every seam publishes beside it does not, and does not need to (see
    ``prefreeze_review_reasons``). The CLI returns a PATH string under the report
    key, which ``_safe_dict`` renders ``{}``, so the fallback is what answers there.
    """

    from t2pw.pipeline.release_status import (
        RELEASE_STATES,
        cap_release_for_prefreeze_declination,
    )

    release = _safe_dict(_safe_dict(pwml_result.get("quarantine_report")).get("release"))
    if _text(release.get("status")) not in RELEASE_STATES:
        return {}
    prefreeze = _safe_dict(pwml_result.get("prefreeze_resolution_report")) or _safe_dict(
        pwml_result.get("prefreeze_review_required")
    )
    return cap_release_for_prefreeze_declination(release, prefreeze)


def _pwml_artifact_name(release: Dict[str, Any]) -> str:
    """D-004's filename for a frozen record. ``""`` means: write no final PWML.

    ``release_ready`` earns the reserved name. ``diagnostic_only`` gets no final
    PWML at all (PRODUCT_CONTRACT 13). Everything else -- ``review_required``, and
    an ABSENT or uninterpretable record -- lands on
    ``pathway.review_required.pwml``: merge rule 7 forbids dropping the bytes, and
    a run that cannot produce its classification certainly cannot prove it is
    release-ready, so "valid, needs review" is the only honest name left.
    """

    from t2pw.pipeline.release_status import DIAGNOSTIC_ONLY, RELEASE_READY

    status = _text(release.get("status"))
    if status == RELEASE_READY:
        return PWML_RELEASE_READY_NAME
    if status == DIAGNOSTIC_ONLY:
        return ""
    return PWML_REVIEW_REQUIRED_NAME


def _add_strict_artifacts(
    artifacts: Dict[str, Any],
    pwml_result: Dict[str, Any],
    out: Dict[str, Any],
) -> str:
    """Write the strict deliverables. Returns the PWML filename, ``""`` if none.

    The name is a function of the FROZEN CLASSIFICATION, not of "some XML exists".
    Before D-004 every strict export landed as ``pathway.pwml``, so four
    independent readers equated that filename with strict success and a
    ``review_required`` fragment was indistinguishable from a release-ready
    pathway on disk. Structured status stays authoritative; the distinct names are
    the safe migration while those readers still exist.
    """

    xml = _xml_bytes(pwml_result)
    name = _pwml_artifact_name(_frozen_release_record(pwml_result)) if xml else ""
    if name:
        out[name] = xml
    ir = pwml_result.get("pwml_ir")
    if isinstance(ir, dict) and ir:
        out["pwml_ir.json"] = _json_text(ir)
    # ``artifact_name``, NOT ``name``: this loop runs unconditionally and its
    # target outlives it, so reusing ``name`` here silently overwrote the PWML
    # filename this function returns with the last JSON report's filename, on
    # every path. The bug had no live effect -- both callers re-derive the name --
    # which is exactly why it survived: it was the one dimension of this seam
    # nothing could observe. It is pinned now in
    # ``tests/test_batch_pwml_artifact_naming.py``.
    for key, artifact_name in (
        ("pwml_ir_report", "pwml_ir_report.json"),
        ("pwml_ir_validation", "pwml_ir_validation.json"),
        ("validation_report", "pwml_validation_report.json"),
        ("qa", "pwml_qa.json"),
        ("required_gate_report", "pwml_required_field_gate_report.json"),
    ):
        value = pwml_result.get(key)
        if isinstance(value, dict) and value:
            out[artifact_name] = _json_text(value)
    # ``final_mapped.json`` is written by _add_identity_artifacts on the common
    # path, which runs on every branch including the failing ones. Re-emitting it
    # here would be a second writer for one filename with no second source.
    return name


def _add_research_artifacts(
    at: Any,
    artifacts: Dict[str, Any],
    out: Dict[str, Any],
    counts: Dict[str, Any],
    *,
    gate_flags: Optional[List[Dict[str, str]]] = None,
) -> str:
    """Write the research deliverables. Returns a note for ``message``.

    The citation report must be built from the **pre-merge** payload
    (``rag_result.payload``): ``pipeline._clean_processes`` strips
    ``rag_provenance`` / ``source_papers`` / ``source_refs`` from every reaction
    during the merge, so tiering the merged payload would report Tier D
    (unsourced) for everything. Identifiers, conversely, only exist on the
    *mapped* payload, so the two are passed separately -- the same split the app
    itself makes.

    ``gate_flags`` is the strict Stage-3 gate's errors, normalized by
    :func:`_research_gate_flags`, for a run that did not fail on them. They are
    written under their own key rather than merged into
    ``biology_provenance_flags`` (which means "the biology is unverified") or
    ``format_rules_skipped`` (which is the app's own relaxation ledger from
    ``export_mode.format_gaps``) -- PMC12444477's research run would otherwise
    have reported 30 extra "biology" flags for what are importer-shape rules.
    They are also deliberately kept out of ``build_citation_report``: the report's
    "PathWhiz format rules skipped (N)" header counts the app's relaxations, and
    adding 30 gate rows to it would misstate what the app actually relaxed.
    """

    flags = _safe_list(artifacts.get("research_review_flags"))
    skipped = _safe_list(artifacts.get("research_skipped_format_rules"))
    preserved = _safe_list(artifacts.get("research_normalization_actions"))
    gate_rows = list(gate_flags or [])
    counts["review_flags"] = len(flags)
    counts["skipped_format_rules"] = len(skipped)
    counts["normalization_actions"] = len(preserved)
    if gate_rows:
        counts["gate_review_flags"] = len(gate_rows)

    out["review_flags.json"] = _json_text(
        {
            "export_mode": artifacts.get("export_mode"),
            "biology_provenance_flags": flags,
            "format_rules_skipped": skipped,
            "content_preserved": preserved,
            RESEARCH_GATE_FLAG_KEY: gate_rows,
        }
    )

    rag_result = _ss(at, "rag_result")
    synthesized = getattr(rag_result, "payload", None)
    if not isinstance(synthesized, dict) or not synthesized:
        # Not a failure: RAG is optional and may legitimately not have triggered.
        return "no RAG synthesis in this run, so no citation report was produced"

    from t2pw.rag.research_report import build_citation_report

    report = build_citation_report(
        synthesized,
        review_flags=flags,
        format_gaps=skipped,
        rag_result=rag_result,
        mode=RESEARCH,
        identity_source=artifacts.get("final_mapped_db") or artifacts.get("final_mapped"),
    )
    out["research_pathway_report.txt"] = report.to_text()
    out["research_pathway_citations.json"] = report.to_json()
    out["research_pathway_elements.csv"] = report.to_csv()

    summary = _safe_dict(report.summary)
    tier_counts = _safe_dict(summary.get("tier_counts"))
    counts["tier"] = {str(tier).lower(): int(value or 0) for tier, value in tier_counts.items()}
    counts["citations"] = int(summary.get("distinct_sources_cited", 0) or 0)
    counts["unsourced"] = int(summary.get("unsourced_count", 0) or 0)
    return ""


# ---------------------------------------------------------------------------
# The driver.
# ---------------------------------------------------------------------------
def run_one(
    paper: Any,
    mode: Any,
    *,
    app_path: Optional[Any] = None,
    timeout: float = DEFAULT_TIMEOUT,
    app_timeout: float = DEFAULT_APP_TIMEOUT,
) -> RunOutcome:
    """Run one paper through the real app once, in one export mode.

    Never raises. Any internal problem -- a bad app path, an AppTest change, a
    bug in this module -- becomes ``status="error"``, ``failure_kind="crash"``
    with the traceback in ``detail``, because an overnight batch that dies on
    paper three is worthless.
    """

    mode_token, mode_label = _resolve_mode(mode)
    outcome = RunOutcome(paper_id=_paper_id(paper), mode=mode_token)
    budget = _Budget(timeout, app_timeout)
    try:
        _drive(paper, mode_label, outcome, budget, app_path)
    except Exception:  # noqa: BLE001 -- the whole point of this wrapper
        outcome.status = _STATUS_ERROR
        outcome.failure_kind = KIND_CRASH
        outcome.message = outcome.message or "batch driver crashed before it could judge the run"
        outcome.detail = ((outcome.detail + "\n\n") if outcome.detail else "") + traceback.format_exc()
    outcome.seconds = round(budget.elapsed, 2)
    return outcome


def _fail(
    outcome: RunOutcome,
    *,
    status: str,
    kind: str,
    message: str,
    detail: str = "",
    codes: Optional[List[str]] = None,
) -> None:
    outcome.status = status
    outcome.failure_kind = kind
    outcome.message = message
    if detail:
        outcome.detail = detail
    if codes:
        for code in codes:
            if code not in outcome.issue_codes:
                outcome.issue_codes.append(code)


def _run_app(at: Any, budget: _Budget) -> Tuple[bool, str]:
    """One ``at.run()``. Returns ``(timed_out, detail)`` -- other errors raise."""

    if budget.remaining <= 0:
        return True, f"whole-run budget of {budget.total:.0f}s was already spent"
    try:
        at.run(timeout=budget.slice())
    except Exception as exc:  # noqa: BLE001
        if _looks_like_timeout(exc):
            return True, f"{exc}"
        raise
    return False, ""


# ---------------------------------------------------------------------------
# Terminal paths. Lifted verbatim out of _drive so each exit is separately
# addressable; they must stay byte-equivalent to what _drive used to do inline.
# ---------------------------------------------------------------------------
def _finalize_timeout(
    outcome: RunOutcome,
    *,
    message: str,
    detail: str,
    codes: Optional[List[str]] = None,
    reason: str = "",
) -> None:
    """Terminal path: an interaction ran out of the leg's wall-clock budget.

    Classifies as well as fails. ``failure_kind="timeout"`` says only that the
    clock was involved; D-005 needs *which* clock, because the leg spending its
    entire budget (``budget_exhausted``) and one interaction overrunning its own
    configured maximum (``operation_timeout``) are separate outcomes that must
    never be conflated. :func:`_run_app` already distinguishes them -- it refuses
    to start an interaction once the whole-run budget is gone and says so in
    ``detail`` -- but that was thrown away here, so every timeout arrived
    downstream as one undifferentiated kind. Both answers are operational, so the
    leg can never be read as a semantic failure, as scientific insufficiency, or
    as retrieval exhaustion: **a timeout is an operational fact, never a
    biological verdict.**

    ``reason`` lets a caller that already knows say so outright, which is how the
    escalation ladder (C-042) will report a rung it could not afford.

    **C-083 / F-092 defect 3.** C-032 wrote the two answers onto ``outcome`` and
    stopped there, recording in this docstring that adding them to
    :meth:`RunOutcome.to_dict` "would edit ``RunOutcome``, outside this card's
    boundary". That was an accurate statement of C-032's boundary and an interim
    state of the record: the classification was computed correctly and then
    dropped by the serializer, so six real timeout legs across
    ``runs_verify/2026-08-21_2239`` and ``runs_verify/2026-08-22_2147`` reached disk
    with ``failure_kind="timeout"`` and no trace of ``operation_timeout`` anywhere.
    D-004 gave the row an owner; this card is the one that owns it. Both names are
    now declared on :class:`RunOutcome` and emitted CONDITIONALLY by
    :meth:`RunOutcome.to_dict` (as ``termination_reason`` /
    ``operational_failure``, the OUTER path's key names), so a run that never timed
    out still writes exactly the row it wrote before and the classification this
    function computes now survives to the manifest.

    **Nothing about the classification itself moves here.** The verdict, its
    inputs, and the closed vocabulary are unchanged; only its persistence is
    corrected. This function still does not know the leg budget --
    :data:`INNER_BUDGET_UNRECORDED` says so in the row rather than inventing one.
    """

    from t2pw.pipeline import deadline as leg_deadline

    _fail(
        outcome,
        status=_STATUS_TIMEOUT,
        kind=KIND_TIMEOUT,
        message=message,
        detail=detail,
        codes=codes,
    )
    # Classified AFTER the row is written: a bad ``reason`` must fail loud, but it must
    # not also abandon the terminal record. The leg really did time out.
    outcome.termination_reason = leg_deadline.classify_interaction_timeout(detail, explicit=reason)
    outcome.termination_is_operational = leg_deadline.is_operational(outcome.termination_reason)


def _gate_failure_semantic_carry(artifacts: Dict[str, Any]) -> Dict[str, Any]:
    """The semantic verdict the quarantine boundary FROZE, as classifier kwargs.

    ``{}`` means the boundary left nothing to carry, and the caller then takes
    ``classify_release_status``'s unwired defaults -- an honest ``not_evaluated``.

    WHY THIS SOURCE, when :func:`_frozen_release_record` reads an in-memory object
    on the PASS path. ``_finalize_gate_failure``'s parameters are ``outcome`` and
    eight verdict-side values; neither ``at`` nor ``artifacts`` (the app's
    ``post_pipeline_artifacts``) is among them, and ``_drive`` RETURNS at the
    gate-failure call, 85 lines before ``pwml_result`` is bound by
    ``_find_pwml_result``. Both in-memory routes -- ``_ss(at,
    "quarantine_report")``, written by ``streamlit_app.py:1143``, and
    ``artifacts["quarantine_report"]``, the same object republished at
    ``streamlit_app.py:3713`` -- would therefore need a new parameter on this
    seam. ``outcome`` is a parameter, and ``_add_common_artifacts`` has already
    run: it reaches :func:`_add_identity_artifacts`, which puts
    ``QUARANTINE_REPORT_FILENAME`` into the artifact set as the verbatim text of
    the file ``write_quarantine_artifacts`` wrote from the same
    ``QuarantineResult``.

    WHY READING IT BACK IS SOUND HERE, when ``_frozen_release_record`` rejects the
    on-disk report for the PASS path. That objection is anchored on the export --
    "the report on disk can therefore describe a DIFFERENT payload than the XML
    being named". No XML is named on this path, and more decisively the export
    never ran: the boundary is invoked ONCE per pipeline run
    (``streamlit_app.py:2561``), and the only other ``quarantine_and_close`` call
    (``streamlit_app.py:3990``, inside ``run_pwml_export``) is unreachable behind
    the ``refinement_gate_errors`` return at ``streamlit_app.py:1911``. One
    decision exists for this leg, and this text is its serialization. Each leg
    also gets a fresh ``AppTest`` session, and ``outputs/`` is cleared ahead of
    every pipeline run, so the file cannot belong to a previous leg.

    ONLY THE SEMANTIC FIELDS TRAVEL. The record is NOT carried wholesale the way
    the PASS path carries it: the boundary's own ``strict_gates_passed`` is its
    closure/lock invariant set, computed BEFORE the final pre-export Stage-3
    suite runs, so on a leg those later gates blocked it can legitimately still
    read ``release_ready``. Copying that status onto a leg that produced no PWML
    would manufacture a strict success -- the opposite of this fix.

    Unrecognised, absent or unparseable input answers ``{}`` rather than being
    trusted, for the reason ``_frozen_release_record`` gives about a fourth
    status string: a record this code cannot interpret is the same predicament as
    having none.
    """

    from t2pw.pipeline.release_status import (
        SEMANTIC_FAILED,
        SEMANTIC_NOT_EVALUATED,
        SEMANTIC_PASSED,
    )
    from t2pw.pipeline.strict_quarantine import QUARANTINE_REPORT_FILENAME

    document = artifacts.get(QUARANTINE_REPORT_FILENAME)
    if isinstance(document, (bytes, bytearray)):
        document = document.decode("utf-8", "replace")
    if isinstance(document, str):
        try:
            document = json.loads(document) if document.strip() else {}
        except ValueError:  # a truncated or swept file is not a verdict
            return {}
    release = _safe_dict(_safe_dict(document).get("release"))
    evaluation = _text(release.get("semantic_evaluation"))
    # A CLOSED-VOCABULARY guard, the same shape as ``_frozen_release_record``'s
    # ``... in RELEASE_STATES``: PRODUCT_CONTRACT 11 defines exactly three semantic
    # outcomes and a fourth string is a record this code cannot interpret. It is
    # NOT an affirmative read of ``passed`` under D-054 section 7 -- all three
    # members are treated identically, nothing branches on which one arrived, and
    # no denominator, numerator or rate is built here or anywhere downstream of it.
    if evaluation not in (SEMANTIC_PASSED, SEMANTIC_FAILED, SEMANTIC_NOT_EVALUATED):
        return {}
    carried: Dict[str, Any] = {
        "semantic_evaluation": evaluation,
        # Names, verbatim. ``classify_release_status`` keeps them only on a
        # ``failed``, so a carried pass cannot arrive holding failing checks.
        "semantic_failed_checks": _safe_list(release.get("semantic_failed_checks")),
        # C-056c's carrier, in the serialized mapping shape the classifier
        # documents itself as tolerating so a record rebuilt from JSON keeps it.
        "semantic_check_evaluability": _safe_list(release.get("semantic_check_evaluability")),
    }
    reason = _text(release.get("semantic_not_evaluated_reason"))
    if reason:
        carried["semantic_not_evaluated_reason"] = reason
    return carried


def _finalize_gate_failure(
    outcome: RunOutcome,
    *,
    verdict: Any,
    blocking_issues: List[Any],
    codes: List[str],
    code_lines: List[str],
    joined: str,
    crashed: bool,
    stage3_gate: Dict[str, Any],
    gate_fail: Dict[str, Any],
) -> None:
    """Terminal path: the gate channel or the contract channel blocked the run.

    Classifies the run as well as failing it. The two are different questions and
    this path used to answer only the first: ``status="fail"`` says the leg did
    not complete, and every downstream reader then had to re-derive "so is there
    anything releasable?" from the absence of a ``pathway.pwml``. The
    classification is attached as a first-class
    :class:`~t2pw.pipeline.release_status.ReleaseStatus`, with
    PRODUCT_CONTRACT 11's states unfolded.

    TWO OF THE THREE ARE THIS PATH'S OWN OBSERVATIONS and stay so: the pipeline
    DID execute, and the strict TECHNICAL gates did NOT pass -- that second one
    is what ``_drive`` just measured off ``FINAL_GATE_REPORT_KEY`` and the
    contract channel, and it is a gate result, never a biological verdict.

    THE THIRD IS NOT THIS PATH'S TO ANSWER (F-055, merge rule 8). This function
    used to pass NO semantic argument at all, so ``classify_release_status``'s
    unwired defaults fired and every gate-failed manifest row recorded
    ``semantic_evaluation: not_evaluated`` -- while the SAME leg's quarantine
    boundary had already computed ``failed`` and named the checks. Both records
    shipped, disagreeing, in ``runs_verify/2026-08-18_1328``. The verdict is now
    CARRIED from the boundary rather than re-derived here: see
    :func:`_gate_failure_semantic_carry` for the source and why it is the only
    one reachable at this seam.

    Carrying is subtractive-safe by construction, not by promise.
    ``strict_gates_passed=False`` pins the status at ``diagnostic_only``
    (``release_status.py:491-493``), which pins ``strict_acceptance_eligible``
    to ``False`` (``:563``) and puts the run below the semantic cap's only entry
    point (``:540``, which acts on ``release_ready`` alone). No semantic input
    this function can carry -- ``failed``, ``passed`` or ``not_evaluated`` --
    can move either field. Nothing here reads the verdict; it is recorded.

    ``not_evaluated`` remains the honest answer when the boundary left no record
    to carry: the defect was recording it when the boundary HAD computed one.

    ``diagnostic_only`` here is a statement about SERIALIZATION, not about the
    biology: the gate channel blocked export, so no PWML exists to review, and
    the reason line says exactly that rather than implying "no defensible core".

    D-004 / C-053 have since promoted the record into the manifest row through
    :func:`_release_status_row`, which copies it wholesale -- so what this
    function attaches is what an offline reader gets.
    """

    from t2pw.pipeline.release_status import classify_release_status

    outcome.release_status = classify_release_status(
        pipeline_executed=True,
        strict_gates_passed=False,
        **_gate_failure_semantic_carry(outcome.artifacts),
    )

    _fail(
        outcome,
        status=_STATUS_FAIL,
        kind=_classify(
            text=joined,
            issue_codes=codes,
            contract_signal=True,
            ambiguous=False,
            no_reactions=False,
            crashed=crashed,
        ),
        message=(
            "post-pipeline validation failed: "
            f"{len(blocking_issues)} blocking issue(s) at "
            f"{_text(verdict.stage) or 'a stage boundary'}"
            + (
                f" [{verdict.reason}]"
                if verdict.source == SOURCE_FAIL_CLOSED
                else ""
            )
        ),
        detail="\n".join(part for part in (joined, *code_lines) if part)
        or _json_text(
            {
                "gate_verdict": {
                    "failed": verdict.failed,
                    "source": verdict.source,
                    "reason": verdict.reason,
                    "stage": verdict.stage,
                },
                FINAL_GATE_REPORT_KEY: stage3_gate,
                # Retained as evidence, never as arithmetic.
                "superseded_gate_fail_report": gate_fail,
            }
        ),
        codes=codes,
    )


def _finalize_pwml_export(
    outcome: RunOutcome,
    *,
    xml: bytes,
    pwml_result: Dict[str, Any],
    joined: str,
    codes: List[str],
) -> None:
    """Terminal path: strict mode produced PWML XML, so the leg is a pass.

    It also RECORDS what was released, which "the leg passed" never said. A
    technical pass is a gate result; ``release_status`` is the release decision,
    and D-004 puts both in the row so no consumer has to re-derive the second from
    the first or from an artifact filename.

    The filename is re-derived here from the same frozen record, by the same pure
    function ``_add_strict_artifacts`` used, rather than threaded back through
    ``_drive``: both helpers already receive ``pwml_result``, so a shared pure
    function costs nothing and keeps the sprint's second-largest hotspot at zero
    changed lines.
    """

    release = _frozen_release_record(pwml_result)
    name = _pwml_artifact_name(release)
    outcome.release_status = release or None
    outcome.pwml_artifact = name
    if not release:
        # Fail loud. The leg is still a pass -- the pipeline ran and produced
        # bytes -- but it must never be summarised as a clean release-ready run,
        # and ``warnings`` is the field that travels into the manifest row.
        outcome.warnings.append(WARN_RELEASE_STATUS_UNAVAILABLE)
    elif not name:
        outcome.warnings.append(WARN_PWML_WITHHELD_DIAGNOSTIC_ONLY)

    outcome.counts["pwml_bytes"] = len(xml)
    pwml_counts = _safe_dict(pwml_result.get("counts"))
    if pwml_counts:
        outcome.counts["pwml"] = {str(k): v for k, v in pwml_counts.items()}
    outcome.status = _STATUS_PASS
    outcome.failure_kind = ""
    outcome.message = (
        f"strict run completed; {name} is {len(xml)} bytes"
        if name
        else f"strict run completed; {len(xml)} bytes of PWML were withheld"
    )
    outcome.detail = joined
    outcome.issue_codes = codes


def _drive(
    paper: Any,
    mode_label: str,
    outcome: RunOutcome,
    budget: _Budget,
    app_path: Optional[Any],
) -> None:
    """The interaction script. Mutates ``outcome`` and returns on any verdict."""

    from streamlit.testing.v1 import AppTest

    target = Path(app_path) if app_path else DEFAULT_APP_PATH
    if not target.is_file():
        _fail(
            outcome,
            status=_STATUS_ERROR,
            kind=KIND_CRASH,
            message=f"app script not found: {target}",
        )
        return

    source_text = _paper_text(paper)
    if not source_text:
        _fail(
            outcome,
            status=_STATUS_FAIL,
            kind=KIND_NO_REACTIONS,
            message="paper has no full text, so there was nothing to extract",
        )
        return

    at = AppTest.from_file(str(target), default_timeout=budget.per_run)

    # ── 1. Load the app ────────────────────────────────────────────────────
    timed_out, detail = _run_app(at, budget)
    if timed_out:
        _finalize_timeout(
            outcome,
            message="the app did not finish its first render",
            detail=detail,
        )
        return
    outcome.stage = STAGE_APP_LOAD
    joined, _errors, exceptions = _collect_app_text(at)
    if exceptions:
        _fail(
            outcome,
            status=_STATUS_ERROR,
            kind=_classify(
                text=joined,
                issue_codes=[],
                contract_signal=False,
                ambiguous=False,
                no_reactions=False,
                crashed=True,
            ),
            message="the app raised while loading, before any input was given",
            detail=joined,
        )
        return

    # ── 2. Choose the mode, paste the text, submit ─────────────────────────
    # Two runs, not one: the export-mode and input-mode radios sit *outside* the
    # form precisely because the app re-renders on them, and letting that render
    # happen is what a human does. The text area only exists while input mode is
    # "Paste text", so it is set after that render, not before.
    at.radio(key=KEY_EXPORT_MODE).set_value(mode_label)
    at.radio(key=KEY_INPUT_MODE).set_value(INPUT_MODE_PASTE)
    timed_out, detail = _run_app(at, budget)
    if timed_out:
        _finalize_timeout(
            outcome,
            message="the app hung while switching to the requested export/input mode",
            detail=detail,
        )
        return
    outcome.stage = STAGE_INPUT

    at.text_area(key=KEY_TEXT).set_value(source_text)
    # Both text areas live inside the ``pwml_pipeline`` form, so the focus is set in
    # the same interaction as the pasted text and arrives with the submit click --
    # which is what a human does, and the only way the app sees it at all.
    _set_extraction_focus(at, paper, outcome)
    submit = next((button for button in at.button if button.label == LABEL_RUN_PIPELINE), None)
    if submit is None:
        _fail(
            outcome,
            status=_STATUS_ERROR,
            kind=KIND_CRASH,
            message=f'the app rendered no "{LABEL_RUN_PIPELINE}" button; its contract changed',
            detail="buttons seen: " + ", ".join(repr(b.label) for b in at.button),
        )
        return
    submit.click()
    timed_out, detail = _run_app(at, budget)
    if timed_out:
        _finalize_timeout(
            outcome,
            message="extraction did not finish inside the time budget",
            detail=detail,
        )
        return

    # ── 3. Judge stage 1 BEFORE looking for artifacts ─────────────────────
    # The app reports failure with st.error(...) + st.stop(), which AppTest sees
    # as a clean run. Only the emitted elements and pipeline_error say otherwise.
    joined, errors, exceptions = _collect_app_text(at)
    pipeline_error = _safe_dict(_ss(at, "pipeline_error"))
    ready = bool(_ss(at, "pipeline_ready", False))

    if _text(pipeline_error.get("status")) == KIND_AMBIGUOUS:
        _fail(
            outcome,
            status=_STATUS_FAIL,
            kind=KIND_AMBIGUOUS,
            message=_text(pipeline_error.get("message"))
            or "multi-example review with no selected example; extraction was refused",
            detail="\n".join(part for part in (_json_text(pipeline_error), joined) if part),
            codes=[KIND_AMBIGUOUS],
        )
        outcome.stage = STAGE_STAGE1
        return

    if not ready:
        outcome.stage = STAGE_STAGE1
        # A structural guard ("Payload must include a processes object") is a
        # StageContractError, not a model fault: it belongs in the contract bucket
        # with its code, even though the app prints an LLM hint next to it.
        guard_codes = _structural_guard_codes(joined)
        _fail(
            outcome,
            status=_STATUS_ERROR if exceptions else _STATUS_FAIL,
            kind=_classify(
                text=joined,
                issue_codes=guard_codes,
                contract_signal=False,
                ambiguous=False,
                no_reactions=False,
                crashed=bool(exceptions),
            ),
            message=(
                errors[0].splitlines()[0]
                if errors
                else "extraction did not complete and the app reported no error"
            ),
            detail=joined or "(the app emitted no error, warning or exception)",
            codes=guard_codes,
        )
        _add_common_artifacts(at, {}, outcome.artifacts)
        return

    outcome.stage = STAGE_STAGE1

    # ── 3b. Reconcile Stage 0's reading with what the batch REQUESTED ──────
    # First point in the run where both halves exist: the app has stored its
    # Stage-0 context and the plan record says what was asked for. The request is
    # never rewritten from the observation -- see _reconcile_stage0_scope.
    if _reconcile_stage0_scope(at, paper, outcome):
        _add_common_artifacts(at, {}, outcome.artifacts)
        return

    merged_counts = _payload_counts(_ss(at, "final_payload"))
    outcome.counts.update(merged_counts)
    if not merged_counts["reactions"] and not merged_counts["transports"]:
        _fail(
            outcome,
            status=_STATUS_FAIL,
            kind=KIND_NO_REACTIONS,
            message="the pipeline produced a pathway with no reactions and no transports",
            detail=joined or "(no app error; the payload is simply empty)",
        )
        _add_common_artifacts(at, {}, outcome.artifacts)
        return

    # ── 4. Audit + DB mapping ─────────────────────────────────────────────
    try:
        post_button = at.button(key=KEY_POST_PIPELINE)
    except Exception as exc:  # noqa: BLE001 -- KeyError when the app never rendered it
        _fail(
            outcome,
            status=_STATUS_FAIL,
            kind=_classify(
                text=joined,
                issue_codes=[],
                contract_signal=False,
                ambiguous=False,
                no_reactions=False,
                crashed=bool(exceptions),
            ),
            message=f'the app never rendered the "{KEY_POST_PIPELINE}" button ({exc})',
            detail=joined,
        )
        _add_common_artifacts(at, {}, outcome.artifacts)
        return

    post_button.click()
    timed_out, detail = _run_app(at, budget)
    if timed_out:
        _finalize_timeout(
            outcome,
            message="audit and DB mapping did not finish inside the time budget",
            detail=detail,
        )
        _add_common_artifacts(at, {}, outcome.artifacts)
        return

    outcome.stage = STAGE_POST_PIPELINE
    joined, errors, exceptions = _collect_app_text(at)
    artifacts = _safe_dict(_ss(at, "post_pipeline_artifacts"))
    # Two scans over two different sets, deliberately: the blocking verdict comes
    # from the top-level *_contract_report objects only, while runtime-schema
    # findings -- which RUNTIME_SCHEMA_MODE="report" records as parent warnings --
    # stay visible as warnings and can never fail the run.
    blocking_reports = _blocking_reports(artifacts)
    codes, code_lines, error_count = _collect_issue_codes(blocking_reports)
    blocking_contract_issues = _report_issues(blocking_reports)
    schema_warnings, schema_findings = _runtime_schema_warnings(_collect_reports(artifacts))
    for warning in schema_warnings:
        if warning not in outcome.warnings:
            outcome.warnings.append(warning)
    if error_count:
        outcome.counts["contract_errors"] = error_count
    if schema_findings:
        outcome.counts["runtime_schema_findings"] = schema_findings
    _add_common_artifacts(at, artifacts, outcome.artifacts)

    if not artifacts:
        # A StageContractError is caught by the app and rendered, which leaves
        # post_pipeline_artifacts absent -- so absence itself is the signal. The
        # code only exists in the sentence the app printed, so recover it.
        for code in _structural_guard_codes(joined):
            if code not in codes:
                codes.append(code)
        _fail(
            outcome,
            status=_STATUS_ERROR if exceptions else _STATUS_FAIL,
            kind=_classify(
                text=joined,
                issue_codes=codes,
                contract_signal=False,
                ambiguous=False,
                no_reactions=False,
                crashed=bool(exceptions),
            ),
            message=(
                errors[0].splitlines()[0]
                if errors
                else "audit and DB mapping stored no artifacts and reported no error"
            ),
            detail="\n".join(part for part in (joined, *code_lines) if part),
            codes=codes,
        )
        return

    mapped_counts = _payload_counts(artifacts.get("final_mapped_db") or artifacts.get("final_mapped"))
    outcome.counts.update({key: value for key, value in mapped_counts.items() if value})

    gate_fail = _safe_dict(artifacts.get("gate_fail_report"))
    stage3_gate = _safe_dict(artifacts.get(FINAL_GATE_REPORT_KEY))
    # ONE derivation, from the report that describes what shipped. The old
    # expression here was ``bool(gate_failed or normalization_gate_failed or
    # gate_fail or stage3_gate.ok is False)`` -- an OR over dictionaries, two of
    # which are *historical*: ``gate_fail_report`` records the pre-audit gate run
    # whose failures the audit is then handed as its repair list. A truthy stale
    # dict with no artifact able to countermand it meant no repair could ever
    # produce a PASS. Across runs/2026-08-02_2130 every strict leg carrying a
    # gate_fail_report.json was filed FAIL (6 of 6), and PMC12782028's final
    # payload passed the strict suite outright.
    #
    # :func:`t2pw.pipeline.gate_reports.gate_verdict` reads the final report and
    # nothing else, fails closed on every uncertainty, and falls back to the old
    # arithmetic only for artifact sets stamped as pre-change -- which genuinely
    # cannot produce a final report and must keep their recorded history.
    verdict = gate_verdict(artifacts)
    gate_failed = verdict.failed
    gate_errors = list(verdict.errors)
    for issue in gate_errors:
        code = _issue_code(issue)
        if code and code not in codes:
            codes.append(code)
    outcome.counts["gate_errors"] = len(gate_errors)

    # This leg used to be mode-blind: it read the two gate reports out of the
    # artifacts dict and failed the run on them with no mode check, and it runs
    # BEFORE the "if outcome.mode == MODE_RESEARCH" branch below, so a research
    # run could never reach its own fail-open policy. PMC12444477's research run
    # in 2026-07-28_0919 was filed FAIL/30 gate errors that way while every
    # contract report it produced said ok=true, still_blocking=0,
    # research_blocked=[]. See :func:`_research_gate_flags` for the doctrine and
    # for which four of those 30 errors research mode had manufactured itself.
    #
    # Research mode therefore records the gate errors three ways -- counts, capped
    # warnings, and the review_flags.json flag channel -- and does not fail.
    # Strict mode is untouched: `blocking_gate` is exactly the old `gate_failed`
    # there, and the contract-error leg (`error_count`) is unchanged in BOTH modes
    # because a *_contract_report that still carries errors after
    # export_mode.relax_report has already had its research relaxation applied.
    research_mode = outcome.mode == MODE_RESEARCH
    research_gate_flags: List[Dict[str, str]] = []
    if research_mode and (gate_failed or gate_errors):
        research_gate_flags, gate_warnings = _research_gate_flags(gate_errors)
        outcome.counts["gate_errors_not_blocking"] = len(gate_errors)
        for warning in gate_warnings:
            if warning not in outcome.warnings:
                outcome.warnings.append(warning)
        if gate_failed and not gate_errors:
            # A gate report that says "failed" but lists no errors still has to be
            # visible, or the only trace of it would be a file nobody opens.
            warning = (
                f"{WARN_RESEARCH_GATE_PREFIX}the Stage-3 gate reported failure at "
                f"{_text(verdict.stage) or 'a stage boundary'} "
                "with no itemised errors"
            )
            if warning not in outcome.warnings:
                outcome.warnings.append(warning)

    # Research mode may CONTINUE past a failing gate. It may not restate the
    # failure as a pass: `verdict.failed` above is the factual outcome and is
    # already recorded in the counts, the warnings and review_flags.json. Only
    # the blocking decision is mode-dependent, and only here.
    blocking_gate = gate_failed and not research_mode
    # Counted once. The gate channel and the contract channel overlap -- the same
    # ALAS2 registry error arrives as a gate error AND as the contract error
    # derived from it -- and PMC12856317 was reported as "2 blocking issue(s)"
    # for one row on one path for one reason. Keyed on (code, JSON pointer) and
    # never on message text: gate messages embed entity names, which is what
    # fragments one defect into dozens of codes in failures_by_code.txt. Both
    # channels still block independently; what changes is the number.
    blocking_issues = (
        blocking_findings(verdict, blocking_contract_issues)
        if blocking_gate
        else dedupe_findings(blocking_contract_issues)
    )
    outcome.counts["blocking_issues"] = len(blocking_issues)
    if blocking_gate or error_count:
        _finalize_gate_failure(
            outcome,
            verdict=verdict,
            blocking_issues=blocking_issues,
            codes=codes,
            code_lines=code_lines,
            joined=joined,
            crashed=bool(exceptions),
            stage3_gate=stage3_gate,
            gate_fail=gate_fail,
        )
        return

    # ── 5. Mode-specific deliverable ──────────────────────────────────────
    if outcome.mode == MODE_RESEARCH:
        outcome.stage = STAGE_RESEARCH_REPORT
        try:
            note = _add_research_artifacts(
                at,
                artifacts,
                outcome.artifacts,
                outcome.counts,
                gate_flags=research_gate_flags,
            )
        except Exception:  # noqa: BLE001 -- a report bug must not be called a pass
            _fail(
                outcome,
                status=_STATUS_ERROR,
                kind=KIND_CRASH,
                message="research report generation failed after a clean pipeline run",
                detail=traceback.format_exc(),
                codes=codes,
            )
            return
        if "research_pathway_report.txt" not in outcome.artifacts and not note:
            _fail(
                outcome,
                status=_STATUS_FAIL,
                kind=KIND_UNKNOWN,
                message="no research report was produced and no reason was given",
                detail=joined,
                codes=codes,
            )
            return
        outcome.status = _STATUS_PASS
        outcome.failure_kind = ""
        outcome.message = "research run completed" + (f"; {note}" if note else " with a citation report")
        outcome.detail = joined
        outcome.issue_codes = codes
        if "research_pathway_report.txt" not in outcome.artifacts:
            # The pipeline ran, so this is a pass -- but there is no research
            # deliverable in the folder (no report, no citations, no tiers), and
            # a night of these must never be summarised as ALL GREEN.
            if WARN_NO_RESEARCH_REPORT not in outcome.warnings:
                outcome.warnings.append(WARN_NO_RESEARCH_REPORT)
        return

    # Strict: PWML is gated behind the review step, so the third click is
    # required. Without it there is no .pwml file and nothing to confirm.
    outcome.stage = STAGE_PWML_EXPORT
    try:
        pwml_button = at.button(key=KEY_GENERATE_PWML)
    except Exception as exc:  # noqa: BLE001
        _fail(
            outcome,
            status=_STATUS_FAIL,
            kind=_classify(
                text=joined,
                issue_codes=codes,
                contract_signal=False,
                ambiguous=False,
                no_reactions=False,
                crashed=bool(exceptions),
            ),
            message=(
                f'the app never rendered the "{KEY_GENERATE_PWML}" button ({exc}), '
                "so the review step that unlocks PWML export was not reached"
            ),
            detail=joined,
            codes=codes,
        )
        return

    pwml_button.click()
    timed_out, detail = _run_app(at, budget)
    if timed_out:
        _finalize_timeout(
            outcome,
            message="PWML export did not finish inside the time budget",
            detail=detail,
            codes=codes,
        )
        return

    joined, errors, exceptions = _collect_app_text(at)
    artifacts = _safe_dict(_ss(at, "post_pipeline_artifacts")) or artifacts
    source_key, pwml_result = _find_pwml_result(at, artifacts)
    _add_strict_artifacts(artifacts, pwml_result, outcome.artifacts)

    pwml_codes = list(codes)
    for report_key in ("required_gate_report", "stage3_contract_report", "pwml_contract_report"):
        for issue in _report_errors(pwml_result.get(report_key)):
            code = _issue_code(issue)
            if code and code not in pwml_codes:
                pwml_codes.append(code)

    if not pwml_result:
        _fail(
            outcome,
            status=_STATUS_FAIL,
            kind=_classify(
                text=joined,
                issue_codes=pwml_codes,
                contract_signal=False,
                ambiguous=False,
                no_reactions=False,
                crashed=bool(exceptions),
            ),
            message="no PWML export result was stored, so a strict pass cannot be confirmed",
            detail=joined or "(the app emitted no error)",
            codes=pwml_codes,
        )
        return

    xml = _xml_bytes(pwml_result)
    if not pwml_result.get("ok") or not xml:
        _fail(
            outcome,
            status=_STATUS_FAIL,
            kind=_classify(
                text="\n".join([joined, _text(pwml_result.get("error"))]),
                issue_codes=pwml_codes,
                contract_signal=bool(pwml_codes),
                ambiguous=False,
                no_reactions=False,
                crashed=bool(exceptions),
            ),
            message="PWML export failed: "
            + (_text(pwml_result.get("error")) or "no XML was produced and no error was reported"),
            detail="\n".join(
                part
                for part in (joined, f"pwml result read from {source_key}", _json_text(pwml_result))
                if part
            ),
            codes=pwml_codes,
        )
        return

    _finalize_pwml_export(
        outcome,
        xml=xml,
        pwml_result=pwml_result,
        joined=joined,
        codes=pwml_codes,
    )


__all__ = [
    "DEFAULT_APP_PATH",
    "DEFAULT_APP_TIMEOUT",
    "DEFAULT_TIMEOUT",
    "DETAIL_LIMIT",
    "KIND_CONTRACT",
    "KIND_CRASH",
    "KIND_LLM",
    "KIND_TIMEOUT",
    "LABEL_EXTRACTION_FOCUS",
    "MODE_RESEARCH",
    "MODE_STRICT",
    "PWML_RELEASE_READY_NAME",
    "PWML_REVIEW_REQUIRED_NAME",
    "RESEARCH",
    "RESEARCH_GATE_FLAG_KEY",
    "RESEARCH_GATE_WARN_LIMIT",
    "RUNTIME_SCHEMA_WARN_LIMIT",
    "STRICT",
    "WARN_NO_FOCUS_BOX",
    "WARN_NO_RESEARCH_REPORT",
    "WARN_PWML_WITHHELD_DIAGNOSTIC_ONLY",
    "WARN_RELEASE_STATUS_UNAVAILABLE",
    "WARN_RESEARCH_GATE_PREFIX",
    "WARN_RUNTIME_SCHEMA_PREFIX",
    "RunOutcome",
    "run_one",
]
