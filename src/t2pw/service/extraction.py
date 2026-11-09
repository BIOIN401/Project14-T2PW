"""The "Run pipeline" handler, moved out of ``streamlit_app.py`` step by step (R3).

R3a moved the input assembly: the pasted texts and the uploaded PDFs become the one
``text`` the stages read. R3b moved Stage 0 (:func:`run_stage_zero`), R3c Stage 1 and
the RAG seed (:func:`run_stage_one`), R3d Stage 2 through the session-state publish
(:func:`run_stage_two`) and the headless composition of all four (:func:`run_extraction`).

Every message goes through the ``reporter`` the caller passes (the app passes
``StreamlitReporter(st)``). There is no default: service code never falls back to
Streamlit silently (R2 review F1). Where the app used to call ``st.stop()`` the
function returns a :class:`InputFailure` instead, and the app calls ``st.stop()``.
"""

from __future__ import annotations

import logging
import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, MutableMapping, Optional, Sequence, Tuple

import t2pw.paths as paths  # PROJECT_ROOT read at call time: the harness patches it
from t2pw.extraction.pdf_parser import SKIP_SECTIONS, parse_pdf
# Module imports, so these are looked up at call time as the app script's per-run
# ``from ... import`` did: a test patch of ``preprocessor.preprocess`` or of
# ``extraction_diagnostics.payload_hash`` / ``activate`` / ``current`` reaches them.
from t2pw.pipeline import extraction_diagnostics as _diag
from t2pw.pipeline import preprocessor as _preprocessor
from t2pw.pipeline.preprocessor import (
    describe_preprocess_status,
    is_ambiguous_multi_example_review_context,
    preprocess_was_recovered,
)
from t2pw.service.reporter import Reporter

logger = logging.getLogger(__name__)

# Failure kinds, one per former ``st.stop()`` in the input-assembly code.
KIND_NO_PDF = "no_pdf_uploaded"
KIND_PDF_FAILED = "pdf_extraction_failed"
KIND_NO_TEXT = "no_text"


@dataclass(frozen=True)
class InputFailure:
    kind: str
    message: str


@dataclass(frozen=True)
class AssembledInputs:
    """``text`` for the stages, or the ``failure`` where the app stops the run."""

    text: str = ""
    failure: Optional[InputFailure] = None


def assemble_inputs(
    text_entries: Iterable[str],
    pdfs: Sequence[Tuple[str, bytes]],
    *,
    uses_pdf_input: bool,
    input_mode: str,
    pdf_page_range: Optional[str],
    pdf_skip_refs: bool,
    pdf_ocr: bool,
    reporter: Reporter,
) -> AssembledInputs:
    """Join the pasted texts and the PDFs' text exactly as the app's submit block did.

    ``pdfs`` are ``(file name, file bytes)`` pairs in upload order. Texts come first,
    then each PDF; parts are stripped, empty ones dropped, and joined with a blank line.
    """

    text_parts = [entry.strip() for entry in text_entries if entry.strip()]

    # PDF extraction runs here — outside the form, so uploaded_pdf is accessible
    if uses_pdf_input:
        if not pdfs:
            if input_mode == "Upload PDF":
                message = "Please upload one or more PDFs using the file uploader above."
                reporter.warning(message)
                return AssembledInputs(failure=InputFailure(KIND_NO_PDF, message))
            pdfs = []

        # Parse page range once and apply it to every uploaded PDF.
        _ps, _pe = 1, None
        _pr = (pdf_page_range or "").strip()
        if _pr:
            _parts = _pr.split("-")
            try:
                _ps = int(_parts[0])
                _pe = int(_parts[1]) if len(_parts) > 1 else _ps
            except ValueError:
                reporter.warning(f"Invalid page range '{_pr}'; extracting all pages.")

        _skip = SKIP_SECTIONS if pdf_skip_refs else set()

        for name, data in pdfs:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as _tmp:
                _tmp.write(data)
                _tmp_path = _tmp.name

            try:
                with reporter.spinner(f"Extracting text from {name}..."):
                    _pdf = parse_pdf(
                        _tmp_path,
                        page_start=_ps,
                        page_end=_pe,
                        skip_sections=_skip,
                        enable_ocr_fallback=bool(pdf_ocr),
                    )
            finally:
                try:
                    os.unlink(_tmp_path)
                except Exception:
                    pass

            if _pdf["error"]:
                message = f"PDF extraction failed for {name}: {_pdf['error']}"
                reporter.error(message)
                return AssembledInputs(failure=InputFailure(KIND_PDF_FAILED, message))

            if _pdf["text"].strip():
                text_parts.append(_pdf["text"].strip())

            for _w in _pdf.get("warnings", []):
                reporter.warning(f"{name}: {_w}")

            reporter.success(
                f"{name}: extracted **{_pdf['pages_used']}** of **{_pdf['total_pages']}** pages "
                f"via **{_pdf['method']}**. "
                f"Sections: {', '.join(_pdf['sections'].keys()) or 'none'}"
            )
            _meta = _pdf.get("metadata", {})
            _mp = [p for p in [
                f"Title: {_meta['title']}"     if _meta.get("title")  else "",
                f"Author(s): {_meta['author']}" if _meta.get("author") else "",
                f"DOI: {_meta['doi']}"          if _meta.get("doi")    else "",
            ] if p]
            if _mp:
                reporter.caption(" | ".join(_mp))

            with reporter.expander(f"Preview extracted text from {name} (first 1000 chars)", expanded=False):
                _preview = _pdf["text"][:1000] + ("..." if len(_pdf["text"]) > 1000 else "")
                reporter.text(_preview)

    text = "\n\n".join(text_parts)
    if not text.strip():
        message = "No text to process. Paste text, upload one or more PDFs, or use both."
        reporter.warning(message)
        return AssembledInputs(text=text, failure=InputFailure(KIND_NO_TEXT, message))

    return AssembledInputs(text=text)


# ---------------------------------------------------------------------------
# R3b: Stage 0. Moved from streamlit_app.py: the Stage-0 helpers, the extraction
# diagnostics start/sync, and the submit block from the diagnostics start through
# the post-Stage-0 sync. The ambiguous-review gate stays in the app (R3b.md).
# ---------------------------------------------------------------------------

# Bounded head of the document used when retrying a failed Stage-0 pass. The
# pathway name / organism / key entities live in the title, abstract and intro,
# and ``t2pw.rag.select`` already caps this same ``preprocess`` call at 6000
# chars for exactly this reason.
_PREPROCESS_RETRY_CHARS = 20000


def _has_usable_context(ctx: Any) -> bool:
    """True when Stage 0 produced at least one term usable as a search anchor.

    Mirrors precisely what ``t2pw.rag.acquire.build_query`` needs: without a
    pathway name, an organism, or any compound/protein/gap term, the literature
    query is the empty string and no paper can ever be fetched.
    """
    if not isinstance(ctx, dict):
        return False
    if str(ctx.get("pathway_name") or "").strip():
        return True
    if str(ctx.get("likely_organism") or ctx.get("organism") or "").strip():
        return True
    for key in ("key_compounds", "key_proteins", "gap_terms"):
        value = ctx.get(key)
        if isinstance(value, list) and any(str(v or "").strip() for v in value):
            return True
    return False


def _seed_context_from_user_scope(
    ctx: Any, user_scope: Optional[str]
) -> Optional[Dict[str, Any]]:
    """Derive a usable Stage-0 context from an explicit user scope string.

    Stage 0 (the preprocessor LLM) fails CLOSED to an empty context, which
    silently disables RAG (``build_query`` returns "", so zero papers are ever
    fetched) and leaves extraction unguided. But the pipeline is not actually
    blind in that case: the user routinely types the scope into the extraction
    focus box (e.g. ``"menaquinone biosynthesis in Lactococcus lactis"``). This
    fallback parses that into a ``pathway_name`` and — from a trailing
    ``" in <organism>"`` — a ``likely_organism``, so a flaky preprocessor call
    can no longer nuke a scope the user already provided.

    Returns an augmented COPY of ``ctx`` (all original keys preserved) that
    ``_has_usable_context`` accepts, or ``None`` when there is no usable scope to
    seed from. Heuristic and transparent: the caller surfaces exactly what
    pathway/organism was inferred so a mis-parse is visible, not silent.
    """
    scope = str(user_scope or "").strip()
    if not scope:
        return None
    head, sep, tail = scope.rpartition(" in ")
    # Treat the trailing "in <X>" as the organism only when it is short (organism
    # names are 1–3 words); otherwise keep the whole scope as the pathway name so
    # a phrase like "... in vitro reconstitution" is not mistaken for an organism.
    if sep and head.strip() and 1 <= len(tail.split()) <= 3:
        pathway_name = head.strip()
        organism_name = tail.strip()
    else:
        pathway_name = scope
        organism_name = ""
    seeded = dict(ctx) if isinstance(ctx, dict) else {}
    seeded["pathway_name"] = pathway_name
    if organism_name and not str(seeded.get("likely_organism") or "").strip():
        seeded["likely_organism"] = organism_name
    if not _has_usable_context(seeded):
        return None
    return seeded


def _stage0_context_rank(ctx: Any) -> int:
    """Order two Stage-0 draws so a retry can only ever move the context UPWARD.

    Higher is both more informative and more fail-safe:

    * ``2`` — a well-formed **Case C**: ``document_type == "multi_example_review"``
      with a blank ``selected_example``. This is the Stage-0 prompt's own refusal
      verdict, and it is the strongest state precisely because it is the only one
      that stops the run. It outranks a usable context on purpose: if one draw
      says "this document is a multi-example review and I could not pick a
      target" and another says "the pathway is X", the safe reading is the
      refusal — extracting X from a review that describes six unrelated examples
      is how a mixed-pathway payload gets built. A second draw must never be able
      to disarm the refusal gate.
    * ``1`` — a usable context: ``_has_usable_context`` accepts it, so extraction
      is guided and ``t2pw.rag.acquire.build_query`` can form a non-empty query.
    * ``0`` — neither: Stage 0 failed closed (swallowed LLM error, unparseable
      reply) and left every field blank.

    Note that Case C and rank 0 look IDENTICAL to ``_has_usable_context``: the
    Stage-0 prompt mandates that Case C leave pathway_name / likely_organism /
    key_compounds / key_proteins / gap_terms blank, so the only thing separating a
    deliberate refusal from a crashed call is ``document_type``. That is exactly
    what the 2026-07-28 clobber destroyed (see ``_run_stage_zero_with_retry``).
    """
    if is_ambiguous_multi_example_review_context(ctx):
        return 2
    if _has_usable_context(ctx):
        return 1
    return 0


def _run_stage_zero_with_retry(
    text: str,
    *,
    temperature: float,
    user_task_context: Optional[str],
    preprocess_fn: Optional[Callable[..., Any]] = None,
) -> Dict[str, Any]:
    """Run Stage 0, retrying once on a bounded head only when that can help.

    Two separate defects lived in the inline version of this block, and both were
    load-bearing in run ``runs/2026-07-28_0919``.

    1. *The retry fired on a correct refusal.* Stage 0 fails CLOSED to an empty
       context (a swallowed LLM error or an unparseable reply are both logged, not
       raised), and an empty context silently disables RAG — ``build_query()``
       returns "" so zero papers can ever be fetched. Hence the retry on a bounded
       head of the document: the pathway name / organism / key entities live in
       the title, abstract and intro, and the smaller payload also dodges
       transient failures. But the retry's trigger was ``not
       _has_usable_context(...)``, and the prompt's Case C — a multi-example
       review with no target selected — MANDATES those same fields be blank. A
       CORRECT Case C was therefore indistinguishable from a FAILED Stage 0 and
       fired the retry every time. With ``_PREPROCESS_RETRY_CHARS`` at 20,000 the
       length condition is met by essentially every topic-fetched review (the
       PMC13278307 leg below is 50,377 chars per that run's plan.json), so this
       was not a rare path.

    2. *The retry's result was adopted unconditionally.* ``pathway_context =
       preprocess(head)`` overwrote the first draw whatever came back. When the
       first draw was a Case C and the second was blank, the overwrite dropped
       ``document_type``, which is the one field
       ``is_ambiguous_multi_example_review_context`` requires. That disarms the
       refusal gate further down and lets the run proceed to an unguided
       extraction over a multi-example review. Because the extraction-focus box is
       now filled in by the batch driver, such a run no longer aborts loudly — it
       reports PASS. Leg PMC13278307/strict of ``runs/2026-07-28_0919`` is exactly
       that shape: the paper is the review "An Overview of Mobile Colistin
       Resistance (mcr) Genes in Gram-Negative Bacilli", the focus box said
       ``lipid A biosynthesis``, and the leg came back **pass** in 2098.21s with
       14 reactions and 0 gate errors — every one of them an mcr / PEtN / L-Ara4N
       lipid A MODIFICATION step, and not one reaction of the lipid A
       biosynthesis pathway it was asked for.

    The fix is the two rules below: skip the retry entirely for a deliberate Case
    C (it is deterministic — re-running never helps, only the user naming a target
    does), and, whenever a retry does run, keep its result in a LOCAL and adopt it
    only when ``_stage0_context_rank`` says it is strictly better. A second draw
    can now still rescue a genuinely failed Stage 0, but it can no longer degrade
    the context it was meant to improve.
    R3b: ``preprocess_fn`` is the Stage-0 call. ``None`` means
    ``t2pw.pipeline.preprocessor.preprocess``, looked up at call time; the app passes
    its own ``preprocess`` binding so a patch of ``streamlit_app.preprocess`` still lands.
    """
    preprocess = preprocess_fn if preprocess_fn is not None else _preprocessor.preprocess
    first_context = preprocess(
        text, temperature=temperature, user_task_context=user_task_context
    )
    if _has_usable_context(first_context):
        # Nothing to rescue: extraction is guided and RAG can build a query.
        return first_context
    if is_ambiguous_multi_example_review_context(first_context):
        # Defect 1: a deliberate Case C has blank pathway fields BY DESIGN. The
        # caller's refusal gate needs the document_type this draw carries, so
        # return it untouched rather than spending a second LLM call that can only
        # return the same verdict or something weaker.
        return first_context
    if len(text) <= _PREPROCESS_RETRY_CHARS:
        # The retry re-reads a bounded head; on a document already shorter than
        # that head it would re-send identical input for an identical answer.
        return first_context

    retry_context = preprocess(
        text[:_PREPROCESS_RETRY_CHARS],
        temperature=temperature,
        user_task_context=user_task_context,
    )
    # Defect 2: adopt only on a strict improvement. ``first_context`` is known to
    # be rank 0 here, so this admits both a usable retry and a retry that is
    # itself a well-formed Case C (the head of a review is often enough for Stage
    # 0 to recognise the review it could not classify from the full text), while
    # rejecting a second blank draw.
    if _stage0_context_rank(retry_context) > _stage0_context_rank(first_context):
        return retry_context
    return first_context


#: Session-state key holding this run's diagnostic artifacts as a
#: ``{filename: object}`` map. The batch driver reads it to write those files into
#: the per-leg directory, and it is populated on the FAILURE paths too -- which is
#: the whole point. The 2026-07-28 legs died with ``files: []`` because every
#: artifact hand-off in the app sat downstream of an ``st.stop()``.
EXTRACTION_DIAGNOSTICS_KEY = "extraction_diagnostics"

#: Session-state key holding the directory this invocation's artifacts were
#: written to. Each run gets its own, so the path has to be reported rather than
#: assumed -- "look in tmp/extraction_diagnostics" is no longer an instruction
#: anyone can follow.
EXTRACTION_DIAGNOSTICS_DIR_KEY = "extraction_diagnostics_dir"

#: Where a run's diagnostics land on disk, under the project ``tmp`` tree the app
#: already uses for Stage-1 lock artifacts. Writing to disk is independent of the
#: session-state hand-off on purpose: an interactive user who never runs the batch
#: driver still gets the files, and a leg that dies before the driver reads
#: session state still leaves them behind.
DIAGNOSTICS_DIRNAME = "extraction_diagnostics"


def _start_extraction_diagnostics(source_text: str, *, state: MutableMapping[str, Any]) -> Path:
    """Begin a fresh diagnostics record for one pipeline run. Returns its directory.

    Called once, before Stage 0. Replacing the recorder rather than clearing it
    means diagnostics from a previous paper can never leak into this run's
    artifacts -- a real hazard in the batch, where one process drives the app
    repeatedly.

    ``unique=True`` is not optional here. Streamlit serves every browser session
    from the same process, so two people (or two batch legs sharing an
    interpreter) extracting different papers at the same moment would write
    ``extraction_boundary_report.json`` over each other. The loser would then be
    diagnosed with the winner's evidence, which is worse than having none: it
    looks authoritative. Each invocation gets its own subdirectory, and the path
    is surfaced in the failure message so an operator knows where to look.
    """

    run_id = _diag.payload_hash(source_text)
    recorder = _diag.activate(
        run_id=run_id,
        artifact_dir=paths.PROJECT_ROOT / "tmp" / DIAGNOSTICS_DIRNAME,
        unique=True,
    )
    state[EXTRACTION_DIAGNOSTICS_KEY] = {}
    state[EXTRACTION_DIAGNOSTICS_DIR_KEY] = str(recorder.artifact_dir or "")
    return Path(recorder.artifact_dir) if recorder.artifact_dir else paths.PROJECT_ROOT / "tmp"


def _sync_extraction_diagnostics(*, state: MutableMapping[str, Any]) -> Dict[str, Any]:
    """Publish the diagnostics collected so far into session state.

    Called at every point the run can end -- after Stage 0, after Stage 1 on both
    the pass and the fail path, and after the post-pipeline step -- because
    ``st.stop()`` means "this script run is over" and anything not in session
    state by then is lost. Never raises: a diagnostics hand-off that can abort the
    run defeats its own purpose.
    """

    try:
        artifacts = _diag.current().artifacts()
    except Exception:  # noqa: BLE001 - diagnostics must never break the run
        logger.debug("extraction diagnostics could not be published", exc_info=True)
        return {}
    state[EXTRACTION_DIAGNOSTICS_KEY] = artifacts
    return artifacts


@dataclass(frozen=True)
class StageZeroResult:
    """Stage 0's context and the directory this run's diagnostics are written to."""

    pathway_context: Any
    diagnostics_dir: Path


def run_stage_zero(
    text: str,
    *,
    temperature: float,
    user_task_context: Optional[str],
    state: MutableMapping[str, Any],
    reporter: Reporter,
    preprocess_fn: Optional[Callable[..., Any]] = None,
) -> StageZeroResult:
    """Stage 0 exactly as the app's submit block ran it, up to the ambiguous-review gate.

    Opens this run's extraction diagnostics (writing their keys into ``state``), runs
    the preprocessor with its bounded-head retry, falls back to the user's scope when
    Stage 0 came back empty, reports what happened through ``reporter``, and publishes
    the diagnostics into ``state``. Nothing here stops the run.
    """

    # Opened BEFORE Stage 0, because Stage 0 is one of the boundaries being
    # diagnosed. Everything recorded from here on is flushed to disk as it is
    # recorded, so a run that dies inside Stage 0 has already written
    # stage0_attempts.json by the time the error is rendered.
    _diagnostics_dir = _start_extraction_diagnostics(text, state=state)

    # Preprocessing: lightweight context summary to guide extraction and inference
    with reporter.spinner("Running preprocessor..."):
        # Stage 0 plus its at-most-one bounded-head retry. The retry's guards
        # (skip a deliberate Case C; adopt the second draw only on a strict
        # improvement) live inside the helper because the inline version ran the
        # retry BEFORE the Case-C test below and let it overwrite the context
        # unconditionally — see _run_stage_zero_with_retry for the run that
        # exposed it.
        pathway_context = _run_stage_zero_with_retry(
            text, temperature=temperature, user_task_context=user_task_context,
            preprocess_fn=preprocess_fn,
        )
        # A deliberate Case C ambiguous multi-example review also has blank
        # pathway fields, but that is the prompt's guardrail, not a flaky call:
        # it is deterministic and re-running never helps. The branch below
        # raises its own error, so stay quiet here.
        if not _has_usable_context(pathway_context) and not is_ambiguous_multi_example_review_context(
            pathway_context
        ):
            # Stage 0 came back empty. Before disabling RAG, fall back to the scope
            # the user typed in the extraction-focus box — a flaky preprocessor call
            # should never nuke a scope we already have. Show exactly what was
            # inferred so a mis-parse is visible rather than silent.
            seeded_context = _seed_context_from_user_scope(pathway_context, user_task_context)
            if seeded_context is not None:
                pathway_context = seeded_context
                _seed_org = str(seeded_context.get("likely_organism") or "").strip()
                reporter.info(
                    "Stage 0 returned no usable context, so the pipeline is falling "
                    "back to the scope you provided — pathway "
                    f"\"{seeded_context.get('pathway_name', '')}\""
                    + (f", organism \"{_seed_org}\"" if _seed_org else "")
                    + ". Extraction and multi-paper RAG will use it. Re-run if you'd "
                    "rather the preprocessor try to read the scope from the paper again."
                )
            else:
                reporter.warning(
                    f"Stage 0 failed: {describe_preprocess_status(pathway_context)}\n\n"
                    "It returned no usable context — no pathway name, "
                    "organism, compounds or proteins, and no scope was provided in the "
                    "extraction-focus box to fall back to. Extraction will be unguided, "
                    "and multi-paper RAG cannot build a literature query, so no papers "
                    "will be fetched. Re-run (usually a transient LLM failure), or type "
                    "the pathway/organism into the focus box so the pipeline can proceed "
                    "without the preprocessor."
                )
        elif preprocess_was_recovered(pathway_context):
            # Not an error: the context is real. But it came from a reply that
            # was cut off mid-JSON, so fields the model never wrote are absent.
            reporter.warning(
                f"Stage 0 partially recovered: {describe_preprocess_status(pathway_context)}\n\n"
                "The reply was cut off and repaired, so some context fields "
                "(e.g. candidate examples, key compounds) may be missing. Re-run "
                "if the context below looks incomplete."
            )

    _sync_extraction_diagnostics(state=state)

    return StageZeroResult(pathway_context=pathway_context, diagnostics_dir=_diagnostics_dir)


# ---------------------------------------------------------------------------
# R3c: Stage 1 and the RAG seed. Moved from streamlit_app.py: render_attempts,
# _last_cleaning_report, and the submit block from the Stage-1 spinner through the
# RAG seed's user_task_context fold, stopping before Stage 2 (R3c.md).
# ---------------------------------------------------------------------------

# Failure kinds, one per former ``st.stop()`` in the Stage-1 code.
KIND_STAGE_ONE_FAILED = "stage_one_failed"
KIND_STAGE_ONE_BOUNDARY_ERROR = "stage_one_boundary_error"
KIND_STAGE_ONE_INCOMPLETE = "stage_one_incomplete"


def render_attempts(label: str, attempts: List[Dict[str, Any]], *, reporter: Reporter) -> None:
    with reporter.expander(label, expanded=False):
        for log in attempts:
            status = "success" if not log.get("error") else "retry"
            phase = log.get("phase")
            phase_label = f" ({phase})" if phase else ""
            note = log.get("note")
            note_label = f" [{note}]" if note else ""
            reporter.markdown(f"**Attempt {log['attempt']}{phase_label}** - {status}{note_label}")
            reporter.code(log["raw"], language="json")
            if log.get("error"):
                reporter.caption(log["error"])


def _last_cleaning_report() -> Dict[str, Any]:
    """The most recent Stage-1 cleaning pass, or ``{}``.

    The boundary needs exactly one fact from it -- whether cleaning discarded
    every process row the model returned -- to tell "the model produced nothing"
    apart from "cleaning removed what it produced". The last recorded pass is the
    right one: Stage 1 records per chunk and then once for the merge, and the
    merge is what the boundary is judging.
    """

    # Imported here, not at module level, for the reason given in run_stage_one.
    from t2pw.service.postpipeline import _safe_dict, _safe_list

    try:
        artifacts = _diag.current().artifacts()
    except Exception:  # noqa: BLE001
        return {}
    passes = _safe_list(_safe_dict(artifacts.get(_diag.CLEANING_REPORT_NAME)).get("passes"))
    return _safe_dict(passes[-1]) if passes else {}


#: Refusal of the ambiguous-review gate (Q-R3d-2, moved from the app in R5). The batch
#: driver's and the web layer's kind of the same name.
KIND_AMBIGUOUS_REVIEW_SCOPE = "ambiguous_review_scope"
#: ``run_extraction`` found the run's deadline already gone before the next stage.
KIND_DEADLINE_EXCEEDED = "deadline_exceeded"
#: Where ``run_pathway`` publishes the run's absolute deadline (``time.time()`` seconds).
#: ``state``, not a parameter: ``run_extraction`` and ``run_stage_two`` take none, and the
#: app has no deadline, so its call sites and their behaviour stay exactly as they were.
DEADLINE_STATE_KEY = "pathway_run_deadline"


def ambiguous_review_gate(
    pathway_context: Any,
    *,
    state: MutableMapping[str, Any],
    reporter: Reporter,
) -> Optional[InputFailure]:
    """Stage 0's multi-example-review refusal, or ``None`` when the run may go on.

    The app's gate, moved here unchanged (Q-R3d-2). Stage 1 would refuse the same context
    before its first LLM call, but only with a generic failure and without these writes.
    """

    if not is_ambiguous_multi_example_review_context(pathway_context):
        return None
    candidate_examples = pathway_context.get("candidate_examples", [])
    state["pipeline_ready"] = False
    state["pathway_context"] = pathway_context
    message = (
        "multi_example_review detected with no selected_example. "
        "Extraction skipped to prevent mixed-pathway output."
    )
    state["pipeline_error"] = {
        "status": KIND_AMBIGUOUS_REVIEW_SCOPE,
        "message": message,
        "candidate_examples": candidate_examples,
    }
    reporter.error(
        "Ambiguous review scope: this looks like a multi-example review, but no target example was selected. "
        "Name the example you want in the \"Optional extraction focus / task context\" box above "
        "(one of the candidate example names below works), then re-run. "
        "Re-running without naming a target will produce this same result."
    )
    if candidate_examples:
        reporter.subheader("Candidate examples")
        reporter.json(candidate_examples)
    warning = pathway_context.get("warning")
    if isinstance(warning, str) and warning.strip():
        reporter.warning(warning.strip())
    return InputFailure(KIND_AMBIGUOUS_REVIEW_SCOPE, message)


def _deadline_failure(state: MutableMapping[str, Any]) -> Optional[InputFailure]:
    """The stop for a run with no time left. Never set on the app's path."""

    deadline = state.get(DEADLINE_STATE_KEY)
    if deadline is not None and time.time() >= float(deadline):
        return InputFailure(KIND_DEADLINE_EXCEEDED, "the run's deadline passed before the next stage started")
    return None


@dataclass(frozen=True)
class StageOneResult:
    """What the rest of the submit block reads after Stage 1 and the RAG seed, or the
    ``failure`` where the app stops the run."""

    stage_one: Any = None
    chunk_details: Any = None
    stage_one_in_scope: Any = None
    out_of_scope_removed_reactions: Any = None
    rag_result: Any = None
    user_task_context: Optional[str] = None
    failure: Optional[InputFailure] = None


def run_stage_one(
    text: str,
    *,
    pathway_context: Any,
    user_task_context: Optional[str],
    diagnostics_dir: Path,
    enable_chunking: bool,
    chunk_size: Any,
    chunk_overlap: Any,
    extract_attempts: Any,
    temperature: float,
    extract_tokens: Any,
    rag_incomplete_flag: Any,
    state: MutableMapping[str, Any],
    reporter: Reporter,
) -> StageOneResult:
    """Stage 1, its boundary, the out-of-scope filter and the RAG seed, exactly as the
    app's submit block ran them, up to Stage 2.

    ``user_task_context`` in the result is the value Stage 2 reads: the input, with the
    RAG evidence folded in when the seed returned some.
    """

    # Imported here, not at module level, so that importing this module loads exactly
    # what it loaded before R3c. They are module objects, so every name is looked up at
    # call time, as the app script's per-run ``from ... import`` was: a test patch of
    # ``pipeline.run_stage_one_with_chunking``, ``stage_one_boundary.settle_stage_one``,
    # ``draft_graph.build_draft_graph``, ``qa_graph.generate_qa_report`` or
    # ``rag_orchestration.maybe_run_rag`` on its source module reaches the call.
    import t2pw.config as _config
    from t2pw.pipeline import draft_graph as _draft_graph
    from t2pw.pipeline import export_mode as _export_mode
    from t2pw.pipeline import pipeline as _pipeline
    from t2pw.pipeline import qa_graph as _qa_graph
    from t2pw.pipeline import stage_one_boundary as _stage_one_boundary
    from t2pw.pipeline.stage_contracts import StageContractError

    # Stage 1: strict extraction with auto-repair
    try:
        with reporter.spinner("Running Stage 1 extraction..."):
            stage_one, chunk_details = _pipeline.run_stage_one_with_chunking(
                text,
                pathway_context=pathway_context,
                user_task_context=user_task_context,
                artifact_dir=paths.PROJECT_ROOT / "tmp",
                enable_chunking=enable_chunking,
                chunk_word_limit=int(chunk_size),
                chunk_overlap=int(chunk_overlap),
                max_attempts=int(extract_attempts),
                temperature=temperature,
                max_tokens=int(extract_tokens),
            )
            # The boundary is settled rather than merely checked: a registry row
            # the extraction referenced but never declared is reconstructed
            # deterministically, and rows the contract rejects get a bounded,
            # localized repair -- neither of which regenerates the pathway. See
            # t2pw.pipeline.stage_one_boundary. When it cannot be settled the
            # payload still comes back and the run reports an INCOMPLETE outcome
            # instead of a pass; nothing is invented to satisfy the contract.
            _boundary = _stage_one_boundary.settle_stage_one(
                stage_one,
                mode=_export_mode.coerce_mode(state.get("export_mode")),
                cleaning_report=_last_cleaning_report(),
            )
            stage_one = _boundary.payload
            state["stage_one_boundary"] = _boundary.to_summary()
    except _pipeline.PipelineFailure as failure:
        _sync_extraction_diagnostics(state=state)
        message = f"Extraction failed: {failure}"
        reporter.error(message)
        render_attempts("Stage 1 attempts", failure.attempts, reporter=reporter)
        return StageOneResult(failure=InputFailure(KIND_STAGE_ONE_FAILED, message))
    except StageContractError as failure:
        _sync_extraction_diagnostics(state=state)
        message = f"Extraction boundary failed: {failure}"
        reporter.error(message)
        reporter.json(failure.report)
        return StageOneResult(failure=InputFailure(KIND_STAGE_ONE_BOUNDARY_ERROR, message))

    _sync_extraction_diagnostics(state=state)
    if not _boundary.ok:
        # Explicitly incomplete, not a silent pass and not a fabricated pathway.
        # The artifacts written above are the deliverable for this run.
        message = (
            "Extraction boundary failed and could not be recovered: "
            f"{_boundary.incomplete_reason}"
        )
        reporter.error(message)
        reporter.info(
            "Localized repair and deterministic registry reconstruction were both "
            "applied; the payload below is as far as the run got. No reaction was "
            "invented to satisfy the contract. The diagnostics for this run "
            f"(stage0_attempts.json, extraction_boundary_report.json, "
            f"cleaning_report.json) are in {diagnostics_dir}."
        )
        reporter.json(_boundary.to_summary())
        if _boundary.failure is not None:
            reporter.json(_boundary.failure.report)
        return StageOneResult(failure=InputFailure(KIND_STAGE_ONE_INCOMPLETE, message))
    if _boundary.reconstruction:
        reporter.info(
            f"Reconstructed {_boundary.reconstruction.get('shell_count', 0)} unresolved "
            "entity shell(s) for participants named by reactions but missing from the "
            "registry. They carry no identifier and no function; every identity gate "
            "downstream still applies to them."
        )

    # Stage 1 wrote its raw output (incl. out-of-scope reactions) to disk and the
    # locked-reaction manifest above, before this filter runs; the manifest never
    # locked out-of-scope reactions in the first place. Drop them here so Stage 2
    # does not spend effort linking modifiers to reactions that are about to be
    # removed, and so out-of-scope material never reaches export.
    stage_one_in_scope, out_of_scope_removed_reactions = _pipeline.filter_out_of_scope_reactions(stage_one)
    if out_of_scope_removed_reactions:
        reporter.info(
            f"Removed {len(out_of_scope_removed_reactions)} out-of-scope reaction(s) "
            f"before Stage 2: {', '.join(out_of_scope_removed_reactions)}"
        )

    # ── RAG orchestration (seam S5) — WIRING ONLY; identical path when off ───
    # Guarded on the deploy switch (RAG_ENABLED) ONLY. With RAG_ENABLED off,
    # nothing here runs and user_task_context / the flow stay exactly as today —
    # the standard single-paper pipeline, byte-for-byte. With RAG_ENABLED on, the
    # decision of WHETHER to run the chain is delegated to t2pw.rag triage inside
    # maybe_run_rag (the separation invariant): the UI toggle is passed through as
    # user_flag, so an ON toggle force-runs RAG while an OFF toggle lets
    # should_run_rag auto-decide from the scope/gap guardrails. Only when triage
    # says run does maybe_run_rag run the whole R1–R5 chain and return its
    # outputs; the app then folds the retrieved evidence through the existing S1
    # (user_task_context) and S2 (the audit's retrieval_context, via
    # session_state) params, and hands the synthesized Payload to the
    # post-pipeline path (S3) below.
    rag_result = None
    if _config.rag_config()["enabled"]:
        from t2pw.service import rag_orchestration as _rag_orchestration  # only with RAG on

        _rag_seed_graph = _draft_graph.build_draft_graph(stage_one_in_scope)
        rag_result = _rag_orchestration.maybe_run_rag(
            pathway_context=pathway_context,
            user_flag=bool(rag_incomplete_flag),
            seed_payload=stage_one_in_scope,
            reports={"qa_graph": _qa_graph.generate_qa_report(_rag_seed_graph, stage_one_in_scope)},
            seed_text=text,
            reporter=reporter,
        )
        if rag_result is not None:
            state["rag_result"] = rag_result
            if rag_result.evidence_context:
                user_task_context = "\n\n".join(
                    part
                    for part in (user_task_context, rag_result.evidence_context)
                    if part
                )  # S1
                state["rag_evidence_context"] = rag_result.evidence_context  # S2

    return StageOneResult(
        stage_one=stage_one,
        chunk_details=chunk_details,
        stage_one_in_scope=stage_one_in_scope,
        out_of_scope_removed_reactions=out_of_scope_removed_reactions,
        rag_result=rag_result,
        user_task_context=user_task_context,
    )


# ---------------------------------------------------------------------------
# R3d: Stage 2 to the end of the run. Moved from streamlit_app.py:
# emit_rag_loop_messages, rag_loop_session_record, and the submit block from
# ``final_payload = stage_one_in_scope`` through the session-state publish. The
# quarantine reset and the token stats stay in the app, after the call (R3d.md).
# ---------------------------------------------------------------------------

# Failure kind of the former ``st.stop()`` in the Stage-2 code.
KIND_STAGE_TWO_FAILED = "stage_two_failed"


def emit_rag_loop_messages(messages: Any, *, reporter: Reporter) -> None:
    """Render the loop's collected UI lines. Best-effort UI only."""

    from t2pw.service.postpipeline import _safe_list  # function-local, as in _last_cleaning_report

    for kind, body in _safe_list(messages):
        if kind == "info":
            reporter.info(body)
        elif kind == "warning":
            reporter.warning(body)
        elif kind == "error":
            reporter.error(body)
        elif kind == "json":
            reporter.json(body)


def rag_loop_session_record(record: Dict[str, Any]) -> Dict[str, Any]:
    """The JSON-safe half of the loop record, for session state and diagnostics.

    ``payload`` is deliberately excluded: the graph the loop settled on is
    ``final_payload``, and a second copy under another key would be a second
    authority for the same fact.
    """

    from t2pw.service.postpipeline import _safe_dict  # function-local, as in _last_cleaning_report

    return {
        key: value for key, value in _safe_dict(record).items()
        if key not in {"payload", "messages"}
    }


@dataclass(frozen=True)
class ExtractionResult:
    """The run's Stage-0 context and final payload, or the ``failure`` where the app
    stops the run. Everything else the run produced is in ``state``, under the keys the
    app's session state has always used (``stage_one``, ``stage_two``, ``draft_graph``...)."""

    pathway_context: Any = None
    final_payload: Any = None
    failure: Optional[InputFailure] = None


def _rag_deadline_kwargs(state: MutableMapping[str, Any]) -> Dict[str, float]:
    """``deadline_seconds`` for the RAG rounds, only when a deadline was set (the app has none)."""

    from t2pw.service.rag_orchestration import rag_loop_deadline_seconds  # the usual budget, as the driver path gives it

    deadline = state.get(DEADLINE_STATE_KEY)
    if deadline is None:
        return {}
    return {"deadline_seconds": max(1.0, min(rag_loop_deadline_seconds(), float(deadline) - time.time()))}


def run_stage_two(
    text: str,
    stage_one_result: StageOneResult,
    *,
    pathway_context: Any,
    run_inference: Any,
    enable_chunking: bool,
    chunk_size: Any,
    chunk_overlap: Any,
    infer_rounds: Any,
    infer_attempts: Any,
    temperature: float,
    infer_tokens: Any,
    rag_incomplete_flag: Any,
    export_mode: Any,
    state: MutableMapping[str, Any],
    reporter: Reporter,
) -> ExtractionResult:
    """Stage 2, both ``merge_additions`` sites, the RAG rounds, the draft graph and the
    session-state publish, exactly as the app's submit block ran them after Stage 1.

    ``stage_one_result`` is :func:`run_stage_one`'s result (no failure). ``export_mode`` is
    the app's own value (the RAG rounds read it); ``pwml_strict_db`` is read from ``state``.
    """

    # Function-local for the reason given in run_stage_one: every callee is looked up at
    # call time, so a patch of ``pipeline.run_stage_two_with_feedback_loop`` or
    # ``pipeline.build_and_save_draft_graph`` on its source module reaches the call.
    # ``merge_additions`` is imported by name because C-060a lifts the two merge statements
    # below by AST and runs them with ``merge_additions`` as a plain name; the import still
    # runs on every call, so a patch of ``pipeline.merge_additions`` reaches it too.
    from t2pw.pipeline import pipeline as _pipeline
    from t2pw.pipeline.pipeline import merge_additions
    from t2pw.service import postpipeline as _postpipeline

    # What the app used to rebind from run_stage_one's result (R3c).
    stage_one = stage_one_result.stage_one
    chunk_details = stage_one_result.chunk_details
    stage_one_in_scope = stage_one_result.stage_one_in_scope
    out_of_scope_removed_reactions = stage_one_result.out_of_scope_removed_reactions
    rag_result = stage_one_result.rag_result
    user_task_context = stage_one_result.user_task_context

    final_payload = stage_one_in_scope
    stage_two = None
    stage_two_chunks: List[Dict[str, Any]] = []
    stage_two_rounds: List[Dict[str, Any]] = []
    qa_hints = None

    # Stage 2: inference/enrichment + auto-repair
    if run_inference:
        try:
            with reporter.spinner("Running Stage 2 inference..."):
                stage_two, stage_two_chunks, stage_two_rounds = _pipeline.run_stage_two_with_feedback_loop(
                    text,
                    stage_one_in_scope,
                    chunk_details=chunk_details,
                    pathway_context=pathway_context,
                    user_task_context=user_task_context,
                    qa_rounds=int(infer_rounds),
                    enable_chunking=enable_chunking,
                    chunk_word_limit=int(chunk_size),
                    chunk_overlap=int(chunk_overlap),
                    max_attempts=int(infer_attempts),
                    temperature=temperature,
                    max_tokens=int(infer_tokens),
                )
        except _pipeline.PipelineFailure as failure:
            message = f"Inference stage failed: {failure}"
            reporter.error(message)
            label = f"{failure.stage} attempts" if getattr(failure, "stage", None) else "Stage 2 attempts"
            render_attempts(label, failure.attempts, reporter=reporter)
            return ExtractionResult(failure=InputFailure(KIND_STAGE_TWO_FAILED, message))

        qa_hints = stage_two.get("qa_hints", {}) if isinstance(stage_two, dict) else {}
        # The C-060 gate's advisory phase gates on isinstance(context,
        # PathwayContext) and `pathway_context` here is the Stage-0 DICT, so
        # forwarding it would type-check and do nothing. The factory builds the
        # real frozen context from the one derivation maybe_run_rag also uses. An
        # empty Stage 0 yields PathwayContext(""), which the gate already coerces
        # a missing context to, so no new refusal path appears. `seed_text` is
        # deliberately not supplied -- see the pipeline.py site for why.
        from t2pw.pipeline.entity_admission import pathway_context_from_stage_zero

        final_payload = merge_additions(
            stage_one_in_scope,
            stage_two if isinstance(stage_two, dict) else {},
            pathway_context=pathway_context_from_stage_zero(pathway_context),
            source_text=text,
        )

    # S3 — inject R5's synthesized reactions/entities INTO the seed payload rather
    # than REPLACING it, ONE ROUND AT A TIME. A no-op with RAG off (rag_result is
    # None): final_payload is untouched and t2pw.rag.controller is never imported.
    #
    # RAG's ``to_payload`` emits a thin, reaction-only payload MISSING the rich
    # Stage-1 shape the post-pipeline assumes (no element_locations, no per-reaction
    # compartment, no per-entity mapping_meta, evidence as a LIST, participants as
    # dicts). Replacing the seed with it crashed the downstream audit/DB-mapping
    # with ``'NoneType' object is not subscriptable``. Instead, conform the RAG
    # payload to a Stage-2 additions envelope and MERGE it into the seed with the
    # SAME machinery Stage 2 uses (merge_additions) — so the payload entering the
    # post-pipeline keeps the seed's rich Stage-1 shape (compartments defaulted by
    # the existing default_compartment="cell" path) with RAG reactions slotted in.
    #
    # That conform/merge/cleanup/schema-gate sequence is unchanged; what changed is
    # that it is now the body of ONE ROUND of ``t2pw.rag.controller.run_rag_loop``
    # (``run_rag_rounds`` above) instead of a single straight-line pass. D-008
    # (LOCKED) and PRODUCT_CONTRACT §10: every RAG round must re-enter normalization,
    # mapping, gates, persistence and classification, the loop must be deadline-aware,
    # and it must checkpoint before each round. The controller enforces all of that
    # structurally; this site supplies only ``stages=``, ``round_runner=``,
    # ``deadline=`` and ``checkpoint=``, per the separation invariant at :161.
    #
    # WHY THE CHAIN STAYS AT :5514 AND THE MERGE HAPPENS HERE. Round 0's chain must
    # run BEFORE Stage 2, because its ``evidence_context`` is folded into Stage 2's
    # prompt through S1/S2; the merge must run AFTER it, because the merge base is
    # Stage 1 PLUS Stage 2 whenever inference ran. So round 0 REUSES the result
    # computed there, and only rounds 1..N-1 re-enter the chain — against their own
    # graph and their own recomputed reports.
    #
    # ``RAG_LOOP_MAX_ROUNDS`` defaults to 1, so an unchanged environment runs exactly
    # one round and produces exactly today's merge, today's messages and today's
    # final_payload.
    #
    # THE MERGE STATEMENT STAYS HERE, and the loop calls it. C-060a's wiring test
    # lifts the two ``merge_additions`` statements out of this file BY AST and
    # executes the real source bytes, to prove each passes a real PathwayContext
    # (tests/test_c060a_requested_pathway_wiring.py:68-104; both must sit at or
    # above OWNED_FLOOR = 790, Stage 2's first). Writing the RAG merge inside a
    # helper defined above would put it out of that instrument's reach and disarm
    # C-060a's protection without changing a line of C-060a's test.
    # R3d: both statements moved here with the rest of the block, verbatim, so C-060a
    # has to read this file instead of streamlit_app.py (R3d.md, open question 1).
    def _rag_merge_site(final_payload, rag_envelope, pathway_context):
        # The RAG leg runs merge_additions a SECOND time, so the same request has
        # to reach the gate on this pass too. Without it the rows RAG adds are
        # screened with no context while the Stage-2 rows were screened with one,
        # and the two passes would disagree about the same species.
        from t2pw.pipeline.entity_admission import pathway_context_from_stage_zero

        merged = merge_additions(
            final_payload,
            rag_envelope,
            pathway_context=pathway_context_from_stage_zero(pathway_context),
        )
        return merged

    if rag_result is not None and rag_result.synthesized and rag_result.payload:
        from t2pw.service import rag_orchestration as _rag_orchestration  # only with RAG on

        rag_loop_record = _rag_orchestration.run_rag_rounds(
            final_payload,
            first_result=rag_result,
            merge_site=_rag_merge_site,
            pathway_context=pathway_context,
            user_flag=bool(rag_incomplete_flag),
            seed_text=text,
            export_mode=export_mode,
            strict_db=bool(state.get("pwml_strict_db", True)),
            reporter=reporter,
            state=state,
            **_rag_deadline_kwargs(state),
        )
        final_payload = rag_loop_record["payload"]
        emit_rag_loop_messages(rag_loop_record["messages"], reporter=reporter)
        # Additive only. ``RagOrchestrationResult`` is a bare SimpleNamespace read
        # through ``getattr(..., default)``, and ``payload`` keeps its meaning — the
        # PRE-MERGE synthesized payload the citation report reads. The loop record is
        # a NEW field beside it, never a redefinition of that one.
        state["rag_loop"] = rag_loop_session_record(rag_loop_record)
        try:
            rag_result.loop = state["rag_loop"]
        except Exception:  # noqa: BLE001 - a diagnostic attachment is never fatal
            pass

    stage2_preservation_report = _postpipeline._write_reaction_preservation_report("after_stage2", final_payload)

    draft_graph, qa_report, reaction_summary = _pipeline.build_and_save_draft_graph(final_payload)
    state["draft_graph"] = draft_graph.to_dict()
    state.pop("draft_graph_png_bytes", None)
    state.pop("draft_graph_render_error", None)
    state["qa_report"] = qa_report
    state["reaction_summary"] = reaction_summary

    state["pipeline_ready"] = True
    state["run_inference_enabled"] = bool(run_inference)
    state["pathway_context"] = pathway_context
    state["user_task_context"] = user_task_context
    state["stage_one"] = stage_one
    state["extraction_diagnostics_recorder"] = _diag.current()
    state["out_of_scope_removed_reactions"] = out_of_scope_removed_reactions
    state["chunk_details"] = chunk_details
    state["stage_two"] = stage_two
    state["stage_two_chunks"] = stage_two_chunks
    state["stage_two_rounds"] = stage_two_rounds
    state["qa_hints"] = qa_hints
    state["final_payload"] = final_payload
    state["final_payload_snapshot"] = final_payload
    state["pipeline_source_text"] = text
    state["reaction_preservation_after_stage2"] = stage2_preservation_report
    state.pop("post_pipeline_artifacts", None)

    return ExtractionResult(pathway_context=pathway_context, final_payload=final_payload)


@dataclass(frozen=True)
class ExtractionRequest:
    """Everything the app's "Run pipeline" form hands the run: the widget values, raw,
    as the app reads them. No field has a default: the caller states every setting."""

    text_entries: Sequence[str]
    pdfs: Sequence[Tuple[str, bytes]]
    uses_pdf_input: bool
    input_mode: str
    pdf_page_range: Optional[str]
    pdf_skip_refs: bool
    pdf_ocr: bool
    user_task_context: Optional[str]
    export_mode: Any
    temperature: float
    run_inference: Any
    enable_chunking: bool
    chunk_size: Any
    chunk_overlap: Any
    extract_attempts: Any
    infer_attempts: Any
    infer_rounds: Any
    extract_tokens: Any
    infer_tokens: Any
    rag_incomplete_flag: Any


def run_extraction(
    request: ExtractionRequest,
    *,
    reporter: Reporter,
    state: MutableMapping[str, Any],
) -> ExtractionResult:
    """The app's "Run pipeline" submit block without Streamlit: input assembly, Stage 0,
    Stage 1 and the RAG seed, Stage 2 and the publish into ``state``, composed in the
    app's order. Each step's failure is returned as it is; nothing else is added.

    The app does not call this: it keeps calling the steps itself, because
    ``reset_refinement_state()`` stays in the app, as do the quarantine reset and the token
    stats after Stage 2. Every step's failure is returned; ``state[DEADLINE_STATE_KEY]`` (absent
    on the app's path) adds a stop before Stage 1 and Stage 2 once it has passed.
    """

    # As the app: the export-mode radio writes the session copy that Stage 1 reads.
    state["export_mode"] = request.export_mode
    user_task_context = (request.user_task_context or "").strip() or None

    assembled = assemble_inputs(
        request.text_entries,
        request.pdfs,
        uses_pdf_input=request.uses_pdf_input,
        input_mode=request.input_mode,
        pdf_page_range=request.pdf_page_range,
        pdf_skip_refs=request.pdf_skip_refs,
        pdf_ocr=request.pdf_ocr,
        reporter=reporter,
    )
    if assembled.failure is not None:
        return ExtractionResult(failure=assembled.failure)
    text = assembled.text

    stage_zero = run_stage_zero(
        text,
        temperature=request.temperature,
        user_task_context=user_task_context,
        state=state,
        reporter=reporter,
    )
    gated = ambiguous_review_gate(stage_zero.pathway_context, state=state, reporter=reporter)
    if gated is not None:
        return ExtractionResult(failure=gated)
    expired = _deadline_failure(state)
    if expired is not None:
        return ExtractionResult(failure=expired)
    stage_one = run_stage_one(
        text,
        pathway_context=stage_zero.pathway_context,
        user_task_context=user_task_context,
        diagnostics_dir=stage_zero.diagnostics_dir,
        enable_chunking=request.enable_chunking,
        chunk_size=request.chunk_size,
        chunk_overlap=request.chunk_overlap,
        extract_attempts=request.extract_attempts,
        temperature=request.temperature,
        extract_tokens=request.extract_tokens,
        rag_incomplete_flag=request.rag_incomplete_flag,
        state=state,
        reporter=reporter,
    )
    if stage_one.failure is not None:
        return ExtractionResult(failure=stage_one.failure)
    expired = _deadline_failure(state)
    if expired is not None:
        return ExtractionResult(failure=expired)
    return run_stage_two(
        text,
        stage_one,
        pathway_context=stage_zero.pathway_context,
        run_inference=request.run_inference,
        enable_chunking=request.enable_chunking,
        chunk_size=request.chunk_size,
        chunk_overlap=request.chunk_overlap,
        infer_rounds=request.infer_rounds,
        infer_attempts=request.infer_attempts,
        temperature=request.temperature,
        infer_tokens=request.infer_tokens,
        rag_incomplete_flag=request.rag_incomplete_flag,
        export_mode=request.export_mode,
        state=state,
        reporter=reporter,
    )
