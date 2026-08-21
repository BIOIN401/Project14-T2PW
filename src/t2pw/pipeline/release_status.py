"""Release status as a FIRST-CLASS runtime classification (D-002, PRODUCT_CONTRACT 4/7/11).

Before this module the question "is this run releasable?" had no answer object. It
was re-derived at each consumer from whatever happened to be nearby -- a boolean
(``quarantine_ok``), an exit status (``status == "pass"``), or an artifact filename
(``pathway.pwml`` exists). Each of those collapses facts the product contract says
must never be collapsed, and none of them can carry the evidence D-002 requires a
below-threshold run to record.

Two shapes live here.

:class:`CoverageVerdict`
    What :func:`t2pw.pipeline.strict_quarantine.evaluate_core_coverage` returns.
    A ``dict`` subclass on purpose: every existing consumer keeps working,
    ``json.dumps`` produces the SAME BYTES, ``deepcopy`` keeps the class, and
    ``==`` against a plain dict still holds -- so the pinned coverage documents
    (``evidence/c011_freeze_seam_before.json``, the replay baseline) do not move.
    What it adds is a NAMED TYPE with semantic accessors, so a consumer asks
    ``verdict.below_coverage_minimum`` instead of string-matching a reason line.
    That is the return-shape change ``MASTER_PLAN.md:230`` records as the reason
    C-053, C-054, C-056b and C-057 cannot build until this lands.

:class:`ReleaseStatus`
    The classification, with PRODUCT_CONTRACT 11's five states kept apart on
    separate fields. ``semantic_evaluation`` is a THREE-valued string, never a
    bool, because "not evaluated" is never ``False``.

What this module deliberately does NOT do: it does not **evaluate** semantics --
that is ``bench.semantic_production``'s, and this module never imports it. What
C-056a added is the *input* and the gating: :func:`semantic_verdict` reduces an
already-computed report to one of PRODUCT_CONTRACT 11's three semantic states, and
:func:`classify_release_status` accepts that verdict and applies D-042's one-step
cap. A caller that passes nothing still gets ``SEMANTIC_NOT_EVALUATED`` with a
stated reason, exactly as before. It also does not name artifacts (D-004 /
C-053): structured status is authoritative and a filename is never an input here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

#: PRODUCT_CONTRACT 4 output states. There is deliberately no fourth.
RELEASE_READY = "release_ready"
REVIEW_REQUIRED = "review_required"
DIAGNOSTIC_ONLY = "diagnostic_only"
RELEASE_STATES: Tuple[str, ...] = (RELEASE_READY, REVIEW_REQUIRED, DIAGNOSTIC_ONLY)

#: PRODUCT_CONTRACT 11 semantic outcomes. THREE values, never a bool: a caller
#: that stored this as ``True``/``False`` would have to invent one of these.
SEMANTIC_PASSED = "passed"
SEMANTIC_FAILED = "failed"
SEMANTIC_NOT_EVALUATED = "not_evaluated"

#: Why no semantic verdict is attached yet. A reason, not a failure. Still the
#: default: a caller that passes no semantic input gets exactly the pre-C-056a
#: record, which is what keeps every unwired consumer byte-identical.
SEMANTIC_INPUT_NOT_WIRED = (
    "no semantic verdict reached this classification: the caller passed no "
    "semantic input, so this is a MISSING INPUT, NOT a semantic failure and NOT "
    "a pass"
)

#: No report reached the classifier at all -- distinct from a report that ran and
#: could not evaluate anything.
SEMANTIC_NO_REPORT = "no semantic report was produced for this run"

#: A report ran, but every GATING check was inapplicable, so there is no semantic
#: evidence either way. PRODUCT_CONTRACT 11 forbids calling that a pass: a verdict
#: with nothing behind it is exactly the "not performed" state, and reporting it
#: as ``passed`` would let an unevaluable run count as semantically confirmed.
SEMANTIC_NO_GATING_CHECK_EVALUABLE = (
    "no gating semantic check could be evaluated on this run; the gating set is "
    "closed and every member was inapplicable"
)

#: The semantic checks that may GATE the release status. The set is CLOSED (D-039
#: section 3); its cardinality is deliberately NOT restated here, nor in any
#: SHIPPED string, because a hard-coded count drifts on the next ratified widening
#: and the test below already forces every addition to be deliberate. Read it from
#: ``len(SEMANTIC_GATING_CHECKS)``.
#:
#: ``CHECK_PLACEHOLDER_IDENTITY`` never gates (PRODUCT_CONTRACT
#: 13 / TRAP-3 -- it is explicitly non-adjudicating); ``CHECK_SUPPORTED_REACTIONS``
#: is always inapplicable in production; and ``CHECK_SOURCE_CARRIER`` and
#: ``CHECK_CONNECTED_CORE`` are RECORDED BUT NON-GATING -- the first documents
#: itself as hygiene that "deliberately does NOT claim the reaction is supported",
#: so blocking a biological release on it would misuse it, and the second
#: duplicates a floor the coverage verdict already enforced at this same seam.
#:
#: LOCAL RESTATEMENT of ``bench.semantic``'s ``CHECK_*`` names, the same house
#: pattern ``semantic_production._contains_token`` uses for ``goldset``: this is
#: ``t2pw.pipeline`` and a module-level ``t2pw.bench`` import here would invert the
#: layering for every importer of this module, not just the one seam that is
#: authorized to. Kept in step BY TEST, not by comment --
#: ``tests/test_semantic_release_gating.py`` asserts this tuple equals the named
#: constants, member by member, and that the set is CLOSED.
SEMANTIC_GATING_CHECKS: Tuple[str, ...] = (
    "requested_pathway_anchors_present",       # bench.semantic.CHECK_ANCHORS
    "organism_compatible",                     # bench.semantic.CHECK_ORGANISM
    "no_real_id_or_name_conflict",             # bench.semantic.CHECK_ID_CONFLICT
    "no_rejected_rag_reaction_reintroduced",   # bench.semantic.CHECK_RAG_REINTRODUCTION
    "actor_named_in_its_own_cited_span",       # bench.semantic.CHECK_ACTOR_EVIDENCE
)

#: Reason vocabulary. Grouped so a report can say WHY without parsing prose.
REASON_PIPELINE_DID_NOT_EXECUTE = "pipeline_did_not_execute"
REASON_STRICT_GATES_BLOCKED = "strict_technical_gates_blocked_export"
REASON_SERIALIZATION_REQUIRES_INVENTION = "serialization_would_require_invention"
REASON_NO_DEFENSIBLE_CONNECTED_CORE = "no_defensible_connected_core"
REASON_COVERAGE_NOT_EVALUATED = "requested_core_coverage_not_evaluated"
REASON_SEMANTIC_EVALUATION_FAILED = "semantic_evaluation_failed"

#: The coverage reason prefixes ``evaluate_core_coverage`` emits, named once so no
#: consumer has to re-type the string it string-matches on.
COVERAGE_REASON_EMPTY = "no_surviving_process"
COVERAGE_REASON_COUNT_BELOW_MINIMUM = "core_process_count_below_minimum"
COVERAGE_REASON_BELOW_MINIMUM = "requested_core_coverage_below_minimum"


def _as_tuple(value: Any) -> Tuple[str, ...]:
    if isinstance(value, str) or not isinstance(value, Sequence):
        return ()
    return tuple(str(item) for item in value)


class CoverageVerdict(dict):
    """The core-coverage decision, as a typed view over the SAME serialized dict.

    Subclassing ``dict`` rather than wrapping it is the whole design: the mapping
    content is byte-for-byte what it always was, so ``quarantine_coverage`` in
    every artifact, fixture and pinned baseline is unchanged, and the accessors
    below are derived rather than stored, so ``deepcopy``/``pickle``/``json``
    cannot see a difference either.
    """

    # -- raw facts, named -------------------------------------------------
    @property
    def declared(self) -> bool:
        """Whether a requested core was declared at all, i.e. whether relevance
        was judgeable. Distinct from "declared and nothing matched"."""

        return bool(self.get("requested_core_declared"))

    @property
    def coverage_ratio(self) -> float:
        try:
            return float(self.get("coverage_ratio") or 0.0)
        except (TypeError, ValueError):
            return 0.0

    @property
    def reasons(self) -> Tuple[str, ...]:
        return _as_tuple(self.get("reasons"))

    @property
    def minimum_core_satisfied(self) -> bool:
        return bool(self.get("minimum_core_satisfied"))

    @property
    def surviving_processes(self) -> int:
        try:
            return int(self.get("surviving_processes") or 0)
        except (TypeError, ValueError):
            return 0

    @property
    def min_core_coverage(self) -> Optional[float]:
        thresholds = self.get("thresholds")
        if not isinstance(thresholds, Mapping):
            return None
        try:
            return float(thresholds.get("min_core_coverage"))
        except (TypeError, ValueError):
            return None

    # -- derived, and the reason this type exists -------------------------
    @property
    def completeness(self) -> Optional[float]:
        """How much of the REQUEST survived, or ``None`` when unjudgeable.

        The serialized ``coverage_ratio`` is ``0.0`` in both "nothing requested
        was found" and "nothing was requested", which are opposite facts. The
        serialized value cannot change (pinned), so the distinction is drawn
        here: undeclared is ``None``, never zero.
        """

        return self.coverage_ratio if self.declared else None

    @property
    def missing_anchors(self) -> Tuple[str, ...]:
        """The requested-core anchors nothing surviving touches. D-002 requires a
        below-threshold run to record exactly these."""

        return _as_tuple(self.get("unmatched_terms"))

    def _has_reason(self, prefix: str) -> bool:
        return any(reason.split(":", 1)[0] == prefix for reason in self.reasons)

    @property
    def below_coverage_minimum(self) -> bool:
        """``requested_core_coverage_below_minimum`` fired. Per D-002 this is a
        trigger for targeted retrieval and then for ``review_required`` -- it is
        NOT, in itself, a refusal."""

        return self._has_reason(COVERAGE_REASON_BELOW_MINIMUM)

    @property
    def core_process_count_below_minimum(self) -> bool:
        return self._has_reason(COVERAGE_REASON_COUNT_BELOW_MINIMUM)

    @property
    def empty_graph(self) -> bool:
        return self._has_reason(COVERAGE_REASON_EMPTY)

    @property
    def has_surviving_core(self) -> bool:
        """Something survived that could be serialized without inventing biology.

        Deliberately the weakest possible test -- "anything at all survived" --
        because the product invariant forbids deleting an incomplete-but-correct
        pathway. A fragment that is merely *shallow*, or merely *not the pathway
        that was requested*, is still a fragment; it becomes ``review_required``,
        never ``diagnostic_only``.
        """

        return self.surviving_processes > 0 and not self.empty_graph


@dataclass(frozen=True)
class ReleaseStatus:
    """One run's release classification, with PRODUCT_CONTRACT 11 kept unfolded."""

    status: str
    #: 11(a). The pipeline ran, whatever it concluded.
    pipeline_executed: bool
    #: 11(b). The strict TECHNICAL gates. Never merged with the biological verdict.
    strict_gates_passed: bool
    #: 11(c)/(d)/(e) as ONE three-valued field. ``not_evaluated`` is never False.
    semantic_evaluation: str = SEMANTIC_NOT_EVALUATED
    semantic_not_evaluated_reason: str = SEMANTIC_INPUT_NOT_WIRED
    #: WHICH gating checks failed, as names (C-056b). Before this the names
    #: survived only inside a ``reasons`` entry -- ``semantic_evaluation_failed:a,b``
    #: -- so a consumer wanting them had to split a prose string on two
    #: separators, which is exactly what the reason VOCABULARY exists to stop.
    #: The benchmark is that consumer: the record travels verbatim into the
    #: manifest row, so ``bench.acceptance`` can say WHICH checks the run recorded
    #: as failed on a leg it did not count as semantically confirmed
    #: (``ModeResult.runtime_semantic_failed_checks``). "The runtime said no" is
    #: not a usable finding without the checks behind it.
    #:
    #: COHERENT BY CONSTRUCTION: non-empty here IFF ``semantic_evaluation`` is
    #: ``failed``. A pass or a non-evaluation has no failing checks to name, so a
    #: reader never has to decide which of two fields to believe.
    semantic_failed_checks: Tuple[str, ...] = ()
    #: HOW MUCH was evaluable behind the verdict (C-056c, F-053 / D-054 section 6).
    #: One entry per :data:`SEMANTIC_GATING_CHECKS` name, in that order, as
    #: ``(check, applicable, inapplicable_reason)``.
    #:
    #: Why it exists: ``semantic_evaluation == "passed"`` alone cannot tell a
    #: manifest reader "one of four gating checks was evaluable and that one
    #: passed" from "four of four were evaluable and all passed". Measured, the
    #: production seam reaches THREE -- ``CHECK_RAG_REINTRODUCTION`` is
    #: structurally inapplicable there because no admission report exists to pass
    #: it (``strict_quarantine.py`` says so at its own call site) -- and a replay
    #: of context-free committed artifacts reaches one. Neither figure is written
    #: down anywhere; both are DERIVED from this record, per run.
    #:
    #: DELIBERATELY UNGATED BY THE VERDICT, unlike ``semantic_failed_checks``
    #: above. Failing check names are meaningful only on a ``failed``, but
    #: evaluability is the missing context of a ``passed`` most of all, so gating
    #: it on any one state would reintroduce exactly the blind spot it closes.
    #:
    #: EMPTY MEANS NOT RECORDED, never "nothing was applicable": a run whose
    #: report did not evaluate has no per-check applicability, and a measured
    #: all-inapplicable report records four entries each naming its own reason.
    #: A reader distinguishes the two by length, never by guessing.
    semantic_check_evaluability: Tuple[Tuple[str, bool, str], ...] = ()
    #: Whether this run may count toward the STRICT benchmark denominator.
    #: ``review_required`` never may -- TRAP-1 / PRODUCT_CONTRACT 13.
    strict_acceptance_eligible: bool = False
    #: D-002's required record.
    completeness: Optional[float] = None
    missing_anchors: Tuple[str, ...] = ()
    retrieval_attempts: Optional[int] = None
    expansion_blocked_reason: str = ""
    coverage_evaluated: bool = False
    reasons: Tuple[str, ...] = ()

    @property
    def semantic_confirmed(self) -> bool:
        """Only an actual ``passed`` counts. ``not_evaluated`` is not a pass."""

        return self.semantic_evaluation == SEMANTIC_PASSED

    @property
    def produced_pwml(self) -> bool:
        """``diagnostic_only`` is the one state with no final PWML."""

        return self.status in (RELEASE_READY, REVIEW_REQUIRED)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "pipeline_executed": self.pipeline_executed,
            "strict_gates_passed": self.strict_gates_passed,
            "semantic_evaluation": self.semantic_evaluation,
            "semantic_not_evaluated_reason": self.semantic_not_evaluated_reason,
            "semantic_failed_checks": list(self.semantic_failed_checks),
            # F-053's carrier, as records rather than a bare count: a reader can
            # recover the count from the map but never the map from a count, and
            # only the map says WHY a check did not answer.
            "semantic_check_evaluability": [
                {"check": check, "applicable": applicable, "inapplicable_reason": reason}
                for check, applicable, reason in self.semantic_check_evaluability
            ],
            "strict_acceptance_eligible": self.strict_acceptance_eligible,
            "completeness": self.completeness,
            "missing_anchors": list(self.missing_anchors),
            "retrieval_attempts": self.retrieval_attempts,
            "expansion_blocked_reason": self.expansion_blocked_reason,
            "coverage_evaluated": self.coverage_evaluated,
            "reasons": list(self.reasons),
        }


def coverage_verdict(value: Any) -> Optional[CoverageVerdict]:
    """Adapt anything a caller already holds into a :class:`CoverageVerdict`.

    A verdict read back from JSON is a plain dict; re-wrapping is free and
    lossless, so a consumer never has to care which side of a serialization
    boundary its coverage came from.
    """

    if value is None:
        return None
    if isinstance(value, CoverageVerdict):
        return value
    if isinstance(value, Mapping):
        return CoverageVerdict(value)
    return None


def semantic_verdict(
    report: Any,
) -> Tuple[str, str, Tuple[str, ...], Tuple[Tuple[str, bool, str], ...]]:
    """Reduce a report to ``(evaluation, reason, failed_checks, evaluability)``.

    Duck-typed on purpose. This is ``t2pw.pipeline``; importing
    ``t2pw.bench.semantic`` to name the type would invert the layering for every
    importer of this module. What is read is the shape D-006 fixed and C-017
    shipped -- ``evaluated``, ``not_evaluated_reason`` and a ``checks`` mapping of
    name to a result carrying ``ok`` and ``applicable``.

    A0-C4, unrelaxed: ``confirmed`` is deliberately NOT consulted. It can never be
    ``True`` on a production run -- ``retained_reactions_match_supported_signatures``
    needs quote-verified gold signatures -- so gating on it would ship nothing,
    ever. ``evaluated``, ``ok`` and applicability are combined instead, and only
    over :data:`SEMANTIC_GATING_CHECKS`.

    The three outcomes, kept apart exactly as PRODUCT_CONTRACT 11 requires:

    * no report, or a report that did not evaluate -> ``not_evaluated`` with the
      report's own stated reason;
    * at least one gating check applicable and failing -> ``failed``, naming them;
    * at least one gating check applicable and all such passing -> ``passed``.

    The fourth case is the one that is easy to get wrong: a report that ran but in
    which **every** gating check was inapplicable. That is ``not_evaluated``, never
    ``passed`` -- there is no evidence behind a pass, and PRODUCT_CONTRACT 11's
    "not_evaluated is never false" cuts the other way too. A NON-GATING check
    cannot reach any branch here, so its failure can never demote a run.

    THE FOURTH RETURN VALUE (C-056c, F-053 / D-054 section 6). ``evaluable`` below
    was, and still is, consumed only as a BOOLEAN at the ``not evaluable`` guard.
    How MANY of the four gating checks answered, and WHICH, were dropped at every
    return -- so a downstream ``passed`` was indistinguishable from a four-of-four
    pass. That is what the fourth value carries: one entry per name in
    :data:`SEMANTIC_GATING_CHECKS`, in that order, as
    ``(check, applicable, inapplicable_reason)``.

    The reason string is RELOCATED, never invented: it is ``bench.semantic``'s own
    ``CheckResult.inapplicable_reason``, read through the same duck-typed accessor
    as ``applicable`` and ``ok``. A check the report never carried, or one carrying
    no reason, records ``""`` -- writing a synthetic reason there would make an
    absence indistinguishable from a measured one (the D-038 rule).

    A count is derivable from this map; the map is not derivable from a count,
    which is why a count is not what travels. And it is a CARRIER only: no verdict
    changes, no branch reads it, and nothing here turns a pass into a numerator --
    F-053's prohibition on affirmative readers of ``passed`` is untouched.
    """

    if report is None:
        return SEMANTIC_NOT_EVALUATED, SEMANTIC_NO_REPORT, (), ()
    if not getattr(report, "evaluated", False):
        reason = str(getattr(report, "not_evaluated_reason", "") or "") or SEMANTIC_NO_REPORT
        # No evaluation happened, so there is no per-check applicability to
        # report. Empty means "not recorded" and is never confused with a
        # measured four-way map, every entry of which names a real check.
        return SEMANTIC_NOT_EVALUATED, reason, (), ()

    checks = getattr(report, "checks", None) or {}
    failed: List[str] = []
    evaluable = 0
    evaluability: List[Tuple[str, bool, str]] = []
    for name in SEMANTIC_GATING_CHECKS:
        result = checks.get(name) if isinstance(checks, Mapping) else None
        # Exactly the negation of the guard this replaced -- a missing result and
        # a non-applicable one are still both non-evaluable, and the arithmetic
        # below is byte-for-byte the arithmetic that was here before.
        applicable = result is not None and bool(getattr(result, "applicable", False))
        evaluability.append((
            name,
            applicable,
            "" if applicable else str(getattr(result, "inapplicable_reason", "") or ""),
        ))
        if not applicable:
            continue
        evaluable += 1
        if not getattr(result, "ok", False):
            failed.append(name)
    applicability = tuple(evaluability)
    if failed:
        return SEMANTIC_FAILED, "", tuple(failed), applicability
    if not evaluable:
        return SEMANTIC_NOT_EVALUATED, SEMANTIC_NO_GATING_CHECK_EVALUABLE, (), applicability
    return SEMANTIC_PASSED, "", (), applicability


def classify_release_status(
    coverage: Any = None,
    *,
    pipeline_executed: bool = True,
    strict_gates_passed: bool = False,
    serializable_without_invention: bool = True,
    retrieval_attempts: Optional[int] = None,
    expansion_blocked_reason: str = "",
    extra_reasons: Sequence[str] = (),
    semantic_evaluation: str = SEMANTIC_NOT_EVALUATED,
    semantic_not_evaluated_reason: str = SEMANTIC_INPUT_NOT_WIRED,
    semantic_failed_checks: Sequence[str] = (),
    semantic_check_evaluability: Sequence[Any] = (),
) -> ReleaseStatus:
    """Classify one run from its coverage verdict and its technical outcome.

    The rules are in the product's order and each refuses exactly one thing:

    1. the pipeline never ran, or
    2. the strict TECHNICAL gates blocked export, or
    3. serialization would require inventing biology, or
    4. nothing survived at all
       -> ``diagnostic_only``: there is no PWML to review.
    5. coverage was never evaluated, or the requested-core threshold was not met
       -> ``review_required``: D-002, the threshold blocks RELEASE-READY status,
       not PWML production. A biologically correct, internally connected fragment
       representable without guessing is exported and flagged, never dropped, and
       never counted as strict success.
    6. otherwise -> ``release_ready``.

    Then, and only then, the SEMANTIC CAP (D-042 section 3). A failing gating
    semantic check **caps** the status at ``review_required``. It is a cap, not a
    move:

    * it never produces ``diagnostic_only`` -- PRODUCT_CONTRACT 13 defines
      ``review_required`` as "valid, needs review", which is exactly a pathway
      whose semantics did not confirm, and merge rule 7 preserves
      incomplete-but-correct work rather than dropping it;
    * it never touches a status already ``review_required`` or
      ``diagnostic_only``, so a technical refusal is never restated as a
      biological one.

    Because a cap is monotone it can only ever **remove** strict successes, never
    create one -- no new strict success without measured evidence.

    ``semantic_evaluation`` still defaults to ``not_evaluated`` with the unwired
    reason, so a caller that passes no semantic input gets byte-identically the
    record it got before this input existed. ``not_evaluated`` is never ``False``
    and never demotes: an unevaluable check produces NO status change.

    ``semantic_check_evaluability`` (C-056c, F-053) is the same kind of input and
    defaults the same way: omit it and the record is byte-identical to the one
    this function produced before the parameter existed, apart from the one new
    ``to_dict`` key, which is then an empty list meaning "not recorded". It is
    RECORDED AND NEVER READ here -- it changes no status, no cap and no
    eligibility, because a carrier that could move a verdict would be a second
    gate wearing a record's name.
    """

    verdict = coverage_verdict(coverage)
    reasons = [str(reason) for reason in extra_reasons or ()]
    completeness = verdict.completeness if verdict is not None else None
    missing = verdict.missing_anchors if verdict is not None else ()

    if not pipeline_executed:
        status = DIAGNOSTIC_ONLY
        reasons.append(REASON_PIPELINE_DID_NOT_EXECUTE)
    elif not strict_gates_passed:
        status = DIAGNOSTIC_ONLY
        reasons.append(REASON_STRICT_GATES_BLOCKED)
    elif not serializable_without_invention:
        status = DIAGNOSTIC_ONLY
        reasons.append(REASON_SERIALIZATION_REQUIRES_INVENTION)
    elif verdict is None:
        status = REVIEW_REQUIRED
        reasons.append(REASON_COVERAGE_NOT_EVALUATED)
    elif not verdict.has_surviving_core:
        status = DIAGNOSTIC_ONLY
        reasons.append(REASON_NO_DEFENSIBLE_CONNECTED_CORE)
        reasons.extend(verdict.reasons)
    elif not verdict.minimum_core_satisfied:
        status = REVIEW_REQUIRED
        reasons.extend(verdict.reasons)
    else:
        status = RELEASE_READY

    # The semantic cap. Applied AFTER the technical chain and only downward from
    # release_ready, so it can never manufacture a status the rules above refused
    # and can never deepen one they already reached.
    evaluation = str(semantic_evaluation or SEMANTIC_NOT_EVALUATED)
    # Recorded whenever the verdict is FAILED, cap or no cap: a run the technical
    # chain already put in review_required whose semantics ALSO failed must still
    # say which checks failed. Tying the names to the verdict rather than to the
    # cap is what makes the field coherent -- a pass or a non-evaluation carries
    # none, so a consumer never reads a failing check name beside "passed".
    failed = (
        tuple(str(name) for name in semantic_failed_checks or ())
        if evaluation == SEMANTIC_FAILED else ()
    )
    # Recorded on ALL THREE verdicts, unlike ``failed`` directly above. F-053 is
    # about a ``passed`` whose evaluability nobody can see, so conditioning this
    # on the verdict would drop it exactly where it is needed. Tolerant of both
    # shapes the record travels in -- the in-memory triple and the serialized
    # mapping ``to_dict`` writes -- so a classification rebuilt from JSON keeps
    # its evaluability instead of silently flattening to "not recorded".
    evaluability: List[Tuple[str, bool, str]] = []
    for entry in semantic_check_evaluability or ():
        if isinstance(entry, Mapping):
            check, applicable, reason = (
                entry.get("check"),
                entry.get("applicable"),
                entry.get("inapplicable_reason"),
            )
        else:
            check, applicable, reason = entry
        evaluability.append((str(check or ""), bool(applicable), str(reason or "")))
    if evaluation == SEMANTIC_FAILED and status == RELEASE_READY:
        status = REVIEW_REQUIRED
        reasons.append(
            f"{REASON_SEMANTIC_EVALUATION_FAILED}:{','.join(failed)}" if failed
            else REASON_SEMANTIC_EVALUATION_FAILED
        )

    return ReleaseStatus(
        status=status,
        pipeline_executed=bool(pipeline_executed),
        strict_gates_passed=bool(strict_gates_passed),
        semantic_evaluation=evaluation,
        # A reason belongs to ``not_evaluated`` alone: carrying the unwired
        # placeholder beside a real ``passed``/``failed`` verdict would tell a
        # reader the evaluation never happened.
        semantic_not_evaluated_reason=(
            str(semantic_not_evaluated_reason or "")
            if evaluation == SEMANTIC_NOT_EVALUATED else ""
        ),
        semantic_failed_checks=failed,
        semantic_check_evaluability=tuple(evaluability),
        # A run may only enter the STRICT denominator when it is release-ready.
        # review_required must never count as strict success (TRAP-1).
        strict_acceptance_eligible=status == RELEASE_READY,
        completeness=completeness,
        missing_anchors=tuple(missing),
        retrieval_attempts=retrieval_attempts,
        expansion_blocked_reason=str(expansion_blocked_reason or ""),
        coverage_evaluated=verdict is not None,
        reasons=tuple(dict.fromkeys(reasons)),
    )


#: Rendering vocabulary, single-sourced so ``batch/report.py`` and
#: ``bench/render.py`` cannot drift into two different words for one fact.
NOT_RECORDED = "not recorded"
SEMANTIC_LABELS: Dict[str, str] = {
    SEMANTIC_PASSED: "passed",
    SEMANTIC_FAILED: "failed",
    SEMANTIC_NOT_EVALUATED: "NOT PERFORMED",
}
#: Printed wherever a technical pass could otherwise be read as release-ready.
NOT_RELEASE_READY_NOTE = (
    "a technical PASS is a GATE result, not a release decision; semantic "
    "evaluation is reported separately and is never implied by it"
)


def describe(status: Any) -> str:
    """One line naming the status and the three facts behind it, for reports.

    Accepts a :class:`ReleaseStatus`, its ``to_dict``, a bare state string or
    ``None``: report code renders rows written by older runs and must never raise.
    """

    if status is None:
        return NOT_RECORDED
    if isinstance(status, ReleaseStatus):
        data = status.to_dict()
    elif isinstance(status, Mapping):
        data = dict(status)
    else:
        text = str(status).strip()
        return text or NOT_RECORDED
    state = str(data.get("status") or "").strip() or NOT_RECORDED
    semantic = str(data.get("semantic_evaluation") or SEMANTIC_NOT_EVALUATED)
    gates = "passed" if data.get("strict_gates_passed") else "failed"
    ran = "ran" if data.get("pipeline_executed") else "did not run"
    return (
        f"{state}  [pipeline {ran}; strict gates {gates}; semantic evaluation "
        f"{SEMANTIC_LABELS.get(semantic, semantic)}]"
    )


__all__ = [
    "RELEASE_READY", "REVIEW_REQUIRED", "DIAGNOSTIC_ONLY", "RELEASE_STATES",
    "SEMANTIC_PASSED", "SEMANTIC_FAILED", "SEMANTIC_NOT_EVALUATED",
    "SEMANTIC_INPUT_NOT_WIRED", "SEMANTIC_LABELS",
    "SEMANTIC_NO_REPORT", "SEMANTIC_NO_GATING_CHECK_EVALUABLE",
    "SEMANTIC_GATING_CHECKS",
    "NOT_RECORDED", "NOT_RELEASE_READY_NOTE",
    "COVERAGE_REASON_EMPTY", "COVERAGE_REASON_COUNT_BELOW_MINIMUM",
    "COVERAGE_REASON_BELOW_MINIMUM",
    "REASON_PIPELINE_DID_NOT_EXECUTE", "REASON_STRICT_GATES_BLOCKED",
    "REASON_SERIALIZATION_REQUIRES_INVENTION",
    "REASON_NO_DEFENSIBLE_CONNECTED_CORE", "REASON_COVERAGE_NOT_EVALUATED",
    "REASON_SEMANTIC_EVALUATION_FAILED",
    "CoverageVerdict", "ReleaseStatus",
    "coverage_verdict", "classify_release_status", "semantic_verdict", "describe",
]
