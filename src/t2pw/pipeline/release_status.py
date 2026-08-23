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
#: so blocking a biological release on it would misuse it.
#:
#: THE SECOND ONE'S OLD JUSTIFICATION WAS FALSE, and C-074 (F-101) replaces it
#: rather than leaving it standing. It read "duplicates a floor the coverage
#: verdict already enforced at this same seam". Measured, the two floors count
#: different things: the coverage verdict's ``min_core_processes`` counts ACCEPTED
#: PROCESSES INCLUDING INTERACTIONS against a threshold of 1, while
#: ``CHECK_CONNECTED_CORE`` counts REACTIONS that join into one chain through
#: shared non-cofactor metabolites. On the two legs F-100/F-101 were registered
#: from they disagree 3-vs-1 and 5-vs-1, so the coverage floor was passing
#: single-reaction payloads the connectivity floor would have stopped.
#:
#: The set below is still CLOSED and ``CHECK_CONNECTED_CORE`` is still not in it:
#: widening the gating set is a ratification (D-039 section 3), and the semantic
#: verdict is not where a structural floor belongs anyway. What C-074 adds instead
#: is :data:`MIN_CONNECTED_CORE_REACTIONS` and the CAP in
#: :func:`classify_release_status` that reads the size the production seam already
#: computes -- a cap, so it can only ever remove ``release_ready``.
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

#: F-094 / PRODUCT_CONTRACT 13. A declared requested core one or more of whose
#: anchors matched NO admitted process. Distinct from
#: :data:`COVERAGE_REASON_BELOW_MINIMUM` on purpose: that one is a RATIO falling
#: under a tunable threshold, this one is the unconditional fact that part of what
#: was asked for is absent from what survived. A run can satisfy the ratio
#: comfortably (0.8 >= 0.5, measured on PMC12452463/strict at T-104) and still be
#: missing three named anchors, which is precisely the case the ratio cannot see.
#: The unmatched anchor names are appended after a ``:`` so the record says WHICH,
#: exactly as ``semantic_evaluation_failed`` names its checks.
REASON_REQUESTED_CORE_ANCHORS_UNMATCHED = "requested_core_anchors_unmatched"

#: C-074 / F-101, PRODUCT_CONTRACT 13. The smallest CHEMICALLY CONNECTED core a
#: run may call ``release_ready``. Two, because one is not a pathway: a single
#: reaction has no step to be connected TO, and PRODUCT_CONTRACT 13 defines a bare
#: ``pathway.pwml`` as "ship it, no review needed" -- which a one-reaction payload
#: emitted for a multi-step request is not.
#:
#: Deliberately NOT the same number as, and not derived from,
#: ``strict_quarantine.DEFAULT_MIN_CORE_PROCESSES``. That one counts accepted
#: processes INCLUDING interactions and is pinned at 1 by
#: ``test_strict_quarantine_release_seam.py:222``; this one counts reactions that
#: join into one chain. Tying them together is what made the old "duplicates a
#: floor" justification above look true.
MIN_CONNECTED_CORE_REACTIONS = 2

#: The largest chemically connected core is under
#: :data:`MIN_CONNECTED_CORE_REACTIONS` and the request did not ask for a
#: single-reaction pathway. The observed and required sizes are appended after a
#: ``:`` so the record says WHICH shortfall, exactly as
#: ``requested_core_coverage_below_minimum`` does.
REASON_CONNECTED_CORE_BELOW_FLOOR = "connected_core_below_minimum"

#: C-074 / F-100. A context that CLAIMS to declare a requested core while naming
#: no pathway at all. Not a coverage failure and deliberately not worded as one:
#: coverage against terms nobody asked for is UNEVALUABLE, the same regime
#: :data:`REASON_COVERAGE_NOT_EVALUATED` names, and "nothing was asked for" must
#: never read as "nothing is missing".
REASON_REQUESTED_PATHWAY_NOT_STATED = "requested_pathway_not_stated"

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

    # -- C-074 / F-100: was the request actually STATED? -------------------
    @property
    def requested_core_source(self) -> str:
        """Which input produced the terms: ``explicit_argument`` /
        ``pathway_context`` / ``payload`` / ``none``. Written by
        ``evaluate_core_coverage``; a record written before that key existed
        reads as ``""`` and is treated as "not recorded", never as a source."""

        return str(self.get("requested_core_source") or "")

    @property
    def requested_context(self) -> Optional[Mapping[str, Any]]:
        """The Stage-0 context exactly as production handed it over, or ``None``
        when no context reached the check at all. The distinction is the whole
        point of the field and it is the whole point of the rule below."""

        context = self.get("requested_context")
        return context if isinstance(context, Mapping) else None

    @property
    def requested_pathway_name(self) -> str:
        context = self.requested_context
        return str((context or {}).get("pathway_name") or "").strip()

    @property
    def declares_core_without_stating_a_pathway(self) -> bool:
        """A context that CLAIMS a requested core while naming no pathway (F-100).

        Measured on the leg F-100 was registered from:
        ``requested_core_declared`` True, ``coverage_ratio`` 1.0,
        ``unmatched_terms`` empty -- and ``pathway_name`` the empty string, with
        the six "requested" terms being the PAPER's own key compounds and
        proteins read by Stage 0. The batch had asked for something else
        entirely. Scoring a request nobody stated
        against terms taken from the paper is not a test, which is the same
        failure mode ``collect_requested_core_terms`` already refuses to commit
        with survivor-derived terms.

        THREE conditions, and each excludes a regime that must not move:

        * ``declared`` -- with no terms at all relevance is already unjudgeable
          and ``completeness`` is already ``None``. The undeclared regime is
          untouched.
        * **the terms came FROM the context** (``requested_core_source ==
          "pathway_context"``). This is the "CLAIMS to declare a core" clause and
          it is doing real work, not decoration. An ``explicit_argument`` source
          means a caller passed ``requested_core=`` by hand, which STATES the
          request whatever the context says (C-074 section 3, arm B). A
          ``payload`` source means the terms were scraped off the payload's own
          ``metadata`` because nobody handed a context over -- a context carrying
          only, say, a PathWhiz category never claimed to declare anything, and
          demoting on it would demote shaped unit-test payloads for a fact about
          the harness. Measured: ``test_semantic_release_gating`` supplies
          exactly that shape.
        * a context mapping is actually present and names no pathway -- a run
          where NO context reached the check carries ``None`` here and behaves
          exactly as it did before this property existed. That is what keeps
          every context-free payload out of the new refusal.
        """

        return bool(
            self.declared
            and self.requested_core_source == "pathway_context"
            and self.requested_context is not None
            and not self.requested_pathway_name
        )

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
    connected_core_reactions: Optional[int] = None,
    min_connected_core_reactions: int = MIN_CONNECTED_CORE_REACTIONS,
    single_reaction_scope_requested: bool = False,
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

    Then the INCOMPLETE-CORE CAP (F-094, PRODUCT_CONTRACT 13), a second cap of the
    same shape and with the same three restrictions, and **independent of the
    semantic one**: when a requested core was DECLARED and one or more of its
    anchors matched no admitted process, ``release_ready`` is not available and the
    status is ``review_required``, whatever the semantic verdict says. It reads no
    semantic field, so a run whose semantics passed and whose structural gates are
    all green is still demoted while part of what was asked for is missing from
    what survived. It is not a blanket demotion: a declared core with every anchor
    matched is untouched.

    Then the CONNECTED-PATHWAY FLOOR (C-074 arm A, F-101) and the UNSTATED-REQUEST
    cap (C-074 arm B, F-100), a third and a fourth cap of exactly the same shape.
    The first refuses ``release_ready`` to a payload whose largest chemically
    connected core is under :data:`MIN_CONNECTED_CORE_REACTIONS` reactions unless
    a single-reaction pathway is what the REQUEST asked for; the second refuses it
    to a context that claims a requested core while naming no pathway, because a
    request nobody stated cannot be reported as satisfied. Both read only inputs a
    caller supplies or the coverage verdict already carries, and
    ``connected_core_reactions`` defaults to ``None`` -- not measured, so never a
    demotion -- which is what keeps every pre-C-074 caller byte-identical.

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

    # The INCOMPLETE-CORE cap (F-094, PRODUCT_CONTRACT 13). A SECOND, INDEPENDENT
    # cap of exactly the same shape as the semantic one above, and independent of
    # it in the only sense that matters: it reads no semantic field, so it holds
    # with ``semantic_evaluation == "passed"`` and every structural gate green.
    #
    # Why it cannot live in ``evaluate_core_coverage``: that function's
    # ``minimum_core_satisfied`` is ``not reasons``, and its three reasons are all
    # THRESHOLD questions -- nothing survived, too few core processes, ratio under
    # ``min_core_coverage``. Unmatched anchors are recorded there
    # (``unmatched_terms``) but are deliberately not one of them, because the
    # pinned 0.5 threshold has to stay load-bearing. So the fact is available and
    # simply never asked, which is how PMC12452463/strict finished T-104 at
    # coverage 0.8 with three unmatched anchors, its own
    # ``expansion_blocked_reason`` saying admitting them "would require unsupported
    # biology", and a ``release_ready`` status that asked for no human review.
    #
    # A cap, not a move, on all four counts:
    #   * only from ``release_ready`` -- it can remove a strict success and can
    #     never manufacture one;
    #   * exactly ONE step, to ``review_required``. Never ``diagnostic_only``: an
    #     incomplete core is "valid, needs review", and merge rule 7 preserves an
    #     incomplete-but-correct pathway rather than dropping it. The payload and
    #     the surviving processes are untouched here, as everywhere in this
    #     function;
    #   * never applied to a status already ``review_required`` or
    #     ``diagnostic_only``, so a technical refusal is never restated as a
    #     completeness one and no reason is accumulated twice;
    #   * only in the DECLARED regime. With no requested core, relevance is
    #     unjudgeable (``completeness`` is ``None`` above by the same rule) and
    #     there is no anchor to be missing.
    if status == RELEASE_READY and verdict is not None and verdict.declared and missing:
        status = REVIEW_REQUIRED
        reasons.append(
            f"{REASON_REQUESTED_CORE_ANCHORS_UNMATCHED}:{','.join(missing)}"
        )

    # The CONNECTED-PATHWAY FLOOR (C-074 arm A, F-101). A THIRD cap of the same
    # shape, and the one the caps above cannot express: F-101's leg had every
    # requested anchor matched, coverage 1.0, semantics passing and every
    # structural gate green, and shipped a bare ``pathway.pwml`` containing ONE
    # reaction. PRODUCT_CONTRACT 13 reads a bare PWML as "ship it, no review
    # needed"; a single reaction is not a multi-step pathway, and gold's own
    # ``export_rationale`` on that leg says emitting the requested pathway from
    # that paper "requires importing seven steps the paper never mentions".
    #
    # WHAT IS COUNTED, and why it is not the coverage floor: the largest
    # CHEMICALLY CONNECTED core -- reactions joined through shared non-cofactor
    # metabolites -- as ``bench.semantic._connected_core`` already computes it on
    # every production run (``semantic_production.py:468``). The coverage floor
    # next to it counts accepted processes INCLUDING INTERACTIONS at a threshold
    # of 1. On F-101's leg those read 1 and 3; on F-100's, 1 and 5. Interactions
    # are not reactions and never enter this number.
    #
    # NOT RECORDED IS NOT A FAILURE. ``None`` -- the default, and what every
    # caller that does not measure connectivity passes -- means the size was not
    # measured, and an unmeasured floor never demotes. A caller that measures it
    # passes the integer. This is the D-038 rule and it is why adding this
    # parameter leaves every existing caller's record byte-identical.
    #
    # THE ONE EXEMPTION IS READ FROM THE REQUEST. ``single_reaction_scope_requested``
    # is derived by the caller from what was ASKED FOR -- never from what
    # survived, which would make the floor score itself, and never from a paper
    # identifier, which would hardcode a benchmark into production.
    #
    # A cap, on all four counts, exactly like the two above: only from
    # ``release_ready``; exactly one step, to ``review_required``, never
    # ``diagnostic_only`` -- the payload and its surviving processes are
    # untouched here and merge rule 7 preserves an incomplete-but-correct pathway
    # rather than dropping it; never applied to a status the chain already
    # lowered; and it can only ever REMOVE a strict success.
    #
    # BOTH ARMS ARE EVALUATED BEFORE EITHER IS APPLIED, which is why they share
    # one block. Applied in sequence, the first to fire would take the status out
    # of ``release_ready`` and silence the second -- and on the leg F-100 was
    # registered from BOTH hold, so the record would have lost the very fact the
    # finding is about. This is the same rule ``semantic_failed_checks`` above
    # already follows: the FACT is recorded whenever it holds, cap or no cap, and
    # only the STATUS is capped once.
    below_connected_core_floor = (
        connected_core_reactions is not None
        and not single_reaction_scope_requested
        and int(connected_core_reactions) < int(min_connected_core_reactions)
    )
    request_was_never_stated = (
        verdict is not None and verdict.declares_core_without_stating_a_pathway
    )
    if status == RELEASE_READY and below_connected_core_floor:
        reasons.append(
            f"{REASON_CONNECTED_CORE_BELOW_FLOOR}:"
            f"{int(connected_core_reactions)}<{int(min_connected_core_reactions)}"
        )

    # THE UNSTATED REQUEST (C-074 arm B, F-100). A FOURTH cap, same shape again.
    # "Nothing was asked for" is currently read as "nothing is missing": a context
    # that declares a core while naming no pathway had its coverage scored against
    # terms Stage 0 read out of THE SAME PAPER, so the ratio came back 1.0 and the
    # gate passed on the declared NEGATIVE CONTROL of the batch.
    #
    # Recorded as UNEVALUABLE, not as a coverage failure -- the reason constant
    # says so and the coverage verdict itself is untouched, so the measured ratio
    # stays on the record as the evidence it is rather than being overwritten by a
    # number this cap invented. That is the same grain as ``completeness`` being
    # ``None`` rather than ``0.0`` in the undeclared regime.
    #
    # See ``CoverageVerdict.declares_core_without_stating_a_pathway`` for why the
    # undeclared regime and the no-context-at-all regime are both excluded by
    # construction.
    if status == RELEASE_READY and request_was_never_stated:
        reasons.append(REASON_REQUESTED_PATHWAY_NOT_STATED)
        reasons.append(REASON_COVERAGE_NOT_EVALUATED)
    # ONE step, once, however many of the two facts hold.
    if status == RELEASE_READY and (below_connected_core_floor or request_was_never_stated):
        status = REVIEW_REQUIRED

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
    "MIN_CONNECTED_CORE_REACTIONS", "REASON_CONNECTED_CORE_BELOW_FLOOR",
    "REASON_REQUESTED_PATHWAY_NOT_STATED",
    "CoverageVerdict", "ReleaseStatus",
    "coverage_verdict", "classify_release_status", "semantic_verdict", "describe",
]
