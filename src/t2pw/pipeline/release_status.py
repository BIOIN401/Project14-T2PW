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

#: PRODUCT_CONTRACT 4 output states. There is deliberately no fourth, and D-065
#: (LOCKED) did not add one: :data:`DISPOSITION_EXTRACTED_NOT_SERIALIZED` is a
#: DISPOSITION recorded on a separate field beside the status, never a member here.
#: Extending this tuple is a change D-065 charters and reviews on its own merits.
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

#: C-087 / F-123, D-068 (LOCKED). The ``stage`` label every report
#: ``t2pw.pwml.prefreeze_resolution`` produces carries -- the one
#: :func:`run_prefreeze_resolution` builds AND the one an entry point writes when a
#: canonicalizer RAISED. It is named here so :func:`prefreeze_review_reasons` can
#: tell a WHOLE REPORT from the bare ``review_required`` sub-mapping production
#: seams publish beside it, and so it can do that WITHOUT importing ``t2pw.pwml``:
#: this is ``t2pw.pipeline`` and a module-level import there would invert the
#: layering for every importer of this module, exactly as the note on
#: :data:`SEMANTIC_GATING_CHECKS` records for ``bench.semantic``.
#:
#: Kept in step BY TEST, not by comment: ``tests/test_c087_prefreeze_declination_
#: demotes_release_status.py`` runs the real ``run_prefreeze_resolution`` and
#: asserts the report it returns carries exactly this label.
PREFREEZE_RESOLUTION_STAGE = "prefreeze_resolution"

#: C-087 / F-123, D-068 (LOCKED). Pre-freeze canonicalization finished ``ok=False``
#: AND named at least one REVIEW-REQUIRED reason: an identity the PathBank DB was
#: never reachable to establish (D-029), or a species rename that was DECLINED
#: rather than guessed (C-082). The offending ``<canonicalizer>:<reason>`` pairs are
#: appended after a ``:`` so the record says WHICH, exactly as
#: ``semantic_evaluation_failed`` names its checks and
#: ``requested_core_anchors_unmatched`` names its anchors.
#:
#: NOT the whole ``ok`` flag, and the asymmetry is deliberate and ruled, not an
#: oversight -- see :func:`prefreeze_review_reasons` for the derivation and the
#: authority.
REASON_PREFREEZE_REVIEW_REQUIRED = "prefreeze_resolution_review_required"

#: C-088 / F-107, **D-065 (LOCKED)**. The one release DISPOSITION this module
#: recognizes: *a defensible pathway core was extracted, and a correct scope guard
#: stopped the run before audit, DB mapping, freeze and PWML serialization.*
#:
#: A DISPOSITION IS NOT A STATUS, and the separation is the ruling's own. D-065
#: offered three readings and preferred the third -- **an additional explicit field
#: beside the existing runtime status** -- precisely so the safe runtime refusal is
#: PRESERVED rather than replaced. So this name is deliberately NOT a member of
#: :data:`RELEASE_STATES`, ``PRODUCT_CONTRACT`` 4 still has exactly three output
#: states, and a run carrying this disposition still reports
#: ``status == diagnostic_only``, ``strict_gates_passed == False`` and
#: ``produced_pwml == False``. Extending ``RELEASE_STATES`` is a change D-065
#: charters and reviews on its own merits; it is not taken here.
#:
#: WHAT IT REMOVES. ``PRODUCT_CONTRACT`` 4's ``diagnostic_only`` gloss reads
#: "recovery and retrieval could not establish a defensible pathway core", and
#: measured, that is UNTRUE of the legs this names: on the committed run
#: ``runs_verify/2026-08-24_1428``, ``PMC12421875``'s two legs each reached a
#: connected core of **9** against that case's gold floor of **7**. The record said
#: something false about them, and this is what makes the record honest without
#: fabricating a gate result nobody measured (C-077's refusal, ratified by D-065).
DISPOSITION_EXTRACTED_NOT_SERIALIZED = "extracted_not_serialized"

#: NOT RECORDED. The default everywhere, and never a fourth disposition: a run this
#: rule cannot affirmatively place carries the empty string, exactly as
#: ``expansion_blocked_reason`` does, and a reader distinguishes "no disposition was
#: established" from "a disposition says X" by emptiness alone (the D-038 rule).
NO_DISPOSITION = ""

#: The closed disposition vocabulary. One member, and a widening is a ratification.
RELEASE_DISPOSITIONS: Tuple[str, ...] = (DISPOSITION_EXTRACTED_NOT_SERIALIZED,)

#: LOCAL RESTATEMENT of ``batch.driver.REASON_STAGE0_SCOPE_CONFLICT``, the same
#: house pattern the note on :data:`SEMANTIC_GATING_CHECKS` records for
#: ``bench.semantic``: this is ``t2pw.pipeline`` and a module-level ``t2pw.batch``
#: import here would invert the layering for every importer of this module.
#:
#: It is the ONLY evidence that the stop was a SCOPE GUARD rather than a gate
#: failure or a crash, which is why the disposition below reads it and nothing
#: else: ``strict_technical_gates_blocked_export`` is emitted by three other
#: terminal paths and cannot tell them apart, and a status alone cannot either.
#: Kept in step BY TEST, not by comment -- ``tests/test_c088_extracted_not_
#: serialized_disposition.py`` asserts this equals the driver's own constant.
SCOPE_GUARD_STOP_REASON = "stage0_scope_conflict_stopped_the_run_before_serialization"

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
    #: C-088 / F-107, **D-065 (LOCKED)**: the explicit disposition BESIDE the status,
    #: which is D-065's preferred reading 3 in one field. Either
    #: :data:`DISPOSITION_EXTRACTED_NOT_SERIALIZED` or :data:`NO_DISPOSITION`.
    #:
    #: LAST, and deliberately so: every field above keeps its position, so a caller
    #: constructing this record positionally is unaffected. It is the LOGICAL
    #: neighbour of ``status`` and ``to_dict`` writes it there.
    #:
    #: IT ADDS NOTHING TO ``status`` AND CONTRADICTS NOTHING IN IT. The status is
    #: still whatever the technical chain concluded; this says which SHAPE of that
    #: conclusion was measured. Derived by :func:`release_disposition` from facts a
    #: caller supplies, never asserted by the caller directly, so the two cannot be
    #: made to disagree from outside this module.
    disposition: str = NO_DISPOSITION

    @property
    def semantic_confirmed(self) -> bool:
        """Only an actual ``passed`` counts. ``not_evaluated`` is not a pass."""

        return self.semantic_evaluation == SEMANTIC_PASSED

    @property
    def produced_pwml(self) -> bool:
        """``diagnostic_only`` is the one state with no final PWML."""

        return self.status in (RELEASE_READY, REVIEW_REQUIRED)

    def to_dict(self) -> Dict[str, Any]:
        record: Dict[str, Any] = {
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
        # CONDITIONAL, and for the reason ``bench.acceptance.ModeResult.to_dict``
        # already states about its own two optional keys: a record that established
        # no disposition serializes BYTE-IDENTICALLY to the one this method produced
        # before the field existed, so not one committed artifact, pinned digest or
        # golden capture moves. Absent means NOT RECORDED; an always-present empty
        # string would be a placeholder that reads like a measurement, and the
        # seven-slot digest in ``tests/test_batch_driver_seam_golden.py`` -- which
        # hashes this exact dict -- would have moved on every leg for a fact none of
        # them measured.
        #
        # Written NEXT TO ``status`` rather than appended, because D-065's reading 3
        # is "an additional explicit field BESIDE the existing runtime status" and a
        # reader of the serialized record should not have to hunt for it.
        if self.disposition:
            reordered: Dict[str, Any] = {}
            for key, value in record.items():
                reordered[key] = value
                if key == "status":
                    reordered["disposition"] = self.disposition
            return reordered
        return record


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


def prefreeze_review_reasons(prefreeze: Any) -> Tuple[str, ...]:
    """The REVIEW-REQUIRED reasons a pre-freeze resolution report names, normalized.

    ``()`` means "no review-required declination reached this call", and it is the
    answer for every input that is not one -- ``None``, a path string, a report that
    finished ``ok=True``, an empty sub-mapping. **Not recorded is not a failure**
    (the D-038 rule), which is what keeps every caller that hands this nothing
    byte-identical to the record it produced before this function existed.

    TWO SHAPES, because production publishes both and they are not interchangeable.
    ``t2pw.pwml.prefreeze_resolution.run_prefreeze_resolution`` returns the WHOLE
    report; the CLI seam (``writer.py:2732``) and both Streamlit seams
    (``streamlit_app.py:4571``/``:4956``) additionally publish its
    ``review_required`` SUB-MAPPING alone, under ``prefreeze_review_required``. The
    two are told apart by ``stage``, which every report the module produces carries
    and no canonicalizer name can be -- including the report an entry point writes
    when a canonicalizer RAISED, which has ``ok=False`` and NO ``review_required``
    key at all and would otherwise be read as a sub-mapping and turned into reasons
    out of its own bookkeeping keys.

    THE ``ok`` GATE, and why only the report shape can carry it. D-068 demotes on
    ``ok=False`` **with a review-required reason**. Handed a whole report this
    checks ``ok is False`` literally, so a report that finished ``ok=True`` returns
    ``()`` whatever else it carries. Handed the bare sub-mapping there is no ``ok``
    to read -- and none is needed, because ``run_prefreeze_resolution`` builds the
    two from one dict in adjacent statements::

        report["ok"] = not failures
        report["review_required"] = {n: r for n, r in failures.items()
                                     if r in _REVIEW_REQUIRED_REASONS}

    so a NON-EMPTY ``review_required`` implies a non-empty ``failures`` implies
    ``ok is False``, by construction. A non-empty sub-mapping IS the ``ok=False``
    evidence; an empty one yields ``()`` and demotes nothing.

    THE ASYMMETRY THIS DELIBERATELY DOES **NOT** COLLAPSE (D-068 requires it be
    decided, documented and justified from the record rather than guessed). An
    ``ok=False`` carrying **no** review-required reason returns ``()`` here and is
    release-status-NEUTRAL, exactly as it is today. Three grounds, all from the
    record:

    * **D-068's grant is exactly that narrow.** It rules on ``ok=False`` *"with a
      review-required reason"*. Extending the demotion to the whole ``ok`` flag
      would be an improvised product decision of precisely the kind D-068 says
      C-082 had no authority to take, and this card has no more authority than
      C-082 did.
    * **The record says the two are different in KIND.**
      ``prefreeze_resolution._REVIEW_REQUIRED_REASONS`` is documented as *"Verdicts
      that mean 'identity was not established', not 'the payload is wrong'"*, and
      holds exactly ``resolution_report_not_ok:db_unavailable`` and
      ``species_rename_declined:AMBIGUOUS_RENAME_TARGET``. The residue --
      ``resolution_report_not_ok`` with the DB REACHABLE (the resolver was consulted
      and rejected the row), a non-benign ``skipped_reason``, ``summary_not_a_
      mapping`` -- is the opposite kind: a statement ABOUT THE PAYLOAD, or about the
      harness. ``PRODUCT_CONTRACT`` 4 defines ``review_required`` as *"Valid, useful
      PWML produced, but one or more important biological uncertainties"*. A payload
      a reachable resolver actively rejected may not be merely uncertain, and
      whether it is ``review_required`` or ``diagnostic_only`` is a question nothing
      in the record answers.
    * **Silence there weakens nothing.** The residue keeps byte-identically the
      behaviour it has at base; no gate moves, nothing is admitted, and merge rule 6
      is untouched. It is registered as an open question for the product owner
      rather than settled here.

    The returned strings are ``"<canonicalizer>:<reason>"`` -- ``compounds`` /
    ``species`` and the reason verbatim -- SORTED, so the reason line a run records
    does not depend on ``PREFREEZE_CANONICALIZERS`` ordering or on dict insertion
    order. An entry whose reason is blank is dropped: a name with no reason behind it
    is not a stated declination.
    """

    if not isinstance(prefreeze, Mapping):
        return ()
    if str(prefreeze.get("stage") or "") == PREFREEZE_RESOLUTION_STAGE:
        if prefreeze.get("ok") is not False:
            return ()
        review: Any = prefreeze.get("review_required")
    else:
        review = prefreeze
    if not isinstance(review, Mapping):
        return ()
    return tuple(sorted(
        f"{name}:{str(reason).strip()}"
        for name, reason in review.items()
        if str(reason or "").strip()
    ))


def _as_measured_int(value: Any) -> Optional[int]:
    """An integer that was actually MEASURED, or ``None``.

    ``None``, ``""``, a bool and anything unparseable are all "not measured", and
    not measured is never a fact (the D-038 rule). ``bool`` is excluded explicitly
    because ``isinstance(True, int)`` is ``True`` in Python and a caller handing a
    flag where a count belongs would otherwise be read as the count ``1``.
    """

    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def release_disposition(
    release: Any,
    *,
    connected_core_reactions: Any = None,
    required_connected_reactions: Any = None,
    min_connected_core_reactions: int = MIN_CONNECTED_CORE_REACTIONS,
    produced_pwml: Optional[bool] = None,
) -> str:
    """The D-065 disposition a release record qualifies for, or :data:`NO_DISPOSITION`.

    **THE SINGLE RULE**, and single on purpose: :func:`classify_release_status`
    calls it while building a record and ``bench.acceptance.ModeResult`` calls it
    over a record another stage already froze, so the runtime field and the
    acceptance record cannot drift into two readings of one ruling. That is the
    same shape C-087 gave :func:`prefreeze_review_reasons`, and it is what D-065
    means by *"so no reader has to decide which of two fields to believe"*.

    **AFFIRMATIVE BY CONSTRUCTION.** Every clause below must be satisfied by a fact
    that was measured; anything absent, unparseable or merely not recorded returns
    :data:`NO_DISPOSITION`. There is no branch that reaches a disposition through an
    ABSENCE, so a run whose evidence never arrived is excluded without anyone
    enumerating the ways evidence can go missing.

    The five clauses are D-065 / C-088 section 4's four conditions, and each names
    where it is read from:

    1. **The pipeline executed** -- ``pipeline_executed is True`` on the record.
       Literal ``True``, not truthiness: this is the machine-readable refutation of
       *"nothing was attempted"* and a string that happens to be non-empty is not it.
    2. **The stop was a SCOPE GUARD**, not a gate failure and not a crash --
       :data:`SCOPE_GUARD_STOP_REASON` is in ``reasons``. Nothing else in the record
       can carry this: ``strict_technical_gates_blocked_export`` is emitted by three
       other terminal paths, and ``status`` alone is emitted by all of them.
       ``strict_gates_passed`` must additionally be ``False``, which is the same
       fact from the other side -- a scope guard stops the run BEFORE the strict
       technical gates run, so a record claiming they passed did not come from one.
    3. **No PWML was written.** ``status`` is ``diagnostic_only``, the one state
       :attr:`ReleaseStatus.produced_pwml` defines as having no final PWML. A caller
       holding an INDEPENDENT observation -- the acceptance scorer reads the
       manifest row's own artifact name -- passes ``produced_pwml`` and it is
       believed over the status, in the refusing direction only: an explicit
       ``True`` withdraws the disposition, and an explicit ``False`` never grants
       one the status did not already allow.
    4. **A defensible connected core was actually reached** -- NOT ASSUMED. Both
       sizes must be measured integers and both floors must clear:

       * ``connected_core_reactions >= required_connected_reactions``, the case's
         OWN floor (``bench.goldset.GoldCase.min_connected_reactions``), which is
         the number the gold set says that paper actually supports; and
       * ``connected_core_reactions >= min_connected_core_reactions``, the sprint's
         floor for calling anything a pathway at all
         (:data:`MIN_CONNECTED_CORE_REACTIONS`, C-074 / F-101: *"one is not a
         pathway: a single reaction has no step to be connected TO"*).

       **BOTH, and the second is not redundant.** Measured on
       ``runs_verify/2026-08-24_1428``, the six scope-conflict legs split 4/2 on it:
       ``PMC12421875`` reaches 9 against a gold floor of 7 and ``PMC12657337``
       reaches 5 and 3 against a floor of 3, but ``PMC12312563``'s two legs reach
       **1** against a gold floor of **1** -- the case floor clears, and that case's
       own gold ``export_rationale`` says in terms *"A single reaction cannot form a
       connected pathway, and no second reaction anywhere in the text shares a
       metabolite with it."* On that leg ``diagnostic_only``'s existing gloss is
       TRUE, so granting it a disposition that asserts a defensible pathway core
       would replace one untruth with another. The gold set is the authority on that
       question and it has already answered it.

    Returns :data:`DISPOSITION_EXTRACTED_NOT_SERIALIZED` or :data:`NO_DISPOSITION`.
    It reads no biology, writes nothing, moves no status, touches no payload and
    creates no route toward strict export -- ``strict_acceptance_eligible`` is
    computed from ``status == release_ready`` and this function cannot reach a
    record whose status is anything but ``diagnostic_only``.
    """

    if isinstance(release, ReleaseStatus):
        record: Mapping[str, Any] = release.to_dict()
    elif isinstance(release, Mapping):
        record = release
    else:
        return NO_DISPOSITION

    status = str(record.get("status") or "")
    if status != DIAGNOSTIC_ONLY:
        return NO_DISPOSITION
    if record.get("pipeline_executed") is not True:
        return NO_DISPOSITION
    if record.get("strict_gates_passed") is not False:
        return NO_DISPOSITION
    if SCOPE_GUARD_STOP_REASON not in {str(r) for r in (record.get("reasons") or ())}:
        return NO_DISPOSITION
    # ``status`` already establishes this; an explicit observation may still
    # WITHDRAW the disposition, never grant it.
    if produced_pwml:
        return NO_DISPOSITION

    core = _as_measured_int(connected_core_reactions)
    required = _as_measured_int(required_connected_reactions)
    if core is None or required is None or required < 1:
        return NO_DISPOSITION
    if core < required or core < int(min_connected_core_reactions):
        return NO_DISPOSITION
    return DISPOSITION_EXTRACTED_NOT_SERIALIZED


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
    prefreeze_review_required: Any = None,
    required_connected_reactions: Optional[int] = None,
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

    Then the PRE-FREEZE DECLINATION cap (C-087 arm, F-123 / **D-068**), a fifth cap
    of exactly the same shape. When pre-freeze canonicalization finished ``ok=False``
    naming a REVIEW-REQUIRED reason -- an identity the database was never reachable
    to establish, or a species rename DECLINED rather than guessed --
    ``release_ready`` is not available. Before D-068 this channel was
    release-status-NEUTRAL: ``report["ok"] = False`` demoted nothing, both consuming
    seams say so in terms, and D-035 clause 8's *"must not become a successful
    export"* was left enforced only by the OTHER gates. An invariant that holds by
    coincidence of gate ordering is not an invariant.

    ``prefreeze_review_required`` defaults to ``None`` -- not recorded, so never a
    demotion -- which is what keeps every pre-C-087 caller byte-identical, and it is
    read only through :func:`prefreeze_review_reasons`, which is where the ``ok``
    gate and the documented asymmetry live. It is a CAP on all four counts like the
    four above: only from ``release_ready``; exactly one step, to
    ``review_required``, never ``diagnostic_only`` -- **the payload, the graph and
    the surviving processes are untouched here as everywhere in this function, the
    declined rename stays declined, and nothing merges** (merge rule 7, D-068's
    *"useful intact biology remains available"*); never applied to a status the chain
    already lowered; and it can only ever REMOVE a strict success.

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

    Finally the D-065 DISPOSITION (C-088 arm, F-107), which is not a sixth cap and
    not any kind of rule about the status. Every branch above has already settled
    ``status``, and it is returned unchanged whatever the disposition evaluates to.
    :attr:`ReleaseStatus.disposition` records WHICH SHAPE of ``diagnostic_only`` was
    measured -- ``extracted_not_serialized`` when a defensible connected core was
    reached and a correct scope guard stopped the run before serialization -- and
    :func:`release_disposition` holds the whole rule, so the runtime field and the
    acceptance record read one ruling. It needs ``required_connected_reactions``,
    which defaults to ``None`` (not measured, so never a disposition) and which no
    production seam supplies today, so every existing caller's record is
    byte-identical: ``to_dict`` writes the key only when a disposition was actually
    established.
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
    # finding is about. Both facts are therefore recorded, and the status is
    # capped ONCE.
    #
    # Recorded FROM ``release_ready`` ONLY, unlike ``semantic_failed_checks``
    # above, which records on a FAILED verdict cap or no cap. The difference is
    # deliberate and is C-072 precedent: a status the chain already lowered was
    # lowered for a TECHNICAL reason, and appending a completeness reason to it
    # would restate that refusal as a biological one. The two arms differ from
    # each other in nothing.
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

    # THE PRE-FREEZE DECLINATION (C-087 arm A, F-123, D-068 LOCKED). A FIFTH cap,
    # same shape again, and the one no gate above can express: it is a fact about
    # CANONICALIZATION, not about coverage, connectivity, serializability or
    # semantics, and until D-068 nothing read it. ``report["ok"] = False`` was
    # persisted and surfaced by three production seams and acted on by none --
    # ``writer.py:2724`` and ``streamlit_app.py:4952`` both say so in terms, because
    # D-029 as split by D-040 section 8 assigned acting on it to NO card and
    # registered it as backlog ``BL-004``. D-068 assigns ``BL-004`` and this is it.
    #
    # WHAT IS READ, and what is deliberately not. Only the REVIEW-REQUIRED channel:
    # a declination whose reason ``prefreeze_resolution`` classifies as "identity was
    # not established", never the bare ``ok`` flag. :func:`prefreeze_review_reasons`
    # holds that derivation, the ``ok`` gate, and the justification for the
    # asymmetry; it is a single normalizer rather than an inline test so this cap and
    # :func:`cap_release_for_prefreeze_declination` -- the seam that applies the same
    # rule to an ALREADY-FROZEN record -- cannot drift into two readings of one
    # ruling.
    #
    # WHAT IS NOT DONE, and D-068 names each one. The payload is not discarded: no
    # branch here touches ``coverage``, ``completeness``, ``missing_anchors`` or any
    # graph, and ``review_required`` is precisely the state PRODUCT_CONTRACT 4
    # reserves for "valid, useful PWML produced" with an uncertainty named. The
    # ambiguous rename REMAINS DECLINED -- this function cannot rename anything and
    # does not try, so no unsafe merge is guessed. And nothing is dropped merely
    # because a rename was ambiguous, which is merge rule 7 and the reason this is a
    # cap to ``review_required`` and never a move to ``diagnostic_only``.
    #
    # RECORDED FROM ``release_ready`` ONLY, the same convention as the four caps
    # above and for the same C-072 reason: a status the chain already lowered was
    # lowered for its own reason, and appending this one to it would restate that
    # refusal as a canonicalization one. The declination itself is never lost by
    # that -- it is persisted to ``pwml_prefreeze_resolution_report.json`` and
    # published under ``prefreeze_review_required`` at every seam, whatever the
    # status.
    prefreeze_reasons = prefreeze_review_reasons(prefreeze_review_required)
    if status == RELEASE_READY and prefreeze_reasons:
        status = REVIEW_REQUIRED
        reasons.append(
            f"{REASON_PREFREEZE_REVIEW_REQUIRED}:{','.join(prefreeze_reasons)}"
        )

    # THE D-065 DISPOSITION (C-088 arm, F-107). NOT a cap, not a gate and not a
    # sixth rule: every branch above has already run and the status it produced is
    # returned UNCHANGED below, whatever this evaluates to. What it adds is the
    # explicit field D-065's reading 3 asks for, beside a runtime refusal that is
    # PRESERVED rather than replaced -- the run still stopped before serialization,
    # ``strict_gates_passed`` is still whatever was measured, ``produced_pwml`` is
    # still ``False``, and ``strict_acceptance_eligible`` below still reads
    # ``status == RELEASE_READY``, which this can never reach.
    #
    # ``required_connected_reactions`` defaults to ``None`` -- not measured, so never
    # a disposition -- and it is the ONLY new input, so every existing caller gets
    # ``NO_DISPOSITION`` and, because ``to_dict`` writes the key only when it is set,
    # a byte-identical record. That is deliberate: the case floor it wants is the
    # GOLD SET's ``min_connected_reactions``, which is a benchmark fact no production
    # seam holds, so the caller that supplies it today is the acceptance scorer.
    disposition = release_disposition(
        {
            "status": status,
            "pipeline_executed": bool(pipeline_executed),
            "strict_gates_passed": bool(strict_gates_passed),
            "reasons": list(reasons),
        },
        connected_core_reactions=connected_core_reactions,
        required_connected_reactions=required_connected_reactions,
        min_connected_core_reactions=min_connected_core_reactions,
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
        disposition=disposition,
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


def cap_release_for_prefreeze_declination(
    release: Any,
    prefreeze: Any = None,
) -> Dict[str, Any]:
    """Apply the D-068 cap to a release record that was ALREADY FROZEN.

    THE SAME RULE AS :func:`classify_release_status`'s fifth cap, reached through
    the same :func:`prefreeze_review_reasons`, applied to the serialized record
    instead of to classifier inputs. It exists because of an ORDERING fact that no
    amount of parameter-passing can change: pre-freeze compound and species
    canonicalization runs at the EXPORT seams
    (``streamlit_app.py:4322``/``:4873``, ``writer.py:2692``), and the release
    classification is frozen EARLIER, at the quarantine boundary
    (``strict_quarantine.py`` -> :func:`classify_release_status`). Nothing in
    ``src`` holds both at once at classification time, so the verdict has to reach
    the frozen record instead of the classifier call.

    THIS IS NOT A RE-CLASSIFICATION, and the distinction is the whole reason this
    is a separate function rather than a second ``classify_release_status`` call at
    the consuming seam. Re-deriving the classification downstream would be an
    exporter answering a biological question after the freeze, which **merge rule 8
    forbids outright** and which ``batch/driver.py::_frozen_release_record`` already
    refuses by name. What happens here instead:

    * it is **MONOTONE**. The only transition is ``release_ready`` ->
      ``review_required``. No other status is reachable from any input, so it can
      only ever REMOVE a strict success and can never manufacture one. A
      ``diagnostic_only`` record is returned unchanged -- it is never PROMOTED to
      ``review_required``, which would be this function inventing a PWML that the
      chain said does not exist;
    * it **reads no biology**. No payload, graph, entity, reaction, name or
      identifier is read, written, added, removed, resolved or reinterpreted. The
      only inputs are a status string and a verdict another stage already reached;
    * it **repairs nothing**. The declined rename stays declined; nothing merges;
      nothing is dropped (merge rule 7, D-068's *"no payload is discarded merely
      because the rename was ambiguous"*);
    * ``strict_acceptance_eligible`` is forced to ``False`` on the demoted record,
      preserving the invariant ``strict_acceptance_eligible == (status ==
      release_ready)`` that ``classify_release_status`` establishes and FINDINGS M-8
      says must live in exactly one place. It is only ever set to ``False`` here,
      never to ``True`` (TRAP-1 / ``PRODUCT_CONTRACT`` 13).

    Returns a NEW dict; the record handed in is never mutated, so a caller holding
    the frozen object still holds the frozen object. A record whose ``status`` is
    absent or outside :data:`RELEASE_STATES` is returned as a plain copy: this
    function interprets a classification, it does not invent one.
    """

    record: Dict[str, Any] = dict(release) if isinstance(release, Mapping) else {}
    reasons = prefreeze_review_reasons(prefreeze)
    if not reasons or str(record.get("status") or "") != RELEASE_READY:
        return record
    existing = [str(reason) for reason in (record.get("reasons") or ())]
    existing.append(f"{REASON_PREFREEZE_REVIEW_REQUIRED}:{','.join(reasons)}")
    record["status"] = REVIEW_REQUIRED
    record["strict_acceptance_eligible"] = False
    record["reasons"] = list(dict.fromkeys(existing))
    return record


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
    "PREFREEZE_RESOLUTION_STAGE", "REASON_PREFREEZE_REVIEW_REQUIRED",
    "DISPOSITION_EXTRACTED_NOT_SERIALIZED", "NO_DISPOSITION",
    "RELEASE_DISPOSITIONS", "SCOPE_GUARD_STOP_REASON",
    "CoverageVerdict", "ReleaseStatus",
    "coverage_verdict", "classify_release_status", "semantic_verdict", "describe",
    "prefreeze_review_reasons", "cap_release_for_prefreeze_declination",
    "release_disposition",
]
