"""The lifecycle of a Stage-3 gate report: version, phase, payload binding, authority.

WHY THIS MODULE EXISTS
----------------------
The pipeline used to record gate *failure* and never record gate *success*. The
first post-normalization gate wrote ``gate_fail_report``; the audit rounds that
repaired the payload re-ran the gates into a local variable and threw the result
away; the pre-export revalidation wrote ``final_stage3_gate_report`` only on its
failing branches. So the batch driver's verdict reduced to ``bool(a_stale_dict)``
-- a truthy dictionary describing a payload that no longer existed, which nothing
downstream was able to countermand. A leg whose payload the audit had genuinely
repaired was still filed ``FAIL``.

The rule this module encodes is the whole fix, and it is narrower than "track the
gates better":

    A strict PASS means the strict gate suite passed on the exact payload that
    shipped.

Three consequences follow, and each is a function here:

1. **A report has to say what it describes.** ``phase`` names the boundary
   (:data:`PHASE_INITIAL_POST_NORMALIZATION`, :data:`PHASE_AUDIT_ROUND`,
   :data:`PHASE_FINAL_PRE_EXPORT`) and :data:`REPORT_SCHEMA_VERSION_KEY` names the
   shape. A consumer that cannot tell an initial report from a final one cannot
   tell a superseded failure from a live one -- which is exactly how the stale
   dict acquired authority in the first place.

2. **A report has to say what it validated.** :func:`payload_sha256` binds the
   report to the object it ran on. Without it, "the gates passed" is a claim
   about an unnamed payload, and the pipeline used to have *three* candidate
   final payloads (pre-enrichment, post-enrichment, post-quarantine) that were
   serialized, exported and validated in three different combinations.

3. **A failure is superseded only by a fresh pass, never by inference.** Not by
   "the audit said it fixed it", not by a round count, not by a later stage's
   report being clean. :func:`gate_verdict` therefore reads exactly one report --
   the final one -- and fails closed when it cannot.

FAILING CLOSED IS THE DEFAULT
-----------------------------
Every uncertainty resolves to "failed". A current-version artifact set with no
final report is a bug in the producer, not a pass. So is an unsupported version,
a report describing the wrong phase, a report whose ``ok: true`` sits next to a
non-empty ``errors`` list, and a report whose payload hash does not match the
payload that shipped. The gate exists because run ``2026-07-28_0919`` shipped
PhoP as NAD+ and ``mcr genes`` as the human mineralocorticoid receptor with zero
gate errors; a bookkeeping fix that can be talked into a pass by a malformed
dictionary would be a worse bug than the one it replaced.

LEGACY ARTIFACTS
----------------
Runs recorded before this change carry no :data:`ARTIFACT_SET_VERSION_KEY`. They
genuinely cannot produce a final report, so refusing them would rewrite the
history of every archived run. They fall back to the old
``bool(gate_fail_report)`` arithmetic -- and *only* they do: the legacy branch is
reachable only for an artifact set that is stamped as pre-change.

This module imports nothing from the app or the batch driver. Both import it.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from t2pw.pipeline import canonical_hash as _canonical

__all__ = [
    "ARTIFACT_SET_VERSION",
    "ARTIFACT_SET_VERSION_KEY",
    "CANONICAL_PAYLOAD_KEY",
    "FINAL_GATE_REPORT_KEY",
    "GateVerdict",
    "INITIAL_GATE_REPORT_KEY",
    "PAYLOAD_SHA256_KEY",
    "PHASE_AUDIT_ROUND",
    "PHASE_FINAL_PRE_EXPORT",
    "PHASE_INITIAL_POST_NORMALIZATION",
    "PHASE_KEY",
    "REPORT_SCHEMA_VERSION",
    "REPORT_SCHEMA_VERSION_KEY",
    "SUPPORTED_ARTIFACT_SET_VERSIONS",
    "SUPPORTED_REPORT_SCHEMA_VERSIONS",
    "dedupe_findings",
    "finding_key",
    "gate_verdict",
    "is_current_artifact_set",
    "payload_sha256",
    "stamp_artifact_set",
    "stamp_report",
]


# ── Versions ───────────────────────────────────────────────────────────────

#: Shape of one gate/contract report. Bumped when a *consumer* would misread the
#: previous shape -- not when a field is added that older readers ignore safely.
REPORT_SCHEMA_VERSION = 1

#: Shape of the whole artifact set one leg produces. Its presence is the signal
#: that this run can be judged by :func:`gate_verdict`'s current-version rules;
#: its absence means "recorded before the report lifecycle existed".
ARTIFACT_SET_VERSION = 1

SUPPORTED_REPORT_SCHEMA_VERSIONS: frozenset = frozenset({REPORT_SCHEMA_VERSION})
SUPPORTED_ARTIFACT_SET_VERSIONS: frozenset = frozenset({ARTIFACT_SET_VERSION})

REPORT_SCHEMA_VERSION_KEY = "report_schema_version"
ARTIFACT_SET_VERSION_KEY = "artifact_set_version"
PHASE_KEY = "phase"
PAYLOAD_SHA256_KEY = "payload_sha256"


# ── Phases ─────────────────────────────────────────────────────────────────

#: The gate run inside ``normalize_process_payload``, before any audit round has
#: seen the payload. Its failures are the audit's *input*, and a report carrying
#: this phase is therefore never a verdict about what shipped.
PHASE_INITIAL_POST_NORMALIZATION = "initial_post_normalization"

#: One audit round's re-run of the same suite on that round's settled payload.
#: Progress evidence; not authoritative either, because a later round -- or the
#: remap that follows every round -- can move the payload again.
PHASE_AUDIT_ROUND = "audit_round"

#: THE authoritative phase. The strict suite re-run on the canonical payload:
#: enriched, post-quarantine, and the same object that is serialized to
#: ``final_mapped.json`` and handed to the SBML build.
PHASE_FINAL_PRE_EXPORT = "final_pre_export"

#: Artifact keys. Spelled here so the app, the driver and the tests agree without
#: any of them importing another.
FINAL_GATE_REPORT_KEY = "final_stage3_gate_report"
#: Where the *first* failure goes once it is history. ``gate_fail_report`` keeps
#: being written to disk for the audit trail; this key is what says "superseded".
INITIAL_GATE_REPORT_KEY = "initial_stage3_gate_report"
#: The object :data:`PHASE_FINAL_PRE_EXPORT` validated, verbatim.
CANONICAL_PAYLOAD_KEY = "canonical_export_payload"


# ── Hashing ────────────────────────────────────────────────────────────────


def payload_sha256(payload: Any) -> str:
    """A stable content hash of ``payload``.

    ``sort_keys`` and the tightest separators make the digest independent of dict
    ordering and of whatever indentation the serializer happened to use, so the
    same payload hashes identically whether it came out of memory or off disk.
    ``default=str`` keeps a stray non-JSON value (a ``Path``, a ``Decimal``) from
    turning a hash computation into a crash mid-pipeline; such a value would be
    unserializable in the artifact anyway.
    """

    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


# ── Stamping ───────────────────────────────────────────────────────────────


def stamp_report(
    report: Optional[Mapping[str, Any]],
    *,
    phase: str,
    payload: Any = None,
    payload_hash: Optional[str] = None,
    hash_schema: int = _canonical.HASH_SCHEMA_VERSION_LEGACY,
) -> Dict[str, Any]:
    """Return ``report`` with its version, phase and (optionally) payload binding.

    A copy, never a mutation: the caller's report is often a nested slice of a
    normalization report that other code still reads.

    ``payload_hash`` wins over ``payload`` so a caller that already hashed the
    object does not pay for a second serialization of a large payload. Passing
    neither leaves the binding off, which is correct for the two non-final phases
    -- their payloads are intermediate by definition and nothing downstream is
    entitled to treat them as what shipped.

    ``hash_schema`` defaults to 1 -- the single ``payload_sha256`` binding, byte
    for byte what this function has always written. A caller that asks for schema
    2 gets the D-001 split as well, and must hand over the payload itself: the two
    canonical projections are computed from the object, not from a digest of it.
    """

    stamped: Dict[str, Any] = dict(report or {})
    stamped[REPORT_SCHEMA_VERSION_KEY] = REPORT_SCHEMA_VERSION
    stamped[PHASE_KEY] = phase
    if payload_hash is not None:
        stamped[PAYLOAD_SHA256_KEY] = payload_hash
    elif payload is not None:
        stamped[PAYLOAD_SHA256_KEY] = payload_sha256(payload)
    if hash_schema == _canonical.HASH_SCHEMA_VERSION:
        if payload is None:
            raise ValueError("hash schema 2 needs the payload, not only its digest")
        stamped[_canonical.CANONICAL_GRAPH_SHA256_KEY] = _canonical.canonical_graph_sha256(payload)
        stamped[_canonical.CANONICAL_PAYLOAD_SHA256_KEY] = _canonical.canonical_payload_sha256(payload)
        stamped[_canonical.HASH_SCHEMA_VERSION_KEY] = _canonical.HASH_SCHEMA_VERSION
    return stamped


def stamp_artifact_set(artifacts: Dict[str, Any]) -> Dict[str, Any]:
    """Mark ``artifacts`` as produced by a build that maintains the lifecycle.

    Mutates and returns the same dict: the app hands this object straight to
    session state and the caller keeps the reference.
    """

    artifacts[ARTIFACT_SET_VERSION_KEY] = ARTIFACT_SET_VERSION
    return artifacts


def is_current_artifact_set(artifacts: Mapping[str, Any]) -> bool:
    """``True`` when the set is stamped at all -- supported or not.

    Deliberately not "supported": an artifact set stamped with a *future* version
    is still a current-lifecycle set, and must fail closed rather than fall
    through to the legacy arithmetic it was never described by.
    """

    return ARTIFACT_SET_VERSION_KEY in (artifacts or {})


# ── Findings ───────────────────────────────────────────────────────────────


def finding_key(issue: Any) -> Tuple[str, str]:
    """The identity of one finding: ``(code, json_pointer)``.

    NOT the message. Gate messages embed entity names -- ``"Protein 'ALAS2' is
    missing a resolved identifier"`` -- so keying on them makes one defect look
    like as many defects as it has victims. That is precisely what makes
    ``failures_by_code.txt`` fragment a single root cause into dozens of
    "distinct" codes, and a deduplicator with that flaw removes nothing.

    Contract reports carry ``code``/``pointer``; gate reports carry ``path`` and a
    prose ``reason``. A gate error therefore has no code of its own and keys on
    ``("", pointer)`` -- which is right: two gate errors at the same pointer from
    the same suite *are* the same finding reported twice, which is the exact
    double-count this exists to remove (PMC12856317 was filed with "2 blocking
    issue(s)" that were one ALAS2 row counted once per channel).
    """

    if isinstance(issue, str):
        return ("", issue.strip())
    data = issue if isinstance(issue, Mapping) else {}
    code = str(data.get("code") or "").strip()
    pointer = str(data.get("pointer") or data.get("path") or "").strip()
    return (code, pointer)


def dedupe_findings(*groups: Iterable[Any]) -> List[Any]:
    """Every finding across ``groups``, first occurrence wins, keyed by identity.

    Order-preserving so the first channel to report a finding is the one whose
    wording survives into the failure detail.
    """

    seen: set = set()
    merged: List[Any] = []
    for group in groups:
        for issue in group or []:
            key = finding_key(issue)
            if key in seen:
                continue
            seen.add(key)
            merged.append(issue)
    return merged


# ── Authority ──────────────────────────────────────────────────────────────

#: ``source`` values on :class:`GateVerdict`, so callers branch on a constant.
SOURCE_FINAL_REPORT = "final_report"
SOURCE_LEGACY = "legacy_artifacts"
SOURCE_FAIL_CLOSED = "fail_closed"

REASON_UNSUPPORTED_ARTIFACT_SET = "unsupported_artifact_set_version"
REASON_FINAL_REPORT_MISSING = "final_gate_report_missing"
REASON_UNSUPPORTED_REPORT_SCHEMA = "unsupported_report_schema_version"
REASON_WRONG_PHASE = "final_gate_report_wrong_phase"
REASON_MALFORMED = "final_gate_report_malformed"
REASON_PAYLOAD_UNBOUND = "final_gate_report_not_bound_to_a_payload"
REASON_PAYLOAD_MISSING = "canonical_payload_missing"
REASON_PAYLOAD_MISMATCH = "final_gate_report_payload_mismatch"
REASON_UNSUPPORTED_HASH_SCHEMA = "unsupported_hash_schema_version"
REASON_CANONICAL_GRAPH_MISMATCH = "canonical_graph_mismatch"
REASON_CANONICAL_PAYLOAD_MISMATCH_GRAPH_INTACT = "canonical_payload_mismatch_graph_intact"


@dataclass(frozen=True)
class GateVerdict:
    """The one gate answer a consumer is allowed to act on.

    ``failed`` is the *factual* gate outcome and never a policy decision. Research
    mode may continue past a failing gate; it may not restate the fact as a pass,
    so the mode check belongs to the caller and not here.
    """

    failed: bool
    errors: List[Any] = field(default_factory=list)
    source: str = SOURCE_FAIL_CLOSED
    reason: str = ""
    stage: str = ""

    @property
    def failed_closed(self) -> bool:
        return self.source == SOURCE_FAIL_CLOSED


def _as_dict(value: Any) -> Dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _as_list(value: Any) -> List[Any]:
    if isinstance(value, list):
        return list(value)
    if isinstance(value, tuple):
        return list(value)
    return []


def _fail_closed(reason: str, stage: str = "") -> GateVerdict:
    return GateVerdict(
        failed=True,
        errors=[{"path": "/", "reason": f"gate lifecycle: {reason}"}],
        source=SOURCE_FAIL_CLOSED,
        reason=reason,
        stage=stage or "final_pre_export_gate_lifecycle",
    )


def gate_verdict(artifacts: Mapping[str, Any]) -> GateVerdict:
    """Derive the blocking gate outcome ONCE, from the reports that are current.

    Current-version artifact sets are judged by :data:`FINAL_GATE_REPORT_KEY`
    alone. ``gate_fail_report`` / :data:`INITIAL_GATE_REPORT_KEY` are history and
    take no part in the arithmetic -- OR-ing them in is the defect: a stale dict
    from before the audit repaired the payload has no artifact able to countermand
    it, so no repair can ever produce a pass.

    Legacy sets keep the old behaviour exactly, because they have nothing else to
    be judged by.
    """

    artifacts = artifacts or {}
    final_report = _as_dict(artifacts.get(FINAL_GATE_REPORT_KEY))

    if not is_current_artifact_set(artifacts):
        # ── Legacy: the pre-change arithmetic, unchanged. ──
        gate_fail = _as_dict(artifacts.get("gate_fail_report"))
        failed = bool(
            artifacts.get("gate_failed")
            or artifacts.get("normalization_gate_failed")
            or gate_fail
            or (final_report and final_report.get("ok") is False)
        )
        return GateVerdict(
            failed=failed,
            errors=[*_as_list(gate_fail.get("errors")), *_as_list(final_report.get("errors"))],
            source=SOURCE_LEGACY,
            reason="legacy_artifact_set" if failed else "",
            stage=str(gate_fail.get("stage") or final_report.get("stage") or ""),
        )

    version = artifacts.get(ARTIFACT_SET_VERSION_KEY)
    if version not in SUPPORTED_ARTIFACT_SET_VERSIONS:
        return _fail_closed(f"{REASON_UNSUPPORTED_ARTIFACT_SET}:{version!r}")

    if not final_report:
        # A current-version run that produced no final report did not measure what
        # it shipped. That is a producer bug, and the safe reading of "unmeasured"
        # is "failed".
        return _fail_closed(REASON_FINAL_REPORT_MISSING)

    report_version = final_report.get(REPORT_SCHEMA_VERSION_KEY)
    if report_version not in SUPPORTED_REPORT_SCHEMA_VERSIONS:
        return _fail_closed(f"{REASON_UNSUPPORTED_REPORT_SCHEMA}:{report_version!r}")

    phase = str(final_report.get(PHASE_KEY) or "")
    if phase != PHASE_FINAL_PRE_EXPORT:
        # A correct report about the wrong boundary. An initial or audit-round
        # report parked under the final key would otherwise let a pass on an
        # intermediate payload speak for the one that shipped.
        return _fail_closed(f"{REASON_WRONG_PHASE}:{phase or '(none)'}")

    errors = _as_list(final_report.get("errors"))
    ok = final_report.get("ok")
    if ok is True and errors:
        return _fail_closed(REASON_MALFORMED)

    hash_schema = final_report.get(
        _canonical.HASH_SCHEMA_VERSION_KEY, _canonical.HASH_SCHEMA_VERSION_LEGACY
    )
    if hash_schema not in _canonical.SUPPORTED_HASH_SCHEMA_VERSIONS:
        return _fail_closed(f"{REASON_UNSUPPORTED_HASH_SCHEMA}:{hash_schema!r}")
    # (recorded key, how to recompute it, why disagreement is fatal). Schema 2
    # keeps its two failures apart instead of collapsing them into
    # REASON_PAYLOAD_MISMATCH: "the exported biology is not the gated biology" and
    # "the evidence record moved, the biology did not" are different findings about
    # different guarantees. Both still fail closed -- a report about a payload that
    # is not the one in hand is not a statement about what shipped, whichever half
    # of it moved.
    checks = (
        [
            (_canonical.CANONICAL_GRAPH_SHA256_KEY, _canonical.canonical_graph_sha256,
             REASON_CANONICAL_GRAPH_MISMATCH),
            (_canonical.CANONICAL_PAYLOAD_SHA256_KEY, _canonical.canonical_payload_sha256,
             REASON_CANONICAL_PAYLOAD_MISMATCH_GRAPH_INTACT),
        ]
        if hash_schema == _canonical.HASH_SCHEMA_VERSION
        else [(PAYLOAD_SHA256_KEY, payload_sha256, REASON_PAYLOAD_MISMATCH)]
    )
    if not all(str(final_report.get(key) or "") for key, _, _ in checks):
        return _fail_closed(REASON_PAYLOAD_UNBOUND)
    canonical = artifacts.get(CANONICAL_PAYLOAD_KEY)
    if not isinstance(canonical, Mapping) or not canonical:
        return _fail_closed(REASON_PAYLOAD_MISSING)
    for key, digest, reason in checks:
        if digest(canonical) != str(final_report.get(key) or ""):
            return _fail_closed(reason)

    failed = ok is not True
    return GateVerdict(
        failed=failed,
        errors=errors,
        source=SOURCE_FINAL_REPORT,
        reason="final_gate_failed" if failed else "",
        stage=str(final_report.get("stage") or PHASE_FINAL_PRE_EXPORT),
    )


def blocking_findings(
    verdict: GateVerdict,
    contract_findings: Sequence[Any] = (),
) -> List[Any]:
    """The findings that block a run, counted once each.

    Two channels report independently -- the gate suite and the stage contracts --
    and they overlap: the same ALAS2 registry error arrives once as a gate error
    and once as a contract error derived from it. Both channels still *block*
    independently; what this changes is the number a human is told, which was
    ``2 blocking issue(s)`` for a single row.
    """

    return dedupe_findings(verdict.errors, contract_findings)
