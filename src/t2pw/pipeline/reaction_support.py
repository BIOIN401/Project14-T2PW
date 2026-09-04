"""F-179 — a canonical pathway must have defensible REACTION-LEVEL support to export.

``PRODUCT_CONTRACT`` § 2 already requires "no unsupported retained reactions" and § 3
requires every externally added process to identify the stage that introduced it and its
supporting source.

**THIS MODULE IS A LEG-LEVEL FLOOR UNDER THOSE CLAUSES, NOT PER-ROW ENFORCEMENT OF THEM.**
It refuses a pathway in which NO reaction has defensible reaction-level support. A payload
with a defensible core plus unattributed rows is allowed through — measured on the
archives, 198 unattributed rows across the committed legs are retained on that basis, and
13 of the 28 rows the discovery census flagged sit in legs this rule permits. Saying it
enforces § 2 outright would overstate it; the narrower per-row question is left open
deliberately, because the only per-row remedy available at an exporter is deletion and
merge rule 8 forbids that.

THE DEFECT IT EXISTS FOR (F-179, ruled a REPEATED PRODUCTION DEFECT / FALSE-POSITIVE
BIOLOGICAL EXPORT). ``PMC12180156`` delivered a canonical payload containing exactly one
reaction, ``glycine -> heme`` — an eight-step pathway collapsed into a single step that
no source states. Stage 1 extracted **zero** reactions for that leg; the row first
appears in the merged payload carrying **no ``provenance_lineage`` and no
``rag_provenance``**. Its participants are paper-stated and database-grounded, and that
is exactly the inference the contract forbids: glycine and heme both resolving to real
identities is evidence about *entities*, never evidence that the *reaction* occurs. The
runtime nevertheless recorded ``semantic_evaluation: passed`` and serialized a PWML, and
the same mechanism reached export on four separate runs across a month.

=============================================================================
THE RULE, AND WHAT IT DELIBERATELY IS NOT
=============================================================================

A reaction row is SUPPORTED when its own row-level provenance carries either:

  **A. target-paper reaction support** — a lineage entry with ``origin="paper_stated"``
     or ``paper_explicit="explicit"``. A stage typed the assertion that the paper states
     this reaction.
  **B. external RAG reaction support** — a lineage entry with ``origin="rag_literature"``,
     or the row-level ``rag_provenance`` carrier. Such rows already passed the admission
     gate's reaction-specificity, pathway and organism checks; this module does not
     re-litigate them.

**C. Deterministic inherited support falls out of the shape of the test rather than
needing a rule of its own.** Support is ANY qualifying entry in the lineage, never the
LAST one, so a paper-stated reaction later rewritten by identifier mapping, normalization
or canonicalization stays supported — its ``paper_stated`` entry is still there, because
:mod:`t2pw.pipeline.lineage` is append-only. That is the distinction the ruling draws:
*identifier mapping modified a supported reaction* is valid; *identifier mapping is the
only discoverable reason the reaction exists* is not.

**NEVER support, on their own** — and this list is the ruling's, restated so it cannot
drift: ``identifier_mapping`` · ``database_grounded`` · a ChEBI, KEGG, DrugBank, CAS,
HMDB, UniProt or PathBank identifier · successful entity normalization · the presence of
the substrate or product names elsewhere in the graph. Identifier and database mapping
prove IDENTITY, not reaction occurrence. (A database record that is itself a REACTION
record would be reaction-level evidence; an entity's accession is not, and only the
latter is what ``identifier_mapping`` produces.)

**THIS IS NOT THE DISCOVERY CENSUS.** The census that found F-179 used "terminal
product", "precursor → terminal shortcut" and gold-signature matching. Those are
detection heuristics and they are deliberately absent here. There is no pathway name, no
paper id, no gold reaction, no reaction-count threshold and no chemistry in this module.
It reads provenance and nothing else.

=============================================================================
WHY A LEG-LEVEL REFUSAL AND NOT A PER-ROW DELETION
=============================================================================

Permanent merge rule 8 forbids an exporter repairing biology after the canonical graph is
frozen, and the ruling forbids silently deleting biology where deletion changes pathway
meaning. Dropping the single row of a one-reaction payload would not be a repair, it
would be deleting the pathway. So the check refuses SERIALIZATION and leaves the payload
untouched. No new output state is invented, and ``review_required`` is not used to ship
knowingly unsupported chemistry.

**HOW THE REFUSAL ACTUALLY REACHES THE EXPORTER, and this is exact because getting it
wrong once already shipped an unfixed defect.** ``validate_pre_export`` writes the issue
into ``report["pwml_contract_report"]`` and sets that inner report's ``ok`` to ``False``.
The production caller catches the ``StageContractError`` and then branches on **that inner
report** — not on the outer one — returning ``ok=False`` with an empty ``output_path`` so
no PWML is written, and persisting the same mapping as
``pwml_required_field_gate_report.json`` so the refusal is auditable on disk. An earlier
revision raised on the OUTER report only; the inner ``ok`` stayed ``True``, the caller
fell through, and the payload was still serialized while eighteen focused tests passed.
Independent review caught it. ``tests/test_f179_reaction_support.py`` now asserts the
inner report directly, which is the variable that decides.

It fires only when **NO** reaction in the payload is supported. A payload with a
defensible core plus some unattributed rows is NOT blocked here — that is a narrower
per-row question this module deliberately leaves open rather than answering with a
per-row deletion it is not allowed to make.

=============================================================================
THE ARCHIVAL-UNCERTAINTY GUARD — NOT EVERY ABSENCE IS A VIOLATION
=============================================================================

:func:`lineage_carrier_active` exists so that missing provenance is not read as proved
invention. Runs predating ``provenance_lineage`` carry none anywhere, and a payload from
one of those cannot be judged by a rule that reads lineage: the answer is
INDETERMINATE, and the check abstains. Only a payload where the carrier is demonstrably
running — some row somewhere has lineage — while **no reaction** carries A or B support
is a positive finding.

Measured over the committed and preserved archives (124 canonical legs with at least one
reaction): **89 allowed · 23 indeterminate (carrier not running) · 12 blocked**. Every
one of the 12 is a leg of ``PMC12180156`` or ``PMC13231680`` — the two gold cases whose
``export_rationale`` independently says nothing is exportable — and five of them had
exported a PWML. Neither ``PMC12096016`` nor ``PMC12782028``, the two gold
``strict_exportable`` cases, is blocked in any run where the carrier was active.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Optional

#: Issue code raised at the pre-export contract.
#:
#: ON EXPORT MODE, STATED AS MEASURED RATHER THAN ASSUMED. This code is not in
#: ``FORMAT_ISSUE_CODES``, so *when the contract is run through*
#: ``stage_contracts.run_stage_contract``, a research run re-severities it to a review
#: flag and continues. **The production PWML caller does NOT route this boundary through
#: ``run_stage_contract``** -- it calls ``validate_pre_export`` directly -- so at that
#: seam no relaxation happens in either mode. That costs nothing in practice, because
#: ``batch.runner`` maps PWML deliverables to STRICT mode only and a research leg emits
#: no PWML to refuse. The distinction is written down because assuming the asymmetry
#: existed here, rather than checking, is the kind of claim that ships a hole.
CODE_NO_DEFENSIBLE_REACTION_SUPPORT = "no_defensible_reaction_support"

#: Lineage origins that constitute reaction-level support, by class.
TARGET_PAPER_ORIGINS = ("paper_stated",)
EXTERNAL_RAG_ORIGINS = ("rag_literature",)

SUPPORT_TARGET_PAPER = "target_paper"
SUPPORT_EXTERNAL_RAG = "external_rag"


def _entries(row: Any) -> List[Mapping[str, Any]]:
    lin = row.get("provenance_lineage") if isinstance(row, Mapping) else None
    if not isinstance(lin, list):
        return []
    return [e for e in lin if isinstance(e, Mapping)]


def reaction_support_class(row: Any) -> Optional[str]:
    """``target_paper`` / ``external_rag`` / ``None`` for one reaction row.

    Reads the ROW's own provenance only. Entity provenance is never consulted: the
    ruling is explicit that "presence of substrate/product names elsewhere in the graph"
    does not establish that a reaction occurs, and F-179 is exactly that inference —
    both participants of ``glycine -> heme`` are paper-stated entities.

    ANY qualifying entry counts, never only the newest, so a supported reaction later
    rewritten by identifier mapping or normalization keeps its support.
    """

    if not isinstance(row, Mapping):
        return None
    for entry in _entries(row):
        if entry.get("origin") in TARGET_PAPER_ORIGINS or entry.get("paper_explicit") == "explicit":
            return SUPPORT_TARGET_PAPER
    for entry in _entries(row):
        if entry.get("origin") in EXTERNAL_RAG_ORIGINS:
            return SUPPORT_EXTERNAL_RAG
    # The row-level RAG carrier (``pipeline._RAG_ROW_CARRIER_KEYS``). A row carrying it
    # was admitted by the RAG gate, which already applied reaction-specificity.
    if row.get("rag_provenance"):
        return SUPPORT_EXTERNAL_RAG
    return None


def _iter_rows(payload: Any) -> Iterable[Mapping[str, Any]]:
    processes = payload.get("processes") if isinstance(payload, Mapping) else None
    rows = (processes or {}).get("reactions") if isinstance(processes, Mapping) else None
    for row in rows or []:
        if isinstance(row, Mapping):
            yield row


def lineage_carrier_active(payload: Any) -> bool:
    """Whether ANY row in the payload carries lineage.

    The archival-uncertainty guard. A payload from a run that predates the
    ``provenance_lineage`` carrier has none anywhere, and absence there is a fact about
    the run, not about the biology. Entities are checked as well as reactions because
    ``_clean_entities`` copied entity rows key-for-key long before the process buckets
    carried lineage, so entity lineage is the earliest reliable signal that the carrier
    was running at all.
    """

    if not isinstance(payload, Mapping):
        return False
    entities = payload.get("entities")
    if isinstance(entities, Mapping):
        for rows in entities.values():
            for row in rows or []:
                if isinstance(row, Mapping) and row.get("provenance_lineage"):
                    return True
    for row in _iter_rows(payload):
        if row.get("provenance_lineage"):
            return True
    return False


def evaluate_reaction_support(payload: Any) -> Dict[str, Any]:
    """Per-payload verdict. Pure: reads the payload and mutates nothing.

    ``verdict`` is one of ``supported`` (a defensible core exists),
    ``indeterminate`` (no reactions, or the lineage carrier was not running) and
    ``no_defensible_core`` (the carrier was running and NO reaction carries support).
    Only the last is a violation; the three are kept apart by name because collapsing
    "we cannot tell" into "it is unsupported" is the failure D-091 was withdrawn for.
    """

    rows = list(_iter_rows(payload))
    classes = [reaction_support_class(r) for r in rows]
    supported = [c for c in classes if c]
    result: Dict[str, Any] = {
        "reactions": len(rows),
        "supported": len(supported),
        "target_paper_supported": sum(1 for c in classes if c == SUPPORT_TARGET_PAPER),
        "external_rag_supported": sum(1 for c in classes if c == SUPPORT_EXTERNAL_RAG),
        "unattributed": sum(1 for c in classes if not c),
        "lineage_carrier_active": lineage_carrier_active(payload),
    }
    if not rows:
        result["verdict"] = "indeterminate"
        result["reason"] = "the payload carries no reactions; nothing to support"
    elif supported:
        result["verdict"] = "supported"
        result["reason"] = (f"{len(supported)} of {len(rows)} reactions carry "
                            f"target-paper or external-RAG reaction-level support")
    elif not result["lineage_carrier_active"]:
        result["verdict"] = "indeterminate"
        result["reason"] = (
            "no reaction carries recorded support, but no row anywhere in this payload "
            "carries a provenance lineage either -- the carrier was not running, so "
            "support is UNKNOWN rather than absent"
        )
    else:
        result["verdict"] = "no_defensible_core"
        result["reason"] = (
            f"the provenance carrier is active in this payload, yet NONE of its "
            f"{len(rows)} reaction(s) carries target-paper or external-RAG "
            f"reaction-level support. Entity identity, database grounding and "
            f"identifier mapping do not establish that a reaction occurs "
            f"(PRODUCT_CONTRACT 2 and 3)"
        )
    return result


def reaction_support_issue(payload: Any) -> Optional[Dict[str, Any]]:
    """The pre-export contract issue for this payload, or ``None`` when there is none."""

    report = evaluate_reaction_support(payload)
    if report["verdict"] != "no_defensible_core":
        return None
    return {
        "code": CODE_NO_DEFENSIBLE_REACTION_SUPPORT,
        "message": (
            "No exported reaction has defensible reaction-level support: "
            + report["reason"]
        ),
        "pointer": "/processes/reactions",
        "support_report": report,
    }
