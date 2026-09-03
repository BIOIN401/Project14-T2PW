"""R-D092-1 -- evaluation-only ROW-LEVEL RAG lineage for canonical reactions.

CHARTERED BY D-093 section 3. EVALUATION-ONLY: read-only over archived run
directories, imports no production module, writes nothing into any run, and produces
NO acceptance verdict. Running it is not a re-score of T-107/T-108/T-109 and does not
touch their dispositions. It changes no runtime behaviour, in keeping with D-093
section 2, which DENIED the F-176 runtime change.

WHAT IT ANSWERS. For each canonical reaction: where did this row come from, and does
its preserved lineage defend it? D-093 section 1 requires THREE support classes rather
than two, because a correctly-attributed cross-paper RAG reaction is NOT hallucinated.

=============================================================================
FIVE MEASURED FACTS THAT DETERMINE THIS MODULE'S DESIGN
=============================================================================

Every one of these was measured on the committed corpus before a line of the
classifier was written. Each of them defeats an obvious implementation.

(1) THE BRIEF'S PREMISE IS OUT OF DATE, AND IN THE CHARTERED DIRECTION.
    "rag_provenance today lives on ENTITIES, not on reaction rows" is true of the
    2026-07-27 run the brief cites, and false of the corpus as a whole. Measured
    across 187 committed payload files: 236 of 1079 reaction rows carry
    ``rag_provenance`` and 431 carry ``provenance_lineage``. ``pipeline.py``'s
    ``_carry_rag_provenance`` shipped that carrier. So this module must handle THREE
    provenance eras, not one, and reconstruct by inheritance only where it must.

(2) LINEAGE ``support="unsupported"`` IS NOT D-093's ``unsupported``.
    Measured: 426 of the 440 reaction lineage entries in the committed corpus are
    ``(origin=paper_stated, support=unsupported, paper_explicit=explicit)``.
    ``lineage.SUPPORT_LEVELS`` grades whether a NAMED SOURCE backs the row; a row the
    paper itself states names no source and is therefore ``unsupported`` in that
    vocabulary while being ``target_paper_supported`` in D-093's. Mapping one onto the
    other relabels 426 paper-explicit reactions as unsupported -- the D-091 failure
    replayed one level down. :func:`classify_support` NEVER reads ``support``.

(3) ``origin="rag_literature"`` DOES NOT MEAN EXTERNAL.
    Of the 14 ``rag_literature`` source references in the corpus, 9 carry the
    ``seed_paper`` sentinel and 2 carry the leg's own target paper id: 11 of 14 point
    AT THE TARGET PAPER. Only 3 are genuinely foreign. Externality is decided by
    resolving ``source_id`` against the leg's target paper (:func:`resolve_source`),
    never by reading ``origin``.

(4) THE REACTION ``evidence`` STRING IS NOT REACTION-SPECIFIC EVIDENCE, AND IT
    SILENTLY CARRIES EXTERNAL TEXT. On the leg the brief cites, reaction row 6's
    ``evidence`` is 35,029 characters and reaction row 5's is 108,335 -- these are
    document dumps, not quotes. Worse, row 6's text is the ABSTRACT OF PMC8091085,
    an external paper, while ``grep`` over that leg's own ``01_source_text.txt``
    returns ZERO occurrences of ``MenI``, ``DHNA-CoA thioesterase`` or
    ``LMRG_02730``. A "row has evidence, therefore the paper supports it" test passes
    every row in the corpus AND launders external text as target-paper support. This
    module therefore treats ``evidence`` as presence-only and never as attribution.

(5) PARTICIPANT INHERITANCE CANNOT ESTABLISH D-093 CONDITION 1.
    D-093 requires "direct REACTION-SPECIFIC evidence, not a span that merely names
    the participants". Entity provenance is exactly a span that names a participant:
    it records the chunk an ENTITY was matched in. Inheriting it onto a reaction
    proves the participants were retrieved, never that the chunk states the reaction.
    So inheritance alone yields ``indeterminate``, never ``external_rag_supported``.
    The ONE deterministic bridge is the chunk join below.

=============================================================================
THE INHERITANCE RULE -- ADOPTED AND DOCUMENTED, AS D-093 REQUIRES
=============================================================================

The brief requires this rule be decided and written down rather than left implicit.

ATTRIBUTION TIERS, in strict precedence order. The tier is always reported, so a
reader can see how much of a classification rests on inference:

  ``row_lineage``      the row carries ``provenance_lineage``. Authoritative:
                       lineage is typed, append-only and per-row.
  ``row_rag_provenance`` the row carries ``rag_provenance``. Authoritative.
  ``participant_inheritance`` neither row carrier is present; provenance is read off
                       the row's participant ENTITIES. INFERRED, and capped at
                       ``indeterminate`` for external sources by fact (5).
  ``no_signal``        no row carrier and no participant carries provenance.

WHY PRECEDENCE AND NOT UNION. A row carrying its own lineage has been attributed by
the stage that introduced it; its participants may have been retrieved separately for
unrelated gaps. Letting participant provenance override a row's own record would let
an entity's retrieval history rewrite a reaction's attribution.

PARTICIPANT LOOKUP IS ONE-TO-MANY, AND THAT IS NOT AN EDGE CASE. Entity names collide
within a single payload: on the cited leg ``isochorismate``, ``SEPHCHC`` and ``MenD``
each appear TWICE, once with no provenance and once carrying the ``seed_paper``
sentinel. A name lookup returning "the" entity would silently pick one. So lookup
collects the SET of provenance records for a name, and a name whose records disagree
about externality makes the row ``indeterminate`` rather than picking a winner.

SOURCE RESOLUTION. ``seed_paper`` (the sentinel documented at
``identity_admission.py:262``) and a ``source_id`` equal to the leg's target paper both
resolve to TARGET. An empty id resolves to UNRESOLVED, never to target -- absence is
not attribution. Anything else is EXTERNAL.

THE CHUNK JOIN, AND ITS HONEST LIMIT. An entity's provenance names a ``chunk_id``;
admission records carry ``evidence.chunk_id`` with the retrieved ``span`` and the
gate's own verdict. Joining them recovers the actual retrieved text behind an
inherited attribution -- measured to resolve 37 of 79 distinct entity chunks, INCLUDING
the ``fb1cf2b2...`` chunk behind the case D-091 tripped on. A join inside one leg is
``within_run``. A join across runs is ``cross_run`` and is reported as such and never
silently: retrieval is temperature-sensitive and a different run's admission verdict is
evidence about that run. Pre-carrier legs have no admission artifact at all, so for
them the join is necessarily cross-run or ``unavailable``.

=============================================================================
POPULATIONS, AND WHY THEY ARE NEVER SUMMED
=============================================================================

F-177 (``eval_semantic_populations.py``) established that ``final_mapped.json``
(CANONICAL) and ``merged_payload.json`` (FALLBACK) are different kinds of object and
that summing them produces a number nobody can act on. That discipline is extended
here, not regressed: measured, the committed corpus splits 115 canonical legs / 433
reactions against 72 fallback legs / 609 reactions. Every table below is
per-population, including empty ones.

THE CORPUS IS THE COMMITTED CORPUS, VIA ``git ls-files``. F-178 cost a wave because
two helpers named ``_committed_legs`` called ``rglob`` and measured the working tree,
going red for exactly the people who had run a benchmark. :func:`committed_paths`
asks git and nothing else.

Usage:
  python rd092_1_reaction_lineage.py <repo-root> [--run <substr> ...] [--json OUT]
"""

from __future__ import annotations

import argparse
import collections
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

# --------------------------------------------------------------------------
# Closed vocabularies. Free strings would get several spellings from several
# writers and make "how many rows are indeterminate?" unanswerable.
# --------------------------------------------------------------------------

#: D-093 section 1's three classes, plus the fourth the ruling itself names.
TARGET_PAPER_SUPPORTED = "target_paper_supported"
EXTERNAL_RAG_SUPPORTED = "external_rag_supported"
UNSUPPORTED = "unsupported"
INDETERMINATE = "indeterminate"
SUPPORT_CLASS_ORDER: Tuple[str, ...] = (
    TARGET_PAPER_SUPPORTED, EXTERNAL_RAG_SUPPORTED, UNSUPPORTED, INDETERMINATE,
)

#: Attribution tiers, strongest first. Always reported beside the class.
TIER_ROW_LINEAGE = "row_lineage"
TIER_ROW_RAG_PROVENANCE = "row_rag_provenance"
TIER_PARTICIPANT_INHERITANCE = "participant_inheritance"
TIER_NO_SIGNAL = "no_signal"
TIER_ORDER: Tuple[str, ...] = (
    TIER_ROW_LINEAGE, TIER_ROW_RAG_PROVENANCE, TIER_PARTICIPANT_INHERITANCE,
    TIER_NO_SIGNAL,
)

#: Source resolution outcomes. ``UNRESOLVED`` is deliberately not ``TARGET``.
SRC_TARGET = "target"
SRC_EXTERNAL = "external"
SRC_UNRESOLVED = "unresolved"

#: Payload populations, from F-177. Never summed.
CANONICAL = "canonical"
FALLBACK = "fallback"
POPULATION_ORDER: Tuple[str, ...] = (CANONICAL, FALLBACK)
POPULATION_BY_FILE: Dict[str, str] = {
    "final_mapped.json": CANONICAL,
    "merged_payload.json": FALLBACK,
}
#: Resolution order. ``final_mapped.json`` wins where both exist, matching the file
#: the Stage-3 gate binds to; the population is recorded, never guessed.
PAYLOAD_FILES: Tuple[str, ...] = ("final_mapped.json", "merged_payload.json")

#: The sentinel a synthesized row carries when its chunk came from the seed paper's
#: own text rather than a retrieved second document
#: (``identity_admission.py:262``). Spelled out rather than imported: this module
#: imports no production code, so an evaluation run can never perturb the runtime.
SEED_PAPER_SENTINEL = "seed_paper"

#: The join between an entity's provenance chunk and an admission record's span.
JOIN_WITHIN_RUN = "within_run"
JOIN_CROSS_RUN = "cross_run"
JOIN_NONE = "no_chunk_match"
JOIN_UNAVAILABLE = "unavailable"

#: The literal D-093 asks for wherever lineage cannot be reconstructed.
UNAVAILABLE = "unavailable"


# --------------------------------------------------------------------------
# Corpus discovery -- git, never the filesystem (F-178)
# --------------------------------------------------------------------------

def committed_paths(repo_root: Path) -> List[str]:
    """Every path tracked at HEAD, as forward-slash repo-relative strings.

    Asks git. A ``rglob`` here would sweep untracked benchmark output into the
    corpus and make this instrument's numbers depend on who had run a pipeline leg
    recently -- which is exactly F-178, and it went red only for the people who had.
    """

    out = subprocess.run(
        ["git", "ls-files"], cwd=str(repo_root), capture_output=True,
        encoding="utf-8", errors="replace",
    )
    if out.returncode != 0:
        raise SystemExit(f"git ls-files failed in {repo_root}: {out.stderr[:400]}")
    return [line.strip().replace("\\", "/") for line in out.stdout.splitlines() if line.strip()]


#: A leg directory is ``<run>/papers/<paper-dir>/<mode>``. The paper directory is
#: sometimes bare (``PMC12312563``) and sometimes slugged
#: (``PMC12312563__structures-of-listeria-...``), so the id is matched, not split.
_PAPER_ID_RE = re.compile(r"/papers/(PMC\d+)")


def target_paper_id(leg_dir: str) -> Optional[str]:
    """The leg's target paper id, or ``None`` if the path does not name one."""

    m = _PAPER_ID_RE.search(leg_dir)
    return m.group(1) if m else None


def discover_legs(paths: Sequence[str], run_filters: Sequence[str]) -> List[Dict[str, str]]:
    """Every committed leg carrying a payload, with its population recorded."""

    by_dir: Dict[str, Set[str]] = collections.defaultdict(set)
    for p in paths:
        head, _, name = p.rpartition("/")
        if name in POPULATION_BY_FILE:
            by_dir[head].add(name)

    legs: List[Dict[str, str]] = []
    for leg_dir in sorted(by_dir):
        if run_filters and not any(f in leg_dir for f in run_filters):
            continue
        present = by_dir[leg_dir]
        source = next((f for f in PAYLOAD_FILES if f in present), None)
        if source is None:  # pragma: no cover -- by_dir only holds known names
            continue
        legs.append({
            "leg_dir": leg_dir,
            "payload_file": source,
            "population": POPULATION_BY_FILE[source],
            "run": leg_dir.split("/papers/")[0],
            "target_paper": target_paper_id(leg_dir) or UNAVAILABLE,
        })
    return legs


def load_json(path: Path) -> Optional[Any]:
    """Parse a JSON artifact, or ``None`` if it is missing or malformed.

    Missing and malformed are distinguished by the caller from the return of
    :func:`artifact_state`, never by a bare ``None`` -- collapsing them is the
    error family F-177 exists for.
    """

    try:
        with path.open(encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


# --------------------------------------------------------------------------
# Source resolution
# --------------------------------------------------------------------------

def resolve_source(source_id: Any, target: Optional[str]) -> str:
    """Resolve one ``source_id`` against the leg's target paper.

    Never reads ``origin``: measured, 11 of the 14 ``rag_literature`` source
    references in the corpus point at the target paper (9 via the sentinel, 2 by id),
    so ``origin`` answers "did RAG touch this row", not "is this external".

    An empty or missing id is ``UNRESOLVED`` and deliberately not ``TARGET``: a row
    with no named source has not been attributed to the paper, it has simply not been
    attributed, and defaulting absence to the target paper is how external content
    acquires target-paper support it never earned.
    """

    if not isinstance(source_id, str) or not source_id.strip():
        return SRC_UNRESOLVED
    sid = source_id.strip()
    if sid == SEED_PAPER_SENTINEL:
        return SRC_TARGET
    if target and sid == target:
        return SRC_TARGET
    return SRC_EXTERNAL


def provenance_sources(prov: Any) -> List[Dict[str, Any]]:
    """Normalize a ``rag_provenance`` mapping to the source shape used below."""

    if not isinstance(prov, dict) or not prov:
        return []
    return [{
        "source_id": prov.get("source_id") or "",
        "source_title": prov.get("source_title") or "",
        "uri": prov.get("source_uri") or "",
        "chunk_id": prov.get("chunk_id") or "",
        "section": prov.get("section") or "",
    }]


def lineage_entries(lineage: Any) -> List[Dict[str, Any]]:
    """A row's lineage entries, or empty when the row carries none.

    THE KEY IS THE SIGNAL, NOT ITS SOURCES. Measured: 650 of the 692 reaction lineage
    entries in the committed corpus are ``(paper_stated, explicit)`` carrying ZERO
    sources, because ``lineage.py`` requires a named source only for
    ``SOURCED_ORIGINS`` -- a row the paper itself stated names no external record and
    correctly carries none. An earlier revision of this module tiered on the presence
    of SOURCES and therefore demoted all 650 typed, explicit, paper-stated
    attributions to participant inheritance or ``no_signal``, discarding the
    strongest evidence in the corpus and inflating ``indeterminate``. That is this
    project's standing defect -- a missing key read as zero -- so the presence test
    is on the entry list and the sources are read separately.
    """

    return [e for e in lineage if isinstance(e, dict)] if isinstance(lineage, list) else []


def lineage_sources(lineage: Any) -> List[Dict[str, Any]]:
    """Every source reference across a row's lineage entries, order preserved."""

    out: List[Dict[str, Any]] = []
    if not isinstance(lineage, list):
        return out
    for entry in lineage:
        if not isinstance(entry, dict):
            continue
        for src in entry.get("sources") or []:
            if not isinstance(src, dict):
                continue
            out.append({
                "source_id": src.get("source_id") or "",
                "source_title": src.get("source_title") or "",
                "uri": src.get("uri") or "",
                # ``LineageSource.locator`` is the chunk pointer; lineage names a
                # record rather than copying it (``lineage.py`` "Pointer, not copy").
                "chunk_id": src.get("locator") or "",
                "section": "",
                "stage": entry.get("stage") or "",
                "origin": entry.get("origin") or "",
                "reason": entry.get("reason") or "",
                "review_required": bool(entry.get("review_required")),
            })
    return out


# --------------------------------------------------------------------------
# Participant inheritance
# --------------------------------------------------------------------------

def entity_provenance_index(payload: Any) -> Dict[str, List[Dict[str, Any]]]:
    """Map lowercased entity name -> EVERY provenance record carried under it.

    A list, not a value, and that is load-bearing. Entity names collide inside one
    payload: on the leg the brief cites, ``isochorismate``, ``SEPHCHC`` and ``MenD``
    each appear twice, once with no provenance and once carrying the sentinel. A
    lookup returning one record would silently pick a winner; returning the set lets
    :func:`classify_support` report the disagreement as ``indeterminate``.

    Entities with NO provenance are indexed as an explicit ``None`` record, because
    "this name exists and carries nothing" and "this name is absent" are different
    facts and only the first is evidence about the row.
    """

    index: Dict[str, List[Dict[str, Any]]] = collections.defaultdict(list)
    if not isinstance(payload, dict):
        return index
    entities = payload.get("entities")
    if not isinstance(entities, dict):
        return index
    for kind, rows in entities.items():
        if not isinstance(rows, list):
            continue
        for row in rows:
            if not isinstance(row, dict):
                continue
            name = row.get("name")
            if not isinstance(name, str) or not name.strip():
                continue
            prov = row.get("rag_provenance")
            srcs = provenance_sources(prov)
            lin = lineage_sources(row.get("provenance_lineage"))
            index[name.strip().lower()].append({
                "kind": kind,
                "sources": srcs + lin,
            })
    return index


#: The enzyme-name key in each payload population. THE TWO POPULATIONS DO NOT AGREE.
#: Measured: every one of the 702 canonical enzyme records is keyed
#: ``{entity, entity_type, role, evidence, confidence, provenance}``, while fallback
#: rows key the name ``protein``. Reading only ``protein`` -- as an earlier revision
#: did -- drops the enzyme from EVERY canonical row's participant set, silently
#: shrinking the inheritance evidence for the population D-093 actually asks about.
ENZYME_NAME_KEYS: Tuple[str, ...] = ("entity", "protein", "name")


def reaction_participants(reaction: Any) -> List[str]:
    """Inputs, outputs and enzyme names -- the row's participant names.

    Handles both payload schemas; see :data:`ENZYME_NAME_KEYS` for why that is not
    defensive coding but a measured difference between the two populations.
    """

    names: List[str] = []
    if not isinstance(reaction, dict):
        return names
    for key in ("inputs", "outputs"):
        for v in reaction.get(key) or []:
            if isinstance(v, str) and v.strip():
                names.append(v.strip())
            elif isinstance(v, dict):
                for nk in ("name", "entity"):
                    if isinstance(v.get(nk), str) and v[nk].strip():
                        names.append(v[nk].strip())
                        break
    for enz in reaction.get("enzymes") or []:
        if isinstance(enz, dict):
            for nk in ENZYME_NAME_KEYS:
                p = enz.get(nk)
                if isinstance(p, str) and p.strip():
                    names.append(p.strip())
                    break
        elif isinstance(enz, str) and enz.strip():
            names.append(enz.strip())
    return names


def enzyme_names(reaction: Any) -> List[Any]:
    """Enzyme names for the record, across both payload schemas."""

    out: List[Any] = []
    for enz in (reaction.get("enzymes") or []) if isinstance(reaction, dict) else []:
        if isinstance(enz, dict):
            out.append(next((enz[k] for k in ENZYME_NAME_KEYS
                             if isinstance(enz.get(k), str) and enz[k].strip()), UNAVAILABLE))
        else:
            out.append(enz)
    return out


def enzyme_extraction_modes(reaction: Any) -> List[str]:
    """The canonical enzyme ``provenance`` field -- ``extracted`` or ``inferred``.

    A DIFFERENT VOCABULARY WEARING THE SAME WORD, and the reason it is reported under
    a name that cannot be mistaken for source attribution. Measured across the
    canonical population: 612 ``extracted`` and 90 ``inferred``. This field records
    HOW the enzyme was obtained, not WHERE it came from; ``rag_provenance`` records
    the source. Folding an ``inferred`` enzyme into source attribution -- or reading
    ``extracted`` as "the paper stated it" -- would be the same category error as
    reading lineage ``support`` as a biological verdict.
    """

    modes: List[str] = []
    for enz in (reaction.get("enzymes") or []) if isinstance(reaction, dict) else []:
        if isinstance(enz, dict) and isinstance(enz.get("provenance"), str):
            modes.append(enz["provenance"])
    return modes


# --------------------------------------------------------------------------
# The chunk join
# --------------------------------------------------------------------------

def admission_index(paths: Sequence[str], repo_root: Path,
                    run_filters: Sequence[str]) -> Dict[str, List[Dict[str, Any]]]:
    """Map ``chunk_id`` -> every committed admission record carrying it.

    Both ``accepted`` and ``rejected`` are indexed, and the group is recorded on each
    record: the gate's verdict on a chunk is the whole point of the join.
    """

    index: Dict[str, List[Dict[str, Any]]] = collections.defaultdict(list)
    for p in paths:
        if not p.endswith("/rag_admission_report.json"):
            continue
        if run_filters and not any(f in p for f in run_filters):
            continue
        doc = load_json(repo_root / p)
        if not isinstance(doc, dict):
            continue
        leg_dir = p.rpartition("/")[0]
        for group in ("accepted", "rejected"):
            for rec in doc.get(group) or []:
                if not isinstance(rec, dict):
                    continue
                ev = rec.get("evidence")
                if not isinstance(ev, dict):
                    continue
                cid = ev.get("chunk_id")
                if not isinstance(cid, str) or not cid:
                    continue
                index[cid].append({
                    "leg_dir": leg_dir,
                    "run": leg_dir.split("/papers/")[0],
                    "group": group,
                    "gap_id": rec.get("gap_id") or UNAVAILABLE,
                    "candidate_name": rec.get("name") or UNAVAILABLE,
                    "inputs": rec.get("inputs") or [],
                    "outputs": rec.get("outputs") or [],
                    "enzymes": rec.get("enzymes") or [],
                    "source_paper": (rec.get("source_paper") or {}).get("source_id") or UNAVAILABLE,
                    "section": ev.get("section") or UNAVAILABLE,
                    "score": ev.get("score", UNAVAILABLE),
                    "span": ev.get("span") or UNAVAILABLE,
                    "reasons": rec.get("reasons") or [],
                    "requested_pathway_match": rec.get("requested_pathway_match") or UNAVAILABLE,
                    "organism_match": rec.get("organism_match") or UNAVAILABLE,
                    "scope_membership": rec.get("scope_membership") or UNAVAILABLE,
                })
    return index


def _norm_set(values: Iterable[Any]) -> Set[str]:
    return {str(v).strip().lower() for v in values if str(v).strip()}


def join_chunk(chunk_id: str, reaction: Any, leg_dir: str,
               index: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
    """Recover the retrieved span and gate verdict behind one chunk pointer.

    Returns the join scope (``within_run`` / ``cross_run`` / ``no_chunk_match`` /
    ``unavailable``) alongside the matching records. A match is ``reaction_specific``
    only when a record on that chunk carries THIS row's input and output sets --
    D-093 condition 1 is "direct reaction-specific evidence, not a span that merely
    names the participants", and a chunk match alone is exactly the latter.
    """

    if not chunk_id:
        return {"scope": JOIN_UNAVAILABLE, "records": [], "reaction_specific": False,
                "gate_verdicts": []}
    records = index.get(chunk_id) or []
    if not records:
        return {"scope": JOIN_NONE, "records": [], "reaction_specific": False,
                "gate_verdicts": []}

    run = leg_dir.split("/papers/")[0]
    within = [r for r in records if r["run"] == run]
    chosen = within or records
    scope = JOIN_WITHIN_RUN if within else JOIN_CROSS_RUN

    want_in = _norm_set(reaction.get("inputs") or [])
    want_out = _norm_set(reaction.get("outputs") or [])
    specific: List[Dict[str, Any]] = []
    for r in chosen:
        if want_in and want_out and _norm_set(r["inputs"]) == want_in and \
                _norm_set(r["outputs"]) == want_out:
            specific.append(r)

    return {
        "scope": scope,
        "records": chosen,
        "matched_records": specific,
        "reaction_specific": bool(specific),
        "gate_verdicts": sorted({r["group"] for r in specific}) or
                         sorted({r["group"] for r in chosen}),
    }


# --------------------------------------------------------------------------
# Classification
# --------------------------------------------------------------------------

def classify_support(tier: str, resolutions: Sequence[str], join: Dict[str, Any],
                     has_evidence: bool,
                     paper_explicit: bool = False) -> Tuple[str, str]:
    """The D-093 support class for one row, with the reason that produced it.

    NEVER reads lineage ``support``. Measured fact (2) in the module docstring: 650
    of 692 reaction lineage entries in the corpus are ``support=unsupported`` while
    being ``paper_explicit=explicit``, because that field grades whether a NAMED
    SOURCE backs the row, not whether the biology is defensible. Reading it would
    relabel every paper-stated reaction as unsupported, which is precisely the
    collapse D-091 was withdrawn for.

    It DOES read ``paper_explicit``, which is the orthogonal three-valued field
    ``lineage.py`` provides for exactly this question: ``explicit`` is a stage's
    typed assertion that the paper stated this row, and ``not_evaluated`` is
    explicitly never ``not_explicit``.
    """

    res = set(resolutions)

    # A typed, explicit paper_stated attribution outranks everything below: it is the
    # extraction stage asserting the TARGET PAPER stated this row, which is what
    # ``target_paper_supported`` means. Checked before the external branches because a
    # row can carry both a paper_stated entry and a later RAG entry, and the paper
    # having stated it does not stop being true when RAG also retrieved it.
    if tier == TIER_ROW_LINEAGE and paper_explicit and SRC_EXTERNAL not in res:
        return TARGET_PAPER_SUPPORTED, "row lineage records paper_explicit=explicit (paper_stated)"

    if tier == TIER_NO_SIGNAL:
        # Nothing attributes this row anywhere. A Stage-1 row from a pre-carrier era
        # leg looks exactly like this. Presence of an evidence string is the only
        # positive signal available -- and by measured fact (4) it is weak, so it
        # buys ``target_paper_supported`` and never more. With no evidence at all the
        # honest answer is ``indeterminate``: absence of provenance is not proof of
        # anything, and D-093 asks for that word rather than a guess.
        if has_evidence:
            return TARGET_PAPER_SUPPORTED, "no provenance carrier; row carries a Stage-1 evidence string"
        return INDETERMINATE, "no provenance carrier and no evidence string"

    if res == {SRC_TARGET}:
        return TARGET_PAPER_SUPPORTED, f"all sources resolve to the target paper ({tier})"

    if SRC_EXTERNAL not in res:
        # No external source, and no explicit paper_stated assertion either. Either
        # the ids present resolve to nothing, or the carrier holds only entries that
        # attribute nothing (``audit_modified`` records a modification, not an
        # origin). Both are genuinely indeterminate and are named apart.
        if not res:
            return INDETERMINATE, f"provenance carrier present but names no source ({tier})"
        return INDETERMINATE, f"sources present but none resolvable ({tier})"

    # From here the row has at least one external source.
    if tier == TIER_PARTICIPANT_INHERITANCE:
        # Measured fact (5): entity provenance records the chunk an ENTITY was
        # matched in, so inheriting it proves the participants were retrieved and
        # never that the chunk states the reaction. The chunk join is the only
        # deterministic bridge, and only when it lands on a record whose inputs and
        # outputs are THIS row's.
        if join.get("reaction_specific"):
            verdicts = join.get("gate_verdicts") or []
            if verdicts == ["accepted"]:
                return (EXTERNAL_RAG_SUPPORTED,
                        f"inherited external source; chunk join ({join['scope']}) found an "
                        f"ACCEPTED reaction-specific admission record")
            if "accepted" in verdicts:
                return (INDETERMINATE,
                        f"inherited external source; chunk join ({join['scope']}) found both "
                        f"accepted and rejected reaction-specific records")
            if join["scope"] == JOIN_CROSS_RUN:
                # A rejection recorded in a DIFFERENT run is evidence about that run.
                # Retrieval is temperature-sensitive and the sprint's standing trap is
                # explicit that identical legs give materially different draws, so a
                # cross-run rejection does not license calling THIS row unsupported.
                # The rejection is still carried on the record for a reader.
                return (INDETERMINATE,
                        "inherited external source; the only reaction-specific admission "
                        "records were REJECTED in a DIFFERENT run (cross_run) -- recorded, "
                        "not charged against this row")
            return (UNSUPPORTED,
                    f"inherited external source; every reaction-specific admission record on "
                    f"this chunk was REJECTED ({join['scope']})")
        return (INDETERMINATE,
                f"external participant provenance, but no reaction-specific evidence "
                f"recoverable (chunk join: {join.get('scope', JOIN_UNAVAILABLE)})")

    # Row-level carrier naming an external source: lineage is preserved by
    # construction (D-093 condition 4). Conditions 1-3 still need the retrieved
    # record, so the join decides between supported and indeterminate.
    if join.get("reaction_specific"):
        verdicts = join.get("gate_verdicts") or []
        if "accepted" in verdicts:
            return (EXTERNAL_RAG_SUPPORTED,
                    f"row-level external lineage with an ACCEPTED reaction-specific "
                    f"admission record ({join['scope']})")
        if join["scope"] == JOIN_CROSS_RUN:
            return (INDETERMINATE,
                    "row-level external lineage; the only reaction-specific admission "
                    "records were REJECTED in a DIFFERENT run (cross_run) -- recorded, "
                    "not charged against this row")
        return (UNSUPPORTED,
                f"row-level external lineage; every reaction-specific admission record on "
                f"this chunk was REJECTED ({join['scope']})")
    return (INDETERMINATE,
            f"row-level external lineage, but no reaction-specific retrieved evidence "
            f"recoverable (chunk join: {join.get('scope', JOIN_UNAVAILABLE)})")


def attribute_row(reaction: Dict[str, Any], entity_index: Dict[str, List[Dict[str, Any]]],
                  target: Optional[str]) -> Tuple[str, List[Dict[str, Any]], Dict[str, Any]]:
    """Pick the attribution tier and collect the sources it yields.

    Precedence, never union: a row carrying its own lineage has been attributed by
    the stage that introduced it, and its participants may have been retrieved for
    unrelated gaps. Letting participant provenance override the row's own record
    would let an entity's retrieval history rewrite a reaction's attribution.
    """

    detail: Dict[str, Any] = {}

    entries = lineage_entries(reaction.get("provenance_lineage"))
    if entries:
        # Tiered on the ENTRIES, not on their sources -- see :func:`lineage_entries`
        # for the 650-row defect that distinction fixes.
        detail["lineage_paper_explicit"] = any(
            e.get("paper_explicit") == "explicit" for e in entries)
        detail["lineage_stages"] = sorted({e.get("stage") for e in entries if e.get("stage")})
        detail["lineage_origins"] = sorted({e.get("origin") for e in entries if e.get("origin")})
        return TIER_ROW_LINEAGE, lineage_sources(reaction.get("provenance_lineage")), detail

    prov = provenance_sources(reaction.get("rag_provenance"))
    if prov:
        return TIER_ROW_RAG_PROVENANCE, prov, detail

    sources: List[Dict[str, Any]] = []
    per_participant: List[Dict[str, Any]] = []
    for name in reaction_participants(reaction):
        records = entity_index.get(name.strip().lower())
        if records is None:
            per_participant.append({"name": name, "entity_found": False, "sources": []})
            continue
        found: List[Dict[str, Any]] = []
        for rec in records:
            found.extend(rec["sources"])
        per_participant.append({
            "name": name, "entity_found": True,
            "resolutions": sorted({resolve_source(s.get("source_id"), target) for s in found}),
            "sources": found,
        })
        sources.extend(found)
    detail["participants"] = per_participant
    if sources:
        return TIER_PARTICIPANT_INHERITANCE, sources, detail
    return TIER_NO_SIGNAL, [], detail


def build_record(reaction: Dict[str, Any], leg: Dict[str, str], row_index: int,
                 entity_index: Dict[str, List[Dict[str, Any]]],
                 adm_index: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
    """One D-093 section 3 lineage record for one canonical reaction row."""

    target = None if leg["target_paper"] == UNAVAILABLE else leg["target_paper"]
    tier, sources, detail = attribute_row(reaction, entity_index, target)

    resolutions = [resolve_source(s.get("source_id"), target) for s in sources]
    chunk_ids = [s.get("chunk_id") for s in sources if s.get("chunk_id")]
    # Join on the first external-bearing chunk; a row whose externality comes from
    # several chunks is rare and the record keeps them all for a reader.
    join_chunk_id = ""
    for src, res in zip(sources, resolutions):
        if res == SRC_EXTERNAL and src.get("chunk_id"):
            join_chunk_id = src["chunk_id"]
            break
    if not join_chunk_id and chunk_ids:
        join_chunk_id = chunk_ids[0]
    join = join_chunk(join_chunk_id, reaction, leg["leg_dir"], adm_index)

    ev = reaction.get("evidence")
    has_evidence = isinstance(ev, str) and bool(ev.strip())
    support_class, reason = classify_support(
        tier, resolutions, join, has_evidence,
        paper_explicit=bool(detail.get("lineage_paper_explicit")))

    matched = join.get("matched_records") or []
    first = matched[0] if matched else None

    return {
        # identity
        "run": leg["run"],
        "leg_dir": leg["leg_dir"],
        "population": leg["population"],
        "payload_file": leg["payload_file"],
        "target_paper": leg["target_paper"],
        "row_index": row_index,
        "reaction_name": reaction.get("name") or UNAVAILABLE,
        "inputs": reaction.get("inputs") or [],
        "outputs": reaction.get("outputs") or [],
        "enzymes": enzyme_names(reaction),
        # HOW the enzyme was obtained, never WHERE it came from -- see
        # :func:`enzyme_extraction_modes` for why this is not source attribution.
        "enzyme_extraction_modes": enzyme_extraction_modes(reaction) or UNAVAILABLE,
        # The row's own scope verdict, which is D-093 condition 2 (pathway/scope
        # compatibility) as the pipeline recorded it. Kept apart from the admission
        # record's ``scope_membership`` below: one is about the row in its pathway,
        # the other about the retrieved candidate.
        "row_scope_membership": reaction.get("scope_membership", UNAVAILABLE),
        "locked_reaction_id": reaction.get("locked_reaction_id", UNAVAILABLE),
        # origin stage -- from lineage where the row carries it, else unavailable.
        # Origin stage comes from the lineage entries where the row carries them --
        # a paper_stated entry names its stage while naming no source, so reading
        # stages off SOURCES alone reports ``unavailable`` for 650 attributed rows.
        "origin_stages": detail.get("lineage_stages")
                         or sorted({s.get("stage") for s in sources if s.get("stage")})
                         or UNAVAILABLE,
        "origins": detail.get("lineage_origins")
                   or sorted({s.get("origin") for s in sources if s.get("origin")})
                   or UNAVAILABLE,
        "paper_explicit": detail.get("lineage_paper_explicit", UNAVAILABLE),
        # target-paper evidence. Presence only -- measured fact (4): these strings run
        # to 100k+ characters and carry EXTERNAL text, so presence is not attribution.
        "row_evidence_present": has_evidence,
        "row_evidence_chars": len(ev) if isinstance(ev, str) else 0,
        # attribution
        "attribution_tier": tier,
        "source_ids": [s.get("source_id") or UNAVAILABLE for s in sources] or UNAVAILABLE,
        "source_resolutions": resolutions or UNAVAILABLE,
        "retrieved_chunk_ids": chunk_ids or UNAVAILABLE,
        "lineage_review_required": any(s.get("review_required") for s in sources),
        # retrieved record, recovered by the chunk join
        "chunk_join_scope": join["scope"],
        "chunk_join_reaction_specific": join.get("reaction_specific", False),
        "rag_candidate_gap_id": first["gap_id"] if first else UNAVAILABLE,
        "retrieved_source_paper": first["source_paper"] if first else UNAVAILABLE,
        "retrieved_span": first["span"] if first else UNAVAILABLE,
        "retrieval_score": first["score"] if first else UNAVAILABLE,
        "retrieval_section": first["section"] if first else UNAVAILABLE,
        "admission_result": first["group"] if first else UNAVAILABLE,
        "rejection_reasons": first["reasons"] if first else UNAVAILABLE,
        "requested_pathway_match": first["requested_pathway_match"] if first else UNAVAILABLE,
        "organism_match": first["organism_match"] if first else UNAVAILABLE,
        "scope_membership": first["scope_membership"] if first else UNAVAILABLE,
        # survival. A row read out of the leg's payload is by construction in that
        # payload; for the canonical population that IS the canonical graph. Stated
        # rather than inferred, and named per population so it is never read as a
        # claim about a leg that exported nothing.
        "survives_in_payload": True,
        # D-093's "whether audit or repair later modified or reintroduced it". Read
        # from the lineage stages first (``audit_repair`` / ``audit_modified`` is the
        # typed record of exactly this) and from the repair keys as a fallback for
        # pre-lineage legs, so neither carrier alone decides it.
        "audit_modified": bool(
            reaction.get("preservation_status")
            or reaction.get("repaired_missing_compound_entities")
            or "audit_repair" in (detail.get("lineage_stages") or [])
            or "audit_modified" in (detail.get("lineage_origins") or [])),
        # verdict
        "support_class": support_class,
        "support_reason": reason,
        "participant_detail": detail.get("participants", UNAVAILABLE),
    }


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------

def summarize(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Per-population tallies. Populations are never summed (F-177)."""

    out: Dict[str, Any] = {}
    for pop in POPULATION_ORDER:
        rows = [r for r in records if r["population"] == pop]
        out[pop] = {
            "reactions": len(rows),
            "legs": len({r["leg_dir"] for r in rows}),
            "support_class": {c: sum(1 for r in rows if r["support_class"] == c)
                              for c in SUPPORT_CLASS_ORDER},
            "attribution_tier": {t: sum(1 for r in rows if r["attribution_tier"] == t)
                                 for t in TIER_ORDER},
            "chunk_join_scope": dict(collections.Counter(r["chunk_join_scope"] for r in rows)),
            "reaction_specific_joins": sum(1 for r in rows if r["chunk_join_reaction_specific"]),
        }
    return out


def render(summary: Dict[str, Any], records: Sequence[Dict[str, Any]]) -> str:
    lines: List[str] = []
    lines.append("R-D092-1 -- row-level RAG lineage over the COMMITTED archived corpus")
    lines.append("EVALUATION-ONLY. No acceptance verdict. No runtime touched. (D-093 s.3)")
    lines.append("")
    for pop in POPULATION_ORDER:
        s = summary[pop]
        lines.append(f"== population: {pop} ==")
        lines.append(f"   legs {s['legs']}   reactions {s['reactions']}   "
                     f"(denominator for every rate below)")
        if not s["reactions"]:
            lines.append("   0 evaluated -- itself a fact about the corpus, not a skip")
            lines.append("")
            continue
        lines.append("   support class (D-093 s.1):")
        for c in SUPPORT_CLASS_ORDER:
            n = s["support_class"][c]
            lines.append(f"      {c:24s} {n:5d}  {100.0*n/s['reactions']:5.1f}%  of {s['reactions']}")
        lines.append("   attribution tier:")
        for t in TIER_ORDER:
            lines.append(f"      {t:26s} {s['attribution_tier'][t]:5d}")
        lines.append(f"   chunk join: {s['chunk_join_scope']}")
        lines.append(f"   reaction-specific joins: {s['reaction_specific_joins']}")
        lines.append("")
    return "\n".join(lines)


def main(argv: List[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("repo_root")
    ap.add_argument("--run", action="append", default=[],
                    help="substring filter on run directory; repeatable")
    ap.add_argument("--json", dest="json_path", default=None,
                    help="full per-reaction records (2.6 MB on the current corpus -- "
                         "regenerable, so it is deliberately NOT committed)")
    ap.add_argument("--summary-json", dest="summary_path", default=None,
                    help="summary only, without the per-row records. THIS is the "
                         "committed artifact: the sprint's .git is already 158 MB and a "
                         "derived dump that one command reproduces does not belong in it")
    args = ap.parse_args(argv)

    repo_root = Path(args.repo_root).resolve()
    paths = committed_paths(repo_root)
    legs = discover_legs(paths, args.run)
    adm_index = admission_index(paths, repo_root, [])

    records: List[Dict[str, Any]] = []
    malformed: List[str] = []
    for leg in legs:
        payload = load_json(repo_root / leg["leg_dir"] / leg["payload_file"])
        if not isinstance(payload, dict):
            malformed.append(leg["leg_dir"])
            continue
        entity_index = entity_provenance_index(payload)
        reactions = (payload.get("processes") or {}).get("reactions") or []
        for i, rx in enumerate(reactions):
            if isinstance(rx, dict):
                records.append(build_record(rx, leg, i, entity_index, adm_index))

    summary = summarize(records)
    report = {
        "instrument": "rd092_1_reaction_lineage",
        "charter": "D-093 s.3 (R-D092-1)",
        "evaluation_only": True,
        "corpus": "committed (git ls-files) -- NOT the working tree (F-178)",
        "legs_discovered": len(legs),
        "legs_malformed": malformed,
        "admission_chunks_indexed": len(adm_index),
        "summary": summary,
        "records": records,
    }
    print(render(summary, records))
    if malformed:
        print(f"artifact_malformed legs: {len(malformed)}")
    if args.json_path:
        out = Path(args.json_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
        print(f"wrote {out}")
    if args.summary_path:
        out = Path(args.summary_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        slim = {k: v for k, v in report.items() if k != "records"}
        slim["records_omitted"] = len(records)
        slim["reproduce"] = ("python docs/pwml_recovery_sprint/evidence/"
                             "rd092_1_reaction_lineage.py . --json <out>")
        out.write_text(json.dumps(slim, indent=1, ensure_ascii=False), encoding="utf-8")
        print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
