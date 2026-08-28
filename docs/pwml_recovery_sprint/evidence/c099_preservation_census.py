"""What C-099 actually preserves, measured on the corpus rather than argued.

ORCH-711 censused the corpus by ``species_ref.source`` alone. This runs the
shipped decision function, ``_wrapper_species_fields``, over the same rows, so
the number reported is the one the code produces rather than one derived from
the source label.

**This is a RECONSTRUCTION from committed post-clobber artifacts, not an
observation of a live run.** The committed ``final_mapped.json`` rows already
carry the sentinel's *Arabidopsis* in the four fields the Stage 6 clobber wrote
(``species``, ``organism``, ``species_id``, ``pathbank_species_id``). Feeding
them in raw would trip the internal-consistency check on every row and report
zero preserved -- an artifact of the input, not the behaviour.

So the pre-clobber row is rebuilt by **copying the committed row and resetting
only those four fields** to what ``hydrate_species_references`` would have
stamped from that row's own ref. Everything else is left exactly as committed,
because the clobber never touched it.

REV-099 Finding 2 -- why the previous version was a ceiling
-----------------------------------------------------------
It built a fresh ``{"name": ...}`` dict and stamped it from the ref. Every field
the consistency check reads then either came from that same ref -- agreeing by
construction -- or was absent and skipped by the ``if value`` / ``if sid is not
None`` guards. The ``disagree`` branch could not fire, so "4 preserved" was an
**upper bound**, and the one path it could not exercise was exactly the
exception C-099 documents: a source-supported row that still ships
*Arabidopsis*.

REV-099 amendment -- the reachable surface, and the trap in counting it
-----------------------------------------------------------------------
Under this reconstruction the ``disagree`` branch is reachable through **exactly
two fields**:

===========================  ==========================================
``species`` / ``organism``   reset, then re-stamped from the ref
                             -> **agree by construction**, cannot fire
``species_id`` /             reset to the ref's id, or dropped when the
``pathbank_species_id``      ref has none -> **equal, or guard-skipped**;
                             cannot fire either way
``species_name``             **survives the reset** -> fires if it differs
``taxonomy_id``              **survives the reset** -> fires if it differs
===========================  ==========================================

That is the entire reachable surface, and it is the residual bound on every
number below.

The trap: the guards are ``if value and ...`` and ``if row_tax and ref_tax and
...``. A row where ``species_name`` and ``taxonomy_id`` are simply **absent**
produces no disagreement for precisely the reason the old reconstruction
produced none -- **the check never ran**. In a bare count that is
indistinguishable from "checked and agreed". So this reports **two facts per
row**: whether those fields were PRESENT (the check was live) and whether they
MATCHED (it ran and agreed). Only the first makes the second mean anything, and
the per-row table below is printed for every row rather than aggregated away.

If no row has a live check, the honest verdict is that the contradiction path
remains **unmeasured**, and this says so rather than reporting a flattering
number.

Remaining limitations, stated because the ``.log`` is what a later session reads:

* Still a reconstruction. It measures the decision function against rebuilt
  inputs, not a live pipeline run.
* Where a row's ref carries no species id, the pre-clobber id is
  **unrecoverable** -- the clobber destroyed it. Those rows are named and their
  id fields dropped rather than guessed; that is one of two possible
  reconstructions and could move their verdict.
* Contradictions a live run would create *after* the mapping stage cannot appear
  here; only mapping-stage artifacts are read.

Stop condition: if either example D-070 section O-1 pins -- the enterobactin
synthase complex (*E. coli*) or the ALAS2 homodimer (*Homo sapiens*) -- is not
among the preserved rows, this exits 2. That outcome contradicts REV-099
Finding 1's premise rather than the production code, and belongs with the
coordinator, not in a total.

Usage::

    <python> c099_preservation_census.py            # run from the tree root
"""

from __future__ import annotations

import copy
import glob
import io
import json
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

from _repo_root import add_src_to_path

add_src_to_path()

from t2pw.mapping import map_ids  # noqa: E402

PATTERN = "runs*/**/papers/*/*/final_mapped.json"
OLD_SET = frozenset(
    {"explicit_entity_species", "single_pathway_species", "biological_state_species"})

#: The four fields, and only these four, that the Stage 6 clobber overwrote.
CLOBBERED = ("species", "organism", "species_id", "pathbank_species_id")

#: The entire surface through which the disagree branch can fire here.
REACHABLE_FIELDS = ("species_name", "taxonomy_id")

#: A species no corpus row resolves to, used only by the positive control.
CONTROL_SPECIES = "Xanthomonas campestris"

DISAGREE_KIND = "row_species_fields_disagree_with_resolution"

#: The two examples D-070 § O-1 names. Carrying ``explicit_entity_species`` gets
#: a row past the SOURCE gate; it does not get it past the CONTRADICTION gate,
#: which this reconstruction makes reachable for the first time. If either is
#: missing from the preserved set, REV-099 Finding 1's premise -- "the narrower
#: set preserves everything the ruling asked for and loses nothing it named" --
#: is false, and that sends the RULING back rather than the card. Exit 2.
PINNED = (
    ("enterobactin synthase complex", "Escherichia coli"),
    ("ALAS2 homodimer", "Homo sapiens"),
)


def _resolution_ref(row: Dict[str, Any]) -> Dict[str, Any]:
    meta = row.get("mapping_meta") if isinstance(row.get("mapping_meta"), dict) else {}
    ref = row.get("species_ref") if isinstance(row.get("species_ref"), dict) else {}
    if not ref:
        candidate = meta.get("species_resolution")
        ref = candidate if isinstance(candidate, dict) else {}
    return ref


def _is_wrapper(row: Any) -> bool:
    if not isinstance(row, dict):
        return False
    meta = row.get("mapping_meta") if isinstance(row.get("mapping_meta"), dict) else {}
    reason = str(row.get("generation_reason") or meta.get("generation_reason") or "")
    return row.get("generated") is True and reason == "single_protein_pathwhiz_wrapper"


def _reconstructed(row: Dict[str, Any], ref: Dict[str, Any]) -> Tuple[Dict[str, Any], bool]:
    """The committed row with ONLY the four clobbered fields put back.

    Returns ``(row, id_recoverable)``. When the ref carries no species id the
    pre-clobber id is unrecoverable; the fields are dropped rather than guessed
    and the caller is told so.
    """

    probe = copy.deepcopy(row)
    probe.pop("generated", None)
    probe.pop("generation_reason", None)
    meta = probe.get("mapping_meta")
    if isinstance(meta, dict):
        meta.pop("species_preservation", None)

    name = str(ref.get("name") or "")
    probe["species"] = name
    probe["organism"] = name

    sid = map_ids._to_positive_int(ref.get("pathbank_species_id") or ref.get("species_id"))
    if sid is not None:
        probe["species_id"] = sid
        probe["pathbank_species_id"] = sid
        return probe, True
    probe.pop("species_id", None)
    probe.pop("pathbank_species_id", None)
    return probe, False


def _decide(probe: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
    fields = map_ids._wrapper_species_fields(probe)
    note = (probe.get("mapping_meta") or {}).get("species_preservation") or {}
    return fields == {}, dict(note)


def _liveness(probe: Dict[str, Any], ref: Dict[str, Any]) -> Dict[str, Any]:
    """Per-field: was the guard satisfied (PRESENT), and did the value AGREE?

    Mirrors ``_wrapper_species_fields``' own guards exactly -- ``if value and``
    for the name fields, ``if row_tax and ref_tax and`` for taxonomy -- so
    "present" here means the same thing it means inside the decision function.
    """

    resolved = map_ids._canonical_name(str(ref.get("name") or ""))
    ref_tax = map_ids._canonical_name(str(ref.get("taxonomy_id") or ""))

    species_name = map_ids._canonical_name(str(probe.get("species_name") or ""))
    row_tax = map_ids._canonical_name(str(probe.get("taxonomy_id") or ""))

    name_present = bool(species_name)
    tax_present = bool(row_tax and ref_tax)
    return {
        "species_name_present": name_present,
        "species_name_value": species_name or "(absent)",
        "species_name_agrees": (
            map_ids._normalize_name(species_name) == map_ids._normalize_name(resolved)
            if name_present else None),
        "taxonomy_id_present": tax_present,
        "taxonomy_id_value": row_tax or "(absent)",
        "ref_taxonomy_id": ref_tax or "(absent)",
        "taxonomy_id_agrees": (row_tax == ref_tax) if tax_present else None,
        "check_live": name_present or tax_present,
    }


def _control(row: Dict[str, Any], ref: Dict[str, Any]) -> bool:
    """Positive control on a REAL corpus row of exactly this shape."""

    probe, _ = _reconstructed(row, ref)
    probe["species_name"] = CONTROL_SPECIES
    preserved, note = _decide(probe)
    kinds = {item.get("kind") for item in note.get("contradictions") or []}
    return (not preserved) and DISAGREE_KIND in kinds


def _yn(value: Optional[bool]) -> str:
    if value is None:
        return "n/a"
    return "yes" if value else "NO"


def main() -> int:
    files = sorted(glob.glob(PATTERN, recursive=True))
    roots = sorted({p.replace("\\", "/").split("/papers/")[0] for p in files})

    total = 0
    sources: Counter = Counter()
    decisions: Counter = Counter()
    rows: List[Dict[str, Any]] = []
    control_row: Optional[Tuple[Dict[str, Any], Dict[str, Any]]] = None

    for path in files:
        payload = json.load(io.open(path, encoding="utf-8"))
        entities = payload.get("entities") or {}
        for row in entities.get("protein_complexes") or []:
            if not _is_wrapper(row):
                continue
            total += 1
            ref = _resolution_ref(row)
            name = str(ref.get("name") or "")
            if not ref or not name or name.strip().casefold() == "unknown species":
                continue
            source = str(ref.get("source") or "(none)")
            sources[source] += 1

            probe, id_recoverable = _reconstructed(row, ref)
            live = _liveness(probe, ref)
            if control_row is None:
                control_row = (row, ref)

            preserved, note = _decide(probe)
            decisions[note.get("decision", "no_note_attached")] += 1
            kinds = {item.get("kind") for item in note.get("contradictions") or []}
            rows.append({
                "wrapper": str(row.get("name") or ""),
                "leg": path.replace("\\", "/").split("/papers/")[-1],
                "resolved": name,
                "source": source,
                "preserved": preserved,
                "decision": note.get("decision", "no_note_attached"),
                "contradiction": DISAGREE_KIND in kinds,
                "id_recoverable": id_recoverable,
                **live,
            })

    with_ref = len(rows)
    preserved_rows = [r for r in rows if r["preserved"]]
    contradiction_rows = [r for r in rows if r["contradiction"]]
    live_rows = [r for r in rows if r["check_live"]]
    moved = [r for r in rows if not r["preserved"] and r["source"] in OLD_SET]
    unrecoverable = [r for r in rows if not r["id_recoverable"]]

    print(f"pattern                          : {PATTERN}")
    print(f"roots scanned                    : {len(roots)}")
    print(f"files scanned                    : {len(files)}")
    print(f"generated single-protein wrappers: {total}")
    print(f"with a RESOLVED species ref      : {with_ref}")
    print("by species_ref.source:")
    for source, count in sorted(sources.items(), key=lambda kv: (-kv[1], kv[0])):
        member = ("SOURCE-SUPPORTED"
                  if source in map_ids._SOURCE_SUPPORTED_SPECIES_SOURCES else "inference")
        print(f"   {source:<28} {count:>3}   [{member}]")
    print("by _wrapper_species_fields decision:")
    for decision, count in sorted(decisions.items(), key=lambda kv: (-kv[1], kv[0])):
        print(f"   {decision:<28} {count:>3}")
    print(f"PRESERVED (measured)             : {len(preserved_rows)}")
    print(f"CONTRADICTION surfaced           : {len(contradiction_rows)}")
    print(f"moved back to today's behaviour by REV-099 Finding 1: {len(moved)}")
    print(f"species id unrecoverable (dropped, not guessed)     : {len(unrecoverable)}")

    print("")
    print("PER-ROW -- the consistency check is reachable ONLY through species_name")
    print("and taxonomy_id. 'present' = the guard was satisfied and the check ran.")
    print("'agrees' means anything ONLY where 'present' is yes.")
    print("")
    header = (f"{'#':<3} {'wrapper':<32} {'source':<24} "
              f"{'sp_name present':<15} {'agrees':<7} "
              f"{'tax present':<12} {'agrees':<7} {'live':<5} {'verdict':<28}")
    print(header)
    print("-" * len(header))
    for index, r in enumerate(rows, 1):
        print(f"{index:<3} {r['wrapper'][:32]:<32} {r['source'][:24]:<24} "
              f"{_yn(r['species_name_present']):<15} {_yn(r['species_name_agrees']):<7} "
              f"{_yn(r['taxonomy_id_present']):<12} {_yn(r['taxonomy_id_agrees']):<7} "
              f"{_yn(r['check_live']):<5} {r['decision']:<28}")
    print("")
    for index, r in enumerate(rows, 1):
        print(f"  [{index}] {r['wrapper']} :: resolved={r['resolved']} :: {r['source']}")
        print(f"      leg              : {r['leg']}")
        print(f"      species_name     : {r['species_name_value']}")
        print(f"      taxonomy_id      : row={r['taxonomy_id_value']} ref={r['ref_taxonomy_id']}")
        print(f"      preserved        : {r['preserved']}")

    print("")
    print(f"rows where the contradiction check was LIVE : {len(live_rows)} of {with_ref}")
    if not live_rows:
        print("   NONE. Both reachable fields are absent on every row, so the")
        print("   check never ran. THE CONTRADICTION PATH REMAINS UNMEASURED --")
        print("   this is NOT 'reachable and did not fire'. Do not read the")
        print("   preserved count as covering it.")
    else:
        agreed = [r for r in live_rows if r["check_live"] and not r["contradiction"]]
        print(f"   of those, ran and AGREED                 : {len(agreed)}")
        print("   These are genuine 'could have fired, did not' rows:")
        for r in agreed:
            print(f"      = {r['wrapper']} :: species_name={r['species_name_value']} "
                  f"taxonomy_id={r['taxonomy_id_value']} vs ref {r['resolved']}"
                  f"/{r['ref_taxonomy_id']}")

    print("")
    print("POSITIVE CONTROL -- is the contradiction path reachable at all?")
    if control_row is None:
        print("   NO CORPUS ROW AVAILABLE. The count above is UNVALIDATED.")
        return 1
    fired = _control(*control_row)
    print(f"   real corpus row, species_name perturbed to {CONTROL_SPECIES!r}")
    print(f"   contradiction raised             : {fired}")
    if not fired:
        print("   FAILED. The disagree branch is NOT reachable on rows of this")
        print("   shape, so the preserved count above is an UPPER BOUND, not a")
        print("   measurement. Do not cite it as one.")
        return 1
    print("   PASSED. The branch fires when a row genuinely disagrees.")

    print("")
    print("PINNED EXAMPLES -- the ones D-070 section O-1 names. Carrying")
    print("explicit_entity_species clears the SOURCE gate only; the CONTRADICTION")
    print("gate is separate and is reachable here for the first time.")
    missing: List[str] = []
    for pinned_name, pinned_species in PINNED:
        hit = next(
            (r for r in preserved_rows
             if map_ids._normalize_name(r["wrapper"]) == map_ids._normalize_name(pinned_name)
             and map_ids._normalize_name(r["resolved"]) == map_ids._normalize_name(pinned_species)),
            None)
        state = "PRESERVED" if hit else "NOT PRESERVED"
        print(f"   {pinned_name} ({pinned_species}): {state}")
        if hit is None:
            missing.append(f"{pinned_name} ({pinned_species})")
    if missing:
        print("")
        print("   *** STOP CONDITION MET ***")
        print("   A pinned example is not preserved. REV-099 Finding 1 rests on")
        print("   the premise that the narrower source set 'preserves everything")
        print("   the ruling asked for and loses nothing it named'. That premise")
        print("   is FALSE for: " + "; ".join(missing))
        print("   This contradicts THE RULING, not the production code. Report to")
        print("   the coordinator; do not adjust map_ids.py or the test file.")
        return 2

    print("")
    print("TWO DIFFERENT 4s -- do not reconcile them, they measure different things:")
    print("  * the 4/2 CENSUS SPLIT -- refs counted by species_ref.source, read")
    print("    straight off committed data, cited in the C-099 charter amendment")
    print("    and in ORCH-711. Round two did NOT touch it and it was never")
    print("    wrong. If the preserved count below ever moves, the charter's 4/2")
    print("    still does not need editing and MUST NOT be edited.")
    print("  * the PRESERVED-ROW COUNT -- a different quantity: how many rows the")
    print("    shipped decision function actually preserves. That is what round")
    print("    two re-measured. The two coincide here by arithmetic accident.")

    print("")
    print("LIMITATIONS -- read before citing any number above:")
    print("  * RECONSTRUCTION, not a live run. The decision function is applied")
    print("    to rows rebuilt from committed post-clobber artifacts by putting")
    print("    back only the four fields the clobber wrote.")
    print("  * RESIDUAL BOUND. The disagree branch is reachable here through")
    print("    species_name and taxonomy_id ONLY. species/organism are")
    print("    re-stamped from the ref and agree by construction;")
    print("    species_id/pathbank_species_id are set equal to the ref id or")
    print("    dropped, so their guard cannot fire. A contradiction living only")
    print("    in those fields would be invisible to this probe.")
    print("  * REV-099 Finding 2: the previous version built a fresh dict from")
    print("    the ref, so NO field could disagree and its 'PRESERVED 4' was an")
    print("    upper bound. This version is a measurement only to the extent the")
    print("    per-row 'live' column above says the check actually ran.")
    print("  * Where a ref carries no species id the pre-clobber id is")
    print("    unrecoverable; those rows have their id fields dropped rather")
    print("    than guessed, which is one of two possible reconstructions.")
    print("  * Contradictions a live run would create after the mapping stage")
    print("    cannot appear here; only mapping-stage artifacts are read.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
