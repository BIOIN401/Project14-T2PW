"""What C-099 actually preserves, measured on the corpus rather than argued.

ORCH-711 censused the corpus by ``species_ref.source`` alone. This runs the
shipped decision function, ``_wrapper_species_fields``, over the same rows, so
the number reported is the one the code produces rather than one derived from
the source label.

**This is a RECONSTRUCTION and says so.** The committed ``final_mapped.json``
rows are post-clobber: they already carry the sentinel's *Arabidopsis* in
``species``/``species_id`` beside their own ``species_ref``. Feeding those rows
straight in would trip the internal-consistency check on every one of them and
report zero preserved, which is an artifact of the input, not the behaviour. So
each row is rebuilt as ``hydrate_species_references`` left it -- a fresh row
stamped by the production ``_stamp_entity_species`` from that row's own
resolution ref -- which is exactly what the wrapper build sees in a live run.

Also reports the count under the pre-correction three-member source set, so the
cost of ruling out ``single_pathway_species`` (REV-099 Finding 1) is measured
rather than asserted.

Usage::

    <python> c099_preservation_census.py            # run from the tree root
"""

from __future__ import annotations

import glob
import io
import json
from collections import Counter
from typing import Any, Dict, List

from _repo_root import add_src_to_path

add_src_to_path()

from t2pw.mapping import map_ids  # noqa: E402

PATTERN = "runs*/**/papers/*/*/final_mapped.json"
OLD_SET = frozenset(
    {"explicit_entity_species", "single_pathway_species", "biological_state_species"})


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


def _rehydrated(row: Dict[str, Any], ref: Dict[str, Any]) -> Dict[str, Any]:
    """The row as hydration left it, before Stage 6 touched its species."""

    rebuilt: Dict[str, Any] = {"name": row.get("name")}
    map_ids._stamp_entity_species(rebuilt, ref)
    return rebuilt


def main() -> int:
    files = sorted(glob.glob(PATTERN, recursive=True))
    roots = sorted({p.replace("\\", "/").split("/papers/")[0] for p in files})

    total = 0
    with_ref = 0
    sources: Counter = Counter()
    decisions: Counter = Counter()
    preserved_rows: List[str] = []
    moved_by_the_ruling: List[str] = []
    notes_attached = 0

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
            with_ref += 1
            source = str(ref.get("source") or "(none)")
            sources[source] += 1

            probe = _rehydrated(row, ref)
            fields = map_ids._wrapper_species_fields(probe)
            note = (probe.get("mapping_meta") or {}).get("species_preservation")
            if note is not None:
                notes_attached += 1
            decision = (note or {}).get("decision", "no_note_attached")
            decisions[decision] += 1
            label = f"{row.get('name')} :: {name} :: {source}"
            if fields == {}:
                preserved_rows.append(label)
            elif source in OLD_SET:
                moved_by_the_ruling.append(label)

    print(f"pattern                          : {PATTERN}")
    print(f"roots scanned                    : {len(roots)}")
    print(f"files scanned                    : {len(files)}")
    print(f"generated single-protein wrappers: {total}")
    print(f"with a RESOLVED species ref      : {with_ref}")
    print("by species_ref.source:")
    for source, count in sorted(sources.items(), key=lambda kv: (-kv[1], kv[0])):
        member = "SOURCE-SUPPORTED" if source in map_ids._SOURCE_SUPPORTED_SPECIES_SOURCES else "inference"
        print(f"   {source:<28} {count:>3}   [{member}]")
    print("by _wrapper_species_fields decision:")
    for decision, count in sorted(decisions.items(), key=lambda kv: (-kv[1], kv[0])):
        print(f"   {decision:<28} {count:>3}")
    print(f"PRESERVED at this tip            : {len(preserved_rows)}")
    for label in preserved_rows:
        print(f"   + {label}")
    print(f"moved back to today's behaviour by REV-099 Finding 1: {len(moved_by_the_ruling)}")
    for label in moved_by_the_ruling:
        print(f"   - {label}")
    print(f"notes attached (of {with_ref} rows with a ref): {notes_attached}")
    print(f"shipped source set               : {sorted(map_ids._SOURCE_SUPPORTED_SPECIES_SOURCES)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
