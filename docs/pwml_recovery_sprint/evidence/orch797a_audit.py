"""ORCH-797a: collision audit of the D-120 reference key over the six ORCH-796 legs (read-only, JSON only).

Usage: python orch797a_audit.py   (PYTHONPATH = the tip tree's src, for ``reference_key``)

Per leg, every name the leg writes (entity rows of stage1/merged/final payloads, every reference string in their
processes, element locations and complex components, quarantined rows' original processes, removed entities) is
grouped by the NEW key. A group whose members differ under an OLD key (``process_normalizer._normalize`` as it was,
or ``ir._norm`` as it was) is a pair the new key merges and that old key did not: printed for review. Then each
quarantined row is checked: does a reference it names miss every entity row under the old keys but hit one under
the new key? Those are the rows D-120 § 1 can recover.
"""

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

from t2pw.pipeline.name_keys import reference_key

LEGS = Path(r"C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW/.claude/worktrees/orch796run/runs_papers/ORCH-796/legs")
STAMPS = {"chlorophyll": "2026-10-29_1632", "wnt_ca": "2026-10-29_1941", "wnt_pcp": "2026-10-29_1729",
          "sour_taste_v2": "2026-10-29_1314", "enterobactin": "2026-10-29_1153", "menaquinone": "2026-10-29_1451"}
FREE_TEXT = {"evidence", "provenance", "provenance_lineage", "reason", "relationship", "interaction_type", "class",
             "biological_state", "from_biological_state", "to_biological_state", "side", "role", "entity_type",
             "locked_reaction_id", "scope_membership", "notes", "description", "pointer", "json_pointer", "detail",
             "sources", "mapped_ids", "ids", "metadata", "states", "core_terms", "state", "origin", "support"}
BUCKETS = ("compounds", "proteins", "protein_complexes", "nucleic_acids", "element_collections")


def old_pn(v):
    return re.sub(r"[^a-z0-9:+ ]+", "", re.sub(r"\s+", " ", str(v or "").strip().casefold()))


def old_ir(v):
    return re.sub(r"[^a-z0-9:+ ]+", " ", re.sub(r"\s+", " ", str(v or "")).strip().casefold()).strip()


def strings(value, top=True):
    if isinstance(value, str):
        yield value
    elif isinstance(value, list):
        for item in value:
            yield from strings(item, False)
    elif isinstance(value, dict):
        for k, v in value.items():
            if k in FREE_TEXT or (top and k == "name"):
                continue
            yield from strings(v, False)


def load(p):
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def main():
    total_pairs = 0
    for paper, stamp in STAMPS.items():
        legdir = next((LEGS / f"{paper}-strict" / stamp / "papers").glob("*/strict"))
        rows, refs = set(), set()
        payloads = [load(legdir / n) for n in ("stage1_payload.json", "merged_payload.json", "final_mapped.json")]
        q = load(legdir / "quarantine_report.json")
        rer = load(legdir / "removed_entity_report.json")
        for p in payloads:
            ents = p.get("entities") or {}
            for b in BUCKETS:
                for r in ents.get(b) or []:
                    if isinstance(r, dict) and r.get("name"):
                        rows.add(str(r["name"]))
                        refs.update(strings(r.get("components")))
            for procs in (p.get("processes") or {}).values():
                refs.update(strings(procs))
            for locs in (p.get("element_locations") or {}).values():
                refs.update(strings(locs))
        for e in rer.get("removed_entities") or []:
            rows.add(str(e.get("name")))
        for row in q.get("quarantined") or []:
            refs.update(strings(row.get("original_process")))
        names = {n.strip() for n in rows | refs if isinstance(n, str) and n.strip() and len(n) < 200}
        groups = defaultdict(set)
        for n in names:
            if reference_key(n):
                groups[reference_key(n)].add(n)
        print(f"===== {paper} ({legdir})  names={len(names)} entity rows={len(rows)}")
        for key, members in sorted(groups.items()):
            if len(members) < 2:
                continue
            if len({old_pn(m) for m in members}) > 1 or len({old_ir(m) for m in members}) > 1:
                total_pairs += 1
                tag = "ROW+ROW" if sum(m in rows for m in members) > 1 else ("ROW+REF" if members & rows else "REF+REF")
                print(f"  MERGED [{tag}] {key!r}: {sorted(members)}")
        row_pn = {old_pn(r) for r in rows}
        row_ir = {old_ir(r) for r in rows}
        row_new = {reference_key(r): r for r in rows}
        for row in q.get("quarantined") or []:
            hits = [(s, row_new[reference_key(s)]) for s in strings(row.get("original_process"))
                    if s.strip() and old_pn(s) not in row_pn and old_ir(s) not in row_ir and reference_key(s) in row_new]
            if hits:
                print(f"  RECOVERABLE? {row.get('bucket')}[{row.get('index')}] {row.get('name')!r} "
                      f"{row.get('reason')}: {hits}")
    print("merged groups total:", total_pairs)
    return 0


if __name__ == "__main__":
    sys.exit(main())
