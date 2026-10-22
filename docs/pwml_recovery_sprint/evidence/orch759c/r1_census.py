"""ORCH-759c round 1 blast radius: REAL base vs tip over every archived strict leg.

Round 0's census emulated base inside the tip process. That cannot cover round 1's
location rule, so this one runs the actual code twice:

    r1_census.py dump <out.json> <root>...   # once with base src on PYTHONPATH, once with tip src
    r1_census.py compare <base.json> <tip.json>

``dump`` runs ``quarantine_and_close`` (``strict_db=True``, ``mode="pathwhiz"``, the leg's own
Stage-0 context when its ``quarantine_report.json`` carries one) on every
``<root>/**/strict/final_mapped.json`` and records, per leg, the outcome, the
admissions, the removed entities and locations, and every surviving location row
that names no entity of its own kind. ``compare`` prints every leg that differs.
Read-only, except the dump file.
"""
import json
import sys
from pathlib import Path

LOCATES = {
    "compound_locations": ("compound", ("compounds",)),
    "protein_locations": ("protein", ("proteins", "protein_complexes")),
    "nucleic_acid_locations": ("nucleic_acid", ("nucleic_acids",)),
    "element_collection_locations": ("element_collection", ("element_collections",)),
}


def dump(out, roots):
    from t2pw.pipeline.process_normalizer import _entity_name_norms, _normalize
    from t2pw.pipeline.strict_quarantine import ACCEPTED_STATES, quarantine_and_close

    legs = sorted({p for root in roots for p in Path(root).rglob("final_mapped.json") if p.parent.name == "strict"})
    record = {}
    for leg in legs:
        try:
            payload = json.loads(leg.read_text(encoding="utf-8"))
            q = leg.parent / "quarantine_report.json"
            ctx = (json.loads(q.read_text(encoding="utf-8")).get("coverage") or {}).get("requested_context") if q.is_file() else None
            r = quarantine_and_close(payload, strict_db=True, pathway_context=ctx, mode="pathwhiz")
        except Exception as exc:  # recorded, never hidden
            record[str(leg)] = {"error": f"{type(exc).__name__}: {exc}"}
            continue
        ents = r.payload.get("entities") or {}
        cross = []
        for bucket, (field, owners) in LOCATES.items():
            own = {n for b in owners for n in _entity_name_norms(ents.get(b) or [])}
            for row in (r.payload.get("element_locations") or {}).get(bucket) or []:
                if isinstance(row, dict):
                    name = row.get(field) or row.get("entity")
                    if name and _normalize(name) not in own:
                        cross.append(f"{bucket}|{name}|{row.get('biological_state')}")
        record[str(leg)] = {
            "ok": r.ok,
            "refusal": r.refusal_reasons,
            "release": (r.quarantine_report.get("release") or {}).get("status"),
            "resulting": r.quarantine_report["resulting_payload_hash"],
            "accepted": sum(1 for a in r.admissions if a["state"] in ACCEPTED_STATES),
            "admissions": sorted(f"{a['pointer']}|{a['state']}|{a.get('reason') or ''}" for a in r.admissions),
            "removed_entities": sorted(f"{e['bucket']}|{e['name']}|{e['reason']}" for e in r.removed_entity_report["removed_entities"]),
            "removed_locations": sorted(f"{e['bucket']}|{e['name']}|{e['reason']}|{e.get('biological_state')}" for e in r.removed_entity_report["removed_locations"]),
            "cross_kind_surviving_locations": sorted(cross),
        }
    Path(out).write_text(json.dumps(record, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"DUMP legs={len(legs)} errors={sum(1 for v in record.values() if 'error' in v)} "
          f"accepted={sum(v.get('accepted', 0) for v in record.values())} "
          f"legs_with_cross_kind_locations={sum(1 for v in record.values() if v.get('cross_kind_surviving_locations'))} -> {out}")


def compare(base_path, tip_path):
    base = json.loads(Path(base_path).read_text(encoding="utf-8"))
    tip = json.loads(Path(tip_path).read_text(encoding="utf-8"))
    assert set(base) == set(tip), "leg sets differ"
    differ = 0
    for leg in sorted(base):
        b, t = base[leg], tip[leg]
        if b == t:
            continue
        differ += 1
        print(f"DIFF {leg}")
        for key in ("ok", "refusal", "release", "accepted"):
            if b.get(key) != t.get(key):
                print(f"   {key}: {b.get(key)} -> {t.get(key)}")
        for key in ("admissions", "removed_entities", "removed_locations", "cross_kind_surviving_locations"):
            only_b = sorted(set(b.get(key, [])) - set(t.get(key, [])))
            only_t = sorted(set(t.get(key, [])) - set(b.get(key, [])))
            if only_b or only_t:
                print(f"   {key}: only base={only_b} | only tip={only_t}")
    print(f"COMPARE legs={len(base)} differ={differ} same={len(base) - differ} "
          f"cross_kind_legs base={sum(1 for v in base.values() if v.get('cross_kind_surviving_locations'))} "
          f"tip={sum(1 for v in tip.values() if v.get('cross_kind_surviving_locations'))}")


if __name__ == "__main__":
    if sys.argv[1] == "dump":
        dump(sys.argv[2], sys.argv[3:])
    else:
        compare(sys.argv[2], sys.argv[3])
