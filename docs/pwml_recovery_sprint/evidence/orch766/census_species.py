"""ORCH-766 census: replay the strict quarantine on every archived leg, record species.

Usage (through bounded_run.py, PYTHONPATH=<tree>/src):

    python -u census_species.py <out.json>

Reads, never writes, the primary checkout's ``runs_papers/`` and ``runs_verify/``:
every ``final_mapped.json`` with a sibling ``quarantine_report.json``. Each leg is
replayed with ITS OWN ``export_mode``, ``strict_db`` and ``coverage.requested_context``
(a wrong mode fabricates removals). No LLM, no network, no pipeline.

Per leg it records what the code under PYTHONPATH does (species rows in and out, the
species rows the quarantine removed, the required-field gate's ``species_*`` errors on
the quarantined payload) AND three INDEPENDENT reference facts per species row of the
quarantined payload, computed here without calling any ORCH-766 code:

* ``name_hits`` / ``id_hits_*``: every scalar leaf of the quarantined payload outside
  ``entities.species`` (plus the requested context) that equals one of the row's
  names or ids, compared casefolded and as alphanumerics only;
* ``ir_state_refs``: biological states whose ``species_key`` ``build_pwml_ir`` resolved
  to the row's IR record, and whether that record is IR ``species[0]`` (the writer's
  pathway / complex / default species, ``writer.py:1002-1003,1186,1310,2702``);
* ``writer_protein_refs``: proteins the writer's ``protein_species_id``
  (``writer.py:1188-1214``, replicated here) resolves to that record.

Run once on the base tree and once on the tip; ``census_compare.py`` diffs the two.
"""
from __future__ import annotations

import json
import re
import sys
import time
from copy import deepcopy
from pathlib import Path

from t2pw.pipeline.strict_quarantine import admitted_payload_hash, quarantine_and_close
from t2pw.pwml.ir import build_pwml_ir, validate_required_pwml_contract

PRIMARY = Path("C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW")
ROOTS = ("runs_papers", "runs_verify")
NAME_KEYS = ("name", "display_name", "raw_name")
LIST_NAME_KEYS = ("aliases", "synonyms")
ID_KEYS = ("taxonomy_id", "taxonomy-id", "pathbank_species_id", "pw_species_id", "pathwhiz_id", "species_id")


def keys_of(value):
    if isinstance(value, bool) or value is None:
        return set()
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    text = re.sub(r"\s+", " ", str(value).strip()).casefold()
    return {k for k in (text, re.sub(r"[^0-9a-z]+", "", text)) if k}


def row_keys(row):
    names, ids = set(), set()
    for k in NAME_KEYS:
        names |= keys_of(row.get(k))
    for k in LIST_NAME_KEYS:
        for v in row.get(k) or []:
            names |= keys_of(v)
    for holder in (row, row.get("mapping_meta") or {}, row.get("mapped_ids") or {}):
        if isinstance(holder, dict):
            for k in ID_KEYS:
                ids |= keys_of(holder.get(k))
    return names, ids


SPECIES_PATH = re.compile(r"species|taxon|organism", re.IGNORECASE)


def species_path(path):
    return any(SPECIES_PATH.search(part) for part in path.split("/") if not part.isdigit())


def leaves(obj, path, skip):
    if path in skip:
        return
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from leaves(v, f"{path}/{k}", skip)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from leaves(v, f"{path}/{i}", skip)
    else:
        yield path, obj


def norm(v):
    t = re.sub(r"\s+", " ", str(v or "").strip()).casefold()
    return re.sub(r"[^a-z0-9:+ ]+", " ", t).strip()


def ir_facts(payload, strict_db):
    """IR species records, state refs and writer protein refs, by IR index."""
    try:
        ir, _rep = build_pwml_ir(deepcopy(payload), strict_db=strict_db, name_index=None)
    except Exception as exc:  # noqa: BLE001
        return {"error": f"{type(exc).__name__}: {exc}"[:300]}
    sp = [r for r in ir.get("species") or [] if isinstance(r, dict)]
    key_to_idx = {r.get("key"): i for i, r in enumerate(sp)}
    state_refs = {}
    for st in ir.get("biological_states") or []:
        i = key_to_idx.get(st.get("species_key"))
        if i is not None:
            state_refs.setdefault(i, []).append(st.get("name"))
    by_name, by_ext = {}, {}
    for i, r in enumerate(sp):
        if r.get("name"):
            by_name[str(r["name"]).strip().casefold()] = i
        for k in ("pathwhiz_id", "pathbank_species_id", "pw_species_id", "species_id", "taxonomy_id"):
            if r.get(k) not in (None, ""):
                by_ext[str(r[k]).strip().casefold()] = i
    prot_refs = {}
    for p in (ir.get("entities") or {}).get("proteins") or []:
        sref = p.get("species_ref") if isinstance(p.get("species_ref"), dict) else {}
        mm = p.get("mapping_meta") if isinstance(p.get("mapping_meta"), dict) else {}
        cands = [p.get("species"), p.get("organism"), p.get("species_id"), p.get("pathbank_species_id"),
                 p.get("taxonomy_id"), sref.get("name"), sref.get("species_id"), sref.get("pathbank_species_id"),
                 sref.get("taxonomy_id"), mm.get("species"), mm.get("species_id")]
        hit = 0 if sp else None  # default_species_id
        for c in cands:
            if isinstance(c, dict):
                c = c.get("name") or c.get("id")
            if c in (None, ""):
                continue
            t = str(c).strip()
            j = by_name.get(t.casefold())
            if j is None:
                j = by_ext.get(t.casefold())
            if j is not None:
                hit = j
                break
        if hit is not None:
            prot_refs.setdefault(hit, []).append(p.get("name"))
    return {"species": [{"name": r.get("name"), "aliases": r.get("aliases") or []} for r in sp],
            "state_refs": state_refs, "protein_refs": prot_refs}


def ir_index_of(row, facts):
    names = {norm(row.get("name"))} | {norm(a) for a in row.get("aliases") or []}
    names.discard("")
    for i, r in enumerate(facts.get("species") or []):
        rn = {norm(r["name"])} | {norm(a) for a in r["aliases"]}
        if names & rn:
            return i
    return None


def leg_record(fm_path, q_path):
    fm = json.loads(fm_path.read_text(encoding="utf-8"))
    q = json.loads(q_path.read_text(encoding="utf-8"))
    mode = q.get("export_mode") or "pathwhiz"
    strict_db = bool(q.get("strict_db", True))
    ctx = (q.get("coverage") or {}).get("requested_context")
    res = quarantine_and_close(deepcopy(fm), strict_db=strict_db, pathway_context=ctx, mode=mode)
    out = res.payload
    gate = validate_required_pwml_contract(deepcopy(out), strict_db=strict_db)
    rows_in = [r for r in (fm.get("entities") or {}).get("species") or [] if isinstance(r, dict)]
    rows_out = [r for r in (out.get("entities") or {}).get("species") or [] if isinstance(r, dict)]
    universe = {}
    for p, v in leaves(out, "", {"/entities/species"}):
        for k in keys_of(v):
            universe.setdefault(k, []).append(p)
    for p, v in leaves(ctx or {}, "ctx", set()):
        for k in keys_of(v):
            universe.setdefault(k, []).append(p)
    facts = ir_facts(out, strict_db)
    species_facts = []
    for idx, row in enumerate(rows_out):
        names, ids = row_keys(row)
        name_hits = sorted({p for k in names for p in universe.get(k, [])})
        id_hits = sorted({p for k in ids for p in universe.get(k, [])})
        i = ir_index_of(row, facts) if "error" not in facts else None
        species_facts.append({
            "index": idx,
            "name": row.get("name"),
            "taxonomy_id": row.get("taxonomy_id"),
            "classification": row.get("classification"),
            "name_hits": name_hits[:12],
            "id_hits_species_path": [p for p in id_hits if species_path(p)][:12],
            "id_hits_other_path": [p for p in id_hits if not species_path(p)][:12],
            "ir_index": i,
            "ir_is_species0": i == 0,
            "ir_state_refs": (facts.get("state_refs") or {}).get(i, []) if i is not None else [],
            "writer_protein_refs": (facts.get("protein_refs") or {}).get(i, []) if i is not None else [],
        })
    return {
        "leg": fm_path.parent.relative_to(PRIMARY).as_posix(),
        "export_mode": mode,
        "strict_db": strict_db,
        "input_hash": admitted_payload_hash(fm),
        "result_hash": admitted_payload_hash(out),
        "result_hash_without_species": admitted_payload_hash(
            {**out, "entities": {k: v for k, v in (out.get("entities") or {}).items() if k != "species"}}),
        "ok": res.ok,
        "release_status": (res.quarantine_report.get("release") or {}).get("status"),
        "refusal_reasons": res.refusal_reasons,
        "closure_iterations": res.closure_report.get("iteration_count"),
        "species_in": [r.get("name") for r in rows_in],
        "species_out": [r.get("name") for r in rows_out],
        "species_removed": [r for r in res.removed_entity_report.get("removed_entities") or []
                            if r.get("bucket") == "species"],
        "removed_entities_total": len(res.removed_entity_report.get("removed_entities") or []),
        "gate_ok": bool(gate.get("ok")),
        "gate_species_errors": [[e.get("code"), e.get("pointer"), e.get("species_name")]
                                for e in gate.get("errors") or [] if str(e.get("code", "")).startswith("species_")],
        "gate_error_codes": sorted({str(e.get("code")) for e in gate.get("errors") or []}),
        "ir_error": facts.get("error"),
        "species_facts": species_facts,
    }


def main():
    out_path = Path(sys.argv[1])
    records, skipped = [], []
    t0 = time.time()
    for root in ROOTS:
        for fm in sorted((PRIMARY / root).rglob("final_mapped.json")):
            q = fm.parent / "quarantine_report.json"
            if not q.exists():
                skipped.append({"leg": fm.parent.relative_to(PRIMARY).as_posix(), "why": "no quarantine_report.json"})
                continue
            try:
                records.append(leg_record(fm, q))
            except Exception as exc:  # noqa: BLE001
                skipped.append({"leg": fm.parent.relative_to(PRIMARY).as_posix(), "why": f"{type(exc).__name__}: {exc}"[:300]})
    out_path.write_text(json.dumps({"records": records, "skipped": skipped}, indent=1, ensure_ascii=False), encoding="utf-8")
    strict = [r for r in records if r["export_mode"] != "research"]
    print(f"legs replayed: {len(records)} (strict {len(strict)}), skipped: {len(skipped)}, {time.time() - t0:.1f}s")
    print("legs with a species row removed:", sum(1 for r in records if r["species_removed"]))
    print("species rows removed:", sum(len(r["species_removed"]) for r in records))
    print("strict legs with species_* gate errors:", sum(1 for r in strict if r["gate_species_errors"]))
    print("ir errors:", sum(1 for r in records if r["ir_error"]))
    print("CENSUS DONE")


if __name__ == "__main__":
    main()
