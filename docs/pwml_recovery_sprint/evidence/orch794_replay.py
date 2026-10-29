"""ORCH-794: offline tail replay of the ORCH-785 chlorophyll and Wnt/Ca2+ strict legs (one name in two kinds).

Usage: python orch794_replay.py --src <tree>/src --out <scratch dir>

Both legs exported, so their ``final_mapped.json`` is the CANONICAL payload: the rows ORCH-794 is about were
already quarantined out of it, and a plain ``replay_leg`` re-quarantines a payload that no longer holds them. This
script therefore RECONSTRUCTS the quarantine input for those rows only, and states exactly how:

* every quarantine record whose reason is ``ambiguous_entity_type:<a>+<b>`` other than the gene/protein pair
  (ORCH-782's) is put back into ``processes[<bucket>]``: the report's ``detail.row`` digest, with each elided
  string restored from the leg's ``merged_payload.json`` row of the same name (a field it cannot restore is
  dropped); the record is taken out of the scratch ``quarantine_report.json`` so it is not counted twice;
* every entity row the leg's closure removed (``removed_entity_report.json``) under a name those rows name is
  put back, from ``merged_payload.json`` when it holds that row (Stage-6 fields are then absent), else as
  ``{"name", "class"}``; their removed location rows are put back the same way;
* ``final_stage3_gate_report.json`` gets the reconstructed payload's hash, so the harness reads the scratch leg
  as a canonical checkpoint (recorded pre-freeze removals carried, as for any canonical leg).

Then the ORCH-768 harness's ``replay_leg`` (imported; it installs the socket block) runs the tail with the
measured tree: quarantine, freeze, export, coverage. Read-only on the legs; writes only under --out.
"""

import argparse
import importlib.util
import json
import re
import shutil
import sys
from copy import deepcopy
from pathlib import Path

HERE = Path(__file__).resolve().parent
HARNESS = HERE / "orch768_replay_corpus.py"
W = Path(r"C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW/.claude/worktrees")
LEGS = {
    "chlorophyll": W / "orch785run/runs_papers/ORCH-785/legs/chlorophyll-strict/2026-10-28_1743/papers/PMC5130987/strict",
    "wnt_ca": W / "orch785run/runs_papers/ORCH-785/legs/wnt_ca-strict/2026-10-28_1910/papers/PMC10425814/strict",
}
ELIDED = re.compile(r"^(?:\d+ item\(s\), )?\d+ chars elided$")
SKIP = object()
LOC_FIELD = {"compound_locations": "compound", "protein_locations": "protein",
             "nucleic_acid_locations": "nucleic_acid", "element_collection_locations": "element_collection"}
#: The ORCH-785 run's Stage-6 id cache (end of run), read-only: a removed protein's recorded accession.
CACHE = json.loads((W / "orch785run/data/id_mapping_cache.json").read_text(encoding="utf-8"))
CACHE = CACHE.get("proteins") or {}
SPECIES_KEYS = ("species", "species_name", "organism", "pathbank_species_id", "species_id", "taxonomy_id", "species_ref")
SINGULAR = {"compounds": "compound", "proteins": "protein", "protein_complexes": "protein_complex",
            "nucleic_acids": "nucleic_acid", "element_collections": "element_collection"}


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def restore(digest, full):
    if isinstance(digest, str) and ELIDED.match(digest):
        return full if full is not None else SKIP
    if isinstance(digest, dict):
        out = {}
        for key, value in digest.items():
            got = restore(value, full.get(key) if isinstance(full, dict) else None)
            if got is not SKIP:
                out[key] = got
        return out
    if isinstance(digest, list):
        got = [restore(v, full[i] if isinstance(full, list) and i < len(full) else None) for i, v in enumerate(digest)]
        return [g for g in got if g is not SKIP]
    return digest


def strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for key, item in value.items():
            if key not in ("evidence", "name", "provenance_lineage", "pointer", "biological_state",
                           "from_biological_state", "to_biological_state", "relationship", "side"):
                yield from strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from strings(item)


def reconstruct(leg_dir):
    fm, q, rer, merged = (load(leg_dir / n) for n in ("final_mapped.json", "quarantine_report.json",
                                                       "removed_entity_report.json", "merged_payload.json"))
    payload = deepcopy(fm)
    targets = [r for r in q.get("quarantined") or []
               if str(r.get("reason", "")).startswith("ambiguous_entity_type:")
               and r.get("reason") != "ambiguous_entity_type:nucleic_acids+proteins"]
    log = []
    names = set()
    locks = set()
    for rec in targets:
        bucket = rec["bucket"]
        if rec.get("original_process"):  # the verbatim row the quarantine kept
            row, how = deepcopy(rec["original_process"]), "verbatim original_process"
        else:
            full = next((r for r in (merged.get("processes") or {}).get(bucket) or [] if r.get("name") == rec["name"]), None)
            row = restore({k: v for k, v in rec["detail"]["row"].items() if k != "pointer"}, full)
            how = f"detail.row digest, evidence restored from merged_payload: {full is not None}"
        locks.add(row.get("locked_reaction_id"))
        payload.setdefault("processes", {}).setdefault(bucket, []).append(row)
        names |= {s.casefold() for s in strings(row)}
        log.append(f"put back {bucket} '{rec['name']}' (was {rec['pointer']}, {rec['reason']}); from {how}")
    locks.discard(None)
    # The leg's own lock bookkeeping lists those rows as quarantined; the replay's quarantine re-decides them.
    stale = [r for r in payload.get("quarantined_locked_reactions") or [] if r.get("locked_reaction_id") in locks
             or (r.get("original_reaction") or {}).get("locked_reaction_id") in locks]
    if stale:
        payload["quarantined_locked_reactions"] = [r for r in payload["quarantined_locked_reactions"] if r not in stale]
        log.append(f"dropped {len(stale)} stale quarantined_locked_reactions entries: {sorted(locks)}")
    ents = payload.setdefault("entities", {})
    # A single-protein wrapper's component goes too ("HEMA1 complex" -> "HEMA1").
    names |= {n[: -len(" complex")] for n in names if n.endswith(" complex")}
    for e in rer.get("removed_entities") or []:
        bucket, name = e.get("bucket"), e.get("name")
        if bucket not in SINGULAR or str(name).casefold() not in names:
            continue
        if any(r.get("name") == name for r in ents.get(bucket) or []):
            continue
        src = next((r for r in (merged.get("entities") or {}).get(bucket) or [] if r.get("name") == name), None)
        how = "merged_payload" if src else "name and class only"
        row = deepcopy(src) if src else {"name": name, "class": SINGULAR[bucket]}
        base = name[: -len(" complex")] if name.endswith(" complex") else ""
        template = next((c for c in ents.get("protein_complexes") or [] if c.get("generated")
                         and len(c.get("components") or []) == 1), None)
        if not src and bucket == "protein_complexes" and template and any(p.get("name") == base for p in ents.get("proteins") or []):
            # A Stage-6 single-protein wrapper (not in merged_payload): this leg's own wrapper shape, renamed.
            row = {**deepcopy(template), "name": name, "components": [{"name": base, "stoichiometry": 1}]}
            how = f"Stage-6 wrapper shape of '{template['name']}', renamed, component '{base}'"
        if src and bucket == "proteins" and e.get("had_external_identity") and not row.get("mapped_ids"):
            # Stage 6 had given it an id (the removal record says so); the run's own id cache holds which one.
            hit = next((v for k, v in CACHE.items() if k.startswith(f"api-v9::{name.casefold()}::tax:")
                        and v.get("status") == "mapped"), None)
            model = next((p for p in ents.get("proteins") or [] if p.get("species_ref")), {})
            if hit:
                row.update({k: deepcopy(model[k]) for k in SPECIES_KEYS if k in model})
                row["mapped_ids"] = dict(hit["mapped_ids"])
                how += f" + uniprot {hit['mapped_ids'].get('uniprot')} from the run's id cache ({hit.get('chosen_rule')})"
        ents.setdefault(bucket, []).append(row)
        log.append(f"put back entity {bucket} '{name}' (closure: {e.get('reason')}); from: {how}")
    locs = payload.setdefault("element_locations", {})
    for e in rer.get("removed_locations") or []:
        bucket = str(e.get("bucket", "")).split("/")[-1]
        if bucket in LOC_FIELD and str(e.get("name")).casefold() in names:
            locs.setdefault(bucket, []).append({LOC_FIELD[bucket]: e["name"], "biological_state": e.get("biological_state")})
            log.append(f"put back location {bucket} '{e['name']}' @ '{e.get('biological_state')}'")
    keep = [r for r in q.get("quarantined") or [] if r not in targets]
    return payload, {**q, "quarantined": keep}, targets, log


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    spec = importlib.util.spec_from_file_location("orch768_replay_corpus", HARNESS)
    H = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(H)
    T = H.load_tree(Path(args.src))
    print("t2pw resolved:", T.t2pw)
    print("offline probe:", H.probe_offline())
    out = Path(args.out)
    for tag, leg_dir in LEGS.items():
        print(f"\n===== {tag}: {leg_dir}")
        payload, qrep, targets, log = reconstruct(leg_dir)
        print("\n".join(log))
        root = out / tag / "inputs"
        scratch = root / "leg"
        scratch.mkdir(parents=True, exist_ok=True)
        for name in H.LEG_FILES:
            if (leg_dir / name).is_file():
                shutil.copyfile(leg_dir / name, scratch / name)
        (scratch / "final_mapped.json").write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        (scratch / "quarantine_report.json").write_text(json.dumps(qrep, ensure_ascii=False), encoding="utf-8")
        grec = load(leg_dir / "final_stage3_gate_report.json")
        grec["payload_sha256"] = T.gr.payload_sha256(load(scratch / "final_mapped.json"))
        (scratch / "final_stage3_gate_report.json").write_text(json.dumps(grec), encoding="utf-8")
        inputs = {n: {"path": f"leg/{n}", "sha256": H.sha256_file(scratch / n)} for n in H.LEG_FILES if (scratch / n).is_file()}
        leg = {"id": f"ORCH-785/{tag}-strict (ORCH-794 reconstruction)", "paper": leg_dir.parent.name, "mode": "strict",
               "inputs": inputs}
        work = out / tag / "work"
        r = H.replay_leg(T, leg, work, primary=root)
        for key in ("checkpoint", "quarantine", "export", "pwml_written", "failure_stage", "network_refused"):
            print(f"{key}: {json.dumps(r.get(key), default=str)}")
        cov = r["pwml_coverage"]
        print("coverage:", json.dumps({k: cov.get(k) for k in ("drawn", "total", "by_design_withheld", "pwml_written", "by_kind")}))
        print("omitted by reason:", json.dumps(cov.get("omitted_by_reason")))
        qr = load(next(work.rglob("quarantine_report.json")))
        rr = load(next(work.rglob("removed_entity_report.json")))
        canonical = load(work / "canonical.json")
        ir = load(work / "pwml_ir.json") if r["pwml_written"] else {}
        print("rows put back, their fate at this tree:")
        recovered = 0
        for rec in targets:
            bucket, name = rec["bucket"], rec["name"]
            fate = next((a["reason"] for a in qr.get("admissions") or [] if a["bucket"] == bucket and a["name"] == name
                         and a.get("index", 0) >= len((load(leg_dir / "final_mapped.json").get("processes") or {}).get(bucket) or [])),
                        "not admitted (no record)")
            in_canon = any(p.get("name") == name for p in (canonical.get("processes") or {}).get(bucket) or [])
            ir_rows = [p for p in (ir.get("processes") or {}).get(bucket) or [] if p.get("name") == " ".join(name.split())]
            drawn = bool(ir_rows) and (bucket != "interactions" or ir_rows[0].get("export_status") == "exportable")
            recovered += drawn
            withheld = ir_rows[0].get("withheld", {}).get("reason") if ir_rows and not drawn else ""
            print(f"  {bucket} '{name}': admission={fate}; in canonical={in_canon}; drawn in PWML={drawn}"
                  + (f"; IR withheld: {withheld}" if withheld else ""))
        print(f"recovered (drawn) {recovered} of {len(targets)} put back")
        print("ORCH-794 removals:", [(e["bucket"], e["name"], e["reason"], (e.get("detail") or {}).get("decided_by"))
                                     for e in rr.get("removed_entities") or []
                                     if str(e.get("reason", "")).startswith("same_name_declared_as_another_kind")])
        dropped_actors = [(e["bucket"], e["name"], e["reason"]) for e in rr.get("removed_entities") or []
                          if (e.get("detail") or {}).get("actor_removed_from")]
        print("actors dropped from kept processes:", dropped_actors)
    return 0


if __name__ == "__main__":
    sys.exit(main())
