"""A4: compact text summary of each sample final_payload (read-only)."""
import json, sys, collections
from pathlib import Path

ROOT = Path("C:/t/corr-assess")
sample = json.loads((ROOT / "sample.json").read_text(encoding="utf-8"))
only = set(int(a) for a in sys.argv[1:])
out = []
for s in sample:
    idx = s["idx"]
    if only and idx not in only:
        continue
    d = json.loads((ROOT / "base" / str(idx) / "final_payload.json").read_text(encoding="utf-8"))
    P = d["processes"]
    E = d["entities"]
    out.append(f"##### idx {idx} | {s['organism']} | {s['pathway_name']}")
    out.append("STATES: " + " ; ".join(f"{b.get('name')}={b.get('subcellular_location')}/{b.get('species')}" for b in d.get("biological_states") or []))
    out.append("SPECIES: " + " ; ".join(str(x.get("name")) for x in E.get("species") or []))
    seen = collections.Counter()
    for i, r in enumerate(P["reactions"]):
        enz = ",".join(f"{e.get('entity')}" for e in r.get("enzymes") or [])
        mods = ",".join(f"{m.get('entity')}[{m.get('role')}]" for m in r.get("modifiers") or [] if m.get("role") not in ("catalyst", "enzyme"))
        key = (tuple(sorted(r["inputs"])), tuple(sorted(r["outputs"])), r.get("biological_state"))
        seen[key] += 1
        prov = (r.get("provenance_lineage") or [{}])[0].get("origin", "?")
        ev = (r.get("evidence") or "").replace("\n", " ")[:160]
        out.append(f" R{i} {' + '.join(r['inputs'])} -> {' + '.join(r['outputs'])} | enz={enz} | mods={mods} | st={r.get('biological_state')} | sp={r.get('spontaneous')} | {prov} | \"{ev}\"")
    d2 = [k for k, v in seen.items() if v > 1]
    if d2:
        out.append(f" EXACT-DUP reaction keys: {len(d2)}")
    for i, t in enumerate(P["transports"]):
        tr = ",".join(str(x.get("entity")) for x in t.get("transporters") or [])
        out.append(f" T{i} {t.get('cargo')}: {t.get('from_biological_state')} -> {t.get('to_biological_state')} | by={tr} | \"{(t.get('evidence') or '')[:120]}\"")
    for i, it in enumerate(P["interactions"]):
        out.append(f" I{i} {it.get('entity_1')} -{it.get('relationship')}-> {it.get('entity_2')} | st={it.get('biological_state')} | \"{(it.get('evidence') or '')[:100]}\"")
    pcs = []
    for pc in E.get("protein_complexes") or []:
        comps = ",".join(str(c.get("name") if isinstance(c, dict) else c) for c in pc.get("components") or [])
        fn = (pc.get("mapping_meta") or {}).get("functional_enzyme_name")
        pcs.append(f"{pc.get('name')}({fn})=[{comps}]")
    out.append(" COMPLEXES: " + " ; ".join(pcs))
    out.append(" PROTEINS: " + " ; ".join(f"{p.get('name')}/{(p.get('mapped_ids') or {}).get('uniprot')}/{p.get('species')}" for p in E.get("proteins") or []))
    out.append(" QUARANTINED_RXN: " + json.dumps(d.get("quarantined_locked_reactions"), ensure_ascii=False)[:600])
    out.append(" QUARANTINED_PROT: " + json.dumps([q.get("name") if isinstance(q, dict) else q for q in d.get("quarantined_proteins") or []], ensure_ascii=False)[:300])
print("\n".join(out))
