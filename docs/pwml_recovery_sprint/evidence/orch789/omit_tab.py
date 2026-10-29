"""Read-only: tabulate every omitted process per leg with its real root cause inputs."""
import json, os, re, sys
from collections import Counter

ROOT = r"C:\Users\Angad\Desktop\SummerBIOIN\Project14-T2PW\.claude\worktrees"
LEGS = {
    "chlorophyll@785": r"orch785run\runs_papers\ORCH-785\legs\chlorophyll-strict\2026-10-28_1743\papers\PMC5130987\strict",
    "wnt_ca@785": r"orch785run\runs_papers\ORCH-785\legs\wnt_ca-strict\2026-10-28_1910\papers\PMC10425814\strict",
    "sour_taste_v2@785": r"orch785run\runs_papers\ORCH-785\legs\sour_taste_v2-strict\2026-10-28_2058\papers\PMC7059786\strict",
    "enterobactin@784": r"orch784run\runs_papers\ORCH-784\legs\enterobactin-strict\2026-10-27_1931\papers\PMC12096016\strict",
    "menaquinone@784": r"orch784run\runs_papers\ORCH-784\legs\menaquinone-strict\2026-10-28_0208\papers\PMC12421875\strict",
    "wnt_pcp@784": r"orch784run\runs_papers\ORCH-784\legs\wnt_pcp-strict\2026-10-27_2309\papers\PMC10425814\strict",
}


def load(d, f):
    p = os.path.join(d, f)
    if not os.path.exists(p):
        return {}
    with open(p, encoding="utf-8") as h:
        return json.load(h)


def norm(s):
    return re.sub(r"[^a-z0-9]", "", str(s or "").lower())


def ents(payload):
    out = {}
    for b, rows in (payload.get("entities") or {}).items():
        for r in rows or []:
            if isinstance(r, dict) and r.get("name"):
                out.setdefault(norm(r["name"]), []).append((b, r))
    return out


def pid(r):
    for c in (r, r.get("mapped_ids") or {}, r.get("ids") or {}, r.get("mapping_meta") or {}):
        for k in ("uniprot", "uniprot_id", "drugbank", "drugbank_id"):
            if isinstance(c, dict) and c.get(k):
                return f"{k}={c.get(k)}"
    return "noid"


def fate(name, merged, final, qprot, removed):
    n = norm(name)
    bits = []
    for b, r in merged.get(n, []):
        bits.append(f"merged:{b}")
    for b, r in final.get(n, []):
        x = f"final:{b}"
        if b == "proteins":
            x += f"({pid(r)})"
        if b == "protein_complexes":
            x += "(comps=" + ",".join(str(c.get("name") if isinstance(c, dict) else c) for c in r.get("components") or []) + ")"
        bits.append(x)
    for q in qprot.get(n, []):
        res = ((q.get("original_row") or {}).get("mapping_meta") or {}).get("resolution") or {}
        amb = ((q.get("original_row") or {}).get("mapping_meta") or {}).get("ambiguity") or {}
        bits.append(f"qprot:{q.get('reason')}|{q.get('pass')}|res={res.get('status')}/{res.get('issue')}/{res.get('failed_check')}|cand={amb.get('candidate_count')}/{amb.get('verified_count')}")
    for r in removed.get(n, []):
        bits.append(f"removed:{r.get('bucket')}:{str(r.get('reason'))[:70]}")
    return "; ".join(bits) or "NOWHERE"


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    only = sys.argv[1:]
    for leg, rel in LEGS.items():
        if only and not any(o in leg for o in only):
            continue
        d = os.path.join(ROOT, rel)
        qr = load(d, "quarantine_report.json")
        merged = ents(load(d, "merged_payload.json"))
        fm = load(d, "final_mapped.json")
        final = ents(fm)
        qprot = {}
        for q in fm.get("quarantined_proteins") or []:
            qprot.setdefault(norm(q.get("name")), []).append(q)
        removed = {}
        for r in (load(d, "removed_entity_report.json").get("removed_entities") or []):
            removed.setdefault(norm(r.get("name")), []).append(r)
        print(f"\n######## {leg}")
        c = Counter()
        for a in qr.get("admissions") or []:
            st = a.get("state", "")
            if not st.startswith("quarantined"):
                continue
            row = (a.get("detail") or {}).get("row") or {}
            reason = a.get("reason")
            c[re.sub(r":.*", "", str(reason))] += 1
            names = []
            for k in ("entity_1", "entity_2", "left", "right", "source", "target"):
                if row.get(k):
                    names.append((k, row[k]))
            for k in ("enzymes", "transporters", "modifiers"):
                for e in row.get(k) or []:
                    if isinstance(e, dict):
                        names.append((k, e.get("entity")))
            for k in ("inputs", "outputs", "cargo", "elements"):
                for e in row.get(k) or []:
                    names.append((k, e if isinstance(e, str) else (e or {}).get("name") or (e or {}).get("entity")))
            print(f"Q|{a.get('bucket')}|{a.get('name')}|{reason}|ess={a.get('essential_participant')}|rel={row.get('relationship')}")
            ess = a.get("essential_participant")
            seen = set()
            for k, nm in names:
                if not nm or norm(nm) in seen:
                    continue
                seen.add(norm(nm))
                print(f"    {k}={nm}: {fate(nm, merged, final, qprot, removed)}")
        ir = load(d, "pwml_ir.json")
        for bucket in ("interactions", "reactions", "transports"):
            for r in (ir.get("processes") or {}).get(bucket) or []:
                if isinstance(r, dict) and r.get("export_status") == "withheld":
                    w = r.get("withheld") or {}
                    c["IR:" + str(w.get("reason"))] += 1
                    print(f"W|{bucket}|{r.get('name')}|{w.get('reason')}|raw={w.get('raw_interaction_type')}|L={w.get('left')}|R={w.get('right')}")
                    if w.get("reason") != "untyped_interaction":
                        for nm in (w.get("left"), w.get("right")):
                            print(f"    {nm}: {fate(nm, merged, final, qprot, removed)}")
        print("COUNTS", dict(c))


main()
