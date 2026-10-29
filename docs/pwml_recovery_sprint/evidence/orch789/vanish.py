"""Read-only: proteins present after Stage 1 merge that reach neither the final payload nor any removal record."""
import json, os, re, sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
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


def walk(o):
    if isinstance(o, dict):
        yield o
        for v in o.values():
            yield from walk(v)
    elif isinstance(o, list):
        for v in o:
            yield from walk(v)


for leg, rel in LEGS.items():
    d = os.path.join(ROOT, rel)
    merged = load(d, "merged_payload.json")
    fm = load(d, "final_mapped.json")
    names_final = set()
    for b, rows in (fm.get("entities") or {}).items():
        for r in rows or []:
            if isinstance(r, dict):
                names_final.add(norm(r.get("name")))
                for s in r.get("synonyms") or []:
                    names_final.add(norm(s))
    qp = {norm(q.get("name")) for q in fm.get("quarantined_proteins") or []}
    rm = {norm(r.get("name")) for r in load(d, "removed_entity_report.json").get("removed_entities") or []}
    gap = {}
    for o in walk(load(d, "gap_iteration_summary.json")):
        k = o.get("issue_key")
        if isinstance(k, str) and k.startswith("protein:") and "status" in o:
            api = ((o.get("background") or {}).get("protein_api") or {})
            gap[norm(k[8:])] = f"gap={o.get('status')}/{api.get('status')}/{api.get('reason')}/cand={api.get('candidate_count')}"
    gate = {}
    for o in walk(load(d, "initial_stage3_gate_report.json")):
        if isinstance(o.get("reason"), str) and o["reason"].startswith("Protein '"):
            m = re.match(r"Protein '(.+?)' is missing", o["reason"])
            if m:
                gate[norm(m.group(1))] = "initgate:no_id"
    print(f"\n######## {leg}")
    for r in (merged.get("entities") or {}).get("proteins") or []:
        n = norm(r.get("name"))
        if n in names_final or n in qp or n in rm:
            continue
        print(f"VANISHED {r.get('name')!r}: {gap.get(n, 'gap=none')} {gate.get(n, '')}")
    # comparison: missing-identity proteins that DID reach quarantine
    for r in load(d, "removed_entity_report.json").get("removed_entities") or []:
        if r.get("bucket") == "proteins" and r.get("had_external_identity") is False:
            n = norm(r.get("name"))
            print(f"REACHED-QUARANTINE {r.get('name')!r}: {gap.get(n, 'gap=none')} {gate.get(n, '')}")
