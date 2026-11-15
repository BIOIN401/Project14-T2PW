import json, os, sys
R = "C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW/"
rows = [json.loads(l) for l in open("C:/t/ev-oa/funnel/out/runs_funnel.jsonl")]
RE = {json.loads(l)["path"]: json.loads(l) for l in open("C:/t/ev-oa/funnel/out/re_reports.jsonl")}
for r in rows:
    if not ("PMC7059786" in r["path"] or "e2e4" in r["path"] or "e2e5" in r["path"]): continue
    d = os.path.dirname(R + r["path"] if not r["path"].startswith("C:") else r["path"])
    print("\n=====", r["path"][-95:], "| qok", r["qok"], "| mode", r["mode"], "| pwml", r.get("pwml_file"))
    for st, n in (("S1", "stage1_payload"), ("F", "final_mapped")):
        p = json.load(open(d + f"/{n}.json", encoding="utf-8"))
        pr = p["processes"]
        print(f"  [{st}] entities: comp {len(p['entities'].get('compounds', []))} prot {len(p['entities'].get('proteins', []))} cplx {len(p['entities'].get('protein_complexes', []))} | rx {len(pr.get('reactions', []))} tr {len(pr.get('transports', []))} ia {len(pr.get('interactions', []))}")
        if st == "S1":
            print("   S1 compounds:", [e["name"] for e in p["entities"].get("compounds", [])])
            print("   S1 proteins:", [e["name"] for e in p["entities"].get("proteins", [])][:14])
        for t in pr.get("transports", []):
            print("   TR:", t.get("name"), "| cargo", t.get("cargo"), "|", [x.get("entity") for x in t.get("transporters", [])], "|", t.get("from_biological_state"), "->", t.get("to_biological_state"))
        for t in pr.get("interactions", []):
            print("   IA:", t.get("entity_1"), "|", t.get("relationship"), "|", t.get("entity_2"))
        for t in pr.get("reactions", []):
            print("   RX:", t.get("name"), t.get("inputs"), "->", t.get("outputs"))
    x = RE[r["path"]]
    print("  pwml_coverage:", json.dumps(x.get("pwml_coverage")))
    print("  omitted:", [(o["kind"], o["name"][:40], o["reason"]) for o in x.get("omitted") or []][:12])
    print("  withheld:", x.get("withheld_reasons"))
    print("  pwml counts:", r.get("pwml"))
