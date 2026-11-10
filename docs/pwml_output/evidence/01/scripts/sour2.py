import json, sys
for run in ("e2e4", "e2e5"):
    d = f"C:/t/{run}-run/work/item-1/attempt-1/out/"
    mg = json.load(open(d + "merged_payload.json", encoding="utf-8")); fm = json.load(open(d + "final_mapped.json", encoding="utf-8"))
    qr = json.load(open(d + "quarantine_report.json", encoding="utf-8")); rem = json.load(open(d + "removed_entity_report.json", encoding="utf-8"))
    print("=====", run)
    print("M interactions:", [(x.get("entity_1"), x.get("relationship"), x.get("entity_2")) for x in mg["processes"].get("interactions", [])])
    print("F interactions:", [(x.get("entity_1"), x.get("relationship"), x.get("entity_2")) for x in fm["processes"].get("interactions", [])])
    print("M proteins:", [e["name"] for e in mg["entities"].get("proteins", [])])
    print("F proteins:", [e["name"] for e in fm["entities"].get("proteins", [])], "F complexes:", [e["name"] for e in fm["entities"].get("protein_complexes", [])])
    print("quarantined:", [(q["process_kind"], q["name"], q["reason"]) for q in qr.get("quarantined", [])])
    print("accepted:", [(q["process_kind"], q["name"]) for q in qr.get("accepted", [])])
    print("removed entities:", [(x["bucket"], x["name"], x["reason"]) for x in rem.get("removed_entities", [])])
    print("locked quarantined:", [(x.get("reaction_name"), x.get("reason")) for x in fm.get("quarantined_locked_reactions", [])])
    print("coverage requested core:", json.load(open(d + "coverage_summary.json", encoding="utf-8")).get("requested_core_terms"))
