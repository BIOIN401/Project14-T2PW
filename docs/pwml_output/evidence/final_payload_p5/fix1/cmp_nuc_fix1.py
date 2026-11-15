import json
a = json.load(open("nucleicD/d_nucleic_measure.json"))["rows"]
b = json.load(open("fix1/d_nucleic_measure_fix1.json"))["rows"]
key = lambda r: (r["idx"], r["name"])
ba = {key(r): r for r in a}
for r in b:
    o = ba.get(key(r))
    if o and (o["status"], o["type"], o["lookup_code"]) != (r["status"], r["type"], r["lookup_code"]):
        print(r["idx"], r["name"], "|", o["status"], o["type"], o["by"], o["paper_reason"], o["lookup_code"], "->", r["status"], r["type"], r["paper_reason"], r["lookup_code"])
print("resolved after:", [(r["idx"], r["name"], r["type"], r["by"]) for r in b if r["status"] == "resolved"])
