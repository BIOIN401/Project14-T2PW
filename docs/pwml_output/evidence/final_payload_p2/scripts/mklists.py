import json
m = json.load(open("C:/t/fp/docs/pwml_output/evidence/final_payload_p2/import/baseline_map.json"))
D = "C:/t/fp/docs/pwml_output/evidence/final_payload_p2/"
for L, key in (("off5", "off5_written"), ("on5", "on5_written")):
    xs = [x for x in m if x[key] and x["off5_written"] and x["on5_written"]]
    open(D + "import/list_%s.txt" % L, "w").write("\n".join(D + "replay/%s/pwml/%d.pwml" % (L, x["idx"]) for x in xs) + "\n")
    print(L, len(xs))
open(D + "import/list_baseline.txt", "w").write("\n".join(x["file"].replace("\\", "/") for x in m) + "\n")
