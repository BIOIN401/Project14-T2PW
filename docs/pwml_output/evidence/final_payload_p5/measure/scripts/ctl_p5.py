import json, os
E = "C:/t/fp/docs/pwml_output/evidence/final_payload_p4/d3/import/controls/results.jsonl"
I = "C:/t/p5scratch/measure/import/"
def L(p): return {os.path.basename(json.loads(l)["file"].replace("\\", "/")): json.loads(l) for l in open(p, encoding="utf-8")}
S = {"P4 d3": L(E), "P5 (a)": L(I + "c03-out/results.jsonl"), "P5 (b)": L(I + "c04-out/results.jsonl")}
for f in sorted(S["P4 d3"]):
    print(f, {n: (sum(len(v) for v in s[f]["errors"].values()), s[f]["success"]) for n, s in S.items()})
