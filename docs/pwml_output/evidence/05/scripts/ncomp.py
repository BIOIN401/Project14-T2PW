import sys, json
sys.path.insert(0, "C:/t/oa/src")
from t2pw.pipeline.pipeline import build_qa_feedback
import hashlib
for q in json.load(open("qa_seed_unique.json")):
    p="C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW/"+q["payload"]
    r=build_qa_feedback(json.load(open(p,encoding="utf-8")))
    print(r["n_components"]-1, len(r["orphan_components"]), len(r["missing_links_suspected"]), q["distinct"])
