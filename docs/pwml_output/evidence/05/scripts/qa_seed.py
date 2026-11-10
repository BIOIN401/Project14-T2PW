import sys, json, hashlib
sys.path.insert(0, "C:/t/oa/src")
from t2pw.pipeline.pipeline import build_qa_feedback
p = json.load(open(sys.argv[1], encoding="utf-8"))
r = build_qa_feedback(p)
print(hashlib.sha1(json.dumps(r, sort_keys=False).encode()).hexdigest()[:10], len(r.get("orphan_components", [])), [c["size"] for c in r.get("orphan_components", [])][:6])
