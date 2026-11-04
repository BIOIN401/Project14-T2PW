"""ORCH-804: the exact delta behind the R3 pin move in tests/test_pwml_ir_duplicate_row_refusal.py.

usage: r3_ir_delta.py dump <src> <leg final_mapped.json> <out.json>   (run once per tree)
       r3_ir_delta.py diff <base.json> <tip.json>
dump reproduces the test's own call sequence and writes the IR plus its sha256.
"""
import hashlib
import json
import sys


class _DownDb:
    last_error = "harvest_db_down"

    def available(self):
        return False


def flat(d, p=""):
    if isinstance(d, dict):
        for k, v in d.items():
            yield from flat(v, f"{p}/{k}")
    elif isinstance(d, list):
        for i, v in enumerate(d):
            yield from flat(v, f"{p}/{i}")
    else:
        yield p, d


if sys.argv[1] == "dump":
    sys.path.insert(0, sys.argv[2])
    from t2pw.pwml.ir import build_pwml_ir
    from t2pw.pwml.prefreeze_resolution import run_prefreeze_resolution

    staged = json.load(open(sys.argv[3], encoding="utf-8"))
    run_prefreeze_resolution(staged, strict_db=True, db_resolver=_DownDb(), name_index=None)
    ir, report = build_pwml_ir(staged, db_resolver=_DownDb(), strict_db=True, name_index=None)
    digest = hashlib.sha256(json.dumps(ir, sort_keys=True, indent=1, default=repr).encode()).hexdigest()
    json.dump({"sha256": digest, "ir": ir, "report": report}, open(sys.argv[4], "w", encoding="utf-8"), default=repr)
    print(digest)
else:
    a, b = (json.load(open(f, encoding="utf-8")) for f in sys.argv[2:4])
    print("base", a["sha256"], "\ntip ", b["sha256"])
    st = {s["key"]: s["name"] for s in b["ir"]["biological_states"]}
    fa, fb = dict(flat({"ir": a["ir"], "report": a.get("report")})), dict(flat({"ir": b["ir"], "report": b.get("report")}))
    for k in sorted(set(fa) | set(fb)):
        if fa.get(k) != fb.get(k):
            print(f"{k}: {fa.get(k)!r} ({st.get(fa.get(k), '')}) -> {fb.get(k)!r} ({st.get(fb.get(k), '')})")
