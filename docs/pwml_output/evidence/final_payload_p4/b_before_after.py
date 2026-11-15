"""Old (git f66b3ef2) vs new process_normalizer on every stored merged_payload.json."""
import importlib.util, json, subprocess, sys, os, glob, copy
ROOT = "C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW"
sys.path.insert(0, "C:/t/fp/src")
old_src = subprocess.check_output(["git", "-C", "C:/t/fp", "show", "f66b3ef2:src/t2pw/pipeline/process_normalizer.py"])
oldp = "C:/t/pn_old.py"
open(oldp, "wb").write(old_src)
spec = importlib.util.spec_from_file_location("t2pw.pipeline.pn_old", oldp)
old = importlib.util.module_from_spec(spec); sys.modules["t2pw.pipeline.pn_old"] = old; spec.loader.exec_module(old)
import t2pw.pipeline.process_normalizer as new
assert old is not new and old.ensure_autostates is not new.ensure_autostates

def diff(a, b, path=""):
    if type(a) != type(b): return [(path, a, b)]
    if isinstance(a, dict):
        out = []
        for k in sorted(set(a) | set(b)):
            if k not in a or k not in b: out.append((f"{path}/{k}", a.get(k, "<absent>"), b.get(k, "<absent>")))
            else: out += diff(a[k], b[k], f"{path}/{k}")
        return out
    if isinstance(a, list):
        if len(a) != len(b): return [(path, f"len {len(a)}", f"len {len(b)}")]
        out = []
        for i, (x, y) in enumerate(zip(a, b)): out += diff(x, y, f"{path}/{i}")
        return out
    return [] if a == b else [(path, a, b)]

files = sorted(glob.glob(ROOT + "/runs_*/**/merged_payload.json", recursive=True))
res = {"n_files": len(files), "compared": 0, "errors": [], "runs_changed": 0, "transports_changed": [],
       "outside_transport_diffs": [], "report_diffs": [], "auto_both_old": [], "auto_both_new": []}
def eq_pairs(d):
    out = []
    for i, t in enumerate(d.get("processes", {}).get("transports", []) or []):
        if isinstance(t, dict) and t.get("from_biological_state") and t.get("from_biological_state") == t.get("to_biological_state"):
            out.append((i, t.get("name"), t.get("cargo"), t["from_biological_state"]))
    return out
for f in files:
    run = f[len(ROOT) + 1:].rsplit("/merged_payload.json", 1)[0]
    try:
        p = json.load(open(f, encoding="utf-8"))
        mode = "research" if "/research" in run else "pathwhiz"
        a, ra = old.normalize_process_payload(copy.deepcopy(p), mode=mode)
        b, rb = new.normalize_process_payload(copy.deepcopy(p), mode=mode)
    except Exception as e:
        res["errors"].append({"run": run, "err": repr(e)[:200]}); continue
    res["compared"] += 1
    res["auto_both_old"] += [{"run": run, "i": i, "transport": n, "cargo": c, "state": s} for i, n, c, s in eq_pairs(a)]
    res["auto_both_new"] += [{"run": run, "i": i, "transport": n, "cargo": c, "state": s} for i, n, c, s in eq_pairs(b)]
    ds = diff(a, b)
    if ds: res["runs_changed"] += 1
    for path, x, y in ds:
        parts = path.split("/")
        if len(parts) >= 5 and parts[1] == "processes" and parts[2] == "transports" and parts[4] in ("from_biological_state", "to_biological_state"):
            ti = int(parts[3]); t = a["processes"]["transports"][ti]
            res["transports_changed"].append({"run": run, "idx": ti, "transport": t.get("name"), "cargo": t.get("cargo"), "field": parts[4], "before": x, "after": y,
                "other_side_after": b["processes"]["transports"][ti].get("to_biological_state" if parts[4].startswith("from") else "from_biological_state")})
        else:
            res["outside_transport_diffs"].append({"run": run, "path": path, "before": str(x)[:100], "after": str(y)[:100]})
    for path, x, y in diff(ra, rb): res["report_diffs"].append({"run": run, "path": path, "before": str(x)[:100], "after": str(y)[:100]})
json.dump(res, open("C:/t/fp/docs/pwml_output/evidence/final_payload_p4/b_before_after.json", "w", encoding="utf-8"), indent=1)
print({k: (len(v) if isinstance(v, list) else v) for k, v in res.items()})
