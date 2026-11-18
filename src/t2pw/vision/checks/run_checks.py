"""Run every development check over stored runs and write one JSON report.

    python -m t2pw.vision.checks.run_checks <run_dir> [<run_dir> ...] --scopes paper,default,full --out report.json
    python -m t2pw.vision.checks.run_checks --runlist C:/t/ev-cm/runlist.json --idx ok_idx.txt --scopes paper,default --out report.json

Per run and scope: generate pwml / sbml / sbgn / kgml with ``generate`` (offline, the run's own export controls),
validate the first three, compare all four (consistency.cross_format). Per run: faithfulness (paper view, extended view).
The report holds per-run flags, totals per scope and up to five examples per failure class.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

FMTS = ("pwml", "sbml", "sbgn", "kgml")


def _short(items, n=3):
    return [(x if isinstance(x, str) else json.dumps(x, default=str))[:200] for x in items[:n]]


def check_scope(big, scope_name: str, export, meta) -> Dict[str, Any]:
    from t2pw.vision import generate as G
    from t2pw.vision.checks import consistency, validate
    scope = G.parse_scope(scope_name)
    outs: Dict[str, bytes] = {}
    errors: Dict[str, str] = {}
    for fmt in FMTS:
        try:
            outs[fmt] = G.generate(big, scope, fmt, export=export, meta=meta)
        except Exception as e:  # GenerateError (refused view) or a crash: recorded, never hidden
            errors[fmt] = "%s: %s" % (type(e).__name__, str(e)[:300])
    rec: Dict[str, Any] = {"generated": sorted(outs), "errors": errors, "sizes": {f: len(b) for f, b in outs.items()}}
    if "sbml" in outs:
        v = validate.validate_sbml(outs["sbml"])
        rec["sbml"] = {"ok": v["ok"], "n_errors": v["n_errors"], "n_warnings": v["n_warnings"], "errors": _short([e["msg"] for e in v["errors"]])}
        rec["sbml"]["error_ids"] = sorted({e["id"] for e in v["errors"]})
    if "sbgn" in outs:
        v = validate.validate_sbgn(outs["sbgn"])
        rec["sbgn"] = {"ok": v["ok"], "xsd_errors": _short(v["xsd_errors"]), "schematron_failures": _short(v["schematron_failures"]),
                       "n_xsd": len(v["xsd_errors"]), "n_schematron": len(v["schematron_failures"])}
    ir = None
    if "pwml" in outs:
        from t2pw.vision.view import view
        ir = G._export_pwml(view(big, scope), dict(export or {}))[1]  # the IR behind the PWML bytes (same view, same controls)
        v = validate.validate_pwml(outs["pwml"], ir=ir)
        rec["pwml"] = {"ok": v["ok"], "n_errors": len(v["errors"]), "errors": _short([e.get("message", str(e)) for e in v["errors"]])}
    if len(outs) == len(FMTS):
        # the SBGN exporter's own report, on the same kept graph and IR generate() gave it (generate returns bytes only)
        from t2pw.vision.export_sbgn import sbgn_export
        sbgn_report = sbgn_export(G._kept(view(big, scope), dict(export or {})), ir=ir)[1]
        c = consistency.cross_format(outs, pwml_ir=ir, sbgn_report=sbgn_report)
        rec["sbgn_report"] = {"exported": sbgn_report["exported"], "omitted": sbgn_report["omitted"]}
        rec["consistency"] = {"ok": c["ok"], "counts": c["counts"], "failures": c["failures"],
                              "known": {k: (v if not isinstance(v, dict) else {x: y for x, y in v.items() if x in ("n", "pairs", "per_format")}) for k, v in c["known"].items()},
                              "known_examples": {k: v.get("examples", [])[:2] for k, v in c["known"].items() if isinstance(v, dict)}}
    return rec


def check_faithfulness(big, run_dir: Optional[str], cache=None) -> Dict[str, Any]:
    from t2pw.vision.checks import faithfulness
    text, path = faithfulness.find_paper_text(run_dir) if run_dir else (None, None)
    return {"text_file": path, "paper": faithfulness.paper_view(big, text), "extended": faithfulness.extended_view(big, cache)}


def check_run(run_dir, scopes: List[str], cache=None) -> Dict[str, Any]:
    from t2pw.clean_model.builder import read_run_meta
    from t2pw.vision import generate as G
    from t2pw.vision.payload import big_from_run
    big = big_from_run(run_dir)
    export, meta = G.export_options_from_run(run_dir), read_run_meta(str(run_dir))
    rec: Dict[str, Any] = {"run": str(run_dir), "scopes": {}}
    try:
        for s in scopes:
            rec["scopes"][s] = check_scope(big, s, export, meta)
        rec["faithfulness"] = check_faithfulness(big, run_dir, cache)
    finally:
        G._MEMO.clear()
        G._KEPT.clear()
    return rec


def aggregate(runs: List[Dict[str, Any]], scopes: List[str]) -> Dict[str, Any]:
    """Totals per scope (n over the runs that were checked) and up to five examples per failure class."""
    out: Dict[str, Any] = {"n_runs": len(runs), "crashed": [r["run"] for r in runs if "crash" in r][:10], "scopes": {}}
    for s in scopes:
        recs = [(r, r["scopes"][s]) for r in runs if s in r.get("scopes", {})]
        t: Dict[str, Any] = {"n": len(recs), "generated_all_formats": sum(1 for _, x in recs if len(x["generated"]) == len(FMTS)),
                             "refused": [], "sbml_valid": 0, "sbgn_valid": 0, "pwml_ok": 0, "cross_ok": 0}
        classes: Dict[str, Dict[str, Any]] = {}
        invalid: Dict[str, Dict[str, Any]] = defaultdict(lambda: {"runs": 0, "examples": []})
        known, renamed, sentinel = Counter(), Counter(), Counter()
        for r, x in recs:
            if len(x["generated"]) < len(FMTS):
                t["refused"].append({"run": r["run"][-90:], "errors": {k: v[:120] for k, v in list(x["errors"].items())[:1]}})
            t["sbml_valid"] += bool(x.get("sbml", {}).get("ok"))
            t["sbgn_valid"] += bool(x.get("sbgn", {}).get("ok"))
            t["pwml_ok"] += bool(x.get("pwml", {}).get("ok"))
            for fmt, key in (("sbml", "errors"), ("sbgn", "schematron_failures"), ("sbgn", "xsd_errors"), ("pwml", "errors")):
                if x.get(fmt) and not x[fmt]["ok"] and x[fmt].get(key):
                    d = invalid["%s:%s" % (fmt, key)]
                    d["runs"] += 1
                    if len(d["examples"]) < 5:
                        d["examples"].append({"run": r["run"][-70:], "msg": x[fmt][key][0][:200]})
            c = x.get("consistency")
            if c:
                t["cross_ok"] += bool(c["ok"])
                for f in c["failures"]:
                    d = classes.setdefault(f["class"], {"runs": 0, "total": 0, "examples": []})
                    d["runs"] += 1
                    d["total"] += f["n"]
                    for e in f["examples"]:
                        if len(d["examples"]) < 5:
                            d["examples"].append({"run": r["run"][-70:], "example": e})
                for k, v in c["known"].items():
                    if isinstance(v, dict):  # every renamed compartment pair, and the Unknown sentinel per format, summed over runs
                        renamed.update(v.get("pairs") or {})
                        for f_, n_ in (v.get("per_format") or {}).items():
                            sentinel[f_] += n_
                    known[k] += (v if isinstance(v, int) and not isinstance(v, bool) else (v.get("n", 0) if isinstance(v, dict) else int(bool(v))))
        t["validation_failure_kinds"] = dict(invalid)
        t["cross_format_failure_classes"] = dict(sorted(classes.items()))
        t["known_difference_totals"] = dict(known)
        t["compartment_renames"] = dict(renamed.most_common())
        t["unknown_sentinel_per_format"] = dict(sentinel)
        out["scopes"][s] = t
    fa = [r["faithfulness"] for r in runs if "faithfulness" in r]
    p = Counter()
    ex, am = [], Counter()
    for f in fa:
        pv = f["paper"]
        p["runs"] += 1
        p["runs_no_text"] += pv["text"] != "found"
        for k in ("n", "verbatim", "fragments", "not_found", "no_text", "short"):
            p[k] += pv[k]
        for k in ("verbatim", "fragments", "not_found"):
            p["before_fold_" + k] += pv["before_fold"][k]
        for k, v in pv["audit_modified_only"].items():
            if isinstance(v, int):
                am[k] += v
        for e in pv["not_found_examples"]:
            if len(ex) < 5:
                ex.append(e)
    found = p["verbatim"] + p["fragments"]
    judged = p["verbatim"] + p["fragments"] + p["not_found"]
    found0 = p["before_fold_verbatim"] + p["before_fold_fragments"]
    judged0 = found0 + p["before_fold_not_found"]
    out["faithfulness"] = {
        "found_rate_before_fold": round(found0 / judged0, 4) if judged0 else None,
        "found_rate_after_fold": round(found / judged, 4) if judged else None,
        "paper_quotes": dict(p), "found_rate_of_judged": round(found / judged, 4) if judged else None,
        "verbatim_rate_of_judged": round(p["verbatim"] / judged, 4) if judged else None,
        "not_found_examples": ex, "audit_modified_only_totals": dict(am),
        "extended": {"no_extension_runs": sum(f["extended"]["status"] == "no extension" for f in fa),
                     "database_elements": sum(f["extended"]["database"]["n"] for f in fa),
                     "database_no_cache": sum(f["extended"]["database"]["no_cache"] for f in fa),
                     "other_paper_quotes": sum(f["extended"]["other_paper"]["n"] for f in fa),
                     "other_paper_no_cache": sum(f["extended"]["other_paper"]["no_cache"] for f in fa)}}
    return out


def _run_list(a) -> List[Dict[str, Any]]:
    runs = [{"run": d, "idx": None} for d in a.run_dirs]
    if a.runlist:
        rl = [x for x in json.loads(Path(a.runlist).read_text(encoding="utf-8")) if x.get("q")]
        idx = [int(i) for i in (Path(a.idx).read_text() if a.idx and Path(a.idx).is_file() else a.idx or "").replace("\n", ",").split(",") if i.strip()]
        runs += [{"run": rl[i]["dir"], "idx": i} for i in (idx or range(len(rl)))]
    return runs


def tree_info() -> Dict[str, Any]:
    """The source tree this run imported t2pw from, and its git HEAD (so a report says which code made it)."""
    import subprocess
    import t2pw
    root = Path(t2pw.__file__).resolve().parents[2]
    git = lambda *a: subprocess.run(["git", "-C", str(root)] + list(a), capture_output=True, text=True, timeout=30).stdout.strip()  # noqa: E731
    try:
        return {"path": str(root), "head": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain", "--", "src"))}
    except Exception as e:  # not a git checkout, or no git: say so instead of failing the run
        return {"path": str(root), "head": None, "dirty": None, "error": str(e)[:100]}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m t2pw.vision.checks.run_checks", description=__doc__.split("\n\n")[0])
    ap.add_argument("run_dirs", nargs="*", help="run directories (final_mapped.json)")
    ap.add_argument("--runlist", help="runlist.json (entries with q=true), with --idx")
    ap.add_argument("--idx", help="comma-separated indexes into the q=true entries, or a file holding them")
    ap.add_argument("--scopes", default="paper,default,full")
    ap.add_argument("--cache", default=None, help="extension cache directory (extended-view checks; nothing is fetched)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--totals-out", default=None, help="also write only the totals (small enough to commit as evidence)")
    a = ap.parse_args(argv)
    scopes = [s.strip() for s in a.scopes.split(",") if s.strip()]
    todo, runs = _run_list(a), []
    t0 = time.time()
    for n, item in enumerate(todo):
        try:
            rec = check_run(item["run"], scopes, a.cache)
        except Exception:
            rec = {"run": item["run"], "scopes": {}, "crash": traceback.format_exc()[-600:]}
        rec["idx"] = item["idx"]
        runs.append(rec)
        print("%d/%d %s %s %.0fs" % (n + 1, len(todo), item["idx"], "CRASH" if "crash" in rec else "ok", time.time() - t0), flush=True)
        if (n + 1) % 10 == 0:
            Path(a.out + ".partial").write_text(json.dumps({"runs": runs}, default=str), encoding="utf-8")
    tree = tree_info()
    report = {"tree": tree, "scopes": scopes, "totals": aggregate(runs, scopes), "runs": runs}
    Path(a.out).write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
    if a.totals_out:
        Path(a.totals_out).write_text(json.dumps({"tree": tree, "scopes": scopes, "totals": report["totals"]}, indent=1, default=str), encoding="utf-8")
    print("tree: %s HEAD %s dirty=%s" % (tree["path"], tree["head"], tree["dirty"]), flush=True)
    print(json.dumps(report["totals"], indent=1, default=str)[:6000])
    return 0


if __name__ == "__main__":
    sys.exit(main())
