"""measure.py [OFF_LABEL] [ON_LABEL]  -> C:/t/ev-fp/replay/<ON>/MEASURE.md, MEASURE.json, diff_sample.md, ids_gained.jsonl"""
import collections, difflib, glob, json, re, sys
from pathlib import Path
from lxml import etree

OFF = sys.argv[1] if len(sys.argv) > 1 else "fp2offb"
ON = sys.argv[2] if len(sys.argv) > 2 else "on1"
R = Path("C:/t/ev-fp/replay")
IDTAGS = ("chebi-id", "kegg-id", "hmdb-id", "pubchem-cid", "cas", "drugbank-id")


def load(label):
    d = {}
    for f in sorted(glob.glob(str(R / label / "results_*.jsonl"))):
        for line in open(f, encoding="utf-8"):
            r = json.loads(line)
            d[r["idx"]] = r
    return d


off, on = load(OFF), load(ON)
M = {"off_label": OFF, "on_label": ON, "runs_off": len(off), "runs_on": len(on)}
idxs = sorted(set(off) & set(on))
grp = lambda i: on[i]["group"].split("_")[0]


def pw(label, i):
    return R / label / "pwml" / ("%d.pwml" % i)


# (a)
tab = collections.defaultdict(lambda: collections.Counter())
changes = []
for i in idxs:
    g = grp(i)
    tab[g]["runs"] += 1
    tab[g]["off_written"] += bool(off[i]["pwml_written"])
    tab[g]["on_written"] += bool(on[i]["pwml_written"])
    if off[i]["pwml_written"] != on[i]["pwml_written"]:
        changes.append({"idx": i, "group": g, "off": off[i]["pwml_written"], "on": on[i]["pwml_written"],
                        "on_error": on[i]["error"][:200], "on_stage": on[i]["stage"],
                        "kind": "REGRESSION" if off[i]["pwml_written"] else "gained"})
M["a_groups"] = {g: dict(c) for g, c in tab.items()}
M["a_changes"] = changes
M["a_regressions"] = [c["idx"] for c in changes if c["kind"] == "REGRESSION"]

# (b)
ents = []
by = {k: collections.Counter() for k in ("kind", "field", "source", "match_type", "kind_field")}
touched = set()
refused = collections.Counter()
for i in idxs:
    for e in on[i].get("fill_entries") or []:
        ents.append({"idx": i, "entity": e.get("entity"), "kind": e.get("kind"), "field": e.get("field"),
                     "accession": e.get("accession"), "source": e.get("source"), "match_type": e.get("match_type")})
        by["kind"][e.get("kind")] += 1
        by["field"][e.get("field")] += 1
        by["source"][str(e.get("source"))] += 1
        by["match_type"][str(e.get("match_type"))] += 1
        by["kind_field"]["%s.%s" % (e.get("kind"), e.get("field"))] += 1
        touched.add(i)
    for k, v in ((on[i].get("fill_summary") or {}).get("refused") or {}).items():
        refused[k] += v
M["b_ids_gained"] = len(ents)
M["b_by"] = {k: dict(v) for k, v in by.items()}
M["b_runs_touched"] = len(touched)
M["b_refused_total"] = sum(refused.values())
M["b_refused_by_reason"] = dict(refused)
M["b_fill_errors"] = [{"idx": i, "error": on[i].get("fill_error")} for i in idxs if on[i].get("fill_error")]
with open(R / ON / "ids_gained.jsonl", "w", encoding="utf-8") as fh:
    for e in ents:
        fh.write(json.dumps(e) + "\n")


# (c)
def comp_ids(path):
    t = etree.parse(str(path))
    n = w = 0
    for c in t.iter("compound"):
        if c.find("name") is None:
            continue
        n += 1
        if any((c.findtext(x) or "").strip() for x in IDTAGS):
            w += 1
    return n, w


only_on = [i for i in idxs if on[i]["pwml_written"] and not off[i]["pwml_written"]]
M["c_only_on"] = [{"idx": i, "group": grp(i), "counts": on[i]["counts"]} for i in only_on]
per = []
for i in idxs:
    if on[i]["pwml_written"] and off[i]["pwml_written"]:
        a, b = comp_ids(pw(OFF, i)), comp_ids(pw(ON, i))
        per.append({"idx": i, "compounds_off": a[0], "with_id_off": a[1], "compounds_on": b[0], "with_id_on": b[1]})
M["c_compounds_with_id"] = {"runs": len(per), "total_compounds_off": sum(p["compounds_off"] for p in per),
                            "with_id_off": sum(p["with_id_off"] for p in per), "with_id_on": sum(p["with_id_on"] for p in per),
                            "runs_increased": sum(p["with_id_on"] > p["with_id_off"] for p in per)}
M["c_per_run"] = per


# (d)
def title_ref(path):
    t = etree.parse(str(path))
    pwy = next(t.iter("pathway"), None)
    nm = (pwy.findtext("name") or "") if pwy is not None else ""
    pm = [(r.findtext("pubmed-id") or "").strip() for r in t.iter("reference")]
    return nm, any(pm), len(pm) > 0


wr = [i for i in idxs if on[i]["pwml_written"]]
tc = collections.Counter()
for i in wr:
    nm, pmid, anyref = title_ref(pw(ON, i))
    tc["written"] += 1
    tc["real_title"] += nm != "Generated Pathway"
    tc["pubmed_id_nonempty"] += pmid
    tc["any_reference_element"] += anyref
    tc["title_source:" + str(on[i].get("title_source"))] += 1
    tc["off_title_real"] += (title_ref(pw(OFF, i))[0] != "Generated Pathway") if off[i]["pwml_written"] else 0
M["d_titles"] = dict(tc)

# (e)(f)
diff = []
for i in idxs:
    if on[i]["pwml_written"] and off[i]["pwml_written"] and on[i]["sha256"] != off[i]["sha256"]:
        diff.append(i)
M["e_differ_from_off"] = len(diff)
M["e_identical_to_off"] = sum(1 for i in idxs if on[i]["pwml_written"] and off[i]["pwml_written"]) - len(diff)
M["e_differ_idx"] = diff


def udiff(i):
    a = pw(OFF, i).read_text(encoding="utf-8").splitlines()
    b = pw(ON, i).read_text(encoding="utf-8").splitlines()
    out = list(difflib.unified_diff(a, b, "flag-off", "flag-on", n=2, lineterm=""))
    return out[:90]


# varied: one per (group, kind-of-change): most ids, a title-only, a reference, a species id, a P0-only-on
def nids(i):
    return len(on[i].get("fill_entries") or [])


pick = []
if diff:
    pick.append(max(diff, key=nids))
    for pred in (lambda i: nids(i) == 0, lambda i: bool(on[i].get("reference")), lambda i: grp(i) == "P0",
                 lambda i: any(e.get("kind") == "species" for e in on[i].get("fill_entries") or []),
                 lambda i: grp(i) == "P1", lambda i: True):
        for i in diff:
            if i not in pick and pred(i):
                pick.append(i)
                break
        if len(pick) >= 5:
            break
with open(R / ON / "diff_sample.md", "w", encoding="utf-8") as fh:
    fh.write("# Diff sample (flag-off vs flag-on), %d of %d differing PWML\n\n" % (len(pick[:5]), len(diff)))
    for i in pick[:5]:
        fh.write("## idx %d (%s), ids written: %d, title: %s\n\n```diff\n%s\n```\n\n" % (i, grp(i), nids(i), on[i].get("title"), "\n".join(udiff(i))))
M["f_diff_sample_idx"] = pick[:5]

# (4) well-formedness and QA
bad = []
for i in wr:
    try:
        etree.parse(str(pw(ON, i)))
    except Exception as e:
        bad.append({"idx": i, "error": str(e)[:200]})
M["wellformed_bad"] = bad
M["wellformed_checked"] = len(wr)
qa = lambda d: {"written": sum(1 for r in d.values() if r["pwml_written"]),
                "qa_ok_false": [i for i, r in d.items() if r["pwml_written"] and r.get("qa_ok") is False],
                "qa_errors_total": sum(r.get("qa_errors") or 0 for r in d.values() if r["pwml_written"]),
                "qa_warnings_total": sum(r.get("qa_warnings") or 0 for r in d.values() if r["pwml_written"])}
M["qa_off"], M["qa_on"] = qa(off), qa(on)

# live requests
reqs = [json.loads(x) for x in open(R / ON / "requests.jsonl", encoding="utf-8")] if (R / ON / "requests.jsonl").exists() else []
M["live_requests_total"] = sum(x["live"] for x in reqs)
M["live_request_errors_total"] = sum(x["errors"] for x in reqs)
M["passes"] = dict(collections.Counter(x["pass"] for x in reqs))
last = collections.defaultdict(int)
for x in reqs:
    last[x["pass"]] += x["live"]
M["live_requests_by_pass"] = dict(last)

json.dump(M, open(R / ON / "MEASURE.json", "w", encoding="utf-8"), indent=1, default=str)
L = ["# MEASURE %s vs %s\n" % (OFF, ON), "## (a) PWML written, flag off -> on\n", "| group | runs | off | on |", "|---|---|---|---|"]
for g, c in sorted(tab.items()):
    L.append("| %s | %d | %d | %d |" % (g, c["runs"], c["off_written"], c["on_written"]))
L.append("\nState changes: %s\n" % (json.dumps(changes) if changes else "none"))
L.append("## (b) ids gained: %d in %d runs; refused %d %s\n" % (len(ents), len(touched), M["b_refused_total"], dict(refused)))
for k, v in M["b_by"].items():
    L.append("- by %s: %s" % (k, v))
L.append("\n## (c)\n- written only with flag on: %s\n- compounds with any db id (runs written both ways): %s\n" % (
    json.dumps(M["c_only_on"]), json.dumps(M["c_compounds_with_id"])))
L.append("## (d) titles/references\n%s\n" % json.dumps(M["d_titles"]))
L.append("## (e) PWML differing from flag-off: %d (identical: %d)\n" % (len(diff), M["e_identical_to_off"]))
L.append("## QA/well-formedness\n- not well-formed: %d of %d\n- qa off: %s\n- qa on: %s\n" % (len(bad), len(wr), M["qa_off"], M["qa_on"]))
L.append("## Live requests\n- total %d (errors %d), by pass %s\n" % (M["live_requests_total"], M["live_request_errors_total"], M["live_requests_by_pass"]))
open(R / ON / "MEASURE.md", "w", encoding="utf-8").write("\n".join(L))
print("\n".join(L))
