"""P5 final measurement: replay comparisons (identity vs P4 d3 / P3 flag-off, flag-on counters, wildcard checker).
Usage: python cmp_p5.py   -> prints and writes evidence/final_payload_p5/measure/cmp_p5.out.txt"""
import collections, hashlib, json
from pathlib import Path
from lxml import etree

E = Path("C:/t/fp/docs/pwml_output/evidence")
NEW = Path("C:/t/p5scratch/measure/replay")
D3, D3OFF = E / "final_payload_p4/d3/replay/on", E / "final_payload_p4/d3/replay/off"
P3OFF, P3ON = E / "final_payload_p3/replay/off", E / "final_payload_p3/replay/on"
out = []


def p(*a):
    print(*a); out.append(" ".join(str(x) for x in a))


def load(d): return {json.loads(l)["idx"]: json.loads(l) for l in open(d / "results_0.jsonl", encoding="utf-8")}
def sha(d, i): return hashlib.sha256((d / "pwml" / f"{i}.pwml").read_bytes()).hexdigest()
def written(r): return {i for i, x in r.items() if x["pwml_written"]}


def states_of(path):
    root = etree.parse(str(path)).getroot()
    rows = {}
    for sec in ("tissues", "cell-types", "subcellular-locations"):
        for n in root.findall(f".//{sec}/{sec[:-1]}"):
            rows[(sec, (n.findtext("id") or "").strip())] = ((n.findtext("name") or "").casefold(), (n.findtext("ontology-id") or "").strip() or None)
    res = []
    for st in root.findall(".//biological-states/biological-state"):
        comps = [(st.findtext("species-id") or "").strip() or None]
        for sec, tag in (("tissues", "tissue-id"), ("cell-types", "cell-type-id"), ("subcellular-locations", "subcellular-location-id")):
            v = (st.findtext(tag) or "").strip()
            comps.append(rows.get((sec, v), ("?" + v, None)) if v else None)
        res.append(tuple(comps))
    return res


def check(d, idxs):
    a_t = b_t = fa = fb = 0
    for i in sorted(idxs):
        sts = states_of(d / "pwml" / f"{i}.pwml")
        a = sum(1 for s in sts if all(c is None for c in s[1:]))
        b = sum(1 for k, s in enumerate(sts) if any(j != k and all(x is None or x == y for x, y in zip(s, t)) for j, t in enumerate(sts)))
        a_t += a; b_t += b; fa += bool(a); fb += bool(b)
    return {"species_only_states": a_t, "files_species_only": fa, "wildcard_bindable_states": b_t, "files_bindable": fb}


off, on = load(NEW / "off"), load(NEW / "on")
d3off, d3on, p3off = load(D3OFF), load(D3), load(P3OFF)
p("== (1) FLAG OFF (T2PW_FINAL_PAYLOAD=0), new vs stored")
p("written new off / P4d3 off / P3 off:", len(written(off)), len(written(d3off)), len(written(p3off)),
  "same set d3:", written(off) == written(d3off), "same set p3:", written(off) == written(p3off))
for name, d, ref in (("P4 d3 off", D3OFF, d3off), ("P3 off", P3OFF, p3off)):
    both = written(off) & written(ref)
    diff = sorted(i for i in both if sha(NEW / "off", i) != sha(d, i))
    p(f"identical to {name}:", len(both) - len(diff), "of", len(both), "different:", diff)
errs = collections.Counter((x["stage"], x["error"][:60]) for x in off.values() if not x["pwml_written"])
p("off unwritten by stage:", dict(collections.Counter(x["stage"] for x in off.values() if not x["pwml_written"])))
p("off unwritten by (stage,error) differing from d3 off:", [(i, off[i]["stage"], d3off[i]["stage"]) for i in off if off[i]["stage"] != d3off[i]["stage"]])
p("== (2) FLAG ON")
p("written new on / P4d3 on:", len(written(on)), len(written(d3on)), "same set:", written(on) == written(d3on), "only new", sorted(written(on) - written(d3on)), "only d3", sorted(written(d3on) - written(on)))
ch = sorted(i for i in written(on) & written(d3on) if sha(NEW / "on", i) != sha(D3, i))
p("on files byte-different from P4 d3 on:", len(ch), "of", len(written(on) & written(d3on)), ch)
p("on files byte-different from P3 on:", sum(1 for i in written(on) & written(load(P3ON)) if sha(NEW / "on", i) != sha(P3ON, i)))
p("non-stage3 errors on:", [(i, x["error"][:80], x["stage"]) for i, x in on.items() if x["error"] and x["stage"] != "stage3_revalidation"][:8])
p("fill_error runs:", [i for i, x in on.items() if x.get("fill_error")])
p("live requests", sum(x.get("live_requests", 0) for x in on.values()), "fetch errors", sum(x.get("fetch_errors", 0) for x in on.values()),
  "paper text found", sum(1 for x in on.values() if x.get("paper_text_found")), "of", len(on))
keys = ["locations_dropped_no_id", "cell_types_dropped_no_id", "tissues_dropped_no_id", "components_kept_wildcard_guard", "locations_kept_state_collision",
        "membranes_kept_unresolved", "nucleic_acids_type_unset"]
allk = set()
for x in on.values(): allk |= set(x.get("layout_debug_counts") or {})
extra = sorted(k for k in allk if "nucleic" in k or "unknown" in k or "unresolved" in k or "wildcard" in k)
for k in dict.fromkeys(keys + extra):
    t = {n: sum((x.get("layout_debug_counts") or {}).get(k, 0) for i, x in r.items() if x["pwml_written"]) for n, r in (("new", on), ("d3", d3on))}
    f = sum(1 for x in on.values() if x["pwml_written"] and (x.get("layout_debug_counts") or {}).get(k))
    p("counter", k, "written-runs sum", t, "files(new)", f)
ont = collections.Counter(); kinds = collections.Counter(); nuc = collections.Counter()
for x in on.values():
    for k, v in ((x.get("ontology_report") or {}).get("counts") or {}).items(): ont[k] += v
    for e in x.get("fill_entries") or []:
        if e.get("field") == "ontology_id": kinds[e["kind"]] += 1
    for k, v in (x.get("nucleic_types_report") or {}).items():
        if isinstance(v, (int, float)): nuc[k] += v
        elif isinstance(v, dict): nuc.update({k + "." + kk: vv for kk, vv in v.items() if isinstance(vv, (int, float))})
p("ontology counts", dict(ont), "ontology_id entries by kind", dict(kinds))
p("nucleic_types_report sums", dict(nuc))
p("wildcard check NEW on ", check(NEW / "on", written(on)))
p("wildcard check P4d3 on", check(D3, written(d3on)))
p("wildcard check NEW off", check(NEW / "off", written(off)))
open(E / "final_payload_p5/measure/cmp_p5.out.txt", "w", encoding="utf-8").write("\n".join(out) + "\n")
