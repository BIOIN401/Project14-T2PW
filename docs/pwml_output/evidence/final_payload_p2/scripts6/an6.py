import collections, json, glob, hashlib, sys
from pathlib import Path
from lxml import etree
E = Path("C:/t/fp/docs/pwml_output/evidence/final_payload_p2")
def load(d, label):
    out = {}
    for f in glob.glob(str(E / d / label / "results_*.jsonl")):
        for l in open(f, encoding="utf-8"):
            r = json.loads(l); out[r["idx"]] = r
    return out
def sha(d, label, i):
    return hashlib.sha256((E / d / label / "pwml" / f"{i}.pwml").read_bytes()).hexdigest()
res = {}
for d, l in (("replay", "base1"), ("replay6", "base6"), ("replay6", "off6"), ("replay6", "on6")):
    res[l] = (d, load(d, l))
def written(l): return {i for i, r in res[l][1].items() if r["pwml_written"]}
bw, ow, nw = written("base6"), written("off6"), written("on6")
print("written base1/base6/off6/on6:", len(written("base1")), len(bw), len(ow), len(nw))
print("base6 sha == base1 sha:", sum(1 for i in bw if i in written("base1") and sha("replay6","base6",i) == sha("replay","base1",i)), "of", len(bw))
print("off6 sha == base6 sha:", sum(1 for i in bw if i in ow and sha("replay6","base6",i) == sha("replay6","off6",i)), "of", len(bw), "; written sets equal:", bw == ow)
print("on6 gained/lost vs off6:", sorted(nw - ow), sorted(ow - nw))
REFS = ["protein-id", "protein-complex-id", "compound-location-id", "bound-visualization-id"]
def scan(p):
    t = etree.parse(str(p)); r = {}
    ids = [(c.findtext("id") or "").strip() for c in t.iter("compound-location")]
    refs = [(e.text or "").strip() for e in t.iter("compound-location-id")]
    r["dang_cl"] = sum(1 for x in refs if x not in set(ids)); r["cl_refs"] = len(refs)
    r["bound"] = sum(1 for _ in t.iter("bound-visualization"))
    r["sp"] = collections.Counter("nil" if e.get("nil") == "true" else (e.text or "").strip() for e in t.iter("spontaneous"))
    tt = list(t.iter("transport-transporter"))
    r["tt"] = len(tt)
    r["tt_prot"] = sum(1 for e in tt if e.find("protein-id") is not None and (e.findtext("protein-id") or "").strip())
    r["tt_cplx"] = sum(1 for e in tt if e.find("protein-complex-id") is not None and (e.findtext("protein-complex-id") or "").strip())
    # dangling protein-complex-id vs protein-complexes ids / protein-id vs proteins ids
    def tbl(tag, itemtag):
        return {(x.findtext("id") or "").strip() for sec in t.iter(tag) for x in sec.findall(itemtag)}
    cx = tbl("protein-complexes", "protein-complex"); pr = tbl("proteins", "protein")
    r["dang_cplx"] = sum(1 for e in t.iter("protein-complex-id") if (e.text or "").strip() and (e.text or "").strip() not in cx)
    r["dang_prot"] = sum(1 for e in t.iter("protein-id") if (e.text or "").strip() and (e.text or "").strip() not in pr)
    # duplicate ids per table (direct children of any plural container having id children)
    dup = collections.Counter()
    for sec in t.getroot().iter():
        kids = [k for k in sec if len(k) and k.find("id") is not None]
        if len(kids) > 1 and len({k.tag for k in kids}) == 1:
            ids2 = [(k.findtext("id") or "").strip() for k in kids]
            d = len(ids2) - len(set(ids2))
            if d: dup[sec.tag] += d
    r["dup"] = dup
    return r
for l in ("base6", "off6", "on6"):
    d, rs = res[l]; tot = collections.Counter(); sp = collections.Counter(); dup = collections.Counter(); dfiles = collections.Counter()
    bad_parse = []
    for i in sorted(written(l)):
        try: s = scan(E / d / l / "pwml" / f"{i}.pwml")
        except Exception as ex: bad_parse.append((i, str(ex)[:80])); continue
        for k in ("dang_cl","cl_refs","bound","tt","tt_prot","tt_cplx","dang_cplx","dang_prot"): tot[k] += s[k]
        for k in ("dang_cl","dang_cplx","dang_prot"): dfiles[k] += 1 if s[k] else 0
        sp.update(s["sp"]); dup.update(s["dup"])
    r = res[l][1]; w = written(l)
    print(f"== {l}: totals {dict(tot)} files_with_dangling {dict(dfiles)}\n   spontaneous {dict(sp)}\n   dup_ids {dict(dup)}\n   xml parse failures {bad_parse}\n   QA errors {sum(r[i].get('qa_errors') or 0 for i in w)} warnings {sum(r[i].get('qa_warnings') or 0 for i in w)} qa_ok_false {[i for i in w if r[i].get('qa_ok') is False]}")
