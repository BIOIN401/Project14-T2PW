import sys, collections
sys.argv = ["x"]
exec(open("C:/t/fp/docs/pwml_output/evidence/final_payload_p5/measure/scripts/cat_p5.py", encoding="utf-8").read().split("# per-file:")[0].replace("print(", "(lambda *a: None)("))
def tx(r): return collections.Counter((b, m) for b, m in msgs(r))
a, b, c = S["P4 d3"], S["P5 (a) 03"], S["P5 (b) 04"]
for i in sorted(a):
    for n, s in (("a", b), ("b", c)):
        new = tx(s[i]) - tx(a[i])
        if new: print("NEWTEXT", n, i, dict(new))
print("--- a vs b differing files")
for i in sorted(a):
    if len(msgs(b[i])) != len(msgs(c[i])): print(i, len(msgs(a[i])), len(msgs(b[i])), len(msgs(c[i])), "tr", [(s[i].get("created") or {}).get("transports") for s in (a, b, c)], (a[i].get("expected") or {}).get("transports"))
print("--- changed vs P4 d3 (a)")
for i in sorted(a):
    if len(msgs(a[i])) != len(msgs(b[i])): print(i, len(msgs(a[i])), "->", len(msgs(b[i])), "->", len(msgs(c[i])), sorted((tx(a[i]) - tx(b[i])).items())[:3])
c2 = collections.Counter()
for n, s in (("d3", a), ("a", b), ("b", c)):
    k = collections.Counter(bb + ": " + m[:60] for r in s.values() for bb, m in msgs(r))
    print(n, k.most_common(12))
