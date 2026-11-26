"""List edges of a PWML whose bbox touches a given rectangle, and which visualizations reference each edge.
usage: probe_edges.py <pwml> x0 y0 x1 y1"""
import re, sys
s = open(sys.argv[1], encoding="utf-8").read()
x0, y0, x1, y1 = map(float, sys.argv[2:6])
refs = {}
for m in re.finditer(r"<([a-z_-]+)>\s*<id[^>]*>(\d+)</id>((?:(?!</\1>).)*?)<edge-id[^>]*>(\d+)</edge-id>", s, re.S):
    refs.setdefault(m.group(4), []).append(m.group(1))
for m in re.finditer(r"<edge>(.*?)</edge>", s, re.S):
    t = m.group(1)
    g = lambda k: (re.search("<" + k + "[^>]*>([^<]*)<", t) or [None, ""])[1]
    nums = [float(v) for v in re.findall(r"-?\d+\.?\d*", g("path"))]
    xs, ys = nums[0::2], nums[1::2]
    if xs and min(xs) <= x1 and max(xs) >= x0 and min(ys) <= y1 and max(ys) >= y0:
        print(g("id"), "tpl", g("visualization-template-id"), "z", g("zindex"), "hidden", g("hidden"),
              "path", g("path")[:90], "opts", g("options")[:60], "refs", refs.get(g("id"), []))
