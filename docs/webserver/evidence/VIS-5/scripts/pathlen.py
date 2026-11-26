"""Longest edge / membrane path strings in a PWML set (PathWhiz edges.path is varchar(255))."""
import sys, re, glob
worst = []
for p in sys.argv[1:]:
    s = open(p, encoding="utf-8").read()
    for tag in ("edge", "membrane-visualization"):
        for m in re.finditer(r"<%s>.*?<path>([^<]*)</path>" % tag, s, re.S):
            worst.append((len(m.group(1)), tag, p.rsplit("/", 1)[-1]))
worst.sort(reverse=True)
print("over 255:", sum(1 for n, t, _ in worst if n > 255 and t == "edge"), "edges;", sum(1 for n, t, _ in worst if n > 255 and t != "edge"), "membranes")
print(worst[:5])
