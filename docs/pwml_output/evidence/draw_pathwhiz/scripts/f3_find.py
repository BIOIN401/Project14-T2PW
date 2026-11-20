"""Which replay runs / showcases draw membrane templates 84, 95, 304 (after arm), and which fixtures name such places.
usage: f3_find.py AFTER_PWML_DIR"""
import collections
import glob
import os
import re
import sys

hits = collections.defaultdict(list)
for f in sorted(glob.glob(os.path.join(sys.argv[1], "*.pwml")), key=lambda p: int(os.path.basename(p)[:-5])):
    s = open(f, encoding="utf-8").read()
    i, j = s.find("<membrane-visualizations>"), s.find("</membrane-visualizations>")
    ts = collections.Counter(re.findall(r'<visualization-template-id type="integer">(\d+)</visualization-template-id>', s[i:j]))
    for t in ("84", "95", "304"):
        if ts.get(t):
            nloc = s.count("<compound-location>") + s.count("<protein-location>")
            hits[t].append((os.path.basename(f), ts[t], nloc, len(s)))
for t, rows in hits.items():
    print(t, rows)
print("templates absent:", [t for t in ("84", "95", "304") if t not in hits])
