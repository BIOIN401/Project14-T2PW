"""Count edges that no <edge-id> refers to, per PWML. usage: orphans.py <pwml>..."""
import re, sys
for p in sys.argv[1:]:
    s = open(p, encoding="utf-8").read()
    ids = re.findall(r"<edge>\s*<id[^>]*>(\d+)</id>", s)
    refd = set(re.findall(r"<edge-id[^>]*>(\d+)</edge-id>", s))
    orph = [i for i in ids if i not in refd]
    print(f"{p.split('/')[-1]}: edges {len(ids)} orphans {len(orph)} {orph[:8]}")
