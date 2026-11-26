"""Orphan edges (no <edge-id> refers to them) that are not template 83, per PWML. usage: orphans_tpl.py <pwml>..."""
import re, sys
for p in sys.argv[1:]:
    s = open(p, encoding="utf-8").read()
    refd = set(re.findall(r"<edge-id[^>]*>(\d+)</edge-id>", s))
    out = []
    for m in re.finditer(r"<edge>\s*<id[^>]*>(\d+)</id>(.*?)</edge>", s, re.S):
        tpl = (re.search(r"<visualization-template-id[^>]*>(\d+)<", m.group(2)) or [None, "?"])[1]
        hidden = "<hidden>true" in m.group(2)
        if m.group(1) not in refd and tpl != "83":
            out.append(f"#{m.group(1)} tpl {tpl}{' hidden' if hidden else ''}")
    if out:
        print(p.replace("\\", "/").split("/")[-1], out)
