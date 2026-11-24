"""Copy before/after PathWhiz screenshots into compare/ and write a side-by-side HTML page.
usage: compare_page.py   (reads pw/base and pw/after results.jsonl; writes compare/index.html)"""
import html
import json
import os
import shutil

W = "C:/t/lane-pwdraw-work"
C = f"{W}/compare5"
SUF = {"base": "2", "after": "5"}
os.makedirs(C, exist_ok=True)
rows = {}
for arm in ("base", "after"):
    for line in open(f"{W}/pw/{arm}{SUF[arm]}/results.jsonl", encoding="utf-8"):
        r = json.loads(line)
        name = os.path.basename(r["file"])[:-5].split("_", 1)[1]
        rows.setdefault(name, {})[arm] = r
parts = ["<!doctype html><meta charset=utf-8><title>PathWhiz before/after</title><style>body{font-family:sans-serif;margin:16px}"
         "table{border-collapse:collapse}td{vertical-align:top;padding:4px;border-top:1px solid #ccc}img{width:100%;max-width:900px}"
         "h2{margin:24px 0 4px}</style><h1>PathWhiz import A/B: base (main fd814b8e) vs draw/pathwhiz 23ffd20c (review round 1)</h1>"]
for name in sorted(rows):
    if not name.startswith("show_"):
        continue
    slug = name[5:]
    parts.append(f"<h2>{html.escape(slug)}</h2><table><tr><th>base</th><th>after</th></tr>")
    for shot in ("", "_zoom"):
        cells = []
        for arm in ("base", "after"):
            r = rows[name].get(arm) or {}
            src = r.get("screenshot")
            cell = "(no screenshot)"
            if src:
                p = os.path.join(W, "pw", arm + SUF[arm], src)
                if shot:
                    p = p.replace(".png", "_zoom.png")
                if os.path.exists(p):
                    dst = f"{slug}_{arm}_pathwhiz_{'zoom' if shot else 'full'}.png"
                    shutil.copyfile(p, os.path.join(C, dst))
                    cell = f"<img src='{dst}'>"
            created = r.get("created") or {}
            cell += "<br><small>%s; errors %d</small>" % (html.escape(", ".join(f"{k} {v}" for k, v in sorted(created.items()) if v)), r.get("error_count", 0))
            cells.append(f"<td>{cell}</td>")
        parts.append("<tr>" + "".join(cells) + "</tr>")
    parts.append("</table>")
open(f"{C}/index.html", "w", encoding="utf-8").write("\n".join(parts))
print("written", len([n for n in rows if n.startswith("show_")]))
