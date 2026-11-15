"""READ-ONLY look at the PathBank production dump (smpdb db_dumps/smpdb_remote_dump.sql, lines 83-84 biological_states, 14454 species):
which species use PathWhiz subcellular_locations rows 10 Cell Membrane, 12/14 Mitochondrial Inner/Outer Membrane, 31 Periplasmic Space,
32 Inner Membrane, 33 Outer Membrane, 40 Periplasm. Output e_pathwhiz_row_usage.txt (counts by species.classification / gram_stain)."""
import collections
import re

D = "C:/Users/Angad/Desktop/SummerBIOIN/smpdb/db_dumps/smpdb_remote_dump.sql"
bs, sp = [], {}
Q = r"'(?:[^'\\]|\\.|'')*'"
SPROW = re.compile(r"\((\d+)," + "(" + Q + r"),(NULL|" + Q + r"),(?:" + Q + "),(?:" + Q + r"),(\d+),(NULL|" + Q + r"),(NULL|" + Q + r"),(NULL|" + Q + r")\)")
with open(D, "r", encoding="utf-8", errors="replace") as f:
    for n, line in enumerate(f, 1):
        if n in (83, 84):
            for m in re.finditer(r"\((\d+),(NULL|\d+),(NULL|\d+),'[^']*','[^']*',(NULL|\d+),(NULL|\d+),(\d+),(?:'[^']*'|NULL)\)", line):
                bs.append((int(m[1]), m[2], m[3], m[4], m[5], m[6]))
        if n == 14454:
            for m in SPROW.finditer(line):
                sp[int(m[1])] = (m[2].strip("'"), m[3].strip("'"), m[5].strip("'"), m[7].strip("'"))
            break
out = [f"biological_states parsed {len(bs)}; species parsed {len(sp)}"]
for loc, name in (("10", "Cell Membrane"), ("12", "Mitochondrial Inner Membrane"), ("14", "Mitochondrial Outer Membrane"), ("31", "Periplasmic Space"),
                  ("32", "Inner Membrane"), ("33", "Outer Membrane"), ("40", "Periplasm")):
    c, ex = collections.Counter(), collections.defaultdict(set)
    for (_i, _t, l, s, _ct, _u) in bs:
        if l == loc:
            x = sp.get(int(s), ("?", "?", "?", "?")) if s != "NULL" else ("NULL species", "", "", "")
            c[(x[2] or "?", x[3])] += 1
            ex[(x[2] or "?", x[3])].add(x[0])
    tot = sum(c.values())
    pro = sum(v for k, v in c.items() if k[0] == "Prokaryote")
    out.append(f"row {loc} {name!r}: {tot} states, {pro} with a Prokaryote species ({100 * pro // max(tot, 1)} %); by (classification, gram_stain): {dict(c)}")
    for k in c:
        if k[0] != "Prokaryote":
            out.append(f"    non-prokaryote species examples {k}: {sorted(ex[k])[:6]}")
open(__file__.replace(".py", ".txt"), "w", encoding="utf-8").write("\n".join(out) + "\n")
print("\n".join(out))
