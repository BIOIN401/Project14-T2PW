"""Compare two proof_generate logs row by row: PWML sha of this branch vs the base tree (main + metrics only).
usage: proofcmp.py <base log> <branch log>"""
import json
import sys


def rows(p):
    out = {}
    for line in open(p, encoding="utf-8", errors="replace"):
        if line.startswith("{") and '"idx"' in line:
            r = json.loads(line)
            out[r["idx"]] = r
    return out


a, b = rows(sys.argv[1]), rows(sys.argv[2])
same = [i for i in a if i in b and a[i].get("got_sha") and a[i].get("got_sha") == b[i].get("got_sha")]
print("BRANCH_VS_BASE identical %d/%d (base rows %d, branch rows %d)" % (len(same), len(set(a) | set(b)), len(a), len(b)))
print("replay-identical: base %d, branch %d" % (sum(r["identical"] for r in a.values()), sum(r["identical"] for r in b.values())))
print("differ:", sorted(set(a) - set(same)))
