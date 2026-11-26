"""PathWhiz import error counts for one file between two checkpoint runs (SQL values cut).
usage: pw_err2.py OLD_results.jsonl NEW_results.jsonl NAME"""
import json
import re
import sys


def errs(p, name):
    for line in open(p, encoding="utf-8"):
        r = json.loads(line)
        if name in r["file"]:
            out = {}
            for k, v in (r.get("errors") or {}).items():
                for m in v:
                    key = (re.sub(r"_\d+$", "", k), re.sub(r"VALUES \(.*", "VALUES (...)", m)[:140])
                    out[key] = out.get(key, 0) + 1
            return out
    return {}


a, b = errs(sys.argv[1], sys.argv[3]), errs(sys.argv[2], sys.argv[3])
for k in sorted(set(a) | set(b)):
    if a.get(k) != b.get(k):
        print(a.get(k), "->", b.get(k), k)
