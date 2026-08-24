import json, sys
from pathlib import Path

base = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
tip = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
assert set(base) == set(tip), (set(base) ^ set(tip))
print("legs compared:", len(base))

same_arms = 0
id_moved, id_same = [], 0
gained, lost, changed_prose = [], [], []
for name in sorted(base):
    b, t = base[name], tip[name]
    for arm in ("placeholder_check", "census", "forged", "backed"):
        if b[arm] != t[arm]:
            print("!! ARM CHANGED", arm, name)
            print("   base:", json.dumps(b[arm], sort_keys=True)[:600])
            print("   tip :", json.dumps(t[arm], sort_keys=True)[:600])
    if all(b[arm] == t[arm] for arm in ("placeholder_check", "census", "forged", "backed")):
        same_arms += 1
    bi, ti = b["id_check"], t["id_check"]
    if bi == ti:
        id_same += 1
        continue
    id_moved.append(name)
    bk = [f.get("kind") for f in bi["findings"]]
    tk = [f.get("kind") for f in ti["findings"]]
    print("--- id_check moved:", name)
    print("    ok  base->tip :", bi["ok"], "->", ti["ok"])
    print("    kinds base    :", bk)
    print("    kinds tip     :", tk)
    print("    summary base  :", bi["summary"])
    print("    summary tip   :", ti["summary"])
    # exact per-finding set comparison on identity keys
    def key(f):
        return (f.get("kind"), f.get("namespace"), f.get("identifier"),
                tuple(f.get("entities") or ()), f.get("pointer"), f.get("name"),
                f.get("rule"), json.dumps(f.get("identifiers"), sort_keys=True))
    bset = {key(f): f for f in bi["findings"]}
    tset = {key(f): f for f in ti["findings"]}
    for k in tset:
        if k not in bset:
            gained.append((name, tset[k]))
            print("    GAINED FINDING:", json.dumps(tset[k], sort_keys=True))
    for k in bset:
        if k not in tset:
            lost.append((name, bset[k]))
            print("    LOST FINDING  :", json.dumps(bset[k], sort_keys=True))
    for k in set(bset) & set(tset):
        if bset[k] != tset[k]:
            diff = {f: (bset[k].get(f), tset[k].get(f))
                    for f in set(bset[k]) | set(tset[k]) if bset[k].get(f) != tset[k].get(f)}
            changed_prose.append((name, diff))
            print("    SAME-IDENTITY FINDING, FIELDS DIFFER:", json.dumps(diff, sort_keys=True))

print()
print("legs where placeholder/census/forged/backed IDENTICAL:", same_arms, "/", len(base))
print("legs where id_check IDENTICAL                       :", id_same)
print("legs where id_check MOVED                           :", len(id_moved), id_moved)
print("findings GAINED anywhere                            :", len(gained))
print("findings LOST anywhere                              :", len(lost))
print("same-identity findings with a field difference      :", len(changed_prose))
for n, d in changed_prose:
    print("   ", n, "fields:", sorted(d))
