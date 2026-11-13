import json
want = {l.strip().rsplit("/", 1)[1] for l in open("../../import/list_on.txt")}
L = [l.strip() for l in open("list_on.txt")]
R = [l for l in open("on/results.jsonl", encoding="utf-8")]
assert len(L) == len(R)
keep = [(p, r) for p, r in zip(L, R) if p.rsplit("/", 1)[1] in want]
print(len(keep))
open("list_on59.txt", "w").write("\n".join(p for p, _ in keep) + "\n")
open("on59_results.jsonl", "w", encoding="utf-8").write("".join(r for _, r in keep))
