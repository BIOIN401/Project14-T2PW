import re
from pathlib import Path

ev = Path("C:/t/ws-R5-tmp/ev")
rows, same, diff = [], 0, 0
SUMMARY = re.compile(r"(?:\d+ (?:passed|failed|skipped|errors?)(?:, )?)+ in ")
for t in sorted((ev / "t").glob("t-*.out")):
    name = t.stem[2:]
    b = ev / "b" / f"b-{name}.out"
    def parse(path):
        text = path.read_text(encoding="utf-8", errors="replace")
        fails = sorted(set(re.findall(r"^(?:FAILED|ERROR) (tests\S+)", text, re.M)))
        m = [x.group(0)[:-4] for x in SUMMARY.finditer(text)]
        return (m[-1] if m else "(no summary)"), fails
    ts, tf = parse(t)
    bs, bf = parse(b) if b.exists() else ("n/a", None)
    same_set = tf == bf
    same += same_set
    diff += not same_set
    rows.append(f"{name}\t{ts}\t{bs}\t{'same' if same_set else 'DIFFERENT'}")
    if not same_set and bf is not None:
        rows.extend(f"   tip-only: {x}" for x in sorted(set(tf) - set(bf)))
        rows.extend(f"   base-only: {x}" for x in sorted(set(bf) - set(tf)))
out = "\n".join(rows + [f"same={same} different={diff}"]) + "\n"
(ev / "compare.txt").write_text(out, encoding="utf-8")
print(out)
