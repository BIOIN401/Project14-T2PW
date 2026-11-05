"""Tip-vs-base comparison of the R2 chunk logs (scratch)."""
import hashlib
import re
from pathlib import Path

EV = Path("C:/t/wsR2tmp/ev")
IDS = ["g1", "g2", "g3", "g4", "2a", "2b", "2b-c055", "2c", "2c-c126", "2d", "2d-stage3",
       "2d-c119g9", "2e", "2e-759b", "2e-hook", "2e-759afold", "2f", "2f-driver", "2f-gate",
       "2g", "2g-c082pp", "2g-c056d", "2g-c119sup", "2g-759d", "2g-revview", "web", "new"]


def read(prefix, cid):
    p = EV / f"{prefix}-{cid}.log"
    if not p.exists():
        return None
    text = p.read_text(encoding="utf-8", errors="replace")
    summary = next((l.strip() for l in text.splitlines()
                    if re.search(r"\d+ (passed|failed)", l) and " in " in l), "?")
    failed = sorted({m.group(1) for m in re.finditer(r"^FAILED (\S+)", text, re.M)}
                    | {m.group(1) for m in re.finditer(r"^ERROR (\S+)", text, re.M)})
    head = re.search(r"wrapper vs HEAD\s*:\s*(.*)", text)
    surv = re.search(r"FINAL SURVIVING COUNT\s*:\s*(\d+)", text)
    return summary.split(" in ")[0], failed, head.group(1).strip() if head else "?", surv.group(1) if surv else "?"


rows = ["| chunk | tip (0941cd5d) | base (4a41b91c) | failing-node set | survivors tip/base |",
        "|---|---|---|---|---|"]
for cid in IDS:
    t, b = read("t", cid), read("b", cid)
    tf = t[1] if t else []
    bf = b[1] if b else None
    if bf is None:
        same = "n/a (new file, absent at base)"
    elif tf == bf:
        same = ("identical, none" if not tf else
                f"identical ({len(tf)}; sha256 {hashlib.sha256(chr(10).join(tf).encode()).hexdigest()[:12]})")
    else:
        same = f"DIFFERENT tip={tf} base={bf}"
    rows.append(f"| {cid} | {t[0] if t else '-'} | {b[0] if b else '-'} | {same} | "
                f"{t[3] if t else '-'}/{b[3] if b else '-'} |")
print("\n".join(rows))
print()
for cid in IDS:
    t = read("t", cid)
    if t and t[1]:
        print(cid, "FAILED:", *t[1], sep="\n  ")
for pfx in ("t", "b"):
    heads = {read(pfx, c)[2] for c in IDS if read(pfx, c)}
    print(pfx, "wrapper vs HEAD:", heads)
