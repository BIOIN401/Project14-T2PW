"""Reviewer scan of committed R3d evidence: surviving counts, secret shapes, smoke numbers, fixture env flags."""
import gzip, json, re, sys
from pathlib import Path
ROOT = Path("C:/t/ws-R3d")
EV = ROOT / "docs/webserver/evidence/R3d"
counts = {}
bad = []
for p in EV.rglob("*.json"):
    try:
        d = json.loads(p.read_text(encoding="utf-8", errors="replace"))
    except Exception:
        continue
    if isinstance(d, dict) and "final_surviving_count" in d:
        c = d.get("final_surviving_count")
        counts[c] = counts.get(c, 0) + 1
        if c != 0:
            bad.append((str(p), c))
print("final_surviving_count histogram over wrapper reports:", counts, "nonzero:", bad[:5])
SHAPES = {
    "openrouter": re.compile(r"sk-or-[A-Za-z0-9-]{8,}"),
    "bearer": re.compile(r"(?i)bearer\s+[A-Za-z0-9._-]{12,}"),
    "db_env": re.compile(r"PATHBANK_DB_(HOST|USER|PASSWORD)\s*="),
    "password_field": re.compile(r"(?i)\"(db_)?password\"\s*:\s*\"[^\"<]"),
    "api_key_param": re.compile(r"(?i)\b(api_?key|access_token)=(?!<redacted>)[A-Za-z0-9_-]{8,}"),
}
hits = {}
scanned = 0
for p in list(EV.rglob("*")) + [ROOT / "docs/webserver/results/R3d.md", ROOT / "tests/test_service_extraction_stage2.py"]:
    if p.is_file():
        scanned += 1
        t = p.read_text(encoding="utf-8", errors="replace")
        for name, rx in SHAPES.items():
            found = rx.findall(t)
            if found:
                m = rx.search(t)
                hits.setdefault(name, []).append((str(p.relative_to(ROOT)), len(found), m.group(0)[:12] + "..."))
print("scanned files:", scanned)
print("secret-shape hits:", hits)
for name in ("r2-smoke.log", "r-eq.log", "r-g1.log", "r-g2.log", "r-g3.log", "r-g4.log", "r-new.log", "r-solo-c060a.log", "r-solo-s2orch.log", "r-2b.log"):
    t = (EV / "post-rulings" / name).read_text(encoding="utf-8", errors="replace")
    line = [l for l in t.splitlines() if re.search(r"\d+ (passed|failed)", l)]
    surv = re.search(r"FINAL SURVIVING COUNT\s*:\s*(\d+)", t)
    head = re.search(r"wrapper vs HEAD\s*:.*", t)
    print(name, "|", line[-1][:70] if line else "?", "| surv", surv.group(1) if surv else "?", "|", head.group(0)[-60:] if head else "?")
fx = json.loads(gzip.decompress((ROOT / "tests/fixtures/equivalence/sour_taste_v2_strict.json.gz").read_bytes()).decode("utf-8"))
print("fixture env RAG flags:", {k: v for k, v in fx["env"].items() if "RAG" in k.upper()})
print("fixture llm/http/db counts:", {k: len(v) for k, v in fx["entries"].items()})
