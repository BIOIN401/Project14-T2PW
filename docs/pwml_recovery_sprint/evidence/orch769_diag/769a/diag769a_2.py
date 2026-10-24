"""ORCH-769a read-only diagnosis, part 2: connectivity of CF-B1, actor-evidence findings,
and whether the inferred actor's cited span occurs in the paper. Writes nothing."""
import json
import re
from copy import deepcopy

from t2pw.pipeline.process_normalizer import run_strict_post_normalization_gates, GateValidationError
from t2pw.bench import semantic as _s
from t2pw.bench.semantic_production import _check_actor_evidence

ROOT = "C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW"
LEG = ROOT + "/.claude/worktrees/orch767/runs_papers/ORCH-767/legs/enterobactin-strict/2026-10-23_2224/papers/PMC12096016"
WRAP = "enterobactin synthase complex"

with open(LEG + "/strict/final_mapped.json", encoding="utf-8") as f:
    fm = json.load(f)
with open(LEG + "/01_source_text.txt", encoding="utf-8") as f:
    src = f.read()


def norm(t):
    return re.sub(r"[^a-z0-9]+", " ", t.lower().replace("\u2010", "-")).strip()


res = _check_actor_evidence(_s._processes(fm), _s._entities(fm))
print("[actor-check] ok =", res.ok, "|", res.summary)
for f_ in res.findings:
    print("   ", f_["pointer"], "|", f_["entity"], "|", f_["cited_span"][:120])

nsrc = norm(src)
for r in fm["processes"]["reactions"]:
    for a in r.get("enzymes") or []:
        ev = a.get("evidence") or ""
        print(f"[span-in-paper] {r.get('name')!r} / {a.get('entity')!r} ({a.get('provenance')}): {norm(ev) in nsrc}")

b = deepcopy(fm)
for row in b["processes"]["reactions"]:
    for key in ("enzymes", "modifiers"):
        row[key] = [a for a in row.get(key) or [] if a.get("entity") != WRAP]
b["entities"]["protein_complexes"] = [c for c in b["entities"]["protein_complexes"] if c.get("name") != WRAP]
try:
    d = run_strict_post_normalization_gates(deepcopy(b), enforce_all_proteins_connected=True)
except GateValidationError as exc:
    d = exc.details or {}
print("[cf-b1] connectivity:", json.dumps(d.get("connectivity"), ensure_ascii=False))
