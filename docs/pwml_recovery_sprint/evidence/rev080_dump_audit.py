"""Dump the FULL 5-tuple of _audit_entities for every committed final_mapped.json.

Run once in the base worktree and once in the tip worktree; the two JSON files are
then compared field by field, with NO normalisation of any kind.
"""
import json, sys
from pathlib import Path

ROOT = Path(sys.argv[1]).resolve()
OUT = Path(sys.argv[2])
sys.path.insert(0, str(ROOT / "src"))
from t2pw.bench import semantic as bench_semantic  # noqa: E402
from t2pw.bench import semantic_production as sp  # noqa: E402

RUNS = ROOT / "runs_verify"
rows = {}
for run in sorted(RUNS.iterdir()):
    papers = run / "papers"
    if not papers.is_dir():
        continue
    for paper in sorted(papers.iterdir()):
        if not paper.is_dir():
            continue
        for leg in sorted(paper.iterdir()):
            fm = leg / "final_mapped.json"
            if not fm.is_file():
                continue
            name = f"{run.name}/papers/{paper.name}/{leg.name}"
            payload = json.loads(fm.read_text(encoding="utf-8"))
            entities = bench_semantic._entities(payload)
            id_check, ph_check, census, forged, backed = sp._audit_entities(entities)
            rows[name] = {
                "id_check": {
                    "name": id_check.name, "ok": id_check.ok,
                    "summary": id_check.summary,
                    "applicable": getattr(id_check, "applicable", None),
                    "inapplicable_reason": getattr(id_check, "inapplicable_reason", None),
                    "findings": id_check.findings,
                },
                "placeholder_check": {
                    "name": ph_check.name, "ok": ph_check.ok,
                    "summary": ph_check.summary,
                    "applicable": getattr(ph_check, "applicable", None),
                    "inapplicable_reason": getattr(ph_check, "inapplicable_reason", None),
                    "findings": ph_check.findings,
                },
                "census": census, "forged": forged, "backed": backed,
            }
OUT.write_text(json.dumps(rows, indent=1, sort_keys=True, default=str), encoding="utf-8")
print("legs with final_mapped.json:", len(rows))
print("wrote", OUT)
