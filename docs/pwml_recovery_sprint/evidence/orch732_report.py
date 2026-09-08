"""ORCH-732 EXPECTED-WORKING-PWML-SMOKE -- morning report.

Reads the run manifest and prints one block per leg plus the yield summary.
Read-only: opens nothing but the run directory. Not pipeline code.

Run it as:
    PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe \
        docs/pwml_recovery_sprint/evidence/orch732_report.py runs_smoke/2026-09-07_2323
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ORGANISM = {
    "PMC9544450": "Escherichia coli",
    "PMC4725005": "Escherichia coli",
    "PMC11487621": "Bacillus subtilis",
    "PMC12051542": "Aquifex aeolicus",
    "PMC10031235": "Homo sapiens",
    "PMC7615680": "Homo sapiens",
}
COHORT = list(ORGANISM)


def _safe(text: object) -> str:
    return str(text if text is not None else "")


def main(argv: list[str]) -> int:
    run_dir = Path(argv[1] if len(argv) > 1 else "runs_smoke/2026-09-07_2323")
    manifest = run_dir / "manifest.jsonl"
    if not manifest.exists():
        print(f"no manifest at {manifest} -- the run has not written a leg yet")
        return 1

    rows: dict[str, dict] = {}
    for line in manifest.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        rows[d.get("paper_id", "?")] = d

    produced: list[tuple[str, str, int]] = []

    for pid in COHORT:
        d = rows.get(pid)
        print("=" * 104)
        if d is None:
            print(f"{pid:<13} {ORGANISM[pid]:<22} NOT RUN (no manifest row)")
            continue

        counts = d.get("counts") or {}
        rel = d.get("release_status") or {}
        pwml_name = d.get("pwml_artifact") or ""
        pwml_bytes = counts.get("pwml_bytes")
        inner = counts.get("pwml") or {}

        print(f"{pid:<13} {ORGANISM[pid]:<22} topic: {_safe(d.get('topic'))}")
        print(f"  pipeline    : status={_safe(d.get('status'))}  stage={_safe(d.get('stage'))}  "
              f"{round(float(d.get('seconds') or 0))}s")
        print(f"  canonical   : reactions={counts.get('reactions')}  transports={counts.get('transports')}  "
              f"proteins={counts.get('proteins')}  compounds={counts.get('compounds')}")

        if pwml_name:
            path = run_dir / _safe(d.get("dir")) / pwml_name
            print(f"  PWML        : YES  {pwml_bytes} bytes")
            print(f"  path        : {path}")
            print(f"  pwml graph  : compounds={inner.get('compounds')} proteins={inner.get('proteins')} "
                  f"reactions={inner.get('reactions')} edges={inner.get('edges')}")
            produced.append((pid, str(path), int(pwml_bytes or 0)))
        else:
            print("  PWML        : NO")

        print(f"  release     : {_safe(rel.get('status'))}  strict_gates_passed={rel.get('strict_gates_passed')}  "
              f"semantic={_safe(rel.get('semantic_evaluation'))}")
        print(f"  completeness: {rel.get('completeness')}")

        reasons = rel.get("reasons") or []
        if reasons:
            print(f"  reasons     : {'; '.join(_safe(r) for r in reasons)}")
        missing = rel.get("missing_anchors") or []
        if missing:
            print(f"  missing     : {', '.join(_safe(m) for m in missing)}")

        msg = _safe(d.get("message"))
        if msg:
            print(f"  one-liner   : {msg[:300]}")

    print("=" * 104)
    ran = len([p for p in COHORT if p in rows])
    print(f"YIELD: {len(produced)} PWML from {ran} leg(s) run, cohort of {len(COHORT)}")
    for pid, path, size in produced:
        print(f"  {pid:<13} {size:>8} bytes  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
