"""ORCH-780: per leg of a replayed corpus out-dir, the reactions in the frozen payload by the support gate's own predicate.

    python support_census.py <corpus.json> <out-dir>

For every leg that wrote a PWML: reactions in canonical.json, how many the gate's row predicate
(``reaction_support_class``) calls target_paper / external_rag / None, and whether the lineage carrier is
active (``lineage_carrier_active``). ``EXPORTED-UNSUPPORTED`` marks a PWML-writing leg whose frozen payload
holds a reaction with no support while the carrier is active. Read-only; imports t2pw from PYTHONPATH.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

from t2pw.pipeline.reaction_support import lineage_carrier_active, reaction_support_class


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    doc = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    out = Path(sys.argv[2])
    bad = 0
    for leg in doc["legs"]:
        r = leg.get("replay") or {}
        if not r.get("pwml_written"):
            continue
        canonical = json.loads((out / re.sub(r"[^A-Za-z0-9._-]+", "_", leg["id"]) / "canonical.json").read_text(encoding="utf-8"))
        rows = [x for x in (canonical.get("processes") or {}).get("reactions") or [] if isinstance(x, dict)]
        carrier = lineage_carrier_active(canonical)
        classes = [reaction_support_class(x) for x in rows]
        none = [x.get("name") for x, c in zip(rows, classes) if c is None]
        flag = "EXPORTED-UNSUPPORTED" if carrier and none else ""
        bad += bool(flag)
        print(f"{leg['id']:58s} carrier={'on ' if carrier else 'off'} reactions={len(rows):3d} "
              f"paper={classes.count('target_paper'):3d} rag={classes.count('external_rag'):3d} none={len(none):3d} {flag}")
        if flag:
            for name in none:
                print(f"    unsupported exported: {name}")
    print(f"legs writing a PWML with an unsupported reaction while the carrier is active: {bad}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
