"""ORCH-754b repair round 2 -- the committed rows the new ``ambiguous_protein_endpoint`` reason moves.

Configuration B of the GOLDEN sweep (DB down, no name index, strict), over every committed
``final_mapped.json`` with interactions; prints each row the tip withholds as ambiguous, with
the entity kinds its endpoint name registers.
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]


class _DownDb:
    last_error = "harvest_db_down"

    def available(self) -> bool:
        return False


def main() -> int:
    logging.disable(logging.CRITICAL)
    from t2pw.pwml.ir import build_pwml_ir
    from t2pw.pwml.prefreeze_resolution import run_prefreeze_resolution

    legs = sorted(p for p in subprocess.run(["git", "ls-files", "runs", "runs_verify"], cwd=REPO, capture_output=True,
                                            text=True, check=True).stdout.split("\n") if p.endswith("/final_mapped.json"))
    found = 0
    for leg in legs:
        payload = json.loads((REPO / leg).read_text(encoding="utf-8"))
        if not (payload.get("processes") or {}).get("interactions"):
            continue
        try:
            run_prefreeze_resolution(payload, strict_db=True, db_resolver=_DownDb(), name_index=None)
            ir, _ = build_pwml_ir(payload, db_resolver=_DownDb(), strict_db=True, name_index=None)
        except Exception:  # noqa: BLE001 -- the export stops on this leg; nothing to report
            continue
        for row in ir["processes"]["interactions"]:
            withheld = row.get("withheld") or {}
            if withheld.get("reason") == "ambiguous_protein_endpoint":
                found += 1
                print(f"{leg}  {row['name']!r}  {withheld['left']} -{withheld['raw_interaction_type']}-> "
                      f"{withheld['right']}  ambiguous: {withheld['ambiguous_endpoints']}")
    print(f"rows: {found}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
