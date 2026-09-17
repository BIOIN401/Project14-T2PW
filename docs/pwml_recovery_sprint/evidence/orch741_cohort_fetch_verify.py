"""ORCH-741: verify every cohort paper fetches through the project's OWN acquisition path.

Same code path the batch run uses (``t2pw.rag.acquire.fetch_full_text``), so no leg
can fail for an acquisition reason that selection could have caught. Read-only:
writes only its ``--out`` JSON, touches no run tree, no cache, no production file.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from t2pw.rag.acquire import CandidatePaper, fetch_full_text  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("pmcids", nargs="+")
    a = ap.parse_args()

    rows = []
    failures = 0
    for pmcid in a.pmcids:
        try:
            candidate = CandidatePaper(id=pmcid, source="europepmc")
            text = fetch_full_text(candidate, use_cache=False) or ""
            err = ""
        except Exception as exc:  # noqa: BLE001 - the failure IS the result
            text, err = "", f"{type(exc).__name__}: {exc}"
        ok = bool(text) and len(text) >= 18000 and not err
        if not ok:
            failures += 1
        rows.append({"pmcid": pmcid, "chars": len(text), "error": err, "ok": ok})
        print(f"  {pmcid:<12} chars={len(text):>7} {'OK' if ok else 'FAIL ' + err}")

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(
        json.dumps({"tool": "orch741_cohort_fetch_verify", "rows": rows,
                    "failures": failures}, indent=1),
        encoding="utf-8",
    )
    print(f"verified {len(rows)} paper(s) through fetch_full_text | failures: {failures}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
