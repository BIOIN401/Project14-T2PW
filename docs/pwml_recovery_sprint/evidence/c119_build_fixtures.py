"""C-119 — build the archived-leg fixtures the deterministic regressions replay.

READ-ONLY over the run archives. It copies four objects VERBATIM out of each
archived strict leg and writes them into one compact JSON file per leg under
``tests/fixtures/c119/``:

===========================  =====================================================
``contract_reports``         ``contract_reports.json`` -- exactly the
                             ``*_contract_report`` / ``*runtime_schema_report``
                             slice of the app's ``post_pipeline_artifacts``
``final_stage3_gate_report`` ``final_stage3_gate_report.json``
``canonical_export_payload`` ``final_mapped.json`` -- which IS the canonical export
                             payload: this script asserts that its
                             ``canonical_graph_sha256`` and
                             ``canonical_payload_sha256`` equal the two digests the
                             final gate report recorded, so the pairing is proved
                             rather than assumed
``quarantine_release``       ``quarantine_report.json``'s ``release`` block -- the
                             record the quarantine boundary FROZE, which is the one
                             object ``batch/driver.py::_frozen_release_record``
                             reads
===========================  =====================================================

NOTHING IS EDITED. The only transformation is JSON whitespace: the archives are
written with ``indent=2`` and these are written with the tightest separators, which
is byte-different and value-identical. No error, report, phase, count, payload,
entity or reaction is added, removed, reordered or rewritten.

``artifact_set_version: 1`` is added by the TEST when it assembles the artifacts
dict, not here, and it is not an invention: ``streamlit_app.py:4392`` wraps every
artifact set this build returns in ``stamp_artifact_set`` unconditionally, and every
contract report in these archives carries ``report_schema_version: 1``, which only
``stamp_report`` from the same build writes. The archives simply do not persist
top-level scalars, because ``contract_reports.json`` is a report-only slice.

Usage (from the worktree root, with the run archives available)::

    PYTHONPATH=src python docs/pwml_recovery_sprint/evidence/c119_build_fixtures.py \
        --runs <path to runs_verify>
"""

from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from t2pw.pipeline import canonical_hash as _canonical  # noqa: E402

#: ``(run, paper, fixture stem)`` for the nine deterministic regressions of C-119
#: section 4. Two papers appear twice, under different runs, and the stem carries
#: the run date so the two archives can never be confused for one another --
#: ORCH-728 section 6 records that the two ``PMC12452463`` archives DISAGREE, and a
#: fixture name that hid which one was loaded would hide exactly that.
LEGS = (
    ("2026-09-06_1425", "PMC7232280", "PMC7232280_strict_2026-09-06"),
    ("2026-09-06_1425", "PMC8510960", "PMC8510960_strict_2026-09-06"),
    ("2026-09-06_1425", "PMC12071552", "PMC12071552_strict_2026-09-06"),
    ("2026-09-06_1425", "PMC12376012", "PMC12376012_strict_2026-09-06"),
    ("2026-09-06_1425", "PMC11405693", "PMC11405693_strict_2026-09-06"),
    ("2026-09-02_2052", "PMC12180156", "PMC12180156_strict_2026-09-02"),
    ("2026-08-28_1816", "PMC12180156", "PMC12180156_strict_2026-08-28"),
    ("2026-09-02_2052", "PMC12452463", "PMC12452463_strict_2026-09-02"),
    ("2026-08-28_1816", "PMC12452463", "PMC12452463_strict_2026-08-28"),
)

OUT_DIR = ROOT / "tests" / "fixtures" / "c119"


def _load(path: Path):
    return json.loads(io.open(path, encoding="utf-8").read())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", required=True, help="path to the runs_verify directory")
    args = parser.parse_args()
    runs = Path(args.runs)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    total = 0
    for run, paper, stem in LEGS:
        leg = runs / run / "papers" / paper / "strict"
        gate = _load(leg / "final_stage3_gate_report.json")
        payload = _load(leg / "final_mapped.json")
        # PROVED, not assumed: final_mapped.json is the object the final
        # pre-export gate validated, so ``gate_verdict``'s payload binding holds
        # over the reassembled artifact set exactly as it did in the live run.
        assert _canonical.canonical_graph_sha256(payload) == gate.get("canonical_graph_sha256"), stem
        assert _canonical.canonical_payload_sha256(payload) == gate.get("canonical_payload_sha256"), stem
        bundle = {
            "archive": f"runs_verify/{run}/papers/{paper}/strict",
            "contract_reports": _load(leg / "contract_reports.json"),
            "final_stage3_gate_report": gate,
            "canonical_export_payload": payload,
            "quarantine_release": (_load(leg / "quarantine_report.json") or {}).get("release"),
        }
        text = json.dumps(bundle, separators=(",", ":"), ensure_ascii=False)
        out = OUT_DIR / f"{stem}.json"
        io.open(out, "w", encoding="utf-8", newline="\n").write(text)
        total += len(text.encode("utf-8"))
        print(f"{stem}: {len(text.encode('utf-8'))} bytes")
    print(f"TOTAL {total} bytes in {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
