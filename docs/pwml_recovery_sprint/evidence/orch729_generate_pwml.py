"""ORCH-729: generate the recovered PWMLs deterministically from archived payloads.

NO PIPELINE LEG IS RE-RUN AND NO LLM DRAW IS TAKEN. The canonical payloads already
exist on disk; this script pushes them through the SAME GATED PRODUCTION EXPORT PATH
the app uses -- ``streamlit_app.run_pwml_export`` -- which runs ``validate_pre_export``
(and therefore **F-179**) at ``streamlit_app.py:4788`` BEFORE
``DeterministicPwmlBuilder`` at ``:5003``.

WHY NOT THE CLI. ``writer.run_pwml_pipeline_export`` would be simpler and is what
``scripts/run_pwml.py`` drives, but ``F-183`` established that path is **not protected
by the F-179 rule**. Using it to produce a deliverable would bypass the anti-invention
floor, which is the one thing this whole wave exists to preserve. The gated function is
used instead, and a leg that its gates refuse produces NO FILE here, exactly as in
production.

WHAT THIS IS EVIDENCE OF, and what it is not
--------------------------------------------
It is evidence that the payloads C-119 unblocked SERIALIZE -- that real PWML bytes come
out of the gated path, not that the biology in them is right. **No human has reviewed
their content.** ``PILOT-MANUAL-REVIEW.md`` is the instrument for that and this script is
not a substitute for it.

Output goes to a NEW directory and overwrites nothing: existing run artifacts under
``runs_verify/`` are protected state and are read, never written.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

#: (paper, mode, archived run, why this leg is here)
LEGS: Tuple[Tuple[str, str, str, str], ...] = (
    ("PMC7232280", "strict", "runs_verify/2026-09-06_1425",
     "RECOVERED by C-119 -- Moco biosynthesis, Neurospora crassa, 5 reactions"),
    ("PMC8510960", "strict", "runs_verify/2026-09-06_1425",
     "RECOVERED by C-119 -- MIA biosynthesis, Catharanthus roseus, 5 reactions"),
    ("PMC12071552", "strict", "runs_verify/2026-09-06_1425",
     "CONTROL -- already shipped a PWML in the pilot; must still serialize"),
    ("PMC12376012", "strict", "runs_verify/2026-09-06_1425",
     "NEGATIVE CONTROL -- must still be REFUSED by the live pre-export gate"),
    ("PMC12180156", "strict", "runs_verify/2026-09-02_2052",
     "NEGATIVE CONTROL -- must still be REFUSED by F-179"),
)


def load(path: Path) -> Optional[Any]:
    if not path.is_file():
        return None
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def pathway_metadata(leg_dir: Path, run_dir: Path, paper: str) -> Dict[str, str]:
    """Name / description / subject, taken from the run's own record.

    Nothing is invented: the plan record is what the batch asked for, and it is the
    same string the app's ``pwml_name`` widget carried for that leg. When the plan is
    silent the generic app defaults are used -- the SAME defaults
    ``streamlit_app.py:2660-2662`` applies -- so the export is not made to look more
    specific than the run was.
    """

    requested_pathway = ""
    requested_organism = ""
    plan = load(run_dir / "plan.json") or {}
    for entry in plan.get("papers") or []:
        if isinstance(entry, dict) and entry.get("paper_id") == paper:
            requested_pathway = str(entry.get("requested_pathway") or "")
            requested_organism = str(entry.get("requested_organism") or "")
            break
    name = requested_pathway or "Generated Pathway"
    description = " ".join(
        part for part in (requested_pathway, f"({requested_organism})" if requested_organism else "")
        if part
    ).strip()
    return {
        "pathway_name": name,
        "pathway_description": description,
        "pathway_subject": "Metabolic",
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("repo_root", nargs="?", default=".")
    parser.add_argument("--out", dest="out_dir", default="outputs/c119_recovered")
    args = parser.parse_args(list(argv) if argv is not None else None)

    root = Path(args.repo_root).resolve()
    sys.path.insert(0, str(root / "src"))
    os.environ.setdefault("STREAMLIT_SERVER_HEADLESS", "true")

    from t2pw.app.streamlit_app import (  # noqa: E402
        PROJECT_ROOT,
        _pwml_reference_path,
        run_pwml_export,
    )

    out_dir = (root / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 96)
    print("ORCH-729 -- deterministic PWML generation from ARCHIVED canonical payloads")
    print("=" * 96)
    print(f"repo root : {root}")
    print(f"out dir   : {out_dir}")
    print("export    : streamlit_app.run_pwml_export -- the GATED path "
          "(validate_pre_export -> F-179 -> DeterministicPwmlBuilder)")
    print("NO pipeline leg is re-run. NO LLM draw is taken. Run artifacts are read-only.")
    print()

    results: List[Dict[str, Any]] = []
    for paper, mode, run, note in LEGS:
        run_dir = root / run
        leg_dir = run_dir / "papers" / paper / mode
        payload = load(leg_dir / "final_mapped.json")
        print("#" * 96)
        print(f"{paper}/{mode}   [{run.split('/')[-1]}]")
        print(f"   {note}")
        if payload is None:
            print("   NO archived canonical payload -- skipped")
            results.append({"leg": f"{paper}/{mode}", "ok": False, "reason": "no payload"})
            print()
            continue

        meta = pathway_metadata(leg_dir, run_dir, paper)
        quarantine = load(leg_dir / "quarantine_report.json") or {}
        print(f"   pathway name : {meta['pathway_name']!r}")
        try:
            result = run_pwml_export(
                payload,
                pathway_name=meta["pathway_name"],
                pathway_description=meta["pathway_description"],
                pathway_subject=meta["pathway_subject"],
                project_root=PROJECT_ROOT,
                ref_path=_pwml_reference_path(PROJECT_ROOT),
                strict_db=True,
                quarantine_report=quarantine or None,
            )
        except Exception as exc:  # noqa: BLE001 -- a generator reports, it does not raise
            print(f"   EXPORT RAISED: {type(exc).__name__}: {exc}")
            results.append(
                {"leg": f"{paper}/{mode}", "ok": False, "reason": f"{type(exc).__name__}: {exc}"}
            )
            print()
            continue

        ok = bool(result.get("ok"))
        gate = result.get("required_gate_report") or {}
        gate_codes = sorted(
            {
                str((item or {}).get("code"))
                for item in (gate.get("errors") or [])
                if isinstance(item, dict)
            }
        )
        release = ((result.get("quarantine_report") or {}).get("release")) or {}
        xml = result.get("xml_bytes") or b""
        print(f"   export ok    : {ok}")
        print(f"   gate ok      : {gate.get('ok')}  errors={len(gate.get('errors') or [])} {gate_codes}")
        print(f"   release      : status={release.get('status')} "
              f"strict_acceptance_eligible={release.get('strict_acceptance_eligible')}")
        print(f"   xml bytes    : {len(xml)}")

        written = ""
        if ok and xml:
            status = str(release.get("status") or "review_required")
            stem = "pathway" if status == "release_ready" else "pathway.review_required"
            written = str(out_dir / f"{paper}_{mode}_{stem}.pwml")
            with open(written, "wb") as handle:
                handle.write(xml if isinstance(xml, bytes) else str(xml).encode("utf-8"))
            print(f"   WROTE        : {written}")
        else:
            print("   NO FILE WRITTEN -- the gated path refused, exactly as in production")

        results.append(
            {
                "leg": f"{paper}/{mode}",
                "run": run,
                "ok": ok,
                "gate_ok": gate.get("ok"),
                "gate_codes": gate_codes,
                "release_status": release.get("status"),
                "strict_acceptance_eligible": release.get("strict_acceptance_eligible"),
                "xml_bytes": len(xml),
                "written": written,
                "note": note,
            }
        )
        print()

    print("=" * 96)
    print("SUMMARY")
    print("=" * 96)
    for row in results:
        print(
            f"  {row['leg']:<24} ok={str(row.get('ok')):<5} "
            f"status={str(row.get('release_status')):<16} "
            f"bytes={row.get('xml_bytes', 0):<8} "
            f"{'WROTE ' + os.path.basename(row['written']) if row.get('written') else 'no file'}"
        )
    generated = [r for r in results if r.get("written")]
    print()
    print(f"  PWML files generated: {len(generated)}")
    print("  NONE of these has had its BIOLOGY reviewed by a human. That is "
          "PILOT-MANUAL-REVIEW.md's job and this script is not a substitute for it.")
    report = out_dir / "ORCH-729-generation-report.json"
    with open(report, "w", encoding="utf-8") as handle:
        json.dump({"instrument": "orch729", "reran_nothing": True, "results": results},
                  handle, indent=1, ensure_ascii=False)
    print(f"  report: {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
