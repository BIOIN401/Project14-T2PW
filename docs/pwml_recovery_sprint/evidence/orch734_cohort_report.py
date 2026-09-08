"""ORCH-734 -- score one smoke run into the charter's per-leg table and mechanism census.

READ-ONLY. Opens committed and uncommitted run artifacts, calls two production
functions on already-written payloads, and writes nothing into the run tree.

The charter asks for two numbers that a single "PWML rate" cannot express, so
they are computed separately and never collapsed:

  * **meaningful-core yield**  -- legs that produced a defensible canonical
    reaction core: ``quarantine_report.counts.core_accepted >= 1`` AND the F-179
    verdict on the leg's own final payload is not ``no_defensible_core``.
  * **serialization yield**    -- of those cores, how many became a PWML on disk.

The gap between them is the whole diagnostic: a core that never serializes is a
*downstream PWML-generation* failure, and a leg with no core is an *extraction*
failure. Reporting only "3/12 PWML" would hide which one the run is suffering
from, which is the question that decides the stopping rule.

F-179 is recomputed here by calling ``evaluate_reaction_support`` on the leg's
own ``final_mapped.json`` rather than scraped out of a report string. The
function is pure and mutates nothing (its own docstring says so), so this reads
the production verdict instead of a paraphrase of it.

MECHANISM CLASSIFICATION. Every no-PWML leg gets exactly one dominant cause from
the charter's ten-way vocabulary. The classifier is ordered earliest-stage-first
because a leg that died at Stage 1 has no opinion about the export gate, and it
refuses to guess: anything it cannot place lands in ``other`` with the evidence
attached, rather than being quietly filed under a plausible neighbour.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

_HERE = Path(__file__).resolve()
_ROOT = _HERE.parents[3]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

# The charter's ten-way vocabulary. Closed on purpose: an invented mechanism
# reaches the aggregate as a string somebody has to bucket by hand.
MECHANISMS = (
    "stage1_provider_delivery",
    "stage1_extraction_content",
    "scope_or_guard_refusal",
    "identity_resolution",
    "quarantine_core_coverage",
    "live_semantic_gate",
    "f179",
    "required_field_export_gate",
    "f192_autostate",
    "other",
)


def _load(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except Exception:  # noqa: BLE001
        return None


def _f179(leg: Path) -> Dict[str, Any]:
    """The production F-179 verdict on this leg's own final payload."""

    for name in ("final_mapped.json", "merged_payload.json", "stage1_payload.json"):
        payload = _load(leg / name)
        if payload is None:
            continue
        try:
            from t2pw.pipeline.reaction_support import evaluate_reaction_support

            out = dict(evaluate_reaction_support(payload))
            out["source"] = name
            return out
        except Exception as exc:  # noqa: BLE001
            return {"verdict": "uncomputable", "reason": f"{exc!r}", "source": name}
    return {"verdict": "unreachable", "reason": "no payload artifact on disk", "source": ""}


def _blob(leg: Path) -> str:
    """Everything textual this leg wrote, for evidence-based classification."""

    parts: List[str] = []
    for name in (
        "RESULT.txt",
        "LEG_TERMINAL.json",
        "extraction_boundary_report.json",
        "stage0_attempts.json",
        "pwml_required_field_gate_report.json",
        "final_stage3_gate_report.json",
    ):
        p = leg / name
        if p.exists():
            parts.append(p.read_text(encoding="utf-8", errors="replace"))
    return "\n".join(parts)


def _classify(row: Dict[str, Any], leg: Path) -> Dict[str, Any]:
    """One dominant mechanism per no-PWML leg. Earliest stage wins."""

    blob = _blob(leg)
    stage = (row.get("stage") or "").lower()
    fk = (row.get("failure_kind") or "").lower()
    msg = ((row.get("message") or "") + " " + (row.get("detail") or "")).lower()
    codes = " ".join(str(c) for c in (row.get("issue_codes") or [])).lower()
    hay = " ".join((blob, stage, fk, msg, codes)).lower()
    ev: List[str] = []

    def hit(*needles: str) -> bool:
        found = [n for n in needles if n in hay]
        ev.extend(found)
        return bool(found)

    # 1. Stage-1 DELIVERY: the provider did not hand back usable text at all.
    #    ORCH-733's classes A-D. Distinguished from content failure by the shape
    #    of the completion, not by the stage name -- both die at "stage1".
    if hit(
        "identical_empty_response",
        "content_chars\": 0",
        "structurally empty payload",
        "failed to produce valid json",
        "jsondecodeerror",
        "expected json object",
        "empty completion",
    ):
        return {"mechanism": "stage1_provider_delivery", "evidence": ev}

    # 2. Scope / guard refusal BEFORE extraction had a chance. Stage 0.
    if hit("scope_conflict", "off_topic", "no_mechanistic_pathway_terms", "stage0") and (
        "stage0" in stage or "screen" in stage
    ):
        return {"mechanism": "scope_or_guard_refusal", "evidence": ev}

    # 3. Stage-1 CONTENT: the provider delivered fine and the content was wrong
    #    or absent. ORCH-733's classes H/I/J, the larger population.
    if "stage1" in stage or "extraction" in stage:
        hit("required container", "entities object", "zero processes", "no reactions")
        return {"mechanism": "stage1_extraction_content", "evidence": ev or ["terminal at stage1"]}

    # 4. F-192 -- the auto-state lifecycle defect. Named explicitly BEFORE the
    #    generic required-field gate, because it surfaces THROUGH that gate and
    #    would otherwise be filed as an ordinary export-gate failure, which is
    #    exactly the misattribution that made it invisible until C-120.
    if hit("no_biological_states"):
        return {"mechanism": "f192_autostate", "evidence": ev}

    # 5. F-179 -- the biological support floor. A refusal here is the safeguard
    #    working, not a reliability defect, and it is reported as its own bucket
    #    so it can never be counted as a bug to be fixed.
    if row.get("f179_verdict") == "no_defensible_core" or hit("no_defensible_core"):
        return {"mechanism": "f179", "evidence": ev or ["f179 verdict"]}

    # 6. Identity resolution -- C-120's territory.
    if hit(
        "species_missing_taxonomy",
        "species_missing_classification",
        "missing_taxonomy",
        "unknown_protein",
        "unresolved",
    ):
        return {"mechanism": "identity_resolution", "evidence": ev}

    # 7. Quarantine / core coverage.
    if row.get("core_accepted") == 0 or hit("core_accepted\": 0", "quarantin", "coverage"):
        return {"mechanism": "quarantine_core_coverage", "evidence": ev or ["core_accepted = 0"]}

    # 8. The live pre-export semantic gate.
    if hit("final_pre_export_stage3", "stage3 gate", "semantic"):
        return {"mechanism": "live_semantic_gate", "evidence": ev}

    # 9. Required-field / export gate, generic.
    if hit("required_field", "gate_errors", "export blocked", "pwml_export"):
        return {"mechanism": "required_field_export_gate", "evidence": ev}

    return {
        "mechanism": "other",
        "evidence": [f"stage={stage}", f"failure_kind={fk}", f"message={msg[:180]}"],
    }


def score(run_dir: Path) -> Dict[str, Any]:
    manifest = run_dir / "manifest.jsonl"
    if not manifest.exists():
        raise SystemExit(f"no manifest.jsonl under {run_dir}")

    rows: List[Dict[str, Any]] = []
    for line in manifest.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))

    legs: List[Dict[str, Any]] = []
    for m in rows:
        leg = run_dir / (m.get("dir") or "")
        counts = m.get("counts") or {}
        release = m.get("release_status") or {}
        quarantine = _load(leg / "quarantine_report.json") or {}
        qcounts = quarantine.get("counts") or {}
        f179 = _f179(leg)
        gate = _load(leg / "final_stage3_gate_report.json") or {}
        reqgate = _load(leg / "pwml_required_field_gate_report.json") or {}

        pwml_name = m.get("pwml_artifact") or ""
        pwml_path = (leg / pwml_name) if pwml_name else None
        pwml_bytes = 0
        if pwml_path is not None and pwml_path.exists():
            pwml_bytes = pwml_path.stat().st_size
        elif not pwml_name:
            for cand in sorted(leg.glob("*.pwml")):
                pwml_name, pwml_bytes = cand.name, cand.stat().st_size
                break

        organisms = ((m.get("observed_context") or {}).get("observed_organisms")) or []
        core = int(qcounts.get("core_accepted") or 0)
        row: Dict[str, Any] = {
            "paper": m.get("paper_id"),
            "mode": m.get("mode"),
            "pathway": m.get("topic"),
            "organism": ", ".join(str(o) for o in organisms) or "(none observed)",
            "status": m.get("status"),
            "stage": m.get("stage"),
            "seconds": round(float(m.get("seconds") or 0.0), 1),
            "failure_kind": m.get("failure_kind") or "",
            "message": (m.get("message") or "")[:220],
            "detail": (m.get("detail") or "")[:220],
            "issue_codes": m.get("issue_codes") or [],
            "reactions": int(counts.get("reactions") or 0),
            "transports": int(counts.get("transports") or 0),
            "core_accepted": core,
            "quarantined": {
                k: v for k, v in qcounts.items() if k.startswith("quarantined") and v
            },
            "f179_verdict": f179.get("verdict"),
            "f179_reason": (f179.get("reason") or "")[:200],
            "f179_supported": f179.get("supported"),
            "f179_reactions": f179.get("reactions"),
            "stage3_gate_ok": gate.get("ok"),
            "required_field_gate_ok": reqgate.get("ok"),
            "required_field_errors": [
                str(e)[:120] for e in (reqgate.get("errors") or [])
            ][:6],
            "release_status": release.get("status") or "",
            "strict_gates_passed": release.get("strict_gates_passed"),
            "superseded_disposition": [
                str(x)
                for x in (release.get("superseded_reports") or release.get("notes") or [])
            ][:4],
            "pwml": bool(pwml_bytes),
            "pwml_file": pwml_name if pwml_bytes else "",
            "pwml_bytes": pwml_bytes,
        }
        row["meaningful_core"] = bool(core >= 1 and row["f179_verdict"] != "no_defensible_core")
        if not row["pwml"]:
            row.update(_classify(row, leg))
        legs.append(row)

    attempted = len(legs)
    with_pwml = [r for r in legs if r["pwml"]]
    cores = [r for r in legs if r["meaningful_core"]]
    cores_serialized = [r for r in cores if r["pwml"]]

    census: Dict[str, int] = {}
    for r in legs:
        if not r["pwml"]:
            census[r["mechanism"]] = census.get(r["mechanism"], 0) + 1

    repeated = sorted(
        ((m, n) for m, n in census.items() if n >= 2), key=lambda kv: -kv[1]
    )

    return {
        "task": "ORCH-734",
        "run_dir": str(run_dir).replace("\\", "/"),
        "dataset_identity": (
            "final expected-working reliability smoke cohort -- NOT the ORCH-724 pilot, "
            "NOT ORCH-730, NOT ORCH-732, NOT the C-120 validation; its numbers may not be "
            "merged into any of those denominators"
        ),
        "papers_attempted": attempted,
        "meaningful_cores": len(cores),
        "pwml_files": len(with_pwml),
        "serialization_yield": f"{len(cores_serialized)}/{len(cores)}",
        "pwml_rate_note": (
            "a 12-paper smoke cohort is not a population-wide statistical rate"
        ),
        "no_pwml_mechanism_census": dict(sorted(census.items(), key=lambda kv: -kv[1])),
        "repeated_mechanisms": [
            {"mechanism": m, "papers": n, "meets_repeat_threshold": True} for m, n in repeated
        ],
        "legs": legs,
    }


def _table(report: Dict[str, Any]) -> str:
    out = [
        "| Paper | Organism | Pathway | Rxns | Core | PWML | Bytes | Disposition | Failure if any |",
        "| --- | --- | --- | ---: | ---: | --- | ---: | --- | --- |",
    ]
    for r in report["legs"]:
        fail = "" if r["pwml"] else f"`{r.get('mechanism','')}` -- {r['message'][:70]}"
        out.append(
            "| `%s` | *%s* | %s | %d | %d | %s | %s | %s | %s |"
            % (
                r["paper"],
                r["organism"][:34],
                r["pathway"],
                r["reactions"],
                r["core_accepted"],
                "**YES**" if r["pwml"] else "NO",
                f"{r['pwml_bytes']:,}" if r["pwml_bytes"] else "--",
                r["release_status"] or r["status"],
                fail,
            )
        )
    return "\n".join(out)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_dir")
    ap.add_argument("--json", dest="json_path", default=None)
    ap.add_argument("--table", action="store_true")
    args = ap.parse_args(argv)

    report = score(Path(args.run_dir))
    text = json.dumps(report, indent=2)
    if args.json_path:
        Path(args.json_path).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_path).write_text(text, encoding="utf-8")
    if args.table:
        print(_table(report))
        print()
        print(
            "attempted %d | meaningful cores %d | PWML %d | serialization %s"
            % (
                report["papers_attempted"],
                report["meaningful_cores"],
                report["pwml_files"],
                report["serialization_yield"],
            )
        )
        print("no-PWML mechanisms:", report["no_pwml_mechanism_census"])
        print("repeated (>=2 papers):", [m["mechanism"] for m in report["repeated_mechanisms"]])
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
