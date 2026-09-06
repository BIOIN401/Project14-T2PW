"""ORCH-724 -- the unseen-pilot product summary and manual-review package.

READ-ONLY over one run directory. Builds the two things the product owner's charter
asks for after the pilot:

  section 11  the per-leg PRODUCT table -- did a PWML generate, what release state, how
              many reactions, is the graph valid, is referential integrity intact, is
              the species right, is there obvious unsupported chemistry;

  section 12  the MANUAL-REVIEW PACKAGE -- for every generated PWML, the paper, mode,
              PWML path, source-text path, final payload path, reaction list, enzymes,
              organisms, RAG-added reactions and warnings, so a human can label each
              PASS / PASS WITH MINOR OMISSIONS / MAJOR BIOLOGICAL ERROR / INVALID.

=============================================================================
WHAT THIS DOES NOT DO
=============================================================================

It does NOT score biology. There is no gold set for these ten papers and none is being
invented: ``release_ready`` is reported as a fact about the runtime, not as a verdict
that the pathway is right, and ``review_required`` is NOT counted as a failure -- the
charter is explicit that a ``review_required`` PWML can be a successful product output
if it is biologically useful. The biological verdict is the human's, and this script
exists to make that judgement cheap rather than to pre-empt it.

"Obvious unsupported chemistry" is likewise a POINTER, not a ruling: it counts retained
reactions carrying no provenance of any kind, using the same reading the F-179 work used.
A flagged row is something for the reviewer to look at first.

Usage:
  python orch724_pilot_summary.py <repo-root> --run <run-dir> [--json OUT] [--md OUT]
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

PROCESS_BUCKETS = ("reactions", "transports", "interactions")


def _load(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _names(value: Any) -> List[str]:
    out: List[str] = []
    for item in (value or []):
        if isinstance(item, str):
            if item.strip():
                out.append(item.strip())
        elif isinstance(item, dict):
            for key in ("name", "compound", "protein", "element", "label", "entity"):
                v = item.get(key)
                if isinstance(v, str) and v.strip():
                    out.append(v.strip())
                    break
    return out


def _has_rag(row: Dict[str, Any]) -> bool:
    if row.get("rag_provenance"):
        return True
    lineage = row.get("provenance_lineage")
    blob = json.dumps(lineage) if lineage else ""
    return "rag" in blob.lower()


def _has_any_provenance(row: Dict[str, Any]) -> bool:
    for key in ("provenance_lineage", "rag_provenance", "source", "source_ref",
                "evidence", "sources", "provenance"):
        if row.get(key):
            return True
    return False


def summarize_leg(leg_dir: Path, paper: str, mode: str) -> Dict[str, Any]:
    payload = _load(leg_dir / "final_mapped.json") or {}
    qa = _load(leg_dir / "pwml_qa.json") or {}
    validation = _load(leg_dir / "pwml_validation_report.json") or {}
    ir_validation = _load(leg_dir / "pwml_ir_validation.json") or {}
    coverage = _load(leg_dir / "coverage_summary.json") or {}
    admission = _load(leg_dir / "rag_admission_report.json") or {}

    pwml_files = sorted(p.name for p in leg_dir.glob("*.pwml"))
    processes = payload.get("processes") or {}
    entities = payload.get("entities") or {}

    rows: List[Dict[str, Any]] = []
    for bucket in PROCESS_BUCKETS:
        for row in (processes.get(bucket) or []):
            if isinstance(row, dict):
                rows.append(row)

    reactions = [r for r in (processes.get("reactions") or []) if isinstance(r, dict)]
    no_prov = [r for r in rows if not _has_any_provenance(r)]
    rag_rows = [r for r in rows if _has_rag(r)]

    species = _names(entities.get("species"))

    reaction_list = []
    for r in rows:
        reaction_list.append({
            "name": r.get("name") or "(unnamed)",
            "inputs": _names(r.get("inputs")),
            "outputs": _names(r.get("outputs")),
            "enzymes": _names(r.get("enzymes")) or _names(r.get("proteins")),
            "rag_added": _has_rag(r),
            "no_provenance": not _has_any_provenance(r),
        })

    enzymes = sorted({e for r in reaction_list for e in r["enzymes"]})

    return {
        "paper": paper,
        "mode": mode,
        "leg_dir": leg_dir.as_posix(),
        # ---- section 11: the product table -------------------------------
        "pwml_generated": bool(pwml_files),
        "pwml_files": pwml_files,
        "pwml_path": (leg_dir / pwml_files[0]).as_posix() if pwml_files else "",
        "reaction_count": len(reactions),
        "process_row_count": len(rows),
        "core_accepted_processes": coverage.get("core_accepted_processes"),
        "coverage_ratio": coverage.get("coverage_ratio"),
        "minimum_core_satisfied": coverage.get("minimum_core_satisfied"),
        "quarantined_processes": coverage.get("quarantined_processes"),
        "graph_valid": validation.get("ok"),
        "graph_issue_count": validation.get("issue_count"),
        "ir_valid": (ir_validation.get("ok") if isinstance(ir_validation, dict) else None),
        "qa_ok": qa.get("ok"),
        "qa_errors": len((qa.get("errors") or [])),
        "qa_warnings": len((qa.get("warnings") or [])),
        "species": species,
        "entity_counts": {k: len(v) for k, v in entities.items() if isinstance(v, list)},
        "rows_without_any_provenance": len(no_prov),
        "rag_added_rows": len(rag_rows),
        "rag_accepted": len((admission.get("accepted") or [])),
        "rag_rejected": len((admission.get("rejected") or [])),
        # ---- section 12: the manual-review package ------------------------
        "final_payload_path": (leg_dir / "final_mapped.json").as_posix(),
        "stage1_payload_path": (leg_dir / "stage1_payload.json").as_posix(),
        "reaction_list": reaction_list,
        "enzymes": enzymes,
    }


def main(argv: List[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("repo_root")
    ap.add_argument("--run", required=True, help="run directory, repo-relative or absolute")
    ap.add_argument("--json", dest="json_path", default=None)
    ap.add_argument("--md", dest="md_path", default=None)
    args = ap.parse_args(argv)

    root = Path(args.repo_root).resolve()
    run_dir = Path(args.run)
    if not run_dir.is_absolute():
        run_dir = root / run_dir

    manifest_rows: Dict[tuple, Dict[str, Any]] = {}
    mf = run_dir / "manifest.jsonl"
    if mf.is_file():
        for line in mf.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except Exception:
                continue
            manifest_rows[(d.get("paper_id"), d.get("mode"))] = d

    legs: List[Dict[str, Any]] = []
    for paper_dir in sorted((run_dir / "papers").glob("*")):
        if not paper_dir.is_dir():
            continue
        for mode_dir in sorted(paper_dir.glob("*")):
            if not mode_dir.is_dir():
                continue
            rec = summarize_leg(mode_dir, paper_dir.name, mode_dir.name)
            m = manifest_rows.get((paper_dir.name, mode_dir.name)) or {}
            # ``release_status`` is a RECORD, not a string. Keep the whole thing --
            # ``semantic_not_evaluated_reason`` and ``missing_anchors`` are exactly the
            # fields a reviewer needs -- but surface the scalar the tables key on, so a
            # dict is never str()'d into a column and truncated into nonsense.
            release = m.get("release_status")
            rec["release_status_record"] = release
            if isinstance(release, dict):
                rec["release_status"] = release.get("status")
                rec["semantic_evaluation"] = release.get("semantic_evaluation")
                rec["strict_gates_passed"] = release.get("strict_gates_passed")
                rec["completeness"] = release.get("completeness")
                rec["missing_anchors"] = release.get("missing_anchors") or []
                rec["release_reasons"] = release.get("reasons") or []
            else:
                rec["release_status"] = release
                rec["semantic_evaluation"] = None
                rec["strict_gates_passed"] = None
                rec["completeness"] = None
                rec["missing_anchors"] = []
                rec["release_reasons"] = []
            rec["run_status"] = m.get("status")
            rec["failure_kind"] = m.get("failure_kind")
            rec["issue_codes"] = m.get("issue_codes")
            rec["seconds"] = m.get("seconds")
            rec["warnings"] = m.get("warnings")
            rec["topic"] = m.get("topic")
            legs.append(rec)

    totals = collections.Counter()
    totals["legs"] = len(legs)
    for leg in legs:
        totals["pwml_generated"] += 1 if leg["pwml_generated"] else 0
        totals["graph_valid"] += 1 if leg["graph_valid"] else 0
        totals["reactions"] += leg["reaction_count"]
        totals["rows_without_provenance"] += leg["rows_without_any_provenance"]
        rs = leg.get("release_status") or "(none)"
        totals["release:" + str(rs)] += 1

    result = {
        "instrument": "orch724_pilot_summary",
        "run_dir": run_dir.as_posix(),
        "evaluation_only": True,
        "note": (
            "release_ready is a runtime fact, not a biological verdict. review_required "
            "is NOT a failure. No gold set exists for these papers and none is invented."
        ),
        "totals": dict(totals),
        "legs": legs,
    }

    # ---- console table ----------------------------------------------------
    print("=" * 118)
    print("ORCH-724 UNSEEN PILOT -- per-leg product summary:", run_dir.as_posix())
    print("=" * 118)
    print("%-13s %-9s %-5s %-17s %-14s %5s %5s %5s %6s %6s %5s" %
          ("paper", "mode", "pwml", "release_status", "semantic", "rxn", "rows", "noprv",
           "graph", "qa_err", "rag+"))
    print("-" * 118)
    for leg in legs:
        print("%-13s %-9s %-5s %-17s %-14s %5d %5d %5d %6s %6d %5d" % (
            leg["paper"], leg["mode"], "YES" if leg["pwml_generated"] else "no",
            str(leg.get("release_status"))[:17], str(leg.get("semantic_evaluation"))[:14],
            leg["reaction_count"], leg["process_row_count"],
            leg["rows_without_any_provenance"], str(leg["graph_valid"]), leg["qa_errors"],
            leg["rag_added_rows"]))
    print("-" * 118)
    print("legs=%d  pwml_generated=%d  graph_valid=%d  total_reactions=%d  rows_without_provenance=%d" % (
        totals["legs"], totals["pwml_generated"], totals["graph_valid"],
        totals["reactions"], totals["rows_without_provenance"]))
    for k in sorted(k for k in totals if k.startswith("release:")):
        print("   %-28s %d" % (k, totals[k]))

    if args.json_path:
        p = Path(args.json_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        print("\nJSON:", p)

    if args.md_path:
        L: List[str] = []
        L.append("# ORCH-724 unseen pilot — manual PWML review package\n")
        L.append("Run: `%s`\n" % run_dir.as_posix())
        L.append("> `release_ready` is a runtime fact, not a biological verdict. "
                 "**`review_required` is not a failure** — a `review_required` PWML is a "
                 "successful product output if it is biologically useful.\n")
        L.append("\nLabel each leg: **PASS** · **PASS WITH MINOR OMISSIONS** · "
                 "**MAJOR BIOLOGICAL ERROR** · **INVALID OR UNUSABLE PWML**\n")
        L.append("\nMinor omissions: ordinary cofactors, currency metabolites, non-defining "
                 "regulators, ancillary proteins. Major errors: fabricated reaction, wrong "
                 "defining substrate/product, missing major pathway branch, wrong organism, "
                 "repeated major enzyme error, unusable PWML.\n")
        for leg in legs:
            L.append("\n---\n")
            L.append("## %s / %s\n" % (leg["paper"], leg["mode"]))
            L.append("| field | value |")
            L.append("|---|---|")
            L.append("| requested scope | %s |" % (leg.get("topic") or ""))
            L.append("| PWML | %s |" % (leg["pwml_path"] or "**none generated**"))
            L.append("| release status | %s |" % leg.get("release_status"))
            L.append("| semantic evaluation | %s |" % leg.get("semantic_evaluation"))
            L.append("| completeness / missing anchors | %s / %s |" % (
                leg.get("completeness"), ", ".join(leg.get("missing_anchors") or []) or "none"))
            L.append("| source text | %s |" % leg["stage1_payload_path"])
            L.append("| final payload | %s |" % leg["final_payload_path"])
            L.append("| organisms | %s |" % (", ".join(leg["species"]) or "(none)"))
            L.append("| enzymes | %s |" % (", ".join(leg["enzymes"]) or "(none)"))
            L.append("| reactions / rows | %d / %d |" % (leg["reaction_count"], leg["process_row_count"]))
            L.append("| graph valid | %s (issues %s) |" % (leg["graph_valid"], leg["graph_issue_count"]))
            L.append("| QA | ok=%s errors=%d warnings=%d |" % (leg["qa_ok"], leg["qa_errors"], leg["qa_warnings"]))
            L.append("| RAG added rows | %d (accepted %d / rejected %d) |" % (
                leg["rag_added_rows"], leg["rag_accepted"], leg["rag_rejected"]))
            L.append("| rows with NO provenance | **%d** |" % leg["rows_without_any_provenance"])
            L.append("\n**Reactions**\n")
            if not leg["reaction_list"]:
                L.append("_none retained_\n")
            else:
                L.append("| # | name | substrates | products | enzymes | RAG | no prov |")
                L.append("|---|---|---|---|---|---|---|")
                for i, r in enumerate(leg["reaction_list"], 1):
                    L.append("| %d | %s | %s | %s | %s | %s | %s |" % (
                        i, r["name"], ", ".join(r["inputs"]) or "—",
                        ", ".join(r["outputs"]) or "—", ", ".join(r["enzymes"]) or "—",
                        "yes" if r["rag_added"] else "", "**YES**" if r["no_provenance"] else ""))
            L.append("\n**Human label:** `PASS` / `PASS WITH MINOR OMISSIONS` / "
                     "`MAJOR BIOLOGICAL ERROR` / `INVALID OR UNUSABLE PWML`\n")
            L.append("**Notes:**\n")
        p = Path(args.md_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("\n".join(L), encoding="utf-8")
        print("MARKDOWN:", p)

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
