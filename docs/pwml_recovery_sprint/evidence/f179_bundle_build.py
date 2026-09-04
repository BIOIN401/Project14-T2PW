"""F-179 evidence bundle builder -- makes the 2026-09-02 diagnosis independently auditable.

**THE SOURCE RUN IS NEVER TOUCHED.** ``runs_verify/2026-09-02_2052`` is generated,
protected run state and stays byte-identical and UNTRACKED. This module only READS it,
records a SHA-256 for every artifact it cites, and writes a compact bundle elsewhere.
Nothing here copies a whole payload: the bundle carries the minimum records that support
each finding, plus the hashes that let a reviewer verify those records against the
original on a machine that has it.

WHY A BUNDLE AND NOT A COMMIT OF THE RUN. Committing the run would put ~10 legs of
generated state into a repository whose ``.git`` is already 158 MB, and the run is on
the protected list. But a diagnosis resting on an untracked directory is exactly what
reversed a wave's headline result before, so the diagnosis must become checkable
WITHOUT the run becoming tracked. Hashes plus minimal extracts do that: a reviewer with
the run can verify every hash; a reviewer without it can still read every record the
findings rest on.

WHAT "MINIMUM PERMISSIBLE" MEANS HERE. For each finding, the smallest record that
carries the claim: the reaction row itself (a shortcut claim is about one row), the
release facts (four scalars), the gate error list (already small), and the failure line
from ``RESULT.txt``. Full payloads, full source text and caches are never copied.

Usage:
  python f179_bundle_build.py <repo-root> [--out <bundle-dir>] [--verify]

``--verify`` re-hashes every inventory entry against the live run and reports drift
instead of rewriting the bundle. That is the reproduction path.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

RUN = "runs_verify/2026-09-02_2052"

#: Legs and the finding each supports. Kept explicit so the bundle's scope is auditable
#: rather than a glob that silently grows.
FINDINGS: Dict[str, Dict[str, Any]] = {
    "F-179": {
        "title": "false-positive biological export: fabricated one-step pathway collapse",
        "paper": "PMC12180156", "mode": "strict",
        "classification": "production defect / false-positive biological export",
    },
    "F-180": {
        "title": "composite tokenizer misparses ionic charge notation (Fe3+)",
        "paper": "PMC12452463", "mode": "strict",
        "classification": "production tokenizer defect, secondary/deferred",
    },
    "F-181": {
        "title": "interactions reference registry entities that were never registered",
        "paper": "PMC12856317", "mode": "strict",
        "classification": "production referential-integrity defect, secondary/deferred",
    },
    "F-182": {
        "title": "final gate report never emitted; leg fails as final_gate_report_missing",
        "paper": "PMC12444477", "mode": "strict",
        "classification": "lifecycle/evaluation observability defect, NOT automatically biological failure",
    },
}

#: Artifacts cited per leg. Hashed always; extracted only where a finding needs it.
CITED = ("final_mapped.json", "quarantine_report.json", "gate_fail_report.json",
         "initial_stage3_gate_report.json", "final_stage3_gate_report.json",
         "RESULT.txt", "LEG_TERMINAL.json")


def sha256(path: Path) -> Optional[str]:
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def result_failure_lines(path: Path) -> List[str]:
    """The failure-bearing lines of a RESULT.txt. Never the whole file."""

    if not path.is_file():
        return []
    keep = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        s = line.strip()
        if s.startswith(("RESULT:", "failure_kind", "message", "issue codes")) or \
                "blocking_issues" in s:
            keep.append(s[:400])
    return keep


def release_facts(leg: Path) -> Dict[str, Any]:
    q = leg / "quarantine_report.json"
    if not q.is_file():
        return {"release_status": "artifact_missing"}
    try:
        rel = (json.loads(q.read_text(encoding="utf-8")).get("release") or {})
    except ValueError:
        return {"release_status": "artifact_malformed"}
    return {k: rel.get(k) for k in
            ("status", "semantic_evaluation", "strict_gates_passed",
             "semantic_not_evaluated_reason")}


def gate_errors(leg: Path) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for fn in ("gate_fail_report.json", "final_stage3_gate_report.json"):
        p = leg / fn
        if not p.is_file():
            out[fn] = "ABSENT"
            continue
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except ValueError:
            out[fn] = "MALFORMED"
            continue
        errs = d.get("errors")
        if isinstance(errs, list):
            # reason + path only; the `detail` blob repeats whole rows.
            out[fn] = {"ok": d.get("ok"), "status": d.get("status"), "phase": d.get("phase"),
                       "errors": [{"path": e.get("path"), "reason": str(e.get("reason"))[:400]}
                                  for e in errs if isinstance(e, dict)]}
        else:
            out[fn] = {"status": d.get("status"), "phase": d.get("phase"),
                       "error": str(d.get("error"))[:400]}
    return out


def shortcut_rows(leg: Path) -> List[Dict[str, Any]]:
    """Reaction rows with their attribution -- the F-179 claim is about these rows."""

    p = leg / "final_mapped.json"
    if not p.is_file():
        return []
    try:
        doc = json.loads(p.read_text(encoding="utf-8"))
    except ValueError:
        return []
    rows = []
    for i, r in enumerate((doc.get("processes") or {}).get("reactions") or []):
        if not isinstance(r, dict):
            continue
        rows.append({
            "row_index": i, "name": r.get("name"),
            "inputs": r.get("inputs"), "outputs": r.get("outputs"),
            "enzymes": [e.get("entity") if isinstance(e, dict) else e
                        for e in (r.get("enzymes") or [])],
            "has_provenance_lineage": bool(r.get("provenance_lineage")),
            "lineage_stages": sorted({e.get("stage") for e in (r.get("provenance_lineage") or [])
                                      if isinstance(e, dict) and e.get("stage")}) or None,
            "has_rag_provenance": bool(r.get("rag_provenance")),
            # presence only: this string reaches 100k+ chars and can carry EXTERNAL text
            "evidence_chars": len(r["evidence"]) if isinstance(r.get("evidence"), str) else 0,
        })
    return rows


def entity_provenance(leg: Path, names: List[str]) -> List[Dict[str, Any]]:
    p = leg / "final_mapped.json"
    if not p.is_file():
        return []
    try:
        doc = json.loads(p.read_text(encoding="utf-8"))
    except ValueError:
        return []
    want = {n.strip().lower() for n in names}
    out = []
    for kind, rows in ((doc.get("entities") or {}).items()):
        for r in rows or []:
            if not isinstance(r, dict) or str(r.get("name", "")).strip().lower() not in want:
                continue
            out.append({
                "kind": kind, "name": r.get("name"),
                "lineage": [{"stage": e.get("stage"), "origin": e.get("origin"),
                             "paper_explicit": e.get("paper_explicit"),
                             "sources": [s.get("source_id") for s in (e.get("sources") or [])
                                         if isinstance(s, dict)]}
                            for e in (r.get("provenance_lineage") or []) if isinstance(e, dict)],
                "rag_provenance_source": (r.get("rag_provenance") or {}).get("source_id"),
            })
    return out


def build(root: Path, out_dir: Path) -> Dict[str, Any]:
    inventory: List[Dict[str, Any]] = []
    extracts: Dict[str, Any] = {}

    for fid, meta in FINDINGS.items():
        leg_rel = f"{RUN}/papers/{meta['paper']}/{meta['mode']}"
        leg = root / leg_rel
        files = []
        for fn in CITED:
            p = leg / fn
            files.append({"path": f"{leg_rel}/{fn}", "sha256": sha256(p),
                          "bytes": p.stat().st_size if p.is_file() else None,
                          "present": p.is_file()})
        inventory.append({"finding": fid, "leg": leg_rel, "artifacts": files})
        rec: Dict[str, Any] = {
            "finding": fid, "title": meta["title"], "classification": meta["classification"],
            "source_leg": leg_rel,
            "result_txt_failure_lines": result_failure_lines(leg / "RESULT.txt"),
            "release_facts": release_facts(leg),
            "gate_errors": gate_errors(leg),
        }
        if fid == "F-179":
            rec["canonical_reaction_rows"] = shortcut_rows(leg)
            rec["participant_provenance"] = entity_provenance(leg, ["glycine", "heme"])
            rec["pwml_files"] = sorted(os.path.basename(x)
                                       for x in glob.glob(str(leg / "*.pwml")))
            for x in glob.glob(str(leg / "*.pwml")):
                inventory[-1]["artifacts"].append(
                    {"path": f"{leg_rel}/{os.path.basename(x)}",
                     "sha256": sha256(Path(x)), "bytes": os.path.getsize(x), "present": True})
        extracts[fid] = rec

    # Run-level artifacts that the scope-trap finding rests on.
    for rel in (f"{RUN}/SUMMARY.txt", f"{RUN}/failures_by_code.txt", "topics_t104.txt"):
        p = root / rel
        inventory.append({"finding": "context", "leg": "-", "artifacts": [
            {"path": rel, "sha256": sha256(p),
             "bytes": p.stat().st_size if p.is_file() else None, "present": p.is_file()}]})

    return {"bundle": "F-179 evidence bundle", "source_run": RUN,
            "source_run_tracked": False,
            "source_run_untouched": True,
            "inventory": inventory, "extracts": extracts}


def verify(root: Path, out_dir: Path) -> int:
    inv_path = out_dir / "INVENTORY.json"
    if not inv_path.is_file():
        print(f"no inventory at {inv_path}")
        return 1
    doc = json.loads(inv_path.read_text(encoding="utf-8"))
    drift = 0
    checked = 0
    for entry in doc["inventory"]:
        for a in entry["artifacts"]:
            p = root / a["path"]
            now = sha256(p)
            checked += 1
            if now != a["sha256"]:
                drift += 1
                print(f"  DRIFT {a['path']}\n    recorded {a['sha256']}\n    now      {now}")
    print(f"verified {checked} artifacts, {drift} drifted")
    return 1 if drift else 0


def main(argv: List[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("repo_root")
    ap.add_argument("--out", default="docs/pwml_recovery_sprint/evidence/f179_bundle")
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args(argv)

    root = Path(args.repo_root).resolve()
    out_dir = root / args.out
    if args.verify:
        return verify(root, out_dir)

    out_dir.mkdir(parents=True, exist_ok=True)
    doc = build(root, out_dir)
    (out_dir / "INVENTORY.json").write_text(
        json.dumps({k: v for k, v in doc.items() if k != "extracts"}, indent=1,
                   ensure_ascii=False), encoding="utf-8")
    (out_dir / "EXTRACTS.json").write_text(
        json.dumps(doc["extracts"], indent=1, ensure_ascii=False), encoding="utf-8")

    n = sum(len(e["artifacts"]) for e in doc["inventory"])
    print(f"bundle written to {out_dir}")
    print(f"  inventory entries : {len(doc['inventory'])}  artifacts hashed: {n}")
    print(f"  source run        : {RUN} (untracked, byte-untouched)")
    for fid in FINDINGS:
        print(f"  extract           : {fid}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
