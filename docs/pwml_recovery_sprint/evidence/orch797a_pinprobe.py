"""ORCH-797a: what the moved pins actually moved (run once per tree; diff the two outputs).

Usage: python orch797a_pinprobe.py --src <tree>/src --tests <worktree>/tests --out <file.json>

1. Corpus collision audit: every committed ``final_mapped.json`` (``git ls-files``) under the worktree, entity rows
   grouped by the D-120 reference key (tip tree only; the base tree has no ``reference_key``): groups whose members
   differ under the pre-D-120 IR key, same kind or across kinds, and references that hit a row only under the new key.
2. ``test_orch766_unreferenced_species``: per kind, the sha256 of each of the four digested components
   (payload, removed, closure, quarantine) and of the quarantine report minus its key-text fields.
"""

import argparse
import gzip
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--src", required=True)
ap.add_argument("--tests", required=True)
ap.add_argument("--out", required=True)
args = ap.parse_args()
sys.path.insert(0, str(Path(args.src).resolve()))
import t2pw  # noqa: E402

print("t2pw resolved:", Path(t2pw.__file__).resolve().parent)
sys.path.insert(1, str(Path(args.tests).resolve()))
ROOT = Path(args.tests).resolve().parent
BUCKETS = ("compounds", "proteins", "protein_complexes", "nucleic_acids", "element_collections")
out = {}


def sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, default=str).encode()).hexdigest()[:16]


def old_ir(v):
    return re.sub(r"[^a-z0-9:+ ]+", " ", re.sub(r"\s+", " ", str(v or "")).strip().casefold()).strip()


def strip_keys(value):
    """Drop the fields that carry a name KEY (not a name): the expected key-text pin move."""
    if isinstance(value, dict):
        return {k: strip_keys(v) for k, v in value.items()
                if k not in {"core_terms", "pathway_context_core", "requested_core", "decision_input_hash",
                             "decision_inputs", "decision_id", "matched_terms", "unmatched_terms"}}
    if isinstance(value, list):
        return [strip_keys(v) for v in value]
    return value


try:
    from t2pw.pipeline.name_keys import reference_key
except ImportError:
    reference_key = None

if reference_key:
    files = subprocess.run(["git", "-C", str(ROOT), "ls-files"], capture_output=True, text=True).stdout.split()
    legs = [f for f in files if re.search(r"final_mapped\.json(\.gz)?$", f)]
    audit = []
    for leg in legs:
        p = ROOT / leg
        try:
            data = json.loads(gzip.open(p, "rt", encoding="utf-8").read() if leg.endswith(".gz") else p.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        groups = {}
        for b in BUCKETS:
            for r in (data.get("entities") or {}).get(b) or []:
                if isinstance(r, dict) and r.get("name"):
                    groups.setdefault(reference_key(r["name"]), []).append((b, str(r["name"])))
        for key, members in sorted(groups.items()):
            if len(members) > 1 and len({old_ir(n) for _b, n in members}) > 1:
                audit.append({"leg": leg, "key": key, "members": members})
    out["corpus_audit"] = {"legs": len(legs), "merged_groups": audit}
    print(f"corpus audit: {len(legs)} committed legs, {len(audit)} groups merged by the new key and not the old")
    for a in audit:
        print("  MERGED", a["leg"], a["members"])

import test_orch766_unreferenced_species as T766  # noqa: E402
from t2pw.pipeline.strict_quarantine import quarantine_and_close  # noqa: E402

o766 = {}
for kind in T766.ALL_KINDS:
    try:
        r = quarantine_and_close(T766._kind_payload(kind), strict_db=True, pathway_context=T766._context(kind),
                                 mode="pathwhiz")
        o766[kind] = {"payload": sha(r.payload), "removed": sha(r.removed_entity_report), "closure": sha(r.closure_report),
                      "quarantine": sha(r.quarantine_report), "quarantine_minus_keys": sha(strip_keys(r.quarantine_report)),
                      "admissions": [(a.get("bucket"), a.get("index"), a.get("reason")) for a in r.quarantine_report.get("admissions") or []]}
    except Exception as exc:  # noqa: BLE001
        o766[kind] = {"error": f"{type(exc).__name__}: {exc}"[:200]}
out["orch766"] = o766
Path(args.out).write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
print("written", args.out)
