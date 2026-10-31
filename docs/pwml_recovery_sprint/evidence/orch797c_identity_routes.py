"""ORCH-797c: the Stage-6 decisions the name-set fold feeds, on the six ORCH-796 legs (run per tree, diff the outputs).

Usage: python orch797c_identity_routes.py --src <tree>/src --out <file.json>
Per leg (its own merged_payload.json and final_mapped.json, read-only):
  * every compound row's route, as ``map_payload`` asks it (``_collect_protein_like_names`` +
    ``route_entity_for_mapping(name, "compound", protein_like_names=...)``);
  * ``_apply_pathbank_unknown_enzyme_fallback`` on a copy of final_mapped.json: its summary and the sha256 of its
    output (the enzyme/transporter role check behind it reads ``compound_name_block_rule(compound_names=...)``).
No network: neither function calls out.
"""

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--src", required=True)
ap.add_argument("--out", required=True)
args = ap.parse_args()
sys.path.insert(0, str(Path(args.src).resolve()))
import t2pw  # noqa: E402
from t2pw.mapping import map_ids as M  # noqa: E402
from t2pw.pipeline.entity_identity import route_entity_for_mapping  # noqa: E402

print("t2pw resolved:", Path(t2pw.__file__).resolve().parent)
LEGS = Path(r"C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW/.claude/worktrees/orch796run/runs_papers/ORCH-796/legs")
STAMPS = {"chlorophyll": "2026-10-29_1632", "wnt_ca": "2026-10-29_1941", "wnt_pcp": "2026-10-29_1729",
          "sour_taste_v2": "2026-10-29_1314", "enterobactin": "2026-10-29_1153", "menaquinone": "2026-10-29_1451"}
out = {}
for paper, stamp in STAMPS.items():
    legdir = next((LEGS / f"{paper}-strict" / stamp / "papers").glob("*/strict"))
    record = {}
    for fname in ("merged_payload.json", "final_mapped.json"):
        payload = json.loads((legdir / fname).read_text(encoding="utf-8"))
        protein_like = M._collect_protein_like_names(payload)
        record[f"routes:{fname}"] = {
            str(row.get("name")): route_entity_for_mapping(str(row.get("name")), "compound", protein_like_names=protein_like)
            for row in (payload.get("entities") or {}).get("compounds") or [] if isinstance(row, dict) and row.get("name")}
    final = json.loads((legdir / "final_mapped.json").read_text(encoding="utf-8"))
    staged = copy.deepcopy(final)
    record["fallback_summary"] = M._apply_pathbank_unknown_enzyme_fallback(staged)  # mutates staged in place
    record["fallback_sha256"] = hashlib.sha256(json.dumps(staged, sort_keys=True, default=str).encode()).hexdigest()
    out[paper] = record
    print(paper, {k: len(v) for k, v in record.items() if isinstance(v, dict)}, record["fallback_sha256"][:16])
Path(args.out).write_text(json.dumps(out, indent=1, sort_keys=True, default=str), encoding="utf-8")
print("written", args.out)
# Wider: every committed final_mapped.json in this worktree (same two measurements).
import gzip  # noqa: E402
import subprocess  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
files = [f for f in subprocess.run(["git", "-C", str(ROOT), "ls-files"], capture_output=True, text=True).stdout.split()
         if f.endswith("final_mapped.json") or f.endswith("final_mapped.json.gz")]
for f in files:
    p = ROOT / f
    try:
        payload = json.loads(gzip.open(p, "rt", encoding="utf-8").read() if f.endswith(".gz") else p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        continue
    if not isinstance(payload, dict):
        continue
    protein_like = M._collect_protein_like_names(payload)
    record = {"routes": {str(row.get("name")): route_entity_for_mapping(str(row.get("name")), "compound",
                                                                         protein_like_names=protein_like)
                         for row in (payload.get("entities") or {}).get("compounds") or []
                         if isinstance(row, dict) and row.get("name")}}
    staged = copy.deepcopy(payload)
    try:
        record["fallback_summary"] = M._apply_pathbank_unknown_enzyme_fallback(staged)
        record["fallback_sha256"] = hashlib.sha256(json.dumps(staged, sort_keys=True, default=str).encode()).hexdigest()
    except Exception as exc:  # noqa: BLE001
        record["fallback_error"] = f"{type(exc).__name__}: {exc}"[:200]
    out[f] = record
print("corpus legs:", len(files))
Path(args.out).write_text(json.dumps(out, indent=1, sort_keys=True, default=str), encoding="utf-8")
print("written (with corpus)", args.out)
