"""ORCH-794b: offline tail replay (quarantine through IR) of the ORCH-796 chlorophyll strict leg.

Usage: python orch794b_replay.py --src <tree>/src --out <scratch dir>

The leg's ``final_mapped.json`` is the payload the quarantine produced, so the two-kind decision has already been
applied to it: compound ``tRNA^Glu`` (and its compound location) were removed as
``same_name_declared_as_another_kind:*``. The quarantine input is rebuilt by putting back exactly the rows the
leg's ``removed_entity_report.json`` records under that reason at iteration 0 (that function's only removals), as
``{"name", "class"}`` (the removed compound had no identity: ``decided_by name_wording``). Everything else is the leg's
payload as written. ``final_stage3_gate_report.json`` gets that payload's hash, so the ORCH-768 harness
(``replay_leg``, imported; socket block installed) reads the scratch leg as a canonical checkpoint and runs the tail
with the measured tree. Read-only on the leg; writes only under --out.
"""

import argparse
import importlib.util
import json
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
HARNESS = HERE / "orch768_replay_corpus.py"
LEG = Path(r"C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW/.claude/worktrees/orch796run/runs_papers/ORCH-796/legs/"
           r"chlorophyll-strict/2026-10-29_1213/papers/PMC5130987/strict")
REASON = "same_name_declared_as_another_kind:"
SINGULAR = {"compounds": "compound", "proteins": "protein", "protein_complexes": "protein_complex",
            "nucleic_acids": "nucleic_acid", "element_collections": "element_collection"}
LOC_FIELD = {"compound_locations": "compound", "protein_locations": "protein",
             "nucleic_acid_locations": "nucleic_acid", "element_collection_locations": "element_collection"}


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    spec = importlib.util.spec_from_file_location("orch768_replay_corpus", HARNESS)
    H = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(H)
    T = H.load_tree(Path(args.src))
    print("t2pw resolved:", T.t2pw)
    print("offline probe:", H.probe_offline())
    payload = load(LEG / "final_mapped.json")
    rer = load(LEG / "removed_entity_report.json")
    for e in rer.get("removed_entities") or []:
        if str(e.get("reason", "")).startswith(REASON) and e.get("bucket") in SINGULAR:
            payload["entities"].setdefault(e["bucket"], []).append({"name": e["name"], "class": SINGULAR[e["bucket"]]})
            print(f"put back {e['bucket']} '{e['name']}' ({e['reason']}, decided_by {(e.get('detail') or {}).get('decided_by')})")
    for e in rer.get("removed_locations") or []:
        bucket = str(e.get("bucket", "")).split("/")[-1]
        if str(e.get("reason", "")).startswith(REASON) and bucket in LOC_FIELD:
            payload.setdefault("element_locations", {}).setdefault(bucket, []).append(
                {LOC_FIELD[bucket]: e["name"], "biological_state": e.get("biological_state")})
            print(f"put back location {bucket} '{e['name']}' @ '{e.get('biological_state')}'")
    out = Path(args.out)
    scratch = out / "inputs" / "leg"
    scratch.mkdir(parents=True, exist_ok=True)
    for name in H.LEG_FILES:
        if (LEG / name).is_file():
            shutil.copyfile(LEG / name, scratch / name)
    (scratch / "final_mapped.json").write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    grec = load(LEG / "final_stage3_gate_report.json")
    grec["payload_sha256"] = T.gr.payload_sha256(load(scratch / "final_mapped.json"))
    (scratch / "final_stage3_gate_report.json").write_text(json.dumps(grec), encoding="utf-8")
    inputs = {n: {"path": f"leg/{n}", "sha256": H.sha256_file(scratch / n)} for n in H.LEG_FILES if (scratch / n).is_file()}
    leg = {"id": "ORCH-796/chlorophyll-strict (ORCH-794b input rebuild)", "paper": "PMC5130987", "mode": "strict",
           "inputs": inputs}
    r = H.replay_leg(T, leg, out / "work", primary=out / "inputs")
    for key in ("checkpoint", "quarantine", "final_gate", "export", "pwml_written", "failure_stage", "network_refused"):
        print(f"{key}: {json.dumps(r.get(key), default=str)}")
    cov = r.get("pwml_coverage") or {}
    print("coverage:", json.dumps({k: cov.get(k) for k in ("drawn", "total", "by_design_withheld", "pwml_written", "no_pwml_cause")}))
    canonical = load(out / "work" / "canonical.json")
    hits = [(b, p.get("name"), [x for x in p.get("inputs", []) + p.get("outputs", []) if "trna" in str(x).casefold()])
            for b in ("reactions",) for p in (canonical.get("processes") or {}).get(b) or []
            if any("trna" in str(x).casefold() for x in p.get("inputs", []) + p.get("outputs", []))]
    print("tRNA reactions in the canonical payload:", json.dumps(hits))
    print("nucleic acids named tRNA:", [n.get("name") for n in (canonical.get("entities") or {}).get("nucleic_acids") or []
                                        if "trna" in str(n.get("name")).casefold()])
    return 0


if __name__ == "__main__":
    sys.exit(main())
