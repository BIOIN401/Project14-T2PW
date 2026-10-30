"""ORCH-797a: offline tail replay (quarantine through IR, export and coverage) of the six ORCH-796 legs.

Usage: python orch797a_replay.py --src <tree>/src --out <scratch dir>

Each leg's OWN files, unmodified (no rebuilt rows, no respelling: the ORCH-794 lesson): the ORCH-768 harness
``replay_leg`` (imported; socket block installed) reads the leg's ``final_mapped.json`` as the canonical
checkpoint and runs the tail with the measured tree. Read-only on the legs; writes only under --out.
"""

import argparse
import importlib.util
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
HARNESS = HERE / "orch768_replay_corpus.py"
LEGS = Path(r"C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW/.claude/worktrees/orch796run/runs_papers/ORCH-796/legs")
STAMPS = {"chlorophyll": "2026-10-29_1632", "wnt_ca": "2026-10-29_1941", "wnt_pcp": "2026-10-29_1729",
          "sour_taste_v2": "2026-10-29_1314", "enterobactin": "2026-10-29_1153", "menaquinone": "2026-10-29_1451"}


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
    out = Path(args.out)
    for paper, stamp in STAMPS.items():
        legdir = next((LEGS / f"{paper}-strict" / stamp / "papers").glob("*/strict"))
        inputs = {n: {"path": n, "sha256": H.sha256_file(legdir / n)} for n in H.LEG_FILES if (legdir / n).is_file()}
        for pwml in sorted(legdir.glob("*.pwml")):
            inputs[pwml.name] = {"path": pwml.name, "sha256": H.sha256_file(pwml)}
        leg = {"id": f"ORCH-796/{paper}-strict/{stamp}", "paper": legdir.parent.name, "mode": "strict", "inputs": inputs}
        print(f"\n===== {paper} {stamp} ({legdir.parent.name})")
        try:
            r = H.replay_leg(T, leg, out / paper, primary=legdir)
        except Exception as exc:  # noqa: BLE001 -- a crash is a result here, recorded per leg
            print(f"REPLAY CRASH: {type(exc).__name__}: {str(exc)[:300]}")
            continue
        for key in ("checkpoint", "quarantine", "export", "pwml_written", "pwml_matches_leg", "failure_stage",
                    "network_refused"):
            print(f"{key}: {json.dumps(r.get(key), default=str)}")
        cov = r.get("pwml_coverage") or {}
        print("coverage:", json.dumps({k: cov.get(k) for k in ("drawn", "total", "by_design_withheld", "pwml_written",
                                                               "no_pwml_cause")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
