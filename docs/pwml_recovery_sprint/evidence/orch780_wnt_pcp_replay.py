"""ORCH-780 (rebased onto 55fd55ed): offline tail replay of the ORCH-796 wnt_pcp strict leg.

Usage: python orch780_wnt_pcp_replay.py --src <tree>/src --out <scratch dir>

The leg wrote no PWML: required-field gate ``no_defensible_reaction_support`` (none of its 2 reactions has reaction-level
support). The ORCH-768 harness's ``replay_leg`` (imported; socket block installed) replays the leg's saved artifacts,
strict mode, with the measured tree. Read-only on the leg; writes only under --out.
"""

import argparse
import importlib.util
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
HARNESS = HERE / "orch768_replay_corpus.py"
LEG = Path(r"C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW/.claude/worktrees/orch796run/runs_papers/ORCH-796/legs/"
           r"wnt_pcp-strict/2026-10-29_1350/papers/PMC10425814/strict")


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
    inputs = {name: {"path": (LEG / name).relative_to(H.PRIMARY).as_posix(), "sha256": H.sha256_file(LEG / name)}
              for name in H.LEG_FILES if (LEG / name).is_file()}
    leg = {"id": "ORCH-796/wnt_pcp-strict", "paper": "PMC10425814", "mode": "strict", "inputs": inputs}
    work = Path(args.out)
    r = H.replay_leg(T, leg, work)
    for key in ("checkpoint", "quarantine", "export", "pwml_written", "failure_stage", "network_refused"):
        print(f"{key}: {json.dumps(r.get(key), default=str)}")
    cov = r.get("pwml_coverage") or {}
    print("coverage:", json.dumps({k: cov.get(k) for k in ("drawn", "total", "by_design_withheld", "pwml_written",
                                                         "no_pwml_cause", "by_kind")}))
    hits = sorted(work.rglob("quarantine_report.json"))
    if hits:
        q = json.loads(hits[-1].read_text(encoding="utf-8"))
        print("omitted by the replayed quarantine:", json.dumps([(x.get("name"), x.get("reason")) for x in q.get("quarantined") or []]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
