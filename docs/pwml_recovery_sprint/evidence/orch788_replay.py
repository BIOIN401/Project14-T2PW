"""ORCH-788: offline replay of the ORCH-784 sour_taste_v2 strict leg (IR: state with no location).

Usage: python orch788_replay.py --src <tree>/src --out <scratch dir>

Reuses the ORCH-768 harness's ``replay_leg`` (imported, not copied; it installs the socket block) on the
leg's saved artifacts, strict mode, then renders the measured tree's own Reaction Evidence text and prints
its "OMITTED FROM THE PWML" section. Read-only on the leg; writes only under --out.
"""

import argparse
import importlib.util
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
HARNESS = HERE / "orch768_replay_corpus.py"
LEG = Path(r"C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW/.claude/worktrees/orch784run/runs_papers/ORCH-784/legs/"
           r"sour_taste_v2-strict/2026-10-27_2155/papers/PMC7059786/strict")


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
    leg = {"id": "ORCH-784/sour_taste_v2-strict", "paper": "PMC7059786", "mode": "strict", "inputs": inputs}
    work = Path(args.out)
    r = H.replay_leg(T, leg, work)
    for key in ("checkpoint", "quarantine", "driver", "export", "pwml_written", "failure_stage", "network_refused"):
        print(f"{key}: {json.dumps(r.get(key), default=str)}")
    cov = r.get("pwml_coverage") or {}
    print("coverage:", json.dumps({k: cov.get(k) for k in ("drawn", "total", "pwml_written", "no_pwml_cause")}))
    arts = {}
    for name in ("quarantine_report.json", "removed_entity_report.json", "coverage_summary.json"):
        hits = sorted(work.rglob(name))
        if hits:
            arts[name] = json.loads(hits[-1].read_text(encoding="utf-8"))
    arts["final_mapped.json"] = json.loads((work / "canonical.json").read_text(encoding="utf-8"))
    if r.get("pwml_written"):
        arts["pwml_ir.json"] = json.loads((work / "pwml_ir.json").read_text(encoding="utf-8"))
        arts[r["export"]["pwml_name"]] = True
    text = T.evidence.render_text(T.evidence.build_report(arts, no_pwml_cause=H.replay_no_pwml_cause(r)))
    start = text.find("OMITTED FROM THE PWML")
    end = text.find("CORE COVERAGE")
    print("---- Reaction Evidence text, OMITTED section ----")
    print(text[max(0, start - 80):end if end > start else None])
    return 0


if __name__ == "__main__":
    sys.exit(main())
