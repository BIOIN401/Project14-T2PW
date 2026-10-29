"""ORCH-795: the PWML coverage headline on the six ORCH-784/785 strict legs, by the measured tree.

Usage: python orch795_coverage.py --src <tree>/src --out <scratch dir>

Two readings per leg, both by the measured tree's ``reaction_evidence``:
* ``record``: ``build_report`` on the leg directory's own artifacts (what its ``reaction_evidence.txt`` says);
* ``replay``: the ORCH-768 harness's ``replay_leg`` (imported; it installs the socket block), whose
  ``pwml_coverage`` the corpus reads.
Read-only on the legs; writes only under --out.
"""

import argparse
import importlib.util
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
HARNESS = HERE / "orch768_replay_corpus.py"
W = Path(r"C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW/.claude/worktrees")
LEGS = {
    "chlorophyll": ("ORCH-785", W / "orch785run/runs_papers/ORCH-785/legs/chlorophyll-strict/2026-10-28_1743/papers/PMC5130987/strict"),
    "wnt_ca": ("ORCH-785", W / "orch785run/runs_papers/ORCH-785/legs/wnt_ca-strict/2026-10-28_1910/papers/PMC10425814/strict"),
    "sour_taste_v2": ("ORCH-785", W / "orch785run/runs_papers/ORCH-785/legs/sour_taste_v2-strict/2026-10-28_2058/papers/PMC7059786/strict"),
    "enterobactin": ("ORCH-784", W / "orch784run/runs_papers/ORCH-784/legs/enterobactin-strict/2026-10-27_1931/papers/PMC12096016/strict"),
    "menaquinone": ("ORCH-784", W / "orch784run/runs_papers/ORCH-784/legs/menaquinone-strict/2026-10-28_0208/papers/PMC12421875/strict"),
    "wnt_pcp": ("ORCH-784", W / "orch784run/runs_papers/ORCH-784/legs/wnt_pcp-strict/2026-10-27_2309/papers/PMC10425814/strict"),
}


def fmt(cov):
    head = cov["total"] - cov["by_design_withheld"]
    pct = lambda n, d: f"{100.0 * n / d:.1f}%" if d else "n/a"  # noqa: E731
    return (f"headline {cov['drawn']}/{head} ({pct(cov['drawn'], head)}) | full {cov['drawn']}/{cov['total']} "
            f"({pct(cov['drawn'], cov['total'])}) | excluded from headline {cov['by_design_withheld']}")


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
    for tag, (run, leg_dir) in LEGS.items():
        record = T.evidence.build_report(T.evidence.load_run_artifacts(leg_dir))["pwml_coverage"]
        inputs = {name: {"path": (leg_dir / name).relative_to(H.PRIMARY).as_posix(), "sha256": H.sha256_file(leg_dir / name)}
                  for name in H.LEG_FILES if (leg_dir / name).is_file()}
        leg = {"id": f"{run}/{tag}-strict", "paper": leg_dir.parent.name, "mode": "strict", "inputs": inputs}
        r = H.replay_leg(T, leg, Path(args.out) / tag)
        cov = r["pwml_coverage"]
        print(f"{tag:14s} [{run}] record: {fmt(record)}")
        print(f"{'':14s} replay ({r['checkpoint']}, pwml_written={r['pwml_written']}, network_refused={len(r['network_refused'])}): "
              f"{fmt(cov)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
