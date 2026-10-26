"""ORCH-773 phase 2: the Reaction Evidence .txt of chosen ORCH-768 corpus legs, replayed at ``--src``.

The report rendered is the one the harness's own ``pwml_coverage`` builds (``build_report`` is wrapped,
not re-implemented), so the .txt and the corpus coverage figure come from the same artifact set.

    python render_leg_txt.py --src <tree>/src --out <scratch> --txt-dir <dir> <leg id> [<leg id> ...]
"""

from __future__ import annotations

import argparse
import importlib.util
import re
import sys
from pathlib import Path

HARNESS = Path(__file__).resolve().parents[1] / "orch768_replay_corpus.py"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--txt-dir", required=True)
    ap.add_argument("legs", nargs="+")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    spec = importlib.util.spec_from_file_location("orch768_replay_corpus_773", HARNESS)
    H = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(H)
    T = H.load_tree(Path(args.src))
    try:
        probe = H.probe_offline()
        print(f"t2pw resolved: {T.t2pw}\noffline probe: {probe}")
        if not probe["ok"]:
            return 3
        real, captured = T.evidence, []

        class Capture:
            def __getattr__(self, name):
                return getattr(real, name)

            def build_report(self, arts):
                captured.append(real.build_report(arts))
                return captured[-1]

        T.evidence = Capture()
        legs = {leg["id"]: leg for leg in H.load(H.MANIFEST)["legs"]}
        txt_dir = Path(args.txt_dir)
        txt_dir.mkdir(parents=True, exist_ok=True)
        for leg_id in args.legs:
            captured.clear()
            slug = re.sub(r"[^A-Za-z0-9._-]+", "_", leg_id)
            record = H.replay_leg(T, legs[leg_id], Path(args.out) / slug)
            (report,) = captured
            text = real.render_text(report)
            (txt_dir / f"{slug}.txt").write_text(text, encoding="utf-8")
            cov = record["pwml_coverage"]
            print(f"\n##### {leg_id}  pwml_written={record['pwml_written']}  coverage={cov['drawn']}/{cov['total']}"
                  f"  -> {txt_dir / (slug + '.txt')}")
            print(text[text.index("OMITTED FROM THE PWML ("): text.index("\nCORE COVERAGE\n")])
    finally:
        H.Offline.uninstall()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
