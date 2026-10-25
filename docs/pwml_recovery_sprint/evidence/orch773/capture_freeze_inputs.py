"""ORCH-773: capture the exact payload each corpus leg hands to ``freeze_canonical_payload``.

Usage: python capture_freeze_inputs.py --src <tree>/src --out <dir> [--legs <id> ...]

Runs the ORCH-768 harness's own ``replay_leg`` (imported, not copied) with ``freeze_canonical_payload``
wrapped so that its input payload and pathway_context are written to ``<dir>/<slug>/freeze_input.json``
before the real function runs. Offline (the harness installs its socket block). Writes only under <dir>.
"""

import argparse
import importlib.util
import json
import re
import sys
from copy import deepcopy
from pathlib import Path

HERE = Path(__file__).resolve().parent
HARNESS = HERE.parent / "orch768_replay_corpus.py"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--legs", nargs="*")
    args = ap.parse_args()
    spec = importlib.util.spec_from_file_location("orch768_replay_corpus", HARNESS)
    H = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(H)
    T = H.load_tree(Path(args.src))
    print("t2pw resolved:", T.t2pw)
    print("offline probe:", H.probe_offline())
    real = T.app.freeze_canonical_payload
    current = {}

    def wrapped(payload, **kw):
        current["dir"].mkdir(parents=True, exist_ok=True)
        (current["dir"] / "freeze_input.json").write_text(
            json.dumps({"payload": deepcopy(payload), "pathway_context": kw.get("pathway_context")},
                       indent=1, sort_keys=True, ensure_ascii=False, default=str), encoding="utf-8")
        return real(payload, **kw)

    T.app.freeze_canonical_payload = wrapped
    manifest = json.loads(H.MANIFEST.read_text(encoding="utf-8"))
    wanted = set(args.legs or [])
    n = 0
    for leg in manifest["legs"]:
        if (wanted and leg["id"] not in wanted) or leg["mode"] not in (None, "strict") or not leg["checkpoint"]:
            continue
        slug = re.sub(r"[^A-Za-z0-9._-]+", "_", leg["id"])
        work = Path(args.out) / slug
        current["dir"] = work
        r = H.replay_leg(T, leg, work)
        n += 1
        print(f"{leg['id']:58s} pwml={r['pwml_written']} stage={r['failure_stage']}")
    print(f"captured {n} legs; network refused total {len(H.Offline.refused)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
