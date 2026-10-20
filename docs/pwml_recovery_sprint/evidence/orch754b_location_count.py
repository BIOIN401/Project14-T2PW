"""ORCH-754b repair round 1 -- exactly which IR locations moved, c9265fb8 -> tip, on the two legs
the cumulative re-pin found with ``/ir/locations/[]/x|y`` changes (correcting RESULT § 6).

Configuration B of the GOLDEN sweep (DB down, no name index, strict), the R3 test's own.
"""

from __future__ import annotations

import io
import json
import os
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
LEGS = [
    "runs_verify/2026-08-04_1754/papers/PMC12856317/strict/final_mapped.json",
    "runs_verify/2026-08-21_2239/papers/PMC12856317/research/final_mapped.json",
]


class _DownDb:
    last_error = "harvest_db_down"

    def available(self) -> bool:
        return False


def _child(tree: Path) -> int:
    import t2pw
    from t2pw.pwml.ir import build_pwml_ir
    from t2pw.pwml.prefreeze_resolution import run_prefreeze_resolution

    assert (tree / "src" / "t2pw").resolve() in Path(t2pw.__file__).resolve().parents, t2pw.__file__
    out = {}
    for leg in LEGS:
        payload = json.loads((REPO / leg).read_text(encoding="utf-8"))
        run_prefreeze_resolution(payload, strict_db=True, db_resolver=_DownDb(), name_index=None)
        ir, _ = build_pwml_ir(payload, db_resolver=_DownDb(), strict_db=True, name_index=None)
        names = {e["key"]: e.get("name") for bucket in ir["entities"].values() for e in bucket if isinstance(e, dict)}
        out[leg] = [{**loc, "entity_name": names.get(loc.get("entity_key"))} for loc in ir["locations"]]
    print(json.dumps(out))
    return 0


def _run(tree: Path) -> dict:
    env = dict(os.environ, PYTHONPATH=str(tree / "src"), T2PW_OFFLINE_CURATOR="1", PYTHONIOENCODING="utf-8")
    res = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--child", str(tree)], cwd=str(tree), env=env,
                         capture_output=True, text=True, encoding="utf-8")
    if res.returncode:
        raise SystemExit(res.stderr)
    return json.loads(res.stdout.strip().splitlines()[-1])


def main() -> int:
    if len(sys.argv) == 3 and sys.argv[1] == "--child":
        return _child(Path(sys.argv[2]))
    tar = subprocess.run(["git", "archive", "--format=tar", "c9265fb8", "src/t2pw"], cwd=REPO, capture_output=True).stdout
    base_tree = Path(tempfile.mkdtemp(prefix="orch754b_loc_"))
    with tarfile.open(fileobj=io.BytesIO(tar)) as handle:
        handle.extractall(base_tree)
    base, tip = _run(base_tree), _run(REPO)
    for leg in LEGS:
        b = {(l["type"], l["entity_key"], l["biological_state_key"]): l for l in base[leg]}
        t = {(l["type"], l["entity_key"], l["biological_state_key"]): l for l in tip[leg]}
        moved = [(k, (b[k]["x"], b[k]["y"]), (t[k]["x"], t[k]["y"]), t[k]["entity_name"])
                 for k in b.keys() & t.keys() if (b[k]["x"], b[k]["y"]) != (t[k]["x"], t[k]["y"])]
        print(f"{leg}: base {len(b)} locations, tip {len(t)}; added {len(t.keys() - b.keys())}, "
              f"removed {len(b.keys() - t.keys())}, moved {len(moved)}")
        for key, old, new, name in sorted(moved):
            print(f"   moved  {name!r} {key[0]} {old} -> {new}")
        for key in sorted(t.keys() - b.keys()):
            print(f"   added  {t[key]['entity_name']!r} {key[0]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
