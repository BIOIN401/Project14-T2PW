"""ORCH-754b repair round 2 -- does Stage 6's report put anything new into the curator's prompt?

The interactive curator's prompt carries ``interactive_curator.compact_mapping_misses(mapping_report)``
under "Mapping Misses" (``interactive_curator.py:270``; the app at ``streamlit_app.py:1884``).
The Stage-6 wrapper step's report reaches it twice in a real ``map_payload`` report: its summary is
merged into ``report["summary"]`` and the whole step report sits at
``report["enzyme_complex_conversion"]``. For every committed ``final_mapped.json`` with
interactions, this re-runs the step offline (``db=None``, cache disabled -- the same replay as
``orch754b_wrapper_base_proof.py``), builds that two-place shape from its report, and compares
the Mapping Misses across three trees:

* ``base`` -- ``f9892cdf``;
* ``r1``   -- ``9da8358e``, the round-1 tip the round-2 review rejected (expected to differ:
  its per-row counter);
* ``tip``  -- this worktree.

It also counts, per tree, the interaction endpoints the step rewrote (from the payload, not
from any report key), so a 0 below is "nothing but the wrapper", not "nothing was wrapped".
"""

from __future__ import annotations

import hashlib
import io
import json
import logging
import os
import subprocess
import sys
import tarfile
import tempfile
from copy import deepcopy
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent / "orch754b_mapping_misses_replay.json"


def _child(tree: Path) -> int:
    logging.disable(logging.CRITICAL)
    import t2pw
    from t2pw.curation.interactive_curator import compact_mapping_misses
    from t2pw.mapping.map_ids import MappingCache, _extract_global_organism, _rewrite_reaction_protein_enzymes_to_complexes

    assert (tree / "src" / "t2pw").resolve() in Path(t2pw.__file__).resolve().parents, t2pw.__file__
    legs = sorted(p for p in subprocess.run(["git", "ls-files", "runs", "runs_verify"], cwd=REPO, capture_output=True,
                                            text=True, check=True).stdout.split("\n") if p.endswith("/final_mapped.json"))
    out = {}
    for leg in legs:
        payload = json.loads((REPO / leg).read_text(encoding="utf-8"))
        if not (payload.get("processes") or {}).get("interactions"):
            continue
        before = [(r.get("entity_1"), r.get("entity_2")) for r in payload["processes"]["interactions"] if isinstance(r, dict)]
        mapped = deepcopy(payload)
        report = _rewrite_reaction_protein_enzymes_to_complexes(
            mapped, db=None, cache=MappingCache(Path("unused.json"), enabled=False),
            global_organism=_extract_global_organism(mapped))
        after = [(r.get("entity_1"), r.get("entity_2")) for r in mapped["processes"]["interactions"] if isinstance(r, dict)]
        misses = compact_mapping_misses({"summary": report["summary"], "enzyme_complex_conversion": report})
        out[leg] = {
            "misses_sha256": hashlib.sha256(json.dumps(misses, sort_keys=True, default=str).encode()).hexdigest(),
            "misses_count": len(misses),
            "endpoints_rewritten": sum((a[0] != b[0]) + (a[1] != b[1]) for a, b in zip(after, before)),
        }
    print(json.dumps({"t2pw": str(Path(t2pw.__file__).resolve()), "legs": out}))
    return 0


def _archive(rev: str) -> Path:
    data = subprocess.run(["git", "archive", "--format=tar", rev, "src/t2pw"], cwd=REPO, capture_output=True).stdout
    tree = Path(tempfile.mkdtemp(prefix=f"orch754b_mm_{rev}_"))
    with tarfile.open(fileobj=io.BytesIO(data)) as handle:
        handle.extractall(tree)
    return tree


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
    runs = {"base": _run(_archive("f9892cdf")), "r1": _run(_archive("9da8358e")), "tip": _run(REPO)}
    legs = sorted(runs["base"]["legs"])
    differ = {name: [leg for leg in legs if runs[name]["legs"][leg]["misses_sha256"] != runs["base"]["legs"][leg]["misses_sha256"]]
              for name in ("r1", "tip")}
    record = {
        "trees": {k: v["t2pw"] for k, v in runs.items()},
        "legs": len(legs),
        "legs_where_mapping_misses_differ_from_base": {k: len(v) for k, v in differ.items()},
        "tip_endpoints_rewritten": sum(runs["tip"]["legs"][leg]["endpoints_rewritten"] for leg in legs),
        "tip_legs_with_a_rewrite": sum(1 for leg in legs if runs["tip"]["legs"][leg]["endpoints_rewritten"]),
        "differing_legs": differ,
        "per_tree": {k: v["legs"] for k, v in runs.items()},
    }
    OUT.write_text(json.dumps(record, indent=1), encoding="utf-8")
    print(json.dumps({k: record[k] for k in ("trees", "legs", "legs_where_mapping_misses_differ_from_base",
                                             "tip_endpoints_rewritten", "tip_legs_with_a_rewrite")}, indent=1))
    ok = not differ["tip"]
    print("VERDICT: " + ("Mapping Misses identical to base on every leg" if ok else "FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
