"""ORCH-802 reviewer F3: replay both cue attachers on every committed merged_payload.json,
old matcher vs new, after stripping the rows the attachers themselves wrote.

Why merged, not stage1: production runs ``_inject_name_based_modifiers`` on the merged payload
(pipeline.py:1260) and ``attach_enzymes_from_reaction_evidence`` at Stage 5, both with more
declared entities than Stage 1. Under the exactly-one gates, removing a spurious second
qualifier can newly ENABLE an injection; this measures that direction too.

WHICH ROWS ARE STRIPPED (exact, from code, no guessing):
* the injector writes (pipeline.py:3016) a ``modifiers`` row with ``confidence: 0.9``,
  ``provenance: "inferred"``, ``evidence: <cue window>``. ``apply_post_merge_cleanup`` ->
  ``_normalize_reaction_actors`` -> ``_clean_enzymes`` (pipeline.py:815/1858) then folds it into
  ``enzymes`` as ``{protein|protein_complex, evidence, provenance, confidence}`` -- the shape that
  reaches merged_payload.json. A row is stripped as an injector row when it is in
  ``enzymes``/``modifiers``, has provenance ``inferred``, confidence 0.9, and its evidence is
  EXACTLY the old ``cue_near_name(reaction name + " " + evidence, actor)`` window, or EXACTLY
  ``evidence[:120]`` (the pre-hardening injector's slice, run 2026-07-28_0919).
* the normalizer writes (process_normalizer.py:3738) an ``enzymes`` row with confidence 0.75,
  provenance ``inferred``, evidence = the cue window. Same exact-evidence test.
Rows written before ``_clean_enzymes`` kept provenance/confidence (2026-07-29) carry no carrier
and cannot be told apart from extractions; they are not stripped and are counted as such.

Self-check: the OLD-matcher replay should re-create stripped rows; the re-creation rate is
reported.

Usage: python orch802_merged_replay.py --old <base enzyme_cues.py> --new <tip enzyme_cues.py>
       --repo <tip> --out <json> [--extra-root <dir> ...]
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List


def load_module(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def reactions(payload: Any) -> List[Dict[str, Any]]:
    rows = ((payload.get("processes") or {}).get("reactions")) if isinstance(payload, dict) else None
    return [r for r in rows or [] if isinstance(r, dict)]


def actor_name(item: Dict[str, Any]) -> str:
    return str(item.get("entity") or item.get("protein") or item.get("protein_complex") or "").strip()


def row_actors(row: Dict[str, Any]) -> List[str]:
    return [actor_name(i) for b in ("enzymes", "modifiers") for i in row.get(b) or []
            if isinstance(i, dict) and actor_name(i)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--old", required=True)
    ap.add_argument("--new", required=True)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--extra-root", action="append", default=[])
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    old = load_module(args.old, "enzyme_cues_old")
    new = load_module(args.new, "enzyme_cues_new")
    repo = Path(args.repo)
    sys.path.insert(0, str(repo / "src"))
    from t2pw.pipeline import pipeline as P  # noqa: E402
    from t2pw.pipeline import process_normalizer as N  # noqa: E402

    tracked = subprocess.run(["git", "-C", str(repo), "ls-files"], capture_output=True, text=True,
                             check=True).stdout.splitlines()
    files = [(repo / p, p, "committed") for p in tracked if p.endswith("/merged_payload.json")]
    for root in args.extra_root:
        files += [(p, p.as_posix(), "extra_untracked") for p in sorted(Path(root).glob("**/merged_payload.json"))]

    def is_attacher_row(item: Any, row: Dict[str, Any]) -> str:
        if not isinstance(item, dict) or item.get("provenance") != "inferred":
            return ""
        name, ev = actor_name(item), item.get("evidence")
        if not name or not isinstance(ev, str):
            return ""
        rev = row.get("evidence") if isinstance(row.get("evidence"), str) else ""
        hay = old.collapse_whitespace(f"{row.get('name') or ''} {rev}")
        window = old.cue_near_name(hay, name) if len(rev) <= old.MAX_INJECTOR_EVIDENCE_CHARS else None
        if item.get("confidence") == 0.9 and (ev == window or (rev and ev == rev[:120])):
            return "injector"
        if item.get("confidence") == 0.75 and ev == window:
            return "normalizer"
        return ""

    def replay(payload: Dict[str, Any], fn) -> Dict[str, Any]:
        P.cue_near_name, N._cue_near_name = fn, fn
        out = deepcopy(payload)
        P._inject_name_based_modifiers(out)
        N.attach_enzymes_from_reaction_evidence(out)
        return out

    legs = stripped_total = recreated = no_carrier = 0
    stripped_by: Dict[str, int] = {"injector": 0, "normalizer": 0}
    diffs: List[Dict[str, Any]] = []
    for path, label, origin in files:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        if not isinstance(payload, dict) or not reactions(payload):
            continue
        legs += 1
        base = deepcopy(payload)
        stripped: List[tuple] = []
        for ridx, row in enumerate(reactions(base)):
            for bucket in ("enzymes", "modifiers"):
                items = row.get(bucket)
                if not isinstance(items, list):
                    continue
                keep = []
                for item in items:
                    kind = is_attacher_row(item, row)
                    if kind:
                        stripped.append((ridx, actor_name(item).lower()))
                        stripped_by[kind] += 1
                    else:
                        if isinstance(item, dict) and "provenance" not in item and "confidence" not in item:
                            no_carrier += 1
                        keep.append(item)
                row[bucket] = keep
        stripped_total += len(stripped)
        a, b = replay(base, old.cue_near_name), replay(base, new.cue_near_name)
        ra_all, rb_all = reactions(a), reactions(b)
        for ridx, name in set(stripped):
            if ridx < len(ra_all) and name in {x.lower() for x in row_actors(ra_all[ridx])}:
                recreated += 1
        for ridx, (ra, rb) in enumerate(zip(ra_all, rb_all)):
            na, nb = row_actors(ra), row_actors(rb)
            if na != nb:
                rev = ra.get("evidence") if isinstance(ra.get("evidence"), str) else ""
                hay = new.collapse_whitespace(f"{ra.get('name') or ''} {rev}")
                added = [x for x in nb if x not in na]
                diffs.append({
                    "file": label, "origin": origin, "reaction_index": ridx,
                    "reaction_name": ra.get("name"), "evidence": rev[:400],
                    "actors_old": na, "actors_new": nb,
                    "dropped": [x for x in na if x not in nb], "added": added,
                    "added_new_windows": {x: new.cue_near_name(hay, x) for x in added},
                    "dropped_old_windows": {x: old.cue_near_name(hay, x) for x in na if x not in nb},
                })
    P.cue_near_name = N._cue_near_name = new.cue_near_name

    report = {
        "task": "ORCH-802", "purpose": "reviewer F3: merged-payload replay, attacher rows stripped",
        "files": {"committed": sum(f[2] == "committed" for f in files),
                  "extra_untracked": sum(f[2] == "extra_untracked" for f in files)},
        "legs_replayed": legs, "attacher_rows_stripped": stripped_total, "stripped_by": stripped_by,
        "stripped_recreated_by_old_replay": recreated, "actor_rows_without_carrier_not_stripped": no_carrier,
        "reaction_rows_differing": len(diffs),
        "rows_with_drops": sum(bool(d["dropped"]) for d in diffs),
        "rows_with_additions": sum(bool(d["added"]) for d in diffs),
        "diffs": diffs,
    }
    Path(args.out).write_text(json.dumps(report, indent=1, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "diffs"}, sort_keys=True))
    for d in diffs:
        print(f"DIFF {d['file']} r{d['reaction_index']} dropped={d['dropped']} added={d['added']} "
              f"name={d['reaction_name']!r}")
        for x, w in d["added_new_windows"].items():
            print(f"   ADDED {x!r} window={w!r}")
        for x, w in d["dropped_old_windows"].items():
            print(f"   DROPPED {x!r} old_window={(w or '')[:200]!r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
