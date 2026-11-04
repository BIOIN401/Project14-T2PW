"""ORCH-802 no-loss scan: the substring cue matcher (base) against the boundary matcher (tip).

Two measurements over every committed run payload, plus any extra leg roots named on the
command line (read only):

A. PREDICATE DIFFERENTIAL. For every reaction row of every ``stage1_payload.json``,
   ``merged_payload.json`` and ``final_mapped.json``, and every protein / complex name the
   same payload declares, evaluate ``cue_near_name`` from the OLD and the NEW
   ``enzyme_cues.py`` (each loaded by file path; the module has no intra-package imports) on
   the attachers' haystack (reaction name + " " + evidence, evidence <= 400 chars). Every
   pair that hits under OLD and not under NEW is listed -- that is the complete set of cue
   decisions the fix can change. ``new hit and not old hit`` is impossible by construction and
   is counted as a check.

B. PRODUCTION REPLAY. On a deep copy of every ``stage1_payload.json`` run the production
   ``pipeline._inject_name_based_modifiers`` and then
   ``process_normalizer.attach_enzymes_from_reaction_evidence``, once with the OLD matcher
   patched in and once with the NEW, and diff the reaction actor rows. This catches both
   directions: an injection the fix drops, and one it newly enables (a real actor that was
   blocked by the exactly-one test only because a spurious second name also matched).

Usage (through bounded_run.py, cwd = the tree under measurement):
    python orch802_cue_noloss_scan.py --old <base>/src/t2pw/pipeline/enzyme_cues.py
        --new <tip>/src/t2pw/pipeline/enzyme_cues.py --repo <tip> --out <json>
        [--extra-root <dir> ...]
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

KINDS = ("stage1_payload.json", "merged_payload.json", "final_mapped.json")


def load_module(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def reactions(payload: Any) -> List[Dict[str, Any]]:
    if not isinstance(payload, dict):
        return []
    rows = (payload.get("processes") or {}).get("reactions")
    if not isinstance(rows, list):
        rows = payload.get("reactions")
    return [r for r in rows or [] if isinstance(r, dict)]


def actor_names(payload: Dict[str, Any]) -> List[str]:
    ents = payload.get("entities") or {}
    out: List[str] = []
    for key in ("protein_complexes", "proteins"):
        for row in ents.get(key) or []:
            if isinstance(row, dict) and isinstance(row.get("name"), str) and row["name"].strip():
                if row["name"] not in out:
                    out.append(row["name"])
    return out


def row_actor_names(row: Dict[str, Any]) -> List[str]:
    return [
        str(item.get(k)).strip()
        for bucket in ("enzymes", "modifiers")
        for item in row.get(bucket) or []
        if isinstance(item, dict)
        for k in ("entity", "protein", "protein_complex")
        if item.get(k)
    ]


def actors_view(payload: Dict[str, Any]) -> List[Any]:
    return [[r.get("name"), r.get("enzymes"), r.get("modifiers")] for r in reactions(payload)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--old", required=True)
    ap.add_argument("--new", required=True)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--extra-root", action="append", default=[])
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")  # snippets carry U+2032 etc.; cp1252 console crashed run 03

    old = load_module(args.old, "enzyme_cues_old")
    new = load_module(args.new, "enzyme_cues_new")
    repo = Path(args.repo)
    tracked = subprocess.run(["git", "-C", str(repo), "ls-files"], capture_output=True, text=True,
                             check=True).stdout.splitlines()
    files = [(repo / p, p, "committed") for p in tracked if p.rsplit("/", 1)[-1] in KINDS]
    for root in args.extra_root:
        for kind in KINDS:
            for p in sorted(Path(root).glob(f"**/{kind}")):
                files.append((p, p.as_posix(), "extra_untracked"))

    # ---- A. predicate differential ----
    pairs = old_hits = new_hits = impossible = 0
    flips: Dict[tuple, Dict[str, Any]] = {}
    for path, label, origin in files:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        names = actor_names(payload) if isinstance(payload, dict) else []
        for ridx, row in enumerate(reactions(payload)):
            ev = row.get("evidence")
            ev = ev if isinstance(ev, str) else ""
            hay = new.collapse_whitespace(f"{row.get('name') or ''} {ev}")
            if not hay or len(ev) > old.MAX_INJECTOR_EVIDENCE_CHARS:
                continue
            attached = {n.lower() for n in row_actor_names(row)}
            for name in names:
                pairs += 1
                o = old.cue_near_name(hay, name)
                n = new.cue_near_name(hay, name)
                old_hits += bool(o)
                new_hits += bool(n)
                if n and not o:
                    impossible += 1
                if o and not n:
                    key = (name, hay)
                    rec = flips.setdefault(key, {"actor": name, "haystack": hay, "old_snippet": o,
                                                 "occurrences": []})
                    inferred = [it for b in ("enzymes", "modifiers") for it in row.get(b) or []
                                if isinstance(it, dict) and it.get("provenance") == "inferred"
                                and name.lower() in {str(it.get(k) or "").strip().lower()
                                                     for k in ("entity", "protein", "protein_complex")}]
                    rec["occurrences"].append({"file": label, "origin": origin, "reaction_index": ridx,
                                               "actor_attached_on_row": name.lower() in attached,
                                               "actor_attached_inferred_on_row": bool(inferred)})

    # ---- B. production replay, old vs new matcher ----
    sys.path.insert(0, str(repo / "src"))
    from t2pw.pipeline import pipeline as P  # noqa: E402
    from t2pw.pipeline import process_normalizer as N  # noqa: E402

    def replay(payload: Dict[str, Any], fn) -> Dict[str, Any]:
        P.cue_near_name, N._cue_near_name = fn, fn
        out = deepcopy(payload)
        P._inject_name_based_modifiers(out)
        N.attach_enzymes_from_reaction_evidence(out)
        return out

    legs = diffs = 0
    replay_diffs: List[Dict[str, Any]] = []
    for path, label, origin in files:
        if not label.endswith("stage1_payload.json"):
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        if not isinstance(payload, dict):
            continue
        legs += 1
        a, b = replay(payload, old.cue_near_name), replay(payload, new.cue_near_name)
        for ridx, (ra, rb) in enumerate(zip(reactions(a), reactions(b))):
            if [ra.get("enzymes"), ra.get("modifiers")] != [rb.get("enzymes"), rb.get("modifiers")]:
                diffs += 1
                na, nb = row_actor_names(ra), row_actor_names(rb)
                replay_diffs.append({"file": label, "origin": origin, "reaction_index": ridx,
                                     "reaction_name": ra.get("name"), "evidence": ra.get("evidence"),
                                     "actors_old": na, "actors_new": nb,
                                     "dropped": [x for x in na if x not in nb],
                                     "added": [x for x in nb if x not in na]})
    P.cue_near_name = N._cue_near_name = new.cue_near_name

    report = {
        "task": "ORCH-802",
        "files_scanned": {"committed": sum(1 for f in files if f[2] == "committed"),
                          "extra_untracked": sum(1 for f in files if f[2] == "extra_untracked")},
        "A_predicate": {"pairs_evaluated": pairs, "old_hits": old_hits, "new_hits": new_hits,
                        "new_hit_without_old_hit": impossible, "distinct_flips": len(flips),
                        "flips": sorted(flips.values(), key=lambda r: (r["actor"], r["haystack"]))},
        "B_replay": {"stage1_legs": legs, "reaction_rows_differing": diffs, "diffs": replay_diffs},
    }
    Path(args.out).write_text(json.dumps(report, indent=1, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    print(json.dumps({k: (v if k != "A_predicate" else {x: y for x, y in v.items() if x != "flips"})
                      for k, v in report.items() if k != "B_replay"}, sort_keys=True))
    print(json.dumps({k: v for k, v in report["B_replay"].items() if k != "diffs"}))
    for f in report["A_predicate"]["flips"]:
        occ = f["occurrences"]
        print(f"FLIP actor={f['actor']!r} n={len(occ)} inferred_on_row={sum(o['actor_attached_inferred_on_row'] for o in occ)}"
              f" snippet={f['old_snippet'][:160]!r}")
    for d in replay_diffs:
        print(f"REPLAY {d['file']} r{d['reaction_index']} dropped={d['dropped']} added={d['added']} name={d['reaction_name']!r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
