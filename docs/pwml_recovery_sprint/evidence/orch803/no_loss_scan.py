"""ORCH-803 no-loss scan: old vs new on every committed paper text and name-gate record.

    scan    --root <worktree> --out <json>    (run once with base src, once with tip src on PYTHONPATH)
    compare --base <json> --tip <json> --out <json>

(a) every abbreviation pair ``abbreviation_pairs`` finds in each distinct paper text;
(b) every committed ``name_gate`` record (entity name, compared names, kind, identity), replayed
    through ``_name_gate_verdict`` after a pooled Stage 1 over that paper's full source text, the
    way production registers it (``run_ensemble``; a no-op on base). The candidate is rebuilt
    from the record's compared names, so base and tip judge identical input: any difference
    is the code change. Standard library only besides t2pw.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

EXTRA_TEXTS = [  # the three ORCH-801 legs: uncommitted, read-only, in the orch801run worktree
    "C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW/.claude/worktrees/orch801run/runs_papers/ORCH-801/legs",
]


def _texts(root: Path):
    out = sorted(root.glob("runs*/**/01_source_text.txt"))
    out += sorted((root / "docs/pwml_recovery_sprint/evidence/orch754a_inputs/papers").glob("*.txt"))
    for extra in EXTRA_TEXTS:
        out += sorted(Path(extra).glob("*/*/papers/*/01_source_text.txt"))
    return out


def _gate_records(node, found):
    if isinstance(node, dict):
        gate = node.get("name_gate")
        if isinstance(gate, dict) and gate.get("entity_name") and gate.get("compared_names"):
            found.append(gate)
        # A row whose name_gate record did not survive (tb dThd: re-added by an audit patch)
        # still carries the PathBank record it was judged against in db_match.chosen.
        chosen = (node.get("db_match") or {}).get("chosen") if isinstance(node.get("db_match"), dict) else None
        if isinstance(chosen, dict) and node.get("raw_name") and chosen.get("name") and chosen.get("id"):
            ids = {"pathbank_compound_id": str(chosen["id"])}
            if chosen.get("chebi_id"):
                ids["chebi"] = f"CHEBI:{chosen['chebi_id']}"
            found.append({"entity_name": node["raw_name"], "kind": "compound", "source": "db_match",
                          "compared_names": [chosen["name"], chosen.get("short_name") or "", chosen.get("biocyc_id") or ""],
                          "identity": ", ".join(f"{k}:{v}" for k, v in sorted(ids.items())),
                          "verdict": node.get("db_status")})
        for value in node.values():
            _gate_records(value, found)
    elif isinstance(node, list):
        for value in node:
            _gate_records(value, found)


def _ids(identity: str):
    out = {}
    for part in str(identity or "").split(", "):
        key, _, value = part.partition(":")
        if key and value:
            out[key.strip()] = value.strip()
    return out


def scan(root: Path, out: Path) -> None:
    from t2pw.mapping.map_ids import _name_gate_verdict
    from t2pw.pipeline import extraction_diagnostics as diag
    from t2pw.pipeline.paper_lexicon import abbreviation_pairs
    from t2pw.pipeline.stage_one_ensemble import run_ensemble

    texts, by_sha = {}, {}
    for path in _texts(root):
        text = path.read_text(encoding="utf-8", errors="replace")
        sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
        by_sha.setdefault(sha, []).append(str(path))
        texts[sha] = text
    pairs = {sha: sorted({f"{s}\t{l}" for s, l, _p in abbreviation_pairs(t)}) for sha, t in texts.items()}

    records = []
    for sha, paths in by_sha.items():
        for p in paths:
            paper_dir = Path(p).parent
            for mapped in sorted(paper_dir.glob("*/final_mapped.json")):
                found = []
                _gate_records(json.loads(mapped.read_text(encoding="utf-8")), found)
                if not found:
                    continue
                diag.activate()
                run_ensemble(lambda: ({"entities": {}, "processes": {"reactions": []}}, []), 2,
                             failure=RuntimeError, paper_text=texts[sha])
                seen = set()
                for g in found:
                    names = [str(n) for n in g["compared_names"]]
                    key = (g["entity_name"], tuple(names), g.get("kind"), g.get("identity"))
                    if key in seen:
                        continue
                    seen.add(key)
                    ids = _ids(g.get("identity"))
                    if not ids:
                        continue
                    cand = {"name": names[0], "short_name": names[1] if len(names) > 1 else "",
                            "resolved_name": names[2] if len(names) > 2 else "", "mapped_ids": ids,
                            "pathbank_compound_id": ids.get("pathbank_compound_id", "")}
                    verdict = _name_gate_verdict(g["entity_name"], candidates=[cand], mapped_ids=ids,
                                                 kind=str(g.get("kind") or "compound"))
                    records.append({"file": str(mapped), "source": g.get("source", "name_gate"), "entity_name": g["entity_name"], "compared": names,
                                    "kind": g.get("kind"), "identity": g.get("identity"),
                                    "recorded": g.get("verdict"), "verdict": verdict["verdict"],
                                    "reason": verdict["reason"]})
        diag.activate()
    out.write_text(json.dumps({"texts": by_sha, "pairs": pairs, "records": records}, indent=1), encoding="utf-8")
    print(f"texts {len(texts)} (files {sum(len(v) for v in by_sha.values())}), "
          f"pairs {sum(len(v) for v in pairs.values())}, gate records {len(records)}")


def compare(base: Path, tip: Path, out: Path) -> None:
    b, t = json.loads(base.read_text(encoding="utf-8")), json.loads(tip.read_text(encoding="utf-8"))
    removed, added = [], []
    for sha, paths in t["texts"].items():
        old, new = set(b["pairs"].get(sha, [])), set(t["pairs"].get(sha, []))
        removed += [{"paper": paths[0], "pair": p} for p in sorted(old - new)]
        added += [{"paper": paths[0], "pair": p} for p in sorted(new - old)]
    def key(r):
        return (r["file"], r["entity_name"], tuple(r["compared"]), r["kind"], r["identity"])
    old_v = {key(r): r for r in b["records"]}
    flips = {"keep_to_reject": [], "reject_to_keep": [], "other": []}
    for r in t["records"]:
        o = old_v.get(key(r))
        if o is None or o["verdict"] == r["verdict"]:
            continue
        row = {**r, "base_verdict": o["verdict"], "base_reason": o["reason"]}
        name = f"{o['verdict']}_to_{r['verdict']}"
        flips[name if name in flips else "other"].append(row)
    result = {"pairs_removed": removed, "pairs_added": added,
              "records_compared": len(t["records"]), "records_base": len(b["records"]), "flips": flips}
    out.write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(f"pairs removed {len(removed)}, added {len(added)}; records base {len(b['records'])} tip {len(t['records'])}; "
          + ", ".join(f"{k} {len(v)}" for k, v in flips.items()))
    for row in removed:
        print("REMOVED", row["pair"].replace("\t", " -> "), "|", Path(row["paper"]).parent.name)
    for row in added:
        print("ADDED  ", row["pair"].replace("\t", " -> "), "|", Path(row["paper"]).parent.name)
    for name, rows in flips.items():
        for row in rows:
            print(f"FLIP {name} [{row.get('source')}]: {row['entity_name']!r} vs {row['compared']} ({row['base_reason']} -> {row['reason']}) | {row['file']}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("scan")
    s.add_argument("--root", required=True, type=Path)
    s.add_argument("--out", required=True, type=Path)
    c = sub.add_parser("compare")
    c.add_argument("--base", required=True, type=Path)
    c.add_argument("--tip", required=True, type=Path)
    c.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()
    if a.cmd == "scan":
        scan(a.root, a.out)
    else:
        compare(a.base, a.tip, a.out)
    sys.exit(0)
