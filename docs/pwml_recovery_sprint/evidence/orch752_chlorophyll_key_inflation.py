"""ORCH-752: pooled-key inflation on the chlorophyll draws, ORCH-750 key vs ORCH-752 key.

ORCH-751 § 2.2 measured three Stage-1 draws (16, 19, 18 reactions) pooling to 41 distinct
keys and attributed it to the dedup key being cofactor-sensitive. This script re-runs that
measurement on the same three payloads under BOTH keys and writes the comparison beside
itself, so the number is checkable against an artifact rather than a claim.

BEFORE is ORCH-750's ``row_key`` as delivered, restated here verbatim from that module
(reaction: exact set of strict-keyed input names x exact set of strict-keyed output names;
interaction: ``(entity_1, relationship, entity_2)`` each strict-keyed). AFTER is the tip's
``merge_draws``. The payloads are the ORCH-751 outputs copied from ``C:/t/cohort/`` into
``orch752_chlorophyll_payloads/``; their sha256 is recorded so the inputs are pinned.

Usage::

    PYTHONPATH=src python docs/pwml_recovery_sprint/evidence/orch752_chlorophyll_key_inflation.py \
        [--payload-dir docs/pwml_recovery_sprint/evidence/orch752_chlorophyll_payloads]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Tuple

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from t2pw.bench.eval_v2.matcher import strict_key  # noqa: E402
from t2pw.pipeline.participant_schema import PARTICIPANT_NAME_KEYS  # noqa: E402
from t2pw.pipeline.stage_one_ensemble import CURRENCY_KEYS, merge_draws, row_key  # noqa: E402

BUCKETS = ("/processes/reactions", "/processes/interactions", "/processes/transports")


# --- BEFORE: ORCH-750 row_key, restated verbatim ------------------------------------------
def _names_750(value: Any) -> List[str]:
    out: List[str] = []
    for item in value if isinstance(value, list) else [value]:
        if isinstance(item, str) and item.strip():
            out.append(item)
        elif isinstance(item, dict):
            name = next((item[k] for k in PARTICIPANT_NAME_KEYS if item.get(k)), None)
            if name:
                out.append(str(name))
    return out


def _keyset_750(names: List[str]) -> frozenset:
    return frozenset(k for k in (strict_key(n) for n in names) if k)


def row_key_750(path: str, row: Dict[str, Any]) -> Tuple[Any, ...]:
    if path == "/processes/reactions":
        return (_keyset_750(_names_750(row.get("inputs"))), _keyset_750(_names_750(row.get("outputs"))))
    if path == "/processes/interactions":
        return tuple(strict_key(row.get(f)) for f in ("entity_1", "relationship", "entity_2"))
    return tuple(strict_key(row.get(f)) for f in ("cargo", "from_biological_state", "to_biological_state"))


def _label(key: Tuple[Any, ...]) -> str:
    if len(key) == 2 and isinstance(key[0], frozenset):
        return " + ".join(sorted(key[0])) + " -> " + " + ".join(sorted(key[1]))
    return " | ".join(str(p) for p in key)


def _pool(payloads: List[Dict[str, Any]], keyfn) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for path in BUCKETS:
        seen: Dict[Tuple[Any, ...], List[int]] = {}
        per_draw = []
        for d, p in enumerate(payloads, start=1):
            rows = p.get("processes", {}).get(path.rsplit("/", 1)[1]) or []
            per_draw.append(len(rows))
            for row in rows:
                seen.setdefault(keyfn(path, row), []).append(d)
        pair_overlap = {f"draw {a} vs draw {b}": sum(1 for ds in seen.values() if a in ds and b in ds)
                        for a, b in combinations(range(1, len(payloads) + 1), 2)}
        out[path] = {
            "rows_per_draw": per_draw,
            "pooled_keys": len(seen),
            "union_to_mean_ratio": round(len(seen) / (sum(per_draw) / len(per_draw)), 3) if sum(per_draw) else None,
            "shared_by_pair": pair_overlap,
            "shared_by_all": sum(1 for ds in seen.values() if len(set(ds)) == len(payloads)),
            "keys": [{"draws": sorted(set(ds)), "key": _label(k)} for k, ds in seen.items()],
        }
    return out


def _cofactor_census(payloads: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Which currency names the draws actually wrote on reaction sides, and whether any
    pooled defining key differs between draws only by its cofactors (the ORCH-751 theory)."""

    currency_seen: Counter = Counter()
    by_defining: Dict[Tuple[Any, ...], set] = {}
    for p in payloads:
        for row in p.get("processes", {}).get("reactions") or []:
            cof = set()
            for slot in ("inputs", "outputs"):
                for name in _names_750(row.get(slot)):
                    k = strict_key(name)
                    if k in CURRENCY_KEYS:
                        currency_seen[name] += 1
                        cof.add(f"{slot}:{k}")
            by_defining.setdefault(row_key("/processes/reactions", row), set()).add(frozenset(cof))
    varying = [_label(k) for k, variants in by_defining.items() if len(variants) > 1]
    return {
        "currency_names_written": dict(currency_seen.most_common()),
        "defining_keys_whose_cofactor_set_varies_between_draws": varying,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("--payload-dir", default=str(HERE / "orch752_chlorophyll_payloads"))
    ap.add_argument("--out", default=str(HERE / "orch752_chlorophyll_key_inflation.json"))
    args = ap.parse_args()

    files = sorted(Path(args.payload_dir).glob("control__run*__payload.json"))
    payloads = [json.loads(f.read_text(encoding="utf-8")) for f in files]
    before, after = _pool(payloads, row_key_750), _pool(payloads, row_key)
    _merged, report = merge_draws(payloads)
    tip = {p: report["buckets"][p]["merged"] for p in BUCKETS if p in report["buckets"]}
    assert all(after[p]["pooled_keys"] == tip[p] for p in tip), "offline pool and merge_draws disagree"

    result = {
        "task": "ORCH-752",
        "payloads": [{"file": f.name, "sha256": hashlib.sha256(f.read_bytes()).hexdigest()} for f in files],
        "orch751_claim": {"reactions_pooled_keys": 41, "rows_per_draw": [16, 19, 18],
                          "attributed_to": "cofactor-sensitive dedup key"},
        "summary": {p: {"rows_per_draw": before[p]["rows_per_draw"],
                        "before_pooled_keys": before[p]["pooled_keys"],
                        "after_pooled_keys": after[p]["pooled_keys"],
                        "merge_draws_merged_rows": tip.get(p)} for p in BUCKETS},
        "cofactor_census": _cofactor_census(payloads),
        "cofactors_added_by_merge": [r for r in report["buckets"]["/processes/reactions"]["rows"]
                                     if r.get("cofactors_added")],
        "before": before,
        "after": after,
    }
    Path(args.out).write_text(json.dumps(result, indent=1, ensure_ascii=False), encoding="utf-8")

    lines = [f"ORCH-752 chlorophyll pooled-key inflation  ({len(files)} draws)", ""]
    for p in BUCKETS:
        s = result["summary"][p]
        lines.append(f"{p:28s} rows/draw {s['rows_per_draw']}  pooled keys: before {s['before_pooled_keys']}"
                     f"  after {s['after_pooled_keys']}")
    lines += ["", "reaction keys shared by pair, before -> after:"]
    for pair in before["/processes/reactions"]["shared_by_pair"]:
        lines.append(f"  {pair}: {before['/processes/reactions']['shared_by_pair'][pair]} -> "
                     f"{after['/processes/reactions']['shared_by_pair'][pair]}")
    c = result["cofactor_census"]
    lines += ["", f"currency names written on reaction sides: {c['currency_names_written']}",
              f"defining keys whose cofactor set varies between draws: "
              f"{c['defining_keys_whose_cofactor_set_varies_between_draws'] or 'NONE'}",
              f"cofactors donated by the merge: {result['cofactors_added_by_merge'] or 'NONE'}", "",
              "AFTER reaction keys (draws, key):"]
    lines += [f"  {k['draws']} {k['key']}" for k in after["/processes/reactions"]["keys"]]
    lines += ["", "AFTER interaction keys (draws, key):"]
    lines += [f"  {k['draws']} {k['key']}" for k in after["/processes/interactions"]["keys"]]
    text = "\n".join(lines) + "\n"
    Path(args.out).with_suffix(".txt").write_text(text, encoding="utf-8")
    sys.stdout.buffer.write(text.encode("utf-8"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
