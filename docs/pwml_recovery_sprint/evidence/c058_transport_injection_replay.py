"""C-058 -- replay ``_inject_name_based_modifiers`` over the committed legs.

Runs the production injector on a deep copy of every committed
``stage1_payload.json`` -- no Stage-2 additions, no RAG, no LLM, no network --
and reports, per transport row:

* the Stage-1 ``transporters`` value,
* what the injector produced from it, and
* what the committed ``merged_payload.json`` actually shipped.

Run it at the base SHA and at the tip and diff the two artifacts. At base the
replayed transporters reproduce the committed merged ones character for
character, which is what pins ``pipeline.py :: _inject_name_based_modifiers``
as the site of F-058; at the tip the fabricated ``EntE`` rows are gone and the
correct Stage-1 ``FepA`` row is still there, untouched.

``--reaction-digest`` additionally emits a SHA-256 over the *reaction* actors
the same call produced, so the reaction branch can be shown byte-identical
across the two revisions rather than merely undiffed in the source.

Usage::  <python> c058_transport_injection_replay.py --out <path.json>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))

from t2pw.pipeline.pipeline import _inject_name_based_modifiers  # noqa: E402


def _rows(payload: Dict[str, Any], key: str) -> List[Any]:
    value = ((payload.get("processes") or {}).get(key)) or []
    return value if isinstance(value, list) else []


def _reaction_actor_digest(payload: Dict[str, Any]) -> str:
    """Stable digest of every reaction actor row, in document order."""
    blob = json.dumps(
        [
            [row.get("name"), row.get("enzymes"), row.get("modifiers")]
            for row in _rows(payload, "reactions")
            if isinstance(row, dict)
        ],
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--roots",
        default="runs_verify/2026-08-18_1328",
        help="comma-separated roots to scan for */stage1_payload.json",
    )
    args = parser.parse_args(argv)

    stage1_paths: List[Path] = []
    for root in args.roots.split(","):
        stage1_paths.extend(sorted((REPO_ROOT / root.strip()).glob("**/stage1_payload.json")))
    legs: List[Dict[str, Any]] = []
    for stage1 in sorted(set(stage1_paths)):
        merged_path = stage1.parent / "merged_payload.json"
        if not merged_path.exists():
            continue
        stage1_payload = json.loads(stage1.read_text(encoding="utf-8"))
        merged_payload = json.loads(merged_path.read_text(encoding="utf-8"))
        replayed = deepcopy(stage1_payload)
        _inject_name_based_modifiers(replayed)

        committed_by_name = {
            str(row.get("name") or ""): row.get("transporters")
            for row in _rows(merged_payload, "transports")
            if isinstance(row, dict)
        }
        rows: List[Dict[str, Any]] = []
        for before, after in zip(
            _rows(stage1_payload, "transports"), _rows(replayed, "transports")
        ):
            if not isinstance(after, dict):
                continue
            name = str(after.get("name") or "")
            committed = committed_by_name.get(name)
            seeded = (before or {}).get("transporters") or []
            replayed_rows = after.get("transporters") or []
            rows.append(
                {
                    "transport_name": name,
                    "evidence": after.get("evidence"),
                    "stage1_transporters": (before or {}).get("transporters"),
                    "replayed_transporters": after.get("transporters"),
                    "committed_merged_transporters": committed,
                    "replay_reproduces_committed": after.get("transporters") == committed,
                    # Merge rule 7: a Stage-1 transporter is correct biology the
                    # injector may add to but must never remove or rewrite.
                    "stage1_transporters_preserved": all(
                        row in replayed_rows for row in seeded
                    ),
                    "injected_rows": len(
                        [
                            row
                            for row in replayed_rows
                            if isinstance(row, dict) and row.get("provenance") == "inferred"
                        ]
                    ),
                }
            )
        legs.append(
            {
                "leg": stage1.parent.relative_to(REPO_ROOT).as_posix(),
                "actors": {
                    "protein_complexes": [
                        r.get("name")
                        for r in ((stage1_payload.get("entities") or {}).get("protein_complexes") or [])
                        if isinstance(r, dict)
                    ],
                    "proteins": [
                        r.get("name")
                        for r in ((stage1_payload.get("entities") or {}).get("proteins") or [])
                        if isinstance(r, dict)
                    ],
                },
                "transports": rows,
                "reaction_actor_digest_after_replay": _reaction_actor_digest(replayed),
            }
        )

    all_rows = [row for leg in legs for row in leg["transports"]]
    report = {
        "task": "C-058",
        "purpose": (
            "replay pipeline._inject_name_based_modifiers on committed stage1 payloads; "
            "at base it reproduces the committed merged transporters byte-exactly, at the "
            "tip the fabricated rows are refused and the correct FepA row survives"
        ),
        "roots": args.roots,
        "legs": legs,
        "reaction_actor_digests": {
            leg["leg"]: leg["reaction_actor_digest_after_replay"] for leg in legs
        },
        "totals": {
            "legs": len(legs),
            "transport_rows": len(all_rows),
            "rows_reproducing_committed": len(
                [r for r in all_rows if r["replay_reproduces_committed"]]
            ),
            "rows_with_injected_transporters": len(
                [r for r in all_rows if r["injected_rows"]]
            ),
            "total_injected_transporter_rows": sum(r["injected_rows"] for r in all_rows),
            "rows_losing_a_stage1_transporter": len(
                [r for r in all_rows if not r["stage1_transporters_preserved"]]
            ),
        },
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report["totals"], sort_keys=True))
    for leg in legs:
        print(f"{leg['leg']}: reactions_digest={leg['reaction_actor_digest_after_replay']}")
        for row in leg["transports"]:
            print(
                f"  {row['transport_name']!r} injected={row['injected_rows']} "
                f"reproduces_committed={row['replay_reproduces_committed']}"
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
