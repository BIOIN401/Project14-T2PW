"""C-034 G9 artifact: behavioural golden of ``settle_stage_one``'s observable output.

C-034 adds provenance lineage at the Stage-1 exit gate. Acceptance clause **A4**
is a *preservation* claim -- the outcome, the status, the diagnostics boundary
records and every non-lineage field of every row must be what they were at the
base SHA. So this capture is the reverse of the usual G9 proof: it must
**reproduce byte-identically at ``bcc0bfe`` and at the branch tip**. It fails
loudly if the tip moved anything the card is not entitled to move.

Two traps this is built against.

* **S9.3 -- a golden captured at tip proves nothing.** Run it at base FIRST and
  keep that digest; the tip run is then a comparison, not a definition. The
  script carries no expected value for exactly this reason.
* **S9.4 -- content-equality hides re-serialization.** The fixtures are
  deliberately NON-canonical: keys in reverse-alphabetical order, non-ASCII
  characters, and a nested bucket the producer would never emit in that shape.
  The capture is serialized with ``sort_keys=False``, so a re-serialization that
  reorders keys changes the digest instead of being absorbed by it.

``provenance_lineage`` is stripped recursively before hashing, which is what
makes the two runs comparable at all: at base there is none, at tip there is,
and that difference is the *new capability*, reported separately in
``lineage_entry_count`` and never folded into the preservation digest.

ONE OBSERVABLE VALUE IS EXPECTED TO MOVE, and it is named rather than absorbed.
``BoundaryOutcome.to_summary()["payload_hash"]`` fingerprints the WHOLE returned
payload, so a payload that gained lineage gets a different one. That is
``PRODUCT_CONTRACT`` line 178 -- "lineage changes must remain detectable" -- not
a decision change: it is a fingerprint of the artifact, not an input to any
judgement, and the boundary's own ``record_boundary`` fingerprint is computed
BEFORE the lineage writes and must still match at base (it does; it is inside
``boundaries``, which the strict digest covers). So two digests are emitted:

* ``strict_sha256`` -- literally everything. It DIFFERS, in exactly this field.
* ``preservation_sha256`` -- the same with that one field replaced by a
  sentinel. It must MATCH. Nothing else is excluded.

``summary_payload_hash_by_leg`` prints the moved values side by side so the
exclusion can be audited instead of trusted.

Usage::

    python c034_boundary_golden.py --out <capture.json>

Exit code is 0 on a clean capture. Compare two captures with ``--compare``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List

_HERE = Path(__file__).resolve()
for _parent in _HERE.parents:
    _candidate = _parent / "src"
    if (_candidate / "t2pw").is_dir():
        if str(_candidate) not in sys.path:
            sys.path.insert(0, str(_candidate))
        break

from t2pw.pipeline.extraction_diagnostics import activate, deactivate  # noqa: E402
from t2pw.pipeline.stage_one_boundary import settle_stage_one  # noqa: E402

LINEAGE_KEY = "provenance_lineage"  # spelled out, not imported: it is absent at base


class _Reply:
    def __init__(self, text: str) -> None:
        self.text = text
        self.diagnostics = {
            "model": "golden-model",
            "finish_reason": "stop",
            "attempts": 1,
            "response_status": "ok",
            "terminal_reason": "",
            "attempt_log": [],
        }


class _Provider:
    """A scripted, deterministic stand-in for the repair model."""

    def __init__(self, *replies: str) -> None:
        self._replies = list(replies)

    def __call__(self, messages: List[Dict[str, str]], **_: Any) -> _Reply:
        if not self._replies:
            raise AssertionError("the boundary made more repair draws than were scripted")
        return _Reply(self._replies.pop(0))


def _non_canonical() -> Dict[str, Any]:
    """Keys reverse-ordered, non-ASCII text, a participant the registry lacks.

    ``processes`` before ``entities``, ``outputs`` before ``name``: no producer
    in this pipeline emits that order, so a capture that comes back sorted is
    reporting a re-serialization rather than the payload the boundary returned.
    """

    return {
        "processes": {
            "reactions": [
                {
                    "outputs": ["ADP", "Pi"],
                    "inputs": ["ATP", "H₂O"],
                    "evidence": "ATP is hydrolysed to ADP and Pᵢ — see § 2.",
                    "name": "ATP hydrolysis — cytosolic",
                    "enzymes": [{"name": "ATPase α", "entity_type": "protein"}],
                }
            ],
            "transports": [],
        },
        "entities": {
            "proteins": [
                {
                    "organism": "Escherichia coli",
                    "name": "LpxA",
                    "evidence": "LpxA acetylates UDP-GlcNAc.",
                }
            ],
            "compounds": [
                {"name": "ATP", "hmdb_id": "HMDB0000538"},
                {"name": "H₂O"},
            ],
        },
        "pathway_name": "Golden — non-canonical fixture",
    }


_NAMELESS = {
    "processes": {
        "reactions": [
            {"outputs": ["product"], "inputs": ["UDP-GlcNAc"], "evidence": "e", "name": "r"}
        ]
    },
    "entities": {"proteins": [{"evidence": "LpxA acetylates UDP-GlcNAc.", "name": ""}]},
}

_REPAIR_REPLY = json.dumps(
    {
        "repaired_rows": [
            {
                "pointer": "/entities/proteins/0",
                "row": {"name": "LpxA", "evidence": "LpxA acetylates UDP-GlcNAc."},
            }
        ]
    }
)


def _strip_lineage(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _strip_lineage(v) for k, v in value.items() if k != LINEAGE_KEY}
    if isinstance(value, list):
        return [_strip_lineage(item) for item in value]
    return value


def _count_lineage(value: Any) -> int:
    if isinstance(value, dict):
        here = len(value.get(LINEAGE_KEY) or []) if LINEAGE_KEY in value else 0
        return here + sum(
            _count_lineage(v) for k, v in value.items() if k != LINEAGE_KEY
        )
    if isinstance(value, list):
        return sum(_count_lineage(item) for item in value)
    return 0


def _leg(name: str, payload: Dict[str, Any], **kwargs: Any) -> Dict[str, Any]:
    """One settle, captured completely enough that a silent change cannot hide."""

    with tempfile.TemporaryDirectory(prefix="c034golden-") as tmp:
        recorder = activate(run_id="c034-golden", artifact_dir=Path(tmp))
        try:
            outcome = settle_stage_one(json.loads(json.dumps(payload)), **kwargs)
            boundaries = recorder.boundaries
        finally:
            deactivate()

    return {
        "leg": name,
        "ok": outcome.ok,
        "outcome": outcome.outcome,
        "incomplete_reason": outcome.incomplete_reason,
        "failure": None if outcome.failure is None else str(outcome.failure),
        "contract_report": _strip_lineage(outcome.contract_report),
        "reconstruction": _strip_lineage(outcome.reconstruction),
        "repair": None
        if outcome.repair is None
        else {
            "outcome": outcome.repair.outcome,
            "attempts": outcome.repair.attempts,
            "repaired_pointers": outcome.repair.repaired_pointers,
            "unrepaired_pointers": outcome.repair.unrepaired_pointers,
            "rejected": _strip_lineage(outcome.repair.rejected),
            "reason": outcome.repair.reason,
        },
        "boundaries": _strip_lineage(boundaries),
        "summary": _strip_lineage(outcome.to_summary()),
        "payload_without_lineage": _strip_lineage(outcome.payload),
        # NEW CAPABILITY, reported beside the preservation digest and never
        # inside it: 0 at base, non-zero at tip.
        "lineage_entry_count": _count_lineage(outcome.payload),
    }


def capture() -> Dict[str, Any]:
    legs = [
        _leg("reconstruct", _non_canonical(), chat_fn=_Provider()),
        _leg("no_reconstruct", _non_canonical(), reconstruct=False, chat_fn=_Provider()),
        _leg("repair", _NAMELESS, chat_fn=_Provider(_REPAIR_REPLY)),
        _leg("incomplete", _NAMELESS, repair_rows=False),
        _leg("not_a_dict", {}, chat_fn=_Provider()),
    ]
    strict = [
        {k: v for k, v in leg.items() if k != "lineage_entry_count"} for leg in legs
    ]
    preserved = json.loads(json.dumps(strict))
    for leg in preserved:
        if isinstance(leg.get("summary"), dict) and "payload_hash" in leg["summary"]:
            # The one named exclusion. See the module docstring; the value it
            # replaces is reported in ``summary_payload_hash_by_leg``.
            leg["summary"]["payload_hash"] = "<covers lineage by design>"

    def digest(value: Any) -> str:
        body = json.dumps(value, ensure_ascii=False, sort_keys=False, indent=None)
        return hashlib.sha256(body.encode("utf-8")).hexdigest()

    return {
        "legs": legs,
        "strict_sha256": digest(strict),
        "preservation_sha256": digest(preserved),
        "lineage_entry_counts": {leg["leg"]: leg["lineage_entry_count"] for leg in legs},
        "summary_payload_hash_by_leg": {
            leg["leg"]: (leg["summary"] or {}).get("payload_hash", "") for leg in legs
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, help="write the capture here")
    parser.add_argument("--compare", nargs=2, type=Path, metavar=("BASE", "TIP"))
    args = parser.parse_args()

    if args.compare:
        base, tip = (
            json.loads(path.read_text(encoding="utf-8")) for path in args.compare
        )
        same = base["preservation_sha256"] == tip["preservation_sha256"]
        for name in ("strict_sha256", "preservation_sha256"):
            print(f"base {name}: {base[name]}")
            print(f"tip  {name}: {tip[name]}")
        print(f"PRESERVED: {same}")
        print(f"base lineage_entry_counts: {base['lineage_entry_counts']}")
        print(f"tip  lineage_entry_counts: {tip['lineage_entry_counts']}")
        print("summary payload_hash (expected to move, lineage is detectable):")
        for leg in base["summary_payload_hash_by_leg"]:
            print(f"  {leg}: base={base['summary_payload_hash_by_leg'][leg]} "
                  f"tip={tip['summary_payload_hash_by_leg'][leg]}")
        for a, b in zip(base["legs"], tip["legs"]):
            for key in a:
                if key == "lineage_entry_count":
                    continue
                if a[key] != b[key]:
                    print(f"  DIFF leg={a['leg']} field={key}")
        return 0 if same else 1

    result = capture()
    text = json.dumps(result, ensure_ascii=False, indent=2)
    if args.out:
        args.out.write_text(text, encoding="utf-8")
    print(f"preservation_sha256: {result['preservation_sha256']}")
    print(f"lineage_entry_counts: {result['lineage_entry_counts']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
