"""C-110 probe -- does a leg that DECLINED actually preserve artifacts?

C-110's condition 3 requires at least one recorded artifact before it will call
an empty result a decision. F-148 justifies that (``files: []`` is the signature
of a child killed with its finalization reserve spent), but it carries a risk in
the other direction: if a genuinely declined leg records nothing either, the
status is unreachable and the card delivers nothing.

**This counts row fields. It scores nothing, loads no gold set, and calls no part
of the acceptance instrument.** It answers exactly one question: on the committed
manifests, do rows whose ``failure_kind`` is a DECLARED DECLINE carry ``files``
and ``counts``, and do rows whose ``failure_kind`` is a CASUALTY carry them?

**T-107 is excluded by name.** ``runs_verify/2026-08-28_1816`` is not opened. Its
verdict is ``NOT ACCEPTED``, it is a fact about the artifacts it produced, and
nothing here re-reads or re-interprets them.

Usage::

    <venv-python> c110_declined_legs_preserve_artifacts.py <worktree-root>
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(sys.argv[1]).resolve()

#: NOT OPENED. T-107's run directory, excluded by name and by assertion.
T107 = "2026-08-28_1816"

DECLINE_KINDS = {"no_reactions", "contract"}
CASUALTY_KINDS = {"timeout", "crash", "network", "llm"}


def main() -> int:
    manifests = sorted(
        path
        for base in ("runs", "runs_verify")
        for path in (ROOT / base).glob("*/manifest.jsonl")
        if T107 not in str(path)
    )
    skipped = sorted(
        str(path.relative_to(ROOT))
        for base in ("runs", "runs_verify")
        for path in (ROOT / base).glob("*/manifest.jsonl")
        if T107 in str(path)
    )

    print("=" * 78)
    print("C-110 -- do DECLINED legs preserve artifacts, and do CASUALTIES not?")
    print("=" * 78)
    print(f"manifests read    : {len(manifests)}")
    print(f"EXCLUDED BY RULE  : {skipped or '(none present)'}   <- T-107, never opened")
    print()

    buckets: dict[str, Counter] = {
        "declared decline": Counter(),
        "casualty": Counter(),
        "other": Counter(),
    }
    examples: dict[str, list[str]] = {key: [] for key in buckets}
    # Per-`failure_kind`, because "declared decline" lumps `no_reactions` in with
    # `contract` and only the first is the shape the ruling was written about.
    per_kind: dict[str, Counter] = {}

    for manifest in manifests:
        for line in manifest.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(row, dict):
                continue
            status = str(row.get("status") or "").strip().casefold()
            if status == "pass":
                continue
            kind = str(row.get("failure_kind") or "").strip().casefold()
            if kind in DECLINE_KINDS:
                bucket = "declared decline"
            elif kind in CASUALTY_KINDS or status in ("timeout", "error"):
                bucket = "casualty"
            else:
                bucket = "other"

            files = [entry for entry in (row.get("files") or ()) if entry]
            counts = row.get("counts") if isinstance(row.get("counts"), dict) else {}
            tally = buckets[bucket]
            tally["legs"] += 1
            tally["with files"] += 1 if files else 0
            tally["with counts"] += 1 if counts else 0
            tally["with a message"] += 1 if str(row.get("message") or "").strip() else 0
            kind_tally = per_kind.setdefault(kind or "(none)", Counter())
            kind_tally["legs"] += 1
            kind_tally["with files"] += 1 if files else 0
            kind_tally["with counts"] += 1 if counts else 0
            if len(examples[bucket]) < 4:
                examples[bucket].append(
                    f"{manifest.parent.name}/{row.get('paper_id')}/{row.get('mode')}"
                    f"  kind={kind or '(none)'}  status={status or '(none)'}"
                    f"  files={len(files)}  counts={'yes' if counts else 'no'}"
                )

    for bucket, tally in buckets.items():
        legs = tally["legs"]
        print(f"-- {bucket.upper()}  ({legs} non-passing legs)")
        if not legs:
            print("     (none in the corpus)")
            print()
            continue
        for key in ("with files", "with counts", "with a message"):
            print(f"     {key:<16}: {tally[key]:>4} / {legs}   ({tally[key] / legs:.0%})")
        for line in examples[bucket]:
            print(f"       e.g. {line}")
        print()

    print("-- PER failure_kind (the coarse buckets above hide the one that matters)")
    print(f"       {'failure_kind':<24}{'legs':>6}{'with files':>12}{'with counts':>13}")
    for kind in sorted(per_kind):
        tally = per_kind[kind]
        print(
            f"       {kind:<24}{tally['legs']:>6}"
            f"{tally['with files']:>12}{tally['with counts']:>13}"
        )
    print()

    print("=" * 78)
    print("READING")
    print("=" * 78)
    print("A high `with files` rate on DECLARED DECLINE and a low one on CASUALTY is")
    print("what makes C-110's artifact condition discriminating rather than merely")
    print("strict. A declined leg that preserved nothing does not earn the status --")
    print("and by PRODUCT_CONTRACT 4 it also failed to preserve what a no-PWML")
    print("outcome owes, so reporting it as unawarded is the honest answer, not a")
    print("false negative.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
