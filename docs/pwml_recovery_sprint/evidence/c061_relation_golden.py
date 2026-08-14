"""C-061 preservation golden: every real candidate span, parsed and judged.

Orchestration tooling, not pipeline code. It imports ``t2pw`` from whatever
``PYTHONPATH`` names, so the SAME script produces the base golden and the tip
golden and the two are directly comparable.

Why this corpus. Clause **A9** asks whether every existing caller still gets its
previous result on every previously-passing input. The honest corpus for that is
not a hand-written fixture: it is every candidate the gate has actually judged on
committed artifacts — 700+ spans across nine papers and both legs, each with the
claim that was made against it. A golden over those answers "did any real
decision move, and which", which is also the merge-rule-6 question.

Usage::

    PYTHONPATH=<tree>/src python c061_relation_golden.py <out.json>
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
RUNS = ROOT / "runs" / "2026-08-02_2130" / "papers"


def _rows():
    for report in sorted(RUNS.glob("*/*/rag_admission_report.json")):
        leg = report.parent.name
        paper = report.parent.parent.name
        payload = json.loads(report.read_text(encoding="utf-8"))
        for bucket in ("accepted", "rejected"):
            for index, candidate in enumerate(payload.get(bucket) or []):
                span = ((candidate.get("evidence") or {}).get("span")) or ""
                if not span:
                    continue
                yield {
                    "paper": paper,
                    "leg": leg,
                    "bucket": bucket,
                    "index": index,
                    "span": span,
                    "inputs": list(candidate.get("inputs") or []),
                    "outputs": list(candidate.get("outputs") or []),
                    "enzymes": list(candidate.get("enzymes") or []),
                    "reversible": bool(candidate.get("reversible")),
                }


def build() -> dict:
    import t2pw
    from t2pw.rag.admission import parse_span_relation, validate_evidence_span

    entries = []
    for row in _rows():
        relation = parse_span_relation(row["span"])
        verdict = validate_evidence_span(
            row["span"],
            inputs=row["inputs"],
            outputs=row["outputs"],
            enzymes=row["enzymes"],
            reversible=row["reversible"],
        )
        entry = {
            "key": f"{row['paper']}/{row['leg']}/{row['bucket']}[{row['index']}]",
            "span_sha256": hashlib.sha256(row["span"].encode("utf-8")).hexdigest()[:16],
            "shim": None
            if relation is None
            else {
                "pattern": relation.pattern,
                "inputs": relation.inputs,
                "outputs": relation.outputs,
                "catalysts": relation.catalysts,
                "reversible": relation.reversible,
            },
            "ok": bool(verdict.ok),
            "reasons": list(verdict.reasons),
            "normalized_reversible": verdict.normalized_reversible,
        }
        entry["digest"] = entry_digest(entry)
        entries.append(entry)
    body = {"t2pw": t2pw.__file__, "n": len(entries), "entries": entries}
    body["sha256"] = hashlib.sha256(
        json.dumps(entries, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()
    return body


def entry_digest(entry) -> str:
    """The whole observable answer for one candidate, in one hash.

    Covers the shim's relation, the verdict, every reason string and the
    reversibility normalization — so a change to any of them shows up, and a
    committed base file of these digests stays small enough to review.
    """
    return hashlib.sha256(
        json.dumps(
            [
                entry["span_sha256"],
                entry["shim"],
                entry["ok"],
                entry["reasons"],
                entry["normalized_reversible"],
            ],
            sort_keys=True,
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()[:16]


def main(argv) -> int:
    out = Path(argv[1]) if len(argv) > 1 else None
    full = Path(argv[2]) if len(argv) > 2 else None
    body = build()
    if out is not None:
        out.write_text(
            json.dumps(
                {
                    "sha256": body["sha256"],
                    "n": body["n"],
                    "digests": [[e["key"], e["digest"]] for e in body["entries"]],
                },
                indent=1,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
    if full is not None:
        full.write_text(
            json.dumps(body, indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
        )
    print("t2pw:", body["t2pw"])
    print("entries:", body["n"])
    print("sha256:", body["sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
