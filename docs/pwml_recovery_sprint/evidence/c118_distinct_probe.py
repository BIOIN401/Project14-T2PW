"""C-118: the ten changed C-061 golden claims, with the relations their spans state.

Orchestration tooling. For each distinct changed claim it prints the claim, the
gate's verdict, and every relation ``parse_span_relations`` returns for the span,
so the restated assertion in ``test_the_delta_is_five_paper_verbatim_reactions``
can be written against measured shapes instead of guessed ones.

Usage::

    PYTHONPATH=<tree>/src python c118_distinct_probe.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))


def main(argv) -> int:
    import c061_relation_golden as golden
    from t2pw.rag.admission import parse_span_relations, validate_evidence_span

    base_digests = dict(
        json.loads(
            (HERE / "c061_relation_golden_base.json").read_text(encoding="utf-8")
        )["digests"]
    )
    tip = {e["key"]: e for e in golden.build()["entries"]}

    distinct = {}
    for row in golden._rows():
        key = f"{row['paper']}/{row['leg']}/{row['bucket']}[{row['index']}]"
        if base_digests[key] == tip[key]["digest"]:
            continue
        distinct[
            (
                row["span"],
                tuple(row["inputs"]),
                tuple(row["outputs"]),
                tuple(row["enzymes"]),
            )
        ] = row

    out = []
    for (span, inputs, outputs, enzymes), row in distinct.items():
        verdict = validate_evidence_span(
            span,
            inputs=list(inputs),
            outputs=list(outputs),
            enzymes=list(enzymes),
            reversible=False,
        )
        out.append(
            {
                "key": f"{row['paper']}/{row['leg']}/{row['bucket']}[{row['index']}]",
                "span": span,
                "claim_inputs": list(inputs),
                "claim_outputs": list(outputs),
                "claim_enzymes": list(enzymes),
                "ok": bool(verdict.ok),
                "reasons": list(verdict.reasons),
                "relations": [
                    {
                        "pattern": r.pattern,
                        "inputs": r.inputs,
                        "outputs": r.outputs,
                        "catalysts": r.catalysts,
                    }
                    for r in parse_span_relations(span)
                ],
            }
        )
    text = json.dumps({"n_distinct": len(out), "rows": out}, indent=2, ensure_ascii=True)
    print(text)
    if len(argv) > 1:
        Path(argv[1]).write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
