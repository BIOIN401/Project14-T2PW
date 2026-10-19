"""Independently verify drafted gold against its pinned text, before anything is trusted.

Every drafter reports that it verified its own quotes. This re-does it with the
repository's own folding rule (``bench.goldset.fold_for_quote``) rather than the drafter's
script, because a drafter that folds differently from the evaluator will pass its own check
and fail the real one. ORCH-747 established why that matters: a text can hash correctly and
still be wrong, so quote verification is the only check that actually binds.

Checks per draft:

1. **Quotes.** Every ``quote``, ``supporting_quotes`` entry and ``excluded_statements``
   quote occurs in the pinned text under the evaluator's folding.
2. **No self-certification.** ``supported_reactions_complete`` is false.
3. **Bucket admission.** Every row in ``reactions``, ``off_pathway_reactions`` and
   ``summary_statements`` carries at least one defining participant on EACH side — the
   rule the loader enforces, and the one a binding row silently violates.
4. **Unique ids** across the three transformation buckets.
5. **Reference-list contamination (flag, not failure).** The pinned text keeps the
   reference list by design, and reference titles state chemistry. A quote found only in
   the last tail of the text is reported for a human to look at.

Usage::

    python scripts/verify_gold_drafts.py --drafts <dir> --fulltext-dir C:/t/eval-fulltext
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from t2pw.bench.goldset import fold_for_quote  # noqa: E402

#: A quote whose only occurrence starts beyond this fraction of the folded text is
#: probably in the reference list. Flagged for a human, never auto-rejected: some papers
#: genuinely state chemistry in their final paragraphs.
TAIL_FRACTION = 0.88


def _rows(case: Dict[str, Any], bucket: str) -> List[Dict[str, Any]]:
    return [r for r in (case.get(bucket) or ()) if isinstance(r, dict)]


def _quotes(case: Dict[str, Any]) -> List[tuple]:
    out: List[tuple] = []
    for bucket in ("reactions", "off_pathway_reactions", "summary_statements",
                   "interactions", "transports"):
        for row in _rows(case, bucket):
            rid = str(row.get("id") or bucket)
            if str(row.get("quote") or "").strip():
                out.append((rid, "quote", row["quote"]))
            for q in row.get("supporting_quotes") or ():
                if str(q or "").strip():
                    out.append((rid, "supporting", q))
    for i, x in enumerate(case.get("excluded_statements") or (), 1):
        if isinstance(x, dict) and str(x.get("quote") or "").strip():
            out.append((f"X{i}", "excluded", x["quote"]))
    return out


def _defining(row: Dict[str, Any], side: str) -> int:
    return sum(1 for t in (row.get(side) or ())
               if isinstance(t, dict) and str(t.get("role") or "defining") == "defining")


def verify(draft_path: Path, fulltext_dir: Path) -> Dict[str, Any]:
    draft = json.loads(draft_path.read_text(encoding="utf-8"))
    case = draft.get("case") or {}
    paper = str(draft.get("case_id") or draft_path.name.split("__")[0])
    text_path = fulltext_dir / f"{paper}.txt"

    res: Dict[str, Any] = {"paper": paper, "draft": str(draft_path), "ok": True,
                           "counts": {b: len(_rows(case, b)) for b in
                                      ("reactions", "off_pathway_reactions",
                                       "summary_statements", "interactions", "transports")},
                           "excluded": len(case.get("excluded_statements") or ()),
                           "problems": [], "flags": []}
    res["counts"]["excluded_statements"] = res.pop("excluded")

    if case.get("supported_reactions_complete") is not False:
        res["problems"].append("supported_reactions_complete is not false: a model may "
                               "never certify its own exhaustiveness")

    ids = [str(r.get("id")) for b in ("reactions", "off_pathway_reactions",
                                      "summary_statements") for r in _rows(case, b)]
    dupes = sorted({i for i in ids if ids.count(i) > 1})
    if dupes:
        res["problems"].append(f"duplicate row ids: {dupes}")

    for bucket in ("reactions", "off_pathway_reactions", "summary_statements"):
        for row in _rows(case, bucket):
            if not _defining(row, "inputs") or not _defining(row, "outputs"):
                res["problems"].append(
                    f"{bucket}/{row.get('id')}: needs a defining participant on BOTH sides "
                    "(a binding belongs in interactions)")

    if not text_path.is_file():
        res["problems"].append(f"pinned text missing: {text_path}")
        res["ok"] = False
        return res

    folded = fold_for_quote(text_path.read_text(encoding="utf-8"))
    quotes = _quotes(case)
    unverified = []
    for rid, kind, q in quotes:
        fq = fold_for_quote(q)
        if not fq:
            res["problems"].append(f"{rid}: empty {kind} after folding")
            continue
        at = folded.find(fq)
        if at < 0:
            unverified.append({"row": rid, "kind": kind, "quote": q[:140]})
        elif at > TAIL_FRACTION * len(folded):
            res["flags"].append({"row": rid, "kind": kind,
                                 "note": "quote occurs only in the tail of the text; check "
                                         "it is not the reference list",
                                 "quote": q[:100]})
    res["quotes_total"] = len(quotes)
    res["quotes_unverified"] = unverified
    if unverified:
        res["problems"].append(f"{len(unverified)} quote(s) not found in the pinned text")
    res["ok"] = not res["problems"]
    return res


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("--drafts", required=True)
    ap.add_argument("--fulltext-dir", default="C:/t/eval-fulltext")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(list(argv) if argv is not None else None)

    drafts = sorted(Path(args.drafts).glob("*__draft.json"))
    if not drafts:
        raise SystemExit(f"no *__draft.json under {args.drafts}")

    results = [verify(p, Path(args.fulltext_dir)) for p in drafts]
    print(f"{'paper':14s} {'rx':>3s} {'off':>4s} {'sum':>4s} {'int':>4s} {'trn':>4s} "
          f"{'excl':>5s} {'quotes':>7s} {'bad':>4s} {'flags':>6s}  verdict")
    for r in results:
        c = r["counts"]
        print(f"{r['paper']:14s} {c['reactions']:3d} {c['off_pathway_reactions']:4d} "
              f"{c['summary_statements']:4d} {c['interactions']:4d} {c['transports']:4d} "
              f"{c['excluded_statements']:5d} {r.get('quotes_total', 0):7d} "
              f"{len(r.get('quotes_unverified') or []):4d} {len(r['flags']):6d}  "
              f"{'OK' if r['ok'] else 'PROBLEMS'}")
    bad = [r for r in results if not r["ok"]]
    if bad:
        print()
        for r in bad:
            print(f"-- {r['paper']}")
            for p in r["problems"]:
                print(f"     {p}")
            for u in (r.get("quotes_unverified") or [])[:5]:
                print(f"     unverified {u['row']} ({u['kind']}): {u['quote']}")
    print()
    print(f"{len(results) - len(bad)} of {len(results)} drafts pass independent verification")
    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=2, ensure_ascii=False),
                                  encoding="utf-8")
        print(f"wrote {args.out}")
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
