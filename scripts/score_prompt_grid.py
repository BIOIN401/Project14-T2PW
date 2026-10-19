"""Score a prompt grid's payloads against gold, one row per variant per paper.

Round 1 compared variants by counting rows on a single paper, which cannot tell a prompt
that extracts more from a prompt that extracts better. Two of the papers here carry
CERTIFIED EXHAUSTIVE gold with a verbatim quote per reaction, so this reports real
precision, recall and F1 through the evaluator's own ``reactions.score_leg`` — the same
instrument eval-v2 uses — instead of counts.

It deliberately does NOT emit a single composite score. The question is which prompt works
best across every kind of paper, and an average hides the variant that wins on one type by
losing another. Per-paper numbers are printed side by side, and a variant whose F1 spread
across papers is wide is flagged rather than ranked.

Interactions are not scored by the evaluator (it scores two-sided conversions only), so
they are measured here directly: count, values outside the allowed vocabulary, rows with no
evidence, and restatements of one claim at two resolutions.

Usage::

    python scripts/score_prompt_grid.py --grid C:/t/prompt-grid/r2 \
      --fulltext-dir C:/t/eval-fulltext \
      --paper-gold docs/evaluation_v2/pathbank_gold/gold_paper_pw701720.json \
      --out C:/t/prompt-grid/r2/scores.json
"""

from __future__ import annotations

import argparse
import itertools
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from t2pw.bench.eval_v2 import reactions as R  # noqa: E402
from t2pw.bench.eval_v2.gold import load_fulltext, load_supplement  # noqa: E402

ALLOWED_RELATIONSHIPS = {"activation", "inhibition", "binding", "interaction"}


def _norm(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value or "").lower()).strip()


def interaction_measures(payload: Dict[str, Any]) -> Dict[str, Any]:
    """What the evaluator does not score. Regulation is not a conversion."""

    rows = [r for r in ((payload.get("processes") or {}).get("interactions") or [])
            if isinstance(r, dict)]
    bad = [str(r.get("relationship") or "").strip().lower() for r in rows]
    bad = [b for b in bad if b and b not in ALLOWED_RELATIONSHIPS]
    no_evidence = sum(1 for r in rows if not str(r.get("evidence") or "").strip())

    # One claim written twice at different grain: same actor, same sign, and one target's
    # name contains the other's ("represses Pc promoter" beside "represses araC gene").
    restatements = 0
    for a, b in itertools.combinations(rows, 2):
        if _norm(a.get("entity_1")) != _norm(b.get("entity_1")):
            continue
        if _norm(a.get("relationship")) != _norm(b.get("relationship")):
            continue
        ta, tb = _norm(a.get("entity_2")), _norm(b.get("entity_2"))
        if not ta or not tb or ta == tb:
            continue
        if ta in tb or tb in ta:
            restatements += 1

    return {"interactions": len(rows), "vocabulary_violations": len(bad),
            "offending_values": sorted(set(bad)), "rows_without_evidence": no_evidence,
            "restatements": restatements}


def score_one(payload: Dict[str, Any], case: Any, text: Optional[str]) -> Dict[str, Any]:
    metrics = R.score_leg(case, payload, [], text, expected_species=case.species)
    table1 = metrics.get("table1_paper_extraction") or {}
    entity = metrics.get("entity_accuracy") or {}
    evidence = metrics.get("evidence") or {}
    return {
        "gold_reactions": table1.get("gold_reactions"),
        "matched": len(table1.get("gold_reactions_matched") or []),
        "precision": table1.get("precision"),
        "recall": table1.get("recall"),
        "f1": table1.get("f1"),
        "rows_claimed": table1.get("rows_claimed_as_paper_extraction"),
        "row_classes": metrics.get("row_classes"),
        "substrate_accuracy": entity.get("substrate_accuracy"),
        "product_accuracy": entity.get("product_accuracy"),
        "evidence_support_rate": evidence.get("evidence_support_rate"),
        "certified_gold": bool(case.complete),
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("--grid", required=True,
                    help="directory (or parent of per-paper directories) of *__payload.json")
    ap.add_argument("--fulltext-dir", default="C:/t/eval-fulltext")
    ap.add_argument("--paper-gold", default=None,
                    help="an extra supplement file, e.g. the uncertified araC paper gold")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(list(argv) if argv is not None else None)

    cases = dict(load_supplement())
    if args.paper_gold:
        extra = load_supplement(Path(args.paper_gold))
        for pid, case in extra.items():
            cases.setdefault(pid, case)

    ft = Path(args.fulltext_dir)
    texts: Dict[str, Optional[str]] = {}
    for pid, case in cases.items():
        try:
            texts[pid] = load_fulltext(case, ft)
        except Exception as exc:  # noqa: BLE001 - a hash mismatch must be visible, not fatal
            texts[pid] = None
            print(f"  WARNING {pid}: pinned text unusable ({exc})", file=sys.stderr)

    payloads = sorted(Path(args.grid).rglob("*__payload.json"))
    if not payloads:
        raise SystemExit(f"no *__payload.json under {args.grid}")

    rows: List[Dict[str, Any]] = []
    for path in payloads:
        stem = path.name.replace("__payload.json", "")
        variant, _, paper_hint = stem.partition("__")
        paper = next((pid for pid in cases if pid in str(path)), None) \
            or next((pid for pid in cases if pid in paper_hint), None)
        payload = json.loads(path.read_text(encoding="utf-8"))
        row: Dict[str, Any] = {"variant": variant, "paper": paper or "(unknown)",
                               "payload": str(path), **interaction_measures(payload)}
        if paper and paper in cases:
            try:
                row.update(score_one(payload, cases[paper], texts.get(paper)))
            except Exception as exc:  # noqa: BLE001
                row["score_error"] = f"{type(exc).__name__}: {exc}"
        rows.append(row)

    papers = sorted({r["paper"] for r in rows})
    variants = sorted({r["variant"] for r in rows})

    print(f"{'variant':30s} " + "  ".join(f"{p[:13]:>13s}" for p in papers)
          + "   spread   worst")
    print("-" * (30 + 15 * len(papers) + 18))
    ranked = []
    for v in variants:
        f1s: Dict[str, Optional[float]] = {}
        for p in papers:
            row = next((r for r in rows if r["variant"] == v and r["paper"] == p), None)
            f1s[p] = (row or {}).get("f1")
        present = [f for f in f1s.values() if isinstance(f, (int, float))]
        spread = (max(present) - min(present)) if len(present) > 1 else None
        worst = min(present) if present else None
        cells = "  ".join(
            (f"{f1s[p]:13.3f}" if isinstance(f1s[p], (int, float)) else f"{'-':>13s}")
            for p in papers)
        print(f"{v:30s} {cells}   "
              f"{(f'{spread:.3f}' if spread is not None else '   -   '):>7s} "
              f"{(f'{worst:.3f}' if worst is not None else '  -  '):>7s}")
        ranked.append({"variant": v, "f1_by_paper": f1s, "spread": spread, "worst": worst})

    print()
    print("Ranked by WORST paper, because the goal is a prompt that serves every kind:")
    for entry in sorted((r for r in ranked if r["worst"] is not None),
                        key=lambda r: (-r["worst"], r["spread"] or 0)):
        print(f"  {entry['variant']:30s} worst F1 {entry['worst']:.3f}"
              f"   spread {entry['spread']:.3f}" if entry["spread"] is not None
              else f"  {entry['variant']:30s} worst F1 {entry['worst']:.3f}")

    print()
    print("Interactions (not scored by the evaluator; araC is where they matter):")
    print(f"  {'variant':30s} {'paper':14s} {'inter':>5s} {'badvocab':>8s} "
          f"{'noevid':>6s} {'restate':>7s}")
    for r in sorted(rows, key=lambda r: (r["variant"], r["paper"])):
        print(f"  {r['variant']:30s} {r['paper'][:14]:14s} {r['interactions']:5d} "
              f"{r['vocabulary_violations']:8d} {r['rows_without_evidence']:6d} "
              f"{r['restatements']:7d}")

    out = {"schema": "t2pw.prompt_grid_scores/1", "rows": rows, "ranked": ranked,
           "note": "No composite score on purpose: an average hides a variant that wins "
                   "on one paper type by losing another. Ranked by worst paper."}
    if args.out:
        Path(args.out).write_text(json.dumps(out, indent=2, ensure_ascii=False),
                                  encoding="utf-8")
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
