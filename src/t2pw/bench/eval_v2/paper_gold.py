"""Turn a drafter output into a real gold supplement, and verify it (ORCH-747).

EVALUATION-ONLY. Two jobs, both read-only with respect to the pipeline:

``assemble``
    A drafter writes ``t2pw.eval_v2.gold_supplement_draft/1`` (see
    ``prompts/paper_gold_drafter_v1.md``). This fills the ``source`` block from the
    committed source manifest and emits ``t2pw.eval_v2.gold_supplement/1``, the schema
    ``gold.load_supplement`` reads. It then loads its own output with that loader and
    verifies every quote against the pinned text with ``gold.verify_quotes``, so a file
    is never published as gold without the repository's own code agreeing.

    ``supported_reactions_complete`` is written ``false`` whatever the draft says. Only a
    human certifies completeness (D-087 clause 3, applied here by analogy), and a drafter
    certifying its own exhaustiveness would be worthless.

``compare``
    Two drafts of the same paper, one written blind and one given the lab's PathBank
    record as a scope hint, side by side. The hint is the obvious way to contaminate a
    gold: a drafter handed a list of expected processes can start finding them in the
    paper. This prints the agreement so that contamination is measurable rather than
    assumed absent -- rows the two passes share, rows unique to each, and whether the
    hinted pass claims in-scope reactions the blind pass never saw.

Usage::

    python -m t2pw.bench.eval_v2.paper_gold assemble --draft D.json --source S.json \
        --out G.json --fulltext-dir C:/t/eval-fulltext
    python -m t2pw.bench.eval_v2.paper_gold compare --blind B.json --hinted H.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple

from t2pw.bench.eval_v2.gold import load_fulltext, load_supplement, verify_quotes

SUPPLEMENT_SCHEMA = "t2pw.eval_v2.gold_supplement/1"
DRAFT_SCHEMA = "t2pw.eval_v2.gold_supplement_draft/1"

#: Fields the loader requires in ``case.source``, filled from the source manifest.
SOURCE_FIELDS = ("pmcid", "source_url", "fetch_date", "xml_sha256", "text_sha256")

#: Draft-only keys that do not belong in a published case.
DROP_KEYS = ("evidence_required",)


def _fold(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value or "").lower())


def _clean(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items() if k not in DROP_KEYS}
    if isinstance(obj, list):
        return [_clean(v) for v in obj]
    return obj


def assemble(draft: Mapping[str, Any], manifest: Mapping[str, Any]) -> Dict[str, Any]:
    """The draft as a loadable supplement. Raises on anything the loader would reject."""

    if str(draft.get("schema")) != DRAFT_SCHEMA:
        raise ValueError(f"not a drafter output: schema={draft.get('schema')!r}")
    case_id = str(draft.get("case_id") or "").strip()
    if not case_id:
        raise ValueError("the draft names no case_id, so it cannot be keyed")
    case = dict(_clean(draft.get("case") or {}))
    if not case:
        raise ValueError(f"{case_id}: the draft carries no case")

    source = {k: manifest.get(k) for k in SOURCE_FIELDS}
    missing = [k for k, v in source.items() if not v]
    if missing:
        raise ValueError(f"{case_id}: the source manifest is missing {missing}")
    case["source"] = {**source, "citation": manifest.get("citation"),
                      "source_format": manifest.get("source_format"),
                      "text_path_outside_repo": manifest.get("text_path_outside_repo")}
    # Never let a model certify its own exhaustiveness.
    case["supported_reactions_complete"] = False
    case["completeness_status"] = (
        "NOT CERTIFIED. Drafted by a model and verified only for verbatim quotation. A "
        "human must read the paper and certify exhaustiveness before any precision, "
        "recall or F1 is computed from this case.")
    case["drafter"] = dict(draft.get("drafter") or {})
    case["text_revision"] = draft.get("text_revision")

    return {
        "schema": SUPPLEMENT_SCHEMA,
        "description": (
            "Paper gold for the PathBank lab pathways (ORCH-747). A SEPARATE file from "
            "gold_v2_supported_reactions.json, which stays byte-identical: three tests pin "
            "its four cases. Score it by passing --supplement to "
            "t2pw.bench.eval_v2.report, never by merging it into the pinned file."),
        "completeness_definition": (
            "As docs/evaluation_v2/README.md section 1, with one difference: no case here "
            "is certified complete yet, so none of these cases feeds precision, recall, F1 "
            "or the unsupported rate. report.evaluate already skips a case whose flag is "
            "false."),
        "cases": {case_id: case},
    }


def verify(path: Path, fulltext_dir: Optional[Path]) -> Dict[str, Any]:
    """Load an assembled supplement with the production loader and verify its quotes."""

    cases = load_supplement(path)
    out: Dict[str, Any] = {"cases": {}, "ok": True}
    for pid, case in cases.items():
        rec: Dict[str, Any] = {
            "complete": case.complete,
            "in_scope_reactions": len(case.reactions),
            "off_pathway": len(case.off_pathway),
            "summaries": len(case.summaries),
            "excluded": len(case.excluded),
        }
        if fulltext_dir:
            text = load_fulltext(case, fulltext_dir)   # raises on a sha256 mismatch
            if text is None:
                rec["quotes"] = "no pinned text found"
                out["ok"] = False
            else:
                unverified = verify_quotes(case, text)
                rec["unverified_quotes"] = [{"row": r, "quote": q[:120]} for r, q in unverified]
                rec["quotes_verified"] = not unverified
                rec["text_chars"] = len(text)
                if unverified:
                    out["ok"] = False
        out["cases"][pid] = rec
    return out


# ---------------------------------------------------------------------------
# compare: is the hinted pass contaminated by the hint?
# ---------------------------------------------------------------------------

def _rows(case: Mapping[str, Any], bucket: str) -> List[Mapping[str, Any]]:
    return [r for r in (case.get(bucket) or ()) if isinstance(r, dict)]


def _name_forms(raw: str) -> Set[str]:
    """A name, plus its outer name when it carries a trailing parenthetical gloss.

    "araC transcript (araC mRNA)" and "araC transcript" are one entity written two ways,
    and the evaluator's own rule already lets a trailing parenthetical corroborate the
    outer name (matcher.name_verdict). Only the trailing parenthetical is stripped, never
    a prefix, so congeners stay distinct: "acetyl-CoA" never meets "CoA" here.
    """
    text = str(raw or "")
    # The lowercased text with its word boundaries INTACT is kept alongside the folded
    # forms: folding removes spaces, so "araC protein" collapses to one token and the
    # token-containment test in _entity_match can never fire. That cost a measurement.
    forms = {_fold(text), re.sub(r"\s+", " ", text).strip().lower()}
    outer = re.sub(r"\s*\([^()]*\)\s*$", "", text).strip()
    if outer:
        forms.add(_fold(outer))
        forms.add(re.sub(r"\s+", " ", outer).lower())
    return {f for f in forms if f}


def _names(row: Mapping[str, Any], *keys: str) -> Set[str]:
    """The names a row declares on the given slots.

    A slot holds a list of participant dicts (reactions) OR a single string
    (interactions and transports, per the drafter prompt). Iterating a string yields its
    characters, so a missing ``isinstance(value, str)`` branch here silently compares
    single letters and reports agreement between any two rows. That defect produced a
    false "10 of 15 interactions matched" before it was caught.
    """
    out: Set[str] = set()
    for k in keys:
        value = row.get(k)
        items: Sequence[Any]
        if isinstance(value, str):
            items = [value]
        elif isinstance(value, (list, tuple)):
            items = value
        else:
            items = []
        for t in items:
            if isinstance(t, dict) and t.get("name"):
                out |= _name_forms(str(t["name"]))
            elif isinstance(t, str) and t.strip():
                out |= _name_forms(t)
    return out


#: Tokens too generic to carry an identity on their own.
_WEAK_TOKENS = frozenset((
    "the", "a", "an", "of", "at", "in", "to", "and", "or", "with", "for", "by", "form",
    "site", "state", "region", "protein", "gene", "operon", "promoter", "complex", "bound",
    "plus", "minus", "between", "inside", "giving", "together",
))


def _tokens(name: str) -> Set[str]:
    return {t for t in re.split(r"[^a-z0-9]+", str(name or "").lower()) if t}


def _entity_match(a: Set[str], b: Set[str]) -> bool:
    """Do two name sets denote the same entity?

    The two sides are written by different authors at different lengths: a run emits
    "araC protein" where the gold writes "araC protein, P2 form". So a match is token
    containment, the shorter name's tokens inside the longer's, requiring at least one
    shared token that is not generic. Exact equality alone would report no agreement
    anywhere and prove nothing.

    REPORTING ONLY. This is deliberately looser than the evaluator's strict matcher and
    would credit a congener (menaquinone inside menaquinone-7). Nothing that scores
    precision, recall or F1 may use it; ``reactions.score_leg`` and ``matcher`` are
    untouched. Always read the printed pairs rather than trusting the count alone.
    """
    if a & b:
        return True
    for x in a:
        for y in b:
            tx, ty = _tokens(x), _tokens(y)
            if not tx or not ty:
                continue
            small, large = (tx, ty) if len(tx) <= len(ty) else (ty, tx)
            if small <= large and (small - _WEAK_TOKENS):
                return True
    return False


def _sides(row: Mapping[str, Any], bucket: str) -> Tuple[Set[str], Set[str], str]:
    if bucket == "interactions":
        return _names(row, "source"), _names(row, "target"), _fold(row.get("effect"))
    if bucket == "transports":
        return _names(row, "entity"), set(), _fold(row.get("to_compartment"))
    return _names(row, "inputs"), _names(row, "outputs"), ""


def _same_statement(a: Mapping[str, Any], b: Mapping[str, Any], bucket: str) -> bool:
    """Do two rows, written by different passes, state the same thing?

    Exact set equality of names is useless here: two drafters name the same entity
    differently ("transcription initiation at Pc" against "araC DNA") and every row then
    looks unique, which would report zero agreement where the biology plainly agrees. So
    a match needs an overlap on each populated side, or the same verbatim quote, which is
    the strongest evidence that two rows rest on one sentence.
    """
    if _fold(a.get("quote")) and _fold(a.get("quote")) == _fold(b.get("quote")):
        return True
    a0, a1, ak = _sides(a, bucket)
    b0, b1, bk = _sides(b, bucket)
    if ak and bk and ak != bk:
        return False
    for x, y in ((a0, b0), (a1, b1)):
        if (x or y) and not _entity_match(x, y):
            return False
    return _entity_match(a0, b0) or _entity_match(a1, b1)


def _related(a: Mapping[str, Any], b: Mapping[str, Any], bucket: str) -> bool:
    """A weaker test: do the two rows concern the same thing at all?

    Used only for the contamination flag. Two passes can model one event differently --
    for this paper, one gave transcription the template DNA as its substrate and the other
    the nucleotides -- which is a modelling difference worth seeing, not evidence that the
    hinted pass invented a reaction. The flag should fire only when a hinted row has no
    counterpart whatsoever.
    """
    if _fold(a.get("quote")) and _fold(a.get("quote")) == _fold(b.get("quote")):
        return True
    a0, a1, ak = _sides(a, bucket)
    b0, b1, bk = _sides(b, bucket)
    if ak and bk and ak != bk:
        return False
    return _entity_match(a0 | a1, b0 | b1)


def _label(row: Mapping[str, Any]) -> str:
    return str(row.get("label") or row.get("id") or "")


def compare(blind: Mapping[str, Any], hinted: Mapping[str, Any]) -> Dict[str, Any]:
    """Agreement between a blind and a hint-given draft of the same paper."""

    b, h = blind.get("case") or {}, hinted.get("case") or {}
    res: Dict[str, Any] = {"case_id": blind.get("case_id"),
                           "match_rule": "same verbatim quote, or an overlapping name on "
                                         "each populated side with a matching effect",
                           "buckets": {}}
    for bucket in ("reactions", "off_pathway_reactions", "summary_statements",
                   "interactions", "transports"):
        br, hr = _rows(b, bucket), _rows(h, bucket)
        taken: Set[int] = set()
        pairs: List[Dict[str, str]] = []
        for x in br:
            for j, y in enumerate(hr):
                if j in taken or not _same_statement(x, y, bucket):
                    continue
                taken.add(j)
                pairs.append({"blind": _label(x), "hinted": _label(y)})
                break
        matched_blind = {p["blind"] for p in pairs}
        unrelated = [_label(y) for j, y in enumerate(hr) if j not in taken
                     and not any(_related(x, y, bucket) for x in br)]
        res["buckets"][bucket] = {
            "blind": len(br), "hinted": len(hr), "matched": len(pairs),
            "blind_only": [_label(x) for x in br if _label(x) not in matched_blind],
            "hinted_only": [_label(y) for j, y in enumerate(hr) if j not in taken],
            "hinted_only_and_unrelated": unrelated,
            "matched_pairs": pairs,
        }
    bq = {_fold(q) for r in _rows(b, "reactions") + _rows(b, "off_pathway_reactions")
          + _rows(b, "interactions") for q in [r.get("quote")] if q}
    hq = {_fold(q) for r in _rows(h, "reactions") + _rows(h, "off_pathway_reactions")
          + _rows(h, "interactions") for q in [r.get("quote")] if q}
    res["quotes"] = {"blind": len(bq), "hinted": len(hq), "shared_verbatim": len(bq & hq)}
    rx = res["buckets"]["reactions"]
    res["contamination_check"] = {
        "rule": "an in-scope reaction in the hinted pass that is not even related to any "
                "reaction the blind pass found. A different modelling of the same event is "
                "reported under hinted_only, not here.",
        "in_scope_reactions_unrelated_to_anything_blind": rx["hinted_only_and_unrelated"],
        "in_scope_reactions_modelled_differently": [p for p in rx["hinted_only"]
                                                    if p not in rx["hinted_only_and_unrelated"]],
        "verdict": ("clean: the hint added no in-scope reaction unrelated to what the blind "
                    "pass found" if not rx["hinted_only_and_unrelated"] else
                    "REVIEW: the hinted pass claims in-scope reactions with no blind counterpart"),
    }
    return res


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("assemble")
    a.add_argument("--draft", required=True)
    a.add_argument("--source", required=True, help="the committed source manifest")
    a.add_argument("--out", required=True)
    a.add_argument("--fulltext-dir", default=None, help="directory holding <PMCID>.txt")
    c = sub.add_parser("compare")
    c.add_argument("--blind", required=True)
    c.add_argument("--hinted", required=True)
    c.add_argument("--out", default=None)
    args = ap.parse_args(list(argv) if argv is not None else None)

    if args.cmd == "compare":
        res = compare(json.loads(Path(args.blind).read_text(encoding="utf-8")),
                      json.loads(Path(args.hinted).read_text(encoding="utf-8")))
        text = json.dumps(res, indent=1, ensure_ascii=False)
        if args.out:
            Path(args.out).write_text(text + "\n", encoding="utf-8")
        print(text)
        return 0

    supplement = assemble(json.loads(Path(args.draft).read_text(encoding="utf-8")),
                          json.loads(Path(args.source).read_text(encoding="utf-8")))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(supplement, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    res = verify(out, Path(args.fulltext_dir) if args.fulltext_dir else None)
    print(json.dumps({"wrote": str(out), **res}, indent=1, ensure_ascii=False))
    return 0 if res["ok"] else 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
