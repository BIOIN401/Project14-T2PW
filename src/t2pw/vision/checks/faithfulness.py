"""Faithfulness checks (development only): is what a view shows backed by the source it cites?

    find_paper_text(run_dir)       -> (text | None, path | None)
    paper_view(big, text)          -> every element with a ``paper`` record and a quote: is the quote in the paper text?
    extended_view(big, cache)      -> every ``database`` element re-found in its cached record, every ``other_paper``
                                      quote found in its cited text, from a cache only (nothing is fetched)

Quotes are compared after normalisation: Unicode NFKC, dashes and quote marks folded, case folded, whitespace collapsed.
A quote written with an ellipsis ("a ... b") is found when each fragment of 12+ characters is found; those are counted
apart from verbatim finds. The extension cache (V4 / V8 write it) is either a dict ``source id -> text`` or a directory
holding ``<source id with every non-alphanumeric character as "_">.<txt|json|xml|html>``. No file, no fetch: the element
is counted as ``no_cache``.
"""
from __future__ import annotations

import re
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from t2pw.vision.payload import evidence_quote, grounded_ids, is_currency, iter_sections, keyed_rows, refs

_FOLD = str.maketrans({"‐": "-", "‑": "-", "‒": "-", "–": "-", "—": "-", "−": "-",
                       "‘": "'", "’": "'", "“": '"', "”": '"', " ": " "})
TEXT_NAMES = ("01_source_text.txt", "source_text.txt")


def norm_text(s: Any) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", str(s or "")).translate(_FOLD).casefold()).strip()


def find_paper_text(run_dir) -> Tuple[Optional[str], Optional[str]]:
    """The stored source text of a run: the first 01_source_text.txt in the run folder or up to four parents above it."""
    d = Path(run_dir)
    for p in [d] + list(d.parents)[:4]:
        for n in TEXT_NAMES:
            if (p / n).is_file():
                return (p / n).read_text(encoding="utf-8", errors="replace"), str(p / n)
    return None, None


def quote_in(quote: str, text_norm: str) -> str:
    """'verbatim', 'fragments' (every 12+ character piece of an ellipsis quote is found) or 'no'."""
    q = norm_text(quote)
    if q and q in text_norm:
        return "verbatim"
    parts = [x.strip(" .,;:") for x in re.split(r"\.\.\.|…|\[\.\.\.\]", q)]
    parts = [x for x in parts if len(x) >= 12]
    return "fragments" if len(parts) > 1 and all(x in text_norm for x in parts) else "no"


def _tally(rows: List[Tuple[str, str, str]], n_text: Optional[str]) -> Dict[str, Any]:
    out = {"n": len(rows), "verbatim": 0, "fragments": 0, "not_found": 0, "no_text": 0, "not_found_examples": []}
    for key, quote, _ in rows:
        if n_text is None:
            out["no_text"] += 1
            continue
        hit = quote_in(quote, n_text)
        if hit == "no":
            out["not_found"] += 1
            if len(out["not_found_examples"]) < 5:
                out["not_found_examples"].append({"key": key, "quote": quote[:160]})
        else:
            out[hit] += 1
    found = out["verbatim"] + out["fragments"]
    out["found_rate"] = round(found / out["n"], 4) if out["n"] and n_text is not None else None
    return out


def paper_view(big: Dict[str, Any], text: Optional[str]) -> Dict[str, Any]:
    """Every (element, ``paper`` record) with a non-empty quote against the paper text, plus VD-10: elements whose only
    lineage is audit_modified (kind ``curation``; the paper view drops them) whose own evidence quote is verbatim in
    the paper, the candidates for promotion."""
    n_text = norm_text(text) if text else None
    rows = {key: row for _, key, row, _ in keyed_rows(big)}
    quoted, audit = [], []
    for key, recs in (big.get("provenance") or {}).items():
        for r in recs:
            if r.get("kind") == "paper" and str(r.get("quote") or "").strip():
                quoted.append((key, str(r["quote"]), ""))
        if recs and all(r.get("kind") == "curation" for r in recs):
            q = evidence_quote(rows.get(key) or {}) if isinstance(rows.get(key), dict) else None
            audit.append((key, q or "", ""))
    with_q = [a for a in audit if a[1]]
    res = _tally(quoted, n_text)
    res["text"] = "found" if n_text is not None else "no text"
    t = _tally(with_q, n_text)
    res["audit_modified_only"] = {"elements": len(audit), "without_quote": len(audit) - len(with_q), "with_quote": len(with_q),
                                  "quote_verbatim": t["verbatim"], "quote_fragments": t["fragments"],
                                  "quote_not_found": t["not_found"], "no_text": t["no_text"],
                                  "promotion_candidates": t["verbatim"] if n_text is not None else None}
    return res


def _cached(cache, sid: str) -> Optional[str]:
    if cache is None or not sid:
        return None
    if isinstance(cache, dict):
        v = cache.get(sid)
        return v if isinstance(v, str) else (None if v is None else str(v))
    stem = re.sub(r"[^0-9A-Za-z]", "_", sid)
    for ext in ("txt", "json", "xml", "html"):
        p = Path(cache) / ("%s.%s" % (stem, ext))
        if p.is_file():
            return p.read_text(encoding="utf-8", errors="replace")
    return None


def _in_record(section: str, row: Any, rec_norm: str) -> bool:
    """The element is in its database record: its name or a synonym, a grounded id, or (a process) every non-currency participant."""
    if not isinstance(row, dict):
        return False
    if any(n and n in rec_norm for n in [norm_text(row.get("name"))] + [norm_text(s) for s in row.get("synonyms") or []]):
        return True
    if any(v and v in rec_norm for _, v in grounded_ids(row)):
        return True
    if section.startswith("processes."):
        names = [n for n in refs(section, row)[0] if not is_currency(n)]
        return bool(names) and all(norm_text(n) in rec_norm for n in names)
    return False


def extended_view(big: Dict[str, Any], cache=None) -> Dict[str, Any]:
    """``database`` elements re-found in their cached source record; ``other_paper`` quotes found in the cited text.
    ``status`` is ``no extension`` when the overlay has no rows (the checks still run on base rows that cite sources)."""
    ext_rows = sum(len(r) for _, r in iter_sections(big.get("extension") or {}))
    rows = {key: (section, row, is_base) for section, key, row, is_base in keyed_rows(big)}
    db = {"n": 0, "found": 0, "not_found": 0, "no_cache": 0, "not_found_examples": []}
    op = {"n": 0, "verbatim": 0, "fragments": 0, "not_found": 0, "no_cache": 0, "not_found_examples": []}
    for key, recs in (big.get("provenance") or {}).items():
        section, row, _ = rows.get(key, ("", None, True))
        for r in recs:
            if r.get("kind") == "database":
                db["n"] += 1
                text = _cached(cache, r.get("source") or "")
                if text is None:
                    db["no_cache"] += 1
                elif _in_record(section, row, norm_text(text)):
                    db["found"] += 1
                else:
                    db["not_found"] += 1
                    if len(db["not_found_examples"]) < 5:
                        db["not_found_examples"].append({"key": key, "source": r.get("source")})
            elif r.get("kind") == "other_paper" and str(r.get("quote") or "").strip():
                op["n"] += 1
                text = _cached(cache, r.get("source") or "")
                if text is None:
                    op["no_cache"] += 1
                    continue
                hit = quote_in(str(r["quote"]), norm_text(text))
                if hit == "no":
                    op["not_found"] += 1
                    if len(op["not_found_examples"]) < 5:
                        op["not_found_examples"].append({"key": key, "source": r.get("source"), "quote": str(r["quote"])[:160]})
                else:
                    op[hit] += 1
    return {"status": "extension" if ext_rows else "no extension", "extension_rows": ext_rows,
            "database": db, "other_paper": op}
