"""Loader and verifier for the v2 supported-reaction supplement (ORCH-742c).

The supplement is ``gold_v2_supported_reactions.json`` beside this module. It is
keyed by pinned-v1 case id and is read ONLY here: ``bench.goldset`` never sees it,
and it sits outside ``bench/gold/`` so no glob over that directory can either.

Quote verification folds exactly like ``bench.goldset.fold_for_quote`` (case,
whitespace and punctuation dropped). Two texts can be checked:

* the EuropePMC full text the labels were written against, kept OUTSIDE the repo;
  its sha256 is pinned per case, so a different download is refused, not trusted;
* the committed ``01_source_text.txt`` a run stored, which is what makes a v2
  number reproducible from the repository alone.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

from t2pw.bench.goldset import fold_for_quote

SUPPLEMENT_PATH = Path(__file__).resolve().parent / "gold_v2_supported_reactions.json"
ROLES = ("defining", "cofactor")


class GoldV2Error(ValueError):
    """The supplement is malformed. Always names the offending case and row."""


@dataclass(frozen=True)
class Term:
    name: str
    aliases: Tuple[str, ...] = ()
    role: str = "defining"


@dataclass(frozen=True)
class Reaction:
    id: str
    label: str
    inputs: Tuple[Term, ...]
    outputs: Tuple[Term, ...]
    enzyme: Optional[Term]
    reversible: bool
    quote: str
    supporting_quotes: Tuple[str, ...] = ()
    catalytic_cofactors: Tuple[Term, ...] = ()
    kind: str = "in_scope"

    def defining(self, side: str) -> Tuple[Term, ...]:
        return tuple(t for t in getattr(self, side) if t.role == "defining")

    def cofactors(self, side: str) -> Tuple[Term, ...]:
        return tuple(t for t in getattr(self, side) if t.role == "cofactor")

    def quotes(self) -> Tuple[str, ...]:
        return (self.quote, *self.supporting_quotes)


@dataclass(frozen=True)
class CaseV2:
    paper_id: str
    complete: bool
    species: str
    scope: str
    source: Mapping[str, Any]
    reactions: Tuple[Reaction, ...]
    off_pathway: Tuple[Reaction, ...]
    excluded: Tuple[Mapping[str, str], ...] = field(default_factory=tuple)
    summaries: Tuple[Reaction, ...] = ()


def _term(raw: Any, where: str) -> Term:
    if not isinstance(raw, dict) or not str(raw.get("name") or "").strip():
        raise GoldV2Error(f"{where}: a participant needs a 'name'")
    role = str(raw.get("role") or "defining")
    if role not in ROLES:
        raise GoldV2Error(f"{where}: role must be one of {ROLES}, got {role!r}")
    return Term(str(raw["name"]).strip(), tuple(str(a) for a in raw.get("aliases") or ()), role)


def _reaction(raw: Mapping[str, Any], where: str, kind: str) -> Reaction:
    inputs = tuple(_term(t, f"{where}.inputs") for t in raw.get("inputs") or ())
    outputs = tuple(_term(t, f"{where}.outputs") for t in raw.get("outputs") or ())
    if not any(t.role == "defining" for t in inputs) or not any(t.role == "defining" for t in outputs):
        raise GoldV2Error(f"{where}: both sides need at least one defining participant")
    quote = str(raw.get("quote") or "").strip()
    if not quote:
        raise GoldV2Error(f"{where}: a reaction without its quote is an assertion, not evidence")
    enz = raw.get("enzyme")
    return Reaction(
        id=str(raw.get("id") or ""),
        label=str(raw.get("label") or ""),
        inputs=inputs,
        outputs=outputs,
        enzyme=_term(enz, f"{where}.enzyme") if enz else None,
        reversible=bool(raw.get("reversible", False)),
        quote=quote,
        supporting_quotes=tuple(str(q) for q in raw.get("supporting_quotes") or ()),
        catalytic_cofactors=tuple(
            _term({**c, "role": "cofactor"}, f"{where}.catalytic_cofactors")
            for c in raw.get("catalytic_cofactors") or ()),
        kind=kind,
    )


def load_supplement(path: Optional[Path] = None) -> Dict[str, CaseV2]:
    target = Path(path) if path else SUPPLEMENT_PATH
    raw = json.loads(target.read_text(encoding="utf-8"))
    cases: Dict[str, CaseV2] = {}
    for pid, c in (raw.get("cases") or {}).items():
        where = f"cases.{pid}"
        if not isinstance(c.get("supported_reactions_complete"), bool):
            raise GoldV2Error(f"{where}: supported_reactions_complete must be an explicit boolean")
        src = c.get("source") or {}
        for key in ("pmcid", "source_url", "fetch_date", "xml_sha256", "text_sha256"):
            if not src.get(key):
                raise GoldV2Error(f"{where}.source: missing {key!r}")
        rx = tuple(_reaction(r, f"{where}.reactions[{i}]", "in_scope")
                   for i, r in enumerate(c.get("reactions") or ()))
        if c["supported_reactions_complete"] and not rx:
            raise GoldV2Error(f"{where}: a complete case with no in-scope reaction is a negative "
                              "control and belongs in pinned_v1's ceiling, not here")
        off = tuple(_reaction(r, f"{where}.off_pathway_reactions[{i}]", str(r.get("kind") or "off_pathway"))
                    for i, r in enumerate(c.get("off_pathway_reactions") or ()))
        summ = tuple(_reaction(r, f"{where}.summary_statements[{i}]", "summary")
                     for i, r in enumerate(c.get("summary_statements") or ()))
        ids = [r.id for r in (*rx, *off, *summ)]
        if len(ids) != len(set(ids)):
            raise GoldV2Error(f"{where}: reaction ids must be unique")
        cases[pid] = CaseV2(
            paper_id=pid,
            complete=c["supported_reactions_complete"],
            species=str(c.get("species") or ""),
            scope=str(c.get("scope") or ""),
            source=src,
            reactions=rx,
            off_pathway=off,
            excluded=tuple(c.get("excluded_statements") or ()),
            summaries=summ,
        )
    return cases


def all_quotes(case: CaseV2) -> List[Tuple[str, str]]:
    """``(row id, quote)`` for every quote the case relies on, excluded statements too."""

    out = [(r.id, q) for r in (*case.reactions, *case.off_pathway, *case.summaries) for q in r.quotes()]
    out += [(f"X{i + 1}", str(x.get("quote") or "")) for i, x in enumerate(case.excluded)]
    return out


def verify_quotes(case: CaseV2, text: str) -> List[Tuple[str, str]]:
    """The ``(row id, quote)`` pairs NOT found in ``text``. Empty means all verified."""

    folded = fold_for_quote(text)
    return [(rid, q) for rid, q in all_quotes(case) if fold_for_quote(q) not in folded]


def load_fulltext(case: CaseV2, fulltext_dir: Path) -> Optional[str]:
    """The fetched full text, or ``None`` if absent. A hash mismatch raises."""

    p = Path(fulltext_dir) / f"{case.paper_id}.txt"
    if not p.is_file():
        return None
    data = p.read_bytes()
    got = hashlib.sha256(data).hexdigest()
    if got != case.source["text_sha256"]:
        raise GoldV2Error(f"{p}: sha256 {got} != pinned {case.source['text_sha256']}; "
                          "this is not the text the labels were written against")
    return data.decode("utf-8")


__all__ = [
    "CaseV2", "GoldV2Error", "Reaction", "SUPPLEMENT_PATH", "Term",
    "all_quotes", "load_fulltext", "load_supplement", "verify_quotes",
]
