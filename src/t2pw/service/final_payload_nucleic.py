"""Final payload, D-W20c: settle the DNA/RNA type of a nucleic-acid row FROM THE PAPER, deterministically (no LLM).

PathWhiz requires ``nucleotide-type`` DNA or RNA on a nucleic acid; the name alone rarely settles it (``CHLH promoter``,
``miR171c``). Flag ``T2PW_FINAL_PAYLOAD`` only (the caller checks it). For every ``entities.nucleic_acids`` row that has no
database id and whose type the name does not settle (``writer.nucleotide_type_for`` is None), find the row's mentions in the
paper text and read a type cue next to each.

Rules (each justified in ``docs/pwml_output/evidence/final_payload_p4/c_nucleic_paper.md``):
* Mention: the row name or one of its ``synonyms``, case-insensitive, whole token (no letter/digit on either side, and no ``.``
  ``-`` ``/`` joining it to a further letter/digit: ``LHCB1`` is not a mention inside ``LHCB1.1``, nor ``RBCS`` inside ``RBCS-1A``);
  spaces and hyphens in the name match any run of spaces/hyphens.
* Cue window: the 2 word tokens before and after the mention, inside the same sentence. A whole-sentence window would hand a
  neighbouring entity's cue to this one ("the CHLH promoter and miR171c transcript"); an adjacent head noun ("X mRNA", "X
  promoter region", "plasmid X") is the claim being read.
* RNA cues: rna mrna mirna sirna shrna lncrna rrna trna srna ncrna snrna snorna transcript(s). DNA cues: dna cdna gdna gene(s)
  promoter(s) locus loci plasmid(s) genomic allele(s) knockout enhancer(s). "transcription of" / "expression of" are NOT cues:
  they are said of a gene as readily as of its RNA.
* A mention with both kinds in its window is conflicting; a mention with none is neutral. All cue-bearing mentions must agree,
  else the row is unresolved (``conflicting``). No cue anywhere is unresolved (``no_cue``), no mention is ``not_mentioned``.
  No majority vote: a symbol used as gene in one place and as transcript in another is exactly the ambiguity to leave alone.
* Intersection: a type is taken only if the window rule AND the whole-sentence rule (the same agreement logic with every cue of
  each mention's sentence) give the same type. A window hit the whole sentence contradicts is ``conflicting``.
* Protein-like names (writer ``_PROTEIN_TOKENS``) are never typed here.
The result goes on the row as ``nucleotide_type_paper`` plus an ``entries`` item in ``final_payload_fill`` with the quoted
sentence, the cue and the character offset. Fails open: any error leaves the payload exactly as it was.
"""

from __future__ import annotations

import copy
import datetime
import re
from typing import Any, Dict, List, Optional

WINDOW = 2
RNA_CUES = frozenset("rna mrna mirna sirna shrna lncrna rrna trna srna ncrna snrna snorna transcript transcripts".split())
DNA_CUES = frozenset("dna cdna gdna gene genes promoter promoters locus loci plasmid plasmids genomic allele alleles "
                     "knockout enhancer enhancers".split())
QUOTE_MAX = 300
_ID_KEYS = ("pathwhiz_id", "pathbank_nucleic_acid_id", "pw_nucleic_acid_id")
_TOKEN = re.compile(r"[A-Za-z0-9]+")
_SENT_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(\[])|\n\s*\n")


def _names(row: Dict[str, Any]) -> List[str]:
    out = [row.get("name")]
    syn = row.get("synonyms")
    for s in syn if isinstance(syn, list) else []:
        out.append(s.get("name") if isinstance(s, dict) else s)
    seen: List[str] = []
    for n in out:
        n = " ".join(str(n).split()) if isinstance(n, str) else ""
        if len(n) >= 2 and n.casefold() not in [x.casefold() for x in seen]:
            seen.append(n)
    return seen


def _pattern(name: str) -> "re.Pattern[str]":
    parts = [re.escape(p) for p in re.split(r"[\s\-]+", name) if p]
    return re.compile(r"(?<![A-Za-z0-9])(?<![A-Za-z0-9][.\-/])" + r"[\s\-]+".join(parts) + r"(?![A-Za-z0-9])(?![.\-/][A-Za-z0-9])", re.I)


def _sentences(text: str):
    start = 0
    for m in _SENT_END.finditer(text):
        yield start, text[start:m.start()]
        start = m.end()
    yield start, text[start:]


def _quote(sent: str, a: int) -> str:
    sent = " ".join(sent.split())
    if len(sent) <= QUOTE_MAX:
        return sent
    lo = max(0, min(a, len(sent) - QUOTE_MAX) - QUOTE_MAX // 4)
    return ("..." if lo else "") + sent[lo:lo + QUOTE_MAX] + "..."


def mentions(text: str, row: Dict[str, Any], window: Optional[int] = WINDOW) -> List[Dict[str, Any]]:
    """Every mention of the row: {offset, sentence, kinds, cues} with the cues found in the window (None = whole sentence)."""
    pats = [_pattern(n) for n in _names(row)]
    found: List[Dict[str, Any]] = []
    for s0, sent in _sentences(text):
        toks = [(m.group().casefold(), m.start(), m.end()) for m in _TOKEN.finditer(sent)]
        spans = sorted({(m.start(), m.end()) for p in pats for m in p.finditer(sent)})
        last_end = -1
        for a, b in spans:
            if a < last_end:  # an alias nested in, or overlapping, a longer match of the same row
                continue
            last_end = b
            idx = [i for i, t in enumerate(toks) if t[1] < b and t[2] > a]
            if window is None:
                near = range(len(toks))
            else:
                near = range(max(0, (idx[0] if idx else 0) - window), min(len(toks), (idx[-1] if idx else 0) + 1 + window))
            cues = [(toks[i][0], "RNA" if toks[i][0] in RNA_CUES else "DNA") for i in near
                    if toks[i][0] in RNA_CUES or toks[i][0] in DNA_CUES]
            found.append({"offset": s0 + a, "sentence": _quote(sent, a), "cues": cues,
                          "kinds": sorted({k for _, k in cues})})
    return found


def classify(text: str, row: Dict[str, Any], window: Optional[int] = WINDOW) -> Dict[str, Any]:
    """{status: resolved|not_mentioned|no_cue|conflicting, type, cue, offset, quote, mentions, cued}. With a window, resolved only
    if the whole-sentence rule gives the same type (see module docstring)."""
    out = _classify(text, row, window)
    if window is not None and out["status"] == "resolved":
        whole = _classify(text, row, None)
        if whole["status"] != "resolved" or whole["type"] != out["type"]:
            out.update(status="conflicting", type=None)
            for k in ("cue", "offset", "quote"):
                out.pop(k, None)
    return out


def _classify(text: str, row: Dict[str, Any], window: Optional[int]) -> Dict[str, Any]:
    ms = mentions(text, row, window)
    cued = [m for m in ms if m["kinds"]]
    out: Dict[str, Any] = {"mentions": len(ms), "cued": len(cued), "type": None}
    if not ms:
        out["status"] = "not_mentioned"
    elif not cued:
        out["status"] = "no_cue"
    elif len({tuple(m["kinds"]) for m in cued}) != 1 or len(cued[0]["kinds"]) != 1:
        out["status"] = "conflicting"
    else:
        m = cued[0]
        out.update(status="resolved", type=m["kinds"][0], cue=m["cues"][0][0], offset=m["offset"], quote=m["sentence"])
    return out


def apply_paper_nucleic_types(payload: Dict[str, Any], text: Any, *, source: str = "paper text",
                              today: Optional[str] = None, window: Optional[int] = WINDOW) -> Dict[str, Any]:
    """Set ``nucleotide_type_paper`` on the unsettled nucleic-acid rows of ``payload`` in place; return a count report."""
    from t2pw.pwml.writer import _PROTEIN_TOKENS, nucleotide_type_for

    rep: Dict[str, Any] = {"enabled": True, "candidates": 0, "resolved": 0, "no_paper_text": 0, "not_mentioned": 0,
                           "no_cue": 0, "conflicting": 0, "protein_like": 0, "error": None}
    snapshot = copy.deepcopy(payload)
    try:
        ents = payload.get("entities") if isinstance(payload.get("entities"), dict) else {}
        rows = [r for r in ents.get("nucleic_acids") or [] if isinstance(r, dict)]
        todo = [r for r in rows if not any(r.get(k) for k in _ID_KEYS) and nucleotide_type_for(r) is None]
        rep["candidates"] = len(todo)
        if not isinstance(text, str) or not text.strip():
            rep["no_paper_text"] = len(todo)
            return rep
        entries = []
        today = today or datetime.date.today().isoformat()
        for r in todo:
            if any(t in _PROTEIN_TOKENS for t in re.split(r"[^a-z0-9]+", str(r.get("name") or "").casefold())):
                rep["protein_like"] += 1
                continue
            c = classify(text, r, window)
            if c["status"] != "resolved":
                rep[c["status"]] += 1
                continue
            r["nucleotide_type_paper"] = c["type"]
            rep["resolved"] += 1
            entries.append({"kind": "nucleic_acid", "entity": r.get("name"), "field": "nucleotide_type_paper",
                            "accession": c["type"], "source": "paper", "match_type": "paper_cue", "cue": c["cue"],
                            "quote": c["quote"], "offset": c["offset"], "source_file": source,
                            "mentions": c["mentions"], "cued_mentions": c["cued"], "retrieved": today})
        if entries:
            blk = payload.get("final_payload_fill")
            if not isinstance(blk, dict):
                blk = payload["final_payload_fill"] = {"schema": 1, "date": today, "written": 0, "by_field": {}, "entries": []}
            blk["entries"] = list(blk.get("entries") or []) + entries
            blk["written"] = int(blk.get("written") or 0) + len(entries)
            by = blk.setdefault("by_field", {})
            by["nucleic_acid.nucleotide_type_paper"] = by.get("nucleic_acid.nucleotide_type_paper", 0) + len(entries)
    except Exception as exc:  # fail open
        payload.clear()
        payload.update(snapshot)
        rep.update(resolved=0, error=f"{type(exc).__name__}: {exc}")
    return rep
