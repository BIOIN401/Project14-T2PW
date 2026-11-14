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

D-W21(d), ``apply_nucleic_types`` (below): rows the paper leaves unsettled are looked up in NCBI Gene and RNAcentral (exact name or
alias, pathway species, one unambiguous answer) and, if still unsettled, flagged ``nucleotide_type_status: "unknown"``.
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
                              today: Optional[str] = None, window: Optional[int] = WINDOW,
                              reasons: Optional[Dict[int, str]] = None) -> Dict[str, Any]:
    """Set ``nucleotide_type_paper`` on the unsettled nucleic-acid rows of ``payload`` in place; return a count report.
    ``reasons`` (optional out-dict) gets ``id(row) -> why the paper left the row unsettled`` (``apply_nucleic_types`` uses it)."""
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
            if reasons is not None:
                reasons.update({id(r): "no_paper_text" for r in todo})
            return rep
        entries = []
        today = today or datetime.date.today().isoformat()
        for r in todo:
            if any(t in _PROTEIN_TOKENS for t in re.split(r"[^a-z0-9]+", str(r.get("name") or "").casefold())):
                rep["protein_like"] += 1
                if reasons is not None:
                    reasons[id(r)] = "protein_like"
                continue
            c = classify(text, r, window)
            if c["status"] != "resolved":
                rep[c["status"]] += 1
                if reasons is not None:
                    reasons[id(r)] = c["status"]
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


# ---- D-W21(d): what the paper does not settle is looked up online, then flagged unknown --------------------------------------
# Order per row: name rules (writer) -> paper rules (above) -> NCBI Gene / RNAcentral by exact name in the pathway species ->
# ``nucleotide_type_status: "unknown"``. Never a guessed type. Row keys written by ``apply_nucleic_types``:
#   nucleotide_type_status          "resolved" (paper or lookup) | "unknown"
#   nucleotide_type_unknown_reason  why the PAPER did not settle it: conflicting no_cue not_mentioned protein_like no_paper_text
#   nucleotide_type_unknown_lookup  why the LOOKUP did not: see LOOKUP_CODES
#   nucleotide_type                 DNA | RNA, lookup-resolved rows only (the key the writer already reads; no writer change)
#   nucleotide_type_source          lookup-resolved rows: {source, accession, match_type, retrieved, quote, species_taxid, candidates}
# plus one ``final_payload_fill`` entry per row (field nucleotide_type or nucleotide_type_status). Rule text: final_payload_p5/nucleicD.
EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
EBI_SEARCH = "https://www.ebi.ac.uk/ebisearch/ws/rest/rnacentral"  # the search behind rnacentral.org; its own /api/v1/rna has no name filter
MAX_IDS = 20
_SYMBOL = re.compile(r"[A-Za-z][A-Za-z0-9_.\-]{2,39}")  # a gene symbol or alias: one token, no space, slash or leading digit
# NCBI Gene type -> nucleic acid type of the record's name. protein-coding -> DNA is a PROPOSED rule (D-W21d only says "try an online
# lookup"; the product owner has not ruled): accepted only when no paper sentence naming the row has a protein word (see
# ``_protein_context``), else the row stays unknown with lookup code ``protein_context``. NCBI aliases are matched exactly, case-fold
# only. RNA-gene types: the gene's only product is an RNA, and a symbol such as RyhB, MIR171c or tRNA-Leu names that RNA.
# pseudo / other / unknown / biological-region are not decisive and stay unresolved.
GENE_TYPE = {"protein-coding": "DNA", "ncRNA": "RNA", "tRNA": "RNA", "rRNA": "RNA", "snRNA": "RNA", "snoRNA": "RNA",
             "scRNA": "RNA", "miscRNA": "RNA"}
LOOKUP_CODES = ("not_attempted", "no_fetcher", "species_unknown", "species_ambiguous", "no_symbol_name", "not_found",
                "ambiguous", "gene_type_not_decisive", "sources_disagree", "too_many_records", "protein_context",
                "refused_by_a_source",
                "lookup_error", "lookup_skipped_after_error", "slot_filled", "writer_ignores")


class _Fail(Exception):
    """An unresolved lookup; the message is its code."""


def _cf(s: Any) -> str:
    return str(s or "").strip().casefold()


def _protein_context(text: Any, row: Dict[str, Any]) -> bool:
    """True when a paper sentence that names the row (name or a synonym) holds a protein word (writer ``_PROTEIN_TOKENS``): the symbol
    is then used for the protein there (CHL27 as a subunit), so a protein-coding gene type does not make the row DNA."""
    from t2pw.pwml.writer import _PROTEIN_TOKENS
    pats = [_pattern(n) for n in _names(row)]
    for _, sent in _sentences(text if isinstance(text, str) else ""):
        if any(p.search(sent) for p in pats) and any(t in _PROTEIN_TOKENS for t in re.findall(r"[a-z0-9]+", sent.casefold())):
            return True
    return False


def _short(s: str) -> str:
    s = " ".join(str(s).split())
    return s if len(s) <= QUOTE_MAX else s[:QUOTE_MAX] + "..."


def _ncbi_gene(f: Any, names: List[str], taxid: str) -> Dict[str, Any]:
    """NCBI Gene records (live, in ``taxid`` or a descendant) whose official symbol or alias equals a name, case-insensitively."""
    cands: Dict[str, Dict[str, Any]] = {}
    for name in names:
        d = f.get_json(EUTILS + "esearch.fcgi", {"db": "gene", "term": f'"{name}"[Gene Name] AND txid{taxid}[Organism:exp]',
                                                 "retmode": "json", "retmax": MAX_IDS})
        res = (d or {}).get("esearchresult") or {}
        ids = [str(i) for i in res.get("idlist") or []]
        if int(res.get("count") or 0) > MAX_IDS:
            raise _Fail("too_many_records")
        sm = f.get_json(EUTILS + "esummary.fcgi", {"db": "gene", "id": ",".join(ids), "retmode": "json"}) if ids else None
        for i in ids:
            r = ((sm or {}).get("result") or {}).get(i)
            if not isinstance(r, dict):
                continue
            aliases = [a.strip() for a in str(r.get("otheraliases") or "").split(",") if a.strip()]
            if _cf(r.get("name")) == _cf(name):
                mt = "exact_symbol"
            elif _cf(name) in [_cf(a) for a in aliases]:
                mt = "exact_alias"
            else:
                continue
            org = r.get("organism") or {}
            cands.setdefault(i, {"id": i, "symbol": r.get("name"), "aliases": aliases, "match_type": mt, "name": name,
                                 "organism": org.get("scientificname"), "taxid": org.get("taxid")})
    for i, c in list(cands.items()):
        xml = f.get_text(EUTILS + "efetch.fcgi", {"db": "gene", "id": i, "retmode": "xml"}) or ""
        t = re.search(r'<Entrezgene_type value="([^"]*)"', xml)
        st = re.search(r'<Gene-track_status value="([^"]*)"', xml)  # absent = live
        if st and st.group(1) != "live":
            del cands[i]  # discontinued or replaced record
        else:
            c["gene_type"] = t.group(1) if t else None
    if not cands:
        raise _Fail("not_found")
    if len(cands) > 1:
        raise _Fail("ambiguous")  # one candidate only: two live records of the name in the species are two genes (or alias clashes)
    c = next(iter(cands.values()))
    if GENE_TYPE.get(c["gene_type"]) is None:
        raise _Fail("gene_type_not_decisive")
    return {"type": GENE_TYPE[c["gene_type"]], "gene_type": c["gene_type"], "source": "NCBI Gene", "accession": "GeneID:" + c["id"], "match_type": c["match_type"],
            "candidates": ["GeneID:" + i for i in sorted(cands, key=int)],
            "quote": _short('GeneID %s (%s): symbol "%s", aliases "%s", gene type %s; matched "%s" as %s'
                            % (c["id"], c["organism"], c["symbol"], ", ".join(c["aliases"]), c["gene_type"], c["name"], c["match_type"]))}


def _rnacentral(f: Any, names: List[str], taxid: str) -> Dict[str, Any]:
    """RNAcentral entries of exactly ``taxid`` whose ``gene`` field equals a name, case-insensitively. Any entry is an RNA molecule;
    several entries are several sequences of one gene name (all RNA), so they are not an ambiguity, unlike two NCBI gene records."""
    hits: Dict[str, Dict[str, Any]] = {}
    for name in names:
        d = f.get_json(EBI_SEARCH, {"query": 'gene:"%s" AND TAXONOMY:%s' % (name, taxid),
                                    "fields": "description,gene,rna_type,active", "size": 100, "format": "json"})
        for e in (d or {}).get("entries") or []:
            fl = e.get("fields") or {}
            if str(e.get("id", "")).endswith("_" + taxid) and _cf(name) in [_cf(g) for g in fl.get("gene") or []] \
                    and "Active" in (fl.get("active") or ["Active"]):
                hits.setdefault(e["id"], {"id": e["id"], "fields": fl})
    if not hits:
        raise _Fail("not_found")
    h = hits[sorted(hits)[0]]
    fl = h["fields"]
    return {"type": "RNA", "source": "RNAcentral", "accession": h["id"].split("_")[0], "match_type": "exact_gene_name",
            "candidates": sorted(x.split("_")[0] for x in hits),
            "quote": _short("%s: gene %s; rna_type %s; %s" % (h["id"], fl.get("gene"), (fl.get("rna_type") or [None])[0],
                                                              (fl.get("description") or [""])[0]))}


class _Lookup:
    """One run's lookups: shared fetcher, the pathway species, a stop after the first fetch error."""

    def __init__(self, fetcher: Any, taxid: Optional[str], why_no_taxid: str) -> None:
        self.f, self.taxid, self.why, self.failed = fetcher, taxid, why_no_taxid, False

    def resolve(self, row: Dict[str, Any]) -> Dict[str, Any]:
        """{status: resolved, type, source, accession, match_type, quote, candidates} or {status: <LOOKUP_CODE>}."""
        from t2pw.clean_model.fill.http import FetchError
        if self.f is None:
            return {"status": "no_fetcher"}
        if not self.taxid:
            return {"status": self.why}
        names = [n for n in _names(row) if _SYMBOL.fullmatch(n)]
        if not names:
            return {"status": "no_symbol_name"}
        if self.failed:
            return {"status": "lookup_skipped_after_error"}
        got: List[Any] = []
        try:
            for source in (_ncbi_gene, _rnacentral):
                try:
                    got.append(source(self.f, names, self.taxid))
                except _Fail as e:
                    got.append(str(e))
        except FetchError:
            self.failed = True
            return {"status": "lookup_error"}
        ok = [g for g in got if isinstance(g, dict)]
        why = [g for g in got if g != "not_found" and not isinstance(g, dict)]  # a source that REFUSED (not merely found nothing)
        if len({g["type"] for g in ok}) > 1:
            return {"status": "sources_disagree"}
        if ok and why:  # one source answered, another saw two genes / a pseudogene / too many records: not one unambiguous answer
            return {"status": "refused_by_a_source", "refusals": why, "found_by": [g["source"] for g in ok]}
        if ok:
            return dict(ok[0], status="resolved", confirmed_by=[g["source"] + " " + g["accession"] for g in ok[1:]])
        return {"status": why[0] if why else "not_found"}


def _taxid(payload: Dict[str, Any]) -> tuple:
    """(taxid, why_not): the one NCBI taxonomy id of the payload's species rows."""
    ents = payload.get("entities") if isinstance(payload.get("entities"), dict) else {}
    ids = {str(r.get("taxonomy_id")).strip() for r in ents.get("species") or []
           if isinstance(r, dict) and re.fullmatch(r"\d+", str(r.get("taxonomy_id") or "").strip())}
    return (ids.pop(), "") if len(ids) == 1 else (None, "species_ambiguous" if ids else "species_unknown")


def apply_nucleic_types(payload: Dict[str, Any], text: Any, *, fetcher: Any = None, cache: Any = None, source: str = "paper text",
                        today: Optional[str] = None, window: Optional[int] = WINDOW) -> Dict[str, Any]:
    """Paper rules, then the online lookup, then the unknown flag, in place. Fails open (the lookup step is rolled back on error).
    Lookup rule: one live NCBI Gene record (or RNAcentral gene name) for an exact name or alias in the pathway species, else unresolved."""
    from t2pw.pwml.writer import nucleotide_type_for
    today = today or datetime.date.today().isoformat()
    reasons: Dict[int, str] = {}
    rep = apply_paper_nucleic_types(payload, text, source=source, today=today, window=window, reasons=reasons)
    rep.update(lookup_resolved=0, lookup_by_source={}, lookup_codes={}, unknown=0, unknown_by_reason={}, lookup_requests=0,
               lookup_cache_hits=0)
    if rep["error"]:
        return rep
    snapshot = copy.deepcopy(payload)
    try:
        ents = payload.get("entities") if isinstance(payload.get("entities"), dict) else {}
        rows = [r for r in ents.get("nucleic_acids") or [] if isinstance(r, dict) and not any(r.get(k) for k in _ID_KEYS)]
        for r in rows:  # settled by the paper just now
            if r.get("nucleotide_type_paper") in ("DNA", "RNA"):
                r["nucleotide_type_status"] = "resolved"
        todo = [r for r in rows if id(r) in reasons]  # the rows the paper left unsettled; name-settled rows are not in it
        entries: List[Dict[str, Any]] = []
        cf = None
        if fetcher is not None:
            from t2pw.clean_model.fill.cache import Cache, CachedFetcher
            cf = CachedFetcher(fetcher, cache if cache is not None else Cache(), today)
        taxid, why = _taxid(payload)
        lk = _Lookup(cf, taxid, why)
        for r in todo:
            reason = reasons[id(r)]
            # protein-like names are never typed here; a row the paper itself calls both DNA and RNA stays unknown (a lookup would
            # silently pick one against the paper's own text)
            out = {"status": "not_attempted"} if reason in ("protein_like", "conflicting") else lk.resolve(r)
            if out["status"] == "resolved" and out.get("gene_type") == "protein-coding" and _protein_context(text, r):
                out = {"status": "protein_context"}
            if out["status"] == "resolved" and _cf(r.get("nucleotide_type")):
                out = {"status": "slot_filled"}
            if out["status"] == "resolved":
                r["nucleotide_type"] = out["type"]
                if nucleotide_type_for(r) != out["type"]:  # the writer reads name tokens too; never claim a type it will not write
                    r.pop("nucleotide_type")
                    out = {"status": "writer_ignores"}
            if out["status"] == "resolved":
                src = {"source": out["source"], "accession": out["accession"], "match_type": out["match_type"],
                       "retrieved": today, "quote": out["quote"], "species_taxid": taxid, "candidates": out["candidates"],
                       "confirmed_by": out["confirmed_by"]}
                r.update(nucleotide_type_status="resolved", nucleotide_type_source=src)
                entries.append({"kind": "nucleic_acid", "entity": r.get("name"), "field": "nucleotide_type", "value": out["type"],
                                "accession": out["accession"], "source": out["source"], "match_type": out["match_type"],
                                "quote": out["quote"], "species_taxid": taxid, "candidates": out["candidates"],
                                "retrieved": today, "paper_reason": reason})
                rep["lookup_resolved"] += 1
                rep["lookup_by_source"][out["source"]] = rep["lookup_by_source"].get(out["source"], 0) + 1
                continue
            r.update(nucleotide_type_status="unknown", nucleotide_type_unknown_reason=reason,
                     nucleotide_type_unknown_lookup=out["status"])
            entries.append({"kind": "nucleic_acid", "entity": r.get("name"), "field": "nucleotide_type_status", "value": "unknown",
                            "accession": None, "source": "paper+lookup", "match_type": reason, "lookup": out["status"],
                            "retrieved": today})
            rep["unknown"] += 1
            rep["unknown_by_reason"][reason] = rep["unknown_by_reason"].get(reason, 0) + 1
            rep["lookup_codes"][out["status"]] = rep["lookup_codes"].get(out["status"], 0) + 1
        if cf is not None:
            rep["lookup_requests"], rep["lookup_cache_hits"] = cf.requests, cf.hits
        if entries:
            blk = payload.get("final_payload_fill")
            if not isinstance(blk, dict):
                blk = payload["final_payload_fill"] = {"schema": 1, "date": today, "written": 0, "by_field": {}, "entries": []}
            blk["entries"] = list(blk.get("entries") or []) + entries
            blk["written"] = int(blk.get("written") or 0) + len(entries)
            by = blk.setdefault("by_field", {})
            for e in entries:
                k = "nucleic_acid." + e["field"]
                by[k] = by.get(k, 0) + 1
    except Exception as exc:  # fail open
        payload.clear()
        payload.update(snapshot)
        rep.update(lookup_resolved=0, unknown=0, unknown_by_reason={}, lookup_by_source={}, lookup_codes={},
                   error=f"{type(exc).__name__}: {exc}")
    return rep
