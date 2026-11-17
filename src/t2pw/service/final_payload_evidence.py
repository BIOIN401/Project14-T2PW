"""D-W24 (6) and (7c), lane L8: evidence checks on every reaction, enzyme assignment and estimated compartment.

Runs in ``final_payload.maybe_apply_final_id_fill`` after the L6 merge (and after L7's compartment estimate once that lane is
in), with the paper text of the run (``state["final_payload_paper_text"]``). Flag ``T2PW_FINAL_PAYLOAD`` off = the payload is
untouched, so the flag-off path is byte-identical. Deterministic: string and name matching against the paper text and the
payload's own evidence fields; no network, no LLM.

A failing check FLAGS the element: ``review_required`` on the row, a ``review_reasons`` entry ``"<check>: <detail>"`` (the
form lanes L6/L7 use) and one ``evidence_checks`` record per failed check (check, element, the evidence read). Interactions
are not checked here: lane L4 checks them in the IR (``no_evidence``, ``evidence_does_not_name_modifier``,
``evidence_does_not_name_step``). Nothing is removed for a failed check. An element is removed ONLY when the paper contradicts
it, and deciding "contradicts" needs a judgement this module does not make: ``judge`` (optional, default None = offline, no
removal) is called for each flagged element and may return ``{"contradicted": True, "quote": ...}``; the quote must be in the
paper text. A contradicted reaction is marked ``scope_membership: out_of_scope`` with the quote, so the strict quarantine
removes it with its full audit record (never silently); a contradicted enzyme leaves its reaction into the payload's
``withheld_reaction_actors`` record (lane L5's field) with the quote.

Checks (names are the ``review_reasons`` prefixes):
- reactions: ``no_evidence``; ``evidence_is_pathway_name`` (the quote only restates the pathway name / title / organism);
  ``evidence_is_article_header`` (a DOI, PMC id or ISSN: the article metadata block); ``evidence_names_no_participant`` (an
  invented step: the quote names none of the reaction's substrates or products, by name, synonym, the parenthetical half of
  a name or an abbreviation the paper defines); ``evidence_not_in_paper`` (a quote fragment of 3+ words that is not in the
  paper text, verbatim or by 60% of its word trigrams; not asked of a RAG-admitted row, whose quote is another paper's);
  merged steps: ``step_skips_intermediate`` (the payload has S -> X and X -> P
  beside this S -> P; X not currency) and ``evidence_describes_several_steps`` (the quote says "sequential", "followed by",
  "further processed", "two steps" ...).
- enzymes: ``enzyme_not_tied_to_step``: the enzyme's evidence (else the reaction's) must name the enzyme and one of the step's
  non-currency substrates or products; failing that, a paper sentence that names both ties it (recorded, not flagged).
- compartments: ``estimated_compartment_without_evidence``: a row whose ``compartment_source`` is ``estimated`` (L6),
  ``neighbour_estimate`` or ``llm_estimate`` (L7) must carry a ``compartment_estimate`` entry with a reason; an LLM estimate's
  quote must be in the paper text; a neighbour estimate must name its neighbours. ``unresolved`` rows are already flagged by L7
  (``compartment_not_stated``) and are not flagged twice.

D-W24 (7c): :func:`strip_placeholder_species` takes every species field off PathBank's ``Unknown`` protein (9659), which stands
for an enzyme nobody identified and was given Arabidopsis thaliana (PathBank's own record) in every payload. PathWhiz binds the
existing record by name and UniProt value when the PWML gives no species (``pwml_parser.store_object``: an omitted column is
not part of ``find_by``); a new Protein needs a species (``protein.rb`` ``validates :species``), so a database without record
9659 cannot import it either way.

Runs once per paper text: a payload that already carries ``final_payload_evidence`` from a run with the paper text is left
alone by a later call without it (the export seam calls the fill again without the run state).
"""

from __future__ import annotations

import copy
import re
import unicodedata
from typing import Any, Callable, Dict, Iterable, List, Optional, Set

FIELD = "final_payload_evidence"
RULING = "D-W24 6, L8"
CHECKS = (
    "no_evidence", "evidence_is_pathway_name", "evidence_is_article_header", "evidence_names_no_participant",
    "evidence_not_in_paper", "step_skips_intermediate", "evidence_describes_several_steps", "enzyme_not_tied_to_step",
    "estimated_compartment_without_evidence",
)
ESTIMATED_SOURCES = ("estimated", "neighbour_estimate", "llm_estimate")
# ponytail: a fixed share of word trigrams; a paraphrase that keeps under 60% of the paper's wording is flagged.
MIN_TRIGRAM_COVERAGE = 0.6
_HEADER = re.compile(r"\b10\.\d{4,9}/|\bpmc\d{5,}\b|\bissn\b|\bdoi\b", re.IGNORECASE)
# ponytail: wording cues only; "then"/"subsequently" order two stated steps, they do not merge them.
_SEVERAL_STEPS = re.compile(
    r"\b(?:sequential(?:ly)?|followed by|further processed|two[- ]steps?|three[- ]steps?|multi[- ]?steps?|several steps"
    r"|series of (?:reactions|steps)|stepwise)\b", re.IGNORECASE)
_ELLIPSIS = re.compile(r"\.\.\.+|…")
_STATE_WORDS = {"in", "of", "the", "a", "an", "and", "pathway", "biosynthesis", "biosynthetic", "metabolism", "synthesis"}
_NAME_FIELDS = ("name", "raw_name", "short_name", "common_name", "display_name", "gene_name", "abbreviation")
_LIST_FIELDS = ("synonyms", "aliases")
_ENTITY_BUCKETS = ("compounds", "proteins", "protein_complexes", "nucleic_acids", "element_collections")
_FLAG_PREFIXES = tuple(f"{c}:" for c in CHECKS)


def norm_text(value: Any) -> str:
    """Casefolded, NFKC, every non-alphanumeric run a single space, padded with spaces (so ``" x "`` is a word match)."""
    text = unicodedata.normalize("NFKC", str(value or "")).casefold()
    return " " + " ".join(re.findall(r"[^\W_]+", text)) + " "


def _compact(value: str) -> str:
    return value.replace(" ", "")


def _text(value: Any) -> str:
    if isinstance(value, list):
        return " ".join(str(x) for x in value if isinstance(x, str))
    return value if isinstance(value, str) else ""


_PAREN = re.compile(r"^(.*?)\s*\(([^()]+)\)\s*$")
_ABBREV = re.compile(r"([^()\n.;]{3,120}?)\s*\(\s*([A-Za-z0-9][^()\n,;]{0,14}?)\s*[,;)]")


def _aliases(row: Dict[str, Any], *extra: Any) -> List[str]:
    out = [row.get(f) for f in _NAME_FIELDS] + [a for f in _LIST_FIELDS for a in (row.get(f) or [])
                                                  if isinstance(row.get(f), list)] + list(extra)
    for a in list(out):  # "dihydrosphingosine (DHS)" is named as either half
        m = _PAREN.match(a) if isinstance(a, str) else None
        if m:
            out += [m.group(1), m.group(2)]
    seen, names = set(), []
    for a in out:
        n = norm_text(a)
        if n.strip() and n not in seen:
            seen.add(n)
            names.append(n)
        if n.endswith(" complex ") and len(n) > 10:  # "SPT complex" is named as "SPT"
            n2 = n[: -len("complex ")]
            if n2 not in seen:
                seen.add(n2)
                names.append(n2)
    return names


def names_any(text_norm: str, aliases: Iterable[str]) -> str:
    """The first alias the normalized text names (word match; whole-word compact match for long names), else ``""``."""
    compact = None
    for a in aliases:
        if len(a.strip()) < 2:
            continue
        if a in text_norm:
            return a.strip()
        if len(_compact(a)) >= 8:  # "Dimethylallylpyrophosphate" is "dimethylallyl pyrophosphate"
            compact = compact if compact is not None else _compact(text_norm)
            if _compact(a) in compact:
                return a.strip()
    return ""


class _Index:
    """Name -> aliases of every declared entity, and the complex -> component aliases."""

    def __init__(self, payload: Dict[str, Any], paper: str = "") -> None:
        ents = payload.get("entities") if isinstance(payload.get("entities"), dict) else {}
        self.rows: Dict[str, Dict[str, Any]] = {}
        for bucket in _ENTITY_BUCKETS:
            for row in ents.get(bucket) or []:
                if isinstance(row, dict) and norm_text(row.get("name")).strip():
                    self.rows.setdefault(norm_text(row.get("name")), row)
        # the paper's own abbreviations: "3-hydroxy-N6-trimethyllysine (HTML)" -> HTML names that compound
        self.abbrev = [(norm_text(m.group(1)), norm_text(m.group(2))) for m in _ABBREV.finditer(paper or "")]
        self._cache: Dict[str, List[str]] = {}

    def aliases(self, name: str) -> List[str]:
        key = norm_text(name)
        if key in self._cache:
            return self._cache[key]
        row = self.rows.get(key, {})
        out = _aliases(row, name)
        for comp in row.get("components") or []:
            cname = comp if isinstance(comp, str) else (comp or {}).get("name") or (comp or {}).get("protein")
            if isinstance(cname, str) and norm_text(cname).strip() != "unknown":
                out += _aliases(self.rows.get(norm_text(cname), {}), cname)
        longs = [a for a in out if len(a.split()) >= 1 and len(a.strip()) >= 4]
        out += [ab for long_form, ab in self.abbrev if any(long_form.endswith(a) for a in longs) and ab not in out]
        self._cache[key] = out
        return out


def _participants(row: Dict[str, Any]) -> List[str]:
    from t2pw.pwml.ir import _coerce_participants

    return [p["name"] for f in ("inputs", "outputs") for p in _coerce_participants(row.get(f))]


def _side(row: Dict[str, Any], field: str) -> List[str]:
    from t2pw.pwml.ir import _coerce_participants

    return [p["name"] for p in _coerce_participants(row.get(field))]


def _currency(name: str) -> bool:
    from t2pw.pipeline.cofactor_policy import VOCABULARY, _normalize

    return _normalize(name) in VOCABULARY


def _actor(actor: Any) -> str:
    from t2pw.pwml.ir import _actor_name_and_hint

    return _actor_name_and_hint(actor)[0]


def _sentences(paper: str) -> List[str]:
    return [s for s in re.split(r"(?<=[.!?])\s+|\n+", paper) if s.strip()]


class _Paper:
    def __init__(self, text: str) -> None:
        self.raw = text
        self.norm = norm_text(text) if text else ""
        self._sents: Optional[List[tuple]] = None

    def has(self, quote: str) -> bool:
        return bool(self.norm) and norm_text(quote) in self.norm

    def sentences(self) -> List[tuple]:
        if self._sents is None:
            self._sents = [(s.strip(), norm_text(s)) for s in _sentences(self.raw)]
        return self._sents


def _missing_fragments(evidence: str, paper: _Paper) -> List[str]:
    """Quote fragments (split at an ellipsis) of 3+ words the paper text does not carry: not verbatim, and fewer than
    :data:`MIN_TRIGRAM_COVERAGE` of their word trigrams are in it (a quote trimmed of a parenthesis or a citation still
    counts as the paper's)."""
    out = []
    for frag in _ELLIPSIS.split(evidence):
        words = norm_text(frag).split()
        if len(words) < 3 or paper.has(frag):
            continue
        grams = [" " + " ".join(words[k:k + 3]) + " " for k in range(len(words) - 2)]
        if sum(1 for g in grams if g in paper.norm) / len(grams) < MIN_TRIGRAM_COVERAGE:
            out.append(frag.strip())
    return out


def _external_source(row: Dict[str, Any]) -> bool:
    """A RAG-admitted row: its quote is from another paper, so the target paper text is not where it is checked."""
    return bool(row.get("rag_provenance")) or any(
        isinstance(e, dict) and e.get("origin") == "rag_literature" for e in row.get("provenance_lineage") or [])


def _flag(row: Dict[str, Any], record: Dict[str, Any], report: Dict[str, Any]) -> None:
    row["review_required"] = True
    row.setdefault("review_reasons", []).append(f"{record['check']}: {record['detail']} ({RULING})")
    row.setdefault("evidence_checks", []).append(record)
    report["flagged"].append({k: record.get(k) for k in ("kind", "row", "name", "check", "element", "detail")})
    report["counts"][record["check"]] = report["counts"].get(record["check"], 0) + 1


def _clear(row: Dict[str, Any]) -> None:
    """Drop this step's earlier flags from a row (a re-run with the paper text replaces them)."""
    if row.pop("evidence_checks", None) is None:
        return
    reasons = [r for r in row.get("review_reasons") or [] if not str(r).startswith(_FLAG_PREFIXES)]
    if reasons:
        row["review_reasons"] = reasons
    else:
        row.pop("review_reasons", None)
        if row.get("review_required") is True:
            row.pop("review_required", None)


def _restates_pathway(words: Set[str], context: Dict[str, Any]) -> bool:
    """Every word of the quote is a word of the pathway name, the organism or a filler word, and one is the pathway's own."""
    own = set(norm_text(context.get("pathway_name")).split()) - _STATE_WORDS
    allowed = own | set(norm_text(context.get("organism")).split()) | _STATE_WORDS
    return len(words) >= 2 and bool(own & words) and words <= allowed


def _skipped_intermediate(i: int, row: Dict[str, Any], rows: List[Any]) -> Optional[tuple]:
    """``(X, A, B)`` when the payload has A: S -> X and B: X -> P beside this S -> P (X not currency, not in this step)."""
    ins = {norm_text(n) for n in _side(row, "inputs") if not _currency(n)}
    outs = {norm_text(n) for n in _side(row, "outputs") if not _currency(n)}
    if not ins or not outs:
        return None
    others = [(j, r) for j, r in enumerate(rows) if j != i and isinstance(r, dict)]
    for j, a in others:
        if not ins & {norm_text(n) for n in _side(a, "inputs")}:
            continue
        for x in _side(a, "outputs"):
            xn = norm_text(x)
            if xn in ins | outs or _currency(x):
                continue
            for k, b in others:
                if k != j and xn in {norm_text(n) for n in _side(b, "inputs")} \
                        and outs & {norm_text(n) for n in _side(b, "outputs")}:
                    return x, a, b
    return None


def _check_reaction(i: int, row: Dict[str, Any], rows: List[Any], idx: _Index, paper: _Paper,
                    context: Dict[str, Any], report: Dict[str, Any]) -> List[Dict[str, Any]]:
    name = str(row.get("name") or "")
    ev = _text(row.get("evidence")).strip()
    base = {"kind": "reaction", "row": i, "name": name, "element": name, "evidence": ev, "decision": RULING}
    out: List[Dict[str, Any]] = []

    def add(check: str, detail: str, **more: Any) -> None:
        out.append({**base, "check": check, "detail": detail, **more})

    parts = _participants(row)
    if not ev:
        add("no_evidence", "the reaction carries no evidence quote")
    else:
        ev_norm = norm_text(ev)
        if _restates_pathway(set(ev_norm.split()), context):
            add("evidence_is_pathway_name", f"the quote {ev!r} only restates the pathway name")
        if _HEADER.search(ev):
            add("evidence_is_article_header", f"the quote is the article metadata: {ev[:160]!r}")
        if parts and not any(names_any(ev_norm, idx.aliases(p)) for p in parts):
            add("evidence_names_no_participant", f"the quote {ev[:200]!r} names none of {parts}")
        if paper.norm and not _external_source(row):
            gone = _missing_fragments(ev, paper)
            if gone:
                add("evidence_not_in_paper", f"not in the paper text: {gone[0][:160]!r}")
        cue = _SEVERAL_STEPS.search(ev)
        if cue:
            add("evidence_describes_several_steps", f"the quote says {cue.group(0)!r}: {ev[:200]!r}")

    skip = _skipped_intermediate(i, row, rows)
    if skip:
        x, a, b = skip
        add("step_skips_intermediate",
            f"the payload also has {a.get('name')!r} and {b.get('name')!r} through {x!r}; this step skips it",
            intermediate=x, via=[a.get("name"), b.get("name")])

    # enzyme belongs to its step
    steps = [p for p in parts if not _currency(p)] or parts
    for e, actor in enumerate(row.get("enzymes") or []):
        ename = _actor(actor)
        if not ename:
            continue
        aev = _text(actor.get("evidence") if isinstance(actor, dict) else "").strip() or ev
        ealiases = idx.aliases(ename)
        anorm = norm_text(aev)
        if aev and names_any(anorm, ealiases) and any(names_any(anorm, idx.aliases(p)) for p in steps):
            continue
        tie = ""
        for raw, snorm in paper.sentences() if paper.norm else []:
            if names_any(snorm, ealiases) and any(names_any(snorm, idx.aliases(p)) for p in steps):
                tie = raw
                break
        if tie:
            report["tied_by_paper_sentence"].append({"row": i, "reaction": name, "enzyme": ename, "sentence": tie[:300]})
            continue
        out.append({**base, "kind": "enzyme", "element": ename, "enzyme_index": e, "evidence": aev,
                    "check": "enzyme_not_tied_to_step",
                    "detail": f"{ename!r}: its evidence {aev[:200]!r} does not name it with one of {steps}, "
                              f"and no paper sentence does"})
    return out


def _check_compartment(kind: str, i: int, row: Dict[str, Any], paper: _Paper) -> List[Dict[str, Any]]:
    src = row.get("compartment_source")
    if src not in ESTIMATED_SOURCES:
        return []
    entries = [e for e in row.get("compartment_estimate") or [] if isinstance(e, dict)]
    base = {"kind": kind, "row": i, "name": row.get("name"), "element": f"compartment ({src})",
            "check": "estimated_compartment_without_evidence", "decision": RULING}
    if not entries or not all(str(e.get("reason") or "").strip() for e in entries):
        return [{**base, "evidence": "", "detail": f"compartment_source {src!r} with no compartment_estimate reason"}]
    out = []
    for e in entries:
        if e.get("source") == "llm_estimate" or (src == "llm_estimate" and "quote" in e):
            quote = str(e.get("quote") or "")
            if not quote.strip() or (paper.norm and not paper.has(quote)):
                out.append({**base, "evidence": quote, "detail": f"LLM estimate quote not in the paper text: {quote[:160]!r}"})
        if e.get("source") == "neighbour_estimate" and not e.get("neighbours"):
            out.append({**base, "evidence": str(e.get("reason")), "detail": "neighbour estimate names no neighbour"})
    return out


def _apply_judge(payload: Dict[str, Any], judge: Callable[..., Any], paper: _Paper, flagged: List[tuple],
                 report: Dict[str, Any]) -> None:
    """Remove only what the paper contradicts: the judge's quote must be in the paper text."""
    from t2pw.pipeline.strict_quarantine import WITHHELD_REACTION_ACTORS_FIELD

    for row, record in flagged:
        try:
            verdict = judge(record, row, paper.raw) or {}
        except Exception as exc:  # noqa: BLE001 - a judge that fails removes nothing
            report["judge_errors"].append(f"{type(exc).__name__}: {exc}"[:200])
            continue
        quote = str(verdict.get("quote") or "")
        if not verdict.get("contradicted"):
            continue
        if not quote.strip() or not paper.has(quote):
            report["judge_rejected"].append({"element": record["element"], "check": record["check"], "quote": quote[:200]})
            continue
        entry = {"element": record["element"], "kind": record["kind"], "check": record["check"], "quote": quote,
                 "decision": RULING}
        scope = row.get("scope_membership")
        if record["kind"] == "reaction":
            if scope == "out_of_scope" or (isinstance(scope, dict) and scope.get("membership") == "out_of_scope"):
                continue
            row["scope_membership"] = {"membership": "out_of_scope", "reason": "contradicted_by_paper", "quote": quote,
                                       "decision": RULING, "was": scope}
            report["removed"].append({**entry, "reaction": row.get("name")})
        elif record["kind"] == "enzyme":
            enzymes = row.get("enzymes") or []
            pos = next((k for k, a in enumerate(enzymes) if _actor(a) == record["element"]), None)
            if pos is None:
                continue
            actor = enzymes.pop(pos)
            payload.setdefault(WITHHELD_REACTION_ACTORS_FIELD, []).append({
                "reaction": row.get("name"), "slots": ["enzymes"], "actor": actor, "name": record["element"],
                "reason": "contradicted_by_paper", "quote": quote, "gate": "final_payload_evidence", "decision": RULING})
            report["removed"].append({**entry, "reaction": row.get("name")})


def strip_placeholder_species(payload: Dict[str, Any]) -> List[str]:
    """D-W24 (7c): no species on PathBank's Unknown protein. Returns the species names taken off."""
    from t2pw.pipeline.entity_identity import is_pathbank_unknown_protein

    ents = payload.get("entities") if isinstance(payload.get("entities"), dict) else {}
    taken: List[str] = []
    for row in ents.get("proteins") or []:
        if not is_pathbank_unknown_protein(row):
            continue
        for key in ("species", "organism", "species_name", "species_id", "pathbank_species_id", "taxonomy_id", "species_ref"):
            value = row.pop(key, None)
            if key in ("species", "organism") and isinstance(value, str) and value not in taken:
                taken.append(value)
        meta = row.get("mapping_meta") if isinstance(row.get("mapping_meta"), dict) else {}
        for key in ("species", "species_id", "species_resolution"):
            meta.pop(key, None)
        if taken or "species_not_asserted" not in row:
            row["species_not_asserted"] = (
                "D-W24 7c, L8: PathBank's Unknown record stands for an unidentified enzyme; no species is asserted for it")
    return taken


def check_evidence(payload: Dict[str, Any], paper_text: Optional[str] = None, context: Optional[Dict[str, Any]] = None,
                   *, judge: Optional[Callable[..., Any]] = None) -> Dict[str, Any]:
    """Run every check; flag failing elements in place; return the report (also stored under ``FIELD``). Never raises."""
    paper_text = paper_text if isinstance(paper_text, str) else ""
    prior = payload.get(FIELD) if isinstance(payload.get(FIELD), dict) else None
    if prior and prior.get("paper_text") and not paper_text:
        return {"skipped": "already checked with the paper text"}
    report: Dict[str, Any] = {"decision": RULING, "paper_text": bool(paper_text), "counts": {}, "flagged": [],
                              "tied_by_paper_sentence": [], "removed": [], "judge": bool(judge), "judge_errors": [],
                              "judge_rejected": [], "placeholder_species_removed": []}
    snapshot = copy.deepcopy(payload)
    try:
        report["placeholder_species_removed"] = strip_placeholder_species(payload)
        procs = payload.get("processes") if isinstance(payload.get("processes"), dict) else {}
        rx = procs.get("reactions") if isinstance(procs.get("reactions"), list) else []
        tr = procs.get("transports") if isinstance(procs.get("transports"), list) else []
        idx, paper = _Index(payload, paper_text), _Paper(paper_text)
        ctx = {k: v for k, v in (context or {}).items() if isinstance(v, str) and v.strip()}
        flagged: List[tuple] = []
        for kind, rows in (("reaction", rx), ("transport", tr)):
            for i, row in enumerate(rows):
                if isinstance(row, dict):
                    _clear(row)
        for i, row in enumerate(rx):
            if not isinstance(row, dict):
                continue
            for record in _check_reaction(i, row, rx, idx, paper, ctx, report) + _check_compartment("reaction", i, row, paper):
                _flag(row, record, report)
                flagged.append((row, record))
        for i, row in enumerate(tr):
            if isinstance(row, dict):
                for record in _check_compartment("transport", i, row, paper):
                    _flag(row, record, report)
                    flagged.append((row, record))
        if judge is not None and paper.norm:
            _apply_judge(payload, judge, paper, flagged, report)
        report["flagged_elements"] = len(report["flagged"])
        if report["paper_text"] or report["flagged"] or report["removed"] or report["placeholder_species_removed"]:
            # nothing checked against a paper and nothing found: the payload is not touched
            payload[FIELD] = {k: report[k] for k in ("decision", "paper_text", "counts", "flagged_elements", "removed",
                                                     "placeholder_species_removed")}
    except Exception as exc:  # noqa: BLE001 - fails open: the payload is exactly what it was
        payload.clear()
        payload.update(snapshot)
        report["error"] = f"{type(exc).__name__}: {exc}"[:300]
    return report
