"""T2PW_REACTION_VERIFIER / T2PW_INTERACTION_TYPING (default off): check Stage-1 rows against the
paper, after the Stage-1 boundary, and repair or flag them. Nothing is deleted silently.

Reactions (one model call per 30 rows; at most :data:`MAX_CALLS` calls and :data:`MAX_SECONDS` per run).
Each row gets a verdict with a quote from the paper: ``ok`` · ``incomplete`` · ``vague`` · ``split`` ·
``summary`` · ``off_pathway`` · ``reversed`` · ``not_stated``. What happens next is deterministic:

* a name the model adds or renames to is used only when it occurs in the paper text;
* ``reversed`` swaps the sides only when the model's quote is in the paper;
* ``off_pathway`` only flags (it never drops a row);
* ``not_stated`` only flags;
* ``split``: a chain through generic intermediates is joined into the whole step from its own rows
  (linked only through an intermediate exactly one row makes and one row uses); a model's whole step is
  used only for a row through a generic intermediate and must keep every real name of that row;
* ``summary`` (policy ``merge``): omitted only when the kept rows make every non-currency product from
  its non-currency substrates (:func:`summary_covered`);
* a row the verifier changed that became identical to another row is folded into it (enzymes kept).

An omitted row is marked ``scope_membership: out_of_scope``, loses its lock id and its entry in the
lock manifest; every omission, repair and flag is in ``reaction_verifier_report.json`` (kept with the
run) with the row, the reason and the quote. Any error fails open (payload unchanged, error recorded).

Interactions (T2PW_INTERACTION_TYPING): only an unsigned row (:func:`retypable`) may become activation
or inhibition, only with an unqualified paper quote that states an effect (see :func:`apply_types`).
"""

from __future__ import annotations

import copy
import json
import os
import re
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

ENV_FLAG = "T2PW_REACTION_VERIFIER"
IX_FLAG = "T2PW_INTERACTION_TYPING"
GRANULARITY_ENV = "T2PW_VERIFIER_GRANULARITY"
#: split / summary handling. omit: both omitted and reported. merge: a split chain is replaced by
#: the paper's whole step (kept when that step cannot be validated); a summary is omitted only when
#: the rows it summarises are all present, otherwise kept. keep: both kept (reported only).
#: combo: split merged (omitted when the whole step cannot be validated); summary omitted.
GRANULARITY = ("omit", "merge", "keep", "combo")
DEFAULT_GRANULARITY = "merge"
MODEL_ENV_VAR = "OPENROUTER_EXTRACTION_MODEL"  # same model as Stage 1
REPORT_FILE = "reaction_verifier_report.json"
REPORT_STATE_KEY = "reaction_verifier_report"
MAX_PAPER_CHARS = 150_000
ROWS_PER_CALL = 30
#: Per-run budget (R1 F9): at most this many model calls and this much wall time for the whole verifier;
#: rows past the budget get no verdict (= "ok", untouched) and the report says so.
MAX_CALLS = 6
MAX_SECONDS = 600.0
VERDICTS = ("ok", "incomplete", "vague", "split", "summary", "off_pathway", "reversed", "not_stated")
SIGNED = ("activation", "inhibition")

_SYSTEM = """You verify reaction rows extracted from a scientific paper. You get the requested pathway
and organism, the paper text, and numbered rows (inputs -> outputs, enzymes, the evidence the
extractor quoted). Judge each row against the paper only, and give one verdict:
- "ok": the paper states this conversion, with these participants, as a step of the requested pathway.
- "incomplete": the paper states this step but names a substrate or co-product of THIS step that the
  row lacks (for example pyruvate, CO2, AMP, PPi, ATP). Put only the missing names in "add_inputs" /
  "add_outputs", spelled as the paper spells them. Never add enzymes, catalytic cofactors (ThDP, FAD,
  PLP, metal ions), water or protons. If a participant in the row is WRONG rather than missing, give
  the correction in "rename" instead (see "vague").
- "vague": a participant is a generic phrase ("activated X", "the product", "precursor") and the paper
  names that compound specifically. Put [{"old": <name in the row>, "new": <the paper's name>}] in
  "rename". If the paper never names it specifically, do not use "vague".
- "split": the row is a partial step of ONE enzyme's single catalytic conversion that runs through
  unnamed or numbered intermediates ("intermediate I", "intermediate II", an enzyme-bound adduct),
  EVEN WHEN the paper describes those intermediates in its mechanism. Put the paper's whole step in
  "merged_step": {"inputs": [...], "outputs": [...], "quote": "..."} with names from the paper.
- "summary": an overall conversion the paper describes as two or more steps catalysed by DIFFERENT
  enzymes (for example "chorismate is converted to DHNA by MenF, MenD, MenH, MenC, MenE and MenB").
  A step one enzyme catalyses, even over several cycles or with an adenylated substrate, is not a summary.
- "off_pathway": the paper states it, but it is not a step of the requested pathway: iron or nutrient
  uptake, binding of the final product to a metal, an assay or reporter reaction, fermentation, central
  metabolism, or a reaction of another pathway that only supplies a precursor.
- "reversed": the paper states this step in the opposite direction.
- "not_stated": the paper does not state this conversion anywhere.
Also give "quote": 6 to 40 CONSECUTIVE words copied character for character from the paper text (you
may leave out a citation in parentheses; no ellipsis, no rewording, no shortening) that state the step,
or "" if there is none. Start from the row's evidence sentence when it states the step.
Never invent. Every name you give must appear in the paper text. When unsure, answer "ok".
Reply with JSON only:
{"rows": [{"id": 0, "verdict": "ok", "quote": "...", "add_inputs": [], "add_outputs": [], "rename": [],
  "merged_step": null, "reason": "at most 20 words"}]}"""

_IX_SYSTEM = """You check the type of regulatory interaction rows extracted from a scientific paper.
Each row says entity_1 acts on entity_2 with a relationship. Decide the type from the AFTER-EFFECTS the
paper reports for entity_2 or for what it controls: if entity_1 (or its presence, overexpression or
binding) lowers entity_2's expression, level, stability or activity, or blocks the pathway or lowers its
product, or if losing entity_1 raises them, the type is "inhibition". If entity_1 raises them, or losing
entity_1 lowers them, the type is "activation". Use "binding" only when the paper states a physical
interaction and no effect on entity_2. Use "unknown" when the paper states no effect and no binding.
Give "quote": 6 to 40 CONSECUTIVE words copied character for character from the paper text (you may
leave out a citation in parentheses; no ellipsis, no rewording, no shortening) that state the effect,
or "" if there is none. Start from the row's evidence sentence when it states the effect.
Reply with JSON only: {"rows": [{"id": 0, "type": "activation", "quote": "...", "reason": "at most 20 words"}]}"""


def enabled() -> bool:
    return os.environ.get(ENV_FLAG, "").strip().casefold() in ("1", "true", "yes")


def typing_enabled() -> bool:
    return os.environ.get(IX_FLAG, "").strip().casefold() in ("1", "true", "yes")


def granularity() -> str:
    g = os.environ.get(GRANULARITY_ENV, "").strip().casefold()
    return g if g in GRANULARITY else DEFAULT_GRANULARITY


def _words(value: Any) -> str:
    from t2pw.pipeline.localized_repair import _normalize_for_match

    return _normalize_for_match(value)


def _key(value: Any) -> str:
    from t2pw.pipeline.process_normalizer import _normalize

    return _normalize(str(value or ""))


def _in_text(phrase: Any, text_words: str, min_words: int = 1) -> bool:
    q = _words(phrase)
    return len(q.split()) >= min_words and f" {q} " in text_words


def text_words(text: str) -> str:
    """The paper as match text, twice: as written, and with short parentheticals (citations such as
    "( 19 )" or "(Zhong et al., 2009)") taken out, since a model quoting a sentence drops them. The
    "|" between the two copies (never part of a word) keeps a quote from matching across them."""
    bare = re.sub(r"\([^()]{0,80}\)|\[[^\[\]]{0,40}\]", " ", str(text or ""))
    return f" {_words(text)} | {_words(bare)} "


#: Words a model may not drop from the middle of a quote: dropping one can reverse what the paper says.
_KEEP_WORDS = frozenset("not no nor never without loss lack lacking absence absent mutant mutants mutation "
                        "mutations deletion deleted knockout knockdown null fail fails failed unable neither".split())


def _gapped(q: List[str], doc: List[str]) -> bool:
    """``q`` is the paper's own words in order, a few left out: the first three words are consecutive in
    the paper, the rest follow in order within 1.5x the quote's length, and no left-out word is in
    :data:`_KEEP_WORDS`. (A model copying a sentence drops a clause such as "encoding an ALAD".)"""
    return len(q) >= 6 and any(_gapped_at(q, doc, start) for start in range(len(doc) - 2))


def _gapped_at(q: List[str], doc: List[str], start: int) -> bool:
    if len(q) < 6 or doc[start:start + 3] != q[:3]:
        return False
    span = int(len(q) * 1.5) + 2
    i, j = 3, start + 3
    while i < len(q) and j < min(len(doc), start + span):
        if doc[j] == q[i]:
            i += 1
        elif doc[j] in _KEEP_WORDS:
            break
        j += 1
    return i == len(q)


def quote_in_paper(evidence: Any, text_words: str) -> bool:
    """Every piece of the quote (split at ellipses) of 3+ words is in the paper, and one has 4+. A piece
    of 6+ words may also leave out a few of the paper's words (see :func:`_gapped`)."""
    parts = [p for p in re.split(r"\.\.\.|…|\[\s*\]", str(evidence or "")) if len(_words(p).split()) >= 3]
    docs = [d.split() for d in text_words.split(" | ")]

    def found(p: str) -> bool:
        return _in_text(p, text_words) or any(_gapped(_words(p).split(), d) for d in docs)
    return bool(parts) and all(found(p) for p in parts) and any(len(_words(p).split()) >= 4 for p in parts)


#: Tokens that end with a period without ending a sentence ("E. coli", "et al.", "Fig. 2", "e.g.").
_ABBREV = frozenset("fig figs e.g i.e al ref refs no nos vs ca approx eq eqs sp spp st dr mr ms cf resp suppl "
                    "vol pp ed eds etc var subsp str".split())


def split_sentences(text: str) -> List[str]:
    """Sentences, cut only at . ! ? ; followed by space and a capital letter or a digit-free opener, and
    never after a known abbreviation or a single capital letter (a genus initial: "E. coli"). A cut at
    "E. coli" would hide a negation before it (review R3 P2)."""
    out, start = [], 0
    for m in re.finditer(r"[.!?;]\s+(?=[A-Z(\[])", str(text or "")):
        before = re.findall(r"([A-Za-z.]+)\.?$", text[start:m.start() + 1].rstrip("."))
        tok = (before[-1] if before else "").strip(".").casefold()
        if text[m.start()] == "." and (tok in _ABBREV or re.fullmatch(r"[a-z]", tok)):
            continue
        out.append(text[start:m.start() + 1])
        start = m.end()
    out.append(text[start:])
    return [x for x in out if x.strip()]


def _sentences(text: str) -> List[List[str]]:
    """The paper's sentences as word lists, as written and with short parentheticals taken out."""
    bare = re.sub(r"\([^()]{0,80}\)|\[[^\[\]]{0,40}\]", " ", str(text or ""))
    return [_words(s).split() for t in (str(text or ""), bare) for s in split_sentences(t)]


def _starts(q: List[str], doc: List[str]) -> List[int]:
    """Where ``q`` begins in ``doc``: contiguous, or gapped as in :func:`_gapped`."""
    out = [k for k in range(len(doc) - len(q) + 1) if doc[k:k + len(q)] == q]
    return out or [k for k in range(len(doc) - 2) if _gapped_at(q, doc, k)]


def quote_unqualified(quote: Any, text: str) -> bool:
    """The quote is in the paper AND, in some sentence where it occurs, no negation or loss word
    (:data:`_KEEP_WORDS`) stands before it that the quote left out. "PIF1 caused increased expression of
    CHLH" is refused when the paper says "Loss of function of PIF1 caused increased expression of CHLH"."""
    q = _words(quote).split()
    if len(q) < 4:
        return False
    for sent in _sentences(text):
        for k in _starts(q, sent):
            if not set(sent[:k]) & _KEEP_WORDS:  # checked whatever the quote itself contains (R3 P2b)
                return True
    return False


#: A quote that re-types an interaction must state an effect, not only a contact: "EntC and EntB interact
#: to form a dynamic channeling system" states none. Whole word forms, not stems: "promoter", "reductase",
#: "reduction", "inducible", "raised" and "accumulation" are not effects (review R3 P1).
_EFFECT_WORDS = re.compile(
    r"^(?:inhibit(?:s|ed|ing|ion)?|repress(?:es|ed|ing|ion)?|activat(?:e|es|ed|ing|ion)|induc(?:e|es|ed|ing|tion)|"
    r"stimulat(?:e|es|ed|ing|ion)|suppress(?:es|ed|ing|ion)?|block(?:s|ed|ing)?|promot(?:e|es|ed|ing)|"
    r"enhanc(?:es|ed|ing|ement)?|increas(?:es|ed|ing|e)|decreas(?:es|ed|ing|e)|elevat(?:e|es|ed|ing)|"
    r"reduc(?:e|es|ed|ing)|lower(?:s|ed|ing)|degrad(?:e|es|ed|ing|ation)|stabiliz(?:e|es|ed|ing|ation)|"
    r"destabiliz(?:e|es|ed|ing|ation)|abolish(?:es|ed|ing)?|up-?regulat(?:e|es|ed|ing|ion)|"
    r"down-?regulat(?:e|es|ed|ing|ion)|silenc(?:e|es|ed|ing)|prevent(?:s|ed|ing)?|attenuat(?:e|es|ed|ing|ion)|"
    r"impair(?:s|ed|ing)?|restor(?:e|es|ed|ing)|abrogat(?:e|es|ed|ing)|antagoniz(?:e|es|ed|ing)|"
    r"diminish(?:es|ed|ing)?|boost(?:s|ed|ing)?|"
    # agent nouns state the sign too ("a competitive inhibitor of EntB"; review R4 N1)
    r"inhibitor(?:s|y)?|inhibitory|activator(?:s)?|repressor(?:s)?)$")


def states_effect(quote: Any) -> bool:
    raw = str(quote or "").casefold()
    words = re.findall(r"[a-z]+(?:-[a-z]+)?", raw)
    return any(_EFFECT_WORDS.match(w) for w in words) or "up regulat" in raw or "down regulat" in raw


#: Words that carry a sign whatever table they are (not) in: "allosteric_inhibitor_of", "inhibited_by",
#: "negatively_regulates", "activated_by". A row whose word has one of these is signed and never re-typed.
_SIGN_STEMS = ("inhibit", "repress", "activat", "induc", "stimulat", "suppress", "block", "promot", "enhanc",
               "upregul", "downregul", "up regul", "down regul", "antagon", "agonist", "prevent", "attenuat",
               "negative", "positive", "silenc")


def retypable(word: Any) -> bool:
    """Only an empty word or an unsigned Binding / Unspecified / Regulation word may be given a sign."""
    from t2pw.pwml.interaction_types import (PWML_INTERACTION_TYPES, PWML_UNSIGNED_INTERACTION_TYPES,
                                             _interaction_type_key)

    key = _interaction_type_key(word)
    if any(s in key.replace("_", " ").replace("-", " ") for s in _SIGN_STEMS) or key in PWML_INTERACTION_TYPES:
        return False
    return key in ("", "interaction") or PWML_UNSIGNED_INTERACTION_TYPES.get(key) in ("Binding", "Unspecified", "Regulation")


def _currency(name: str) -> bool:
    from t2pw.rag.synthesize import COFACTOR_NAMES

    k = re.sub(r"\s+", " ", str(name or "").strip().casefold())
    return k in COFACTOR_NAMES or k in _MORE_CURRENCY


_MORE_CURRENCY = frozenset({
    "nad", "nadp", "nad(p)h", "nad(p)+", "carbon dioxide", "diphosphate", "pyrophosphate", "phosphate",
    "orthophosphate", "inorganic phosphate", "hydrogen ion", "hydron", "h2o2", "hydrogen peroxide", "ammonia",
    "ammonium", "adenosine triphosphate", "adenosine diphosphate", "adenosine monophosphate", "coa",
    "coenzyme a", "nadh2", "fmn", "fmnh2", "thdp", "thiamine diphosphate", "s-adenosylmethionine",
    "s-adenosyl-l-methionine", "s-adenosylhomocysteine", "s-adenosyl-l-homocysteine", "magnesium", "mg2+",
    "fe2+", "fe3+", "mn2+", "zn2+", "electron", "e-",
})


def _core(side: Any) -> set:
    """The side's non-currency names (keys): what a pathway step is about."""
    return {_key(n) for n in _names(side) if not _currency(n)}


def _name(item: Any) -> str:
    if isinstance(item, dict):
        for k in ("name", "entity", "compound", "protein"):
            if isinstance(item.get(k), str):
                return item[k]
        return ""
    return str(item or "")


def _names(side: Any) -> List[str]:
    return [n for n in (_name(x) for x in side or []) if n.strip()]


def _side_key(side: Any) -> Tuple[str, ...]:
    return tuple(sorted({_key(n) for n in _names(side)}))


def _rows(payload: Dict[str, Any], kind: str) -> List[Any]:
    procs = payload.get("processes")
    rows = procs.get(kind) if isinstance(procs, dict) else None
    return rows if isinstance(rows, list) else []


def _enzymes(row: Dict[str, Any]) -> List[str]:
    out = []
    for e in row.get("enzymes") or []:
        n = e.get("protein") or e.get("protein_complex") or e.get("name") if isinstance(e, dict) else e
        if n:
            out.append(str(n))
    return out


def _paper(text: str) -> str:
    return text if len(text) <= MAX_PAPER_CHARS else text[:MAX_PAPER_CHARS]


class Budget:
    """Calls and seconds left for one run of the verifier."""

    def __init__(self, calls: Optional[int] = None, seconds: Optional[float] = None) -> None:
        import time

        self.calls = MAX_CALLS if calls is None else calls
        self.deadline = time.monotonic() + (MAX_SECONDS if seconds is None else seconds)
        self.skipped = 0

    def take(self) -> Optional[Tuple[float, int]]:
        """``(per-attempt timeout, attempts)`` for the next call such that the client's own worst case
        (:func:`t2pw.llm.client.worst_case_call_seconds`: every attempt, backoff and SDK retry) fits in
        the time left, or None when the budget is spent. This makes :data:`MAX_SECONDS` a real
        wall-clock bound across retries (review R3 P3), not a check between calls."""
        import time

        from t2pw.llm.client import worst_case_call_seconds

        left = self.deadline - time.monotonic()
        if self.calls > 0:
            for attempts in (2, 1):
                for t in (300.0, 240.0, 180.0, 120.0, 90.0, 60.0):
                    if worst_case_call_seconds(timeout=t, max_retries=attempts) <= left:
                        self.calls -= 1
                        return t, attempts
        self.skipped += 1
        return None


def _ask(system: str, user: str, chat_fn: Optional[Callable[..., Any]], stage: str,
         limit: Optional[Tuple[float, int]] = None) -> Dict[str, Any]:
    if chat_fn is None:
        from t2pw.llm.client import chat_detailed as chat_fn  # noqa: PLW2901
    extra = {"timeout": limit[0], "max_retries": limit[1]} if limit else {}
    result = chat_fn([{"role": "system", "content": system}, {"role": "user", "content": user}],
                     temperature=0.0, max_tokens=32000, response_json=True,
                     model_env_var=MODEL_ENV_VAR, stage_name=stage, **extra)
    raw = str(getattr(result, "text", result)).strip()
    raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw)
    parsed = json.loads(raw)
    if not isinstance(parsed, dict):
        raise ValueError("reply is not a JSON object")
    return parsed


def _context(pathway: Dict[str, Any]) -> Dict[str, str]:
    return {"requested_pathway": str(pathway.get("pathway_name") or pathway.get("name") or ""),
            "organism": str(pathway.get("likely_organism") or pathway.get("organism") or ""),
            "task": str(pathway.get("task") or "")}


def ask_reactions(payload: Dict[str, Any], text: str, pathway: Dict[str, Any],
                  chat_fn: Optional[Callable[..., Any]] = None, budget: Optional[Budget] = None) -> Dict[int, Dict[str, Any]]:
    """Model verdicts per reaction index (one call per :data:`ROWS_PER_CALL` rows)."""
    rows = [(i, r) for i, r in enumerate(_rows(payload, "reactions")) if isinstance(r, dict)]
    out: Dict[int, Dict[str, Any]] = {}
    for start in range(0, len(rows), ROWS_PER_CALL):
        chunk = rows[start:start + ROWS_PER_CALL]
        items = [{"id": i, "name": r.get("name"), "inputs": _names(r.get("inputs")), "outputs": _names(r.get("outputs")),
                  "enzymes": _enzymes(r), "evidence": str(r.get("evidence") or "")[:600]} for i, r in chunk]
        user = json.dumps({**_context(pathway), "rows": items, "paper_text": _paper(text)}, ensure_ascii=False)
        left = budget.take() if budget is not None else None
        if budget is not None and left is None:
            break
        reply = _ask(_SYSTEM, user, chat_fn, "Stage 1 reaction verifier", left)
        for a in reply.get("rows") or []:
            if isinstance(a, dict) and isinstance(a.get("id"), int):
                out[a["id"]] = a
    return out


def summary_covered(row: Dict[str, Any], kept: Sequence[Tuple[int, Dict[str, Any]]]) -> bool:
    """True only when the kept rows themselves make EVERY non-currency product of ``row`` from its
    non-currency substrates: each product is an output of some kept row and is reachable from EACH of the
    row's substrates through kept rows, walking non-currency names only (a co-substrate alone, such as
    L-serine, cannot cover it: Gemma run 3). A co-product (CO2, AMP, PPi) can
    neither stand in for a product nor carry the walk, so dropping a covered summary loses nothing the
    kept rows do not already state (review R1 F3). A real one-enzyme step (EntF: DHB-AMP + L-serine ->
    enterobactin) is dropped only if another kept row already makes enterobactin from DHB-AMP."""
    src, dst = _core(row.get("inputs")), _core(row.get("outputs"))
    if not src or not dst:
        return False
    edges: Dict[str, set] = {}
    for _, r in kept:
        for a in _core(r.get("inputs")):
            edges.setdefault(a, set()).update(_core(r.get("outputs")))
    for s in src:
        seen, todo = set(), [s]
        while todo:
            for nxt in edges.get(todo.pop(), ()):
                if nxt not in seen:
                    seen.add(nxt)
                    todo.append(nxt)
        if not dst <= seen:
            return False
    return True


def _lineage(row: Dict[str, Any], reason: str) -> None:
    from t2pw.pipeline import lineage

    lineage.record(row, {"stage": "audit_repair", "origin": "audit_modified", "support": "direct",
                         "paper_explicit": "explicit", "reason": "reaction verifier: " + reason,
                         "review_required": False, "uncertainty": "",
                         "sources": [{"source_id": "seed_paper", "source_type": "paper"}]})


def _add_compound(payload: Dict[str, Any], name: str, reason: str) -> None:
    ents = payload.setdefault("entities", {})
    known = {_key(_name(e)) for bucket in ents.values() if isinstance(bucket, list) for e in bucket}
    if _key(name) in known:
        return
    row = {"name": name, "class": "compound", "confidence": 1.0, "provenance": "extracted"}
    _lineage(row, reason)
    ents.setdefault("compounds", []).append(row)


_GENERIC_INTERMEDIATE = re.compile(
    r"^(?:the\s+|an?\s+)?(?:(?:first|second|third|enzyme[\s\-]bound|covalent|tetrahedral)\s+)?"
    r"(?:intermediate|adduct)(?:\s+(?:[ivx]+|\d+|[a-z]))?$", re.IGNORECASE)


def _generic(name: str) -> bool:
    return bool(_GENERIC_INTERMEDIATE.match(name.strip()))


def _chain_steps(split_rows: Sequence[Tuple[int, Dict[str, Any]]]) -> Dict[int, Dict[str, Any]]:
    """Rows linked through generic intermediates, joined into one whole step: every input and output of
    the chain except the generic names and what one row makes and another uses. Maps each member row
    index to the (shared) joined row. Only names already in the rows are used."""
    # Two rows are linked only through a generic name that exactly ONE row makes and exactly ONE row uses,
    # and only when they share a catalyst (or one names none). Two unrelated mechanisms that both say
    # "intermediate I" (two makers or two users) are never joined (review R1 F7b).
    def gen(side: str, r: Dict[str, Any]) -> set:
        return {_key(n) for n in _names(r.get(side)) if _generic(n)}

    makers: Dict[str, List[int]] = {}
    users: Dict[str, List[int]] = {}
    for i, r in split_rows:
        for g in gen("outputs", r):
            makers.setdefault(g, []).append(i)
        for g in gen("inputs", r):
            users.setdefault(g, []).append(i)
    by_index = dict(split_rows)
    parent = {i: i for i, _ in split_rows}

    def root(i: int) -> int:
        while parent[i] != i:
            i = parent[i]
        return i
    for g, mk in makers.items():
        us = users.get(g, [])
        if len(mk) != 1 or len(us) != 1 or mk[0] == us[0]:
            continue
        ea, eb = ({_key(e) for e in _enzymes(by_index[x])} for x in (mk[0], us[0]))
        if ea and eb and not ea & eb:
            continue
        parent[root(us[0])] = root(mk[0])
    comp: Dict[int, List[Tuple[int, Dict[str, Any]]]] = {}
    for i, r in split_rows:
        comp.setdefault(root(i), []).append((i, r))
    groups = list(comp.values())
    out: Dict[int, Dict[str, Any]] = {}
    for g in groups:
        made = {_key(n) for _, r in g for n in _names(r.get("outputs"))}
        used = {_key(n) for _, r in g for n in _names(r.get("inputs"))}
        ins, outs = [], []
        for _, r in sorted(g, key=lambda x: x[0]):
            for side, acc, other in (("inputs", ins, made), ("outputs", outs, used)):
                for n in _names(r.get(side)):
                    if not _generic(n) and _key(n) not in other and _key(n) not in {_key(x) for x in acc}:
                        acc.append(n)
        made_g = set().union(*(gen("outputs", r) for _, r in g))
        used_g = set().union(*(gen("inputs", r) for _, r in g))
        if len(g) < 2 or not ins or not outs or made_g != used_g:  # every intermediate made AND used inside
            continue
        first = sorted(g, key=lambda x: x[0])[0][1]
        enz: List[Any] = []
        for _, r in g:
            for e in r.get("enzymes") or []:
                if e not in enz:
                    enz.append(copy.deepcopy(e))
        row = {"name": first.get("name") or "joined step", "inputs": ins, "outputs": outs, "enzymes": enz,
               "biological_state": first.get("biological_state"),
               "evidence": " ... ".join(str(r.get("evidence") or "") for _, r in sorted(g, key=lambda x: x[0]) if r.get("evidence")),
               "scope_membership": "core"}
        for i, _ in g:
            out[i] = row
    return out


def _spelling(name: Any) -> str:
    """One key for spellings of one compound: case, punctuation, and acid / -ate (-ous acid / -ite) folded."""
    k = re.sub(r"[^a-z0-9]+", "", str(name or "").casefold())
    return re.sub(r"ousacid$", "ite", re.sub(r"icacid$", "ate", k))


def _payload_names(payload: Dict[str, Any], skip: Optional[Dict[str, Any]] = None, registry: bool = True) -> Dict[str, str]:
    """spelling key -> the spelling the payload already uses (every participant of every other row, and
    every registry entity)."""
    out: Dict[str, str] = {}
    for r in _rows(payload, "reactions"):
        if isinstance(r, dict) and r is not skip:
            for n in _names(r.get("inputs")) + _names(r.get("outputs")):
                out.setdefault(_spelling(n), n)
    for bucket in (payload.get("entities") or {}).values() if registry else ():
        for e in bucket if isinstance(bucket, list) else []:
            if _name(e):
                out.setdefault(_spelling(_name(e)), _name(e))
    return out


def _abbreviated(name: Any) -> bool:
    """The name contains an abbreviation token (2+ capitals, at most 8 characters: DHB, DHB-AMP, CoA, SAM).
    A synonym of a full name the payload already uses cannot be recognised from spelling alone (review R4
    B1: "DHB-AMP" next to "2,3-dihydroxybenzoyl-AMP"), so such a name is only ever taken in the payload's
    own exact spelling."""
    return any(len(t) <= 8 and sum(c.isupper() for c in t) >= 2 for t in re.findall(r"[A-Za-z0-9]+", str(name or "")))


def _one_spelling(name: str, known: Dict[str, str]) -> str:
    """The payload's own spelling when it already names this compound, else ``name`` (review R3 P4: never
    a second spelling of one compound, the AMBIGUOUS_RENAME_TARGET shape that ends a run)."""
    return known.get(_spelling(name), name)


def _apply_renames(row: Dict[str, Any], renames: Any, tw: str, payload: Optional[Dict[str, Any]] = None) -> List[List[str]]:
    """Rename participants of ``row`` in place, only to names that occur in the paper. A rename is applied
    only when the new name is one the payload already uses (its spelling is kept) or the old name occurs in
    no other row (so no row keeps the old spelling next to the new one); otherwise it is refused (R3 P4).
    -> [[old, new]]."""
    known = _payload_names(payload or {}, skip=row)
    others = _payload_names(payload or {}, skip=row, registry=False)
    done = []
    for ren in renames if isinstance(renames, list) else []:
        old, new = (ren.get("old"), ren.get("new")) if isinstance(ren, dict) else (None, None)
        if not (isinstance(old, str) and isinstance(new, str) and len(new.strip()) > 1 and _in_text(new, tw)):
            continue
        if _spelling(new) in known:
            new = known[_spelling(new)]
        elif _spelling(old) in others or _abbreviated(new):  # R3 P4, R4 B1
            continue
        if _key(new) == _key(old):
            continue
        for side in ("inputs", "outputs"):
            for j, item in enumerate(row.get(side) or []):
                if _key(_name(item)) == _key(old):
                    if isinstance(item, dict):
                        item[next(k for k in ("name", "entity", "compound", "protein") if k in item)] = new.strip()
                    else:
                        row[side][j] = new.strip()
                    done.append([old, new.strip()])
    return done


def _currency_compatible(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
    """Same non-currency names already established; on each side one row's currency must contain the
    other's (one lists CO2 the other leaves out) -- never two different cofactors. The Stage-1 ensemble
    refuses the same merge."""
    for side in ("inputs", "outputs"):
        ca = {_key(n) for n in _names(a.get(side)) if _currency(n)}
        cb = {_key(n) for n in _names(b.get(side)) if _currency(n)}
        if not (ca <= cb or cb <= ca):
            return False
    return True


def _omit(row: Dict[str, Any]) -> None:
    row["scope_membership"] = "out_of_scope"
    row.pop("locked_reaction_id", None)
    row.pop("locked", None)


def apply_verdicts(payload: Dict[str, Any], text: str, answers: Dict[int, Dict[str, Any]],
                   policy: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Deterministic half: returns ``(new payload, report)``. ``payload`` is not changed."""
    tw = text_words(text)
    out = copy.deepcopy(payload)
    rows = _rows(out, "reactions")
    rep: Dict[str, Any] = {"policy": policy, "rows": [], "omitted": [], "repaired": [], "flagged": [], "merged": []}
    live = [(i, r) for i, r in enumerate(rows) if isinstance(r, dict)
            and str(r.get("scope_membership") or "").strip().casefold() != "out_of_scope"]
    verdict = {i: str((answers.get(i) or {}).get("verdict") or "ok") for i, _ in live}
    # A row through a generic intermediate ("intermediate I", "an adduct") is a split step whatever the
    # model said: the name is no compound, so the row cannot be one step of the pathway as drawn.
    for i, r in live:
        if verdict[i] != "off_pathway" and any(_generic(n) for n in _names(r.get("inputs")) + _names(r.get("outputs"))):
            verdict[i] = "split"
    chains = _chain_steps([(i, r) for i, r in live if verdict[i] == "split"])
    # a row the model judged wrong (not stated, off pathway) stays in the payload but can never be the
    # coverage that drops another row (Gemma run 3: DHB + serine -> enterobactin "covered" the real EntF step)
    plain = [(i, r) for i, r in live if verdict[i] not in ("summary", "split", "not_stated", "off_pathway")]
    new_rows: List[Dict[str, Any]] = []
    chain_done: set = set()

    for i, r in live:
        a = answers.get(i) or {}
        v = verdict[i] if verdict[i] in VERDICTS else "ok"
        mq = str(a.get("quote") or "")
        rec = {"index": i, "name": r.get("name"), "inputs": _names(r.get("inputs")), "outputs": _names(r.get("outputs")),
               "verdict": v, "reason": str(a.get("reason") or "")[:200], "row_quote": str(r.get("evidence") or "")[:600],
               "row_quote_in_paper": quote_in_paper(r.get("evidence"), tw), "model_quote": mq[:400],
               "model_quote_in_paper": quote_in_paper(mq, tw), "action": "kept",
               "lock_id": r.get("locked_reaction_id")}
        rep["rows"].append(rec)
        if v in ("incomplete", "vague"):
            # A participant that is wrong rather than missing comes as a rename (the model may give one with
            # either verdict, R1 F7c); a missing co-product comes as an addition. Every name must be the
            # paper's own.
            done = _apply_renames(r, a.get("rename"), tw, out)
            added = []
            if v == "incomplete":
                known = _payload_names(out)
                for side, field in (("inputs", "add_inputs"), ("outputs", "add_outputs")):
                    have = set(_side_key(r.get(side)))
                    names = a.get(field)
                    for n in [names] if isinstance(names, str) else (names or []):
                        if isinstance(n, str) and len(n.strip()) > 1 and _key(n) not in have and _in_text(n, tw):
                            if _spelling(n) not in known and _abbreviated(n):  # R4 B1
                                continue
                            n = _one_spelling(n.strip(), known)
                            if _key(n) in have:
                                continue
                            r.setdefault(side, []).append(n.strip())
                            have.add(_key(n))
                            added.append([side, n.strip()])
            for _, new in done:
                _add_compound(out, new, "specific name the paper gives")
            for _, n in added:
                _add_compound(out, n, "participant named by the paper for this step")
            if done or added:
                rec.update(action="repaired", renamed=done, added=added)
                _lineage(r, "; ".join(["renamed " + ", ".join(f"{o} -> {n}" for o, n in done)] * bool(done)
                                      + ["added " + ", ".join(n for _, n in added)] * bool(added)))
        elif v == "reversed":
            if quote_in_paper(mq, tw):
                r["inputs"], r["outputs"] = r.get("outputs") or [], r.get("inputs") or []
                rec["action"] = "repaired"
                _lineage(r, "direction swapped to the paper's")
        elif v == "off_pathway":
            # Flag only (review R3 P6): even an isolated row with a verified quote can be a gold step that
            # Stage 1 extracted disconnected, or a step linked only through the pathway's own cofactor.
            rec["action"] = "flagged"
        elif v == "not_stated":
            # Flag only. On the replay the model called five certified paper steps "not stated" (their
            # quotes were not verbatim); a negative judgement by the model never deletes a row.
            rec["action"] = "flagged"
        elif v in ("split", "summary"):
            act = "flagged"
            if policy == "omit" or (policy == "combo" and v == "summary"):
                act = "omitted"
            elif v == "summary" and policy == "merge":
                act = "omitted" if summary_covered(r, [x for x in plain if x[0] != i]) else "flagged"
            elif v == "split" and policy in ("merge", "combo"):
                # The chain's own rows first (deterministic, no new names). The model's whole step only for a
                # row through a generic intermediate, and only when it keeps every real (non-generic,
                # non-currency) name of that row: it must be the same step, not a replacement (R1 F7a).
                merged, ok = None, False
                if i in chains:
                    merged = None if id(chains[i]) in chain_done else copy.deepcopy(chains[i])
                    chain_done.add(id(chains[i]))
                    ok = True
                elif any(_generic(n) for n in rec["inputs"] + rec["outputs"]):
                    m = a.get("merged_step") if isinstance(a.get("merged_step"), dict) else {}
                    ins = [n for n in m.get("inputs") or [] if isinstance(n, str) and n.strip()]
                    outs = [n for n in m.get("outputs") or [] if isinstance(n, str) and n.strip()]
                    real = {_key(n) for n in rec["inputs"] + rec["outputs"] if not _generic(n) and not _currency(n)}
                    ok = bool(ins and outs and all(_in_text(n, tw) for n in ins + outs)
                              and quote_in_paper(m.get("quote"), tw) and real <= {_key(n) for n in ins + outs})
                    if ok:
                        merged = {"name": (r.get("name") or "merged step"), "inputs": [n.strip() for n in ins],
                                  "outputs": [n.strip() for n in outs], "enzymes": copy.deepcopy(r.get("enzymes") or []),
                                  "biological_state": r.get("biological_state"), "evidence": str(m.get("quote")).strip(),
                                  "scope_membership": "core"}
                        for n in merged["inputs"] + merged["outputs"]:
                            _add_compound(out, n, "participant of the paper's whole step")
                if ok:
                    if merged is not None:
                        _lineage(merged, "whole step replacing partial steps through intermediates")
                        new_rows.append(merged)
                        rec["merged_step"] = {"inputs": _names(merged["inputs"]), "outputs": _names(merged["outputs"]),
                                              "quote": merged["evidence"]}
                    act = "omitted"
                elif policy == "combo":
                    act = "omitted"
            if act == "omitted":
                _omit(r)
            rec["action"] = act

    # A row the verifier changed (or a whole step it appended) that is now identical to another row is
    # folded into it. Duplicates the verifier did not make are left to the stages that own them.
    changed = {rec["index"] for rec in rep["rows"] if rec["action"] == "repaired"}
    for m in new_rows:
        changed.add(len(rows))
        rows.append(m)
    seen: Dict[Tuple, int] = {}
    for i, r in enumerate(rows):
        if not isinstance(r, dict) or str(r.get("scope_membership") or "").strip().casefold() == "out_of_scope":
            continue
        # identical on non-currency names: a joined step without the CO2 the paper's own row lists is the
        # same step (and the paper's row, with its currency, is the one kept)
        ci, co = _core(r.get("inputs")), _core(r.get("outputs"))
        if not ci or not co:
            continue
        k = (tuple(sorted(ci)), tuple(sorted(co)))
        if k not in seen:
            seen[k] = i
            continue
        if i not in changed and seen[k] not in changed:
            continue
        keep = rows[seen[k]]
        if not _currency_compatible(keep, r):  # NAD vs NADP, ATP vs GTP: different rows (R3 P5)
            continue
        have = {_key(e) for e in _enzymes(keep)}
        for e in r.get("enzymes") or []:
            n = e.get("protein") or e.get("protein_complex") or e.get("name") if isinstance(e, dict) else e
            if n and _key(n) not in have:
                keep.setdefault("enzymes", []).append(copy.deepcopy(e))
        _omit(r)
        rep["merged"].append({"index": i, "into": seen[k], "name": r.get("name")})
        for rec in rep["rows"]:
            if rec["index"] == i:
                rec["action"] = "folded_into_%d" % seen[k]
    for rec in rep["rows"]:
        bucket = {"omitted": "omitted", "repaired": "repaired", "flagged": "flagged"}.get(rec["action"])
        if bucket:
            rep[bucket].append(rec["index"])
        if rec["action"] != "kept":  # the row itself says what the verifier did, with the quote
            rows[rec["index"]]["verifier"] = {k: rec.get(k) for k in ("verdict", "action", "reason", "model_quote")}
    rep["appended"] = len(new_rows)
    return out, rep


def ask_interactions(payload: Dict[str, Any], text: str, chat_fn: Optional[Callable[..., Any]] = None,
                     budget: Optional[Budget] = None) -> Dict[int, Dict[str, Any]]:
    rows = [(i, r) for i, r in enumerate(_rows(payload, "interactions")) if isinstance(r, dict)]
    out: Dict[int, Dict[str, Any]] = {}
    for start in range(0, len(rows), 40):
        items = [{"id": i, "entity_1": r.get("entity_1"), "entity_2": r.get("entity_2"), "relationship": r.get("relationship"),
                  "evidence": str(r.get("evidence") or "")[:400]} for i, r in rows[start:start + 40]]
        left = budget.take() if budget is not None else None
        if budget is not None and left is None:
            break
        reply = _ask(_IX_SYSTEM, json.dumps({"rows": items, "paper_text": _paper(text)}, ensure_ascii=False),
                     chat_fn, "Stage 1 interaction typing", left)
        for a in reply.get("rows") or []:
            if isinstance(a, dict) and isinstance(a.get("id"), int):
                out[a["id"]] = a
    return out


def apply_types(payload: Dict[str, Any], text: str, answers: Dict[int, Dict[str, Any]]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """A row changes only when (a) its word carries no sign anywhere (:func:`retypable`: empty, or an
    unsigned Binding / Unspecified / Regulation word; "inhibited_by", "allosteric_inhibitor_of" are signed),
    (b) the model's type is activation or inhibition, (c) the quote is the paper's and no negation or loss
    word the quote left out stands before it in its sentence (:func:`quote_unqualified`), and (d) the quote
    states an effect, not only a contact (:func:`states_effect`; R1 F5, F6)."""
    tw = text_words(text)
    out = copy.deepcopy(payload)
    rep: Dict[str, Any] = {"rows": [], "changed": []}
    for i, r in enumerate(_rows(out, "interactions")):
        if not isinstance(r, dict):
            continue
        a = answers.get(i) or {}
        want, quote = str(a.get("type") or "").strip().casefold(), str(a.get("quote") or "")
        old = str(r.get("interaction_type") or r.get("relationship") or "")  # as the exporter reads it (R3 P7)
        rec = {"index": i, "entity_1": r.get("entity_1"), "entity_2": r.get("entity_2"), "old": old, "model": want,
               "quote": quote[:400], "quote_in_paper": quote_in_paper(quote, tw), "action": "kept"}
        rec["quote_unqualified"] = rec["quote_in_paper"] and quote_unqualified(quote, text)
        rec["states_effect"] = states_effect(quote)
        if want in SIGNED and retypable(old) and rec["quote_unqualified"] and rec["states_effect"]:
            r["relationship"] = want
            if r.get("interaction_type"):
                r["interaction_type"] = want
            r.setdefault("evidence", quote)
            r["type_evidence"] = quote
            _lineage(r, f"type {old or 'none'} -> {want} from the stated after-effect")
            rec["action"] = "changed"
            rep["changed"].append(i)
        rep["rows"].append(rec)
    return out, rep


def verify_stage_one(payload: Dict[str, Any], text: str, pathway: Dict[str, Any], *, reactions: bool = True,
                     interactions: bool = False, policy: Optional[str] = None,
                     chat_fn: Optional[Callable[..., Any]] = None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Return ``(payload, report)``. Fail-open per step: on error the payload as of the last finished step comes
    back (the same ``payload`` object if the reaction step did not finish), so ``report["reactions"]`` exists
    exactly when its decisions are in the returned payload, and the hook prunes the lock manifest only for those."""
    import time

    report: Dict[str, Any] = {"flags": {ENV_FLAG: reactions, IX_FLAG: interactions}, "error": None}
    budget = Budget()
    started = time.monotonic()
    try:
        from t2pw.llm.client import _resolve_model

        report["model"] = _resolve_model(model_env_var=MODEL_ENV_VAR)
    except Exception:  # noqa: BLE001 - audit only
        report["model"] = None
    try:
        out = payload
        if reactions and _rows(payload, "reactions"):
            ans = ask_reactions(payload, text, pathway, chat_fn, budget)
            report["reaction_answers"] = {str(k): v for k, v in ans.items()}
            out, report["reactions"] = apply_verdicts(out, text, ans, policy or granularity())
        if interactions and _rows(out, "interactions"):
            ians = ask_interactions(out, text, chat_fn, budget)
            report["interaction_answers"] = {str(k): v for k, v in ians.items()}
            out, report["interactions"] = apply_types(out, text, ians)
        report["calls_skipped_over_budget"] = budget.skipped
        return out, report
    except Exception as exc:  # noqa: BLE001 - fail open
        report["error"] = f"{type(exc).__name__}: {exc}"
        return out, report  # a failed typing call keeps the finished reaction step (Gemma run 3 pruned locks, kept rows)
    finally:  # measured, for the live criteria (at most MAX_CALLS calls, at most MAX_SECONDS)
        report["calls"] = MAX_CALLS - budget.calls
        report["seconds"] = round(time.monotonic() - started, 1)


def omitted_lock_ids(report: Dict[str, Any]) -> List[str]:
    return [x["lock_id"] for x in ((report.get("reactions") or {}).get("rows") or [])
            if x.get("lock_id") and (x["action"] == "omitted" or x["action"].startswith("folded"))]


def prune_lock_manifest(manifest_dir: Any, lock_ids: Sequence[str]) -> int:
    """Drop the omitted rows' entries from ``locked_reaction_manifest.json`` (written by Stage 1 before this
    hook ran) so the manifest and the payload agree (R1 F8). -> entries removed. Never raises."""
    if not lock_ids or manifest_dir is None:
        return 0
    try:
        from pathlib import Path

        from t2pw.pipeline.reaction_lock_manifest import MANIFEST_FILENAME

        f = Path(manifest_dir) / MANIFEST_FILENAME
        if not f.is_file():
            return 0
        data = json.loads(f.read_text(encoding="utf-8"))
        rows = data if isinstance(data, list) else data.get("locked_reactions") if isinstance(data, dict) else None
        if not isinstance(rows, list):
            return 0
        keep = [e for e in rows if not (isinstance(e, dict) and e.get("locked_reaction_id") in set(lock_ids))]
        if len(keep) == len(rows):
            return 0
        if isinstance(data, list):
            data = keep
        else:
            data["locked_reactions"] = keep
        f.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        return len(rows) - len(keep)
    except Exception:  # noqa: BLE001 - bookkeeping, never a reason to fail
        return 0


def apply_to_stage_one(stage_one: Dict[str, Any], text: str, state: Any, diagnostics_dir: Any,
                       pathway_context: Any = None, task: Any = None, manifest_dir: Any = None) -> Dict[str, Any]:
    """The hook ``run_stage_one`` calls when a flag is on: verify, prune the lock manifest, record.

    The full report goes into ``state[REPORT_STATE_KEY]``, which the batch driver writes into the run's own
    output folder as ``reaction_verifier_report.json`` (R1 F8); a copy also goes to ``diagnostics_dir``."""
    ctx = dict(pathway_context) if isinstance(pathway_context, dict) else {}
    ctx["task"] = str(task or "")[:600]
    fixed, report = verify_stage_one(stage_one, text, ctx, reactions=enabled(), interactions=typing_enabled())
    report["lock_manifest_entries_removed"] = prune_lock_manifest(manifest_dir, omitted_lock_ids(report))
    state["reaction_verifier"] = {k: report.get(k) for k in ("flags", "error")}
    state[REPORT_STATE_KEY] = report
    try:
        from pathlib import Path

        Path(diagnostics_dir).mkdir(parents=True, exist_ok=True)
        (Path(diagnostics_dir) / REPORT_FILE).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    except Exception:  # noqa: BLE001 - the side file is audit, never a reason to fail
        pass
    return fixed
