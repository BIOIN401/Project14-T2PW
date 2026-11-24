"""T2PW_REACTION_VERIFIER / T2PW_INTERACTION_TYPING (default off): check Stage-1 rows against the
paper, after the Stage-1 boundary, and repair or flag them. Nothing is deleted silently.

Reactions (one model call per 30 rows). Each row gets a verdict with a quote from the paper:
``ok`` · ``incomplete`` (a substrate or co-product the paper names for that step is missing) ·
``vague`` (a generic participant name the paper names specifically elsewhere) · ``split`` (a
partial step through made-up intermediates) · ``summary`` (an overall multi-enzyme conversion) ·
``off_pathway`` · ``reversed`` · ``not_stated``. What happens next is deterministic:

* a name the model adds or renames to is used only when it occurs in the paper text;
* ``reversed`` swaps the sides only when the model's quote is verbatim in the paper;
* ``off_pathway`` rows, and ``not_stated`` rows whose own quote and the model's quote are both
  absent from the paper, are omitted;
* ``split`` / ``summary`` rows follow ``T2PW_VERIFIER_GRANULARITY`` (see :data:`GRANULARITY`);
* a row a repair made identical to another row is folded into it (enzymes kept).

An omitted row is marked ``scope_membership: out_of_scope`` and loses its lock id, so the
existing out-of-scope filter removes it before Stage 2; every omission, repair and flag is in
``reaction_verifier_report.json`` with the row, the reason and the quote. Any error fails open
(payload unchanged, error recorded).

Interactions (T2PW_INTERACTION_TYPING): the sign is read from what the paper says happens
downstream (the target's level or activity, the product, the pathway). A relationship changes
only to ``activation`` / ``inhibition`` and only with a verbatim effect quote.
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
MAX_PAPER_CHARS = 150_000
ROWS_PER_CALL = 30
VERDICTS = ("ok", "incomplete", "vague", "split", "summary", "off_pathway", "reversed", "not_stated")
SIGNED = ("activation", "inhibition")

_SYSTEM = """You verify reaction rows extracted from a scientific paper. You get the requested pathway
and organism, the paper text, and numbered rows (inputs -> outputs, enzymes, the evidence the
extractor quoted). Judge each row against the paper only, and give one verdict:
- "ok": the paper states this conversion, with these participants, as a step of the requested pathway.
- "incomplete": the paper states this step but names a substrate or co-product of THIS step that the
  row lacks (for example pyruvate, CO2, AMP, PPi, ATP). Put only the missing names in "add_inputs" /
  "add_outputs", spelled as the paper spells them. Never add enzymes, catalytic cofactors (ThDP, FAD,
  PLP, metal ions), water or protons.
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
    if len(q) < 6:
        return False
    span = int(len(q) * 1.5) + 2
    for start in range(len(doc) - 2):
        if doc[start:start + 3] != q[:3]:
            continue
        i, j = 3, start + 3
        while i < len(q) and j < min(len(doc), start + span):
            if doc[j] == q[i]:
                i += 1
            elif doc[j] in _KEEP_WORDS:
                break
            j += 1
        if i == len(q):
            return True
    return False


def quote_in_paper(evidence: Any, text_words: str) -> bool:
    """Every piece of the quote (split at ellipses) of 3+ words is in the paper, and one has 4+. A piece
    of 6+ words may also leave out a few of the paper's words (see :func:`_gapped`)."""
    parts = [p for p in re.split(r"\.\.\.|…|\[\s*\]", str(evidence or "")) if len(_words(p).split()) >= 3]
    docs = [d.split() for d in text_words.split(" | ")]

    def found(p: str) -> bool:
        return _in_text(p, text_words) or any(_gapped(_words(p).split(), d) for d in docs)
    return bool(parts) and all(found(p) for p in parts) and any(len(_words(p).split()) >= 4 for p in parts)


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


def _ask(system: str, user: str, chat_fn: Optional[Callable[..., Any]], stage: str) -> Dict[str, Any]:
    if chat_fn is None:
        from t2pw.llm.client import chat_detailed as chat_fn  # noqa: PLW2901
    result = chat_fn([{"role": "system", "content": system}, {"role": "user", "content": user}],
                     temperature=0.0, max_tokens=32000, response_json=True,
                     model_env_var=MODEL_ENV_VAR, stage_name=stage)
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
                  chat_fn: Optional[Callable[..., Any]] = None) -> Dict[int, Dict[str, Any]]:
    """Model verdicts per reaction index (one call per :data:`ROWS_PER_CALL` rows)."""
    rows = [(i, r) for i, r in enumerate(_rows(payload, "reactions")) if isinstance(r, dict)]
    out: Dict[int, Dict[str, Any]] = {}
    for start in range(0, len(rows), ROWS_PER_CALL):
        chunk = rows[start:start + ROWS_PER_CALL]
        items = [{"id": i, "name": r.get("name"), "inputs": _names(r.get("inputs")), "outputs": _names(r.get("outputs")),
                  "enzymes": _enzymes(r), "evidence": str(r.get("evidence") or "")[:600]} for i, r in chunk]
        user = json.dumps({**_context(pathway), "rows": items, "paper_text": _paper(text)}, ensure_ascii=False)
        reply = _ask(_SYSTEM, user, chat_fn, "Stage 1 reaction verifier")
        for a in reply.get("rows") or []:
            if isinstance(a, dict) and isinstance(a.get("id"), int):
                out[a["id"]] = a
    return out


def _reach(rows: Sequence[Tuple[int, Dict[str, Any]]], src: set, dst: set) -> bool:
    """True when some name in ``src`` reaches some name in ``dst`` through 2+ of ``rows``."""
    edges: Dict[str, set] = {}
    for _, r in rows:
        for a in _side_key(r.get("inputs")):
            edges.setdefault(a, set()).update(_side_key(r.get("outputs")))
    frontier = {(s, 0) for s in src}
    seen = set()
    while frontier:
        node, hops = frontier.pop()
        for nxt in edges.get(node, ()):
            if nxt in dst and hops + 1 >= 2:
                return True
            if (nxt, hops + 1) not in seen and hops < 12:
                seen.add((nxt, hops + 1))
                frontier.add((nxt, hops + 1))
    return False


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
    def generic_keys(rows: Sequence[Tuple[int, Dict[str, Any]]]) -> set:
        return {_key(n) for _, x in rows for n in _names(x.get("inputs")) + _names(x.get("outputs")) if _generic(n)}

    groups: List[List[Tuple[int, Dict[str, Any]]]] = []
    for i, r in split_rows:
        gen = generic_keys([(i, r)])
        if not gen:
            continue
        hit = [g for g in groups if gen & generic_keys(g)]
        merged = [x for g in hit for x in g] + [(i, r)]
        groups = [g for g in groups if g not in hit] + [merged]
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
        if len(g) < 2 or not ins or not outs:
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


def _bridges(i: int, row: Dict[str, Any], rows: Sequence[Tuple[int, Dict[str, Any]]]) -> bool:
    """True when another row makes one of this row's inputs AND another row uses one of its outputs."""
    ins, outs = set(_side_key(row.get("inputs"))), set(_side_key(row.get("outputs")))
    made = any(ins & set(_side_key(r.get("outputs"))) for j, r in rows if j != i)
    used = any(outs & set(_side_key(r.get("inputs"))) for j, r in rows if j != i)
    return made and used


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
    plain = [(i, r) for i, r in live if verdict[i] not in ("summary", "split")]
    new_rows: List[Dict[str, Any]] = []
    chain_done: set = set()

    for i, r in live:
        a = answers.get(i) or {}
        v = verdict[i] if verdict[i] in VERDICTS else "ok"
        mq = str(a.get("quote") or "")
        rec = {"index": i, "name": r.get("name"), "inputs": _names(r.get("inputs")), "outputs": _names(r.get("outputs")),
               "verdict": v, "reason": str(a.get("reason") or "")[:200], "row_quote": str(r.get("evidence") or "")[:600],
               "row_quote_in_paper": quote_in_paper(r.get("evidence"), tw), "model_quote": mq[:400],
               "model_quote_in_paper": quote_in_paper(mq, tw), "action": "kept"}
        rep["rows"].append(rec)
        if v == "incomplete":
            added = []
            for side, field in (("inputs", "add_inputs"), ("outputs", "add_outputs")):
                have = set(_side_key(r.get(side)))
                for n in a.get(field) or []:
                    if isinstance(n, str) and n.strip() and _key(n) not in have and _in_text(n, tw):
                        r.setdefault(side, []).append(n.strip())
                        have.add(_key(n))
                        _add_compound(out, n.strip(), "participant named by the paper for this step")
                        added.append([side, n.strip()])
            if added:
                rec.update(action="repaired", added=added)
                _lineage(r, "added " + ", ".join(n for _, n in added))
        elif v == "vague":
            done = []
            for ren in a.get("rename") or []:
                old, new = (ren.get("old"), ren.get("new")) if isinstance(ren, dict) else (None, None)
                if not (isinstance(old, str) and isinstance(new, str) and new.strip() and _in_text(new, tw)):
                    continue
                for side in ("inputs", "outputs"):
                    for j, item in enumerate(r.get(side) or []):
                        if _key(_name(item)) == _key(old):
                            if isinstance(item, dict):
                                item[next(k for k in ("name", "entity", "compound", "protein") if k in item)] = new.strip()
                            else:
                                r[side][j] = new.strip()
                            done.append([old, new.strip()])
            if done:
                for _, new in done:
                    _add_compound(out, new, "specific name the paper gives")
                rec.update(action="repaired", renamed=done)
                _lineage(r, "renamed " + "; ".join(f"{o} -> {n}" for o, n in done))
        elif v == "reversed":
            if quote_in_paper(mq, tw):
                r["inputs"], r["outputs"] = r.get("outputs") or [], r.get("inputs") or []
                rec["action"] = "repaired"
                _lineage(r, "direction swapped to the paper's")
        elif v == "off_pathway":
            # A branch (a source or a sink of the drawn pathway) is omitted; a row that other rows feed AND
            # drain is a link of the chain, and a model's "off pathway" is not enough to cut the chain.
            if _bridges(i, r, live):
                rec["action"] = "flagged"
            else:
                _omit(r)
                rec["action"] = "omitted"
        elif v == "not_stated":
            # Flag only. On the replay the model called five certified paper steps "not stated" (their
            # quotes were not verbatim); a negative judgement by the model never deletes a row.
            rec["action"] = "flagged"
        elif v in ("split", "summary"):
            act = "flagged"
            if policy == "omit" or (policy == "combo" and v == "summary"):
                act = "omitted"
            elif v == "summary" and policy == "merge":
                act = "omitted" if _reach([x for x in plain if x[0] != i], set(_side_key(r.get("inputs"))),
                                          set(_side_key(r.get("outputs")))) else "flagged"
            elif v == "split" and policy in ("merge", "combo"):
                m = a.get("merged_step") if isinstance(a.get("merged_step"), dict) else {}
                ins, outs = [n for n in m.get("inputs") or [] if isinstance(n, str)], [n for n in m.get("outputs") or [] if isinstance(n, str)]
                ok = ins and outs and all(_in_text(n, tw) for n in ins + outs) and quote_in_paper(m.get("quote"), tw)
                if ok:
                    merged = {"name": (r.get("name") or "merged step"), "inputs": [n.strip() for n in ins],
                              "outputs": [n.strip() for n in outs], "enzymes": copy.deepcopy(r.get("enzymes") or []),
                              "biological_state": r.get("biological_state"), "evidence": str(m.get("quote")).strip(),
                              "scope_membership": "core"}
                    for n in merged["inputs"] + merged["outputs"]:
                        _add_compound(out, n, "participant of the paper's whole step")
                elif i in chains:  # no valid whole step from the model: join the chain's own rows
                    merged = None if id(chains[i]) in chain_done else copy.deepcopy(chains[i])
                    chain_done.add(id(chains[i]))
                    ok = True
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
        k = (_side_key(r.get("inputs")), _side_key(r.get("outputs")))
        if k not in seen:
            seen[k] = i
            continue
        if i not in changed and seen[k] not in changed:
            continue
        keep = rows[seen[k]]
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


def ask_interactions(payload: Dict[str, Any], text: str, chat_fn: Optional[Callable[..., Any]] = None) -> Dict[int, Dict[str, Any]]:
    rows = [(i, r) for i, r in enumerate(_rows(payload, "interactions")) if isinstance(r, dict)]
    out: Dict[int, Dict[str, Any]] = {}
    for start in range(0, len(rows), 40):
        items = [{"id": i, "entity_1": r.get("entity_1"), "entity_2": r.get("entity_2"), "relationship": r.get("relationship"),
                  "evidence": str(r.get("evidence") or "")[:400]} for i, r in rows[start:start + 40]]
        reply = _ask(_IX_SYSTEM, json.dumps({"rows": items, "paper_text": _paper(text)}, ensure_ascii=False),
                     chat_fn, "Stage 1 interaction typing")
        for a in reply.get("rows") or []:
            if isinstance(a, dict) and isinstance(a.get("id"), int):
                out[a["id"]] = a
    return out


def apply_types(payload: Dict[str, Any], text: str, answers: Dict[int, Dict[str, Any]]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    from t2pw.pwml.interaction_types import pwml_interaction_type

    tw = text_words(text)
    out = copy.deepcopy(payload)
    rep: Dict[str, Any] = {"rows": [], "changed": []}
    for i, r in enumerate(_rows(out, "interactions")):
        if not isinstance(r, dict):
            continue
        a = answers.get(i) or {}
        want, quote = str(a.get("type") or "").strip().casefold(), str(a.get("quote") or "")
        old = str(r.get("relationship") or "")
        rec = {"index": i, "entity_1": r.get("entity_1"), "entity_2": r.get("entity_2"), "old": old, "model": want,
               "quote": quote[:400], "quote_in_paper": quote_in_paper(quote, tw), "action": "kept"}
        if want in SIGNED and rec["quote_in_paper"] and not pwml_interaction_type(old):
            r["relationship"] = want
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
    """Return ``(payload, report)``; the same ``payload`` object comes back on error."""
    report: Dict[str, Any] = {"flags": {ENV_FLAG: reactions, IX_FLAG: interactions}, "error": None}
    try:
        out = payload
        if reactions and _rows(payload, "reactions"):
            ans = ask_reactions(payload, text, pathway, chat_fn)
            report["reaction_answers"] = {str(k): v for k, v in ans.items()}
            out, report["reactions"] = apply_verdicts(out, text, ans, policy or granularity())
        if interactions and _rows(out, "interactions"):
            ians = ask_interactions(out, text, chat_fn)
            report["interaction_answers"] = {str(k): v for k, v in ians.items()}
            out, report["interactions"] = apply_types(out, text, ians)
        return out, report
    except Exception as exc:  # noqa: BLE001 - fail open
        report["error"] = f"{type(exc).__name__}: {exc}"
        return payload, report


def apply_to_stage_one(stage_one: Dict[str, Any], text: str, state: Any, diagnostics_dir: Any,
                       pathway_context: Any = None, task: Any = None) -> Dict[str, Any]:
    """The hook ``run_stage_one`` calls when a flag is on: verify, record, write the side file."""
    ctx = dict(pathway_context) if isinstance(pathway_context, dict) else {}
    ctx["task"] = str(task or "")[:600]
    fixed, report = verify_stage_one(stage_one, text, ctx, reactions=enabled(), interactions=typing_enabled())
    state["reaction_verifier"] = {k: report.get(k) for k in ("flags", "error")}
    try:
        from pathlib import Path

        Path(diagnostics_dir).mkdir(parents=True, exist_ok=True)
        (Path(diagnostics_dir) / REPORT_FILE).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    except Exception:  # noqa: BLE001 - the side file is audit, never a reason to fail
        pass
    return fixed
