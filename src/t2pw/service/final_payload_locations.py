"""Lane loc/assign: put every reaction, transport side and element where the paper says; where it does not, infer a place
and FLAG it as inferred (PO: inferred locations are allowed when always flagged and never override a stated one).

Runs in ``final_payload.maybe_apply_final_id_fill`` right after L7 (``final_payload_compartments``) and before the L8
evidence checks; a live run whose payload already carries L6/L7 gets this step alone. Offline unless ``T2PW_LOCATION_LLM``
is on. Fails open: on any error the payload is exactly what it was. Runs once (``final_payload_locations`` marks it).

Order:
1. States whose only evidence is the Stage-0 "Expected compartments: ..." hint are flagged (never moved).
2. Transport sides to fill: an unplaced side (empty, or a placeholder state with no place); an estimated side (L6
   ``estimated``, L7 estimate) of a transport whose two ends are now the same place; an export's destination L7 filled
   from neighbours. A stated side is never touched.
3. LLM pass (``T2PW_LOCATION_LLM``, gemini-3.8-flash; its own prompt file and cache id, L7's gate): unplaced reactions and the sides of
   step 2. A candidate state or a new location from a closed per-organism vocabulary; the verbatim quote must name the
   place and the item; the two ends of a transport never become one place. Flag ``inferred: llm_quote``.
4. Gram-positive bacterium with no periplasm / outer membrane state (organism default): a side still open on a
   transport whose NAME says export / import goes to the cytoplasm (inside) or the extracellular space (outside). Flag
   ``inferred: organism_default``. Gram-negative or unknown: no name default ("LPS export" ends in the outer membrane).
   3b. (PO ruling 2) Same flag: a transport whose two STATED ends are one place gets its own call (its own prompt and
   cache id); a side is corrected only with a verbatim quote naming the item and the new place, the ends must then
   differ. Exactly one side is corrected; a name that gives the direction fixes which (export / secretion / efflux: "to";
   import / uptake: "from"); only against the paper text (no paper text: no call). Flag ``inferred: llm_corrected``, the
   stated value kept as ``stated_was``. Otherwise unchanged and flagged.
5. Bacterium: unplaced reactions and element rows go to the cytoplasm (an existing plain cytoplasm state, else a new
   "<organism> cytoplasm"). Flag ``inferred: organism_default``. Eukaryotes get no default. (PO ruling 1, "just use main
   organism") Steps 4-5 use the paper's MAIN organism (the organism hint, else the most frequent species among the
   states) even when other organisms are present; the flag names the organism assumed and the others.
6. L6's state merge again (spelling variants, leaflets; it guards transport ends itself): after steps 3-5 a placeholder or
   synonym state no transport end keeps apart merges.
7. A transport whose ends are still one place is flagged ``transport_ends_same_place`` (both ends stated: never changed).

Every placed row gets ``inferred`` (``inferred_sides`` on a transport), ``compartment_source`` (``organism_default`` |
``llm_estimate``), a ``compartment_estimate`` entry with the reason (and quote), and a ``location_inferred`` review reason;
the list is in the report (``inferred``), which the text record prints as "INFERRED LOCATIONS". In the PWML an inferred
place is written like any estimated compartment (D-W24 2): the row sits in that state; the flag lives in the payload, the
text record and our SBML. The UniProt prior (step 3 of the lane plan) is NOT here: measured agreement with paper-stated
locations is 77.9% (< 80%, docs/pwml_output/LOCATIONS.md).
"""

from __future__ import annotations

import copy
import os
import re
from typing import Any, Callable, Dict, List, Optional, Tuple

from t2pw.service import final_payload_compartments as C
from t2pw.service import final_payload_merge as M

FIELD = "final_payload_locations"
LLM_ENV = "T2PW_LOCATION_LLM"
LLM_MODEL = "google/gemini-3.8-flash"
PROMPT_FILE = "location_assign_system.txt"  # its own prompt; L7's (compartment_estimate_system.txt) is untouched
PROMPT_ID = "t2pw:location_assign:v1"
CONFLICT_PROMPT_FILE = "location_conflict_system.txt"  # PO ruling 2: its own call, so the pass's cached replies stay valid
CONFLICT_PROMPT_ID = "t2pw:location_conflict:v1"
MAX_TOKENS = 8000
HINT = "expected compartments"
#: Transport names that say the cargo leaves the cell / goes into it (read from the name only).
EXPORT = re.compile(r"\b(export\w*|secret\w*|efflux\w*|excret\w*|extru\w*)\b", re.I)
IMPORT = re.compile(r"\b(import\w*|uptake|influx\w*|internali[sz]\w*)\b", re.I)
#: Gram-positive genera (one membrane, no periplasm): only for these is an export's destination the outside. The
#: mycolic-acid genera (Corynebacterium, Nocardia, Rhodococcus, Mycobacterium, Gordonia, ...) are left out: they have an
#: outer mycomembrane.
# ponytail: a genus list, not a taxonomy lookup; an unlisted genus counts as unknown and gets no name default.
GRAM_POSITIVE = frozenset({
    "bacillus", "lactococcus", "lactobacillus", "lactiplantibacillus", "lacticaseibacillus", "streptococcus",
    "staphylococcus", "enterococcus", "listeria", "clostridium", "clostridioides", "streptomyces",
    "micrococcus", "bifidobacterium", "leuconostoc", "pediococcus", "paenibacillus", "geobacillus", "actinomyces",
    "saccharopolyspora", "amycolatopsis", "micromonospora", "cutibacterium", "propionibacterium"})
SIDES = C.SIDES
VOCABULARY = {
    "bacterium": ["cytoplasm", "inner membrane", "periplasm", "outer membrane", "cell wall", "extracellular space"],
    "eukaryote": ["cytosol", "nucleus", "mitochondrion", "mitochondrial matrix", "mitochondrial inner membrane",
                  "endoplasmic reticulum", "endoplasmic reticulum membrane", "golgi apparatus", "lysosome",
                  "peroxisome", "vacuole", "chloroplast", "plastid", "plasma membrane", "extracellular space"],
}  # PathWhiz locations only; a place the run itself declares (synaptic vesicle, thylakoid) is offered as a candidate state
_RANK = ("organism_default", "llm_estimate", "neighbour_estimate", "estimated")  # weakest first


def _on() -> bool:
    return (os.getenv(LLM_ENV) or "").strip().lower() in C._TRUE


def _unplaced_state(s: Dict[str, Any]) -> bool:
    loc = M._state_loc(s)
    return not M._norm(loc) or (M.is_placeholder(s) and M._text(loc) in ("cell", "intracellular"))


def _unplaced(ctx: C._Ctx, ref: Any) -> bool:
    """No place: empty, an unknown placeholder name, or a state with no location / a placeholder at "cell". A generated
    ``AutoState_<species><location>`` WITH a location (Membrane, Outer Membrane) is stated."""
    n = M._norm(ref)
    if not n:
        return True
    if n not in ctx.last:
        return M.is_placeholder({"name": ref})
    return _unplaced_state(ctx.states[ctx.last[n]])


def _same_place(ctx: C._Ctx, a: Any, b: Any) -> bool:
    """One place at both ends, or possibly so: two unplaced ends (``__auto_state__`` -> ``__auto_state__``, or two
    differently named placeholders that are both "cell") count as the same place."""
    ua, ub = _unplaced(ctx, a), _unplaced(ctx, b)
    if ua or ub:
        return ua and ub
    return C._same(("state", ctx.name(a)), ("state", ctx.name(b)), ctx)


def _estimated(row: Dict[str, Any], field: str, sources: Tuple[str, ...] = ()) -> bool:
    for e in row.get("compartment_estimate") or []:
        if isinstance(e, dict) and e.get("field") == field and e.get("to_state"):
            if not sources or e.get("source", "estimated") in sources:
                return True
    return False


def _resolve_source(row: Dict[str, Any]) -> None:
    srcs = {e.get("source", "estimated") for e in row.get("compartment_estimate") or [] if isinstance(e, dict) and e.get("to_state")}
    ranked = [s for s in _RANK if s in srcs]
    if ranked:
        row["compartment_source"] = ranked[0]


def _place(row: Dict[str, Any], field: str, state: str, kind: str, reason: str, report: Dict[str, Any], what: str,
           quote: str = "", resolved: bool = True, extra: Optional[Dict[str, Any]] = None) -> None:
    old = row.get(field)
    src = "llm_estimate" if kind in ("llm_quote", "llm_corrected") else kind
    entry = {"step": "locations", "field": field, "from_state": old or None, "to_state": state, "source": src,
             "inferred": kind, "reason": reason, **(extra or {})}
    if quote:
        entry["quote"] = quote
    row[field] = state
    row.setdefault("compartment_estimate", []).append(entry)
    if field == "biological_state":
        row["inferred"] = kind
    else:
        row.setdefault("inferred_sides", {})[field] = kind
    if resolved:  # L7's "left where the PWML puts it" no longer holds
        row["review_reasons"] = [r for r in row.get("review_reasons") or [] if not str(r).startswith(C.NOT_STATED)]
        if row.get("compartment_source") == "unresolved":
            row.pop("compartment_source")
    row["review_required"] = True
    row.setdefault("review_reasons", []).append(f"location_inferred: {kind}: {reason}")
    _resolve_source(row)
    report["inferred"].append({"kind": what, "name": row.get("name") or row.get("compound") or row.get("protein")
                               or row.get("nucleic_acid") or row.get("element_collection") or row.get("entity"),
                               "field": field, "from_state": old or None, "to_state": state, "inferred": kind,
                               "reason": reason, **({"quote": quote} if quote else {}), **(extra or {})})


def _flag_hints(payload: Dict[str, Any], report: Dict[str, Any]) -> None:
    for s in payload.get("biological_states") or []:
        if isinstance(s, dict) and HINT in str(s.get("evidence") or "").casefold() and not s.get("evidence_is_stage0_hint"):
            s["evidence_is_stage0_hint"] = True
            s["review_required"] = True
            s.setdefault("review_reasons", []).append(
                "location_evidence_is_stage0_hint: the only evidence is the preprocessing hint "
                f"{str(s.get('evidence'))[:120]!r}, not a sentence of the paper")
            report["hint_states"].append({"state": s.get("name"), "location": M._state_loc(s), "evidence": s.get("evidence")})


def _side_todo(ctx: C._Ctx, tr: List[Any]) -> Dict[int, List[str]]:
    todo: Dict[int, List[str]] = {}
    for i, t in enumerate(tr):
        if not isinstance(t, dict) or any(isinstance(e, dict) and any(M._norm(e.get(f)) for f in C._ELEMENT_SIDES)
                                          for e in t.get("transport_elements") or []):
            continue
        same = _same_place(ctx, t.get(SIDES[0]), t.get(SIDES[1]))
        # menaquinone (an older L7 fill): "DHNA export" cytoplasm -> intracellular is one place, so its estimated side is redone
        sides = [f for f in SIDES if _unplaced(ctx, t.get(f)) or (same and _estimated(t, f))]
        if sides:
            todo[i] = sides
    return todo


def _ends_ok(ctx: C._Ctx, t: Dict[str, Any], f: str, target: Tuple[str, str], open_sides: List[str]) -> bool:
    other = SIDES[1] if f == SIDES[0] else SIDES[0]
    if other in open_sides or _unplaced(ctx, t.get(other)):
        return True
    return not C._same(target, ("state", ctx.name(t.get(other))), ctx)


# ---- step 3: LLM -------------------------------------------------------------------------------------------------------
def _with_synonyms(payload: Dict[str, Any], names: List[str]) -> List[str]:
    """The names plus every synonym their entity rows carry ("DHNA" for 1,4-dihydroxy-2-naphthoic acid), so a quote that
    uses the paper's abbreviation still names the item."""
    want = {M._norm(n) for n in names if isinstance(n, str)}
    out = list(names)
    for rows in (payload.get("entities") or {}).values():
        for e in rows if isinstance(rows, list) else []:
            if isinstance(e, dict) and M._norm(e.get("name")) in want:
                out += [s for k in ("synonyms", "aliases", "abbreviation", "abbreviations") for s in
                        (e.get(k) if isinstance(e.get(k), list) else [e.get(k)]) if isinstance(s, str) and s.strip()]
    return out


def _answered(ans: Any) -> bool:
    return isinstance(ans, dict) and bool(ans.get("state") or ans.get("new_location"))

def _llm_pass(payload: Dict[str, Any], ctx: C._Ctx, rx: List[Any], tr: List[Any], side_todo: Dict[int, List[str]],
              state: Any, cache: Any, chat_fn: Any, fetcher: Any, today: Optional[str], report: Dict[str, Any]) -> None:
    from t2pw.pwml.ir import _actor_name_and_hint
    from t2pw.service.transport_sides import _window, _words

    rep = report["llm"]
    if not _on():
        rep["skipped"] = "off"
        return
    vocab = VOCABULARY["bacterium" if ctx.pro else "eukaryote"]
    items: List[Dict[str, Any]] = []
    ask: Dict[int, Tuple[str, int, List[str], List[str]]] = {}
    for i, r in enumerate(rx):
        if not isinstance(r, dict) or not _unplaced(ctx, r.get("biological_state")):
            continue
        ins, outs = C._participants({"inputs": r.get("inputs")}), C._participants({"outputs": r.get("outputs")})
        enz = [_actor_name_and_hint(a)[0] for a in r.get("enzymes") or []]
        items.append({"id": len(items), "kind": "reaction", "name": r.get("name"),
                      "equation": " + ".join(ins) + " -> " + " + ".join(outs), "enzymes": enz, "evidence": C._text(r.get("evidence"))})
        ask[items[-1]["id"]] = ("reaction", i, [], _with_synonyms(payload, list(C._linking(ins + outs).values()) + enz))
    for i, sides in side_todo.items():
        t = tr[i]
        items.append({"id": len(items), "kind": "transport", "name": t.get("name"), "cargo": C._cargo(t),
                      "evidence": C._text(t.get("evidence")),
                      "current_from": None if SIDES[0] in sides else ctx.name(t.get(SIDES[0])),
                      "current_to": None if SIDES[1] in sides else ctx.name(t.get(SIDES[1])),
                      "ask": [f.split("_")[0] for f in sides]})
        # the cargo itself names the item, currency or not (H+ through OTOP1)
        ask[items[-1]["id"]] = ("transport", i, sides, _with_synonyms(payload, C._cargo(t)
                                + [_actor_name_and_hint(a)[0] for a in t.get("transporters") or []]))
    items = items[:C.MAX_ITEMS]
    if not items:
        return
    paper = state.get("final_payload_paper_text") if hasattr(state, "get") else None
    text = paper if isinstance(paper, str) and paper.strip() else "\n".join(it["evidence"] for it in items if it.get("evidence"))
    if not text.strip():
        rep["skipped"] = "no_text"
        return
    for it in items:
        it["text_around_evidence"] = _window(text, it["evidence"]) if it.get("evidence") else ""
    import datetime

    answers = C._llm(items, ctx, text, state, cache, chat_fn, fetcher, today or datetime.date.today().isoformat(), report,
                     vocabulary=vocab, model=LLM_MODEL, prompt_file=PROMPT_FILE, prompt_id=PROMPT_ID,
                     max_tokens=MAX_TOKENS)
    text_words = f" {_words(text)} "
    for n, (kind, i, sides, names) in ask.items():
        if n >= len(items):
            continue
        ans = answers.get(n)
        if kind == "reaction":
            target, quote = C._accept(ans, ctx, text_words, names, vocab) if _answered(ans) else (None, "")
            if target is None:
                if _answered(ans):
                    rep["rejected"].append({"reaction": rx[i].get("name"), "reason": quote})
                continue
            st = C._place(target, quote, ctx)
            _place(rx[i], "biological_state", st, "llm_quote", f"not stated for this step; placed from the paper: \"{quote}\"",
                   report, "reaction", quote)
            continue
        t = tr[i]
        got = {}
        for f in sides:
            side = ans.get(f.split("_")[0]) if isinstance(ans, dict) else None
            target, quote = C._accept(side, ctx, text_words, names, vocab) if _answered(side) else (None, "")
            if target is not None:
                got[f] = (target, quote)
            elif _answered(side):
                rep["rejected"].append({"transport": t.get("name"), "side": f, "reason": quote})
        fin = {f: got[f][0] if f in got else (None if f in sides else ("state", ctx.name(t.get(f)))) for f in SIDES}
        if fin[SIDES[0]] and fin[SIDES[1]] and C._same(fin[SIDES[0]], fin[SIDES[1]], ctx):
            rep["rejected"].append({"transport": t.get("name"), "reason": "both ends would be the same place"})
            continue
        for f, (target, quote) in got.items():
            st = C._place(target, quote, ctx)
            _place(t, f, st, "llm_quote", f"{f.split('_')[0]} side not stated; placed from the paper: \"{quote}\"",
                   report, "transport", quote)
            sides.remove(f)


def _conflicts(ctx: C._Ctx, tr: List[Any], side_todo: Dict[int, List[str]]) -> List[int]:
    """Transports whose two STATED ends are one place (run 102: "enterobactin secretion" cytoplasm -> cytoplasm)."""
    out = []
    for i, t in enumerate(tr):
        if not isinstance(t, dict) or i in side_todo or any(
                isinstance(e, dict) and any(M._norm(e.get(f)) for f in C._ELEMENT_SIDES) for e in t.get("transport_elements") or []):
            continue
        a, b = t.get(SIDES[0]), t.get(SIDES[1])
        if not _unplaced(ctx, a) and not _unplaced(ctx, b) and _same_place(ctx, a, b):
            out.append(i)
    return out


def _conflict_pass(payload: Dict[str, Any], ctx: C._Ctx, tr: List[Any], side_todo: Dict[int, List[str]], state: Any,
                   cache: Any, chat_fn: Any, fetcher: Any, today: Optional[str], report: Dict[str, Any]) -> None:
    """PO ruling 2 (contradictory stated locations: "whatever is most correct ... the LLM should be able to reason between
    the lines"), scoped to a transport whose two stated ends are one place. The model gets the paper and the conflict; a
    side is corrected only with a verbatim quote that names the item (cargo, transporter or the process) and the new
    place, and only when the two ends then differ. Flag ``inferred: llm_corrected``; the stated value is kept in the
    estimate entry (``stated_was``). No quote, a failed check or a model error: no change, the row stays flagged."""
    from t2pw.pwml.ir import _actor_name_and_hint
    from t2pw.service.transport_sides import _window, _words

    rep = report["llm_conflict"]
    if not _on():
        rep["skipped"] = "off"
        return
    todo = _conflicts(ctx, tr, side_todo)[:C.MAX_ITEMS]
    if not todo:
        return
    paper = state.get("final_payload_paper_text") if hasattr(state, "get") else None
    if not (isinstance(paper, str) and paper.strip()):
        # a STATED location is corrected only against the paper itself, never the extraction's own evidence strings
        rep["skipped"] = "no_paper_text"
        return
    text = paper
    vocab = VOCABULARY["bacterium" if ctx.pro else "eukaryote"]
    items, names = [], []
    for i in todo:
        t = tr[i]
        ev = C._text(t.get("evidence"))
        stated = ctx.name(t.get(SIDES[0]))
        loc = M._state_loc(ctx.states[ctx.last[M._norm(t.get(SIDES[0]))]]) if M._norm(t.get(SIDES[0])) in ctx.last else ""
        items.append({"id": len(items), "kind": "transport_conflict", "name": t.get("name"), "cargo": C._cargo(t),
                      "evidence": ev, "text_around_evidence": _window(text, ev) if ev else "",
                      "stated_from": stated, "stated_to": ctx.name(t.get(SIDES[1])),
                      "conflict": f"both ends are stated as '{stated}' ({loc}); a transport moves its cargo between two "
                                  "different places"})
        names.append(_with_synonyms(payload, C._cargo(t) + [str(t.get("name") or "")]
                                    + [_actor_name_and_hint(a)[0] for a in t.get("transporters") or []]))
    import datetime

    answers = C._llm(items, ctx, text, state, cache, chat_fn, fetcher, today or datetime.date.today().isoformat(),
                     {"llm": rep}, vocabulary=vocab, model=LLM_MODEL, prompt_file=CONFLICT_PROMPT_FILE,
                     prompt_id=CONFLICT_PROMPT_ID, max_tokens=MAX_TOKENS)
    text_words = f" {_words(text)} "
    for n, i in enumerate(todo):
        t, ans = tr[i], answers.get(n)
        got = {}
        for f in SIDES:
            side = ans.get(f.split("_")[0]) if isinstance(ans, dict) else None
            if not _answered(side):
                continue
            target, quote = C._accept(side, ctx, text_words, names[n], vocab)
            if target is None:
                rep["rejected"].append({"transport": t.get("name"), "side": f, "reason": quote})
            elif C._same(target, ("state", ctx.name(t.get(f))), ctx):
                rep["rejected"].append({"transport": t.get("name"), "side": f, "reason": "same place as stated"})
            else:
                got[f] = (target, quote)
        if not got:
            continue
        if len(got) != 1:  # both stated ends would be overwritten: only one end of a stated pair is ever corrected
            rep["rejected"].append({"transport": t.get("name"), "reason": "both ends returned; exactly one may be corrected"})
            continue
        name = str(t.get("name") or "")
        allowed = (SIDES[1] if EXPORT.search(name) and not IMPORT.search(name) else
                   SIDES[0] if IMPORT.search(name) and not EXPORT.search(name) else None)
        if allowed and allowed not in got:  # "secretion" moves the cargo out: only where it ends up can be wrong
            rep["rejected"].append({"transport": name, "side": next(iter(got)),
                                    "reason": f"the name '{name}' gives the direction; only the {allowed.split('_')[0]} side "
                                              "may be corrected"})
            continue
        fin = {f: got[f][0] if f in got else ("state", ctx.name(t.get(f))) for f in SIDES}
        if C._same(fin[SIDES[0]], fin[SIDES[1]], ctx):
            rep["rejected"].append({"transport": t.get("name"), "reason": "both ends would still be one place"})
            continue
        for f, (target, quote) in got.items():
            was = ctx.name(t.get(f))
            was_loc = M._state_loc(ctx.states[ctx.last[M._norm(t.get(f))]]) if M._norm(t.get(f)) in ctx.last else ""
            st = C._place(target, quote, ctx)
            _place(t, f, st, "llm_corrected",
                   f"{f.split('_')[0]} side was stated as '{was}' ({was_loc}), the same place as the other end; corrected "
                   f"from the paper: \"{quote}\"", report, "transport", quote,
                   extra={"stated_was": was, "stated_location_was": was_loc})


# ---- steps 4 and 6: bacterium ------------------------------------------------------------------------------------------
def _target(ctx: C._Ctx, loc: str) -> str:
    hit = ctx.at(loc)
    return str(hit["name"]) if hit is not None else ctx.add(loc)


def _main_organism_note(payload: Dict[str, Any], ctx: C._Ctx) -> str:
    """PO ruling (mixed-organism papers, "just use main organism"): the default is the MAIN organism's (the organism hint,
    else the most frequent species among the states); the flag names it and any other organism the paper lists."""
    names = [r.get("name") for r in M._species_rows(payload)] + [s.get("species") for s in ctx.states]
    other = sorted({str(n) for n in names if isinstance(n, str) and n.strip() and not ctx.org.same(n, ctx.organism)})
    return (f"main organism assumed: {ctx.organism}" + (f" (the paper also names {', '.join(other)})" if other else ""))


def _name_default_blocked(ctx: C._Ctx) -> str:
    """Why a transport's destination is not read from its name ('' = it may be): a Gram-negative or unknown Gram status
    ("LPS export", Sec "protein export" end in the outer membrane or the periplasm, not outside), or a periplasm / outer
    membrane state in the paper."""
    if (ctx.organism.split() or [""])[0].strip(".").casefold() not in GRAM_POSITIVE:
        return f"Gram status of {ctx.organism} not known to be positive"
    for s in ctx.states:
        if M.location_key(M._state_loc(s) or s.get("name"), True) in ("periplasm", "outer membrane"):
            return f"the paper has a periplasm / outer membrane state ('{s.get('name')}')"
    return ""


def _transport_default(ctx: C._Ctx, tr: List[Any], side_todo: Dict[int, List[str]], report: Dict[str, Any],
                       note: str) -> None:
    for i, sides in side_todo.items():
        t = tr[i]
        name = str(t.get("name") or "")
        m_out, m_in = EXPORT.search(name), IMPORT.search(name)
        if not sides or bool(m_out) == bool(m_in):
            continue
        word = (m_out or m_in).group(0)
        place = {SIDES[0]: "cytoplasm", SIDES[1]: "extracellular space"} if m_out else {SIDES[0]: "extracellular space", SIDES[1]: "cytoplasm"}
        for f in list(sides):
            if not _ends_ok(ctx, t, f, ("new", place[f]), [s for s in sides if s != f]):
                continue
            st = _target(ctx, place[f])
            reason = (f"{f.split('_')[0]} side not stated; the transport is named '{name}' ('{word}'), so in the bacterium "
                      f"{ctx.organism} the cargo moves {place[SIDES[0]]} -> {place[SIDES[1]]}; organism default, not stated; {note}")
            _place(t, f, st, "organism_default", reason, report, "transport", extra={"organism": ctx.organism})
            sides.remove(f)


def _bacterial_default(payload: Dict[str, Any], ctx: C._Ctx, rx: List[Any], report: Dict[str, Any], note: str) -> None:
    reason = (f"no compartment stated in the paper record; {ctx.organism} is a bacterium, so it is placed in its cytoplasm; "
              f"organism default, not stated; {note}")
    target = None
    for r in rx:
        if isinstance(r, dict) and _unplaced(ctx, r.get("biological_state")):
            target = target or _target(ctx, "cytoplasm")
            _place(r, "biological_state", target, "organism_default", reason, report, "reaction", extra={"organism": ctx.organism})
    for bucket, rows in (payload.get("element_locations") or {}).items():
        for r in rows if isinstance(rows, list) else []:
            if isinstance(r, dict) and _unplaced(ctx, r.get("biological_state")):
                target = target or _target(ctx, "cytoplasm")
                _place(r, "biological_state", target, "organism_default", reason, report, bucket,
                       extra={"organism": ctx.organism})


# ---- the step ----------------------------------------------------------------------------------------------------------
def _rerun_state_merge(payload: Dict[str, Any], report: Dict[str, Any]) -> None:
    rep = {"states": [], "deferred": [], "eukaryote_cell_as_cytosol": [], "state_references_repointed": 0,
           "placeholder_references_estimated": 0}
    M._merge_states(payload, rep)
    report["merge"] = {"states": [{k: s.get(k) for k in ("state", "into", "location", "into_location", "placeholder")} for s in rep["states"]],
                       "state_references_repointed": rep["state_references_repointed"]}


def _run(payload: Dict[str, Any], state: Any, report: Dict[str, Any], cache: Any, chat_fn: Any, fetcher: Any,
         today: Optional[str], organism_hint: str) -> None:
    _flag_hints(payload, report)
    procs = payload.get("processes") if isinstance(payload.get("processes"), dict) else {}
    rx = procs.get("reactions") if isinstance(procs.get("reactions"), list) else []
    tr = procs.get("transports") if isinstance(procs.get("transports"), list) else []
    ctx = C._Ctx(payload, organism_hint)
    side_todo = _side_todo(ctx, tr)
    _llm_pass(payload, ctx, rx, tr, side_todo, state, cache, chat_fn, fetcher, today, report)
    _conflict_pass(payload, ctx, tr, side_todo, state, cache, chat_fn, fetcher, today, report)
    if ctx.pro and ctx.organism:  # the MAIN organism is a prokaryote (PO ruling: other organisms do not stop it)
        note = _main_organism_note(payload, ctx)
        report["main_organism"] = note
        blocked = _name_default_blocked(ctx)
        if blocked:
            report["transport_default_skipped"] = blocked
        else:
            _transport_default(ctx, tr, side_todo, report, note)
        _bacterial_default(payload, ctx, rx, report, note)  # before the merge: a row on a placeholder is flagged as a default
    _rerun_state_merge(payload, report)
    ctx = C._Ctx(payload, organism_hint)
    for t in tr:
        if isinstance(t, dict) and _same_place(ctx, t.get(SIDES[0]), t.get(SIDES[1])):
            why = (f"transport_ends_same_place: both ends are '{ctx.name(t.get(SIDES[0]))}' / '{ctx.name(t.get(SIDES[1]))}'; "
                   "no stated or inferable other side")
            t["review_required"] = True
            t.setdefault("review_reasons", []).append(why)
            report["same_place_left"].append({"transport": t.get("name"), "from": t.get(SIDES[0]), "to": t.get(SIDES[1])})
    report["organism"], report["prokaryote"] = ctx.organism, ctx.pro
    report["counts"] = {k: sum(1 for x in report["inferred"] if x["inferred"] == k)
                        for k in ("organism_default", "llm_quote", "llm_corrected")}


def assign_locations(payload: Dict[str, Any], state: Any = None, *, cache: Any = None,
                     chat_fn: Optional[Callable[..., Any]] = None, fetcher: Any = None, today: Optional[str] = None,
                     organism_hint: str = "") -> Dict[str, Any]:
    """Apply the step to ``payload`` in place. Never raises; on error the payload is restored and the report says why."""
    report: Dict[str, Any] = {"schema": 1, "inferred": [], "hint_states": [], "same_place_left": [], "merge": None,
                              "counts": {}, "llm": {"calls": 0, "cache_hits": 0, "skipped": None, "error": None, "rejected": []},
                              "llm_conflict": {"calls": 0, "cache_hits": 0, "skipped": None, "error": None, "rejected": []},
                              "error": None}
    if not isinstance(payload, dict) or FIELD in payload:
        report["skipped"] = "already_assigned" if isinstance(payload, dict) else "not_a_payload"
        return report
    snapshot = copy.deepcopy(payload)
    try:
        _run(payload, state, report, cache, chat_fn, fetcher, today, str(organism_hint or ""))
        if report["inferred"] or report["hint_states"] or report["same_place_left"] or (report["merge"] or {}).get("states"):
            payload[FIELD] = copy.deepcopy(report)
    except Exception as exc:  # fail open
        payload.clear()
        payload.update(snapshot)
        report.update(inferred=[], hint_states=[], same_place_left=[], merge=None, error=f"{type(exc).__name__}: {exc}")
    return report
