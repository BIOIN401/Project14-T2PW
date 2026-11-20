"""D-W24 (2) and (7a), lane L7: estimate the compartment of reactions and transport sides that have none.

Runs in ``final_payload.maybe_apply_final_id_fill`` right AFTER the L6 state merge (flag ``T2PW_FINAL_PAYLOAD``; off = the
payload is untouched, so the flag-off path is byte-identical). After the merge, so the neighbours and the candidate states
are the merged ones (two "ER state" rows are one place), and a placeholder L6 already merged into a stated state is not
estimated twice. Cost of that order: two reactions that become exact duplicates only once this step places them are not
merged by L6 (counted in the report as ``new_duplicate_pairs``).

"No stated compartment" = an empty ``biological_state`` or a reference to a placeholder (``__auto_state__``,
``AutoState_*``). A name that resolves to no state is left alone (that is a different error the IR reports). Today the IR
puts such a reaction in ``biological_states[0]`` (ir.py ``default_bs_key``), whatever that state is.

Order, per reaction:
1. Neighbours (deterministic): walk the reactions that share a non-currency, non-hub participant with it
   (``cofactor_policy.VOCABULARY`` is excluded: ATP, H2O, CoA ... link every compartment), through other unstated reactions,
   stopping at stated ones. If every stated reaction reached sits in ONE state, use it (``neighbour_estimate``); the
   evidence is the shared participants and the neighbour reaction names. Estimates never feed other estimates.
2. DEFERRED (``LLM_ESTIMATE = False``; never measured). When on: LLM (``llm/prompts/compartment_estimate_system.txt``): one call per payload for everything step 1 left, temperature 0,
   JSON, with the paper text (or the evidence sentences), the organism and the candidate states. An answer is accepted only
   when its quote (3+ words) is verbatim in the text, names one of the item's participants (not a currency or hub
   compound) or enzymes, holds a phrase whose location key is the chosen place's (one shared generic word such as "cell"
   or "space" is not enough), and the place is a candidate state or a new location that is a PathWhiz location. A provider failure puts only
   ``compartment_estimate_unavailable`` on the rows; the error, paths redacted, stays in the report. Replies are cached in the fill cache (key: prompt + request sha256), so a replay is
   deterministic; ``T2PW_FINAL_PAYLOAD_BUDGET_S=0`` (and ``T2PW_OFFLINE_CURATOR`` without an injected ``chat_fn``) skips the
   call (a cached reply is still used, as the other fills do) and the row stays flagged (``llm_estimate``).
3. DEFERRED (``GENERAL_STATE_FALLBACK = False``): the reaction keeps its state (the IR puts it in the referenced
   placeholder or the first listed state), ``unresolved``, ``review_required`` with reason ``compartment_not_stated``
   naming where it lands and that the placement is not evidence. When switched on: the organism's general cell state (a stated state of the organism with location cytoplasm (prokaryote) /
   cell (eukaryote) and no cell type or tissue; else the organism's ``__auto_state__`` renamed "<organism> <location>" with
   every reference re-pointed; else a new "<organism> cytoplasm|cell" state), ``review_required`` with reason
   ``compartment_not_stated`` (``unresolved``).

Transports: the same for each missing side. Neighbours only fill a side when the other side is stated (the cargo's other
reactions, minus the stated side's state, sit in one state); with both sides missing only the LLM can give a direction.
Step 3 does not write a side (the general cell state as one end turned a right-by-accident "apical side" into "cell", idx
3): an unestimated side keeps what it had and the row is flagged. Sides that would end up equal are not written either.
Element-level sides are not touched.

Every changed row gets ``compartment_source`` (neighbour_estimate | llm_estimate | unresolved; L6's ``estimated`` = a
placeholder merged into a stated state; absent = stated) and a ``compartment_estimate`` entry with a human ``reason``; the
text record and our SBML print them. The report goes under ``final_payload_compartments`` when anything changed. No row
is removed. Fails open: on any error the payload is exactly what it was.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

from t2pw.service import final_payload_merge as M

FIELD = "final_payload_compartments"
PROMPT_FILE = "compartment_estimate_system.txt"
PROMPT_ID = "t2pw:compartment_estimate:v1"
MODEL_ENV_VAR = "OPENROUTER_EXTRACTION_MODEL"  # the Stage-1 model
MAX_PAPER_CHARS = 60_000  # ponytail: head of the paper only; centre it on the items if long papers lose answers
MAX_ITEMS = 40
LLM_TIMEOUT_S = 120.0
MIN_QUOTE_WORDS = 3
NOT_STATED = "compartment_not_stated"
#: Step 3 (the general cell state) is OFF: on the replays it moved 22 reactions off a first state that was sometimes right
#: by accident (idx 134 TML dioxygenase, mitochondrion; idx 86/98 DHCR24, ER), and without a measured LLM step no-loss cannot
#: be shown (orchestrator ruling under the PO's no-regression rule). Off: an unresolved reaction keeps its state and is
#: flagged. Turn on only after the live check passes.
GENERAL_STATE_FALLBACK = False
#: Step 2 (the LLM estimate) is OFF: it has never been measured (the live check got 0 answers) and under D-W24 (9) an
#: unmeasured biology change is deferred. Off: no model call, no cached reply read; the rows stay flagged. The live-check
#: script switches it on for its own process only; the product never does.
LLM_ESTIMATE = False
#: The fixed row reason when the model step was not usable (provider error): the detail stays in the report only.
UNAVAILABLE = "compartment_estimate_unavailable"
RULING = "D-W24 2, L7"
SOURCES = ("unresolved", "llm_estimate", "neighbour_estimate")  # weakest first
SIDES = ("from_biological_state", "to_biological_state")
_ELEMENT_SIDES = ("left_biological_state", "right_biological_state", "from_biological_state", "to_biological_state")
_TRUE = frozenset({"1", "true", "yes", "on", "y", "t"})


def _norm(v: Any) -> str:
    return M._norm(v)


def _unstated(ref: Any) -> bool:
    return not _norm(ref) or M.is_placeholder({"name": ref})


def _text(v: Any) -> str:
    if isinstance(v, list):
        return " ".join(str(x) for x in v if isinstance(x, str))
    return str(v) if isinstance(v, str) else ""


def _linking(names: List[str]) -> Dict[str, str]:
    """norm -> name of the participants that can link two reactions (no currency, no hub)."""
    from t2pw.pipeline.cofactor_policy import VOCABULARY, _normalize

    return {_norm(n): n for n in names if _norm(n) and _normalize(n) not in VOCABULARY}


def _participants(row: Dict[str, Any]) -> List[str]:
    from t2pw.pwml.ir import _coerce_participants

    return [p["name"] for f in ("inputs", "outputs") for p in _coerce_participants(row.get(f))]


def _cargo(row: Dict[str, Any]) -> List[str]:
    out = []
    for e in row.get("transport_elements") or []:
        n = e if isinstance(e, str) else (e.get("element") or e.get("cargo") or e.get("name")) if isinstance(e, dict) else None
        if isinstance(n, str) and n.strip():
            out.append(n)
    if not out:
        n = row.get("cargo") or row.get("cargo_complex")
        if isinstance(n, str) and n.strip():
            out.append(n)
    return out


def _organism(payload: Dict[str, Any], states: List[Dict[str, Any]], hint: str = "") -> str:
    if hint.strip():
        org = M._Organisms(payload, [str(s.get("species") or "") for s in states])
        for name in [s.get("species") for s in states] + [r.get("name") for r in M._species_rows(payload)]:
            if isinstance(name, str) and name.strip() and org.same(name, hint):
                return name.strip()
    counts: Dict[str, int] = {}
    first: Dict[str, str] = {}
    for s in states:
        sp = str(s.get("species") or "").strip()
        if sp:
            counts[_norm(sp)] = counts.get(_norm(sp), 0) + 1
            first.setdefault(_norm(sp), sp)
    if counts:
        order = list(first)
        return first[max(order, key=lambda k: (counts[k], -order.index(k)))]
    from t2pw.pipeline.process_normalizer import _select_default_species_name

    ents = payload.get("entities")
    return _select_default_species_name(ents) if isinstance(ents, dict) else ""


class _Ctx:
    """The payload's states, the organism, and the places this step may write to."""

    def __init__(self, payload: Dict[str, Any], organism_hint: str = "") -> None:
        self.payload = payload
        if not isinstance(payload.get("biological_states"), list):
            payload["biological_states"] = []
        self.rows = payload["biological_states"]
        self.states = [s for s in self.rows if isinstance(s, dict)]
        _, self.last = M._resolver(self.states)
        self.organism = _organism(payload, self.states, organism_hint)
        self.org = M._Organisms(payload, [str(s.get("species") or "") for s in self.states] + [self.organism])
        self.pro = self.org.prokaryote(self.organism) if self.organism else False
        self.cands = {_norm(s.get("name")): s for s in self.states if _norm(s.get("name")) and not M.is_placeholder(s)}

    def key(self, ref: Any) -> Tuple[str, Any]:
        n = _norm(ref)
        return ("row", id(self.states[self.last[n]])) if n in self.last else ("name", n)

    def informative(self, ref: Any) -> Tuple[str, Any]:
        """``key``, but a eukaryote "cell" / "intracellular" state (no compartment named) is keyed "general"."""
        k = self.key(ref)
        if k[0] == "row":
            s = self.states[self.last[_norm(ref)]]
            pro = self.org.prokaryote(s.get("species"))
            if not pro and M.location_key(M._state_loc(s), pro) == "cell":
                return ("general", k[1])
        return k

    def name(self, ref: Any) -> str:
        n = _norm(ref)
        return str(self.states[self.last[n]].get("name")) if n in self.last else str(ref)

    def _plain(self, s: Dict[str, Any]) -> bool:
        return self.org.same(s.get("species"), self.organism) and not s.get("cell_type") and not s.get("tissue")

    def _bare_name(self, s: Dict[str, Any]) -> bool:
        """The name says nothing beyond organism and location ("E. coli state", "cytoplasmic state"), so "afferent nerve fiber
        state" or "barley seedling" (location "cell") is not the organism's general state."""
        key = M.location_key(M._state_loc(s), self.pro)
        words = M._tokens(M._state_loc(s))
        for k, v in M.LOCATION_SYNONYMS.items():
            if M.location_key(k, self.pro) == key:
                words |= M._tokens(k) | M._tokens(v)
        return not M._name_residue(s, words, M._species_rows(self.payload))

    def at(self, loc: str) -> Optional[Dict[str, Any]]:
        """A stated state of the organism at ``loc`` with no cell type or tissue, whose name names nothing else."""
        want = M.location_key(loc, self.pro)
        return next((s for s in self.states if not M.is_placeholder(s) and self._plain(s)
                     and M.location_key(M._state_loc(s), self.pro) == want and self._bare_name(s)), None)

    def free(self, name: str) -> str:
        taken = {_norm(s.get("name")) for s in self.states}
        return name if _norm(name) not in taken else name + " (compartment not stated)"

    def add(self, loc: str, evidence: Optional[str] = None) -> str:
        row = {"name": self.free(f"{self.organism} {loc}"), "species": self.organism, "subcellular_location": loc}
        if evidence:
            row["evidence"] = evidence
        self.rows.append(row)
        self.states.append(row)
        self.last[_norm(row["name"])] = len(self.states) - 1
        self.cands[_norm(row["name"])] = row
        ents = self.payload.get("entities")
        if not isinstance(ents, dict):
            ents = self.payload["entities"] = {}
        locs = ents.get("subcellular_locations")
        if not isinstance(locs, list):
            locs = ents["subcellular_locations"] = []
        if not any(isinstance(r, dict) and _norm(r.get("name")) == _norm(loc) for r in locs):
            locs.append({"name": loc})
        return row["name"]

    def general(self, report: Dict[str, Any]) -> Optional[str]:
        """Step 3's state (created or renamed once, then reused); None when the payload names no organism."""
        if report.get("general_state"):
            return report["general_state"]["name"]
        if not self.organism:
            return None
        loc = "cytoplasm" if self.pro else "cell"
        hit = self.at(loc)
        if hit is not None:
            report["general_state"] = {"name": str(hit["name"]), "how": "existing_state"}
            return str(hit["name"])
        def general_place(s: Dict[str, Any]) -> bool:
            pro = self.org.prokaryote(s.get("species"))
            return not s.get("cell_type") and not s.get("tissue") and M.location_key(M._state_loc(s), pro) == ("cytoplasm" if pro else "cell")

        ph = [s for s in self.states if M.is_placeholder(s) and general_place(s)]
        ph = [s for s in ph if self._plain(s)] or ph  # the organism's own first; else the one the pipeline made
        if len(ph) == 1 and sum(_norm(s.get("name")) == _norm(ph[0].get("name")) for s in self.states) == 1:
            old = str(ph[0]["name"])
            new = self.free(f"{str(ph[0].get('species') or '').strip() or self.organism} {M._state_loc(ph[0])}")
            n = 0
            for holder, k in list(M._iter_state_refs(self.payload)):
                if _norm(holder.get(k)) == _norm(old):
                    holder[k] = new
                    n += 1
            ph[0]["name"] = new
            self.last[_norm(new)] = self.last.pop(_norm(old))
            report["general_state"] = {"name": new, "how": "renamed_placeholder", "from": old, "references_repointed": n}
            return new
        name = self.add(loc)
        report["general_state"] = {"name": name, "how": "created", "location": loc}
        return name


# ---- step 1 ------------------------------------------------------------------------------------------------------------
def _neighbours(rx: List[Any], i: int, links: Dict[int, Dict[str, str]], by_key: Dict[str, List[int]], ctx: _Ctx):
    """Every stated reaction reachable from ``i`` through shared participants and unstated reactions, grouped by state."""
    found: Dict[Tuple[str, Any], List[Dict[str, Any]]] = {}
    seen, queue, via = {i}, [i], []
    while queue:
        j = queue.pop(0)
        for k in sorted(links.get(j, {})):
            for n in by_key.get(k, ()):
                if n in seen:
                    continue
                seen.add(n)
                ref = rx[n].get("biological_state")
                if _unstated(ref):
                    queue.append(n)
                    via.append(str(rx[n].get("name")))
                else:
                    shared = sorted(links[j][x] for x in set(links[j]) & set(links[n]))
                    found.setdefault(ctx.informative(ref), []).append({"reaction": rx[n].get("name"), "state": ctx.name(ref), "shared": shared})
    return found, via


def _neighbour_reason(state: str, nbs: List[Dict[str, Any]], via: List[str]) -> str:
    named = "; ".join(f"'{nb['reaction']}' (shares {', '.join(nb['shared'])})" for nb in nbs[:5])
    more = f" and {len(nbs) - 5} more" if len(nbs) > 5 else ""
    through = f", through {len(via)} other reaction(s) with no stated compartment" if via else ""
    return (f"no compartment stated; every reaction linked to it by shared participants sits in '{state}': "
            f"{named}{more}{through} ({RULING})")


# ---- step 2 ------------------------------------------------------------------------------------------------------------
def _llm_gate(state: Any, chat_fn: Any, fetcher: Any) -> Optional[str]:
    from t2pw.service import final_payload as fp

    if fp._env_budget() == 0:
        return "offline_switch"
    if chat_fn is None:
        if (os.getenv("T2PW_OFFLINE_CURATOR") or "").strip().lower() in _TRUE:
            return "offline_curator"
        if fetcher is not None and fp._circuit_open(fetcher):
            return "network_down"
    dl = state.get(fp.DEADLINE_STATE_KEY) if hasattr(state, "get") else None
    if isinstance(dl, (int, float)) and not isinstance(dl, bool) and dl - time.time() - fp.DEADLINE_MARGIN_S < LLM_TIMEOUT_S:
        return "job_deadline"
    return None


def _system(prompt_file: str = PROMPT_FILE) -> str:
    from t2pw.paths import PROMPTS_DIR

    return (PROMPTS_DIR / prompt_file).read_text(encoding="utf-8")


def _ask(system: str, user: str, chat_fn: Optional[Callable[..., Any]], model: str = "",
         max_tokens: int = 4000) -> Dict[str, Any]:
    if chat_fn is None:
        from t2pw.llm.client import chat_detailed as chat_fn  # noqa: PLW2901
    how = {"model_override": model} if model else {"model_env_var": MODEL_ENV_VAR}
    res = chat_fn([{"role": "system", "content": system}, {"role": "user", "content": user}],
                  temperature=0.0, max_tokens=max_tokens, response_json=True, **how,
                  stage_name="Final payload compartment estimate", timeout=LLM_TIMEOUT_S)
    parsed = json.loads(str(getattr(res, "text", res)))
    if not isinstance(parsed, dict) or not isinstance(parsed.get("items"), list):
        raise ValueError("reply is not {\"items\": [...]}")
    return parsed


def build_request(items: List[Dict[str, Any]], ctx: _Ctx, text: str, vocabulary: Optional[List[str]] = None,
                  prompt_file: str = PROMPT_FILE) -> Tuple[str, str]:
    """(system, user) of the one call; also used by the live-check script to count tokens. ``vocabulary``: the closed list
    a new location must come from (the location pass); absent = any PathWhiz location (L7's own, off, step)."""
    cands = []
    for s in ctx.cands.values():
        c = {"name": s.get("name"), "location": M._state_loc(s), "species": s.get("species")}
        c.update({k: s[k] for k in ("cell_type", "tissue") if s.get(k)})
        cands.append(c)
    body = {"organism": ctx.organism, "candidate_states": cands, "items": items}
    if vocabulary:
        body["new_location_vocabulary"] = vocabulary
    user = json.dumps({**body, "paper_text": text[:MAX_PAPER_CHARS]}, ensure_ascii=False)
    return _system(prompt_file), user


def _names_place(quote: str, loc: Any, pro: bool) -> bool:
    """The quote holds a phrase (1-4 words) whose location key is the chosen place's ("mitochondrial matrix" names the
    mitochondrion, "Golgi complex" the Golgi apparatus); one shared generic word ("cell", "complex", "membrane", "space")
    does not."""
    key = M.location_key(loc, pro)
    if not key:
        return False
    words = re.findall(r"[a-z0-9]+", str(quote).casefold())
    return any(M.location_key(" ".join(words[i:i + n]), pro) == key
               for n in range(1, 5) for i in range(len(words) - n + 1))


def _names_one(quote: str, names: List[str]) -> bool:
    from t2pw.service.transport_sides import _words

    q = f" {_words(quote)} "
    return any(_words(n) and f" {_words(n)} " in q for n in names if isinstance(n, str))


def _accept(ans: Any, ctx: _Ctx, text_words: str, names: List[str],
            vocabulary: Optional[List[str]] = None) -> Tuple[Optional[Tuple[str, str]], str]:
    """((kind, value), quote) with kind "state" (a candidate name) or "new" (a location), or (None, why). The quote must be
    verbatim in the text, name the chosen place, and name one of the item's participants or enzymes (``names``). With
    ``vocabulary``, a new location must be one of it (same location key)."""
    from t2pw.service.transport_sides import _quote_in_text, _words

    if not isinstance(ans, dict):
        return None, "no answer"
    quote = str(ans.get("quote") or "").strip()
    if len(_words(quote).split()) < MIN_QUOTE_WORDS or not _quote_in_text(quote, text_words):
        return None, "quote not found verbatim in the paper text"
    if not _names_one(quote, names):
        return None, "quote names none of the item's participants or enzymes"
    st = str(ans.get("state") or "").strip()
    if st:
        row = ctx.cands.get(_norm(st))
        if row is None:
            return None, f"state '{st}' is not a candidate"
        if not _names_place(quote, M._state_loc(row) or row.get("name"), ctx.pro):
            return None, f"quote does not name the place of '{row['name']}'"
        return ("state", str(row["name"])), quote
    loc = str(ans.get("new_location") or "").strip()
    if not loc or len(loc) > 40:
        return None, "no state and no location given"
    if not _names_place(quote, loc, ctx.pro):
        return None, f"location '{loc}' is not in the quote"
    if vocabulary is not None and M.location_key(loc, ctx.pro) not in {M.location_key(v, ctx.pro) for v in vocabulary}:
        return None, f"location '{loc}' is not in the organism's vocabulary"
    from t2pw.pwml.vocabulary_map import map_vocabulary_name

    if map_vocabulary_name("subcellular-location", loc, None)[2] not in ("exact", "mapped"):
        return None, f"location '{loc}' is not a PathWhiz location"
    hit = ctx.at(loc)
    return ((("state", str(hit["name"])) if hit is not None else ("new", loc)), quote)


# ponytail: a path runs to the next quote, <, >, |, comma, semicolon or line end (spaces and "(x86)" included), so text
# after a path on the same line is cut too; the report loses detail, never leaks a path.
_PATH = re.compile(r"(?:[A-Za-z]:[\\/]|\\\\|~[\\/]|(?<![\w.])/(?=[\w. -]+/))[^'\"<>|,;\n]*")


def _redact(text: str) -> str:
    """A provider error with filesystem paths (a server may name its .env, in a folder with spaces: N3) cut out."""
    return _PATH.sub("<path>", text)


def _same(a: Tuple[str, str], b: Tuple[str, str], ctx: _Ctx) -> bool:
    """Same row, or the same place (F5): location key, position words ("apical" is not "basolateral"), cell type and
    tissue (a new location has no cell type or tissue)."""
    if a[0] == b[0] == "state" and ctx.key(a[1]) == ctx.key(b[1]):
        return True

    def place(t: Tuple[str, str]) -> Tuple[str, frozenset, str, str]:
        if t[0] == "new":
            return M.location_key(t[1], ctx.pro), M.position_words({"name": t[1]}), "", ""
        n = _norm(t[1])
        row = ctx.states[ctx.last[n]] if n in ctx.last else {}
        return (M.location_key(M._state_loc(row) or row.get("name"), ctx.pro), M.position_words(row),
                M.context_key(row.get("cell_type")), M.context_key(row.get("tissue")))

    pa, pb = place(a), place(b)
    return bool(pa[0]) and pa == pb


def _place(target: Tuple[str, str], quote: str, ctx: _Ctx) -> str:
    if target[0] == "state":
        return target[1]
    hit = ctx.at(target[1])
    return str(hit["name"]) if hit is not None else ctx.add(target[1], quote)


def _llm(items: List[Dict[str, Any]], ctx: _Ctx, text: str, state: Any, cache: Any, chat_fn: Any, fetcher: Any,
         today: str, report: Dict[str, Any], vocabulary: Optional[List[str]] = None, model: str = "",
         prompt_file: str = PROMPT_FILE, prompt_id: str = PROMPT_ID, max_tokens: int = 4000) -> Dict[int, Any]:
    """L7's own call with the defaults (its cache key unchanged); the location step passes its own prompt, id, model,
    vocabulary and token cap."""
    from t2pw.clean_model.fill.cache import Cache, request_key

    rep = report["llm"]
    system, user = build_request(items, ctx, text, vocabulary, prompt_file)
    tail = f"\nmodel: {model}" if model else ""  # in the cache key only: another model never reads this reply
    sha = hashlib.sha256((system + "\n" + user + tail).encode("utf-8")).hexdigest()
    key = request_key("llm", prompt_id, {"sha256": sha})
    cache = cache if cache is not None else Cache()
    hit = cache.get(key)
    rep["request_sha256"] = sha
    rep["items"] = len(items)
    if hit is not None and isinstance(hit.get("value"), dict):
        rep["cache_hits"] += 1
        reply = hit["value"]
    else:
        why = _llm_gate(state, chat_fn, fetcher)
        if why:
            rep["skipped"] = why
            return {}
        rep["calls"] += 1
        try:
            reply = _ask(system, user, chat_fn, model, max_tokens)
        except Exception as exc:  # noqa: BLE001 - the step goes on without the model
            rep["error"] = _redact(f"{type(exc).__name__}: {exc}")[:300]
            return {}
        cache.put(key, prompt_id, {"sha256": sha}, reply, today)
    return {a.get("id"): a for a in reply.get("items") or [] if isinstance(a, dict)}


# ---- the step ----------------------------------------------------------------------------------------------------------
def _mark(row: Dict[str, Any], source: str, entry: Dict[str, Any]) -> None:
    old = row.get("compartment_source")
    if old not in SOURCES or SOURCES.index(source) < SOURCES.index(old):
        row["compartment_source"] = source
    row.setdefault("compartment_estimate", []).append(entry)


def _flag(row: Dict[str, Any], reason: str) -> None:
    row["review_required"] = True
    row.setdefault("review_reasons", []).append(f"{NOT_STATED}: {reason}")


def _run(payload: Dict[str, Any], state: Any, report: Dict[str, Any], cache: Any, chat_fn: Any, fetcher: Any,
         today: Optional[str], organism_hint: str) -> None:
    procs = payload.get("processes") if isinstance(payload.get("processes"), dict) else {}
    rx = procs.get("reactions") if isinstance(procs.get("reactions"), list) else []
    tr = procs.get("transports") if isinstance(procs.get("transports"), list) else []
    todo = [i for i, r in enumerate(rx) if isinstance(r, dict) and _unstated(r.get("biological_state"))]
    ttodo = [i for i, t in enumerate(tr) if isinstance(t, dict)
             and not any(isinstance(e, dict) and any(_norm(e.get(f)) for f in _ELEMENT_SIDES) for e in t.get("transport_elements") or [])
             and any(_unstated(t.get(f)) for f in SIDES)]
    if not todo and not ttodo:
        return
    ctx = _Ctx(payload, organism_hint)
    orig = {i: rx[i].get("biological_state") for i in todo}  # step 3 may rename a placeholder these refer to
    t_orig = {i: {f: tr[i].get(f) for f in SIDES} for i in ttodo}
    crossing = {k for t in tr if isinstance(t, dict) for k in _linking(_cargo(t))}
    every = {i: _linking(_participants(r)) for i, r in enumerate(rx) if isinstance(r, dict)}
    links = {i: {k: v for k, v in ks.items() if k not in crossing} for i, ks in every.items()}
    by_key: Dict[str, List[int]] = {}  # reactions by participant (a transported one included: the transport step reads it)
    for i, ks in every.items():
        for k in ks:
            by_key.setdefault(k, []).append(i)

    # step 1 (all from the stated reactions only, then applied)
    est: Dict[int, Tuple[str, str, Dict[str, Any]]] = {}
    why: Dict[int, str] = {}
    hints: Dict[int, List[str]] = {}
    for i in todo:
        found, via = _neighbours(rx, i, links, by_key, ctx)
        if len(found) == 1 and next(iter(found))[0] == "row":
            nbs = next(iter(found.values()))
            st = nbs[0]["state"]
            est[i] = ("neighbour_estimate", st, {"reason": _neighbour_reason(st, nbs, via), "neighbours": nbs[:10], "via": via[:10]})
        else:
            why[i] = ("linked reactions sit only in a state that names no compartment" if found and all(k[0] == "general" for k in found)
                      else f"linked reactions sit in {len(found)} different states" if found
                      else "no linked reaction has a stated compartment")
            hints[i] = sorted({nb["state"] for nbs in found.values() for nb in nbs})
    t_est: Dict[Tuple[int, str], Tuple[str, str, Dict[str, Any]]] = {}
    t_why: Dict[int, str] = {}
    for i in ttodo:
        t = tr[i]
        missing = [f for f in SIDES if _unstated(t.get(f))]
        if len(missing) == 2:
            t_why[i] = "neither side stated; a direction is not inferred from neighbours"
            continue
        other = t.get(SIDES[1] if missing[0] == SIDES[0] else SIDES[0])
        cargo = _linking(_cargo(t))
        found: Dict[Tuple[str, Any], List[Dict[str, Any]]] = {}
        for k in sorted(cargo):
            for n in by_key.get(k, ()):
                ref = rx[n].get("biological_state")
                # by place, not by state name: "intracellular" is the cytoplasm of a bacterium
                if not _unstated(ref) and not _same(("state", ctx.name(ref)), ("state", ctx.name(other)), ctx):
                    found.setdefault(ctx.informative(ref), []).append({"reaction": rx[n].get("name"), "state": ctx.name(ref), "shared": [cargo[k]]})
        if len(found) == 1 and next(iter(found))[0] == "row":
            nbs = next(iter(found.values()))
            st = nbs[0]["state"]
            t_est[(i, missing[0])] = ("neighbour_estimate", st, {"reason": (
                f"{missing[0].split('_')[0]} side not stated; the other reactions of the cargo, outside '{ctx.name(other)}', "
                f"sit in '{st}': " + "; ".join(f"'{nb['reaction']}'" for nb in nbs[:5]) + f" ({RULING})"), "neighbours": nbs[:10]})
        else:
            t_why[i] = (f"the cargo's other reactions sit in {len(found)} states besides the stated side" if found
                        else "no other reaction of the cargo has a stated compartment")

    # step 2
    items: List[Dict[str, Any]] = []
    ask: Dict[int, Tuple[str, int, List[str]]] = {}
    from t2pw.pwml.ir import _actor_name_and_hint
    names: Dict[int, List[str]] = {}
    for i in [i for i in todo if i not in est]:
        r = rx[i]
        ins = [p for p in _participants({"inputs": r.get("inputs")})]
        outs = [p for p in _participants({"outputs": r.get("outputs")})]
        ev = _text(r.get("evidence"))
        item = {"id": len(items), "kind": "reaction", "name": r.get("name"), "equation": " + ".join(ins) + " -> " + " + ".join(outs),
                "enzymes": [_actor_name_and_hint(a)[0] for a in (r.get("enzymes") or [])], "evidence": ev}
        if hints.get(i):
            item["states_of_linked_reactions"] = hints[i]
        ask[item["id"]] = ("reaction", i, [])
        names[item["id"]] = list(_linking(ins + outs).values()) + item["enzymes"]  # N2: no currency / hub compound
        items.append(item)
    for i in ttodo:
        t = tr[i]
        sides = [f for f in SIDES if _unstated(t.get(f)) and (i, f) not in t_est]
        if not sides:
            continue
        item = {"id": len(items), "kind": "transport", "name": t.get("name"), "cargo": _cargo(t), "evidence": _text(t.get("evidence")),
                "current_from": None if _unstated(t.get(SIDES[0])) else ctx.name(t.get(SIDES[0])),
                "current_to": None if _unstated(t.get(SIDES[1])) else ctx.name(t.get(SIDES[1])),
                "ask": [f.split("_")[0] for f in sides]}
        ask[item["id"]] = ("transport", i, sides)
        names[item["id"]] = list(_linking(_cargo(t)).values()) + [_actor_name_and_hint(a)[0] for a in (t.get("transporters") or [])]
        items.append(item)
    items, dropped = items[:MAX_ITEMS], items[MAX_ITEMS:]
    for it in dropped:
        ask.pop(it["id"], None)
    paper = state.get("final_payload_paper_text") if hasattr(state, "get") else None
    text = paper if isinstance(paper, str) and paper.strip() else "\n".join(it["evidence"] for it in items if it.get("evidence"))
    if items:
        report["llm"]["text_source"] = "paper_text" if text is paper else "evidence_sentences"
    answers: Dict[Any, Any] = {}
    if items and not LLM_ESTIMATE:
        report["llm"]["skipped"] = "deferred"
    elif items and text.strip():
        from t2pw.service.transport_sides import _window

        for it in items:
            it["text_around_evidence"] = _window(text, it["evidence"]) if it.get("evidence") else ""
        import datetime

        answers = _llm(items, ctx, text, state, cache, chat_fn, fetcher, today or datetime.date.today().isoformat(), report)
    elif items:
        report["llm"]["skipped"] = "no_text"
    from t2pw.service.transport_sides import _words
    text_words = f" {_words(text)} "
    for n, (kind, i, sides) in ask.items():
        ans = answers.get(n)
        if kind == "reaction":
            target, quote = _accept(ans, ctx, text_words, names[n]) if ans is not None else (None, "")
            if target is None:
                if ans is not None:
                    report["llm"]["rejected"].append({"reaction": rx[i].get("name"), "reason": quote})
                    why[i] += f"; the model's answer was rejected: {quote}"
                continue
            st = _place(target, quote, ctx)
            est[i] = ("llm_estimate", st, {"reason": f"no compartment stated; estimated from the paper: \"{quote}\" ({RULING})",
                                           "quote": quote})
            continue
        got = {}
        for f in sides:
            side = (ans or {}).get(f.split("_")[0]) if isinstance(ans, dict) else None
            target, quote = _accept(side, ctx, text_words, names[n]) if side is not None else (None, "")
            if target is not None:
                got[f] = (target, quote)
            elif side is not None:
                report["llm"]["rejected"].append({"transport": tr[i].get("name"), "side": f, "reason": quote})
        # the two final sides must differ
        t = tr[i]
        fin = {f: got[f][0] if f in got else (None if _unstated(t.get(f)) else ("state", ctx.name(t.get(f)))) for f in SIDES}
        if fin[SIDES[0]] and fin[SIDES[1]] and _same(fin[SIDES[0]], fin[SIDES[1]], ctx):
            report["llm"]["rejected"].append({"transport": t.get("name"), "reason": "both sides would be the same state"})
            got = {}
        for f, (target, quote) in got.items():
            st = _place(target, quote, ctx)
            t_est[(i, f)] = ("llm_estimate", st, {"reason": f"{f.split('_')[0]} side not stated; estimated from the paper: \"{quote}\" ({RULING})",
                                                  "quote": quote})

    # apply 1 and 2
    for i, (src, st, ev) in sorted(est.items()):
        r = rx[i]
        old = r.get("biological_state")
        r["biological_state"] = st
        _mark(r, src, {"step": "L7", "field": "biological_state", "from_state": old or None, "to_state": st, "source": src, **ev})
        report["reactions"].append({"row": i, "reaction": r.get("name"), "from_state": old or None, "to_state": st, "source": src,
                                    "reason": ev["reason"]})
    for (i, f), (src, st, ev) in sorted(t_est.items()):
        t = tr[i]
        old = t.get(f)
        t[f] = st
        _mark(t, src, {"step": "L7", "field": f, "from_state": old or None, "to_state": st, "source": src, **ev})
        report["transports"].append({"row": i, "transport": t.get("name"), "field": f, "from_state": old or None, "to_state": st,
                                     "source": src, "reason": ev["reason"]})

    # step 3
    # Rows carry only fixed words; a provider's error text (it can name server paths) stays in the report, redacted (F3).
    llm_note = ("; " + UNAVAILABLE) if report["llm"].get("error") else (
        f"; model step {report['llm']['skipped']}" if report["llm"].get("skipped") else "")
    for i in todo:
        if i in est:
            continue
        r = rx[i]
        old = orig[i]
        if not GENERAL_STATE_FALLBACK:  # deferred (orchestrator ruling under the no-regression rule): keep, flag, say where it lands
            first = (_norm(old) and ctx.name(old)) or (str(ctx.states[0].get("name")) if ctx.states else "a generated default state")
            reason = (f"no compartment stated and none could be estimated ({why.get(i)}{llm_note}); left where the PWML puts it, "
                      f"'{first}' (the referenced placeholder, or the first listed state), because no estimate was available ({RULING})")
            _flag(r, f"left in '{first}' because no estimate was available; that placement is not evidence ({RULING})")
            _mark(r, "unresolved", {"step": "L7", "field": "biological_state", "from_state": old or None, "to_state": old or None,
                                    "placed_in": first, "source": "unresolved", "reason": reason})
            report["reactions"].append({"row": i, "reaction": r.get("name"), "from_state": old or None, "to_state": old or None,
                                        "placed_in": first, "source": "unresolved", "reason": reason})
            continue
        g = ctx.general(report)
        if g is None:
            reason = f"no compartment stated and no organism to place it in ({why.get(i)}{llm_note}); left to the default state ({RULING})"
            _flag(r, reason)
            _mark(r, "unresolved", {"step": "L7", "field": "biological_state", "from_state": old or None, "to_state": None,
                                    "source": "unresolved", "reason": reason})
            report["reactions"].append({"row": i, "reaction": r.get("name"), "from_state": old or None, "to_state": None,
                                        "source": "unresolved", "reason": reason})
            continue
        reason = (f"no compartment stated and none could be estimated ({why.get(i)}{llm_note}); placed in '{g}', the general "
                  f"cell state of {ctx.organism} ({RULING})")
        r["biological_state"] = g
        _flag(r, f"placed in '{g}', the general cell state of {ctx.organism} ({RULING})")
        _mark(r, "unresolved", {"step": "L7", "field": "biological_state", "from_state": old or None, "to_state": g,
                                "source": "unresolved", "reason": reason})
        report["reactions"].append({"row": i, "reaction": r.get("name"), "from_state": old or None, "to_state": g,
                                    "source": "unresolved", "reason": reason})
    for i in ttodo:
        t = tr[i]
        missing = [f for f in SIDES if _unstated(t_orig[i][f]) and (i, f) not in t_est]
        if not missing:
            continue
        # A transport side is written only from neighbours or the model. The general cell state is NOT used for a side: on
        # the replays it turned "apical side -> Type III cell" (idx 3, right by accident) into "cell -> Type III cell", a
        # movement from the cell into itself. The side keeps what it had (the IR's default) and the row is flagged.
        reason = (f"side(s) {', '.join(f.split('_')[0] for f in missing)} not stated and not estimated "
                  f"({t_why.get(i) or 'no answer'}{llm_note}); a side is not placed in the general cell state, never "
                  f"written when both sides would be the same state, and no direction is invented; left as they were ({RULING})")
        _flag(t, reason)
        _mark(t, "unresolved", {"step": "L7", "field": ",".join(missing), "from_state": None, "to_state": None,
                                "source": "unresolved", "reason": reason})
        report["transports"].append({"row": i, "transport": t.get("name"), "field": ",".join(missing), "from_state": None,
                                     "to_state": None, "source": "unresolved", "reason": reason})
    report["organism"] = ctx.organism
    report["prokaryote"] = ctx.pro
    report["counts"] = {s: sum(1 for x in report["reactions"] if x["source"] == s) for s in SOURCES}
    report["counts"]["transport_sides"] = {s: sum(1 for x in report["transports"] if x["source"] == s) for s in SOURCES}
    report["new_duplicate_pairs"] = _new_duplicates(rx, [x["row"] for x in report["reactions"] if x["to_state"]])


def _new_duplicates(rx: List[Any], moved: List[int]) -> int:
    """Pairs (moved reaction, other) now with the same state and sides: L6 ran before this step and will not merge them."""
    from t2pw.service.final_payload_merge import _side

    def sig(r: Dict[str, Any]):
        return (_norm(r.get("biological_state")), _side(r.get("inputs")), _side(r.get("outputs")))

    sigs = {i: sig(r) for i, r in enumerate(rx) if isinstance(r, dict)}
    return sum(1 for i in moved for j, s in sigs.items() if j != i and s == sigs[i] and (j not in moved or j > i))


def estimate_compartments(payload: Dict[str, Any], state: Any = None, *, cache: Any = None,
                          chat_fn: Optional[Callable[..., Any]] = None, fetcher: Any = None,
                          today: Optional[str] = None, organism_hint: str = "") -> Dict[str, Any]:
    """Apply the estimate to ``payload`` in place. Never raises; on error the payload is restored and the report says why."""
    report: Dict[str, Any] = {"schema": 1, "reactions": [], "transports": [], "general_state": None, "counts": {},
                              "llm": {"calls": 0, "cache_hits": 0, "skipped": None, "error": None, "rejected": []},
                              "error": None}
    if not isinstance(payload, dict) or FIELD in payload:
        report["skipped"] = "already_estimated" if isinstance(payload, dict) else "not_a_payload"
        return report
    snapshot = copy.deepcopy(payload)
    try:
        _run(payload, state, report, cache, chat_fn, fetcher, today, str(organism_hint or ""))
        if report["reactions"] or report["transports"]:
            payload[FIELD] = copy.deepcopy(report)
    except Exception as exc:  # fail open
        payload.clear()
        payload.update(snapshot)
        report.update(reactions=[], transports=[], general_state=None, error=f"{type(exc).__name__}: {exc}")
    return report
