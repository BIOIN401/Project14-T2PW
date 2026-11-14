"""T2PW_TRANSPORT_SIDES (test only, default off): re-ask the model for the missing or equal
from/to states of Stage-1 transports (D-W21c, candidate B of the equal-side trace).

Stage 1 sometimes leaves a transport's sides empty or writes one state on both. The
normalizer then fills ``__auto_state__`` on both sides and the transport goes in and out
of the same state. This step runs once, after the Stage-1 boundary, and only for those
rows:

* a row with two distinct sides is never sent and never touched; no deficient row means
  zero model calls and the very same payload object back;
* a side is accepted only when the verbatim quote the model gives for it is found in the
  paper text (punctuation and case folded, so U+2010 and ASCII hyphens agree);
* an equal pair is never written; a side nobody names stays empty;
* only ``from_biological_state`` / ``to_biological_state`` and, when needed, one new
  ``biological_states`` row change; a rejected row stays exactly as it was;
* a repair that would make the row a copy of another transport (same cargo, same sides)
  is skipped; any error fails open (payload unchanged, error recorded).
"""

from __future__ import annotations

import copy
import json
import os
import re
from typing import Any, Callable, Dict, List, Optional, Tuple

ENV_FLAG = "T2PW_TRANSPORT_SIDES"
MAX_ROWS = 12
MAX_PAPER_CHARS = 100_000
MODEL_ENV_VAR = "OPENROUTER_EXTRACTION_MODEL"  # same model as Stage 1
REPORT_FILE = "transport_sides_report.json"

_SYSTEM = """You complete the two sides of transport rows extracted from a paper.
For each row you get the cargo, the sentence the row was extracted from, the text around
it, the biological states already declared, and the sides the row already has.
"from" is the state the cargo is in before the transport, "to" the state it ends in.
For every side that is missing (or that must be replaced because both sides are equal):
- name the state only if the paper names it or plainly states it. A phrase such as
  "secreted", "out of the cell" or "efflux" states an extracellular destination; "imported",
  "uptake" or "into the cell" states an extracellular origin. Never guess a more specific
  compartment than the text gives; if the text does not say, return null for that side.
- give the verbatim quote from the paper text (copied exactly, at least two words) that
  supports it. A side without such a quote must be null.
- if one of the declared states fits, reuse its name exactly; otherwise give a short plain
  name and the compartment it stands for (for example "extracellular").
- the two sides must never be the same state.
Reply with JSON only:
{"rows": [{"id": 0, "from": {"state": "...", "compartment": "...", "quote": "..."} or null,
           "to": {"state": "...", "compartment": "...", "quote": "..."} or null}]}"""


def enabled() -> bool:
    return os.environ.get(ENV_FLAG, "").strip().casefold() in ("1", "true", "yes")


def _key(value: Any) -> str:
    from t2pw.pipeline.process_normalizer import _normalize

    return _normalize(str(value or ""))


def _words(value: Any) -> str:
    from t2pw.pipeline.localized_repair import _normalize_for_match

    return _normalize_for_match(value)


def _side(row: Dict[str, Any], field: str, side: str) -> str:
    value = str(row.get(field) or "").strip()
    if value:
        return value
    names = {
        str(e.get("biological_state") or "").strip()
        for e in row.get("elements_with_states") or []
        if isinstance(e, dict) and str(e.get("side", "")).strip().lower() == side
    }
    names.discard("")
    return names.pop() if len(names) == 1 else ""


def _sides(row: Dict[str, Any]) -> Tuple[str, str]:
    return _side(row, "from_biological_state", "left"), _side(row, "to_biological_state", "right")


def _deficient(row: Dict[str, Any]) -> bool:
    a, b = _sides(row)
    return not a or not b or _key(a) == _key(b)


def _transports(payload: Dict[str, Any]) -> List[Any]:
    procs = payload.get("processes")
    rows = procs.get("transports") if isinstance(procs, dict) else None
    return rows if isinstance(rows, list) else []


def _window(text: str, evidence: str) -> str:
    """The evidence sentence with one sentence either side (empty when not located)."""
    sentences = re.split(r"(?<=[.!?])\s+", text)
    want = _words(evidence)
    if not want:
        return ""
    head = " ".join(want.split()[:8])
    for i, s in enumerate(sentences):
        w = _words(s)
        if want in w or w in want and w or head in w:
            return " ".join(sentences[max(0, i - 1): i + 2])
    return ""


def _quote_in_text(quote: Any, text_words: str) -> bool:
    q = _words(quote)
    return len(q.split()) >= 2 and f" {q} " in text_words


def _request(rows: List[Tuple[int, Dict[str, Any]]], text: str, states: List[str]) -> str:
    items = []
    for n, (_, row) in enumerate(rows):
        evidence = str(row.get("evidence") or "")
        a, b = _sides(row)
        items.append({"id": n, "name": row.get("name"), "cargo": row.get("cargo"), "evidence": evidence,
                      "text_around_evidence": _window(text, evidence), "current_from": a, "current_to": b})
    paper = text
    if len(paper) > MAX_PAPER_CHARS:
        w = _window(text, str(rows[0][1].get("evidence") or ""))
        at = max(text.find(w[:200]), 0) if w else 0
        paper = text[max(0, at - MAX_PAPER_CHARS // 2): at + MAX_PAPER_CHARS // 2]
    return json.dumps({"declared_states": states, "rows": items, "paper_text": paper}, ensure_ascii=False)


def _ask(user: str, chat_fn: Optional[Callable[..., Any]]) -> Dict[str, Any]:
    if chat_fn is None:
        from t2pw.llm.client import chat_detailed as chat_fn  # noqa: PLW2901
    result = chat_fn(
        [{"role": "system", "content": _SYSTEM}, {"role": "user", "content": user}],
        temperature=0.0, max_tokens=4000, response_json=True,
        model_env_var=MODEL_ENV_VAR, stage_name="Stage 1 transport sides",
    )
    parsed = json.loads(str(getattr(result, "text", result)))
    if not isinstance(parsed, dict):
        raise ValueError("reply is not a JSON object")
    return parsed


def _species(states: List[Any]) -> str:
    names = {str(s.get("species") or "").strip() for s in states if isinstance(s, dict)}
    names.discard("")
    return names.pop() if len(names) == 1 else ""


def repair_transport_sides(
    payload: Dict[str, Any], text: str, *, chat_fn: Optional[Callable[..., Any]] = None
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Return ``(payload, report)``. The same ``payload`` object comes back unless a row was repaired."""
    report: Dict[str, Any] = {"flag": ENV_FLAG, "deficient": [], "calls": 0, "repairs": [],
                              "rejected": [], "new_states": [], "error": None}
    try:
        rows = _transports(payload)
        todo = [(i, r) for i, r in enumerate(rows) if isinstance(r, dict) and _deficient(r)][:MAX_ROWS]
        report["deficient"] = [{"pointer": f"/processes/transports/{i}", "name": r.get("name"),
                                "sides": list(_sides(r))} for i, r in todo]
        if not todo:
            return payload, report
        known = payload.get("biological_states")
        known = known if isinstance(known, list) else []
        names = [str(s.get("name")) for s in known if isinstance(s, dict) and s.get("name")]
        report["calls"] = 1
        reply = _ask(_request(todo, str(text or ""), names), chat_fn)
        text_words = f" {_words(text)} "
        answers = {a.get("id"): a for a in reply.get("rows") or [] if isinstance(a, dict)}
        out = copy.deepcopy(payload)
        out_rows = _transports(out)
        states = out.setdefault("biological_states", []) if isinstance(out.get("biological_states", []), list) else None
        if states is None:
            raise ValueError("biological_states is not a list")

        for n, (i, row) in enumerate(todo):
            ptr = f"/processes/transports/{i}"
            old = _sides(row)
            both_equal = bool(old[0]) and bool(old[1]) and _key(old[0]) == _key(old[1])
            new = {}
            for side, fld, slot in (("from", "from_biological_state", 0), ("to", "to_biological_state", 1)):
                if old[slot] and not both_equal:  # a stated side is kept
                    continue
                got = (answers.get(n) or {}).get(side)
                if isinstance(got, dict) and str(got.get("state") or "").strip() and _quote_in_text(got.get("quote"), text_words):
                    new[fld] = got
            final = [old[0], old[1]]
            for fld, got in new.items():
                final[0 if fld.startswith("from") else 1] = str(got["state"]).strip()
            reason = None
            if not new:
                reason = "no side with a quote found in the paper text"
            elif final[0] and final[1] and _key(final[0]) == _key(final[1]):
                reason = "result would be an equal pair"
            elif final[0] and final[1]:
                for j, other in enumerate(out_rows):
                    if j != i and isinstance(other, dict) and _key(other.get("cargo")) == _key(row.get("cargo")) \
                            and tuple(map(_key, _sides(other))) == (_key(final[0]), _key(final[1])):
                        reason = f"would duplicate transport {j}"
                        break
            if reason:
                report["rejected"].append({"pointer": ptr, "name": row.get("name"), "reason": reason})
                continue
            target = out_rows[i]
            for fld, got in new.items():
                state = str(got["state"]).strip()
                existing = next((s for s in states if isinstance(s, dict) and _key(s.get("name")) == _key(state)), None)
                if existing is not None:
                    state = str(existing["name"])
                else:
                    add = {"name": state}
                    if _species(states):
                        add["species"] = _species(states)
                    comp = str(got.get("compartment") or "").strip()
                    if comp and len(comp) <= 40:
                        add["subcellular_location"] = comp
                    add["evidence"] = str(got.get("quote")).strip()
                    states.append(add)
                    report["new_states"].append({"pointer": ptr, "state": add})
                target[fld] = state
            report["repairs"].append({"pointer": ptr, "name": row.get("name"), "old": list(old),
                                      "new": [target.get("from_biological_state") or old[0], target.get("to_biological_state") or old[1]],
                                      "quotes": {f: str(g.get("quote")) for f, g in new.items()}})
        return (out if report["repairs"] else payload), report
    except Exception as exc:  # noqa: BLE001 - fail open
        report["error"] = f"{type(exc).__name__}: {exc}"
        return payload, report


def apply_to_stage_one(stage_one: Dict[str, Any], text: str, state: Any, diagnostics_dir: Any) -> Dict[str, Any]:
    """The hook ``run_stage_one`` calls when the flag is on: repair, record, write the side list."""
    fixed, report = repair_transport_sides(stage_one, text)
    state["transport_sides"] = report
    try:
        from pathlib import Path

        Path(diagnostics_dir).mkdir(parents=True, exist_ok=True)
        (Path(diagnostics_dir) / REPORT_FILE).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    except Exception:  # noqa: BLE001 - the side list is audit, never a reason to fail
        pass
    return fixed
