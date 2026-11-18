"""Scope layer (docs/vision/PLAN.md section 2.2, V5): an LLM reads a request and sets the depth of the view;
the user can re-scope afterwards with no pipeline re-run.

``choose_scope(request)`` and ``rescope(current, message)`` ask the LLM for strict JSON
{"paper_only": bool, "max_hops": int|null, "reason": str}. Any error, timeout or invalid answer falls back to a
small keyword table; if that is unsure too: DEFAULT (choose) / unchanged (rescope). ``llm`` is a
callable(system, user) -> str (tests inject fakes). With ``llm=None`` the real client is used, unless
T2PW_OFFLINE_CURATOR is truthy or the provider is not openrouter: then only the fallback runs (no network).
"""

from __future__ import annotations

import json
import os
import re
import threading
from pathlib import Path
from typing import Callable, Optional, Tuple

from t2pw.vision.view import DEFAULT, FULL, PAPER_ONLY, ScopeSpec, less_detailed, more_detailed

STAGE = "vision_scope"
LLM_TIMEOUT_S = 45.0  # the client's own retry loop can run for minutes; a user waiting on a re-scope cannot
MAX_HOPS = 10
LLM = Callable[[str, str], str]

_SYSTEM = (Path(__file__).parent / "prompts" / "scope_system.txt").read_text(encoding="utf-8")


# --- LLM path -------------------------------------------------------------------------------------------

def _offline() -> bool:
    return (os.getenv("T2PW_OFFLINE_CURATOR") or "").strip().lower() in {"1", "true", "yes", "on", "y", "t"}


def _default_llm() -> Optional[LLM]:
    if _offline():
        return None
    try:
        from t2pw.llm import client  # imported lazily: it reads .env and builds a client at import time
    except Exception:
        return None
    if client.PROVIDER != "openrouter":
        return None  # no hosted provider configured: do not wait on a local server

    def call(system: str, user: str) -> str:
        return client.chat_detailed(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=0.0, response_json=True, stage_name=STAGE,
            model_env_var="OPENROUTER_VISION_SCOPE_MODEL", timeout=LLM_TIMEOUT_S,
        ).text

    return call


def _call_with_timeout(llm: LLM, system: str, user: str) -> str:
    # LLM_TIMEOUT_S bounds only this caller's wait. On a timeout the daemon thread (and the client's own retry
    # loop, whose budget is the global LLM_MAX_RETRIES env var) may keep running in the background; its result
    # is discarded.
    box: dict = {}

    def run() -> None:
        try:
            box["out"] = llm(system, user)
        except Exception as e:  # noqa: BLE001 - any failure means fallback
            box["err"] = e

    t = threading.Thread(target=run, daemon=True)
    t.start()
    t.join(LLM_TIMEOUT_S)
    if t.is_alive():
        raise TimeoutError("scope LLM timed out")
    if "err" in box:
        raise box["err"]
    return box["out"]


def parse_scope(text: str) -> Tuple[ScopeSpec, str]:
    """Strict validation of the LLM answer; raises ValueError on anything else."""
    s = text.strip()
    if s.startswith("```"):
        s = s.strip("`").removeprefix("json").strip()
    d = json.loads(s)
    if not isinstance(d, dict) or "paper_only" not in d or "max_hops" not in d:
        raise ValueError("scope JSON needs paper_only and max_hops")
    po, hops, reason = d["paper_only"], d["max_hops"], d.get("reason", "")
    if not isinstance(po, bool):
        raise ValueError("paper_only must be a bool")
    if hops is not None and (not isinstance(hops, int) or isinstance(hops, bool) or not 0 <= hops <= MAX_HOPS):
        raise ValueError(f"max_hops must be null or an integer 0..{MAX_HOPS}")
    if not isinstance(reason, str):
        raise ValueError("reason must be a string")
    return (PAPER_ONLY if po else ScopeSpec(False, hops)), reason.strip()[:300]


def _ask(llm: Optional[LLM], user: str) -> Optional[Tuple[ScopeSpec, str]]:
    llm = llm or _default_llm()
    if llm is None:
        return None
    try:
        spec, reason = parse_scope(_call_with_timeout(llm, _SYSTEM, user))
    except Exception:  # noqa: BLE001 - error, timeout or invalid answer: use the keyword table
        return None
    return spec, reason or "llm"


# --- keyword fallback -----------------------------------------------------------------------------------

_NUM = {w: i for i, w in enumerate(
    "zero one two three four five six seven eight nine ten".split())}
# A negator, then up to 3 words on the same clause (whitespace only: a comma ends the window, so
# "no problem, show the whole pathway" is not negated).
_NEG = r"(?:\b(?:not|never|no|without)|n't)\s+(?:\w+\s+){0,3}?"
_FULL = re.compile(r"\b(?:(?:whole|entire|complete|full|exhaustive)\s+(?:pathway|network|map|diagram|graph|picture|thing)"
                   r"|everything (?:known|available|you know|there is)|all (?:that is |that's )?known)\b")
_NOT_FULL = re.compile(r"\b(?:full|whole|entire|complete)\s+(?:text|paper|article|manuscript|document|study)\b")
_PAPER = re.compile(
    r"\b(?:only|just|strictly|solely|purely)\b\W+(?:\w+\W+){0,4}?(?:paper|article|manuscript)\b"
    r"|\b(?:paper|article|manuscript)\W+only\b|\bnothing (?:beyond|but|else|outside)\b"
    r"|\bstrictly (?:from|in|based on)\b\W+(?:\w+\W+){0,2}?(?:paper|article|manuscript)\b")
_HOPS = re.compile(r"\b(\d+|" + "|".join(_NUM) + r")\s+(?:steps?|hops?|levels?|layers?)\b")
_MORE = re.compile(r"\b(?:more\s+(?:detail\w*|context|info\w*|background|reactions)|expand\w*|elaborate|"
                   r"broader|bigger|deeper|add more|show more|zoom out|wider|enrich\w*)\b")
_LESS = re.compile(r"\b(?:simpler|simplif\w+|less|fewer|smaller|shrink\s+(?:it|this)|trim\s+(?:it|this|down)|"
                   r"narrower|narrow\s+(?:it|this)|zoom in|cut\s+(?:it\s+)?down|pare\s+(?:it\s+)?down|"
                   r"reduce\s+(?:it|this|the\s+(?:detail|size|scope|context)))\b")
_TOO_MUCH = re.compile(r"\btoo (?:much|detailed|busy|complex|complicated|cluttered|crowded|big|large)\b")
_TOO_LITTLE = re.compile(r"\b(?:too (?:simple|little|sparse|small|few|basic|thin)|not (?:detailed|enough)|"
                         r"lacks? (?:detail|context))\b")


def _clean(text: str) -> Tuple[str, bool]:
    """Lowercase; drop negated depth words ('not the whole pathway') and report whether any was dropped."""
    t = re.sub(r"\s+", " ", text.lower().replace("’", "'"))
    t = re.sub(r"\bnot only\b", "also", t)
    t = _NOT_FULL.sub(" ", t)  # "the full text of the paper" is not a request for the full pathway
    # the negator AND the depth phrase are consumed together
    t, n = re.subn(_NEG + "(?:" + _FULL.pattern + ")", " ", t)
    t, n2 = re.subn(_NEG + "(?:" + _PAPER.pattern + ")", " ", t)
    return t, bool(n or n2)


def _absolute(t: str) -> Optional[Tuple[ScopeSpec, str]]:
    m = _HOPS.search(t)
    if m:
        k = int(m.group(1)) if m.group(1).isdigit() else _NUM[m.group(1)]
        if k <= MAX_HOPS:
            return ScopeSpec(False, k), f"keyword: explicit depth {k}"
    if _PAPER.search(t):
        return PAPER_ONLY, "keyword: paper only"
    if _FULL.search(t):
        return FULL, "keyword: full pathway"
    return None


def fallback_choose(text: str) -> Tuple[ScopeSpec, str]:
    return _absolute(_clean(text)[0]) or (DEFAULT, "keyword: no depth stated, default")


def fallback_rescope(current: ScopeSpec, text: str) -> Tuple[ScopeSpec, str]:
    t, negated = _clean(text)
    hit = _absolute(t)
    if hit:
        return hit
    more, t = _TOO_LITTLE.search(t) is not None, _TOO_LITTLE.sub(" ", t)  # "not detailed enough" asks for more
    less, t = _TOO_MUCH.search(t) is not None, _TOO_MUCH.sub(" ", t)
    t = re.sub(r"\b(?:less|fewer)\s+detail\w*", " less ", t)  # "less detailed" must not count as "detailed"
    more, less = more or bool(_MORE.search(t)), less or bool(_LESS.search(t))
    if more != less:
        return (more_detailed(current), "keyword: more detail") if more else (less_detailed(current), "keyword: less detail")
    return current, "keyword: unsure, unchanged" + (" (negated depth word)" if negated else "")


# --- public ---------------------------------------------------------------------------------------------

def choose_scope(request_text: str, *, llm: Optional[LLM] = None) -> Tuple[ScopeSpec, str]:
    got = _ask(llm, f"MODE: initial\nREQUEST: {request_text}")
    return got or fallback_choose(request_text)


def rescope(current: ScopeSpec, message: str, *, llm: Optional[LLM] = None) -> Tuple[ScopeSpec, str]:
    cur = json.dumps({"paper_only": current.paper_only, "max_hops": current.max_hops})
    got = _ask(llm, f"MODE: rescope\nCURRENT: {cur}\nMESSAGE: {message}")
    return got or fallback_rescope(current, message)
