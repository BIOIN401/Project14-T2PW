"""Stage R4.5 — LLM prose→reaction extraction (evidence-bound).

Turn a retrieved evidence passage that describes a reaction in **prose** (not as a
clean ``A + B -> C`` equation) into structured reactions, so cross-paper synthesis
is no longer limited to the deterministic arrow-equation parser in
:mod:`t2pw.rag.synthesize`. A sentence like "NdmA catalyzes the N1-demethylation of
caffeine to yield theobromine and formaldehyde" carries no arrow, so the arrow
parser transcribes nothing; this module recovers that reaction.

Strictly evidence-bound (the whole point of RAG here)
-----------------------------------------------------
The model is instructed to transcribe **only** reactions the passage explicitly
states or describes — no inference, no background knowledge, no filling-in. The
extracted reaction is therefore faithful to the chunk it is attached to (WP5
attaches that chunk's ``source_id`` / ``source_uri`` as provenance), so the
"no element without evidence" guarantee (WP6) still holds: a reaction the passage
does not state is never produced. If the passage states no reaction, extraction
returns ``[]`` — it never fabricates one.

Separation invariant (docs/rag/03_separation_invariant.md)
----------------------------------------------------------
All of this lives in ``t2pw.rag``; the dependency arrow points **RAG -> core
only**. This module reuses the existing OpenAI-compatible chat client
(``t2pw.llm.client.chat``, imported **lazily** so importing this module needs no
LLM client / key / network) and edits no stage module.

Offline-first / opt-in
----------------------
Extraction is **opt-in**: :func:`t2pw.rag.synthesize.synthesize_with_report` runs
it only when the orchestrator passes a ``prose_extractor`` (the app builds one
from the real ``chat`` when ``RAG_EXTRACT_REACTIONS`` is on; tests inject a fake
``chat_fn``; offline/unit runs pass nothing and behave exactly as before). Every
call fails **closed**: any error — no endpoint, a malformed response, a model that
rejects JSON mode — degrades to ``[]``, so prose extraction can only *add*
reactions, never break synthesis.
"""

from __future__ import annotations

import json
from typing import Any, Callable, Dict, List, Optional

# A ``chat_fn`` takes (system_prompt, user_text) and returns the raw model text.
# Injected in tests; defaults to the shared LLM client in production.
ChatFn = Callable[[str, str], str]

# Bound the model's output so one passage can never balloon the response.
_MAX_TOKENS = 1500

# The system prompt is the guarantee. It forbids invention explicitly and
# repeatedly, because an invented reaction here would be attached to a real
# paper's provenance — a false attribution the downstream gates cannot catch.
_SYSTEM_PROMPT = (
    "You extract biochemical reactions from a passage of scientific text and "
    "return them as JSON.\n\n"
    "STRICT RULES:\n"
    "1. Transcribe ONLY reactions the passage explicitly states or describes. "
    "Do NOT add reactions from your own knowledge. Do NOT infer or complete "
    "steps the passage does not state. If in doubt, leave it out.\n"
    "2. For each reaction, list the inputs (substrates) and outputs (products) "
    "using the exact chemical/metabolite names the passage uses.\n"
    "3. Include a stoichiometric coefficient ONLY when the passage states it; "
    "otherwise use null.\n"
    "4. List the catalyzing enzyme(s)/protein(s) by the exact name or gene "
    "symbol the passage uses (e.g. \"NdmA\"). If none is named, use [].\n"
    "5. Set \"reversible\" to true ONLY if the passage describes the reaction "
    "as reversible/bidirectional; otherwise false.\n"
    "6. Give a short descriptive \"name\" when the passage implies one "
    "(e.g. \"caffeine N1-demethylation\"); otherwise use \"\".\n"
    "7. If the passage states NO explicit reaction, return {\"reactions\": []}. "
    "Never fabricate a reaction to fill the gap.\n\n"
    "Return ONLY a JSON object of the form:\n"
    "{\"reactions\": [{\"name\": \"\", \"inputs\": [{\"name\": \"\", "
    "\"stoichiometry\": null}], \"outputs\": [{\"name\": \"\", "
    "\"stoichiometry\": null}], \"enzymes\": [\"\"], \"reversible\": false}]}"
)

# OpenAI-compatible ``response_format`` json_schema. The shared client degrades
# gracefully (json_object, then no format) if a model rejects it, and the parser
# below is lenient regardless, so this is a best-effort steer, not a hard gate.
_RESPONSE_JSON_SCHEMA: Dict[str, Any] = {
    "name": "rag_reaction_extraction",
    "schema": {
        "type": "object",
        "properties": {
            "reactions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "inputs": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "name": {"type": "string"},
                                    "stoichiometry": {"type": ["number", "null"]},
                                },
                                "required": ["name"],
                            },
                        },
                        "outputs": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "name": {"type": "string"},
                                    "stoichiometry": {"type": ["number", "null"]},
                                },
                                "required": ["name"],
                            },
                        },
                        "enzymes": {"type": "array", "items": {"type": "string"}},
                        "reversible": {"type": "boolean"},
                    },
                    "required": ["inputs", "outputs"],
                },
            }
        },
        "required": ["reactions"],
    },
}


def _default_chat_fn(system_prompt: str, user_text: str) -> str:
    """Production ``chat_fn``: call the shared client (imported lazily)."""
    from t2pw.llm.client import chat  # lazy: importing this module needs no client

    return chat(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_text},
        ],
        temperature=0.0,
        max_tokens=_MAX_TOKENS,
        response_json=True,
        json_schema=_RESPONSE_JSON_SCHEMA,
        model_env_var="OPENROUTER_RAG_EXTRACT_MODEL",
        stage_name="rag_prose_extraction",
    )


def _loads_lenient(raw: Any) -> Optional[Dict[str, Any]]:
    """Parse model output to a dict, tolerating markdown fences / surrounding prose."""
    text = str(raw or "").strip()
    if not text:
        return None
    # Strip a ```json ... ``` fence if present.
    if text.startswith("```"):
        text = text.strip("`").strip()
        if text[:4].lower() == "json":
            text = text[4:].strip()
    try:
        data = json.loads(text)
        return data if isinstance(data, dict) else None
    except Exception:  # noqa: BLE001 - fall through to brace extraction
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            data = json.loads(text[start : end + 1])
            return data if isinstance(data, dict) else None
        except Exception:  # noqa: BLE001 - unparseable -> treated as no reactions
            return None
    return None


def _clean_side(value: Any) -> List[Dict[str, Any]]:
    """Normalize an inputs/outputs list into ``[{"name", "stoichiometry"}]``.

    Accepts either bare strings or ``{"name"/"compound"/..., "stoichiometry"}``
    dicts (models emit both), keeps a numeric coefficient when present, and drops
    empty/blank names. Canonicalization is left to synthesis
    (``_participants_from_field``), so names are passed through verbatim here.
    """
    out: List[Dict[str, Any]] = []
    if not isinstance(value, list):
        return out
    for item in value:
        name = ""
        stoich: Any = None
        if isinstance(item, str):
            name = item.strip()
        elif isinstance(item, dict):
            raw_name = (
                item.get("name")
                or item.get("compound")
                or item.get("metabolite")
                or item.get("entity")
                or item.get("species")
            )
            name = str(raw_name or "").strip()
            coeff = item.get("stoichiometry")
            if coeff is None:
                coeff = item.get("coefficient")
            if isinstance(coeff, (int, float)) and not isinstance(coeff, bool):
                stoich = float(coeff)
        if name:
            out.append({"name": name, "stoichiometry": stoich})
    return out


def _parse_reactions_json(raw: Any) -> List[Dict[str, Any]]:
    """Turn raw model output into a clean list of reaction dicts (never raises).

    Output shape per reaction (matching what ``synthesize`` consumes):
    ``{"name": str, "inputs": [{"name", "stoichiometry"}], "outputs": [...],
    "enzymes": [str], "reversible": bool}``. A reaction with neither inputs nor
    outputs is dropped.
    """
    data = _loads_lenient(raw)
    if not isinstance(data, dict):
        return []
    reactions = data.get("reactions")
    if not isinstance(reactions, list):
        return []
    out: List[Dict[str, Any]] = []
    for item in reactions:
        if not isinstance(item, dict):
            continue
        inputs = _clean_side(item.get("inputs"))
        outputs = _clean_side(item.get("outputs"))
        if not inputs and not outputs:
            continue
        enzymes = [
            e.strip()
            for e in (item.get("enzymes") or [])
            if isinstance(e, str) and e.strip()
        ]
        out.append(
            {
                "name": str(item.get("name") or "").strip(),
                "inputs": inputs,
                "outputs": outputs,
                "enzymes": enzymes,
                "reversible": bool(item.get("reversible")),
            }
        )
    return out


def extract_reactions_from_text(
    text: str, *, chat_fn: Optional[ChatFn] = None
) -> List[Dict[str, Any]]:
    """Extract the reactions a passage explicitly states, as clean reaction dicts.

    ``chat_fn(system, user) -> str`` is the model call; it defaults to the shared
    OpenAI-compatible client. Fails **closed**: an empty passage, any client error,
    or an unparseable response yields ``[]`` (never raises), so prose extraction
    can only add reactions to synthesis, never break it.
    """
    passage = str(text or "").strip()
    if not passage:
        return []
    fn = chat_fn if chat_fn is not None else _default_chat_fn
    try:
        raw = fn(_SYSTEM_PROMPT, passage)
    except Exception:  # noqa: BLE001 - no endpoint / model error -> no reactions
        return []
    return _parse_reactions_json(raw)


def make_prose_extractor(chat_fn: Optional[ChatFn] = None) -> Callable[[str], List[Dict[str, Any]]]:
    """Return a ``text -> [reaction dict]`` callable for ``synthesize_with_report``.

    Synthesis owns the per-chunk memoization and passage cap; this is only the
    text→reactions step. Pass a ``chat_fn`` to inject a stub (tests); omit it to
    use the shared client (production).
    """

    def _run(text: str) -> List[Dict[str, Any]]:
        return extract_reactions_from_text(text, chat_fn=chat_fn)

    return _run


__all__ = ["extract_reactions_from_text", "make_prose_extractor", "ChatFn"]
