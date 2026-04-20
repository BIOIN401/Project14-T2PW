from __future__ import annotations

import copy
import json
import time
from typing import Any
from urllib.parse import quote_plus
from xml.etree import ElementTree

import requests

from llm_client import _client, _model
from stoich_classifier import classify_reaction
from stoich_templates import REACTION_TEMPLATES

_KEGG_CACHE: dict = {}
_CHEBI_CACHE: dict = {}

_SESSION = requests.Session()
_SESSION.headers.update({"User-Agent": "Project14-T2PW-StoichAgent/1.0"})


def _http_get(url: str, *, params: dict | None = None, retries: int = 3, backoff: float = 0.6) -> requests.Response:
    last_exc: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            resp = _SESSION.get(url, params=params, timeout=15)
            if resp.status_code >= 500:
                raise requests.HTTPError(f"Server error {resp.status_code}")
            return resp
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if attempt < retries:
                time.sleep(backoff * attempt)
    raise RuntimeError(f"HTTP request failed after retries: {url}; last error: {last_exc}")


def kegg_reaction_search(query: str) -> list[dict]:
    cache_key = f"search:{query}"
    if cache_key in _KEGG_CACHE:
        return _KEGG_CACHE[cache_key]

    url = f"https://rest.kegg.jp/find/reaction/{quote_plus(query)}"
    try:
        resp = _http_get(url)
    except RuntimeError:
        return []
    if resp.status_code != 200 or not resp.text.strip():
        return []

    results: list[dict] = []
    for line in resp.text.strip().splitlines():
        if "\t" not in line:
            continue
        reaction_id, description = line.split("\t", 1)
        # strip "rn:" prefix if present
        reaction_id = reaction_id.strip().removeprefix("rn:")
        results.append({"id": reaction_id, "description": description.strip()})
        if len(results) >= 5:
            break

    _KEGG_CACHE[cache_key] = results
    return results


def kegg_reaction_get(reaction_id: str) -> dict:
    cache_key = f"get:{reaction_id}"
    if cache_key in _KEGG_CACHE:
        return _KEGG_CACHE[cache_key]

    url = f"https://rest.kegg.jp/get/rn:{reaction_id}"
    try:
        resp = _http_get(url)
    except RuntimeError:
        return {}
    if resp.status_code != 200 or not resp.text.strip():
        return {}

    equation_raw = ""
    substrates: list[str] = []
    products: list[str] = []

    for line in resp.text.splitlines():
        if line.startswith("EQUATION"):
            equation_raw = line[len("EQUATION"):].strip()
            break

    if equation_raw:
        # Format: "A + B <=> C + D" — split on <=> or =>
        sep = "<=>" if "<=>" in equation_raw else "=>"
        if sep in equation_raw:
            left, right = equation_raw.split(sep, 1)
            substrates = _parse_kegg_equation_side(left)
            products = _parse_kegg_equation_side(right)

    result = {"equation_raw": equation_raw, "substrates": substrates, "products": products}
    _KEGG_CACHE[cache_key] = result
    return result


def _parse_kegg_equation_side(side: str) -> list[str]:
    """Extract compound names/IDs from one side of a KEGG equation string."""
    import re
    # Strip stoichiometric coefficients (leading digits) and split on " + "
    parts = [p.strip() for p in side.split("+")]
    compounds: list[str] = []
    for part in parts:
        # Remove leading coefficient like "2 " or "n "
        cleaned = re.sub(r"^\d+\s+", "", part).strip()
        cleaned = re.sub(r"^[a-z]\s+", "", cleaned).strip()
        if cleaned:
            compounds.append(cleaned)
    return compounds


def chebi_verify(compound_name: str) -> dict:
    cache_key = compound_name.casefold()
    if cache_key in _CHEBI_CACHE:
        return _CHEBI_CACHE[cache_key]

    url = "https://www.ebi.ac.uk/webservices/chebi/2.0/test/getLiteEntity"
    params = {
        "search": compound_name,
        "searchCategory": "ALL NAMES",
        "maximumResults": 5,
        "stars": "ALL",
    }
    try:
        resp = _http_get(url, params=params)
    except RuntimeError:
        result = {"found": False, "chebi_id": None, "canonical_name": None}
        _CHEBI_CACHE[cache_key] = result
        return result

    if resp.status_code != 200:
        result = {"found": False, "chebi_id": None, "canonical_name": None}
        _CHEBI_CACHE[cache_key] = result
        return result

    try:
        root = ElementTree.fromstring(resp.text)
    except ElementTree.ParseError:
        result = {"found": False, "chebi_id": None, "canonical_name": None}
        _CHEBI_CACHE[cache_key] = result
        return result

    best: dict[str, Any] | None = None
    query_cf = compound_name.casefold()
    for node in root.iter():
        if not node.tag.lower().endswith("liteentity"):
            continue
        chebi_id = ""
        chebi_name = ""
        for child in node:
            tag = child.tag.split("}")[-1]
            text = (child.text or "").strip()
            if tag == "chebiId":
                chebi_id = f"CHEBI:{text}" if not text.upper().startswith("CHEBI:") else text
            elif tag == "chebiAsciiName":
                chebi_name = text
        if not chebi_id:
            continue
        # prefer exact case-insensitive match
        if best is None or chebi_name.casefold() == query_cf:
            best = {"chebi_id": chebi_id, "canonical_name": chebi_name}
        if chebi_name.casefold() == query_cf:
            break

    if best:
        result = {"found": True, "chebi_id": best["chebi_id"], "canonical_name": best["canonical_name"]}
    else:
        result = {"found": False, "chebi_id": None, "canonical_name": None}

    _CHEBI_CACHE[cache_key] = result
    return result


def get_template_suggestion(reaction_class: str) -> dict:
    template = REACTION_TEMPLATES.get(reaction_class)
    if template is None:
        return {"required_inputs": [], "required_outputs": [], "note": f"Unknown reaction class: {reaction_class!r}"}
    return {
        "required_inputs": list(template["required_inputs"]),
        "required_outputs": list(template["required_outputs"]),
        "note": template["note"],
    }


def apply_stoich_fix(
    pathway_json: dict,
    reaction_name: str,
    add_inputs: list[str],
    add_outputs: list[str],
) -> dict:
    added_inputs: list[str] = []
    added_outputs: list[str] = []
    skipped: list[str] = []

    reactions: list[dict] = pathway_json.get("reactions", [])
    target_reaction: dict | None = None
    for rxn in reactions:
        if (rxn.get("name") or "").casefold() == reaction_name.casefold():
            target_reaction = rxn
            break

    if target_reaction is None:
        return {"added_inputs": [], "added_outputs": [], "skipped": list(add_inputs) + list(add_outputs)}

    compounds_list: list[dict] = pathway_json.setdefault("entities", {}).setdefault("compounds", [])
    existing_names_cf = {(c.get("name") or "").casefold() for c in compounds_list}

    def _ensure_compound(name: str) -> None:
        if name.casefold() not in existing_names_cf:
            compounds_list.append({"name": name, "class": "compound", "mapped_ids": {}})
            existing_names_cf.add(name.casefold())

    rxn_inputs: list[dict] = target_reaction.setdefault("inputs", [])
    rxn_outputs: list[dict] = target_reaction.setdefault("outputs", [])

    existing_input_names_cf = {(x.get("name") or "").casefold() for x in rxn_inputs}
    existing_output_names_cf = {(x.get("name") or "").casefold() for x in rxn_outputs}

    for compound in add_inputs:
        if compound.casefold() in existing_input_names_cf:
            skipped.append(compound)
        else:
            _ensure_compound(compound)
            rxn_inputs.append({"name": compound})
            existing_input_names_cf.add(compound.casefold())
            added_inputs.append(compound)

    for compound in add_outputs:
        if compound.casefold() in existing_output_names_cf:
            skipped.append(compound)
        else:
            _ensure_compound(compound)
            rxn_outputs.append({"name": compound})
            existing_output_names_cf.add(compound.casefold())
            added_outputs.append(compound)

    return {"added_inputs": added_inputs, "added_outputs": added_outputs, "skipped": skipped}


# ---------------------------------------------------------------------------
# OpenAI tool schemas for all 5 callable functions
# ---------------------------------------------------------------------------
STOICH_TOOLS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "kegg_reaction_search",
            "description": "Search KEGG for reactions matching a query string. Returns up to 5 results with reaction IDs and descriptions.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query (reaction name, enzyme name, etc.)"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "kegg_reaction_get",
            "description": "Fetch full details of a KEGG reaction by ID, including substrate and product compound IDs/names.",
            "parameters": {
                "type": "object",
                "properties": {
                    "reaction_id": {"type": "string", "description": "KEGG reaction ID, e.g. 'R00200'"},
                },
                "required": ["reaction_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "chebi_verify",
            "description": "Verify that a compound name exists in ChEBI and return its canonical name and ChEBI ID.",
            "parameters": {
                "type": "object",
                "properties": {
                    "compound_name": {"type": "string", "description": "Compound name to verify in ChEBI"},
                },
                "required": ["compound_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_template_suggestion",
            "description": "Get the required inputs and outputs for a reaction class from the built-in template library.",
            "parameters": {
                "type": "object",
                "properties": {
                    "reaction_class": {"type": "string", "description": "Reaction class name, e.g. 'phosphorylation'"},
                },
                "required": ["reaction_class"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "apply_stoich_fix",
            "description": "Apply stoichiometry corrections to a reaction, adding missing input or output compounds.",
            "parameters": {
                "type": "object",
                "properties": {
                    "reaction_name": {"type": "string", "description": "Name of the reaction to fix"},
                    "add_inputs": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Compound names to add as inputs",
                    },
                    "add_outputs": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Compound names to add as outputs",
                    },
                },
                "required": ["reaction_name", "add_inputs", "add_outputs"],
            },
        },
    },
]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _dispatch_tool(fn_name: str, fn_args: dict, working: dict) -> Any:
    if fn_name == "kegg_reaction_search":
        return kegg_reaction_search(fn_args.get("query", ""))
    if fn_name == "kegg_reaction_get":
        return kegg_reaction_get(fn_args.get("reaction_id", ""))
    if fn_name == "chebi_verify":
        return chebi_verify(fn_args.get("compound_name", ""))
    if fn_name == "get_template_suggestion":
        return get_template_suggestion(fn_args.get("reaction_class", ""))
    if fn_name == "apply_stoich_fix":
        # Always operate on the live working copy, not whatever the LLM serialised
        return apply_stoich_fix(
            working,
            fn_args.get("reaction_name", ""),
            fn_args.get("add_inputs", []),
            fn_args.get("add_outputs", []),
        )
    return {"error": f"Unknown tool: {fn_name}"}


def _parse_json_from_text(text: str) -> dict:
    """Extract the first JSON object from a text string."""
    start = text.find("{")
    end = text.rfind("}") + 1
    if start == -1 or end == 0:
        return {}
    try:
        return json.loads(text[start:end])
    except json.JSONDecodeError:
        return {}


def _parse_verdict(content: str) -> dict:
    data = _parse_json_from_text(content)
    verdict = data.get("verdict", "uncertain")
    if verdict not in ("add", "skip", "uncertain"):
        verdict = "uncertain"
    return {
        "verdict": verdict,
        "add_inputs": data.get("add_inputs", []),
        "add_outputs": data.get("add_outputs", []),
        "reasoning": data.get("reasoning", ""),
        "sources_used": data.get("sources_used", []),
    }


def _remove_addition(working: dict, reaction_name: str, compound: str, direction: str) -> None:
    side_key = "inputs" if direction == "input" else "outputs"
    for rxn in working.get("reactions", []):
        if (rxn.get("name") or "").casefold() == reaction_name.casefold():
            rxn[side_key] = [
                item for item in rxn.get(side_key, [])
                if (item.get("name") if isinstance(item, dict) else str(item)).casefold()
                != compound.casefold()
            ]
            break


# ---------------------------------------------------------------------------
# Main agentic function
# ---------------------------------------------------------------------------

def run_stoich_agent(
    mapped_json: dict,
    *,
    max_tool_rounds: int = 5,
    timeout_seconds: int = 120,
    temperature: float = 0.1,
) -> tuple[dict, list[dict]]:
    working = copy.deepcopy(mapped_json)
    audit_log: list[dict] = []
    deadline = time.time() + timeout_seconds

    for reaction in working.get("reactions", []):
        if time.time() >= deadline:
            break

        reaction_name = reaction.get("name", "unknown")
        classification = classify_reaction(reaction)
        rxn_class = classification.get("class")
        confidence = classification.get("confidence", "low")

        template_suggestion: dict = {}
        if rxn_class:
            template_suggestion = get_template_suggestion(rxn_class)

        system_prompt = (
            "You are a biochemistry expert auditing pathway stoichiometry. "
            "Use the provided tools to research whether this reaction is missing required "
            "inputs or outputs, then emit a final JSON verdict as your last message:\n"
            '{"verdict": "add"|"skip"|"uncertain", "add_inputs": [...], "add_outputs": [...], '
            '"reasoning": "...", "sources_used": ["kegg"|"chebi"|"template"|"prior_knowledge"]}\n'
            '"add" means corrections are needed. "skip" means the reaction is complete. '
            '"uncertain" means insufficient evidence.'
        )
        user_content = (
            f"Reaction record:\n{json.dumps(reaction, indent=2)}\n\n"
            f"Classification: {json.dumps(classification, indent=2)}\n\n"
            f"Template suggestion: {json.dumps(template_suggestion, indent=2)}\n\n"
            "Research this reaction and provide your JSON verdict."
        )

        messages: list[dict] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]

        tool_calls_made = 0
        final_content = ""

        for _ in range(max_tool_rounds):
            if time.time() >= deadline:
                break

            response = _client.chat.completions.create(
                model=_model,
                messages=messages,
                tools=STOICH_TOOLS,
                temperature=temperature,
            )

            choice = response.choices[0]
            assistant_dict: dict = {"role": "assistant", "content": choice.message.content}
            if choice.message.tool_calls:
                assistant_dict["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                    }
                    for tc in choice.message.tool_calls
                ]
            messages.append(assistant_dict)

            if choice.finish_reason == "stop" or not choice.message.tool_calls:
                final_content = choice.message.content or ""
                break

            for tc in choice.message.tool_calls:
                tool_calls_made += 1
                try:
                    fn_args = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    fn_args = {}
                tool_result = _dispatch_tool(tc.function.name, fn_args, working)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": json.dumps(tool_result),
                })

        verdict_data = _parse_verdict(final_content)
        llm_verdict = verdict_data["verdict"]
        add_inputs = verdict_data["add_inputs"]
        add_outputs = verdict_data["add_outputs"]

        if llm_verdict == "add":
            apply_stoich_fix(working, reaction_name, add_inputs, add_outputs)

        audit_log.append({
            "reaction_name": reaction_name,
            "class": rxn_class,
            "confidence": confidence,
            "template_suggestion": template_suggestion,
            "llm_verdict": llm_verdict,
            "add_inputs": add_inputs,
            "add_outputs": add_outputs,
            "reasoning": verdict_data["reasoning"],
            "sources_used": verdict_data["sources_used"],
            "tool_calls_made": tool_calls_made,
            "audit_verdict": "not_audited",
        })

    # ------------------------------------------------------------------
    # Audit pass: verify every compound that was added
    # ------------------------------------------------------------------
    for entry in audit_log:
        if entry["llm_verdict"] != "add":
            continue
        if time.time() >= deadline:
            break

        reaction_name = entry["reaction_name"]
        reversed_inputs: list[str] = []
        reversed_outputs: list[str] = []

        for compound in list(entry["add_inputs"]):
            if time.time() >= deadline:
                break
            audit_response = _client.chat.completions.create(
                model=_model,
                messages=[
                    {"role": "system", "content": "You are a biochemistry expert."},
                    {
                        "role": "user",
                        "content": (
                            f"Is it biochemically correct that {compound} is consumed in {reaction_name}? "
                            'Reply JSON: {"verdict": "correct"|"incorrect"|"uncertain", "reasoning": "..."}'
                        ),
                    },
                ],
                temperature=0.0,
                max_tokens=300,
            )
            av = _parse_json_from_text(audit_response.choices[0].message.content or "").get("verdict", "uncertain")
            if av == "incorrect":
                _remove_addition(working, reaction_name, compound, "input")
                reversed_inputs.append(compound)

        for compound in list(entry["add_outputs"]):
            if time.time() >= deadline:
                break
            audit_response = _client.chat.completions.create(
                model=_model,
                messages=[
                    {"role": "system", "content": "You are a biochemistry expert."},
                    {
                        "role": "user",
                        "content": (
                            f"Is it biochemically correct that {compound} is produced in {reaction_name}? "
                            'Reply JSON: {"verdict": "correct"|"incorrect"|"uncertain", "reasoning": "..."}'
                        ),
                    },
                ],
                temperature=0.0,
                max_tokens=300,
            )
            av = _parse_json_from_text(audit_response.choices[0].message.content or "").get("verdict", "uncertain")
            if av == "incorrect":
                _remove_addition(working, reaction_name, compound, "output")
                reversed_outputs.append(compound)

        if reversed_inputs or reversed_outputs:
            entry["audit_verdict"] = "reversed"
            entry["reversed_inputs"] = reversed_inputs
            entry["reversed_outputs"] = reversed_outputs
        else:
            entry["audit_verdict"] = "confirmed"

    return working, audit_log
