from __future__ import annotations

import time
from typing import Any
from urllib.parse import quote_plus
from xml.etree import ElementTree

import requests

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
