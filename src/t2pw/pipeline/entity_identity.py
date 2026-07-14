from __future__ import annotations

import re
from typing import Any, Dict, Mapping, Optional, Set


PATHBANK_UNKNOWN_PROTEIN_ID = 9659
PATHBANK_UNKNOWN_PROTEIN_NAME = "Unknown"
PATHBANK_UNKNOWN_PROTEIN_UNIPROT = "Unknown"
PATHBANK_UNKNOWN_FALLBACK_RULE = "pathbank_unknown_protein_fallback"


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, dict) else {}


def _canonical(value: Any) -> str:
    text = str(value or "").strip()
    text = (
        text.replace("\u2013", "-")
        .replace("\u2014", "-")
        .replace("\u2212", "-")
        .replace("\u00a0", " ")
    )
    return re.sub(r"\s+", " ", text).strip()


def _normalize(value: Any) -> str:
    lowered = re.sub(r"\s+", " ", _canonical(value).casefold())
    return re.sub(r"[^a-z0-9 ]+", "", lowered)


def protein_external_identity(row: Any) -> str:
    """Return a protein's first usable UniProt or DrugBank identifier."""

    if not isinstance(row, dict):
        return ""
    keys = ("uniprot", "uniprot_id", "uniprot-id", "drugbank", "drugbank_id", "drugbank-id")
    for container in (
        row,
        _mapping(row.get("mapped_ids")),
        _mapping(row.get("ids")),
        _mapping(row.get("mapping_meta")),
    ):
        for key in keys:
            value = _canonical(container.get(key))
            if value:
                return value
    return ""


def has_protein_external_identity(row: Any) -> bool:
    """Whether a row has the UniProt/DrugBank identity required by PathWhiz."""

    return bool(protein_external_identity(row))


def is_pathbank_unknown_protein(row: Any) -> bool:
    """Recognize the explicit PathBank Unknown-protein fallback sentinel."""

    if not isinstance(row, dict):
        return False
    mapped_ids = _mapping(row.get("mapped_ids"))
    meta = _mapping(row.get("mapping_meta"))
    try:
        pathbank_id = int(
            row.get("pathbank_protein_id")
            or mapped_ids.get("pathbank_protein_id")
            or meta.get("pathbank_protein_id")
            or 0
        )
    except (TypeError, ValueError):
        pathbank_id = 0
    uniprot = _canonical(
        row.get("uniprot")
        or row.get("uniprot_id")
        or mapped_ids.get("uniprot")
    )
    return bool(
        pathbank_id == PATHBANK_UNKNOWN_PROTEIN_ID
        and _normalize(row.get("name")) == _normalize(PATHBANK_UNKNOWN_PROTEIN_NAME)
        and uniprot.casefold() == PATHBANK_UNKNOWN_PROTEIN_UNIPROT.casefold()
        and (
            meta.get("chosen_rule") == PATHBANK_UNKNOWN_FALLBACK_RULE
            or meta.get("cross_species_placeholder") is True
        )
    )


def protein_species_context(row: Any) -> Any:
    """Return the first species value used by the mapping and PWML gates."""

    if not isinstance(row, dict):
        return ""
    species_ref = _mapping(row.get("species_ref"))
    mapping_meta = _mapping(row.get("mapping_meta"))
    return (
        row.get("species")
        or row.get("organism")
        or row.get("taxonomy_id")
        or row.get("species_id")
        or row.get("pathbank_species_id")
        or species_ref.get("pathbank_species_id")
        or species_ref.get("name")
        or mapping_meta.get("species")
        or mapping_meta.get("species_id")
    )


def is_generated_complex_wrapper(row: Any) -> bool:
    """Recognize generated wrappers, including narrow legacy-cache markers."""

    if not isinstance(row, dict):
        return False
    if row.get("generated") is True:
        return True

    # Legacy fallbacks are retained only for cache rows created before the
    # authoritative ``generated`` flag was consistently persisted.
    meta = _mapping(row.get("mapping_meta"))
    resolution = _mapping(meta.get("resolution"))
    reason = _canonical(row.get("generation_reason") or meta.get("generation_reason")).casefold()
    chosen_rule = _canonical(row.get("chosen_rule") or meta.get("chosen_rule")).casefold()
    order_step = _canonical(resolution.get("order_step")).casefold()
    return bool(
        reason == "single_protein_pathwhiz_wrapper"
        or chosen_rule == "novel_enzyme_single_component_complex"
        or order_step == "novel_enzyme_single_component_complex"
    )


def _looks_protein_like_name(name: str) -> bool:
    return bool(
        _normalize(name)
        and re.search(
            r"(protein|globulin|peroxidase|deiodinase|kinase|phosphatase|ligase|atpase|receptor|transporter|enzyme)",
            _normalize(name),
            flags=re.IGNORECASE,
        )
    )


def _is_biochemical_colon_name(name: str) -> bool:
    text = _canonical(name)
    if ":" not in text:
        return False
    if re.search(r"\(\s*\d+\s*:\s*\d+\s*\)", text):
        return True
    if re.search(r"\b[A-Za-z][A-Za-z0-9]*-\d+\s*:\s*\d+-CoA\b", text, flags=re.IGNORECASE):
        return True
    if re.search(r":\s*CoA\b", text, flags=re.IGNORECASE):
        return True
    return bool(re.search(r"(?<![A-Za-z0-9])\d+\s*:\s*\d+(?![A-Za-z0-9])", text))


def _is_explicit_complex_colon_name(name: str, protein_like_names: Set[str]) -> bool:
    text = _canonical(name)
    if ":" not in text or _is_biochemical_colon_name(text):
        return False
    parts = [part.strip() for part in text.split(":") if part.strip()]
    if len(parts) < 2:
        return False
    return bool(
        re.search(r"\bcomplex\b", text, flags=re.IGNORECASE)
        or any(_looks_protein_like_name(part) for part in parts)
        or any(_normalize(part) in protein_like_names for part in parts)
    )


def route_entity_for_mapping(
    entity_name: str,
    entity_type_hint: str,
    *,
    protein_like_names: Optional[Set[str]] = None,
) -> Dict[str, str]:
    """Classify an entity for compound, protein, or complex ID mapping."""

    name = _canonical(entity_name)
    hint = _canonical(entity_type_hint).lower()
    norm = _normalize(name)
    protein_like_set = {value for value in (protein_like_names or set()) if value}
    if hint in {"complex", "protein_complex"}:
        return {"route": "complex", "reason": "complex_entity"}
    if hint in {"protein", "enzyme", "modifier"}:
        return {"route": "protein", "reason": "type_hint"}
    if ":" in name and _is_explicit_complex_colon_name(name, protein_like_set):
        return {"route": "complex", "reason": "complex_entity"}
    if norm in protein_like_set:
        return {"route": "protein", "reason": "known_protein_like"}
    if _looks_protein_like_name(name):
        return {"route": "protein", "reason": "name_pattern"}
    return {"route": "compound", "reason": "default_compound_route"}
