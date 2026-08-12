from __future__ import annotations

import re
from typing import Any, Dict, Mapping, Optional, Set


PATHBANK_UNKNOWN_PROTEIN_ID = 9659
PATHBANK_UNKNOWN_PROTEIN_NAME = "Unknown"
PATHBANK_UNKNOWN_PROTEIN_UNIPROT = "Unknown"
PATHBANK_UNKNOWN_FALLBACK_RULE = "pathbank_unknown_protein_fallback"

#: ``mapping_meta.identity_status`` values. One authoritative vocabulary, so a
#: reader never has to infer "is this a real match?" from a rule name.
IDENTITY_VERIFIED = "verified"
IDENTITY_PLACEHOLDER = "placeholder"
IDENTITY_UNRESOLVED = "unresolved"

# ── verification status (D-003) ─────────────────────────────────────────────
#
# ``identity_status`` says what the row *is*. This second vocabulary says what
# the identity ladder was *able to establish*, and the two are not the same
# question: a row can be ``unresolved`` because its accession was refuted, or
# because nothing could be looked up. Collapsing those is the failure D-003
# forbids in terms -- "network failure is not evidence that an accession is
# false" -- so ``unavailable`` and ``not_evaluated`` are kept strictly distinct
# from ``rejected``, and no code path may map one onto another.

#: The ladder ran and every rung passed on affirmative evidence.
VERIFICATION_VERIFIED = "verified"
#: The ladder ran and a rung *refuted* the identity -- wrong species, a name
#: that cannot be this molecule, a score under the floor, an unbroken tie.
VERIFICATION_REJECTED = "rejected"
#: An evidence lookup was attempted and could not be completed: the network, the
#: database, or the upstream record was not there. NOT a refutation.
VERIFICATION_UNAVAILABLE = "unavailable"
#: No evidence lookup was attempted, or the ladder had nothing to judge on. The
#: default, and the state of every offline re-export. NOT a refutation.
VERIFICATION_NOT_EVALUATED = "not_evaluated"



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


def identity_status(row: Any) -> str:
    """The declared identity status of a protein / protein_complex row.

    Falls back to inferring it from the sentinel shape so a row written before
    ``identity_status`` existed still reports honestly.
    """

    if not isinstance(row, dict):
        return ""
    meta = _mapping(row.get("mapping_meta"))
    declared = _canonical(row.get("identity_status") or meta.get("identity_status")).casefold()
    if declared:
        return declared
    if is_pathbank_unknown_protein(row):
        return IDENTITY_PLACEHOLDER
    if meta.get("cross_species_placeholder") is True or meta.get("fallback_used") is True:
        return IDENTITY_PLACEHOLDER
    if has_protein_external_identity(row):
        return IDENTITY_VERIFIED
    return IDENTITY_UNRESOLVED


def is_placeholder_identity(row: Any) -> bool:
    """Whether the row admits it is a placeholder rather than a real mapping."""

    return identity_status(row) == IDENTITY_PLACEHOLDER


def unverified_identity_claim(row: Any) -> Dict[str, Any]:
    """The submitted-but-unverified accession a row is preserving, if any.

    D-003: an accession whose verification was ``unavailable`` or
    ``not_evaluated`` is "preserved as an unverified claim -- do not promote it,
    do not erase it". It is therefore *not* in ``mapped_ids`` (that would be
    promotion) and *not* only in ``rejected_mapped_ids`` (that would read as a
    refutation). It lives here, with the state that produced it.
    """

    if not isinstance(row, dict):
        return {}
    claim = _mapping(_mapping(row.get("mapping_meta")).get("unverified_identity_claim"))
    return dict(claim) if claim else {}


def placeholder_claims_real_identity(row: Any) -> str:
    """Return the rule id when a placeholder row is posing as a real mapping.

    A placeholder that ships a plausible UniProt accession, or that reports a
    ``matched`` resolution, is the one failure mode no other gate can see: every
    identity check passes and the pathway is wrong. Returns ``""`` for a
    correctly-formed placeholder.
    """

    if not isinstance(row, dict) or not is_placeholder_identity(row):
        return ""
    meta = _mapping(row.get("mapping_meta"))
    accession = _canonical(
        row.get("uniprot")
        or row.get("uniprot_id")
        or _mapping(row.get("mapped_ids")).get("uniprot")
    )
    if accession and accession.casefold() != PATHBANK_UNKNOWN_PROTEIN_UNIPROT.casefold():
        return "placeholder_ships_real_uniprot_accession"
    drugbank = _canonical(
        row.get("drugbank")
        or row.get("drugbank_id")
        or _mapping(row.get("mapped_ids")).get("drugbank")
    )
    if drugbank:
        return "placeholder_ships_real_drugbank_accession"
    resolution = _mapping(meta.get("resolution"))
    if _canonical(resolution.get("status")).casefold() == "matched":
        return "placeholder_reports_matched_resolution"
    if meta.get("fallback_used") is not True:
        return "placeholder_does_not_record_fallback_used"
    return ""


# ── compound-vs-protein name shape ──────────────────────────────────────────
#
# PROTEIN_LIKE_RE matches the substring "enzyme", which is inside "coenzyme":
# every "coenzyme A" name in run 2026-07-28_2122 was therefore read as a protein
# and filed in entities.proteins, where it can never acquire a UniProt ID. Eight
# of that night's 27 distinct post-gate issue codes are one of two cofactors
# misfiled this way. These two rules say what the regex cannot: a CoA- or
# ACP-bound acyl name is a metabolite, and the enzyme that acts on it is not.

def _moiety_tokens(value: Any) -> str:
    """Space-separated word tokens, punctuation-insensitive (``succinyl-CoA`` -> ``succinyl coa``)."""

    text = re.sub(r"[^a-z0-9]+", " ", _canonical(value).casefold())
    return re.sub(r"\s+", " ", text).strip()


#: Words that make a name the *catalyst* rather than the substrate. ``\w+ase``
#: covers dehydratase/ligase/synthase/...; the denylist keeps ordinary English
#: words that merely end in "ase" from disqualifying a real metabolite name.
_ENZYME_HEAD_NOUN_RE = re.compile(
    r"\b(?:[a-z]{3,}ase|synthase|carrier\s+protein|protein|complex|subunit|enzyme"
    r"|receptor|transporter|symporter|antiporter|permease|factor|domain)\b",
    flags=re.IGNORECASE,
)
_NON_ENZYME_ASE_WORDS = frozenset(
    {"base", "case", "phase", "increase", "decrease", "release", "disease", "purchase", "database"}
)
#: Words that name the ACP *protein* rather than a loaded acyl species -- but
#: only as the whole prefix. ``acyl-ACP`` is the generic carrier; ``trans-2-acyl-ACP``
#: is a specific metabolite, and the moiety qualifier in front is what says so.
_ACP_PROTEIN_STATE_PREFIXES = frozenset({"holo", "apo", "free", "acyl", "unloaded", "loaded"})


def _has_enzyme_head_noun(name: str) -> bool:
    for match in _ENZYME_HEAD_NOUN_RE.finditer(_canonical(name)):
        if match.group(0).strip().casefold() not in _NON_ENZYME_ASE_WORDS:
            return True
    return False


def compound_moiety_verdict(name: Any) -> str:
    """Rule id when ``name`` denotes a carrier-bound metabolite, else ``""``.

    ``coa_thioester`` covers free CoA and every acyl-CoA/acyl-coenzyme A;
    ``acp_thioester`` covers acyl-ACP species such as ``beta-hydroxyacyl-ACP``.
    Both defer to :func:`_has_enzyme_head_noun`, so ``succinyl-CoA synthetase``
    and ``beta-hydroxyacyl-ACP dehydratase`` stay proteins -- the distinction the
    regex could not make.
    """

    tokens = _moiety_tokens(name).split()
    if not tokens or _has_enzyme_head_noun(name):
        return ""
    if "coa" in tokens:
        return "coa_thioester"
    for index in range(len(tokens) - 1):
        if tokens[index] == "coenzyme" and tokens[index + 1] == "a":
            return "coa_thioester"
    if "acp" in tokens:
        index = tokens.index("acp")
        # Bare "ACP", and "holo-/apo-/acyl-ACP", are the carrier protein itself.
        # Anything with a moiety qualifier ahead of that -- "malonyl-ACP",
        # "trans-2-acyl-ACP" -- names the loaded species, which is a metabolite.
        state_prefix_only = index == 1 and tokens[0] in _ACP_PROTEIN_STATE_PREFIXES
        if index > 0 and not state_prefix_only:
            return "acp_thioester"
    return ""


def component_stoichiometry(component: Any) -> Optional[int]:
    """Return a complex component's explicit stoichiometry, or None when unstated.

    PathWhiz stores this on ``protein_complex_proteins`` as a nullable column
    validated with ``allow_nil: true``, and its PWML parser skips blank nodes,
    so an unstated coefficient is left blank rather than assumed. This is the
    single definition shared by mapping, IR construction, and the writer; they
    previously each had their own and disagreed.
    """

    if isinstance(component, str) or not isinstance(component, dict):
        return None
    for key in ("stoichiometry", "coefficient"):
        raw = component.get(key)
        if raw is None or isinstance(raw, bool):
            continue
        try:
            parsed = int(float(str(raw).strip()))
        except (TypeError, ValueError):
            continue
        if parsed >= 1:
            return parsed
    return None


def component_has_explicit_stoichiometry(component: Any) -> bool:
    """Whether the source stated a usable stoichiometry for this component."""

    return component_stoichiometry(component) is not None


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
    lenient_names: bool = False,
    compound_names: Optional[Set[str]] = None,
) -> Dict[str, str]:
    """Classify an entity for compound, protein, or complex ID mapping.

    ``lenient_names`` is the research-mode switch. Normally a ``:`` in a name is
    read as PathWhiz complex syntax (``receptor:kinase``), which is right for
    curated input but wrong for a novel pathway, where a colon is just a
    character in a biological name. With ``lenient_names=True`` an unhinted name
    is never re-routed to the complex mapper on the strength of a colon alone --
    it keeps whatever route its own wording implies. Explicit type hints still
    win, and the biochemical-colon rules (``18:1-CoA``) are unaffected because
    they already route away from complex.

    ``compound_names`` is the payload's own compound registry. A name the
    payload already declares as a compound, or one whose shape says it is a
    carrier-bound metabolite (:func:`compound_moiety_verdict`), is never handed
    to the protein mapper on the strength of the name regex alone -- that is the
    ``coenzyme A`` -> "contains 'enzyme'" -> protein defect. An explicit type
    hint and an explicit protein declaration both still outrank it, so nothing
    that was declared a protein changes route. ``blocked_rule`` records the
    block so a caller can count it instead of the block being invisible.
    """

    name = _canonical(entity_name)
    hint = _canonical(entity_type_hint).lower()
    norm = _normalize(name)
    protein_like_set = {value for value in (protein_like_names or set()) if value}
    if hint in {"complex", "protein_complex"}:
        return {"route": "complex", "reason": "complex_entity"}
    if hint in {"protein", "enzyme", "modifier"}:
        return {"route": "protein", "reason": "type_hint"}
    if not lenient_names and ":" in name and _is_explicit_complex_colon_name(name, protein_like_set):
        return {"route": "complex", "reason": "complex_entity"}
    if norm in protein_like_set:
        return {"route": "protein", "reason": "known_protein_like"}
    blocked = compound_name_block_rule(name, compound_names=compound_names)
    if blocked:
        return {
            "route": "compound",
            "reason": "declared_or_shaped_as_compound",
            "blocked_rule": blocked,
        }
    if _looks_protein_like_name(name):
        return {"route": "protein", "reason": "name_pattern"}
    return {"route": "compound", "reason": "default_compound_route"}


def compound_name_block_rule(
    name: Any,
    *,
    compound_names: Optional[Set[str]] = None,
) -> str:
    """Rule id when ``name`` must be read as a compound, else ``""``.

    Two independent sources, in order: the payload's own compound registry
    (authoritative -- the extractor said so) and the carrier-moiety name shape
    (the backstop for a participant that never got a registry row).
    """

    norm = _normalize(name)
    if not norm:
        return ""
    if norm in {value for value in (compound_names or set()) if value}:
        return "compound_registry"
    return compound_moiety_verdict(name)
