from __future__ import annotations

import logging
import re
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from t2pw.pipeline.entity_identity import (
    component_stoichiometry,
    is_generated_complex_wrapper as _is_generated_complex_row,
    protein_external_identity as _protein_external_id,
    protein_species_context as _entity_species_context,
)
from t2pw.pwml.compound_resolution import _compound_external_ids, _resolve_compound_rows
from t2pw.pwml.compound_templates import select_compound_template_id
from t2pw.pwml.db_resolver import PathWhizCompoundResolver, apply_compound_db_resolution, normalize_chebi_id
from t2pw.pwml.name_index import PathwhizNameIndex, default_name_index

logger = logging.getLogger(__name__)

# Sentinel so callers can pass ``name_index=None`` to explicitly disable the
# offline canonicalization, distinct from "not provided -> load the default".
_NAME_INDEX_UNSET = object()

#: The field a compound row carries once compound resolution has ruled on it.
#: Written on **every** row by the resolution machinery -- ``matched``,
#: ``unmatched``, ``legacy_id_unverified``, ``identity_refused_review_required``
#: or ``matched_offline_name_index`` -- so its presence is the verdict and its
#: absence means no resolution ever ran on the row.
COMPOUND_RESOLUTION_VERDICT_FIELD = "db_status"

#: Where the pre-freeze sequence records, **on the payload**, whether a live
#: PathBank resolution DB was actually consulted before the freeze.
#:
#: ``_emit_canonicalization_preflight`` decides whether to warn from
#: ``report["db_resolution"]["available"]``, and D-032 clause 6 (LOCKED) rules
#: ``preflight`` product-visible export content rather than a diagnostic. The
#: value is produced at ``compound_resolution.py:485``, from
#: ``db_resolver.available()`` after a ``None`` resolver has been replaced by
#: ``PathBankDbResolver.from_env()`` -- so reproducing it here would be the
#: exporter probing an external service after the freeze, which D-015 forbids
#: outright. It is **carried**, never inferred.
#:
#: Nor is it inferable from the rows. On an all-legacy population every row
#: carries a ``pathbank_compound_id``, so ``compound_resolution.py:504`` takes
#: the legacy branch and ``continue``s before the resolver is consulted: the
#: resolved rows are byte-identical while ``available`` differs.
#:
#: The payload is the carrier for the same reason
#: :data:`SPECIES_CANONICALIZATION_FIELD` is -- it is the one object every
#: export entry point hands to :func:`build_pwml_ir`, and it survives the
#: freeze, the JSON round trip and ``deepcopy`` intact. It is provenance, not
#: biology: the name is not in ``canonical_hash.GRAPH_FIELDS`` or
#: ``GRAPH_SECTIONS``, so the graph hash never sees it, and nothing projects it
#: into the IR, so the emitted PWML is unchanged.
PREFREEZE_DB_RESOLUTION_FIELD = "prefreeze_db_resolution"


class UnresolvedCompoundRowError(RuntimeError):
    """A compound row reached the exporter carrying no resolution verdict.

    D-015 (LOCKED) puts compound identity resolution **before** the canonical
    freeze and ``PRODUCT_CONTRACT`` §5 forbids an exporter to "add, remove,
    resolve or reinterpret biological content after the canonical graph is
    frozen". So the exporter refuses such a row loudly rather than resolving it
    late: no defaulted verdict, no warn-and-continue, no silent drop.
    """

    code = "PWML_IR_COMPOUND_VERDICT_MISSING"

    def __init__(self, rows: Sequence[Dict[str, Any]], *, pointer_prefix: str) -> None:
        self.rows = [dict(row) for row in rows]
        self.pointer_prefix = pointer_prefix
        self.names = [_canonical(row.get("name")) for row in self.rows]
        self.keys = [str(row.get("key") or "") for row in self.rows]
        located = ", ".join(
            f"{name or '<unnamed>'} (key={key or '<none>'})"
            for name, key in zip(self.names, self.keys)
        )
        super().__init__(
            f"{self.code}: {len(self.rows)} compound row(s) at {pointer_prefix} carry no "
            f"'{COMPOUND_RESOLUTION_VERDICT_FIELD}' resolution verdict, so compound "
            f"resolution never ran on them before the canonical payload was frozen. The "
            f"PWML exporter does not resolve identity after the freeze (D-015, "
            f"PRODUCT_CONTRACT section 5). Run t2pw.pwml.prefreeze_resolution."
            f"run_prefreeze_resolution on the payload first. Offending rows: {located}"
        )


class DuplicateNamedRowError(RuntimeError):
    """Two rows in one bucket collapse onto a single exporter name key.

    Modelled on :class:`UnresolvedCompoundRowError` above and raised for the same
    reason: ``build_pwml_ir`` runs **after** the canonical freeze at every entry
    point that builds an IR, and ``PRODUCT_CONTRACT`` section 5 with permanent
    merge rule 8 forbid the exporter to "add, remove, resolve or reinterpret
    biological content" there. Dropping the second row is exactly that.

    It is **not** a dangling reference some later gate would catch:
    :func:`_dedupe_named_rows` groups on :func:`_norm`, and ``entity_by_name`` /
    ``component_by_name`` are keyed on the **same** ``_norm``, so
    ``resolve_entity`` finds the *survivor* and ``unresolved_entity_reference``
    never fires. The reference silently repoints to a biologically different row
    -- measured on a real leg (F-039): reaction 9 ``"lipid IV A -> lipid A
    precursor"`` was frozen consuming PathBank 40738 / ChEBI 60365 and exported
    consuming PathBank 40982 / ChEBI 58603, with one warning and no error.

    A blocking report issue alone would not do: ``_add_issue(report, "error", …)``
    sets ``report["ok"] = False`` but does **not** stop IR construction, so the row
    would still be dropped and an invalid IR still returned. Only raising fails
    before an invalid IR can be emitted.

    This refuses; it never repairs. Consolidating the rows would be post-freeze
    biological repair of a different kind: D-035 is unamended and D-036 defers the
    consolidation engine.
    """

    code = "PWML_IR_DUPLICATE_NAMED_ROW"

    def __init__(
        self,
        *,
        norm_key: str,
        pointer_prefix: str,
        rows: Sequence[Dict[str, Any]],
    ) -> None:
        #: Row *descriptors* -- ``name`` / ``key`` / ``pointer`` -- not the raw
        #: payload rows. The same objects are attached to the structured
        #: ``duplicate_named_record`` report issue, so the machine-readable
        #: account of the collision is identical whether it is read off the
        #: report or off the exception.
        self.rows = [dict(row) for row in rows]
        self.norm_key = norm_key
        self.pointer_prefix = pointer_prefix
        self.names = [str(row.get("name") or "") for row in self.rows]
        self.keys = [str(row.get("key") or "") for row in self.rows]
        self.pointers = [str(row.get("pointer") or "") for row in self.rows]
        located = " and ".join(
            f"'{name or '<unnamed>'}' (key={key or '<unassigned>'}, pointer={pointer})"
            for name, key, pointer in zip(self.names, self.keys, self.pointers)
        )
        super().__init__(
            f"{self.code}: {len(self.rows)} rows at {pointer_prefix} share the PWML "
            f"exporter name key '{norm_key}': {located}. The PWML exporter runs after "
            f"the canonical freeze, so it may not choose between them, drop one, or "
            f"merge them (PRODUCT_CONTRACT section 5, merge rule 8). Dropping one "
            f"silently repoints every reference to the survivor, which carries "
            f"different identifiers. Resolve the collision in the payload before the "
            f"freeze -- see t2pw.pwml.prefreeze_resolution.run_prefreeze_resolution."
        )


ENTITY_BUCKETS = {
    "compound": "compounds",
    "protein": "proteins",
    "nucleic_acid": "nucleic_acids",
    "element_collection": "element_collections",
    "protein_complex": "protein_complexes",
    "bound": "bounds",
}

PROCESS_BUCKETS = {
    "reaction": "reactions",
    "reaction_coupled_transport": "reaction_coupled_transports",
    "transport": "transports",
    "interaction": "interactions",
    "sub_pathway": "sub_pathways",
}


def _safe_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _safe_list(value: Any) -> List[Any]:
    return value if isinstance(value, list) else []


def _canonical(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip())


def _norm(value: Any) -> str:
    text = _canonical(value).casefold()
    return re.sub(r"[^a-z0-9:+ ]+", " ", text).strip()


def _to_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    try:
        text = str(value).strip()
        if not text:
            return None
        return int(float(text))
    except (TypeError, ValueError):
        return None


def _first_nonempty(row: Dict[str, Any], keys: Sequence[str]) -> Any:
    for key in keys:
        if key in row and row.get(key) not in (None, ""):
            return row.get(key)
    meta = _safe_dict(row.get("mapping_meta"))
    for key in keys:
        if key in meta and meta.get(key) not in (None, ""):
            return meta.get(key)
    mapped = _safe_dict(row.get("mapped_ids"))
    for key in keys:
        if key in mapped and mapped.get(key) not in (None, ""):
            return mapped.get(key)
    candidates = _safe_list(meta.get("candidates"))
    if candidates and isinstance(candidates[0], dict):
        top = candidates[0]
        for key in keys:
            if key in top and top.get(key) not in (None, ""):
                return top.get(key)
    return None


def _db_id(row: Dict[str, Any], keys: Sequence[str]) -> Optional[int]:
    return _to_int(_first_nonempty(row, keys))


SPECIES_CREATE_DEFAULTS: Dict[str, Dict[str, Any]] = {
    "narcissus sp aff pseudonarcissus": {
        "name": "Narcissus aff. pseudonarcissus MK-2014",
        "taxonomy_id": "1540222",
        "classification": "Eukaryote",
        "common_name": "Daffodil",
        "aliases": ["Narcissus sp. aff. pseudonarcissus"],
    },
}

SUBCELLULAR_LOCATION_CREATE_DEFAULTS: Dict[str, Dict[str, Any]] = {
    "cell": {"name": "cell", "ontology_id": "GO:0005623"},
}


def _display_component_name(value: Any) -> str:
    name = _canonical(value)
    if name and "_" in name and " " not in name:
        name = name.replace("_", " ")
    return _canonical(name)


def _biological_state_component_name(raw: Dict[str, Any], source_key: str) -> str:
    if source_key == "species":
        return _display_component_name(raw.get("species") or raw.get("organism"))
    if source_key == "subcellular_locations":
        return _display_component_name(
            raw.get("subcellular_location")
            or raw.get("subcellular-location")
            or raw.get("compartment")
            or raw.get("compartment_canonical")
        )
    if source_key == "cell_types":
        return _display_component_name(raw.get("cell_type") or raw.get("cell-type"))
    if source_key == "tissues":
        return _display_component_name(raw.get("tissue"))
    return ""


def _hydrate_component_rows_from_biological_states(
    component_rows: Sequence[Any],
    *,
    source_key: str,
    raw_states: Sequence[Dict[str, Any]],
    report: Dict[str, Any],
) -> List[Dict[str, Any]]:
    rows = [dict(row) for row in component_rows if isinstance(row, dict)]
    seen = set()
    for row in rows:
        for alias in [row.get("name"), *_safe_list(row.get("aliases"))]:
            norm = _norm(alias)
            if norm:
                seen.add(norm)

    for idx, raw in enumerate(raw_states):
        name = _biological_state_component_name(raw, source_key)
        norm = _norm(name)
        if not name or not norm or norm in seen:
            continue
        rows.append(
            {
                "name": name,
                "mapping_meta": {
                    "resolution": {
                        "status": "novel",
                        "issue": "inferred_from_biological_state",
                    }
                },
            }
        )
        seen.add(norm)
        _add_issue(
            report,
            "warning",
            "component_inferred_from_biological_state",
            f"Added missing {source_key} component '{name}' from biological_state context.",
            pointer=f"/biological_states/{idx}",
            component_type=source_key,
            name=name,
        )

    return rows


def _dedupe_aliases(values: Iterable[Any]) -> List[str]:
    aliases: List[str] = []
    seen = set()
    for value in values:
        text = _canonical(value)
        norm = _norm(text)
        if not text or not norm or norm in seen:
            continue
        seen.add(norm)
        aliases.append(text)
    return aliases


def _create_default_key(value: Any) -> str:
    return re.sub(r"\s+", " ", _norm(value)).strip()


def _apply_create_defaults(raw: Dict[str, Any], defaults: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    name = _canonical(raw.get("name"))
    default = defaults.get(_create_default_key(name))
    if not default:
        return raw

    record = dict(raw)
    default_name = _canonical(default.get("name"))
    existing_aliases = _safe_list(record.get("aliases"))
    default_aliases = _safe_list(default.get("aliases"))
    aliases = [name, *existing_aliases, *default_aliases]

    if default_name:
        record["name"] = default_name
        aliases.append(default_name)

    for key, value in default.items():
        if key in {"name", "aliases"}:
            continue
        if record.get(key) in (None, ""):
            record[key] = value

    deduped_aliases = _dedupe_aliases(aliases)
    if deduped_aliases:
        record["aliases"] = deduped_aliases
    return record


def _new_report() -> Dict[str, Any]:
    return {
        "ok": True,
        "errors": [],
        "warnings": [],
        "db_resolution": {"compounds": []},
        "unresolved": {
            "db_identities": [],
            "entity_references": [],
            "biological_state_references": [],
        },
        "counts": {},
        "generated_default_state": False,
    }


def _add_issue(
    report: Dict[str, Any],
    severity: str,
    code: str,
    message: str,
    *,
    pointer: str = "",
    **extra: Any,
) -> None:
    issue = {"code": code, "message": message}
    if pointer:
        issue["pointer"] = pointer
    issue.update(extra)
    bucket = "errors" if severity == "error" else "warnings"
    report.setdefault(bucket, []).append(issue)
    if severity == "error":
        report["ok"] = False


def _empty_ir(
    pathway_name: str,
    pathway_subject: str,
    width: int,
    height: int,
) -> Dict[str, Any]:
    return {
        "pathway": {
            "key": "pathway_1",
            "name": pathway_name,
            "subject": pathway_subject,
            "width": int(width),
            "height": int(height),
        },
        "species": [],
        "subcellular_locations": [],
        "cell_types": [],
        "tissues": [],
        "biological_states": [],
        "entities": {
            "compounds": [],
            "proteins": [],
            "nucleic_acids": [],
            "element_collections": [],
            "protein_complexes": [],
            "bounds": [],
        },
        "processes": {
            "reactions": [],
            "reaction_coupled_transports": [],
            "transports": [],
            "interactions": [],
            "sub_pathways": [],
        },
        "locations": [],
        "protein_complex_visualizations": [],
        "bound_visualizations": [],
        "edges": [],
        "process_visualizations": [],
    }


def is_pwml_ir(payload: Any) -> bool:
    if not isinstance(payload, dict):
        return False
    return (
        isinstance(payload.get("pathway"), dict)
        and isinstance(payload.get("entities"), dict)
        and isinstance(payload.get("processes"), dict)
        and isinstance(payload.get("locations"), list)
        and isinstance(payload.get("process_visualizations"), list)
    )


class _KeyFactory:
    def __init__(self) -> None:
        self._counters: Dict[str, int] = defaultdict(int)

    def next(self, prefix: str) -> str:
        self._counters[prefix] += 1
        return f"{prefix}_{self._counters[prefix]}"


def _dedupe_named_rows(
    rows: Iterable[Any],
    *,
    key_prefix: str,
    report: Dict[str, Any],
    pointer_prefix: str,
) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    out: List[Dict[str, Any]] = []
    by_norm: Dict[str, Dict[str, Any]] = {}
    # Where each surviving ``_norm`` group was first seen, so a refusal can point
    # at the survivor as well as the intruder. Kept beside ``by_norm`` rather than
    # on the record: nothing here may add a field to a row that reaches the IR.
    first_idx: Dict[str, int] = {}
    idx = 0
    for raw in rows:
        if not isinstance(raw, dict):
            continue
        name = _canonical(raw.get("name"))
        if not name:
            continue
        n = _norm(name)
        if n in by_norm:
            # C-050i / F-039. This used to warn and ``continue``, destroying the
            # row after the freeze while every reference to it silently repointed
            # to the survivor (``entity_by_name`` and ``component_by_name`` are
            # keyed on this same ``_norm``). It now refuses. See
            # :class:`DuplicateNamedRowError` for why a report issue alone is not
            # sufficient, and why this blocks rather than consolidates.
            survivor = by_norm[n]
            conflicting = [
                {
                    "name": str(survivor.get("name") or ""),
                    "key": str(survivor.get("key") or ""),
                    "pointer": f"{pointer_prefix}/{first_idx[n]}",
                },
                {
                    "name": name,
                    # The intruder never reached the ``record["key"] = ...``
                    # assignment below, so it has no exporter key to name.
                    "key": str(raw.get("key") or ""),
                    "pointer": f"{pointer_prefix}/{idx}",
                },
            ]
            error = DuplicateNamedRowError(
                norm_key=n, pointer_prefix=pointer_prefix, rows=conflicting
            )
            # The structured diagnostic is preserved, and promoted from warning
            # to error. ``build_pwml_ir`` builds its report locally so the raise
            # carries it away, but a caller that owns the report dict -- and any
            # future caller of this helper -- still sees the machine-readable
            # account under the code it has always had.
            _add_issue(
                report,
                "error",
                "duplicate_named_record",
                str(error),
                pointer=f"{pointer_prefix}/{idx}",
                norm_key=n,
                rows=conflicting,
            )
            raise error
        record = dict(raw)
        record["name"] = name
        record["key"] = f"{key_prefix}_{len(out) + 1}"
        out.append(record)
        by_norm[n] = record
        first_idx[n] = idx
        idx += 1
    return out, by_norm


def _copy_common_entity_fields(record: Dict[str, Any], raw: Dict[str, Any]) -> None:
    for key in [
        "mapped_ids",
        "mapping_meta",
        "generated",
        "generation_reason",
        "raw_name",
        "db_status",
        "db_id",
        "db_row",
        "db_match",
        "chosen_rule",
        "class",
        "confidence",
        "provenance",
        "source_refs",
        "evidence",
        "species",
        "taxonomy_id",
        "species_ref",
        "organism",
        "species_id",
        "pathbank_species_id",
        "pw_protein_id",
        "hmdb_id",
        "kegg_id",
        "chebi_id",
        "pubchem_cid",
        "cas",
        "biocyc_id",
        "chemspider_id",
        "moldb_inchi",
        "moldb_inchikey",
        "short_name",
        "synonyms",
        "type",
        "compound_class",
        "compound_type",
        "uniprot",
        "uniprot_id",
        "drugbank",
        "drugbank_id",
        "gene_name",
        "ec_number",
        "ec_numbers",
        "element_states",
        "components",
    ]:
        if key in raw:
            record[key] = raw[key]


def _component_record(raw: Dict[str, Any], key: str, db_keys: Sequence[str]) -> Dict[str, Any]:
    record: Dict[str, Any] = {"key": key, "name": _canonical(raw.get("name"))}
    pathwhiz_id = _db_id(raw, db_keys)
    if pathwhiz_id is not None:
        record["pathwhiz_id"] = pathwhiz_id
    aliases = _dedupe_aliases(_safe_list(raw.get("aliases")))
    if aliases:
        record["aliases"] = aliases
    for src, dst in [
        ("taxonomy_id", "taxonomy_id"),
        ("taxonomy-id", "taxonomy_id"),
        ("ontology_id", "ontology_id"),
        ("ontology-id", "ontology_id"),
        ("classification", "classification"),
        ("common_name", "common_name"),
        ("common-name", "common_name"),
    ]:
        value = raw.get(src)
        if value not in (None, ""):
            record[dst] = value
    return record


def _entity_record(
    raw: Dict[str, Any],
    key: str,
    db_keys: Sequence[str],
    db_field: str,
) -> Dict[str, Any]:
    record: Dict[str, Any] = {"key": key, "name": _canonical(raw.get("name"))}
    db_value = _db_id(raw, db_keys)
    if db_value is not None:
        record["pathwhiz_id"] = db_value
        record[db_field] = db_value
    _copy_common_entity_fields(record, raw)
    return record


def _direction(value: Any) -> str:
    text = _canonical(value).casefold()
    if text in {"<", "left", "reverse"}:
        return "<"
    if text in {"<>", "<=>", "both", "reversible"}:
        return "<>"
    return ">"


def _coerce_participants(value: Any) -> List[Dict[str, Any]]:
    items = value if isinstance(value, list) else []
    out: List[Dict[str, Any]] = []
    for item in items:
        if isinstance(item, str):
            name = _canonical(item)
            if name:
                out.append({"name": name, "stoichiometry": 1})
        elif isinstance(item, dict):
            name = _canonical(
                item.get("name")
                or item.get("compound")
                or item.get("protein")
                or item.get("protein_complex")
                or item.get("element")
                or item.get("entity")
            )
            if not name:
                continue
            stoich = _to_int(item.get("stoichiometry") or item.get("coefficient")) or 1
            out.append({"name": name, "stoichiometry": max(1, stoich)})
    return out


def _component_protein_name(component: Any) -> str:
    if isinstance(component, str):
        return _canonical(component)
    if not isinstance(component, dict):
        return ""
    return _canonical(
        component.get("protein")
        or component.get("protein_name")
        or component.get("name")
        or component.get("component")
        or component.get("entity")
    )


def _component_stoichiometry(component: Any) -> Optional[int]:
    return component_stoichiometry(component)


def _actor_name_and_hint(item: Any) -> Tuple[str, str, str]:
    if isinstance(item, str):
        return _canonical(item), "", ""
    if not isinstance(item, dict):
        return "", "", ""
    for field, hint in [
        ("protein_complex", "protein_complex"),
        ("protein-complex", "protein_complex"),
        ("protein", "protein"),
        ("entity", _canonical(item.get("entity_type")).lower()),
        ("name", _canonical(item.get("entity_type")).lower()),
    ]:
        value = _canonical(item.get(field))
        if value:
            return value, hint, _canonical(item.get("role")).lower()
    return "", "", _canonical(item.get("role")).lower()


def _entity_type_from_location_bucket(bucket: str) -> Tuple[str, str]:
    return {
        "compound_locations": ("compound", "compound"),
        "protein_locations": ("protein", "protein"),
        "nucleic_acid_locations": ("nucleic_acid", "nucleic_acid"),
        "element_collection_locations": ("element_collection", "element_collection"),
    }.get(bucket, ("", ""))


# Culture-collection acronyms and strain markers used to detect the
# strain-qualifier suffix on an organism name. Kept deliberately small and
# case-insensitive; the heuristic below only strips a *trailing run* of such
# tokens, so unrelated words in the binomial are never affected.
_STRAIN_MARKERS = {
    "str",
    "str.",
    "strain",
    "substr",
    "substr.",
    "subsp",
    "subsp.",
    "serovar",
    "biovar",
    "pathovar",
    "pv",
    "pv.",
    "var",
    "var.",
    "isolate",
    "clone",
    "type",
}


def _looks_like_strain_token(token: str) -> bool:
    """True when a trailing token is part of a strain/substrain designation."""
    if not token:
        return False
    low = token.casefold().strip(".")
    if low in {marker.strip(".") for marker in _STRAIN_MARKERS}:
        return True
    if any(ch.isdigit() for ch in token):
        return True  # collection numbers, e.g. "15032", "K-12", "MG1655", "S288C"
    # All-caps alphanumeric codes / culture-collection acronyms, e.g. "IAM",
    # "ATCC", "DSM". Require length >= 2 to avoid stripping a stray initial.
    letters = re.sub(r"[^A-Za-z]", "", token)
    if len(token) >= 2 and letters and token == token.upper() and token != token.casefold():
        return True
    return False


# Sub-species / strain rank markers (period-stripped, lowercased). When one of
# these appears *after* the binomial, everything from it onward is a
# strain/sub-species qualifier and is dropped for determinism.
_STRAIN_RANK_MARKERS = frozenset(
    {"str", "strain", "substr", "subsp", "ssp", "serovar", "biovar", "pathovar", "pv", "var"}
)


def _deterministic_species_name(name: Any) -> str:
    """Return a run-stable canonical species name.

    The LLM/extraction emits the same organism under drifting strain-qualified
    variants (e.g. "Herbaspirillum huttiense IAM 15032" vs "Herbaspirillum
    huttiense" vs "Herbaspirillum huttiense subsp. huttiense IAM 15032"), which
    then collide on import against whichever variant the DB stored first. This
    collapses such variants deterministically by (a) truncating at the first
    sub-species/strain rank marker after the binomial and (b) stripping a
    trailing run of strain-code tokens, always preserving at least the leading
    genus + species epithet. It is a pure function of the input string, so the
    same taxonomy_id yields the same name on every run.
    """
    text = _canonical(name)
    if not text:
        return ""
    tokens = text.split(" ")
    cut = len(tokens)
    # (a) truncate at the first infix strain/sub-species rank marker.
    for idx in range(2, len(tokens)):
        if tokens[idx].casefold().rstrip(".") in _STRAIN_RANK_MARKERS:
            cut = idx
            break
    # (b) strip a trailing run of strain-code tokens (collection numbers, codes).
    while cut > 2 and _looks_like_strain_token(tokens[cut - 1]):
        cut -= 1
    return " ".join(tokens[:cut])


_SPECIES_CREATE_DEFAULT_CANON_NORMS = frozenset(
    _norm(entry.get("name"))
    for entry in SPECIES_CREATE_DEFAULTS.values()
    if _norm(entry.get("name"))
)

#: Where the pre-freeze species stage records **what it decided**, on the payload
#: row it decided about. ``DECISIONS.md`` **D-016 (LOCKED)** moves
#: :func:`_canonicalize_species_offline` upstream of the freeze, so the status it
#: returns is no longer produced inside ``build_pwml_ir`` and cannot be read off
#: the call stack there any more. It travels on the row instead, which is the
#: only carrier that survives ``_hydrate_component_rows_from_biological_states``,
#: ``_apply_create_defaults`` and ``_dedupe_named_rows`` intact.
#:
#: It is provenance, not biology: the name is **not** in
#: ``canonical_hash.GRAPH_FIELDS``, so it is stripped from the graph hash, and
#: ``_component_record`` does not copy it, so the emitted IR shape is unchanged --
#: which is what the docstring below has always promised about this function.
SPECIES_CANONICALIZATION_FIELD = "species_canonicalization"

#: The db-identity keys ``build_pwml_ir``'s species ``component_spec`` names.
#: Kept beside the reader so the pre-freeze caller and ``_component_record``
#: project the same id from the same row.
SPECIES_DB_KEYS: Tuple[str, ...] = ("pathbank_species_id", "pw_species_id", "pathwhiz_id")


def _canonicalize_species_offline(
    record: Dict[str, Any],
    *,
    name_index: Optional[PathwhizNameIndex],
    report: Dict[str, Any],
) -> str:
    """Emit a deterministic, preferably-canonical name for a species record.

    Precedence (documented in docs/setup.md -> "Deterministic species names"):
      1. live resolution-DB name (applied upstream in stage-2 hydration; if the
         record already carries a DB-matched name it flows through unchanged);
      2. offline name-index species-by-taxonomy / pathbank-species-id, plus the
         curated ``SPECIES_CREATE_DEFAULTS`` (both are already canonical);
      3. (reserved) a canonical NCBI-taxonomy-id derived name -- no reliable
         offline source is bundled today, so this is a documented follow-up;
      4. a deterministic normalization of the candidate name (strip strain
         qualifiers) for any *taxonomy-identified* species so the same
         taxonomy_id always emits the same name across runs.

    Species uniqueness in PathWhiz is on both ``name`` and ``taxonomy_id``.
    Truly novel organisms (no taxonomy_id) keep their extraction name.

    Returns a status: ``"canonical"`` (covered by the DB/offline index/curated
    default), ``"deterministic"`` (taxonomy-identified but only normalized -- may
    still collide on import), or ``"novel"`` (no taxonomy id, left untouched). The
    record itself is never tagged, so its emitted shape is unchanged.

    **Where this runs (D-016, LOCKED).** It used to be called from inside
    ``build_pwml_ir``, on the output of :func:`_component_record` -- i.e. by the
    exporter, on a payload the freeze had already hashed, which
    ``PRODUCT_CONTRACT`` §5 and permanent merge rule 8 forbid. Its one caller is
    now :func:`t2pw.pwml.prefreeze_resolution.resolve_species_prefreeze`, which
    hands it the **payload** row instead. The ladder below is unchanged; only the
    two identity reads were widened, so that a payload row and the
    ``_component_record`` projection of that same row resolve to the same
    taxonomy id and the same PathBank id. Both widenings are inert on a
    ``_component_record``, which carries neither an alternate spelling nor a
    ``mapping_meta``.
    """
    canon = report.setdefault("name_canonicalization", {}).setdefault("species", [])

    # (2) offline index by taxonomy / pathbank id -- authoritative when present.
    if name_index is not None:
        hit = name_index.species_canonical(
            # ``_component_record`` folds ``taxonomy-id`` onto ``taxonomy_id``
            # and ``_db_id`` reads the PathBank id through ``_first_nonempty``,
            # which also looks inside ``mapping_meta``/``mapped_ids``. A raw
            # payload row has had neither done to it yet.
            #
            # F-6, recorded not fixed: this ``or`` gives ``taxonomy_id``
            # precedence, while ``_component_record`` assigns both spellings in
            # list order and so lets ``taxonomy-id`` win on a row carrying both.
            # It selects which id is *reported*, never whether a rename applies,
            # and no committed payload carries both spellings.
            taxonomy_id=record.get("taxonomy_id") or record.get("taxonomy-id"),
            pathbank_species_id=_first_nonempty(record, SPECIES_DB_KEYS),
            # F-5, recorded not fixed: ``name`` is accepted and ignored by
            # ``PathwhizNameIndex.species_canonical`` (reserved for a future
            # name-based lookup). The pre-freeze equivalence argument for rows
            # hydrated from ``biological_states`` -- they carry only a name, so
            # they always classified ``novel`` -- rests on that. Should the index
            # ever answer on ``name``, those rows become renameable and the
            # equivalence needs re-measuring.
            name=record.get("name"),
        )
        canonical = _canonical(hit.get("name")) if hit else ""
        if canonical:
            extraction_name = _canonical(record.get("name"))
            if extraction_name and _norm(extraction_name) == _norm(canonical):
                return "canonical"
            record.setdefault("raw_name", extraction_name)
            record["name"] = canonical
            if extraction_name:
                aliases = _dedupe_aliases([*_safe_list(record.get("aliases")), extraction_name])
                if aliases:
                    record["aliases"] = aliases
            canon.append(
                {
                    "from": extraction_name,
                    "to": canonical,
                    "taxonomy_id": record.get("taxonomy_id"),
                    "source": "pathwhiz_id_db.json",
                }
            )
            return "canonical"

    # (4) deterministic normalization for taxonomy-identified species only.
    taxonomy_id = _canonical(record.get("taxonomy_id") or record.get("taxonomy-id"))
    if not taxonomy_id:
        return "novel"  # truly novel / unidentified -> keep extraction name verbatim
    extraction_name = _canonical(record.get("name"))
    if not extraction_name:
        return "novel"
    # A curated create-default name is already the canonical, deterministic form.
    if _norm(extraction_name) in _SPECIES_CREATE_DEFAULT_CANON_NORMS:
        return "canonical"
    canonical = _deterministic_species_name(extraction_name)
    if canonical and _norm(canonical) != _norm(extraction_name):
        record.setdefault("raw_name", extraction_name)
        record["name"] = canonical
        aliases = _dedupe_aliases([*_safe_list(record.get("aliases")), extraction_name])
        if aliases:
            record["aliases"] = aliases
        canon.append(
            {
                "from": extraction_name,
                "to": canonical,
                "taxonomy_id": taxonomy_id,
                "source": "deterministic_strain_normalization",
            }
        )
    return "deterministic"


def _emit_canonicalization_preflight(ir: Dict[str, Any], report: Dict[str, Any]) -> None:
    """Loudly warn when names may be non-canonical and will collide on import.

    Silent offline degradation is the root bug: when the live resolution DB was
    never consulted *and* the offline index does not cover an entity, T2PW emits
    the extraction name, which mismatches the DB's canonical name and fails the
    importer's name+id lookup. This surfaces that condition at WARNING level and
    in the run report (``report['preflight']``) instead of hiding it.
    """
    db_resolution = _safe_dict(report.get("db_resolution"))
    db_available = bool(db_resolution.get("available"))

    at_risk_compounds: List[str] = []
    for rec in _safe_list(_safe_dict(ir.get("entities")).get("compounds")):
        if not isinstance(rec, dict):
            continue
        if _canonical(_safe_dict(rec.get("db_row")).get("name")):
            continue  # canonicalized by the live DB or the offline index
        if rec.get("pathwhiz_id") is not None:
            continue  # has a strong DB identity -> importer matches by id, not name
        ids = _compound_external_ids(rec)
        if any(ids.get(key) for key in ("hmdb", "kegg", "chebi", "pubchem", "drugbank")):
            at_risk_compounds.append(_canonical(rec.get("name")))

    # Species at-risk set is computed during canonicalization (a taxonomy-
    # identified species whose name was only deterministically normalized, not
    # confirmed against the DB / offline index).
    at_risk_species = [
        str(name) for name in report.pop("_species_at_risk", []) if str(name).strip()
    ]

    if db_available or (not at_risk_compounds and not at_risk_species):
        return

    detail = {
        "db_available": False,
        "db_reason": str(db_resolution.get("reason") or "db_not_configured"),
        "compounds": at_risk_compounds,
        "species": at_risk_species,
    }
    report["preflight"] = detail
    message = (
        "Resolution DB unavailable ({reason}) and the offline name index does not "
        "cover {nc} compound(s) and {ns} species; their names may be non-canonical "
        "and will collide with existing rows on PathWhiz import. "
        "Configure the resolution DB (see docs/setup.md) or refresh the offline "
        "index. compounds={cn}; species={sn}".format(
            reason=detail["db_reason"],
            nc=len(at_risk_compounds),
            ns=len(at_risk_species),
            cn=at_risk_compounds[:20],
            sn=at_risk_species[:20],
        )
    )
    logger.warning(message)
    _add_issue(
        report,
        "warning",
        "noncanonical_names_collision_risk",
        message,
        db_reason=detail["db_reason"],
        compounds=at_risk_compounds,
        species=at_risk_species,
    )


def build_pwml_ir(
    payload: dict,
    *,
    pathway_name: str = "Generated Pathway",
    pathway_subject: str = "Metabolic",
    strict_db: bool = True,
    db_resolver: Any = None,
    width: int = 6400,
    height: int = 1400,
    name_index: Any = _NAME_INDEX_UNSET,
) -> tuple[dict, dict]:
    report = _new_report()
    if not isinstance(payload, dict):
        ir = _empty_ir(pathway_name, pathway_subject, width, height)
        _add_issue(report, "error", "invalid_payload", "PWML IR input payload must be an object.")
        return ir, report

    # The pre-freeze stage's verdict on DB reachability, carried on the payload
    # (:data:`PREFREEZE_DB_RESOLUTION_FIELD`) and read here. Read-only, and only
    # when the marker is actually there: a payload no pre-freeze stage ever
    # touched keeps exactly the ``db_resolution`` shape ``_new_report`` gives it,
    # so nothing this exporter already emits moves.
    carried_db_resolution = _safe_dict(payload.get(PREFREEZE_DB_RESOLUTION_FIELD))
    if isinstance(carried_db_resolution.get("available"), bool):
        report["db_resolution"]["available"] = carried_db_resolution["available"]
    # An absent reason is defaulted below to ``"db_not_configured"`` -- false of
    # a DB that was configured and down. Carried for the same reason.
    if str(carried_db_resolution.get("reason") or "").strip():
        report["db_resolution"]["reason"] = str(carried_db_resolution["reason"])

    resolved_name_index = default_name_index() if name_index is _NAME_INDEX_UNSET else name_index

    ir = _empty_ir(pathway_name, pathway_subject, width, height)
    keys = _KeyFactory()
    entities = _safe_dict(payload.get("entities"))
    processes = _safe_dict(payload.get("processes"))

    pathway_meta = _safe_dict(payload.get("metadata"))
    if pathway_meta.get("pathway_name") and pathway_name == "Generated Pathway":
        ir["pathway"]["name"] = _canonical(pathway_meta.get("pathway_name"))
    if pathway_meta.get("pathway_subject") and pathway_subject == "Metabolic":
        ir["pathway"]["subject"] = _canonical(pathway_meta.get("pathway_subject"))

    component_specs = [
        ("species", "species", "sp", ["pathbank_species_id", "pw_species_id", "pathwhiz_id"]),
        (
            "subcellular_locations",
            "subcellular_locations",
            "scl",
            ["pathbank_subcellular_location_id", "pw_subcellular_location_id", "pathwhiz_id"],
        ),
        ("cell_types", "cell_types", "ct", ["pathbank_cell_type_id", "pw_cell_type_id", "pathwhiz_id"]),
        ("tissues", "tissues", "tis", ["pathbank_tissue_id", "pw_tissue_id", "pathwhiz_id"]),
    ]
    component_by_name: Dict[str, Dict[str, Dict[str, Any]]] = {}
    raw_states_for_components = [
        row for row in _safe_list(payload.get("biological_states")) if isinstance(row, dict)
    ]
    for source_key, ir_key, prefix, db_keys in component_specs:
        component_rows = _hydrate_component_rows_from_biological_states(
            _safe_list(entities.get(source_key)),
            source_key=source_key,
            raw_states=raw_states_for_components,
            report=report,
        )
        if source_key == "species":
            component_rows = [
                _apply_create_defaults(row, SPECIES_CREATE_DEFAULTS)
                for row in component_rows
                if isinstance(row, dict)
            ]
        elif source_key == "subcellular_locations":
            component_rows = [
                _apply_create_defaults(row, SUBCELLULAR_LOCATION_CREATE_DEFAULTS)
                for row in component_rows
                if isinstance(row, dict)
            ]
        rows, lookup = _dedupe_named_rows(
            component_rows,
            key_prefix=prefix,
            report=report,
            pointer_prefix=f"/entities/{source_key}",
        )
        ir[ir_key] = [_component_record(row, row["key"], db_keys) for row in rows]
        if source_key == "species":
            # D-016 (LOCKED): the canonicalization itself now happens BEFORE the
            # freeze, in ``prefreeze_resolution.resolve_species_prefreeze``. What
            # is left here is a **reader**, not a second pass: it replays the
            # decision the pre-freeze stage recorded on the row so that this
            # report keeps carrying exactly what it carried before -- the
            # ``name_canonicalization["species"]`` log, in row order, and the
            # ``_species_at_risk`` set ``_emit_canonicalization_preflight``
            # consumes below. Nothing is re-derived and no name is rewritten
            # here; a row with no marker (one hydrated from a biological state,
            # which the ladder always classified ``novel``) contributes nothing,
            # exactly as before.
            species_at_risk: List[str] = []
            if ir[ir_key]:
                # The ladder created this key on its first call, so it was
                # present -- as ``{"species": []}`` -- for every payload that had
                # a species row at all, whether or not anything was renamed.
                # ``pwml_ir_report.json`` is a committed artifact and C-040's
                # golden hashes this report, so the empty case is part of the
                # shape, not an accident of it.
                report.setdefault("name_canonicalization", {}).setdefault("species", [])
            for row, record in zip(rows, ir[ir_key]):
                marker = _safe_dict(row.get(SPECIES_CANONICALIZATION_FIELD))
                entry = marker.get("entry")
                if isinstance(entry, dict):
                    report.setdefault("name_canonicalization", {}).setdefault(
                        "species", []
                    ).append(entry)
                if marker.get("status") == "deterministic":
                    species_at_risk.append(_canonical(record.get("name")))
            report["_species_at_risk"] = species_at_risk
        by_name: Dict[str, Dict[str, Any]] = {}
        for row in ir[ir_key]:
            for alias in [row.get("name"), *_safe_list(row.get("aliases"))]:
                norm = _norm(alias)
                if norm:
                    by_name[norm] = row
        component_by_name[source_key] = by_name

    entity_specs = [
        (
            "compounds",
            "compound",
            "cmp",
            ["pathbank_compound_id", "pw_compound_id", "pathwhiz_id"],
            "pathbank_compound_id",
            False,
        ),
        (
            "proteins",
            "protein",
            "prot",
            ["pathbank_protein_id", "pw_protein_id", "pathwhiz_id"],
            "pathbank_protein_id",
            False,
        ),
        (
            "nucleic_acids",
            "nucleic_acid",
            "na",
            ["pathbank_nucleic_acid_id", "pw_nucleic_acid_id", "pathwhiz_id"],
            "pathbank_nucleic_acid_id",
            False,
        ),
        (
            "element_collections",
            "element_collection",
            "ec",
            ["pathbank_element_collection_id", "pw_element_collection_id", "pathwhiz_id"],
            "pathbank_element_collection_id",
            False,
        ),
        (
            "protein_complexes",
            "protein_complex",
            "pc",
            ["pathbank_complex_id", "pathbank_protein_complex_id", "pw_complex_id", "pathwhiz_id"],
            "pathbank_complex_id",
            False,
        ),
    ]

    entity_by_name: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    entity_by_key: Dict[str, Dict[str, Any]] = {}
    for source_key, entity_type, prefix, db_keys, db_field, strict_required in entity_specs:
        rows, _ = _dedupe_named_rows(
            _safe_list(entities.get(source_key)),
            key_prefix=prefix,
            report=report,
            pointer_prefix=f"/entities/{source_key}",
        )
        if source_key == "compounds":
            # D-021 (LOCKED) section 1: "C-051 -- assert only. Delete the
            # now-redundant in-IR resolution call and replace it with a
            # fail-closed assertion that every compound row already carries a
            # resolution verdict. C-051 resolves nothing and repairs nothing."
            #
            # The deleted call was ``_resolve_compound_rows(...)``: the exporter
            # resolving compound identity AFTER the canonical graph was frozen,
            # which permanent merge rule 8 and ``PRODUCT_CONTRACT`` section 5
            # forbid. D-015 (LOCKED) put that work in
            # ``prefreeze_resolution.resolve_compounds_prefreeze``, and D-032
            # (LOCKED) wired the pre-freeze sequence at BOTH production export
            # entry points -- ``streamlit_app`` and
            # ``writer.run_pwml_pipeline_export`` -- so by the time the IR is
            # built the verdict is always already on the row.
            #
            # Nothing here constructs, resolves, renames or repairs a row: the
            # only outcomes are "carry on unchanged" and "refuse loudly".
            unresolved_rows = [
                row
                for row in rows
                if not str(row.get(COMPOUND_RESOLUTION_VERDICT_FIELD) or "").strip()
            ]
            if unresolved_rows:
                raise UnresolvedCompoundRowError(
                    unresolved_rows, pointer_prefix=f"/entities/{source_key}"
                )
        bucket = ENTITY_BUCKETS[entity_type]
        for row in rows:
            rec = _entity_record(row, row["key"], db_keys, db_field)
            if strict_db and strict_required and rec.get("pathwhiz_id") is None:
                issue = {
                    "entity_type": entity_type,
                    "entity_key": rec["key"],
                    "name": rec["name"],
                    "required_fields": list(db_keys),
                }
                report["unresolved"]["db_identities"].append(issue)
                _add_issue(
                    report,
                    "error" if strict_db else "warning",
                    "missing_db_identity",
                    f"{entity_type} '{rec['name']}' has no required DB-backed PWML identity.",
                    pointer=f"/entities/{source_key}",
                    **issue,
                )
            ir["entities"][bucket].append(rec)
            typed = dict(rec)
            typed["entity_type"] = entity_type
            entity_by_key[typed["key"]] = typed
            for alias in [
                rec.get("name"),
                rec.get("raw_name"),
                rec.get("short_name"),
                rec.get("common_name"),
            ]:
                alias_norm = _norm(alias)
                if alias_norm:
                    entity_by_name[alias_norm].append(typed)
            for alias in _safe_list(rec.get("synonyms")):
                alias_norm = _norm(alias)
                if alias_norm:
                    entity_by_name[alias_norm].append(typed)

    proteins_by_key = {
        str(protein.get("key")): protein
        for protein in ir["entities"]["proteins"]
        if isinstance(protein, dict) and protein.get("key")
    }
    proteins_by_name = {
        _norm(protein.get("name")): protein
        for protein in ir["entities"]["proteins"]
        if isinstance(protein, dict) and _norm(protein.get("name"))
    }
    proteins_by_db_id: Dict[int, Dict[str, Any]] = {}
    proteins_by_uniprot: Dict[str, Dict[str, Any]] = {}
    for protein in ir["entities"]["proteins"]:
        if not isinstance(protein, dict):
            continue
        for db_key in ["pathbank_protein_id", "pw_protein_id", "pathwhiz_id"]:
            db_id = _to_int(protein.get(db_key))
            if db_id is not None:
                proteins_by_db_id[db_id] = protein
        uniprot = _canonical(
            _first_nonempty(
                protein,
                ["uniprot", "uniprot_id", "uniprot-id"],
            )
        ).casefold()
        if uniprot:
            proteins_by_uniprot[uniprot] = protein

    for pc_idx, pc in enumerate(ir["entities"]["protein_complexes"]):
        raw_components = _safe_list(pc.get("components"))
        if not raw_components:
            _add_issue(
                report,
                "warning",
                "protein_complex_missing_components",
                f"Protein complex '{pc.get('name')}' has no protein components.",
                pointer=f"/entities/protein_complexes/{pc_idx}/components",
                entity_key=pc.get("key"),
                entity_name=pc.get("name"),
            )
            pc["components"] = []
            continue

        structured_components: List[Dict[str, Any]] = []
        for comp_idx, component in enumerate(raw_components):
            pointer = f"/entities/protein_complexes/{pc_idx}/components/{comp_idx}"
            protein: Optional[Dict[str, Any]] = None
            if isinstance(component, dict):
                protein_key = _canonical(component.get("protein_key") or component.get("entity_key"))
                if protein_key:
                    protein = proteins_by_key.get(protein_key)
                if protein is None:
                    comp_db_id = _db_id(
                        component,
                        ["pathbank_protein_id", "pw_protein_id", "pathwhiz_id", "protein_id"],
                    )
                    if comp_db_id is not None:
                        protein = proteins_by_db_id.get(comp_db_id)
                if protein is None:
                    comp_uniprot = _canonical(
                        _first_nonempty(
                            component,
                            ["uniprot", "uniprot_id", "uniprot-id"],
                        )
                    ).casefold()
                    if comp_uniprot:
                        protein = proteins_by_uniprot.get(comp_uniprot)
            comp_name = _component_protein_name(component)
            if protein is None and comp_name:
                protein = proteins_by_name.get(_norm(comp_name))

            stoich = _component_stoichiometry(component)
            if stoich is None:
                _add_issue(
                    report,
                    "warning",
                    "component_stoichiometry_unstated",
                    f"Component[{comp_idx}] in complex '{pc.get('name')}' has no stated stoichiometry; leaving it blank.",
                    pointer=pointer,
                    complex_key=pc.get("key"),
                    complex_name=pc.get("name"),
                    component_name=comp_name,
                )
            if protein is None:
                _add_issue(
                    report,
                    "warning",
                    "component_protein_unresolved",
                    f"Component '{comp_name or comp_idx}' in complex '{pc.get('name')}' does not reference an existing protein.",
                    pointer=pointer,
                    complex_key=pc.get("key"),
                    complex_name=pc.get("name"),
                    component_name=comp_name,
                )
                continue

            structured_record: Dict[str, Any] = {"protein_key": str(protein["key"])}
            if stoich is not None:
                structured_record["stoichiometry"] = stoich
            structured_components.append(structured_record)

        pc["components"] = structured_components
        if not structured_components:
            _add_issue(
                report,
                "warning",
                "protein_complex_missing_components",
                f"Protein complex '{pc.get('name')}' has no resolved protein components.",
                pointer=f"/entities/protein_complexes/{pc_idx}/components",
                entity_key=pc.get("key"),
                entity_name=pc.get("name"),
            )

    def resolve_component(source_key: str, value: Any, pointer: str) -> Optional[str]:
        name = _canonical(value)
        if not name:
            return None
        found = component_by_name.get(source_key, {}).get(_norm(name))
        if found:
            return str(found["key"])
        report["unresolved"]["biological_state_references"].append(
            {"component_type": source_key, "name": name, "pointer": pointer}
        )
        _add_issue(
            report,
            "error" if strict_db else "warning",
            "unresolved_biological_state_component",
            f"Biological state component '{name}' was not found in entities.{source_key}.",
            pointer=pointer,
            component_type=source_key,
            name=name,
        )
        return None

    raw_states = [row for row in _safe_list(payload.get("biological_states")) if isinstance(row, dict)]
    if not raw_states:
        report["generated_default_state"] = True
        context_exists = any(ir[key] for key in ["species", "subcellular_locations", "cell_types", "tissues"])
        raw_states = [{"name": "Default state"}]
        _add_issue(
            report,
            "warning",
            "generated_default_state",
            "No biological_states records were present; generated a default PWML biological state.",
        )
        if strict_db and not context_exists:
            _add_issue(
                report,
                "warning",
                "default_state_without_context",
                "Generated default biological state has no species, subcellular location, cell type, or tissue context.",
            )

    state_by_name: Dict[str, Dict[str, Any]] = {}
    for idx, raw in enumerate(raw_states):
        name = _canonical(raw.get("name")) or f"State {idx + 1}"
        state_key = f"bs_{idx + 1}"
        species_key = resolve_component("species", raw.get("species"), f"/biological_states/{idx}/species")
        subcellular_key = resolve_component(
            "subcellular_locations",
            raw.get("subcellular_location")
            or raw.get("subcellular-location")
            or raw.get("compartment")
            or raw.get("compartment_canonical"),
            f"/biological_states/{idx}/subcellular_location",
        )
        cell_type_key = resolve_component("cell_types", raw.get("cell_type"), f"/biological_states/{idx}/cell_type")
        tissue_key = resolve_component("tissues", raw.get("tissue"), f"/biological_states/{idx}/tissue")

        # If the state is named after one component, use that as deterministic context.
        if not species_key:
            species_key = (component_by_name.get("species") or {}).get(_norm(name), {}).get("key")
        if not subcellular_key:
            subcellular_key = (component_by_name.get("subcellular_locations") or {}).get(_norm(name), {}).get("key")
        if not cell_type_key:
            cell_type_key = (component_by_name.get("cell_types") or {}).get(_norm(name), {}).get("key")
        if not tissue_key:
            tissue_key = (component_by_name.get("tissues") or {}).get(_norm(name), {}).get("key")

        state = {
            "key": state_key,
            "name": name,
            "species_key": species_key,
            "subcellular_location_key": subcellular_key,
            "cell_type_key": cell_type_key,
            "tissue_key": tissue_key,
        }
        if not species_key:
            _add_issue(
                report,
                "error",
                "biological_state_missing_species",
                f"Biological state '{name}' has no resolved species reference.",
                pointer=f"/biological_states/{idx}",
                biological_state_key=state_key,
            )
        if not subcellular_key:
            _add_issue(
                report,
                "error",
                "biological_state_missing_subcellular_location",
                f"Biological state '{name}' has no resolved subcellular location reference.",
                pointer=f"/biological_states/{idx}",
                biological_state_key=state_key,
            )
        if not any([species_key, subcellular_key, cell_type_key, tissue_key]):
            _add_issue(
                report,
                "warning",
                "biological_state_without_components",
                f"Biological state '{name}' has no resolved species/location/cell/tissue references.",
                pointer=f"/biological_states/{idx}",
                biological_state_key=state_key,
            )
        ir["biological_states"].append(state)
        state_by_name[_norm(name)] = state

    default_bs_key = ir["biological_states"][0]["key"] if ir["biological_states"] else None

    def resolve_state(value: Any, pointer: str, *, required: bool = False) -> Optional[str]:
        name = _canonical(value)
        if not name:
            if required and default_bs_key:
                return default_bs_key
            return default_bs_key
        found = state_by_name.get(_norm(name))
        if found:
            return str(found["key"])
        report["unresolved"]["biological_state_references"].append(
            {"component_type": "biological_state", "name": name, "pointer": pointer}
        )
        _add_issue(
            report,
            "error" if strict_db or required else "warning",
            "unresolved_biological_state",
            f"Biological state '{name}' was not found.",
            pointer=pointer,
            biological_state=name,
        )
        return default_bs_key

    preferred_order = {
        "reaction_member": ["compound", "nucleic_acid", "element_collection", "protein_complex", "protein"],
        "enzyme": ["protein_complex", "protein"],
        "transporter": ["protein_complex", "protein"],
        "interaction": ["protein", "protein_complex", "compound", "nucleic_acid", "element_collection"],
    }

    def resolve_entity(name: Any, role: str, pointer: str, hint: str = "") -> Optional[Dict[str, Any]]:
        clean = _canonical(name)
        if not clean:
            return None
        candidates = entity_by_name.get(_norm(clean), [])
        if not candidates:
            report["unresolved"]["entity_references"].append({"name": clean, "role": role, "pointer": pointer})
            _add_issue(
                report,
                "error",
                "unresolved_entity_reference",
                f"Process member '{clean}' was not found in entity registries.",
                pointer=pointer,
                name=clean,
                role=role,
            )
            return None
        ordered = [hint] if hint in ENTITY_BUCKETS else []
        ordered.extend(preferred_order.get(role, []))
        for wanted in ordered:
            for candidate in candidates:
                if candidate.get("entity_type") == wanted:
                    return candidate
        if len(candidates) > 1:
            _add_issue(
                report,
                "warning",
                "ambiguous_entity_reference",
                f"Process member '{clean}' matched multiple entity types; using {candidates[0]['entity_type']}.",
                pointer=pointer,
                name=clean,
                entity_types=[c["entity_type"] for c in candidates],
            )
        return candidates[0]

    def ensure_enzyme_complex_entity(entity: Dict[str, Any], pointer: str) -> Dict[str, Any]:
        if entity.get("entity_type") == "protein_complex":
            return entity
        if entity.get("entity_type") != "protein":
            return entity
        protein_name = _canonical(entity.get("name")) or str(entity.get("key") or "<unnamed>")
        _add_issue(
            report,
            "error",
            "reaction_enzyme_must_be_protein_complex",
            f"Reaction enzyme '{protein_name}' is still a bare protein; Stage 6 must create its PathWhiz protein-complex wrapper before export.",
            pointer=pointer,
            protein_key=entity.get("key"),
            protein_name=protein_name,
            entity_type="protein",
        )
        return entity

    location_by_tuple: Dict[Tuple[str, str, str], Dict[str, Any]] = {}

    def ensure_location(
        entity_type: str,
        entity_key: str,
        biological_state_key: Optional[str],
        *,
        x: int,
        y: int,
    ) -> Optional[Dict[str, Any]]:
        if not biological_state_key:
            return None
        entity = entity_by_key.get(entity_key)
        if not entity:
            return None
        cache_key = (entity_type, entity_key, biological_state_key)
        if cache_key in location_by_tuple:
            return location_by_tuple[cache_key]
        location_type = {
            "compound": "compound_location",
            "protein": "protein_location",
            "protein_complex": "protein_complex_location",
            "nucleic_acid": "nucleic_acid_location",
            "element_collection": "element_collection_location",
        }.get(entity_type)
        if not location_type:
            return None
        prefix = {
            "compound": "cl",
            "protein": "pl",
            "protein_complex": "pcl",
            "nucleic_acid": "nal",
            "element_collection": "ecl",
        }[entity_type]
        loc = {
            "key": keys.next(prefix),
            "type": location_type,
            "entity_type": entity_type,
            "entity_key": entity_key,
            "biological_state_key": biological_state_key,
            "x": int(x),
            "y": int(y),
            "zindex": 10 if entity_type == "compound" else 8,
            "visualization_template_id": (
                select_compound_template_id(entity)
                if entity_type == "compound"
                else 2
            ),
            "hidden": False,
        }
        ir["locations"].append(loc)
        location_by_tuple[cache_key] = loc
        return loc

    def add_edge(x1: int, y1: int, x2: int, y2: int) -> Dict[str, Any]:
        edge = {
            "key": keys.next("edge"),
            "path": f"M{x1} {y1} L{x2} {y2}",
            "visualization_template_id": 5,
            "zindex": 18,
            "hidden": False,
        }
        ir["edges"].append(edge)
        return edge

    for idx, state in enumerate(ir["biological_states"]):
        ir["bound_visualizations"].append(
            {
                "key": keys.next("bv"),
                "biological_state_key": state["key"],
                "x": 0,
                "y": max(0, int(idx * (height / max(1, len(ir["biological_states"]))))),
                "width": int(width),
                "height": max(120, int(height / max(1, len(ir["biological_states"])))),
                "zindex": 1,
                "hidden": False,
            }
        )

    # Honor explicit element_locations as visual references without duplicating entities.
    # NOTE: this used to run before reactions, which caused the flat grid to win over the
    # reaction-aware coordinates for every compound. We now build it as a deferred list
    # and only register entities that aren't placed by a reaction/transport later.
    explicit_locations_to_register: List[Tuple[str, str, Optional[str]]] = []
    element_locations = _safe_dict(payload.get("element_locations"))
    for bucket, rows in element_locations.items():
        entity_type, name_field = _entity_type_from_location_bucket(bucket)
        if not entity_type:
            continue
        for row_idx, raw in enumerate(_safe_list(rows)):
            if not isinstance(raw, dict):
                continue
            pointer_field = name_field if raw.get(name_field) else "entity"
            entity_name = _canonical(raw.get(name_field) or raw.get("entity"))
            if entity_name and not entity_by_name.get(_norm(entity_name)):
                _add_issue(
                    report,
                    "warning",
                    "location_entity_not_found",
                    f"Location for '{entity_name}' does not reference an existing entity.",
                    pointer=f"/element_locations/{bucket}/{row_idx}/{pointer_field}",
                    entity_name=entity_name,
                )
                continue
            entity = resolve_entity(
                entity_name,
                "reaction_member",
                f"/element_locations/{bucket}/{row_idx}/{pointer_field}",
                hint=entity_type,
            )
            if not entity:
                continue
            bs_key = resolve_state(raw.get("biological_state"), f"/element_locations/{bucket}/{row_idx}/biological_state")
            explicit_locations_to_register.append((entity["entity_type"], entity["key"], bs_key))

    process_by_name: Dict[str, Dict[str, Any]] = {}
    process_by_key: Dict[str, Dict[str, Any]] = {}

    def process_xy(index: int) -> Tuple[int, int]:
        # Leave enough room on the left for reactant nodes (REACTANT_OFFSET + half a
        # node width + margin) so the leftmost compounds stay on-canvas.
        margin_left = 380
        col_w = 540
        usable_w = max(col_w, width - margin_left - 200)
        cols = max(1, usable_w // col_w)
        col = index % cols
        row = index // cols
        # First reaction sits roughly a third of the way down the canvas so enzymes
        # placed above (rx_y - ~150) don't run off the top.
        return margin_left + col * col_w, max(260, int(height * 0.35)) + row * 240

    allowed_enzyme_entity_types = {"protein", "protein_complex"}
    reaction_index = 0
    for ridx, raw in enumerate(_safe_list(processes.get("reactions"))):
        if not isinstance(raw, dict):
            continue
        reaction_index += 1
        rx_key = f"rxn_{reaction_index}"
        bs_key = resolve_state(raw.get("biological_state"), f"/processes/reactions/{ridx}/biological_state", required=True)
        rx_x, rx_y = process_xy(reaction_index - 1)
        reaction = {
            "key": rx_key,
            "type": "reaction",
            "name": _canonical(raw.get("name")) or f"Reaction {reaction_index}",
            "direction": _direction(raw.get("direction")),
            "biological_state_key": bs_key,
            "left": [],
            "right": [],
            "enzymes": [],
            # Spontaneity is not modeled currently; every reaction is exported as non-spontaneous.
            "spontaneous": False,
        }
        members_for_viz: List[Dict[str, Any]] = []

        REACTANT_OFFSET = 240
        PRODUCT_OFFSET = 240
        NODE_SPACING_Y = 95
        NODE_W = 160
        NODE_H = 60

        # Pre-count valid participants per side to center their vertical stacks on rx_y
        side_counts: Dict[str, int] = {}
        for side_name, raw_key in [("left", "inputs"), ("right", "outputs")]:
            n = 0
            for part in _coerce_participants(raw.get(raw_key)):
                if part.get("name"):
                    n += 1
            side_counts[side_name] = n

        for side_name, raw_key, x_offset in [
            ("left", "inputs", -REACTANT_OFFSET),
            ("right", "outputs", PRODUCT_OFFSET),
        ]:
            total = side_counts[side_name]
            y_top = rx_y - ((total - 1) * NODE_SPACING_Y) // 2 - NODE_H // 2
            for midx, part in enumerate(_coerce_participants(raw.get(raw_key))):
                entity = resolve_entity(
                    part["name"],
                    "reaction_member",
                    f"/processes/reactions/{ridx}/{raw_key}/{midx}",
                )
                if not entity:
                    continue
                member_key = f"{rx_key}_{side_name}_{len(reaction[side_name]) + 1}"
                member = {
                    "key": member_key,
                    "entity_type": entity["entity_type"],
                    "entity_key": entity["key"],
                    "stoichiometry": int(part.get("stoichiometry") or 1),
                }
                reaction[side_name].append(member)
                idx_on_side = len(reaction[side_name]) - 1
                node_x = rx_x + x_offset - NODE_W // 2
                node_y = y_top + idx_on_side * NODE_SPACING_Y
                loc = ensure_location(
                    entity["entity_type"],
                    entity["key"],
                    bs_key,
                    x=node_x,
                    y=node_y,
                )
                if loc:
                    # Side-aware edge anchor: reactants exit at the right edge, products
                    # are entered at the left edge.
                    lx = int(loc["x"])
                    ly = int(loc["y"])
                    if side_name == "left":
                        ax, ay = lx + NODE_W, ly + NODE_H // 2
                        edge = add_edge(ax, ay, rx_x, rx_y)
                    else:
                        ax, ay = lx, ly + NODE_H // 2
                        edge = add_edge(rx_x, rx_y, ax, ay)
                    members_for_viz.append(
                        {
                            "process_member_key": member_key,
                            "role": side_name,
                            "member_type": entity["entity_type"],
                            "location_key": loc["key"],
                            "edge_key": edge["key"],
                        }
                    )

        enzyme_sources: List[Tuple[Any, bool]] = [
            (actor, False) for actor in _safe_list(raw.get("enzymes"))
        ]
        for mod in _safe_list(raw.get("modifiers")):
            if not isinstance(mod, dict):
                continue
            role = _canonical(mod.get("role")).lower()
            if role in {"catalyst", "activator", "inhibitor", "enzyme"}:
                enzyme_sources.append((mod, True))
        explicit_enzyme_actors: set[Tuple[str, str]] = set()
        for eidx, (actor, is_modifier) in enumerate(enzyme_sources):
            name, hint, role = _actor_name_and_hint(actor)
            if not name:
                if isinstance(actor, dict) and actor.get("evidence"):
                    _add_issue(
                        report,
                        "warning",
                        "enzyme_without_resolvable_actor",
                        "Reaction enzyme/modifier evidence did not include an explicit entity reference.",
                        pointer=f"/processes/reactions/{ridx}/enzymes/{eidx}",
                    )
                continue
            entity = resolve_entity(name, "enzyme", f"/processes/reactions/{ridx}/enzymes/{eidx}", hint=hint)
            if not entity:
                continue
            entity_type = str(entity.get("entity_type") or "").casefold()
            if entity_type not in allowed_enzyme_entity_types:
                _add_issue(
                    report,
                    "warning",
                    "non_protein_catalyst_dropped",
                    f"Reaction enzyme/modifier '{name}' resolved to {entity_type or 'unknown'} and was dropped.",
                    pointer=f"/processes/reactions/{ridx}/enzymes/{eidx}",
                    name=name,
                    entity_type=entity_type,
                    role=role or "catalyst",
                )
                continue
            entity = ensure_enzyme_complex_entity(entity, f"/processes/reactions/{ridx}/enzymes/{eidx}")
            logical_actor = (str(entity.get("key") or ""), role or "catalyst")
            if is_modifier and logical_actor in explicit_enzyme_actors:
                continue
            if not is_modifier:
                explicit_enzyme_actors.add(logical_actor)
            member_key = f"{rx_key}_enzyme_{len(reaction['enzymes']) + 1}"
            reaction["enzymes"].append(
                {"key": member_key, "entity_type": entity["entity_type"], "entity_key": entity["key"], "role": role or "catalyst"}
            )
            enzyme_w = 200
            enzyme_h = 60
            loc = ensure_location(
                entity["entity_type"],
                entity["key"],
                bs_key,
                x=rx_x - enzyme_w // 2,
                y=rx_y - 150 - enzyme_h // 2,
            )
            enzyme_edge_key: Optional[str] = None
            if loc:
                ex = int(loc["x"]) + enzyme_w // 2
                ey = int(loc["y"]) + enzyme_h
                enzyme_edge = add_edge(ex, ey, rx_x, rx_y)
                # Dashed catalysis-style edge
                enzyme_edge["visualization_template_id"] = 83
                enzyme_edge_key = enzyme_edge["key"]
                members_for_viz.append(
                    {
                        "process_member_key": member_key,
                        "role": "enzyme",
                        "member_type": entity["entity_type"],
                        "location_key": loc["key"],
                        "edge_key": enzyme_edge_key,
                    }
                )

        ir["processes"]["reactions"].append(reaction)
        process_by_key[rx_key] = reaction
        process_by_name[_norm(reaction["name"])] = reaction
        ir["process_visualizations"].append(
            {
                "key": keys.next("rv"),
                "type": "reaction_visualization",
                "process_key": rx_key,
                "biological_state_key": bs_key,
                "x": rx_x,
                "y": rx_y,
                "members": members_for_viz,
            }
        )

    transport_index = 0
    for tidx, raw in enumerate(_safe_list(processes.get("transports"))):
        if not isinstance(raw, dict):
            continue
        transport_index += 1
        tr_key = f"tr_{transport_index}"
        tr_x, tr_y = process_xy(reaction_index + transport_index - 1)
        transport = {
            "key": tr_key,
            "type": "transport",
            "name": _canonical(raw.get("name")) or f"Transport {transport_index}",
            "transport_elements": [],
            "transporters": [],
        }
        members_for_viz = []
        cargo_items = _safe_list(raw.get("transport_elements"))
        if not cargo_items:
            cargo_name = _canonical(raw.get("cargo") or raw.get("cargo_complex"))
            cargo_items = [{"element": cargo_name}] if cargo_name else []
        for cidx, cargo in enumerate(cargo_items):
            if isinstance(cargo, str):
                cargo = {"element": cargo}
            if not isinstance(cargo, dict):
                continue
            name = _canonical(cargo.get("element") or cargo.get("cargo") or cargo.get("name"))
            entity = resolve_entity(name, "reaction_member", f"/processes/transports/{tidx}/transport_elements/{cidx}")
            if not entity:
                continue
            left_bs = resolve_state(
                cargo.get("left_biological_state")
                or cargo.get("from_biological_state")
                or raw.get("from_biological_state"),
                f"/processes/transports/{tidx}/from_biological_state",
                required=True,
            )
            right_bs = resolve_state(
                cargo.get("right_biological_state")
                or cargo.get("to_biological_state")
                or raw.get("to_biological_state"),
                f"/processes/transports/{tidx}/to_biological_state",
                required=True,
            )
            member_key = f"{tr_key}_el_{len(transport['transport_elements']) + 1}"
            transport["transport_elements"].append(
                {
                    "key": member_key,
                    "entity_type": entity["entity_type"],
                    "entity_key": entity["key"],
                    "left_biological_state_key": left_bs,
                    "right_biological_state_key": right_bs,
                    "stoichiometry": _to_int(cargo.get("stoichiometry")) or 1,
                }
            )
            left_loc = ensure_location(entity["entity_type"], entity["key"], left_bs, x=tr_x - 145, y=tr_y)
            right_loc = ensure_location(entity["entity_type"], entity["key"], right_bs, x=tr_x + 145, y=tr_y)
            if left_loc:
                edge = add_edge(int(left_loc["x"]) + 60, int(left_loc["y"]) + 20, tr_x, tr_y)
                members_for_viz.append(
                    {
                        "process_member_key": member_key,
                        "role": "left",
                        "member_type": entity["entity_type"],
                        "location_key": left_loc["key"],
                        "edge_key": edge["key"],
                    }
                )
            if right_loc:
                edge = add_edge(tr_x, tr_y, int(right_loc["x"]) + 60, int(right_loc["y"]) + 20)
                members_for_viz.append(
                    {
                        "process_member_key": member_key,
                        "role": "right",
                        "member_type": entity["entity_type"],
                        "location_key": right_loc["key"],
                        "edge_key": edge["key"],
                    }
                )

        for aidx, actor in enumerate(_safe_list(raw.get("transporters"))):
            name, hint, _role = _actor_name_and_hint(actor)
            entity = resolve_entity(name, "transporter", f"/processes/transports/{tidx}/transporters/{aidx}", hint=hint)
            if not entity:
                continue
            member_key = f"{tr_key}_transporter_{len(transport['transporters']) + 1}"
            bs_key = resolve_state(
                actor.get("biological_state") if isinstance(actor, dict) else "",
                f"/processes/transports/{tidx}/transporters/{aidx}/biological_state",
                required=True,
            )
            transport["transporters"].append(
                {"key": member_key, "entity_type": entity["entity_type"], "entity_key": entity["key"], "biological_state_key": bs_key}
            )
            loc = ensure_location(entity["entity_type"], entity["key"], bs_key, x=tr_x, y=tr_y - 90)
            if loc:
                members_for_viz.append(
                    {
                        "process_member_key": member_key,
                        "role": "transporter",
                        "member_type": entity["entity_type"],
                        "location_key": loc["key"],
                        "edge_key": None,
                    }
                )

        ir["processes"]["transports"].append(transport)
        process_by_key[tr_key] = transport
        process_by_name[_norm(transport["name"])] = transport
        ir["process_visualizations"].append(
            {
                "key": keys.next("tv"),
                "type": "transport_visualization",
                "process_key": tr_key,
                "biological_state_key": default_bs_key,
                "x": tr_x,
                "y": tr_y,
                "members": members_for_viz,
            }
        )

    interaction_index = 0
    for iidx, raw in enumerate(_safe_list(processes.get("interactions"))):
        if not isinstance(raw, dict):
            continue
        left_name = _canonical(raw.get("left") or raw.get("entity_1") or raw.get("source"))
        right_name = _canonical(raw.get("right") or raw.get("entity_2") or raw.get("target"))
        left_entity = resolve_entity(left_name, "interaction", f"/processes/interactions/{iidx}/entity_1")
        right_entity = resolve_entity(right_name, "interaction", f"/processes/interactions/{iidx}/entity_2")
        interaction_index += 1
        int_key = f"int_{interaction_index}"
        bs_key = resolve_state(raw.get("biological_state"), f"/processes/interactions/{iidx}/biological_state", required=True)
        interaction = {
            "key": int_key,
            "type": "interaction",
            "name": _canonical(raw.get("name")) or f"Interaction {interaction_index}",
            "interaction_type": _canonical(raw.get("interaction_type") or raw.get("relationship") or raw.get("class") or "interaction"),
            "biological_state_key": bs_key,
            "left": (
                {"entity_type": left_entity["entity_type"], "entity_key": left_entity["key"]}
                if left_entity
                else None
            ),
            "right": (
                {"entity_type": right_entity["entity_type"], "entity_key": right_entity["key"]}
                if right_entity
                else None
            ),
        }
        ir["processes"]["interactions"].append(interaction)
        process_by_key[int_key] = interaction
        process_by_name[_norm(interaction["name"])] = interaction

    # Fallback grid for entities referenced via explicit element_locations but not
    # placed by any reaction/transport. They use a flat grid below the reaction band.
    orphan_idx = 0
    for entity_type, entity_key, bs_key in explicit_locations_to_register:
        cache_key = (entity_type, entity_key, bs_key)
        if cache_key in location_by_tuple:
            continue
        orphan_idx += 1
        ensure_location(
            entity_type,
            entity_key,
            bs_key,
            x=40 + (orphan_idx % 8) * 200,
            y=int(height * 0.75) + (orphan_idx // 8) * 100,
        )

    # Preserve RCT/sub-pathway records in typed buckets. Full visualization is left
    # explicit so the validator can reject incomplete records before serialization.
    for idx, raw in enumerate(_safe_list(processes.get("reaction_coupled_transports"))):
        if not isinstance(raw, dict):
            continue
        rct = {
            "key": f"rct_{idx + 1}",
            "type": "reaction_coupled_transport",
            "name": _canonical(raw.get("name")) or f"Reaction coupled transport {idx + 1}",
            "left": [],
            "right": [],
            "enzymes": [],
            "source": raw,
        }
        ir["processes"]["reaction_coupled_transports"].append(rct)
    for idx, raw in enumerate(_safe_list(processes.get("sub_pathways"))):
        if not isinstance(raw, dict):
            continue
        ir["processes"]["sub_pathways"].append(
            {
                "key": f"subp_{idx + 1}",
                "type": "sub_pathway",
                "name": _canonical(raw.get("name")) or f"Sub pathway {idx + 1}",
                "sub_pathway_type": _canonical(raw.get("sub_pathway_type") or raw.get("class")),
                "reference_pathway_id": raw.get("reference_pathway_id"),
                "alttext": _canonical(raw.get("alttext") or raw.get("description")),
            }
        )

    # Protein complex visualizations are explicit IR artifacts. PWML represents
    # complexes through visualizations rather than the standard protein-location
    # model, so these carry the location-like fields needed downstream.
    complex_locs = [loc for loc in ir["locations"] if loc.get("type") == "protein_complex_location"]
    for loc in complex_locs:
        ir["protein_complex_visualizations"].append(
            {
                "key": keys.next("pcv"),
                "entity_key": loc["entity_key"],
                "biological_state_key": loc["biological_state_key"],
                "x": int(loc.get("x") or 0),
                "y": int(loc.get("y") or 0),
                "zindex": int(loc.get("zindex") or 10),
                "hidden": bool(loc.get("hidden", False)),
                "visualization_template_id": int(loc.get("visualization_template_id") or 2),
                "location_key": loc["key"],
                "protein_member_location_keys": [],
                "compound_member_location_keys": [],
            }
        )

    _emit_canonicalization_preflight(ir, report)

    validation = validate_pwml_ir(ir)
    report["ir_validation"] = validation
    report["counts"] = {
        "species": len(ir["species"]),
        "subcellular_locations": len(ir["subcellular_locations"]),
        "biological_states": len(ir["biological_states"]),
        "compounds": len(ir["entities"]["compounds"]),
        "proteins": len(ir["entities"]["proteins"]),
        "reactions": len(ir["processes"]["reactions"]),
        "transports": len(ir["processes"]["transports"]),
        "locations": len(ir["locations"]),
        "edges": len(ir["edges"]),
        "process_visualizations": len(ir["process_visualizations"]),
    }
    if validation.get("errors"):
        for err in validation["errors"]:
            _add_issue(
                report,
                "error",
                str(err.get("code", "ir_validation_error")),
                str(err.get("message", "PWML IR validation error.")),
                pointer=str(err.get("pointer", "")),
            )
    return ir, report


def validate_required_pwml_contract(payload_or_ir: Any, *, strict_db: bool = True) -> Dict[str, Any]:
    """
    Pre-export contract validator for the hydrated payload or built IR.

    Checks every required field defined by the PWML-ready contract without
    mutating the input or making any DB calls. Returns a structured report
    with ``errors``, ``warnings``, ``summary``, and ``ok``.
    """
    errors: List[Dict[str, Any]] = []
    warnings: List[Dict[str, Any]] = []
    error_keys: set[Tuple[str, str]] = set()

    def err(code: str, message: str, pointer: str = "", **extra: Any) -> None:
        issue_key = (code, pointer or message)
        if issue_key in error_keys:
            return
        error_keys.add(issue_key)
        issue: Dict[str, Any] = {"code": code, "message": message}
        if pointer:
            issue["pointer"] = pointer
        issue.update(extra)
        errors.append(issue)

    def warn(code: str, message: str, pointer: str = "", **extra: Any) -> None:
        issue: Dict[str, Any] = {"code": code, "message": message}
        if pointer:
            issue["pointer"] = pointer
        issue.update(extra)
        warnings.append(issue)

    if not isinstance(payload_or_ir, dict):
        err("invalid_input", "Input must be a dict (payload or IR).")
        return {"ok": False, "errors": errors, "warnings": warnings, "summary": {}}

    is_ir = is_pwml_ir(payload_or_ir)

    def db_id(row: Any, keys: Sequence[str]) -> Optional[int]:
        if not isinstance(row, dict):
            return None
        return _db_id(row, keys)

    def resolution_status(row: Any) -> str:
        if not isinstance(row, dict):
            return ""
        status = _safe_dict(row.get("resolution")).get("status")
        if not status:
            status = _safe_dict(_safe_dict(row.get("mapping_meta")).get("resolution")).get("status")
        return _canonical(status).casefold()

    def require_db_identity(
        row: Dict[str, Any],
        keys: Sequence[str],
        *,
        code: str,
        message: str,
        pointer: str,
        **extra: Any,
    ) -> None:
        if not strict_db:
            return
        if db_id(row, keys) is not None:
            return
        err(
            code,
            message,
            pointer,
            required_fields=list(keys),
            resolution_status=resolution_status(row),
            **extra,
        )

    def component_index(bucket: str, keys: Sequence[str]) -> Dict[str, Dict[str, Any]]:
        out: Dict[str, Dict[str, Any]] = {}
        rows = (
            _safe_list(payload_or_ir.get(bucket))
            if is_ir
            else _safe_list(_safe_dict(payload_or_ir.get("entities")).get(bucket))
        )
        for row in rows:
            if not isinstance(row, dict):
                continue
            for value in [row.get("key"), row.get("name"), row.get("display_name")]:
                norm = _norm(value)
                if norm:
                    out[norm] = row
            ident = db_id(row, keys)
            if ident is not None:
                out[str(ident)] = row
        return out

    def protein_external_id(row: Dict[str, Any]) -> str:
        value = _first_nonempty(
            row,
            [
                "uniprot",
                "uniprot_id",
                "uniprot-id",
                "drugbank",
                "drugbank_id",
                "drugbank-id",
            ],
        )
        return _canonical(value)

    # ── PATHWAY ──────────────────────────────────────────────────────────────
    if is_ir:
        pathway = _safe_dict(payload_or_ir.get("pathway"))
        pw_id = pathway.get("pw_id") or pathway.get("pathbank_pathway_id")
        name = pathway.get("name")
        subject = pathway.get("subject")
        width = pathway.get("width")
        height = pathway.get("height")
        base = "/pathway"
    else:
        meta = _safe_dict(payload_or_ir.get("metadata"))
        pw_id = meta.get("pw_id") or meta.get("pathbank_pathway_id") or meta.get("pathway_id")
        name = meta.get("pathway_name") or meta.get("name")
        subject = meta.get("pathway_subject") or meta.get("subject")
        width = meta.get("width")
        height = meta.get("height")
        base = "/metadata"

    if not pw_id:
        warn("pathway_missing_pw_id", "Pathway has no pw_id / pathbank_pathway_id.", f"{base}/pw_id")
    if not name:
        err("pathway_missing_name", "Pathway is missing a name.", f"{base}/name")
    if not subject:
        err("pathway_missing_subject", "Pathway is missing a subject.", f"{base}/subject")
    if is_ir:
        if not width:
            err("pathway_missing_width", "Pathway IR is missing width.", f"{base}/width")
        if not height:
            err("pathway_missing_height", "Pathway IR is missing height.", f"{base}/height")
    else:
        if not width:
            warn("pathway_missing_width", "Payload metadata has no explicit width (will default at IR build time).", f"{base}/width")
        if not height:
            warn("pathway_missing_height", "Payload metadata has no explicit height (will default at IR build time).", f"{base}/height")

    # ── ENTITY REGISTRIES ─────────────────────────────────────────────────────
    entities = _safe_dict(payload_or_ir.get("entities"))
    all_entity_names: set = set()
    all_entity_keys: set = set()
    entity_type_by_key: Dict[str, str] = {}
    protein_names: set = set()
    proteins_by_key: Dict[str, Dict[str, Any]] = {}
    proteins_by_name: Dict[str, Dict[str, Any]] = {}
    proteins_by_uniprot: Dict[str, Dict[str, Any]] = {}
    proteins_by_drugbank: Dict[str, Dict[str, Any]] = {}
    proteins_by_pathbank_id: Dict[int, Dict[str, Any]] = {}
    protein_complex_names: set = set()

    for bucket in ENTITY_BUCKETS.values():
        for e in _safe_list(entities.get(bucket)):
            if isinstance(e, dict):
                n = _canonical(e.get("name"))
                if n:
                    all_entity_names.add(_norm(n))
                    if bucket == "proteins":
                        protein_names.add(_norm(n))
                    if bucket == "protein_complexes":
                        protein_complex_names.add(_norm(n))
                k = e.get("key")
                if k:
                    all_entity_keys.add(k)
                    entity_type_by_key[str(k)] = next(
                        (entity_type for entity_type, entity_bucket in ENTITY_BUCKETS.items() if entity_bucket == bucket),
                        "",
                    )
                if bucket == "proteins":
                    if k:
                        proteins_by_key[str(k)] = e
                    if n:
                        proteins_by_name[_norm(n)] = e
                    uniprot = _canonical(_first_nonempty(e, ["uniprot", "uniprot_id", "uniprot-id"])).casefold()
                    if uniprot:
                        proteins_by_uniprot[uniprot] = e
                    drugbank = _canonical(_first_nonempty(e, ["drugbank", "drugbank_id", "drugbank-id"])).casefold()
                    if drugbank:
                        proteins_by_drugbank[drugbank] = e
                    pathbank_id = db_id(e, ["pathbank_protein_id", "pw_protein_id", "pathwhiz_id", "protein_id"])
                    if pathbank_id is not None:
                        proteins_by_pathbank_id[pathbank_id] = e

    # ── PROTEINS ─────────────────────────────────────────────────────────────
    for idx, comp in enumerate(_safe_list(entities.get("compounds"))):
        if not isinstance(comp, dict):
            continue
        cname = _canonical(comp.get("name"))
        if not cname:
            err(
                "compound_missing_name",
                "Compound is missing a name.",
                f"/entities/compounds/{idx}/name",
            )

    for idx, prot in enumerate(_safe_list(entities.get("proteins"))):
        if not isinstance(prot, dict):
            continue
        pname = _canonical(prot.get("name"))
        if not pname:
            err(
                "protein_missing_name",
                "Protein is missing a name.",
                f"/entities/proteins/{idx}/name",
            )
        species = _entity_species_context(prot)
        if not species:
            err(
                "protein_missing_species",
                f"Protein '{pname}' is missing species/organism.",
                f"/entities/proteins/{idx}",
                entity_name=pname,
            )
        if not _protein_external_id(prot):
            err(
                "protein_missing_external_identity",
                f"Protein '{pname}' is missing a UniProt or DrugBank identifier.",
                f"/entities/proteins/{idx}",
                entity_name=pname,
                required_fields=["uniprot", "uniprot_id", "drugbank", "drugbank_id"],
            )

    # ── PROTEIN COMPLEXES ────────────────────────────────────────────────────
    for idx, pc in enumerate(_safe_list(entities.get("protein_complexes"))):
        if not isinstance(pc, dict):
            continue
        pcname = _canonical(pc.get("name"))
        species = _entity_species_context(pc)
        if not species:
            err(
                "protein_complex_missing_species",
                f"Protein complex '{pcname}' is missing species.",
                f"/entities/protein_complexes/{idx}",
                entity_name=pcname,
            )
        generated_complex = _is_generated_complex_row(pc)
        components = _safe_list(pc.get("components"))
        if not components:
            (err if generated_complex else warn)(
                "protein_complex_missing_components",
                f"Protein complex '{pcname}' has no protein components.",
                f"/entities/protein_complexes/{idx}/components",
                entity_name=pcname,
            )
        else:
            for cidx, comp in enumerate(components):
                pointer = f"/entities/protein_complexes/{idx}/components/{cidx}"
                protein_key = ""
                if isinstance(comp, dict):
                    protein_key = _canonical(comp.get("protein_key") or comp.get("entity_key"))
                comp_name = _component_protein_name(comp)
                stoich = _component_stoichiometry(comp)
                if stoich is None:
                    # PathWhiz's protein_complex_proteins.stoichiometry is nullable and
                    # validated with allow_nil, so an unstated coefficient is exportable.
                    # Papers routinely omit subunit counts; never block export on it.
                    warn(
                        "component_stoichiometry_unstated",
                        f"Component[{cidx}] in complex '{pcname}' has no stated stoichiometry; exporting it blank.",
                        pointer,
                        complex_name=pcname,
                        component_name=comp_name,
                    )
                matched_protein: Optional[Dict[str, Any]] = None
                if protein_key:
                    matched_protein = proteins_by_key.get(protein_key)
                    if matched_protein is None:
                        (err if generated_complex else warn)(
                            "component_protein_unresolved",
                            f"Component '{protein_key}' in complex '{pcname}' does not reference an existing protein.",
                            pointer,
                            complex_name=pcname,
                            protein_key=protein_key,
                        )
                elif comp_name:
                    matched_protein = proteins_by_name.get(_norm(comp_name))
                    if isinstance(comp, dict):
                        comp_uniprot = _canonical(
                            _first_nonempty(comp, ["uniprot", "uniprot_id", "uniprot-id"])
                        ).casefold()
                        comp_drugbank = _canonical(
                            _first_nonempty(comp, ["drugbank", "drugbank_id", "drugbank-id"])
                        ).casefold()
                        comp_pathbank_id = db_id(
                            comp,
                            ["pathbank_protein_id", "pw_protein_id", "pathwhiz_id", "protein_id"],
                        )
                        if matched_protein is None and comp_uniprot:
                            matched_protein = proteins_by_uniprot.get(comp_uniprot)
                        if matched_protein is None and comp_drugbank:
                            matched_protein = proteins_by_drugbank.get(comp_drugbank)
                        if matched_protein is None and comp_pathbank_id is not None:
                            matched_protein = proteins_by_pathbank_id.get(comp_pathbank_id)
                    if matched_protein is None:
                        (err if generated_complex else warn)(
                            "component_protein_unresolved",
                            f"Component '{comp_name}' in complex '{pcname}' does not reference an existing protein.",
                            pointer,
                            complex_name=pcname,
                            component_name=comp_name,
                        )
                else:
                    (err if generated_complex else warn)(
                        "component_protein_unresolved",
                        f"Component[{cidx}] in complex '{pcname}' does not reference an existing protein.",
                        pointer,
                        complex_name=pcname,
                    )
                if generated_complex and matched_protein is not None:
                    if not _entity_species_context(matched_protein):
                        err(
                            "generated_complex_component_missing_species",
                            f"Generated protein complex '{pcname}' component protein '{matched_protein.get('name')}' is missing species/organism.",
                            pointer,
                            complex_name=pcname,
                            component_name=matched_protein.get("name"),
                        )
                    if not _protein_external_id(matched_protein):
                        err(
                            "generated_complex_component_missing_external_identity",
                            f"Generated protein complex '{pcname}' component protein '{matched_protein.get('name')}' is missing a UniProt or DrugBank identifier.",
                            pointer,
                            complex_name=pcname,
                            component_name=matched_protein.get("name"),
                            required_fields=["uniprot", "uniprot_id", "drugbank", "drugbank_id"],
                        )

    # ── SPECIES ──────────────────────────────────────────────────────────────
    # A species with no reference-DB identity will be created fresh in Rails,
    # which requires a numeric taxonomy-id and a Prokaryote/Eukaryote
    # classification. Missing either is fatal under strict_db so the run stops
    # with the organism named — instead of failing opaquely at PWML upload when
    # Species#save rolls back and cascades into the pathway.
    species_rows = (
        _safe_list(payload_or_ir.get("species"))
        if is_ir
        else _safe_list(_safe_dict(payload_or_ir.get("entities")).get("species"))
    )
    for idx, sp in enumerate(species_rows):
        if not isinstance(sp, dict):
            continue
        if not strict_db:
            continue
        if db_id(sp, ["pathbank_species_id", "pw_species_id", "pathwhiz_id", "species_id"]) is not None:
            continue
        sp_name = _canonical(sp.get("name") or sp.get("display_name") or f"species[{idx}]")
        pointer = f"/species/{idx}" if is_ir else f"/entities/species/{idx}"
        taxonomy_id = _canonical(str(sp.get("taxonomy_id") or sp.get("taxonomy-id") or ""))
        classification = _canonical(str(sp.get("classification") or ""))
        if not (taxonomy_id.isdigit() and int(taxonomy_id) > 0):
            err(
                "species_missing_taxonomy",
                f"No taxonomy ID found for species '{sp_name}' — Rails cannot create it. "
                f"Resolve the organism against NCBI Taxonomy or supply a numeric taxonomy_id.",
                pointer,
                species_name=sp_name,
            )
        if classification not in ("Prokaryote", "Eukaryote"):
            err(
                "species_missing_classification",
                f"Species '{sp_name}' has no valid classification "
                f"(expected 'Prokaryote' or 'Eukaryote').",
                pointer,
                species_name=sp_name,
            )

    # ── BIOLOGICAL STATES ────────────────────────────────────────────────────
    bio_states = _safe_list(payload_or_ir.get("biological_states"))
    if not bio_states:
        err("no_biological_states", "No biological_states defined; export requires an explicit state.")
    for idx, state in enumerate(bio_states):
        if not isinstance(state, dict):
            continue
        sname = state.get("name", f"state[{idx}]")
        if is_ir:
            has_species = bool(state.get("species_key"))
            has_subcell = bool(state.get("subcellular_location_key"))
            species_ref = state.get("species_key")
            subcell_ref = state.get("subcellular_location_key")
            has_other_component = bool(state.get("tissue_key") or state.get("cell_type_key"))
        else:
            species_ref = state.get("species") or state.get("organism") or state.get("taxonomy_id")
            subcell_ref = (
                state.get("subcellular_location")
                or state.get("subcellular-location")
                or state.get("compartment")
                or state.get("compartment_canonical")
            )
            has_species = bool(
                species_ref
                or state.get("species_id")
                or state.get("pathbank_species_id")
            )
            has_subcell = bool(
                subcell_ref
                or state.get("subcellular_location_id")
                or state.get("pathbank_subcellular_location_id")
            )
            has_other_component = bool(
                state.get("tissue")
                or state.get("tissue_id")
                or state.get("cell_type")
                or state.get("cell_type_id")
            )
        # PathWhiz's BiologicalState#has_at_least_one_component requires one of
        # species / tissue / cell_type / subcellular_location — not every one of
        # them. A state carrying any of the four is exportable, so only a fully
        # empty state is fatal; the individual gaps stay visible as warnings.
        if not (has_species or has_subcell or has_other_component):
            err(
                "biological_state_missing_components",
                f"Biological state '{sname}' has no species, tissue, cell type, or subcellular location.",
                f"/biological_states/{idx}",
                state_name=sname,
            )
        else:
            if not has_species:
                warn(
                    "biological_state_missing_species",
                    f"Biological state '{sname}' is missing species.",
                    f"/biological_states/{idx}",
                    state_name=sname,
                )
            if not has_subcell:
                warn(
                    "biological_state_missing_subcellular_location",
                    f"Biological state '{sname}' has no subcellular location.",
                    f"/biological_states/{idx}",
                    state_name=sname,
                )

    # ── REACTIONS ─────────────────────────────────────────────────────────────
    processes = _safe_dict(payload_or_ir.get("processes"))
    for idx, rx in enumerate(_safe_list(processes.get("reactions"))):
        if not isinstance(rx, dict):
            continue
        rname = rx.get("name", f"reaction[{idx}]")

        if is_ir:
            left = _safe_list(rx.get("left"))
            right = _safe_list(rx.get("right"))
            left_key, right_key = "left", "right"
        else:
            left = _safe_list(rx.get("inputs") or rx.get("left") or rx.get("substrates"))
            right = _safe_list(rx.get("outputs") or rx.get("right") or rx.get("products"))
            left_key = "inputs" if rx.get("inputs") else "left"
            right_key = "outputs" if rx.get("outputs") else "right"

        if not left:
            err(
                "reaction_missing_left_participants",
                f"Reaction '{rname}' has no left/input participants.",
                f"/processes/reactions/{idx}/{left_key}",
                reaction_name=rname,
            )
        if not right:
            err(
                "reaction_missing_right_participants",
                f"Reaction '{rname}' has no right/output participants.",
                f"/processes/reactions/{idx}/{right_key}",
                reaction_name=rname,
            )

        for side_label, side in [(left_key, left), (right_key, right)]:
            for midx, part in enumerate(side):
                if isinstance(part, str):
                    warn(
                        "reaction_participant_missing_stoichiometry",
                        f"Reaction '{rname}' {side_label}[{midx}] is a bare string with no explicit stoichiometry (defaults to 1).",
                        f"/processes/reactions/{idx}/{side_label}/{midx}",
                        reaction_name=rname,
                    )
                    continue
                if not isinstance(part, dict):
                    continue
                stoich = part.get("stoichiometry") or part.get("coefficient")
                if stoich is None:
                    warn(
                        "reaction_participant_missing_stoichiometry",
                        f"Reaction '{rname}' {side_label}[{midx}] has no explicit stoichiometry (defaults to 1).",
                        f"/processes/reactions/{idx}/{side_label}/{midx}",
                        reaction_name=rname,
                    )

        # Enzyme references against entity registry.  Stage 6 is the sole
        # wrapper-creation boundary, so every surviving protein catalyst must
        # already reference a declared protein_complex here.
        enzyme_field = "enzymes" if rx.get("enzymes") is not None else "catalysts"
        enzyme_raw = _safe_list(rx.get(enzyme_field))

        actor_sources: List[Tuple[Any, str, bool]] = [
            (actor, f"/processes/reactions/{idx}/{enzyme_field}/{eidx}", False)
            for eidx, actor in enumerate(enzyme_raw)
        ]
        if not is_ir:
            catalytic_modifier_roles = {"catalyst", "activator", "inhibitor", "enzyme"}
            for midx, modifier in enumerate(_safe_list(rx.get("modifiers"))):
                if not isinstance(modifier, dict):
                    continue
                role = _canonical(modifier.get("role")).casefold()
                if role in catalytic_modifier_roles:
                    actor_sources.append(
                        (modifier, f"/processes/reactions/{idx}/modifiers/{midx}", True)
                    )

        # ``modifiers`` is the canonical actor mirror retained alongside
        # ``enzymes`` by Stage 6.  Repeating an actor within either collection
        # is invalid, but the same actor once in each collection represents one
        # logical catalyst and is not an intra-field duplicate.
        seen_enzyme_complexes_by_source: Dict[bool, set[str]] = {
            False: set(),
            True: set(),
        }
        for actor, actor_pointer, is_modifier in actor_sources:
            if is_ir:
                if not isinstance(actor, dict):
                    continue
                entity_key = _canonical(actor.get("entity_key"))
                declared_type = _canonical(actor.get("entity_type") or actor.get("type")).casefold()
                referenced_type = entity_type_by_key.get(entity_key, "")
                if entity_key and entity_key not in all_entity_keys:
                    err(
                        "enzyme_reference_not_found",
                        f"Reaction '{rname}' enzyme entity_key '{entity_key}' not found in entities.",
                        actor_pointer,
                        reaction_name=rname,
                        entity_key=entity_key,
                    )
                if declared_type != "protein_complex" or referenced_type not in {"", "protein_complex"}:
                    err(
                        "reaction_enzyme_must_be_protein_complex",
                        f"Reaction '{rname}' enzyme '{entity_key or '<missing>'}' must reference a protein_complex entity.",
                        actor_pointer,
                        reaction_name=rname,
                        entity_key=entity_key,
                        entity_type=referenced_type or declared_type,
                    )
                continue

            if isinstance(actor, str):
                enzyme_name = _canonical(actor)
                declared_type = ""
            elif isinstance(actor, dict):
                enzyme_name, field_hint, _ = _actor_name_and_hint(actor)
                declared_type = _canonical(actor.get("entity_type") or actor.get("type") or field_hint).casefold()
            else:
                continue

            enzyme_norm = _norm(enzyme_name)
            resolved_type = ""
            if declared_type == "protein_complex" and enzyme_norm in protein_complex_names:
                resolved_type = "protein_complex"
            elif declared_type == "protein" and enzyme_norm in protein_names:
                resolved_type = "protein"
            elif not declared_type:
                if enzyme_norm in protein_complex_names:
                    resolved_type = "protein_complex"
                elif enzyme_norm in protein_names:
                    resolved_type = "protein"

            # Non-protein catalytic modifiers are cofactors/effectors and are
            # intentionally dropped by IR construction; they do not need a
            # protein-complex wrapper.  Explicit enzyme rows have no exemption.
            proteinish_modifier = bool(
                declared_type in {"protein", "protein_complex"}
                or resolved_type in {"protein", "protein_complex"}
                or enzyme_norm in protein_names
                or enzyme_norm in protein_complex_names
            )
            if is_modifier and not proteinish_modifier:
                continue

            if resolved_type != "protein_complex":
                err(
                    "reaction_enzyme_must_be_protein_complex",
                    f"Reaction '{rname}' enzyme/modifier '{enzyme_name}' must reference a protein_complex entity.",
                    actor_pointer,
                    reaction_name=rname,
                    enzyme_name=enzyme_name,
                    entity_type=resolved_type or declared_type,
                )

            if declared_type == "protein_complex" and enzyme_name and enzyme_norm not in protein_complex_names:
                err(
                    "enzyme_reference_not_found",
                    f"Reaction '{rname}' enzyme protein complex '{enzyme_name}' not found in entity registries.",
                    actor_pointer,
                    reaction_name=rname,
                    enzyme_name=enzyme_name,
                )
            elif declared_type == "protein" and enzyme_name and enzyme_norm not in protein_names:
                err(
                    "enzyme_reference_not_found",
                    f"Reaction '{rname}' enzyme protein '{enzyme_name}' not found in entity registries.",
                    actor_pointer,
                    reaction_name=rname,
                    enzyme_name=enzyme_name,
                )
            elif enzyme_name and enzyme_norm not in all_entity_names:
                err(
                    "enzyme_reference_not_found",
                    f"Reaction '{rname}' enzyme '{enzyme_name}' not found in entity registries.",
                    actor_pointer,
                    reaction_name=rname,
                    enzyme_name=enzyme_name,
                )

            if resolved_type == "protein_complex":
                seen_enzyme_complexes = seen_enzyme_complexes_by_source[is_modifier]
                if enzyme_norm in seen_enzyme_complexes:
                    err(
                        "duplicate_reaction_enzyme_complex",
                        f"Reaction '{rname}' assigns protein complex '{enzyme_name}' more than once.",
                        actor_pointer,
                        reaction_name=rname,
                        enzyme_name=enzyme_name,
                    )
                seen_enzyme_complexes.add(enzyme_norm)

    # ── LOCATION / VISUALIZATION REFERENCES (IR only) ────────────────────────
    if not is_ir:
        element_locations = _safe_dict(payload_or_ir.get("element_locations"))
        location_fields = {
            "compound_locations": "compound",
            "protein_locations": "protein",
            "nucleic_acid_locations": "nucleic_acid",
            "element_collection_locations": "element_collection",
        }
        for bucket, name_field in location_fields.items():
            rows = _safe_list(element_locations.get(bucket))
            kept_locations: List[Any] = []
            for lidx, loc in enumerate(rows):
                if not isinstance(loc, dict):
                    kept_locations.append(loc)
                    continue
                entity_name = _canonical(loc.get(name_field) or loc.get("entity"))
                if entity_name and _norm(entity_name) not in all_entity_names:
                    warn(
                        "location_entity_not_found",
                        f"Location for '{entity_name}' does not reference an existing entity.",
                        f"/element_locations/{bucket}/{lidx}/{name_field}",
                        entity_name=entity_name,
                    )
                    continue
                kept_locations.append(loc)
                if not loc.get("biological_state"):
                    err(
                        "visible_entity_missing_location_state",
                        f"Visible entity '{entity_name or lidx}' has no biological_state location.",
                        f"/element_locations/{bucket}/{lidx}/biological_state",
                        entity_name=entity_name,
                    )
            if len(kept_locations) != len(rows):
                element_locations[bucket] = kept_locations

    if is_ir:
        bio_state_keys = {
            s.get("key")
            for s in _safe_list(payload_or_ir.get("biological_states"))
            if isinstance(s, dict)
        }
        process_keys: set = set()
        for bucket in PROCESS_BUCKETS.values():
            for p in _safe_list(_safe_dict(payload_or_ir.get("processes")).get(bucket)):
                if isinstance(p, dict) and p.get("key"):
                    process_keys.add(p["key"])
        location_keys = {
            loc.get("key")
            for loc in _safe_list(payload_or_ir.get("locations"))
            if isinstance(loc, dict)
        }

        for idx, loc in enumerate(_safe_list(payload_or_ir.get("locations"))):
            if not isinstance(loc, dict):
                continue
            if loc.get("entity_key") not in all_entity_keys:
                err(
                    "location_missing_entity",
                    f"Location '{loc.get('key', idx)}' references unknown entity_key '{loc.get('entity_key')}'.",
                    f"/locations/{idx}",
                )
            if loc.get("biological_state_key") not in bio_state_keys:
                err(
                    "location_missing_biological_state",
                    f"Location '{loc.get('key', idx)}' references unknown biological_state_key '{loc.get('biological_state_key')}'.",
                    f"/locations/{idx}",
                )

        for idx, viz in enumerate(_safe_list(payload_or_ir.get("process_visualizations"))):
            if not isinstance(viz, dict):
                continue
            if viz.get("process_key") not in process_keys:
                err(
                    "visualization_missing_process",
                    f"Process visualization '{viz.get('key', idx)}' references unknown process_key '{viz.get('process_key')}'.",
                    f"/process_visualizations/{idx}",
                )
            if viz.get("biological_state_key") and viz.get("biological_state_key") not in bio_state_keys:
                err(
                    "visualization_missing_biological_state",
                    f"Process visualization '{viz.get('key', idx)}' references unknown biological_state_key.",
                    f"/process_visualizations/{idx}",
                )
            for midx, member in enumerate(_safe_list(viz.get("members"))):
                if not isinstance(member, dict):
                    continue
                if member.get("location_key") and member.get("location_key") not in location_keys:
                    err(
                        "visualization_member_missing_location",
                        f"Visualization member references unknown location_key '{member.get('location_key')}'.",
                        f"/process_visualizations/{idx}/members/{midx}",
                    )

    error_codes = sorted({e["code"] for e in errors})
    warning_codes = sorted({w["code"] for w in warnings})
    return {
        "ok": not errors,
        "errors": errors,
        "warnings": warnings,
        "summary": {
            "error_count": len(errors),
            "warning_count": len(warnings),
            "error_codes": error_codes,
            "warning_codes": warning_codes,
            "checked_as": "ir" if is_ir else "payload",
            "strict_db": bool(strict_db),
        },
    }


def validate_pwml_ir(ir: Dict[str, Any]) -> Dict[str, Any]:
    errors: List[Dict[str, Any]] = []
    warnings: List[Dict[str, Any]] = []

    def error(code: str, message: str, pointer: str = "", **extra: Any) -> None:
        issue = {"code": code, "message": message}
        if pointer:
            issue["pointer"] = pointer
        issue.update(extra)
        errors.append(issue)

    def warning(code: str, message: str, pointer: str = "", **extra: Any) -> None:
        issue = {"code": code, "message": message}
        if pointer:
            issue["pointer"] = pointer
        issue.update(extra)
        warnings.append(issue)

    if not is_pwml_ir(ir):
        error("invalid_ir_shape", "Input does not look like a PWML IR object.", "/")
        return {"ok": False, "errors": errors, "warnings": warnings, "counts": {}}

    entity_keys_by_type: Dict[str, set] = {entity_type: set() for entity_type in ENTITY_BUCKETS}
    all_entity_keys: set = set()
    for entity_type, bucket in ENTITY_BUCKETS.items():
        for idx, record in enumerate(_safe_list(_safe_dict(ir.get("entities")).get(bucket))):
            if not isinstance(record, dict):
                continue
            key = record.get("key")
            if not key:
                error("missing_entity_key", f"{bucket}[{idx}] is missing key.", f"/entities/{bucket}/{idx}")
                continue
            if key in all_entity_keys:
                error("duplicate_entity_key", f"Duplicate entity key '{key}'.", f"/entities/{bucket}/{idx}")
            entity_keys_by_type[entity_type].add(key)
            all_entity_keys.add(key)

    for idx, record in enumerate(_safe_list(_safe_dict(ir.get("entities")).get("protein_complexes"))):
        if not isinstance(record, dict):
            continue
        components = _safe_list(record.get("components"))
        if not components:
            # A complex with a real, confirmed PathBank complex-level identity
            # may legitimately have zero listed subunits -- reference exports
            # (e.g. reference/PW1.pwml's "alanine aminotransferase (ALT)"
            # complex, pwp-id PW_P000036) carry exactly this shape. Only a
            # complex with no real identity of its own -- which, after Stage
            # 6's PathBank Unknown-sentinel fallback runs, should only be a
            # pipeline-generated wrapper that failed to get even that
            # fallback -- is required to have at least one member.
            has_real_identity = bool(_to_int(record.get("pathbank_complex_id")) or _to_int(record.get("pathwhiz_id")))
            (warning if has_real_identity else error)(
                "protein_complex_missing_components",
                f"Protein complex '{record.get('name') or idx}' has no protein components.",
                f"/entities/protein_complexes/{idx}/components",
            )
            continue
        for cidx, component in enumerate(components):
            pointer = f"/entities/protein_complexes/{idx}/components/{cidx}"
            if not isinstance(component, dict):
                error(
                    "protein_complex_component_not_structured",
                    "Protein complex component must be a structured record.",
                    pointer,
                )
                continue
            protein_key = component.get("protein_key")
            if protein_key not in entity_keys_by_type["protein"]:
                warning(
                    "component_protein_unresolved",
                    f"Protein complex component references unknown protein_key '{protein_key}'.",
                    pointer,
                    protein_key=protein_key,
                )
            if "stoichiometry" in component and _component_stoichiometry(component) is None:
                # A present-but-unusable value is a real defect; an absent one is
                # simply an unstated coefficient, which PathWhiz accepts as nil.
                warning(
                    "component_stoichiometry_unusable",
                    "Protein complex component has a stoichiometry that is not a positive integer; exporting it blank.",
                    pointer,
                    protein_key=protein_key,
                )

    biological_state_keys = {
        state.get("key")
        for state in _safe_list(ir.get("biological_states"))
        if isinstance(state, dict) and state.get("key")
    }
    for idx, state in enumerate(_safe_list(ir.get("biological_states"))):
        if not isinstance(state, dict):
            continue
        pointer = f"/biological_states/{idx}"
        if not state.get("key"):
            error("missing_biological_state_key", f"biological_states[{idx}] is missing key.", pointer)
        if not state.get("species_key"):
            error(
                "biological_state_missing_species",
                f"Biological state '{state.get('name') or idx}' is missing species_key.",
                pointer,
            )
        if not state.get("subcellular_location_key"):
            error(
                "biological_state_missing_subcellular_location",
                f"Biological state '{state.get('name') or idx}' is missing subcellular_location_key.",
                pointer,
            )
    location_keys = {
        loc.get("key")
        for loc in _safe_list(ir.get("locations"))
        if isinstance(loc, dict) and loc.get("key")
    }
    edge_keys = {
        edge.get("key")
        for edge in _safe_list(ir.get("edges"))
        if isinstance(edge, dict) and edge.get("key")
    }

    for idx, loc in enumerate(_safe_list(ir.get("locations"))):
        if not isinstance(loc, dict):
            continue
        ekey = loc.get("entity_key")
        bs_key = loc.get("biological_state_key")
        if ekey not in all_entity_keys:
            error("unresolved_location_entity", f"Location references unknown entity_key '{ekey}'.", f"/locations/{idx}")
        if bs_key not in biological_state_keys:
            error(
                "unresolved_location_biological_state",
                f"Location references unknown biological_state_key '{bs_key}'.",
                f"/locations/{idx}",
            )

    protein_complex_viz_by_entity_state: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for idx, viz in enumerate(_safe_list(ir.get("protein_complex_visualizations"))):
        if not isinstance(viz, dict):
            continue
        pointer = f"/protein_complex_visualizations/{idx}"
        entity_key = viz.get("entity_key")
        bs_key = viz.get("biological_state_key")
        if entity_key not in entity_keys_by_type["protein_complex"]:
            error(
                "protein_complex_visualization_unknown_entity",
                f"Protein complex visualization references unknown entity_key '{entity_key}'.",
                pointer,
                entity_key=entity_key,
            )
        if bs_key not in biological_state_keys:
            error(
                "protein_complex_visualization_unknown_biological_state",
                f"Protein complex visualization references unknown biological_state_key '{bs_key}'.",
                pointer,
                biological_state_key=bs_key,
            )
        for field in ["x", "y", "zindex", "hidden", "visualization_template_id"]:
            if field not in viz:
                error(
                    "protein_complex_visualization_missing_location_field",
                    f"Protein complex visualization is missing required field '{field}'.",
                    pointer,
                    field=field,
                )
        if entity_key and bs_key:
            protein_complex_viz_by_entity_state[(str(entity_key), str(bs_key))] = viz

    process_keys: set = set()
    process_member_keys: set = set()
    process_bucket_by_key: Dict[str, str] = {}
    process_members_by_key: Dict[str, Dict[str, Any]] = {}
    processes = _safe_dict(ir.get("processes"))

    def validate_member(member: Dict[str, Any], pointer: str) -> None:
        mkey = member.get("key")
        entity_type = member.get("entity_type")
        entity_key = member.get("entity_key")
        if not mkey:
            error("missing_process_member_key", "Process member is missing key.", pointer)
        elif mkey in process_member_keys:
            error("duplicate_process_member_key", f"Duplicate process member key '{mkey}'.", pointer)
        else:
            process_member_keys.add(mkey)
            process_members_by_key[mkey] = member
        if entity_type not in ENTITY_BUCKETS:
            error("invalid_member_entity_type", f"Invalid member entity_type '{entity_type}'.", pointer)
        elif entity_key not in entity_keys_by_type[entity_type]:
            error(
                "unresolved_member_entity",
                f"Process member references unknown {entity_type} entity_key '{entity_key}'.",
                pointer,
            )

    for idx, reaction in enumerate(_safe_list(processes.get("reactions"))):
        if not isinstance(reaction, dict):
            continue
        key = reaction.get("key")
        if key:
            process_keys.add(key)
            process_bucket_by_key[key] = "reactions"
        if reaction.get("biological_state_key") not in biological_state_keys:
            error("unresolved_reaction_biological_state", "Reaction has unresolved biological_state_key.", f"/processes/reactions/{idx}")
        left = _safe_list(reaction.get("left"))
        right = _safe_list(reaction.get("right"))
        if not left:
            error("reaction_missing_left", "Reaction must have at least one left member.", f"/processes/reactions/{idx}/left")
        if not right:
            error("reaction_missing_right", "Reaction must have at least one right member.", f"/processes/reactions/{idx}/right")
        for midx, member in enumerate(left):
            if isinstance(member, dict):
                validate_member(member, f"/processes/reactions/{idx}/left/{midx}")
        for midx, member in enumerate(right):
            if isinstance(member, dict):
                validate_member(member, f"/processes/reactions/{idx}/right/{midx}")
        for midx, member in enumerate(_safe_list(reaction.get("enzymes"))):
            if isinstance(member, dict):
                validate_member(member, f"/processes/reactions/{idx}/enzymes/{midx}")
                if member.get("entity_type") != "protein_complex":
                    error(
                        "reaction_enzyme_must_be_protein_complex",
                        "Reaction enzyme must reference a protein_complex entity.",
                        f"/processes/reactions/{idx}/enzymes/{midx}",
                        entity_type=member.get("entity_type"),
                    )
                else:
                    complex_viz = protein_complex_viz_by_entity_state.get(
                        (str(member.get("entity_key")), str(reaction.get("biological_state_key")))
                    )
                    if complex_viz is None:
                        error(
                            "reaction_enzyme_missing_complex_visualization",
                            "Reaction enzyme protein_complex has no visible complex visualization in the reaction biological state.",
                            f"/processes/reactions/{idx}/enzymes/{midx}",
                            entity_key=member.get("entity_key"),
                            biological_state_key=reaction.get("biological_state_key"),
                        )
                    elif complex_viz.get("hidden") is True:
                        error(
                            "reaction_enzyme_hidden_complex_visualization",
                            "Reaction enzyme protein_complex visualization is hidden.",
                            f"/processes/reactions/{idx}/enzymes/{midx}",
                            entity_key=member.get("entity_key"),
                            biological_state_key=reaction.get("biological_state_key"),
                        )

    for idx, transport in enumerate(_safe_list(processes.get("transports"))):
        if not isinstance(transport, dict):
            continue
        key = transport.get("key")
        if key:
            process_keys.add(key)
            process_bucket_by_key[key] = "transports"
        elements = _safe_list(transport.get("transport_elements"))
        if not 1 <= len(elements) <= 3:
            error(
                "transport_element_count",
                "Transport must have 1 to 3 transport elements.",
                f"/processes/transports/{idx}/transport_elements",
                count=len(elements),
            )
        for midx, member in enumerate(elements):
            if not isinstance(member, dict):
                continue
            validate_member(member, f"/processes/transports/{idx}/transport_elements/{midx}")
            if member.get("left_biological_state_key") not in biological_state_keys:
                error("transport_missing_left_state", "Transport element has unresolved left biological state.", f"/processes/transports/{idx}/transport_elements/{midx}")
            if member.get("right_biological_state_key") not in biological_state_keys:
                error("transport_missing_right_state", "Transport element has unresolved right biological state.", f"/processes/transports/{idx}/transport_elements/{midx}")
        for midx, member in enumerate(_safe_list(transport.get("transporters"))):
            if isinstance(member, dict):
                validate_member(member, f"/processes/transports/{idx}/transporters/{midx}")

    for idx, rct in enumerate(_safe_list(processes.get("reaction_coupled_transports"))):
        if not isinstance(rct, dict):
            continue
        key = rct.get("key")
        if key:
            process_keys.add(key)
            process_bucket_by_key[key] = "reaction_coupled_transports"
        if not _safe_list(rct.get("left")):
            error("rct_missing_left", "Reaction-coupled transport must have a left member.", f"/processes/reaction_coupled_transports/{idx}/left")
        if not _safe_list(rct.get("right")):
            error("rct_missing_right", "Reaction-coupled transport must have a right member.", f"/processes/reaction_coupled_transports/{idx}/right")
        if not _safe_list(rct.get("enzymes")):
            error("rct_missing_enzyme", "Reaction-coupled transport must have at least one enzyme.", f"/processes/reaction_coupled_transports/{idx}/enzymes")

    for idx, interaction in enumerate(_safe_list(processes.get("interactions"))):
        if not isinstance(interaction, dict):
            continue
        key = interaction.get("key")
        if key:
            process_keys.add(key)
            process_bucket_by_key[key] = "interactions"
        if not interaction.get("interaction_type"):
            error("interaction_missing_type", "Interaction must have interaction_type.", f"/processes/interactions/{idx}")
        left = interaction.get("left")
        right = interaction.get("right")
        if not isinstance(left, dict) or not isinstance(right, dict):
            error("interaction_missing_side", "Interaction must have exactly one left and one right member.", f"/processes/interactions/{idx}")
        else:
            for side_name, side in [("left", left), ("right", right)]:
                entity_type = side.get("entity_type")
                entity_key = side.get("entity_key")
                if entity_type not in ENTITY_BUCKETS or entity_key not in entity_keys_by_type.get(entity_type, set()):
                    error(
                        "unresolved_interaction_entity",
                        f"Interaction {side_name} references unresolved entity.",
                        f"/processes/interactions/{idx}/{side_name}",
                    )

    for idx, subp in enumerate(_safe_list(processes.get("sub_pathways"))):
        if not isinstance(subp, dict):
            continue
        key = subp.get("key")
        if key:
            process_keys.add(key)
            process_bucket_by_key[key] = "sub_pathways"
        if not subp.get("sub_pathway_type"):
            error("sub_pathway_missing_type", "Sub-pathway has no sub_pathway_type.", f"/processes/sub_pathways/{idx}")
        if not subp.get("reference_pathway_id") and not subp.get("alttext"):
            error(
                "sub_pathway_missing_reference",
                "Sub-pathway needs either reference_pathway_id or alttext.",
                f"/processes/sub_pathways/{idx}",
            )

    expected_bucket_by_viz_type = {
        "reaction_visualization": "reactions",
        "reaction_coupled_transport_visualization": "reaction_coupled_transports",
        "transport_visualization": "transports",
        "interaction_visualization": "interactions",
        "sub_pathway_visualization": "sub_pathways",
    }
    for idx, viz in enumerate(_safe_list(ir.get("process_visualizations"))):
        if not isinstance(viz, dict):
            continue
        process_key = viz.get("process_key")
        viz_type = viz.get("type")
        if process_key not in process_keys:
            error("visualization_unknown_process", f"Visualization references unknown process_key '{process_key}'.", f"/process_visualizations/{idx}")
        elif expected_bucket_by_viz_type.get(viz_type) != process_bucket_by_key.get(process_key):
            warning(
                "visualization_type_process_mismatch",
                "Visualization type does not match process bucket.",
                f"/process_visualizations/{idx}",
                visualization_type=viz_type,
                process_bucket=process_bucket_by_key.get(process_key),
            )
        bs_key = viz.get("biological_state_key")
        if bs_key and bs_key not in biological_state_keys:
            error("visualization_unknown_biological_state", f"Visualization references unknown biological_state_key '{bs_key}'.", f"/process_visualizations/{idx}")
        for midx, member in enumerate(_safe_list(viz.get("members"))):
            if not isinstance(member, dict):
                continue
            pointer = f"/process_visualizations/{idx}/members/{midx}"
            member_key = member.get("process_member_key")
            if member_key not in process_member_keys:
                error("visualization_unknown_member", f"Visualization member references unknown process_member_key '{member_key}'.", pointer)
            loc_key = member.get("location_key")
            if loc_key not in location_keys:
                error("visualization_unknown_location", f"Visualization member references unknown location_key '{loc_key}'.", pointer)
            edge_key = member.get("edge_key")
            if edge_key and edge_key not in edge_keys:
                error("visualization_unknown_edge", f"Visualization member references unknown edge_key '{edge_key}'.", pointer)
            if member.get("role") in {"left", "right"} and not edge_key:
                error("visualization_member_missing_edge", "Left/right visualization members require an edge_key.", pointer)

    counts = {
        "entities": len(all_entity_keys),
        "biological_states": len(biological_state_keys),
        "processes": len(process_keys),
        "process_members": len(process_member_keys),
        "locations": len(location_keys),
        "edges": len(edge_keys),
        "process_visualizations": len(_safe_list(ir.get("process_visualizations"))),
    }
    return {"ok": not errors, "errors": errors, "warnings": warnings, "counts": counts}
