"""N-draw ensembling for Stage 1 extraction (ORCH-750 (dispatched as ORCH-750; renumbered because another session had already claimed 748 in the ledger for the pathway-mining lane)). OFF by default.

``docs/evaluation_v2/PAPER_FINDINGS.md`` §§ 5.3, 5.5: single-draw reaction precision is
near-ceiling, recall varies wildly between identical temperature-0 draws (9, then 7, then
4 matched reactions for one prompt on one paper), and each draw misses DIFFERENT
reactions -- so the UNION of 3 draws reached recall 1.000 at precision 1.000 in 3 of 4
measured conditions, with no false row entering any pool. This module turns that
measurement into a setting.

The knob
--------
``T2PW_EXTRACTION_DRAWS``, or the ``draws=`` parameter of
:func:`t2pw.pipeline.pipeline.run_extraction_pipeline`, which overrides it. Default 1.
At 1 nothing in this module runs: ``run_extraction_pipeline`` issues exactly one
``_run_json_stage`` call with exactly the arguments it always has. Production (D-102) is
unchanged until someone sets the variable. Cost and latency scale linearly in N.

Merge semantics
---------------
Union across draws, first occurrence wins, its ``evidence`` kept verbatim. Rows are
deduplicated by the keys the v2 evaluator already uses, so a row this merge keeps as
distinct is a row the evaluator scores as distinct (``reactions.score_leg``, R-D092-2):

* reaction: (set of DEFINING input keys, set of DEFINING output keys). A cofactor --
  ``role: cofactor`` on the participant, or a name in :data:`CURRENCY_KEYS` -- never
  defines a reaction, exactly as the evaluator scores it (``gold.ROLES``; ORCH-751 § 2.2
  measured the exact-set key inflating pooled rows). A side holding nothing but currency
  keeps every name, so ``ADP + Pi -> ATP`` is not ``GDP + Pi -> GTP``. That key only
  PROPOSES a merge (ORCH-754a finding 1, which withdrew "the cofactor-blind key stays"):
  rows sharing it are one reaction only when :func:`_conflict` finds them compatible --
  one row omitting a cofactor, enzyme or state the other names -- and never when they
  conflict (NAD+ vs NADP+, different enzymes, different biological states, or anything it
  cannot classify), within one draw and across draws. A conflicting row stays its own row,
  and the side file says why (``kept_apart_from``). A survivor keeps its own evidence and
  participants, and gains a later row's cofactor only when its OWN quote names it; a
  cofactor it cannot support is listed with the donor's quote (``cofactors_withheld``)
* interaction: (entity_1, relationship, entity_2); the names strict-keyed, the relationship
  by the type the exporter will emit for it (``ir.PWML_INTERACTION_TYPES``: ``activates``
  and ``activation`` are one ``Activation``; a word that table does not know keeps its
  strict key, so ``binds`` stays apart from ``interacts_with``)
* transport: (cargo, from_biological_state, to_biological_state)
* reaction_coupled_transport: (reaction, transport)
* ``element_locations`` rows: (located entity, biological_state), the key
  ``pipeline._dedup_element_locations`` already uses
* any other row with a ``name`` (every entity registry, biological states,
  compartments, visualizations): the strict key of its name
* a row nothing above can key (all key fields blank): its JSON serialisation, so
  distinct unkeyable rows are never collapsed into one

``strict_key`` is :func:`t2pw.bench.eval_v2.matcher.strict_key` (NFKC, Greek spelled
out, casefold, trailing charge dropped, ``-ic acid`` -> ``-ate``, pyrophosphate ==
diphosphate). It is imported inside the function because ``t2pw.pipeline`` must not
import ``t2pw.bench`` at module level (release_status.py:112; strict_quarantine.py:2704
is the precedent seam).

Which draw(s) produced each row goes to the side file ``stage1_ensemble_report.json``
beside the run's other extraction diagnostics, and a bounded summary goes into
``extraction_boundary_report.json`` as a ``stage1_ensemble`` boundary record.

The paper is the dictionary (ORCH-754a, D-104)
----------------------------------------------
ORCH-752 measured that pooled rows inflate because draws write one molecule several ways
(``GSA``, ``Glu 1-semialdehyde (GSA)``, ``porphobilogen``). When the paper text reaches
:func:`merge_draws` (``run_extraction_pipeline`` passes its input text), every draw is
first rewritten by :class:`~t2pw.pipeline.paper_lexicon.PaperLexicon`:

* rule 1, abbreviation: a name the paper defines as an abbreviation -- or a ``X (Y)`` name
  whose halves the paper defines as one pair -- becomes the paper's LONG form;
* rule 2, typo: a name absent from the paper snaps to the paper's spelling when exactly one
  safe candidate exists (safety rules in that module);
* rule 4, long form only: the rewrite reaches every row that names the participant -- the
  reaction sides, the entity registry, element locations, transport cargo, and enzyme or
  interaction endpoints naming it -- so a renamed entity is never left dangling. Enzyme
  names are COMPARED through the paper (finding 1) but written as the draw wrote them. Evidence is never
  edited. A currency name (:data:`CURRENCY_KEYS`) is never rewritten: ``ATP`` stays ``ATP``
  so rule 7's cofactor-blind key still sees it. Every rewrite is listed in the side file
  (``names_rewritten``), so no original spelling is lost.

Rule 5, support, narrowed by the product owner on 2026-10-20: the marks go to HUMAN-FACING
outputs only, because every text the pipeline already produces reaches a model --
``reaction_summary.txt`` feeds the curator and the enrichment agent, and this merge's
payload is pasted whole into the Stage-2 prompt. So the payload gains NO key (the schema is
frozen, as ORCH-750 designed it), and every row of every bucket carries ``support`` and, at
1/N, ``review_required`` in the side file, which :func:`render_report_text` also writes as
``stage1_ensemble_report.txt`` for the hand comparison. Nothing reads either file back. A
1/N row is KEPT -- PRODUCT_CONTRACT § 7: incomplete-but-correct is reviewed, never
dropped. Rule 6: every collapse is listed on its row in the side file with the rule(s) that
bridged it (``exact`` -- which includes the cofactor-blind key -- ``abbreviation``,
``typo``). Rule 3 (database identity) runs after Stage 6 mapping: see
:func:`merge_on_mapped_identity`.
"""

from __future__ import annotations

import json
import logging
import os
from copy import deepcopy
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from t2pw.pipeline.extraction_diagnostics import OUTCOME_OK, current as current_diagnostics
from t2pw.pipeline.paper_lexicon import RULE_DATABASE_ID, RULE_EXACT, PaperLexicon
from t2pw.pipeline.participant_schema import PARTICIPANT_NAME_KEYS

logger = logging.getLogger(__name__)

ENV_VAR = "T2PW_EXTRACTION_DRAWS"
BOUNDARY_STAGE1_ENSEMBLE = "stage1_ensemble"
REPORT_NAME = "stage1_ensemble_report.json"
#: :func:`merge_on_mapped_identity`'s record: its return key, and its side file.
IDENTITY_REPORT_KEY = "ensemble_identity_merge"
IDENTITY_REPORT_NAME = "ensemble_identity_merge_report.json"
#: Fields of a row that NAME an entity, rewritten in step with it (rule 4).
_REFERENCE_FIELDS = {
    "/processes/reactions": ("inputs", "outputs", "enzymes", "modifiers"),
    "/processes/transports": ("cargo",),
    "/processes/interactions": ("entity_1", "entity_2"),
}

#: The buckets the bounded summary totals over; the side file carries every bucket.
_SUMMARISED_PREFIXES = ("/processes/", "/entities/")
_REACTIONS = "/processes/reactions"
#: The fields ``_dedup_element_locations`` reads the located entity from.
_LOCATION_ENTITY_KEYS = ("compound", "element_collection", "nucleic_acid", "protein", "element")

DrawFn = Callable[[], Tuple[Dict[str, Any], List[Dict[str, Any]]]]

#: Participants that never define a reaction's identity (ORCH-752, from ORCH-751 § 2.2):
#: the evaluator's ``cofactor`` role, which ``reactions.score_leg`` lists apart and never
#: fails a match on. Spelled as ``strict_key`` output -- casefolded, trailing charge dropped,
#: so ``NADP+`` is ``nadp`` and ``Mg2+`` is ``mg2`` -- and matched whole-name only: no
#: stemming, no fuzzy matching, so what collapses is readable from this table. A payload
#: participant carrying ``role: cofactor`` counts whatever its name. Known ceiling: a
#: chelatase's metal and ATP synthase's ADP/Pi ARE the chemistry, which is why a side made
#: only of currency keeps every name (``_side``). ``pipeline.cofactor_policy._CURRENCY`` is
#: the advisory cousin of this table: unwired by design, normalised differently, no metal
#: ions, so it is not imported.
CURRENCY_KEYS = frozenset({
    "atp", "adp", "amp",
    "nad", "nadh", "nadp", "nadph",
    "fad", "fadh2",
    "coa", "coenzyme a",                      # free CoA only; acyl-CoAs are defining
    "h2o", "water",
    "pi", "ppi", "phosphate", "diphosphate",  # strict_key spells pyrophosphate as diphosphate
    "h", "proton",                            # H+
    "co2", "o2",
    "sam", "s adenosyl l methionine", "s adenosylmethionine",
    "sah", "s adenosyl l homocysteine", "s adenosylhomocysteine",
    "mg2", "mg", "fe2", "fe3", "fe", "zn2", "zn", "mn2", "mn", "ca2", "ca", "cu2", "cu",
    "k", "na", "cl",
})


def extraction_draws(override: Optional[int] = None) -> int:
    """How many Stage-1 draws to issue: ``override`` if given, else the env var, else 1.

    A value that is not an integer >= 1 raises rather than silently falling back to one
    draw: a run that believes it ensembled and did not is the § 6.2 failure mode.
    """

    raw: Any = override if override is not None else (os.environ.get(ENV_VAR, "").strip() or "1")
    try:
        count = int(str(raw).strip())
    except ValueError:
        raise ValueError(f"{ENV_VAR} must be an integer >= 1, got {raw!r}") from None
    if count < 1:
        raise ValueError(f"{ENV_VAR} must be an integer >= 1, got {raw!r}")
    return count


def _strict_key(value: Any) -> str:
    from t2pw.bench.eval_v2.matcher import strict_key  # function-local: layering, see module doc

    return strict_key(value)


def _participants(value: Any) -> List[Tuple[Any, str]]:
    """(item, name) per participant in a slot of strings or participant dicts, production key order."""

    out: List[Tuple[Any, str]] = []
    for item in value if isinstance(value, list) else [value]:
        if isinstance(item, str) and item.strip():
            out.append((item, item))
        elif isinstance(item, dict):
            name = next((item[k] for k in PARTICIPANT_NAME_KEYS if item.get(k)), None)
            if name:
                out.append((item, str(name)))
    return out


def _is_cofactor(item: Any, key: str) -> bool:
    role = item.get("role") if isinstance(item, dict) else None
    return str(role or "").strip().casefold() == "cofactor" or key in CURRENCY_KEYS


def _side(value: Any) -> Tuple[frozenset, List[Tuple[Any, str, str]]]:
    """One reaction side as (defining key set, cofactors as (item, name, key)).

    A side made only of currency has nothing incidental on it: its defining set is every key."""

    defining, every, cofactors = set(), set(), []
    for item, name in _participants(value):
        key = _strict_key(name)
        if not key:
            continue
        every.add(key)
        if _is_cofactor(item, key):
            cofactors.append((item, name, key))
        else:
            defining.add(key)
    return frozenset(defining or every), cofactors


def _relationship_key(value: Any) -> str:
    """The type the exporter emits for a relationship, else its strict key (see module doc)."""

    from t2pw.pwml.ir import PWML_INTERACTION_TYPES, _interaction_type_key  # function-local: layering

    text = str(value or "")
    return PWML_INTERACTION_TYPES.get(_interaction_type_key(text)) or _strict_key(text)


#: ORCH-754a finding 1. Spellings of ONE molecule among :data:`CURRENCY_KEYS`, so ``CoA``
#: beside ``coenzyme A`` is never read as two cofactors.
_SAME_COFACTOR = {
    "coenzyme a": "coa", "water": "h2o", "phosphate": "pi", "diphosphate": "ppi", "proton": "h",
    "s adenosyl l methionine": "sam", "s adenosylmethionine": "sam",
    "s adenosyl l homocysteine": "sah", "s adenosylhomocysteine": "sah",
    "mg": "mg2", "zn": "zn2", "ca": "ca2",          # these metals occur as the 2+ ion only
}
#: Cofactors that can stand in for one another in ONE slot: two rows naming different
#: members of a group on one side are different reactions (NAD+ vs NADP+, ATP vs ADP,
#: Mg2+ vs Mn2+). A currency molecule in no group is unique, so it can never conflict;
#: a cofactor outside :data:`CURRENCY_KEYS` (``role: cofactor`` on a payload participant)
#: cannot be classified at all, and :func:`_conflict` then fails closed.
_COFACTOR_ROLE = {
    **dict.fromkeys(("nad", "nadh", "nadp", "nadph"), "nicotinamide"),
    **dict.fromkeys(("atp", "adp", "amp"), "adenine nucleotide"),
    **dict.fromkeys(("fad", "fadh2"), "flavin"),
    **dict.fromkeys(("mg2", "fe2", "fe3", "fe", "zn2", "mn2", "mn", "ca2", "cu2", "cu", "k", "na"), "metal ion"),
}


def _cofactor_molecules(value: Any) -> frozenset:
    """One side's cofactors as molecules: a currency key folded by :data:`_SAME_COFACTOR`,
    or ``("unclassified", key)`` for a cofactor only the payload's role marks as one."""

    return frozenset(_SAME_COFACTOR.get(key, key) if key in CURRENCY_KEYS else ("unclassified", key)
                     for _item, _name, key in _side(value)[1])


def _enzyme_names(row: Dict[str, Any]) -> List[str]:
    return [name for slot in ("enzymes", "modifiers") for _item, name in _participants(row.get(slot))]


def _state_profile(label: Any, states: Dict[str, Dict[str, Any]]) -> Dict[str, str]:
    """What a reaction's ``biological_state`` label denotes. Resolved through the
    ``biological_states`` row of that name in the same payload: its compartment
    (``compartment_canonical``, else ``subcellular_location`` with a plural ``-s`` folded,
    so ``plastids`` is ``plastid``), cell type, tissue and species. A label with no
    definition is only its own text."""

    key = _strict_key(label)
    row = states.get(key)
    if not key or not isinstance(row, dict):
        return {"label": key} if key else {}
    loc = _strict_key(row.get("compartment_canonical")) or _strict_key(row.get("subcellular_location"))
    return {
        "label": key,
        "location": " ".join(t[:-1] if len(t) > 3 and t.endswith("s") else t for t in loc.split()),
        **{f: _strict_key(row.get(f)) for f in ("cell_type", "tissue", "species")},
    }


def _states_by_key(payload: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    rows = payload.get("biological_states") if isinstance(payload, dict) else None
    return {_strict_key(r.get("name")): r for r in rows or [] if isinstance(r, dict) and r.get("name")}


def _profile(row: Dict[str, Any], states: Dict[str, Dict[str, Any]],
             enzyme_key: Callable[[str], Any]) -> Dict[str, Any]:
    """Everything :func:`_conflict` compares, computed once per row."""

    return {
        "inputs": _cofactor_molecules(row.get("inputs")),
        "outputs": _cofactor_molecules(row.get("outputs")),
        "enzymes": frozenset(enzyme_key(name) for name in _enzyme_names(row)),
        "state": _state_profile(row.get("biological_state"), states),
    }


def _conflict(a: Optional[Dict[str, Any]], b: Optional[Dict[str, Any]]) -> str:
    """Why two rows under one reaction key are NOT one reaction, or ``''`` (finding 1).

    Compatible: one row naming a cofactor, enzyme or state the other omits. Conflict, and
    never merged: both naming different cofactors in one role (:data:`_COFACTOR_ROLE`);
    any cofactor that cannot be classified unless one row's cofactors contain the
    other's (fail closed); both naming enzymes and the enzyme sets differ; both stating a
    biological state and the states differ -- or cannot be compared."""

    if not a or not b:
        return ""
    for slot in ("inputs", "outputs"):
        ma, mb = a[slot], b[slot]
        if any(isinstance(m, tuple) for m in ma | mb) and not (ma <= mb or mb <= ma):
            return f"{slot}: cofactors {_show(ma)} vs {_show(mb)} include one that cannot be classified"
        for role in sorted({_COFACTOR_ROLE[m] for m in ma | mb if m in _COFACTOR_ROLE}):
            ra = {m for m in ma if _COFACTOR_ROLE.get(m) == role}
            rb = {m for m in mb if _COFACTOR_ROLE.get(m) == role}
            if ra and rb and ra != rb:
                return f"{slot}: different {role} cofactors {_show(ra)} vs {_show(rb)}"
    if a["enzymes"] and b["enzymes"] and a["enzymes"] != b["enzymes"]:
        return f"different enzymes {_show(a['enzymes'])} vs {_show(b['enzymes'])}"
    sa, sb = a["state"], b["state"]
    if sa and sb and sa["label"] != sb["label"]:
        if "location" not in sa or "location" not in sb:
            return f"biological_state {sa['label']!r} vs {sb['label']!r} cannot be compared"
        for field in ("location", "cell_type", "tissue", "species"):
            if sa[field] and sb[field] and sa[field] != sb[field]:
                return f"biological_state {field} {sa[field]!r} vs {sb[field]!r}"
    return ""


def _show(values: Any) -> str:
    return "{" + ", ".join(sorted(v[1] if isinstance(v, tuple) else str(v) for v in values)) + "}"


def _donate_cofactors(survivor: Dict[str, Any], later: Dict[str, Any]) -> Tuple[List[str], List[Dict[str, Any]]]:
    """Give ``survivor`` each cofactor ``later`` names on a side it does not already carry
    there -- but only one its OWN evidence quote names (finding 1: a survivor's evidence
    must support every participant it carries, and quotes are never edited or joined).
    A cofactor the survivor's quote does not support is withheld and returned with the
    donor's quote, for the side file. Returns ``(added names, withheld records)``."""

    from t2pw.bench.eval_v2.matcher import mentions, text_key  # function-local: layering

    evidence = text_key(survivor.get("evidence") if isinstance(survivor.get("evidence"), str) else "")
    added: List[str] = []
    withheld: List[Dict[str, Any]] = []
    for slot in ("inputs", "outputs"):
        have = survivor.get(slot)
        if not isinstance(have, list):
            continue
        present = {_strict_key(name) for _, name in _participants(have)}
        for item, name, key in _side(later.get(slot))[1]:
            if key in present:
                continue
            present.add(key)
            if mentions(name, evidence):
                have.append(deepcopy(item))
                added.append(name)
            else:
                withheld.append({"cofactor": name, "side": slot,
                                 "donor_evidence": later.get("evidence")})
    return added, withheld


def _signature(row: Any) -> str:
    try:
        return json.dumps(row, sort_keys=True, ensure_ascii=False)
    except TypeError:
        return repr(row)


def row_key(path: str, row: Any) -> Tuple[Any, ...]:
    """The dedup key of one list item under JSON-pointer ``path`` (see module doc)."""

    if isinstance(row, str):
        return ("str", _strict_key(row) or row)
    if not isinstance(row, dict):
        return ("json", _signature(row))
    key: Tuple[Any, ...]
    if path == "/processes/reactions":
        key = (_side(row.get("inputs"))[0], _side(row.get("outputs"))[0])
    elif path == "/processes/interactions":
        key = (_strict_key(row.get("entity_1")), _relationship_key(row.get("relationship")),
               _strict_key(row.get("entity_2")))
    elif path == "/processes/transports":
        key = tuple(_strict_key(row.get(f))
                    for f in ("cargo", "from_biological_state", "to_biological_state"))
    elif path == "/processes/reaction_coupled_transports":
        key = tuple(_strict_key(row.get(f)) for f in ("reaction", "transport"))
    elif path.startswith("/element_locations/"):
        entity = next((row[k] for k in _LOCATION_ENTITY_KEYS if row.get(k)), "")
        key = (_strict_key(entity), _strict_key(row.get("biological_state")))
    else:
        key = (_strict_key(row.get("name")),)
    if not any(key):
        return ("json", _signature(row))
    return key


def _label(key: Tuple[Any, ...]) -> str:
    if len(key) == 2 and isinstance(key[0], frozenset):
        return " + ".join(sorted(key[0])) + " -> " + " + ".join(sorted(key[1]))
    if key and key[0] in ("str", "json"):
        return str(key[1])[:200]
    return " | ".join(str(part) for part in key)


def _row_name(row: Any) -> str:
    if isinstance(row, dict):
        return str(row.get("name") or "")
    return str(row)[:200]


#: Per row (by ``id``), what the paper rewrote on it: ``{"refs": every name key it carries,
#: "renamed": {new key: (original key, rules)}}``. Built by :func:`_canonicalise`.
Notes = Dict[int, Dict[str, Any]]


def _bridge(a: Optional[Dict[str, Any]], b: Optional[Dict[str, Any]]) -> List[str]:
    """Rule 6: the rule(s) that made two rows one. A name both rows carry under the same new
    key but different original keys was bridged by whatever rewrote it; nothing else was."""

    a, b = a or {"refs": set(), "renamed": {}}, b or {"refs": set(), "renamed": {}}
    rules: set = set()
    for key in a["refs"] & b["refs"]:
        (raw_a, rules_a), (raw_b, rules_b) = (n["renamed"].get(key, (key, ())) for n in (a, b))
        if raw_a != raw_b:
            rules.update(rules_a, rules_b)
    return sorted(rules) or [RULE_EXACT]


def _absorb(profile: Optional[Dict[str, Any]], other: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """A survivor's profile after a compatible row joins it: every cofactor either claimed,
    and the first stated enzymes and state. Later rows are judged against ALL of it, so a
    row compatible with each of two conflicting rows can never join them into one."""

    if not profile or not other:
        return profile or other
    return {
        "inputs": profile["inputs"] | other["inputs"],
        "outputs": profile["outputs"] | other["outputs"],
        "enzymes": profile["enzymes"] or other["enzymes"],
        "state": profile["state"] or other["state"],
    }


def _union(dest: List[Any], items: List[Any], path: str, draw: int,
           bucket: Dict[Tuple[Any, ...], List[Dict[str, Any]]], notes: Notes) -> None:
    for item in items:
        key = row_key(path, item)
        note = notes.get(id(item)) or {}
        records = bucket.setdefault(key, [])
        record = None
        kept_apart: List[Dict[str, Any]] = []
        for candidate in records:
            reason = _conflict(candidate["_profile"], note.get("profile")) if path == _REACTIONS else ""
            if not reason:
                record = candidate
                break
            kept_apart.append({"name": candidate["name"], "draws": list(candidate["draws"]), "reason": reason})
        if record is None:
            dest.append(deepcopy(item))
            # "_row" is the surviving object, for cofactor donation; "_notes" what the paper
            # rewrote on it, for rule 6; "_profile" what finding 1 compares. merge_draws strips them.
            records.append({"key": _label(key), "name": _row_name(item), "draws": [draw], "_row": dest[-1],
                            "_notes": note, "_profile": note.get("profile"),
                            **({"kept_apart_from": kept_apart} if kept_apart else {})})
            continue
        if record["draws"][-1] != draw:
            record["draws"].append(draw)
        record.setdefault("merges", []).append(
            {"draw": draw, "name": _row_name(item), "bridged_by": _bridge(record["_notes"], note)})
        if path == _REACTIONS and isinstance(item, dict):
            record["_profile"] = _absorb(record["_profile"], note.get("profile"))
            added, withheld = _donate_cofactors(record["_row"], item)
            if added:
                record.setdefault("cofactors_added", []).extend(added)
            if withheld:
                record.setdefault("cofactors_withheld", []).extend(withheld)


def _merge_into(target: Dict[str, Any], source: Dict[str, Any], path: str, draw: int,
                seen: Dict[str, Dict[Tuple[Any, ...], List[Dict[str, Any]]]],
                per_draw: Dict[str, List[int]], n: int, notes: Notes) -> None:
    for key, value in source.items():
        child = f"{path}/{key}"
        if isinstance(value, dict):
            if key not in target:
                target[key] = {}
            if isinstance(target[key], dict):
                _merge_into(target[key], value, child, draw, seen, per_draw, n, notes)
        elif isinstance(value, list):
            if key not in target:
                target[key] = []
            per_draw.setdefault(child, [0] * n)[draw - 1] += len(value)
            if isinstance(target[key], list):
                _union(target[key], value, child, draw, seen.setdefault(child, {}), notes)
        elif key not in target or target[key] in ("", None):
            # First non-empty scalar wins, as _merge_dict_in_place already does for chunks.
            target[key] = value


def _rename_slot(value: Any, rename: Callable[[str], str]) -> Any:
    """A slot of strings or participant dicts with every name passed through ``rename``."""

    if isinstance(value, str):
        return rename(value)
    if not isinstance(value, list):
        return value
    out: List[Any] = []
    for item in value:
        if isinstance(item, str):
            out.append(rename(item))
        elif isinstance(item, dict):
            field = next((k for k in PARTICIPANT_NAME_KEYS if isinstance(item.get(k), str) and item[k]), None)
            out.append({**item, field: rename(item[field])} if field else item)
        else:
            out.append(item)
    return out


Canonical = Callable[[str], Tuple[str, List[str], str]]


#: strict key of a participant spelling -> (name to write, rules, pattern)
Rewrite = Dict[str, Tuple[str, List[str], str]]


def _rewrite_map(payloads: Sequence[Optional[Dict[str, Any]]], canonical: Canonical) -> Rewrite:
    """Rules 1 and 2 over every reaction participant ANY draw wrote, so a spelling one draw
    uses only as an interaction endpoint or location still follows the rename."""

    rewrite: Rewrite = {}
    for payload in payloads:
        processes = payload.get("processes") if isinstance(payload, dict) else None
        for row in (processes or {}).get("reactions") or []:
            if not isinstance(row, dict):
                continue
            for slot in ("inputs", "outputs"):
                for _item, name in _participants(row.get(slot)):
                    key = _strict_key(name)
                    if key and key not in rewrite and key not in CURRENCY_KEYS:
                        rewrite[key] = canonical(name)
    return rewrite


def _canonicalise(draw: Dict[str, Any], rewrite: Rewrite, canonical: Canonical, number: int,
                  renames: List[Dict[str, Any]], notes: Notes) -> Dict[str, Any]:
    """Rules 1, 2 and 4 on a deep copy of one draw (see module doc). The names rewritten
    are the reaction participants; every other row naming one follows it. Enzyme names
    are compared through the paper too (finding 1) but written as the draw wrote them."""

    out = deepcopy(draw)
    processes = out.get("processes") if isinstance(out.get("processes"), dict) else {}
    done: Dict[str, Dict[str, Any]] = {}

    def rename(name: str) -> str:
        hit = rewrite.get(_strict_key(name))
        if not hit or not hit[1]:
            return name
        if name not in done:
            done[name] = {"draw": number, "from": name, "to": hit[0], "rules": hit[1],
                          **({"pattern": hit[2]} if hit[2] else {})}
        return hit[0]

    def note(row: Dict[str, Any], names: List[str]) -> None:
        renamed = {}
        refs = set()
        for name in names:
            raw = _strict_key(name)
            if raw in rewrite:
                new, rules = rewrite[raw][:2]
            elif raw in CURRENCY_KEYS:
                new, rules = name, []
            else:  # an enzyme or interaction endpoint: compared through the paper, never renamed
                new, rules = canonical(name)[:2]
            key = _strict_key(new) if rules else raw
            refs.add(key)
            if rules:
                renamed[key] = (raw, tuple(rules))
        notes[id(row)] = {"refs": refs, "renamed": renamed}

    for path, fields in _REFERENCE_FIELDS.items():
        for row in processes.get(path.rsplit("/", 1)[1]) or []:
            if not isinstance(row, dict):
                continue
            names = [n for f in fields for _i, n in _participants(row.get(f))]
            note(row, names)
            for f in fields:
                if f in row:
                    row[f] = _rename_slot(row[f], rename)
    entities = out.get("entities") if isinstance(out.get("entities"), dict) else {}
    for rows in entities.values():
        for row in rows if isinstance(rows, list) else []:
            if isinstance(row, dict) and isinstance(row.get("name"), str):
                note(row, [row["name"]])
                row["name"] = rename(row["name"])
    locations = out.get("element_locations") if isinstance(out.get("element_locations"), dict) else {}
    for rows in locations.values():
        for row in rows if isinstance(rows, list) else []:
            if isinstance(row, dict):
                for f in _LOCATION_ENTITY_KEYS:
                    if isinstance(row.get(f), str):
                        row[f] = rename(row[f])
    renames.extend(done.values())
    return out


def merge_draws(payloads: Sequence[Optional[Dict[str, Any]]],
                paper_text: Optional[str] = None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Union the draws into one payload. Position = draw number; ``None`` = a failed draw.

    ``paper_text`` switches on D-104 rules 1, 2 and 4 (module doc); without it names are
    keyed exactly as ORCH-752 keyed them. Finding 1's compatibility rule applies either way.
    Returns ``(merged, report)``: ``merged`` carries no key the draws did not carry -- it is
    the Stage-2 prompt, so no support mark or merge note may ride on it -- and ``report``
    is the per-row attribution described in the module doc.
    """

    n = len(payloads)
    if not any(isinstance(p, dict) for p in payloads):
        raise ValueError("merge_draws needs at least one successful draw")
    succeeded = sum(1 for p in payloads if isinstance(p, dict))
    lexicon = PaperLexicon(paper_text) if paper_text and str(paper_text).strip() else None
    renames: List[Dict[str, Any]] = []
    notes: Notes = {}
    enzyme_key: Callable[[str], str] = _strict_key
    if lexicon is not None:
        canonical: Canonical = lru_cache(maxsize=None)(lexicon.canonical)
        rewrite = _rewrite_map(payloads, canonical)
        payloads = [_canonicalise(p, rewrite, canonical, i, renames, notes) if isinstance(p, dict) else p
                    for i, p in enumerate(payloads, start=1)]
        enzyme_key = lambda name: _strict_key(canonical(name)[0])  # noqa: E731
    for payload in payloads:
        if isinstance(payload, dict):
            states = _states_by_key(payload)
            for row in (payload.get("processes") or {}).get("reactions") or []:
                if isinstance(row, dict):
                    notes.setdefault(id(row), {"refs": set(), "renamed": {}})["profile"] = _profile(
                        row, states, enzyme_key)
    merged: Dict[str, Any] = {}
    seen: Dict[str, Dict[Tuple[Any, ...], List[Dict[str, Any]]]] = {}
    per_draw: Dict[str, List[int]] = {}
    for index, payload in enumerate(payloads, start=1):
        if isinstance(payload, dict):
            _merge_into(merged, payload, "", index, seen, per_draw, n, notes)

    buckets: Dict[str, Dict[str, Any]] = {}
    for path, bucket in seen.items():
        records = [record for group in bucket.values() for record in group]
        for record in records:
            record["support"] = f"{len(record['draws'])}/{succeeded}"
            if succeeded > 1 and len(record["draws"]) == 1:
                record["review_required"] = "found by one draw only"
        rows = [{k: v for k, v in r.items() if not k.startswith("_")} for r in records]
        buckets[path] = {
            "merged": len(rows),
            "rows_per_draw": per_draw.get(path, [0] * n),
            "first_seen_in_draw": [sum(1 for r in rows if r["draws"][0] == d) for d in range(1, n + 1)],
            "unique_to_one_draw": sum(1 for r in rows if len(r["draws"]) == 1),
            "in_every_successful_draw": sum(1 for r in rows if len(r["draws"]) == succeeded),
            "rows": rows,
        }
    report = {
        "draws_requested": n,
        "draws_succeeded": succeeded,
        "draws_failed": n - succeeded,
        "paper_lexicon": lexicon.summary() if lexicon is not None else {
            "status": "not_evaluated", "reason": "no paper text reached the merge"},
        "names_rewritten": renames,
        "buckets": buckets,
    }
    return merged, report


def _summary(report: Dict[str, Any], report_path: str) -> Dict[str, Any]:
    """The bounded form for the boundary record: totals over processes + entities."""

    n = int(report["draws_requested"])
    chosen = {p: b for p, b in report["buckets"].items() if p.startswith(_SUMMARISED_PREFIXES)}
    totals = {
        key: [sum(b[key][d] for b in chosen.values()) for d in range(n)]
        for key in ("rows_per_draw", "first_seen_in_draw")
    }
    return {
        "draws_requested": n,
        "draws_succeeded": report["draws_succeeded"],
        "draws_failed": report["draws_failed"],
        "failed_draws": report.get("failed_draws", []),
        "merged_rows": sum(b["merged"] for b in chosen.values()),
        "unique_to_one_draw": sum(b["unique_to_one_draw"] for b in chosen.values()),
        "in_every_successful_draw": sum(b["in_every_successful_draw"] for b in chosen.values()),
        **totals,
        "by_bucket": {p: {"merged": b["merged"], "unique_to_one_draw": b["unique_to_one_draw"]}
                      for p, b in chosen.items()},
        "report_path": report_path,
    }


def render_report_text(report: Dict[str, Any]) -> str:
    """The side file as text a person reads (D-104 rule 5; product-owner ruling 2026-10-20).

    Written ONLY to ``stage1_ensemble_report.txt`` beside the JSON side file: nothing in
    this repository reads either file back, and no model ever sees them. That is the
    point -- ``reaction_summary.txt`` feeds the curator and the enrichment agent, and the
    pooled payload itself is the Stage-2 prompt, so the support marks and merge notes are
    kept out of both. Reaction rows only. A pure function of ``report``.
    """

    lexicon = report.get("paper_lexicon") or {}
    if "abbreviation_pairs" in lexicon:
        ambiguous = ", ".join(sorted(lexicon.get("ambiguous_abbreviations") or {})) or "none"
        dictionary = f"{lexicon['abbreviation_pairs']} abbreviation pair(s); ambiguous, bridging nothing: {ambiguous}"
    else:
        dictionary = f"not evaluated -- {lexicon.get('reason') or 'no paper text'}"
    rows = ((report.get("buckets") or {}).get(_REACTIONS) or {}).get("rows") or []
    lines = [
        "STAGE-1 ENSEMBLE: POOLED REACTIONS",
        f"Draws: {report.get('draws_succeeded')} of {report.get('draws_requested')} succeeded"
        + "".join(f"; draw {f.get('draw')} failed ({f.get('error')})" for f in report.get("failed_draws") or []),
        f"Paper dictionary: {dictionary}",
        f"Pooled reactions: {len(rows)} ({sum(1 for r in rows if r.get('review_required'))} found by one draw only)",
        "",
    ]
    for number, row in enumerate(rows, 1):
        flag = "  -- REVIEW: found by one draw only" if row.get("review_required") else ""
        lines.append(f"REACTION {number}  [{row.get('support')} draws: {', '.join(map(str, row.get('draws') or []))}]{flag}")
        lines.append(f"  Name: {row.get('name') or '(unnamed)'}")
        lines.append(f"  Key:  {row.get('key')}")
        for merge in row.get("merges") or []:
            lines.append(f"  Merged draw {merge.get('draw')} row {merge.get('name')!r}: bridged by "
                         f"{', '.join(merge.get('bridged_by') or [])}")
        for other in row.get("kept_apart_from") or []:
            lines.append(f"  Kept apart from {other.get('name')!r} (draws {', '.join(map(str, other.get('draws') or []))}): "
                         f"{other.get('reason')}")
        if row.get("cofactors_added"):
            lines.append(f"  Cofactors added from a merged row: {', '.join(row['cofactors_added'])}")
        for withheld in row.get("cofactors_withheld") or []:
            lines.append(f"  Cofactor withheld ({withheld.get('side')}): {withheld.get('cofactor')} -- this row's quote "
                         f"does not name it; the donor's quote: {withheld.get('donor_evidence')!r}")
        lines.append("")
    renames = report.get("names_rewritten") or []
    lines.append(f"NAMES REWRITTEN ({len(renames)})")
    for rename in renames:
        pattern = f", {rename['pattern']}" if rename.get("pattern") else ""
        lines.append(f"  draw {rename.get('draw')}: {rename.get('from')!r} -> {rename.get('to')!r} "
                     f"[{', '.join(rename.get('rules') or [])}{pattern}]")
    return "\n".join(lines) + "\n"


def _write_side_file(artifact_dir: Optional[Path], report: Dict[str, Any]) -> str:
    """Append ``report`` to the run's side file, and its text rendering to the ``.txt``
    beside it. Never raises: a report that can kill the run it describes is worse than no
    report. Returns the JSON path written, or ``''``."""

    if artifact_dir is None:
        return ""
    path = Path(artifact_dir) / REPORT_NAME
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        existing = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
        if not isinstance(existing, list):
            existing = [existing]
        existing.append(report)
        path.write_text(json.dumps(existing, indent=2, ensure_ascii=False, default=str),
                        encoding="utf-8")
        with path.with_suffix(".txt").open("a", encoding="utf-8") as text:
            text.write(render_report_text(report) + "\n")
    except (OSError, ValueError, TypeError) as exc:
        logger.warning("stage1 ensemble report could not be written to %s: %s", path, exc)
        return ""
    return str(path)


def run_ensemble(draw: DrawFn, n: int, *, failure: type,
                 paper_text: Optional[str] = None) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Issue ``draw`` ``n`` times, union the successes, raise only when every draw failed.

    ``failure`` is the exception class that means "this draw produced nothing usable"
    (``PipelineFailure`` in production); it is a parameter so this module never imports
    ``pipeline``. Anything else propagates untouched and immediately: a leg deadline or a
    cancelled run is not a bad sample. When every draw fails the last failure is
    re-raised, so callers keep seeing the type they always caught.
    """

    payloads: List[Optional[Dict[str, Any]]] = []
    attempts: List[Dict[str, Any]] = []
    failures: List[Tuple[int, BaseException]] = []
    for index in range(1, n + 1):
        try:
            payload, logs = draw()
        except failure as exc:  # type: ignore[misc]
            failures.append((index, exc))
            payloads.append(None)
            logs = getattr(exc, "attempts", None) or []
        else:
            payloads.append(payload)
        attempts.extend({**entry, "draw": index} for entry in logs if isinstance(entry, dict))
    if not any(isinstance(p, dict) for p in payloads):
        raise failures[-1][1]

    merged, report = merge_draws(payloads, paper_text=paper_text)
    report["failed_draws"] = [
        {"draw": index, "stage": str(getattr(exc, "stage", "") or ""), "error": str(exc)}
        for index, exc in failures
    ]
    diagnostics = current_diagnostics()
    report_path = _write_side_file(diagnostics.artifact_dir, report)
    diagnostics.record_boundary(
        stage="extraction",
        boundary=BOUNDARY_STAGE1_ENSEMBLE,
        attempts=n,
        outcome=OUTCOME_OK,
        ensemble=_summary(report, report_path),
    )
    return merged, attempts


# ----------------------------------------------------------------------------- rule 3

#: Entity buckets whose rows carry database identities a reaction participant can resolve to.
_IDENTITY_BUCKETS = ("compounds", "proteins", "protein_complexes", "nucleic_acids", "element_collections")
#: ``map_payload`` counts a mapping under this confidence as ``low_confidence_mappings``; such
#: an identity is not confident enough to merge two reactions on.
_CONFIDENT = 0.78


def _admitted_ids(row: Dict[str, Any]) -> Dict[str, str]:
    """The identities rule 3 may merge on: real accessions ``map_payload`` left in
    ``mapped_ids`` after ``_admit_identities`` withheld the unsupported ones -- and only
    from a confident, non-placeholder mapping. Candidates (``mapping_meta.candidates``),
    withheld ids (``rejected_mapped_ids``), best-effort and low-confidence matches and
    Unknown-backed placeholders never count."""

    from t2pw.mapping.identity_admission import external_accessions  # function-local: layering
    from t2pw.pipeline.entity_identity import (IDENTITY_PLACEHOLDER, IDENTITY_UNRESOLVED,
                                               is_pathbank_unknown_protein)

    meta = row.get("mapping_meta") if isinstance(row.get("mapping_meta"), dict) else {}
    if is_pathbank_unknown_protein(row) or meta.get("identity_status") in (IDENTITY_PLACEHOLDER,
                                                                             IDENTITY_UNRESOLVED):
        return {}
    try:
        confidence = float(meta.get("confidence") or 0.0)
    except (TypeError, ValueError):
        return {}
    if meta.get("best_effort") or confidence < _CONFIDENT:
        return {}
    return external_accessions(row)


def _lock_id(row: Any) -> str:
    value = row.get("locked_reaction_id") if isinstance(row, dict) else None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return str(value)
    return value.strip() if isinstance(value, str) else ""


def _account_locks(payload: Dict[str, Any], quarantined: List[Dict[str, Any]]) -> None:
    """File merged-away locked reactions exactly as ``dedupe_processes`` files a duplicate
    locked reaction, and recompute ``locked_reaction_filter_report`` by its rule, so no
    lock goes ``unaccounted``."""

    existing = payload.get("quarantined_locked_reactions")
    records = [r for r in existing if isinstance(r, dict)] if isinstance(existing, list) else []
    records.extend(quarantined)
    payload["quarantined_locked_reactions"] = records
    active = {_lock_id(r) for r in (payload.get("processes") or {}).get("reactions") or [] if _lock_id(r)}
    held = {_lock_id(r) for r in records if _lock_id(r)}
    prior = payload.get("locked_reaction_filter_report")
    prior = prior if isinstance(prior, dict) else {}
    found = max(int(prior.get("locked_reactions_found") or 0), len(active | held))
    if found:
        payload["locked_reaction_filter_report"] = {
            "locked_reactions_found": found,
            "exported_locked_reactions": len(active),
            "quarantined_locked_reactions": len(held),
            "unaccounted_locked_reactions": max(0, found - len(active | held)),
        }


def _pooled_draws() -> int:
    """N for the run in progress, or 0 when its Stage 1 did not pool (see
    :func:`merge_on_mapped_identity`)."""

    try:
        n = extraction_draws()
    except ValueError:
        return 0
    if n <= 1:
        return 0
    pooled = [b for b in current_diagnostics().boundaries if b.get("boundary") == BOUNDARY_STAGE1_ENSEMBLE]
    succeeded = max((int((b.get("ensemble") or {}).get("draws_succeeded") or 0) for b in pooled), default=0)
    return n if succeeded > 1 else 0


def merge_on_mapped_identity(payload: Dict[str, Any]) -> Dict[str, Any]:
    """D-104 rule 3: after Stage 6 mapping, two reactions whose defining participants
    resolve to the same identities are one reaction. Mutates ``payload`` in place.

    Called once, at the end of ``map_ids.map_payload`` -- after identities are assigned
    and admitted, before the canonical freeze (merge rule 8). Applies only when THIS run's
    Stage 1 pooled (:func:`_pooled_draws`): ``T2PW_EXTRACTION_DRAWS`` > 1 AND the run's
    diagnostics hold the ``stage1_ensemble`` boundary of 2+ successful draws. Both, because
    the payload can carry no marker (it is a model prompt; module doc) and the variable
    alone would merge a saved N=1 payload re-mapped under a ``.env`` that says 3.
    Otherwise it returns ``{}`` and touches nothing, so N=1 output does not move; when it
    applies it returns ``{IDENTITY_REPORT_KEY: record}`` and files the record in the run's
    side files (:func:`_write_identity_side_file`) -- never in the mapping report.

    Per participant, identity = the admitted accessions of the entity row it names
    (:func:`_admitted_ids`; rows sharing an accession of one kind are one identity, and a
    group whose accessions disagree in any namespace is no identity at all), else its name
    -- which at N>1 Stage 1 has already written as the paper's long form (rules 1 and 4).
    The key is ORCH-752's cofactor-blind key over those identities (rule 7): a participant
    is currency by name or by sharing an identity with a currency-named entity, and a side
    made only of currency keeps every identity, so a chelatase's metal still separates Fe
    from Mg. Rows sharing a key merge only when :func:`_conflict` finds them compatible --
    cofactor roles, enzymes (by identity) and biological state, exactly as at Stage 1;
    a conflicting pair stays two rows and is listed under ``refused``. The first row
    survives with its own evidence and participants, and gains each cofactor of the later
    row that its own quote supports; a merged-away locked reaction is filed as a
    quarantined duplicate. A row a ``reaction_coupled_transport`` names is never merged
    away, because its reference would dangle.
    """

    n = _pooled_draws()
    if n <= 1 or not isinstance(payload, dict):
        return {}
    from t2pw.mapping.identity_admission import entity_kind_class  # function-local: layering

    path = "/processes/reactions"
    entities = payload.get("entities") if isinstance(payload.get("entities"), dict) else {}
    parent: Dict[Any, Any] = {}

    def find(x: Any) -> Any:
        while parent.setdefault(x, x) != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    rows: List[Tuple[str, List[Tuple[str, str, str]]]] = []
    for bucket in _IDENTITY_BUCKETS:
        for row in entities.get(bucket) or []:
            if isinstance(row, dict) and isinstance(row.get("name"), str) and row["name"].strip():
                kind = entity_kind_class(bucket)
                tokens = [(kind, ns, value) for ns, value in sorted(_admitted_ids(row).items())]
                for token in tokens:
                    parent[find(token)] = find(tokens[0])
                rows.append((row["name"], tokens))
    values: Dict[Any, Dict[Tuple[str, str], set]] = {}
    for _name, tokens in rows:
        for kind, ns, value in tokens:
            values.setdefault(find(tokens[0]), {}).setdefault((kind, ns), set()).add(value)
    conflicted = {root for root, by_ns in values.items() if any(len(v) > 1 for v in by_ns.values())}
    exact: Dict[str, Any] = {}
    by_key: Dict[str, set] = {}
    for name, tokens in rows:
        root = find(tokens[0]) if tokens else None
        ident = ("id", root) if tokens and root not in conflicted else None
        exact.setdefault(name, ident)
        by_key.setdefault(_strict_key(name), set()).add(ident)
    currency = {exact[name] for name, _t in rows if exact[name] and _strict_key(name) in CURRENCY_KEYS}

    def identity(name: str) -> Any:
        ident = exact.get(name)
        if name not in exact:
            found = by_key.get(_strict_key(name), set())
            ident = next(iter(found)) if len(found) == 1 else None
        return ident or ("name", _strict_key(name))

    def side(value: Any) -> frozenset:
        defining, every = set(), set()
        for item, name in _participants(value):
            key = _strict_key(name)
            if not key:
                continue
            ident = identity(name)
            every.add(ident)
            if not (_is_cofactor(item, key) or ident in currency):
                defining.add(ident)
        return frozenset(defining or every)

    processes = payload.get("processes") if isinstance(payload.get("processes"), dict) else {}
    reactions = processes.get("reactions") if isinstance(processes.get("reactions"), list) else []
    coupled = {_strict_key(r.get("reaction")) for r in processes.get("reaction_coupled_transports") or []
               if isinstance(r, dict) and r.get("reaction")}
    states = _states_by_key(payload)
    # Enzymes compare by identity too: the same UniProt under two names is one enzyme.
    survivors: Dict[Tuple[frozenset, frozenset], List[List[Any]]] = {}
    kept: List[Any] = []
    merges: List[Dict[str, Any]] = []
    refused: List[Dict[str, Any]] = []
    quarantined: List[Dict[str, Any]] = []
    for index, row in enumerate(reactions):
        if not isinstance(row, dict):
            kept.append(row)
            continue
        key = (side(row.get("inputs")), side(row.get("outputs")))
        profile = _profile(row, states, identity)
        group = survivors.setdefault(key, [])
        slot = None
        for candidate in group:
            reason = _conflict(candidate[1], profile)
            if not reason:
                slot = candidate
                break
            refused.append({"name": _row_name(row), "would_merge_into": _row_name(candidate[0]),
                            "json_pointer": f"{path}/{index}", "reason": reason})
        if slot is None:
            group.append([row, profile])
            kept.append(row)
            continue
        first = slot[0]
        if _strict_key(row.get("name")) in coupled:
            refused.append({"name": _row_name(row), "would_merge_into": _row_name(first),
                            "json_pointer": f"{path}/{index}", "reason": "named by a reaction_coupled_transport"})
            kept.append(row)
            continue
        slot[1] = _absorb(slot[1], profile)
        added, withheld = _donate_cofactors(first, row)
        merges.append({
            "survivor": _row_name(first), "merged": _row_name(row), "json_pointer": f"{path}/{index}",
            "bridged_by": RULE_EXACT if row_key(path, row) == row_key(path, first) else RULE_DATABASE_ID,
            **({"cofactors_added": added} if added else {}),
            **({"cofactors_withheld": withheld} if withheld else {}),
        })
        if _lock_id(row):
            quarantined.append({
                "reaction_name": _row_name(row) or "<unnamed>", "reason": "ensemble_identity_duplicate",
                "action": "merged_into_identity_duplicate", "json_pointer": f"{path}/{index}",
                "original_reaction": deepcopy(row), "locked_reaction_id": _lock_id(row),
                "merged_into": _lock_id(first) or _row_name(first),
            })
    if merges:
        processes["reactions"] = kept
    if quarantined:
        _account_locks(payload, quarantined)
    record = {
        "draws": n,
        "reactions_before": len(reactions),
        "reactions_after": len(kept),
        "entities_with_admitted_identity": sum(1 for name, _t in rows if exact.get(name)),
        "conflicting_identity_groups": len(conflicted),
        "merges": merges,
        "refused": refused,
    }
    _write_identity_side_file(current_diagnostics().artifact_dir, record)
    return {IDENTITY_REPORT_KEY: record}


def _write_identity_side_file(artifact_dir: Optional[Path], record: Dict[str, Any]) -> None:
    """File one rule-3 pass beside the Stage-1 side file: appended to
    ``ensemble_identity_merge_report.json`` and, as text, to ``stage1_ensemble_report.txt``.
    NOT in the mapping report: ``interactive_curator.compact_mapping_misses`` walks that
    report into a model prompt (ruling 2026-10-20). Never raises."""

    if artifact_dir is None:
        return
    path = Path(artifact_dir) / IDENTITY_REPORT_NAME
    lines = [f"DATABASE-IDENTITY MERGE (rule 3, after mapping): {record['reactions_before']} -> "
             f"{record['reactions_after']} reactions"]
    lines += [f"  {m['survivor']!r} <- {m['merged']!r}: bridged by {m['bridged_by']}" for m in record["merges"]]
    lines += [f"  kept apart: {r['name']!r} from {r['would_merge_into']!r}: {r['reason']}" for r in record["refused"]]
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        existing = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
        existing = existing if isinstance(existing, list) else [existing]
        path.write_text(json.dumps([*existing, record], indent=2, ensure_ascii=False, default=str), encoding="utf-8")
        with (Path(artifact_dir) / REPORT_NAME).with_suffix(".txt").open("a", encoding="utf-8") as text:
            text.write("\n".join(lines) + "\n\n")
    except (OSError, ValueError, TypeError) as exc:
        logger.warning("ensemble identity merge record could not be written to %s: %s", path, exc)
