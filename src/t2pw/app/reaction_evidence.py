"""Reaction-evidence review model and deterministic plain-text export (ORCH-742b).

PRESENTATION ONLY. Everything here is read from a run's EXISTING artifacts --
``coverage_summary.json``, ``quarantine_report.json`` and
``removed_entity_report.json`` (the strict-quarantine set written by
``strict_quarantine.write_quarantine_artifacts``), ``final_mapped.json`` (the
canonical payload, ``batch.driver._add_identity_artifacts``, including its
``quarantined_locked_reactions`` record, which ``pipeline.filter_unresolvable_reactions``,
``process_normalizer.dedupe_processes`` and strict quarantine all append to; a row
from it is labelled by that record name, never by a guessed stage), ``pwml_ir.json``
(the IR the PWML was written from, ``batch.driver._add_pwml_artifacts``: each
interaction's export verdict is READ from it, never recomputed, and only when a PWML
was written) and the first
record of ``LEG_TRACE.jsonl`` for the paper id. Nothing
is recomputed, repaired, merged or deleted, and no model, embedding or network
call is made: this module imports only the standard library, the quarantine
filename constants and the interaction sign table (``t2pw.pwml.interaction_types``,
itself standard-library only).

A value an artifact does not carry is rendered ``Not recorded``; it is never
guessed. A missing artifact (``Not recorded``) and a recorded empty list
(``None recorded``) stay distinct, and ``False`` is a value, not an absence.

The duplicate check is a REVIEWER WARNING only: two surviving reactions with the
same normalized substrates and products are flagged "Possible duplicate
reaction" and both stay in the report exactly as the pathway has them.
"""

from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

# Standard library only (the sign table, no exporter): used to check that an IR row
# belongs to the payload row beside it, never to decide a verdict.
from t2pw.pwml.interaction_types import interaction_relationship_word, pwml_interaction_type

from t2pw.pipeline.strict_quarantine import (
    COVERAGE_REPORT_FILENAME,
    QUARANTINE_REPORT_FILENAME,
    REMOVED_ENTITY_REPORT_FILENAME,
)

FINAL_MAPPED_FILENAME = "final_mapped.json"
PWML_IR_FILENAME = "pwml_ir.json"
#: The PWML a strict leg writes (``batch.driver``: release-ready / review-required). A verdict
#: is shown only when one of them is present: an IR exists even when validation stopped the
#: export (``streamlit_app`` returns ``ok: False`` with ``pwml_ir``; the driver still writes
#: ``pwml_ir.json``), and a verdict then would claim a row was drawn in a file that was never
#: written.
PWML_FILENAMES = ("pathway.pwml", "pathway.review_required.pwml")
LEG_TRACE_FILENAME = "LEG_TRACE.jsonl"
JSON_ARTIFACTS = (
    COVERAGE_REPORT_FILENAME,
    QUARANTINE_REPORT_FILENAME,
    REMOVED_ENTITY_REPORT_FILENAME,
    FINAL_MAPPED_FILENAME,
    PWML_IR_FILENAME,
)
NOT_RECORDED = "Not recorded"
NONE_RECORDED = "None recorded"
DUPLICATE_WARNING = "Possible duplicate reaction"
LOCKED_RECORD = "final payload quarantined_locked_reactions"

#: Identifier fields shown verbatim, by their payload key, so nothing is relabelled.
_ID_FIELDS = (
    "pathbank_compound_id",
    "chebi_id",
    "kegg_id",
    "hmdb_id",
    "uniprot",
    "pathbank_protein_id",
    "pathbank_protein_complex_id",
)


def _dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _list(value: Any) -> List[Any]:
    return value if isinstance(value, list) else []


def _text(value: Any) -> str:
    return "" if value is None else str(value).strip()


def _show(value: Any) -> str:
    """One scalar for display. ``False`` and ``0`` are values; only absence is Not recorded."""

    if value is None or value == "":
        return NOT_RECORDED
    if isinstance(value, (bool, int, float)):
        return json.dumps(value)
    return _text(value)


def _texts(value: Any) -> Optional[List[str]]:
    """A recorded list of strings, or ``None`` when the artifact does not carry one."""

    return [_text(v) for v in value if _text(v)] if isinstance(value, list) else None


def _norm(name: Any) -> str:
    return " ".join(unicodedata.normalize("NFKC", _text(name)).casefold().split())


def _participant(entry: Any) -> str:
    """A bare name, or ``{name, stoichiometry}`` rendered as ``"2 ATP"``."""

    if not isinstance(entry, dict):
        return _text(entry)
    name = _text(entry.get("name"))
    stoich = _text(entry.get("stoichiometry"))
    return f"{stoich} {name}".strip() if stoich and stoich not in {"1", "1.0"} else name


def _side(row: Mapping[str, Any], key: str) -> List[str]:
    return [p for p in (_participant(e) for e in _list(row.get(key))) if p]


def _participant_names(row: Mapping[str, Any]) -> List[str]:
    """Participant NAMES, without stoichiometry, for the entity lookup."""

    names = [
        _text(_dict(e).get("name")) if isinstance(e, dict) else _text(e)
        for key in ("inputs", "outputs")
        for e in _list(row.get(key))
    ]
    return list(dict.fromkeys(n for n in names if n))


def _equation(inputs: List[str], outputs: List[str]) -> str:
    # Built from the lists: names like "NAD+" contain "+", so this string is
    # display-only and is never split back apart.
    return (
        f"{' + '.join(inputs) or '(no substrates recorded)'} -> "
        f"{' + '.join(outputs) or '(no products recorded)'}"
    )


def _quotes(row: Mapping[str, Any]) -> List[str]:
    """Verbatim evidence strings on a process row and its enzymes, first-seen order."""

    found: List[str] = []

    def add(value: Any) -> None:
        if isinstance(value, dict):
            value = value.get("quote") or value.get("span") or value.get("text")
        for item in value if isinstance(value, list) else [value]:
            if isinstance(item, (dict, list)):
                add(item)
            elif _text(item) and _text(item) not in found:
                found.append(_text(item))

    add(row.get("evidence"))
    for enzyme in _list(row.get("enzymes")):
        add(_dict(enzyme).get("evidence"))
    return found


def _lineage(row: Mapping[str, Any]) -> List[str]:
    out = []
    for step in map(_dict, _list(row.get("provenance_lineage"))):
        text = f"{_show(step.get('stage'))}/{_show(step.get('origin'))} (paper_explicit={_show(step.get('paper_explicit'))}"
        out.append(text + (", review_required)" if step.get("review_required") is True else ")"))
    return out


def _sources(row: Mapping[str, Any], paper_id: str) -> List[str]:
    out: List[str] = []
    rag = _dict(row.get("rag_provenance"))
    if rag:
        out.append(
            f"RAG {_show(rag.get('source_id'))}; section={_show(rag.get('section'))}; "
            f"chunk={_show(rag.get('chunk_id'))}"
        )
    for paper in map(_dict, _list(row.get("source_papers"))):
        out.append(" ".join(p for p in (_text(paper.get(k)) for k in ("source_id", "title", "uri")) if p))
    lineage = [_dict(step) for step in _list(row.get("provenance_lineage"))]
    for source in (_dict(s) for step in lineage for s in _list(step.get("sources"))):
        if source:
            out.append(
                f"{_show(source.get('source_type'))} {_show(source.get('source_id'))} "
                f"({_show(source.get('locator'))})"
            )
    if any(step.get("stage") == "paper_extraction" for step in lineage):
        out.append(f"supplied paper {paper_id} via paper_extraction")
    return list(dict.fromkeys(line for line in out if line))


def _entity_index(payload: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    index: Dict[str, Dict[str, Any]] = {}
    for bucket, rows in _dict(payload.get("entities")).items():
        if bucket not in ("species", "subcellular_locations"):
            for row in map(_dict, _list(rows)):
                index.setdefault(_norm(row.get("name")), row)
    return index


def _mapping(name: str, entity: Mapping[str, Any]) -> str:
    if not entity:
        return f"{name}: {NOT_RECORDED}"
    parts = [_show(_dict(_dict(entity.get("mapping_meta")).get("resolution")).get("status"))]
    if _text(entity.get("db_status")):
        parts.append(f"db_status={_text(entity['db_status'])}")
    ids = [f"{key}={_text(entity[key])}" for key in _ID_FIELDS if _text(entity.get(key))]
    parts.append(", ".join(ids) or "no identifier recorded")
    return f"{name}: " + "; ".join(parts)


def _enzyme(enzyme: Mapping[str, Any], index: Mapping[str, Dict[str, Any]]) -> str:
    name = _text(enzyme.get("entity") or enzyme.get("name"))
    components = [_text(_dict(c).get("name")) for c in _list(_dict(index.get(_norm(name))).get("components"))]
    components = [c for c in components if c]
    detail = [_text(enzyme.get("entity_type"))] + ([f"components: {', '.join(components)}"] if components else [])
    detail = [d for d in detail if d]
    return f"{name} ({'; '.join(detail)})" if detail else name


def _same_process(lock_id: str, name: Any, row: Mapping[str, Any]) -> bool:
    if lock_id and row["lock_id"]:
        return lock_id == row["lock_id"]
    return bool(_norm(name)) and _norm(name) == _norm(row["name"])


#: ORCH-754b / D-104. What each exporter reason means, worded to be true whatever led
#: to it -- in particular ``bare_protein_endpoint`` says only that no wrapper exists,
#: never why Stage 6 did not build one.
_WITHHELD_GLOSS = {
    "untyped_interaction": (
        "the relationship word is neither Activation nor Inhibition, the only interaction types PathWhiz stores"
    ),
    "bare_protein_endpoint": (
        "a protein endpoint is not wrapped in a single-protein complex in this payload, "
        "and PathWhiz has no bare-protein interaction member"
    ),
    "ambiguous_protein_endpoint": (
        "a protein endpoint's name also names another kind of entity in this payload, "
        "so it is not known to be the protein and is not drawn as one"
    ),
}
_MISSING = {"species": "species", "external_identity": "UniProt/DrugBank identity"}


def _pwml_verdict(row: Mapping[str, Any]) -> str:
    """One IR interaction row's export verdict, as the IR states it."""

    status = _text(row.get("export_status"))
    if status == "exportable":
        return f"drawn as {_text(row.get('interaction_type')) or NOT_RECORDED}"
    if status != "withheld":
        return f"{NOT_RECORDED} (this IR predates per-row export verdicts)"
    withheld = _dict(row.get("withheld"))
    reason = _text(withheld.get("reason")) or NOT_RECORDED
    gloss = _WITHHELD_GLOSS.get(reason, "")
    lacking = [
        f"{_text(u.get('name')) or NOT_RECORDED} has no "
        + " and ".join(_MISSING.get(_text(m), _text(m)) for m in _list(u.get("missing")))
        for u in map(_dict, _list(withheld.get("unwrappable_proteins")))
    ]
    if lacking:
        gloss = "; ".join(lacking) + ", which a single-protein complex wrapper requires"
    shared = [
        f"{_text(a.get('name')) or NOT_RECORDED} also names a "
        + " and a ".join(_text(k).replace("_", " ") for k in _list(a.get("also_names")))
        for a in map(_dict, _list(withheld.get("ambiguous_endpoints")))
    ]
    if shared:
        gloss = "; ".join(shared) + " in this payload, so it is not known to be the protein and is not drawn as one"
    return f"withheld -- {reason}" + (f": {gloss}" if gloss else "")


def _one_line(value: Any) -> str:
    return " ".join(_text(value).split())


def _ir_norm(value: Any) -> str:
    # The exporter's name key (``ir._norm``), used only to see whether the entity an IR row
    # was bound to carries the name the payload row writes.
    return re.sub(r"[^a-z0-9:+ ]+", " ", _one_line(value).casefold()).strip()


def _same_row(row: Mapping[str, Any], ir_row: Mapping[str, Any], ir_entities: Mapping[str, Any]) -> bool:
    """Was ``ir_row`` built from ``row``? Word and both endpoints, as the exporter read them."""

    word = interaction_relationship_word(row)
    left = _one_line(row.get("left") or row.get("entity_1") or row.get("source"))
    right = _one_line(row.get("right") or row.get("entity_2") or row.get("target"))
    if _text(ir_row.get("export_status")) == "withheld":
        withheld = _dict(ir_row.get("withheld"))
        return (_text(withheld.get("raw_interaction_type")), _text(withheld.get("left")), _text(withheld.get("right"))) == (
            word, left, right)
    # A drawn row keeps the sign, not the word, and the entity each side was bound to.
    if pwml_interaction_type(word) != _text(ir_row.get("interaction_type")):
        return False
    for side, written in (("left", left), ("right", right)):
        entity = _dict(ir_entities.get(_text(_dict(ir_row.get(side)).get("entity_key"))))
        names = [entity.get(f) for f in ("name", "raw_name", "short_name", "common_name")]
        if _ir_norm(written) not in {_ir_norm(n) for n in [*names, *_list(entity.get("synonyms"))] if _ir_norm(n)}:
            return False
    return True


def _interactions(payload: Mapping[str, Any], ir: Any, ir_source: str, no_pwml: str) -> List[Dict[str, Any]]:
    """Every interaction in the final payload, with the IR's own verdict on it.

    ``no_pwml`` is why no PWML was written, or ``""`` when one was.
    """

    rows = [(i, r) for i, r in enumerate(_list(_dict(payload.get("processes")).get("interactions"))) if isinstance(r, dict)]
    ir_rows = _list(_dict(_dict(ir).get("processes")).get("interactions")) if isinstance(ir, dict) else None
    # The IR keeps one row per dict row, in order, named as below (``ir.build_pwml_ir``).
    # Anything else means the IR was built from a different payload: no verdict is shown.
    lined_up = ir_rows is not None and len(ir_rows) == len(rows) and all(
        _text(_dict(v).get("name")) == (" ".join(_text(r.get("name")).split()) or f"Interaction {k}")
        for k, ((_i, r), v) in enumerate(zip(rows, ir_rows), 1)
    )
    ir_entities = {
        _text(e.get("key")): e
        for bucket in _dict(_dict(ir).get("entities")).values()
        for e in map(_dict, _list(bucket))
        if _text(e.get("key"))
    }
    out: List[Dict[str, Any]] = []
    for k, (index, row) in enumerate(rows, 1):
        ir_row = _dict(ir_rows[k - 1]) if lined_up else {}
        if ir_rows is None:
            verdict = f"{NOT_RECORDED} (no {PWML_IR_FILENAME}: no PWML IR was built for this run)"
        elif not lined_up:
            verdict = f"{NOT_RECORDED} ({ir_source} does not line up with the interactions of the final payload)"
        elif _text(ir_row.get("export_status")) not in ("exportable", "withheld"):
            verdict = _pwml_verdict(ir_row)  # an IR older than per-row verdicts says so
        elif no_pwml:
            verdict = f"{NOT_RECORDED} (no PWML written: {no_pwml})"
        elif _text(ir_row.get("export_status")) == "exportable" and not all(
            isinstance(ir_row.get(side), dict) for side in ("left", "right")
        ):
            verdict = f"{NOT_RECORDED} (the IR resolved no entity for an endpoint of this row)"
        elif not _same_row(row, ir_row, ir_entities):
            verdict = f"{NOT_RECORDED} ({ir_source} row {k} was built from a different relationship or endpoints)"
        else:
            verdict = _pwml_verdict(ir_row)
        out.append(
            {
                "number": k,
                "pointer": f"/processes/interactions/{index}",
                "name": _text(row.get("name")),
                "relationship": _text(row.get("relationship")),
                "source": _text(row.get("entity_1") or row.get("left") or row.get("source")),
                "target": _text(row.get("entity_2") or row.get("right") or row.get("target")),
                "state": _text(row.get("biological_state")),
                "pwml": verdict,
                "evidence": _quotes(row),
            }
        )
    return out


def load_run_artifacts(directory: Any) -> Dict[str, Any]:
    """Read whichever of the known artifacts exist in ``directory``. Missing = absent key."""

    root = Path(directory)
    found: Dict[str, Any] = {}
    for name in JSON_ARTIFACTS:
        try:
            found[name] = json.loads((root / name).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
    try:
        with open(root / LEG_TRACE_FILENAME, encoding="utf-8") as handle:
            found[LEG_TRACE_FILENAME] = json.loads(handle.readline())
    except (OSError, ValueError):
        pass
    for name in PWML_FILENAMES:  # presence only: the report never reads a PWML
        if (root / name).is_file():
            found[name] = True
    return found


def build_report(
    artifacts: Mapping[str, Any],
    *,
    payload_source: str = FINAL_MAPPED_FILENAME,
    ir_source: str = PWML_IR_FILENAME,
    pwml_written: Optional[bool] = None,
    no_pwml_cause: str = "",
) -> Dict[str, Any]:
    """The review model, a plain dict built only from ``artifacts`` (``{filename: parsed}``)."""

    coverage = _dict(artifacts.get(COVERAGE_REPORT_FILENAME))
    quarantine = artifacts.get(QUARANTINE_REPORT_FILENAME)
    removed = artifacts.get(REMOVED_ENTITY_REPORT_FILENAME)
    payload = _dict(artifacts.get(FINAL_MAPPED_FILENAME))
    context = _dict(coverage.get("requested_context"))
    paper_id = _show(_dict(artifacts.get(LEG_TRACE_FILENAME)).get("paper_id"))
    index = _entity_index(payload)
    processes = _dict(payload.get("processes"))

    admitted: Dict[str, set] = {}
    for row in map(_dict, _list(_dict(quarantine).get("accepted"))):
        if row.get("bucket") == "reactions":
            admitted.setdefault(_norm(row.get("name")), set()).add(_text(row.get("state")))

    strict_rows: Optional[List[Dict[str, Any]]] = None
    if isinstance(quarantine, dict):
        strict_rows = []
        for row in map(_dict, _list(quarantine.get("quarantined"))):
            # ``detail.row`` carries ELIDED evidence ("123 chars elided"); only the
            # verbatim ``original_process`` is ever quoted.
            original = _dict(row.get("original_process"))
            inputs, outputs = _side(original, "inputs"), _side(original, "outputs")
            state, reason = _text(row.get("state")), _text(row.get("reason"))
            essential = _text(row.get("essential_participant"))
            strict_rows.append(
                {
                    "source": "",
                    "name": _text(row.get("name") or original.get("name")),
                    "pointer": f"pre-quarantine {_text(row['pointer'])}" if _text(row.get("pointer")) else "",
                    "lock_id": _text(original.get("locked_reaction_id")),
                    "kind": _text(row.get("process_kind")),
                    "equation": _equation(inputs, outputs) if (inputs or outputs) else "",
                    "state": state,
                    "reason": reason,
                    "essential_participant": essential,
                    "fields": [("State", state), ("Reason", reason), ("Essential participant", essential)],
                    "evidence": _quotes(original),
                }
            )

    # The payload's quarantined_locked_reactions record has three writers (the
    # locked-reaction filter, dedupe_processes, strict quarantine); the entry's own
    # action / quarantine_stage, when recorded, is the only stage shown. A lock
    # strict quarantine already lists is shown once.
    locked_records = payload.get("quarantined_locked_reactions")
    strict_locks = {row["lock_id"] for row in strict_rows or [] if row["lock_id"]}
    locked_rows: List[Dict[str, Any]] = []
    already_listed: List[str] = []
    for entry in map(_dict, _list(locked_records)):
        original = _dict(entry.get("original_reaction"))
        lock_id = _text(entry.get("locked_reaction_id") or original.get("locked_reaction_id"))
        if lock_id in strict_locks:
            already_listed.append(lock_id)
            continue
        inputs, outputs = _side(original, "inputs"), _side(original, "outputs")
        fields = [("Reason", _text(entry.get("reason")))]
        if isinstance(entry.get("missing_entities"), list):
            fields.append(("Missing entities", ", ".join(_texts(entry["missing_entities"])) or NONE_RECORDED))
        for key, label in (("action", "Action"), ("quarantine_stage", "Quarantine stage"), ("admission_state", "Admission state")):
            if _text(entry.get(key)):
                fields.append((label, _text(entry[key])))
        locked_rows.append(
            {
                "source": LOCKED_RECORD,
                "name": _text(entry.get("reaction_name") or original.get("name")),
                "pointer": f"json_pointer {_text(entry['json_pointer'])}" if _text(entry.get("json_pointer")) else "",
                "lock_id": lock_id,
                "kind": "",
                "equation": _equation(inputs, outputs) if (inputs or outputs) else "",
                "state": _text(entry.get("admission_state")),
                "reason": _text(entry.get("reason")),
                "essential_participant": "",
                "fields": fields,
                "evidence": _quotes(original),
            }
        )
    quarantined: Optional[List[Dict[str, Any]]] = None
    if strict_rows is not None or isinstance(locked_records, list):
        quarantined = [*(strict_rows or []), *locked_rows]
    locked_line = f"From {LOCKED_RECORD}: "
    if not isinstance(locked_records, list):
        locked_line += NOT_RECORDED
    else:
        locked_line += f"{len(locked_records)}"
        if already_listed:
            locked_line += (
                f" ({len(already_listed)} already listed from {QUARANTINE_REPORT_FILENAME} "
                f"by lock id: {', '.join(already_listed)})"
            )
    quarantine_sources = [f"From {QUARANTINE_REPORT_FILENAME} quarantined: {_count(strict_rows)}", locked_line]

    # Whether the quarantine decision was applied to the payload read above, in the
    # report's own terms. Research mode flags without removing; ok=false refused it.
    research = quarantine.get("research_mode") if isinstance(quarantine, dict) else None
    release_status = _show(_dict(_dict(quarantine).get("release")).get("status"))
    surviving_note: List[str] = []
    quarantined_note: List[str] = []
    if isinstance(research, dict) and research.get("applied") is False:
        surviving_note.append("research_mode.applied=false: quarantine removed nothing; reactions it flagged are still listed here")
        quarantined_note.append(f"research_mode.applied=false: {QUARANTINE_REPORT_FILENAME} rows are review flags, not removals")
    if _dict(quarantine).get("ok") is False:
        refused = f"{QUARANTINE_REPORT_FILENAME} ok=false: payload not released (release status {release_status})"
        surviving_note.append(refused)
        quarantined_note.append(refused)

    reactions: List[Dict[str, Any]] = []
    for number, row in enumerate(map(_dict, _list(processes.get("reactions"))), 1):
        inputs, outputs = _side(row, "inputs"), _side(row, "outputs")
        enzyme_names = [_text(_dict(e).get("entity") or _dict(e).get("name")) for e in _list(row.get("enzymes"))]
        states = sorted(admitted.get(_norm(row.get("name")), ()))
        lock_id = _text(row.get("locked_reaction_id"))
        reactions.append(
            {
                "number": number,
                "pointer": f"/processes/reactions/{number - 1}",
                "lock_id": lock_id,
                "name": _text(row.get("name")),
                "substrates": inputs,
                "products": outputs,
                "equation": _equation(inputs, outputs),
                "enzymes": [_enzyme(_dict(e), index) for e in _list(row.get("enzymes"))],
                "species": sorted({_text(index.get(_norm(n), {}).get("species")) for n in enzyme_names} - {""}),
                # Matched by name; ambiguous (two states for one name) is not guessed.
                "admission": states[0] if len(states) == 1 else "",
                # The same reaction listed under QUARANTINED / REJECTED: by lock id when
                # both carry one, else by name.
                "quarantine": [
                    f"listed as QUARANTINED {n}: state={q['state'] or NOT_RECORDED}, reason={q['reason'] or NOT_RECORDED}"
                    for n, q in enumerate(quarantined or [], 1)
                    if _same_process(lock_id, row.get("name"), q)
                ],
                "scope": _text(row.get("scope_membership")),
                "preservation": _text(row.get("preservation_status")),
                "lineage": _lineage(row),
                "mapping": [_mapping(n, index.get(_norm(n), {})) for n in [*_participant_names(row), *enzyme_names] if n],
                "evidence": _quotes(row),
                "sources": _sources(row, paper_id),
                "duplicate_of": [],
                "_key": (tuple(sorted(map(_norm, inputs))), tuple(sorted(map(_norm, outputs)))),
            }
        )
    for reaction in reactions:
        if any(reaction["_key"]):
            reaction["duplicate_of"] = [
                other["number"] for other in reactions if other is not reaction and other["_key"] == reaction["_key"]
            ]
    for reaction in reactions:
        del reaction["_key"]

    removed_entities: Optional[List[str]] = None
    if isinstance(removed, dict):
        removed_entities = [
            f"{_show(e.get('bucket'))} {_show(e.get('name'))}: {_show(e.get('reason'))}"
            for e in map(_dict, _list(removed.get("removed_entities")))
        ]

    return {
        "paper_id": paper_id,
        "pathway": _show(context.get("pathway_name")),
        "requested_organism": _show(context.get("likely_organism")),
        "species": [
            f"{_text(s.get('name'))} (taxonomy {_show(s.get('taxonomy_id'))})"
            for s in map(_dict, _list(_dict(payload.get("entities")).get("species")))
            if _text(s.get("name"))
        ],
        "coverage_ratio": _show(coverage.get("coverage_ratio")),
        "requested_core_terms": _texts(coverage.get("requested_core_terms")),
        "matched_terms": _texts(coverage.get("matched_terms")),
        "unmatched_terms": _texts(coverage.get("unmatched_terms")),
        "coverage_reasons": _texts(coverage.get("reasons")) or [],
        "surviving_processes": _show(coverage.get("surviving_processes")),
        "quarantined_processes": _show(coverage.get("quarantined_processes")),
        "minimum_core_satisfied": _show(coverage.get("minimum_core_satisfied")),
        "release_status": release_status,
        "export_mode": _show(_dict(quarantine).get("export_mode")),
        "quarantine_ok": _show(_dict(quarantine).get("ok")),
        "research_mode": (
            [f"Research mode applied: {_show(research.get('applied'))}", f"Research mode note: {_show(research.get('note'))}"]
            if isinstance(research, dict)
            else []
        ),
        "surviving_note": "; ".join(surviving_note),
        "quarantined_note": "; ".join(quarantined_note),
        "quarantine_sources": quarantine_sources,
        "payload_source": payload_source if payload else NOT_RECORDED,
        "transport_count": len(_list(processes.get("transports"))),
        "interaction_count": len(_list(processes.get("interactions"))),
        "artifacts": sorted(
            (payload_source if name == FINAL_MAPPED_FILENAME else name for name in artifacts), key=str.lower
        ),
        "reactions": reactions,
        "interactions": _interactions(
            payload,
            artifacts.get(PWML_IR_FILENAME),
            ir_source,
            # A run directory says by its files; the app says by its export result.
            ""
            if (any(name in artifacts for name in PWML_FILENAMES) if pwml_written is None else pwml_written)
            else (no_pwml_cause or f"neither {' nor '.join(PWML_FILENAMES)} is in this run"),
        ),
        "ir_source": ir_source if PWML_IR_FILENAME in artifacts else NOT_RECORDED,
        "quarantined": quarantined,
        "removed_entities": removed_entities,
    }


def _terms(label: str, terms: Optional[List[str]]) -> str:
    if terms is None:
        return f"{label}: {NOT_RECORDED}"
    return f"{label} ({len(terms)}): {', '.join(terms) or 'none'}"


def coverage_lines(report: Mapping[str, Any]) -> List[str]:
    lines = [
        _terms("Requested", report["requested_core_terms"]),
        _terms("Matched", report["matched_terms"]),
        _terms("Unmatched", report["unmatched_terms"]),
    ]
    if report["coverage_reasons"]:
        lines.append(f"Reasons: {'; '.join(report['coverage_reasons'])}")
    return lines


def header_lines(report: Mapping[str, Any]) -> List[str]:
    return [
        f"Paper: {report['paper_id']}",
        f"Pathway: {report['pathway']}",
        f"Requested organism (Stage 0): {report['requested_organism']}",
        f"Species (final payload): {', '.join(report['species']) or NOT_RECORDED}",
        f"Coverage: {report['coverage_ratio']}",
        f"Minimum core satisfied: {report['minimum_core_satisfied']}",
        f"Surviving processes: {report['surviving_processes']}",
        f"Quarantined processes: {report['quarantined_processes']}",
        f"Release status: {report['release_status']}",
        f"Export mode: {report['export_mode']}",
        f"Quarantine ok: {report['quarantine_ok']}",
        *report["research_mode"],
        f"Reactions itemised: {len(report['reactions'])}; interactions itemised: {len(report['interactions'])} "
        f"(transports: {report['transport_count']}; counted above, not itemised)",
        f"Surviving reactions read from: {report['payload_source']}",
        f"Artifacts read: {', '.join(report['artifacts']) or 'none'}",
    ]


def _evidence(quotes: List[str]) -> List[str]:
    return ["Evidence:", *([f'  {i}. "{q}"' for i, q in enumerate(quotes, 1)] or [f"  {NOT_RECORDED}"])]


def reaction_lines(reaction: Mapping[str, Any]) -> List[str]:
    # Pointers index the payload each artifact describes: survivors the final payload,
    # quarantined rows the pre-quarantine payload. The lock id is the cross-payload key.
    key = f"final payload {reaction['pointer']}" + (f"; lock {reaction['lock_id']}" if reaction["lock_id"] else "")
    status = [f"admission={reaction['admission']}"] if reaction["admission"] else []
    status += reaction["quarantine"] or ([] if status else [f"admission={NOT_RECORDED}"])
    status.append(f"scope={reaction['scope'] or NOT_RECORDED}")
    if reaction["preservation"]:
        status.append(f"preservation={reaction['preservation']}")
    lines = [
        f"REACTION {reaction['number']}  [{key}]",
        f"Name: {reaction['name'] or NOT_RECORDED}",
        f"Equation: {reaction['equation']}",
        f"Substrates: {', '.join(reaction['substrates']) or NOT_RECORDED}",
        f"Products: {', '.join(reaction['products']) or NOT_RECORDED}",
        f"Enzyme(s): {'; '.join(reaction['enzymes']) or NOT_RECORDED}",
        f"Species: {', '.join(reaction['species']) or NOT_RECORDED}",
        f"Status: {'; '.join(status)}",
        f"Lineage: {' | '.join(reaction['lineage']) or NOT_RECORDED}",
        "Mapping:",
        *[f"  - {m}" for m in reaction["mapping"] or [NOT_RECORDED]],
        *_evidence(reaction["evidence"]),
        "Source:",
        *[f"  - {s}" for s in reaction["sources"] or [NOT_RECORDED]],
    ]
    if reaction["duplicate_of"]:
        others = ", ".join(str(n) for n in reaction["duplicate_of"])
        lines.append(f"WARNING: {DUPLICATE_WARNING} (same substrates and products as reaction {others})")
    return lines


def interaction_lines(item: Mapping[str, Any]) -> List[str]:
    return [
        f"INTERACTION {item['number']}  [final payload {item['pointer']}]",
        f"Name: {item['name'] or NOT_RECORDED}",
        f"Relationship (as extracted): {item['relationship'] or NOT_RECORDED}",
        f"Source: {item['source'] or NOT_RECORDED}",
        f"Target: {item['target'] or NOT_RECORDED}",
        f"State: {item['state'] or NOT_RECORDED}",
        f"PWML: {item['pwml']}",
        *_evidence(item["evidence"]),
    ]


def interactions_title(report: Mapping[str, Any]) -> str:
    return f"INTERACTIONS ({len(report['interactions'])}) -- PWML verdicts read from: {report['ir_source']}"


def quarantined_lines(number: int, row: Mapping[str, Any]) -> List[str]:
    parts = (row["source"], row["pointer"], row["kind"], f"lock {row['lock_id']}" if row["lock_id"] else "")
    key = "; ".join(p for p in parts if p) or NOT_RECORDED
    lines = [f"QUARANTINED {number}  [{key}]", f"Name: {row['name'] or NOT_RECORDED}"]
    if row["equation"]:
        lines.append(f"Equation: {row['equation']}")
    return lines + [f"{label}: {value or NOT_RECORDED}" for label, value in row["fields"]] + _evidence(row["evidence"])


def with_note(title: str, note: str) -> str:
    return f"{title} -- {note}" if note else title


def surviving_title(report: Mapping[str, Any]) -> str:
    return with_note(f"SURVIVING REACTIONS ({len(report['reactions'])})", report["surviving_note"])


def quarantined_title(report: Mapping[str, Any]) -> str:
    return with_note(f"QUARANTINED / REJECTED ({_count(report['quarantined'])})", report["quarantined_note"])


def _section(title: str) -> List[str]:
    return [title, "-" * len(title)]


def _count(rows: Optional[List[Any]]) -> str:
    return NOT_RECORDED if rows is None else str(len(rows))


def _absent(rows: Optional[List[Any]]) -> List[str]:
    return [NOT_RECORDED if rows is None else NONE_RECORDED, ""]


def render_text(report: Mapping[str, Any]) -> str:
    """The .txt export: a pure function of ``report``, so same artifacts give the same bytes."""

    lines = ["PATHWAY EVIDENCE REPORT", "=======================", *header_lines(report), ""]
    lines += _section(surviving_title(report))
    for reaction in report["reactions"]:
        lines += [*reaction_lines(reaction), ""]
    if not report["reactions"]:
        lines += _absent(None if report["payload_source"] == NOT_RECORDED else [])
    lines += _section(interactions_title(report))
    for item in report["interactions"]:
        lines += [*interaction_lines(item), ""]
    if not report["interactions"]:
        lines += _absent(None if report["payload_source"] == NOT_RECORDED else [])
    quarantined = report["quarantined"]
    lines += [*_section(quarantined_title(report)), *report["quarantine_sources"], ""]
    for number, row in enumerate(quarantined or [], 1):
        lines += [*quarantined_lines(number, row), ""]
    if not quarantined:
        lines += _absent(quarantined)
    removed = report["removed_entities"]
    lines += _section(f"REMOVED ENTITIES ({_count(removed)})")
    lines += [*(f"- {e}" for e in removed), ""] if removed else _absent(removed)
    lines += [*_section("CORE COVERAGE"), *coverage_lines(report)]
    return "\n".join(lines) + "\n"
