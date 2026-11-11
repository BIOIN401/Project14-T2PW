"""Build the clean model from the frozen payload (final_mapped.json) plus local run metadata. No network."""

from __future__ import annotations

import copy
import json
import os
import re
from typing import Any, Dict, List, Optional

from . import vocab
from .model import (
    CONTAINERS, ENTITY_LISTS, PLACEMENT_KEYS, PROCESS_LISTS, CleanModel, Field, MappingRecord, Placement, Prov,
    Removed, State, frozen,
)

TOOL = "t2pw"
SENTINEL_PROTEIN_ID = 9659  # PathBank "Unknown" protein
_HEADER_DOI = re.compile(r"\b(10\.\d{4,9}/\S+)")
_PMC_IN_PATH = re.compile(r"^PMC\d+$")
_RUN_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}_\d{4}$")


def _read_json(path: str) -> Any:
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def _typed_list(v: Any) -> bool:
    return isinstance(v, list) and all(isinstance(x, dict) for x in v)


def _name(f: Field) -> Optional[str]:
    return f.value if isinstance(f.value, str) else None


def _is_sentinel(obj: Any) -> bool:
    """True for the PathBank Unknown placeholder (protein 9659 or name Unknown)."""
    if isinstance(obj, str):
        return obj.strip().casefold() == "unknown"
    if not isinstance(obj, dict):
        return False
    ids = obj.get("mapped_ids") if isinstance(obj.get("mapped_ids"), dict) else {}
    return (str(obj.get("name", "")).strip().casefold() == "unknown"
            or SENTINEL_PROTEIN_ID in (obj.get("pathbank_protein_id"), obj.get("pw_protein_id"), ids.get("pathbank_protein_id")))


# ---- run metadata (local files only) ------------------------------------------------------------------------------

def read_run_meta(run_dir: str) -> Dict[str, Any]:
    """pmcid / pmid / doi / title / year / mode / run_date from the run folder and its ancestors."""
    meta: Dict[str, Any] = {}
    run_dir = os.path.normpath(run_dir)
    parts = run_dir.replace("\\", "/").split("/")
    for p in reversed(parts):
        if _PMC_IN_PATH.match(p) and "pmcid" not in meta:
            meta["pmcid"] = p
        if _RUN_DATE.match(p) and "run_date" not in meta:
            meta["run_date"] = p
    if parts[-1] in ("strict", "research"):
        meta["mode"] = parts[-1]
    paper_dir = os.path.dirname(run_dir)
    # a `metadata` block (ir.py reads payload["metadata"]); none of the 152 canonical payloads has one
    for name in ("stage1_payload.json", "merged_payload.json"):
        blk = (_read_json(os.path.join(run_dir, name)) or {}).get("metadata")
        if isinstance(blk, dict):
            for k in ("title", "pmid", "pmcid", "doi", "authors", "year", "organism"):
                if blk.get(k) not in (None, "", "(unknown)"):
                    meta.setdefault(k, blk[k])
    try:
        with open(os.path.join(paper_dir, "00_PAPER.txt"), encoding="utf-8") as fh:
            for line in fh.read().splitlines()[:12]:
                m = re.match(r"^(title|year)\s*:\s*(.*)$", line)
                if m and m.group(2).strip() not in ("", "(unknown)"):
                    meta.setdefault(m.group(1), m.group(2).strip())
    except OSError:
        pass
    req = _read_json(os.path.join(paper_dir, "request.json"))  # single-PMID e2e requests
    if isinstance(req, dict) and isinstance(req.get("pmids"), list) and len(req["pmids"]) == 1:
        meta.setdefault("pmid", str(req["pmids"][0]))
    try:  # the PMC text opens with the journal header: ... PMC<id> <id> <id> <pmid> <doi> <title ...>
        with open(os.path.join(paper_dir, "01_source_text.txt"), encoding="utf-8") as fh:
            head = fh.read(1500)
        d = _HEADER_DOI.search(head)
        if d:
            meta.setdefault("doi", d.group(1).rstrip(".,;"))
            prev = head[:d.start()].split()
            pmc_num = (meta.get("pmcid") or "")[3:]
            if prev and prev[-1].isdigit() and prev[-1] != pmc_num and "pmid" not in meta:
                meta["pmid"] = prev[-1]  # a token position guess, not a structured field
                meta["pmid_source"] = "PMC text header"
            # the title follows the DOI but cannot be cut from the author block reliably: kept raw, not as the title
            meta.setdefault("header_after_doi", head[d.end():d.end() + 300].strip())
    except OSError:
        pass
    return meta


def _fill_pathway(model: CleanModel, meta: Dict[str, Any], src: str) -> None:
    pw = model.pathway
    for key, attr in (("title", "title"), ("pmid", "pmid"), ("pmcid", "pmcid"), ("doi", "doi"),
                      ("authors", "authors"), ("year", "year")):
        if meta.get(key) not in (None, ""):
            guess = key == "pmid" and meta.get("pmid_source")
            setattr(pw, attr, frozen(meta[key], "inferred" if guess else "paper", source=meta["pmid_source"] if guess else src,
                                     accession=str(meta[key]) if key != "title" else None))
    if meta.get("mode"):
        pw.mode = frozen(meta["mode"], "default", source=src)
    if meta.get("run_date"):
        pw.run_date = frozen(meta["run_date"], "default", source=src)
    pw.tool = frozen(TOOL, "default")
    sp = (model.species or [None])[0]
    if sp is not None and not sp.name.absent:
        pw.organism = Field(sp.name.value, sp.name.prov)
        if not sp.taxonomy_id.absent:
            pw.taxon_id = Field(sp.taxonomy_id.value, sp.taxonomy_id.prov)
    elif meta.get("organism"):
        pw.organism = frozen(meta["organism"], "paper", source=src)
    if meta.get("header_after_doi"):
        pw.extras["header_after_doi"] = meta["header_after_doi"]


# ---- derived views ------------------------------------------------------------------------------------------------

def _state_locations(model: CleanModel) -> Dict[str, str]:
    return {_name(s.name): _name(s.location) for s in (model.states or []) if _name(s.name) and _name(s.location)}


def _flag_unknown(model: CleanModel) -> None:
    sentinel_names = {_name(p.name) for p in (model.proteins or [])
                      if p.flags or _is_sentinel({"name": _name(p.name)})}
    for c in model.complexes or []:
        for m in c.members or []:
            if _is_sentinel(m.entity.value) or m.entity.value in sentinel_names or _is_sentinel(m.extras):
                m.unknown, m.entity.prov.label = True, "unknown"
                m.flags.append("unknown_sentinel")
                if "unknown_member" not in c.flags:
                    c.flags.append("unknown_member")
        if c.members and all(m.unknown for m in c.members):
            c.flags.append("all_members_unknown")
    for rx in model.reactions or []:
        for lst in (rx.catalysts, rx.modifiers):
            for m in lst or []:
                if m.entity_key and (m.entity.value in sentinel_names or _is_sentinel(m.entity.value)):
                    m.unknown, m.entity.prov.label = True, "unknown"
                    m.flags.append("unknown_sentinel")


def _resolve_locations(model: CleanModel, order: Any) -> None:
    sloc = _state_locations(model)
    for rx in model.reactions or []:
        st = _name(rx.state)
        if st in sloc:  # the reaction's state applies to its participants; derived, so `inferred`
            for ref in (rx.reactants or []) + (rx.products or []):
                ref.location = Field(sloc[st], Prov("inferred", source="biological_state:" + st, frozen=True))
        if isinstance(order, list) and _name(rx.name) in order and order.count(_name(rx.name)) == 1:
            rx.step_index = frozen(order.index(_name(rx.name)), "inferred", source="payload.reaction_order")
    for t in model.transports or []:
        for src, dst in ((t.from_state, "from_location"), (t.to_state, "to_location")):
            if _name(src) in sloc:
                setattr(t, dst, Field(sloc[_name(src)], Prov("inferred", source="biological_state:" + _name(src), frozen=True)))


def _classify_interactions(model: CleanModel) -> None:
    declared = set()
    for lst in (model.compounds, model.proteins, model.complexes, model.nucleic_acids):
        declared |= {_name(x.name) for x in lst or []}
    ec = model.extras["entities"].get("element_collections")
    if isinstance(ec, list):
        declared |= {x.get("name") for x in ec if isinstance(x, dict)}
    for ix in model.interactions or []:
        raw = ix.relationship.value if not ix.relationship.absent else None
        kind, sign = vocab.classify(raw)
        mt = "table" if vocab.is_mapped(raw) else "unmapped"
        ix.kind = Field(kind, Prov("inferred", source="clean_model.vocab", match_type=mt))
        ix.sign = Field(sign, Prov("inferred", source="clean_model.vocab", match_type=mt))
        ix.endpoints_declared = _name(ix.source) in declared and _name(ix.target) in declared
        if not ix.endpoints_declared:
            ix.flags.append("undeclared_endpoint")


def _mapping_records(model: CleanModel) -> None:
    for kind, lst in (("compound", model.compounds), ("protein", model.proteins), ("complex", model.complexes),
                      ("nucleic_acid", model.nucleic_acids), ("location", model.locations), ("species", model.species)):
        for e in lst or []:
            mm = e.mapping.value if not e.mapping.absent and isinstance(e.mapping.value, dict) else {}
            res = mm.get("resolution") if isinstance(mm.get("resolution"), dict) else {}
            cr = getattr(e, "chosen_rule", None)
            rule = mm.get("chosen_rule") or (cr.value if cr is not None and not cr.absent else None)
            if mm or rule:
                model.mapping_records.append(MappingRecord(
                    kind, _name(e.name), rule, mm.get("identity_status") or res.get("issue"),
                    res.get("status"), mm.get("provider") or mm.get("source")))


def _removed(model: CleanModel, payload: Dict[str, Any], run_dir: Optional[str]) -> None:
    def add(kind, entry, source):
        if isinstance(entry, dict):
            name = next((entry[k] for k in ("name", "entity", "reaction_name", "actor") if isinstance(entry.get(k), str)), None)
            reason = next((entry[k] for k in ("reason", "rule") if isinstance(entry.get(k), str)), None)
            model.removed.append(Removed(kind, name, reason, source, copy.deepcopy(entry)))

    for key, kind in (("quarantined_proteins", "protein"), ("quarantined_locked_reactions", "reaction"),
                      ("quarantined_process_claims", "process_claim")):
        for e in payload.get(key) or []:
            add(kind, e, "payload." + key)
    adm = payload.get("entity_admission_report")
    if isinstance(adm, dict):
        for key in ("removed", "demoted"):
            for e in adm.get(key) or []:
                add(key, e, "payload.entity_admission_report." + key)
    rep = _read_json(os.path.join(run_dir, "removed_entity_report.json")) if run_dir else None
    if isinstance(rep, dict):  # sibling file, not part of the frozen payload
        for key, kind in (("removed_entities", "entity"), ("removed_locations", "location"),
                          ("removed_biological_states", "biological_state")):
            for e in rep.get(key) or []:
                add(kind, e, "removed_entity_report." + key)


# ---- entry points -------------------------------------------------------------------------------------------------

def build_from_payload(payload: Dict[str, Any], meta: Optional[Dict[str, Any]] = None,
                       run_dir: Optional[str] = None) -> CleanModel:
    """Build a CleanModel; ``payload`` is never mutated (every value is deep-copied)."""
    m = CleanModel()
    ents = payload.get("entities") if isinstance(payload.get("entities"), dict) else None
    procs = payload.get("processes") if isinstance(payload.get("processes"), dict) else None
    locs = payload.get("element_locations") if isinstance(payload.get("element_locations"), dict) else None
    for name, d in (("entities", ents), ("processes", procs), ("element_locations", locs)):
        if d is not None:
            m.containers.append(name)
    for k, v in payload.items():
        if k == "biological_states" and _typed_list(v):
            m.states = [State.from_raw(x) for x in v]
        elif k in CONTAINERS and isinstance(v, dict):
            continue
        else:
            m.extras["top"][k] = copy.deepcopy(v)
    for container, table, store in (("entities", ENTITY_LISTS, ents), ("processes", PROCESS_LISTS, procs)):
        for k, v in (store or {}).items():
            if k in table and _typed_list(v):
                attr, cls = table[k]
                setattr(m, attr, [cls.from_raw(x) for x in v])
            else:
                m.extras[container][k] = copy.deepcopy(v)
    for k, v in (locs or {}).items():
        if k in PLACEMENT_KEYS and _typed_list(v):
            m.placements[k] = [Placement.from_raw(x) for x in v]
        else:
            m.extras["element_locations"][k] = copy.deepcopy(v)
    for p in m.proteins or []:
        if _is_sentinel({"name": _name(p.name), "pathbank_protein_id": p.extras.get("pathbank_protein_id"),
                         "pw_protein_id": p.extras.get("pw_protein_id")}):
            p.flags.append("unknown_sentinel")
    _flag_unknown(m)
    _resolve_locations(m, payload.get("reaction_order"))
    _classify_interactions(m)
    _mapping_records(m)
    _removed(m, payload, run_dir)
    meta = dict(meta or {})
    if isinstance(payload.get("metadata"), dict):
        for k in ("title", "pmid", "pmcid", "doi", "authors", "year", "organism"):
            if payload["metadata"].get(k) not in (None, "", "(unknown)"):
                meta.setdefault(k, payload["metadata"][k])
    _fill_pathway(m, meta, meta.get("source") or "run folder")
    return m


def build_from_run(run_dir: str) -> CleanModel:
    with open(os.path.join(run_dir, "final_mapped.json"), encoding="utf-8") as fh:
        payload = json.load(fh)
    return build_from_payload(payload, read_run_meta(run_dir), run_dir=run_dir)
