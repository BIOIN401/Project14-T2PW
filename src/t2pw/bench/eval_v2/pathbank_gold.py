"""Read-only PathBank pathway extractor for evaluation gold (ORCH-747).

EVALUATION-ONLY, like every other module in this package: it reads the live PathBank
MySQL instance, writes JSON outside the pipeline, calls no model and touches no run.

Two files per pathway:

* ``<PW>__extract.json`` -- the whole curated pathway as the lab stored it: every
  process (reaction, interaction, transport, reaction-coupled transport) with its
  participants, their external identifiers, the enzyme complexes with member proteins,
  the biological states (species, compartment, cell type, tissue), the curator's cited
  references, plus ``updated_at`` and a content hash. Neutral; no eval vocabulary.
* ``<PW>__draft_case.json`` -- the same reactions in the shape of the evaluation-v2 gold
  supplement, with **empty quotes on purpose**: a PathBank record is not evidence about
  any paper, so ``gold.load_supplement`` refuses this file. It is the scope hint for the
  paper-gold drafter (``prompts/paper_gold_drafter_v1.md``) and nothing else.

A public accession ``PW701720`` is ``super_pathway_visualizations.pw_id``; the pathway is
its ``named_for_id``. Looking the number up in ``pathways.id`` silently returns a
different pathway, so the accession is always resolved through that table.

Usage::

    python -m t2pw.bench.eval_v2.pathbank_gold --out docs/evaluation_v2/pathbank_gold PW701720
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from t2pw.config import resolution_db_config

SCHEMA_EXTRACT = "t2pw.eval_v2.pathbank_extract/1"
SCHEMA_DRAFT = "t2pw.eval_v2.gold_supplement_draft/1"

#: ``reactions.direction`` values that mean the curator drew the step as reversible.
REVERSIBLE_DIRECTIONS = {"<>", "=", "<=>", "<->", "both"}

#: Guard for ``Bound`` expansion: a bound holds elements that may themselves be bounds.
MAX_BOUND_DEPTH = 4

PROCESS_TABLES = {
    # process_type -> (process table, element table, element fk, enzyme table, enzyme fk)
    "Reaction": ("reactions", "reaction_elements", "reaction_id",
                 "reaction_enzymes", "reaction_id"),
    "Transport": ("transports", "transport_elements", "transport_id",
                  "transport_transporters", "transport_id"),
    "Interaction": ("interactions", "interaction_elements", "interaction_id", None, None),
    "ReactionCoupledTransport": ("reaction_coupled_transports",
                                 "reaction_coupled_transport_elements",
                                 "reaction_coupled_transport_id",
                                 "reaction_coupled_transport_enzymes",
                                 "reaction_coupled_transport_id"),
}


class PathBankExtractError(RuntimeError):
    """The database cannot answer. Never a partial extract written as if whole."""


def connect(overrides: Optional[Mapping[str, Any]] = None):
    """A read-only-by-use pymysql connection from ``.env`` (never a hardcoded host)."""

    try:
        import pymysql
        from pymysql.cursors import DictCursor
    except Exception as exc:  # noqa: BLE001 - optional dependency, named in requirements
        raise PathBankExtractError(f"pymysql unavailable: {exc}") from exc
    cfg = resolution_db_config(dict(overrides or {}))
    if not cfg.get("host") or not cfg.get("user"):
        raise PathBankExtractError("PATHBANK_DB_HOST / _USER are not configured")
    return pymysql.connect(
        host=cfg["host"], port=int(cfg.get("port") or 3306), user=cfg["user"],
        password=cfg.get("password") or "", database=cfg.get("schema") or "pathbank",
        connect_timeout=int(cfg.get("connect_timeout") or 10),
        read_timeout=int(cfg.get("read_timeout") or 120),
        cursorclass=DictCursor, autocommit=False)


class _Reader:
    """Thin query helper with an entity cache; every statement is a SELECT."""

    def __init__(self, conn: Any) -> None:
        self._cur = conn.cursor()
        self._entities: Dict[Tuple[str, int], Dict[str, Any]] = {}
        self._states: Dict[int, Dict[str, Any]] = {}

    def rows(self, sql: str, params: Sequence[Any] = ()) -> List[Dict[str, Any]]:
        self._cur.execute(sql, tuple(params))
        return list(self._cur.fetchall())

    def one(self, sql: str, params: Sequence[Any] = ()) -> Optional[Dict[str, Any]]:
        got = self.rows(sql, params)
        return got[0] if got else None

    def by_ids(self, sql: str, ids: Sequence[int]) -> List[Dict[str, Any]]:
        """``sql`` must contain one ``{ph}`` placeholder list marker."""

        ids = [int(i) for i in dict.fromkeys(ids)]
        if not ids:
            return []
        ph = ",".join(["%s"] * len(ids))
        return self.rows(sql.format(ph=ph), ids)

    # -- entities -----------------------------------------------------------
    def entity(self, element_type: Optional[str], element_id: Optional[int],
               depth: int = 0) -> Dict[str, Any]:
        if not element_type or element_id in (None, ""):
            return {"element_type": element_type, "element_id": element_id,
                    "name": None, "resolved": False}
        key = (str(element_type), int(element_id))
        if key in self._entities:
            return self._entities[key]
        out = self._resolve(str(element_type), int(element_id), depth)
        self._entities[key] = out
        return out

    def _resolve(self, element_type: str, element_id: int, depth: int) -> Dict[str, Any]:
        base = {"element_type": element_type, "element_id": element_id, "resolved": True}
        if element_type == "Compound":
            r = self.one("SELECT id,name,short_name,synonyms,hmdb_id,kegg_id,chebi_id,"
                         "pubchem_cid,cas,drugbank_id,biocyc_id,pwc_id,moldb_formula,"
                         "moldb_inchikey FROM compounds WHERE id=%s", (element_id,))
            if not r:
                return {**base, "resolved": False, "name": None}
            return {**base, "name": r["name"], "short_name": r["short_name"],
                    "synonyms": _synonyms(r.get("synonyms")),
                    "identifiers": _ids(r, ("hmdb_id", "kegg_id", "chebi_id", "pubchem_cid",
                                            "cas", "drugbank_id", "biocyc_id", "pwc_id")),
                    "formula": r.get("moldb_formula"), "inchikey": r.get("moldb_inchikey")}
        if element_type == "ProteinComplex":
            r = self.one("SELECT id,name,species_id,pwp_id FROM protein_complexes WHERE id=%s",
                         (element_id,))
            if not r:
                return {**base, "resolved": False, "name": None}
            members = self.rows(
                "SELECT pcp.stoichiometry, p.id, p.name, p.uniprot_id, p.gene_name, "
                "p.ec_numbers, p.hmdbp_id, p.locus, p.synonyms, p.species_id "
                "FROM protein_complex_proteins pcp JOIN proteins p ON p.id=pcp.protein_id "
                "WHERE pcp.protein_complex_id=%s ORDER BY p.id", (element_id,))
            cof = self.rows(
                "SELECT pcc.stoichiometry, c.id, c.name, c.kegg_id, c.chebi_id "
                "FROM protein_complex_cofactors pcc JOIN compounds c ON c.id=pcc.compound_id "
                "WHERE pcc.protein_complex_id=%s ORDER BY c.id", (element_id,))
            return {**base, "name": r["name"], "identifiers": _ids(r, ("pwp_id",)),
                    "species": self._species(r.get("species_id")),
                    "components": [{"protein_id": m["id"], "name": m["name"],
                                    "gene_name": m["gene_name"], "uniprot": m["uniprot_id"],
                                    "ec_numbers": m["ec_numbers"], "locus": m.get("locus"),
                                    "hmdbp_id": m.get("hmdbp_id"),
                                    "synonyms": _synonyms(m.get("synonyms")),
                                    "stoichiometry": m.get("stoichiometry"),
                                    "species": self._species(m.get("species_id"))}
                                   for m in members],
                    "cofactors": [{"compound_id": c["id"], "name": c["name"],
                                   "kegg_id": c.get("kegg_id"), "chebi_id": c.get("chebi_id"),
                                   "stoichiometry": c.get("stoichiometry")} for c in cof]}
        if element_type == "Protein":
            r = self.one("SELECT id,name,uniprot_id,gene_name,ec_numbers,hmdbp_id,locus,"
                         "synonyms,species_id FROM proteins WHERE id=%s", (element_id,))
            if not r:
                return {**base, "resolved": False, "name": None}
            return {**base, "name": r["name"], "gene_name": r["gene_name"],
                    "synonyms": _synonyms(r.get("synonyms")),
                    "identifiers": _ids(r, ("uniprot_id", "ec_numbers", "hmdbp_id", "locus")),
                    "species": self._species(r.get("species_id"))}
        if element_type == "NucleicAcid":
            r = self.one("SELECT id,name,short_name,gene_name,nucleotide_type,chebi_id,"
                         "genbank_id,pwna_id FROM nucleic_acids WHERE id=%s", (element_id,))
            if not r:
                return {**base, "resolved": False, "name": None}
            return {**base, "name": r["name"], "short_name": r.get("short_name"),
                    "gene_name": r.get("gene_name"), "nucleotide_type": r.get("nucleotide_type"),
                    "identifiers": _ids(r, ("chebi_id", "genbank_id", "pwna_id"))}
        if element_type == "ElementCollection":
            r = self.one("SELECT id,name,short_name,collection_type,element_type,element_id,"
                         "external_id,external_id_type,pwec_id FROM element_collections "
                         "WHERE id=%s", (element_id,))
            if not r:
                return {**base, "resolved": False, "name": None}
            return {**base, "name": r["name"], "short_name": r.get("short_name"),
                    "collection_type": r.get("collection_type"),
                    "member_element_type": r.get("element_type"),
                    "member_element_ids": str(r.get("element_id") or ""),
                    "identifiers": _ids(r, ("external_id", "external_id_type", "pwec_id"))}
        if element_type == "Bound":
            r = self.one("SELECT id,pwb_id,signature FROM bounds WHERE id=%s", (element_id,))
            if not r:
                return {**base, "resolved": False, "name": None}
            members: List[Dict[str, Any]] = []
            if depth < MAX_BOUND_DEPTH:
                for m in self.rows("SELECT element_type,element_id,stoichiometry FROM "
                                   "bound_elements WHERE bound_id=%s ORDER BY id", (element_id,)):
                    members.append({"stoichiometry": m.get("stoichiometry"),
                                    **self.entity(m["element_type"], m["element_id"], depth + 1)})
            name = " + ".join(str(m.get("name") or "?") for m in members) or str(r.get("signature") or "")
            return {**base, "name": name, "kind": "bound_complex",
                    "identifiers": _ids(r, ("pwb_id",)), "members": members,
                    "members_truncated": depth >= MAX_BOUND_DEPTH}
        return {**base, "resolved": False, "name": None,
                "note": f"unhandled element_type {element_type!r}"}

    # -- states -------------------------------------------------------------
    def state(self, state_id: Optional[int]) -> Optional[Dict[str, Any]]:
        if state_id in (None, ""):
            return None
        sid = int(state_id)
        if sid not in self._states:
            r = self.one(
                "SELECT b.id, b.pwbs_id, s.name species, s.taxonomy_id, sl.name location, "
                "ct.name cell_type, t.name tissue FROM biological_states b "
                "LEFT JOIN species s ON s.id=b.species_id "
                "LEFT JOIN subcellular_locations sl ON sl.id=b.subcellular_location_id "
                "LEFT JOIN cell_types ct ON ct.id=b.cell_type_id "
                "LEFT JOIN tissues t ON t.id=b.tissue_id WHERE b.id=%s", (sid,))
            self._states[sid] = dict(r) if r else {"id": sid, "resolved": False}
        return self._states[sid]

    def _species(self, species_id: Optional[int]) -> Optional[Dict[str, Any]]:
        if species_id in (None, ""):
            return None
        return self.one("SELECT id,name,taxonomy_id,common_name FROM species WHERE id=%s",
                        (int(species_id),))


def _synonyms(value: Any) -> List[str]:
    """PathBank stores synonyms as a newline- or semicolon-separated blob."""

    text = str(value or "").strip()
    if not text:
        return []
    parts = [p.strip() for chunk in text.split("\n") for p in chunk.split(";")]
    return [p for p in dict.fromkeys(parts) if p]


def _ids(row: Mapping[str, Any], keys: Sequence[str]) -> Dict[str, str]:
    return {k: str(row[k]).strip() for k in keys
            if row.get(k) is not None and str(row[k]).strip()}


def resolve_accession(r: _Reader, token: str) -> Dict[str, Any]:
    """``PW701720`` or a bare ``pathways.id``, to the pathway it names."""

    token = str(token).strip()
    if token.upper().startswith("PW") and not token.upper().startswith("PW_"):
        spv = r.one("SELECT id, pw_id, named_for_id, named_for_type, cached_name, "
                    "cached_subject, export, private_pathway FROM "
                    "super_pathway_visualizations WHERE pw_id=%s", (token.upper(),))
        if not spv:
            raise PathBankExtractError(f"no super_pathway_visualization carries pw_id {token!r}")
        if str(spv.get("named_for_type")) != "Pathway":
            raise PathBankExtractError(
                f"{token}: named_for_type is {spv.get('named_for_type')!r}, not 'Pathway'")
        return {"accession": spv["pw_id"], "spv_id": spv["id"],
                "pathway_id": int(spv["named_for_id"]), "spv": dict(spv)}
    if not token.isdigit():
        raise PathBankExtractError(f"not an accession or a numeric pathway id: {token!r}")
    pid = int(token)
    spv = r.one("SELECT id, pw_id FROM super_pathway_visualizations "
                "WHERE named_for_type='Pathway' AND named_for_id=%s ORDER BY id", (pid,))
    return {"accession": (spv or {}).get("pw_id"), "spv_id": (spv or {}).get("id"),
            "pathway_id": pid, "spv": dict(spv) if spv else None}


def _process(r: _Reader, process_type: str, process_id: int,
             elements_state: Optional[int]) -> Dict[str, Any]:
    table, el_table, el_fk, enz_table, enz_fk = PROCESS_TABLES[process_type]
    head = r.one(f"SELECT * FROM `{table}` WHERE id=%s", (process_id,)) or {}
    out: Dict[str, Any] = {
        "process_type": process_type, "process_id": process_id,
        "accession": next((head[k] for k in ("pwr_id", "pwi_id", "pwt_id", "pwrct_id")
                           if head.get(k)), None),
        "direction": head.get("direction"),
        "interaction_type": head.get("interaction_type"),
        "transport_type": head.get("transport_type"),
        "spontaneous": bool(head.get("spontaneous")) if head.get("spontaneous") is not None else None,
        "source": _ids(head, ("source_db", "source_id")) or None,
        "signature": str(head.get("signature") or "") or None,
        "biological_state": r.state(elements_state),
        "participants": [], "catalysts": [],
    }
    for e in r.rows(f"SELECT * FROM `{el_table}` WHERE `{el_fk}`=%s ORDER BY id", (process_id,)):
        side = str(e.get("type") or "")
        part: Dict[str, Any] = {
            "side": {"ReactionLeftElement": "left", "ReactionRightElement": "right",
                     "InteractionLeftElement": "left", "InteractionRightElement": "right",
                     "ReactionCoupledTransportLeftElement": "left",
                     "ReactionCoupledTransportRightElement": "right"}.get(side, side or "moved"),
            "stoichiometry": e.get("stoichiometry"),
            "currency": bool(e.get("currency")) if e.get("currency") is not None else None,
            "altext": e.get("altext"),
            **r.entity(e.get("element_type"), e.get("element_id")),
        }
        if e.get("left_biological_state_id") or e.get("right_biological_state_id"):
            part["from_state"] = r.state(e.get("left_biological_state_id"))
            part["to_state"] = r.state(e.get("right_biological_state_id"))
            part["transport_direction"] = e.get("direction")
        if e.get("biological_state_id"):
            part["state"] = r.state(e.get("biological_state_id"))
        out["participants"].append(part)
    if enz_table:
        col = "protein_complex_id"
        for z in r.rows(f"SELECT * FROM `{enz_table}` WHERE `{enz_fk}`=%s ORDER BY id",
                        (process_id,)):
            out["catalysts"].append({
                "enzyme_class": z.get("enzyme_class"), "altext": z.get("altext"),
                "state": r.state(z.get("biological_state_id")),
                **r.entity("ProteinComplex", z.get(col))})
    return out


def extract(conn: Any, token: str) -> Dict[str, Any]:
    """The whole curated pathway, as stored. Raises rather than returning a part."""

    r = _Reader(conn)
    ref = resolve_accession(r, token)
    pid = ref["pathway_id"]
    pw = r.one("SELECT p.*, s.name species_name, s.taxonomy_id, s.common_name "
               "FROM pathways p LEFT JOIN species s ON s.id=p.species_id WHERE p.id=%s", (pid,))
    if not pw:
        raise PathBankExtractError(f"{token}: pathways.id {pid} does not exist")

    rows = r.rows("SELECT DISTINCT process_type, process_id, biological_state_id FROM "
                  "pathway_elements WHERE pathway_id=%s ORDER BY process_type, process_id", (pid,))
    seen: set = set()
    processes: List[Dict[str, Any]] = []
    unhandled: List[Dict[str, Any]] = []
    for row in rows:
        ptype, pidn = str(row["process_type"] or ""), row["process_id"]
        if pidn in (None, "") or not str(pidn).isdigit():
            continue
        key = (ptype, int(pidn))
        if key in seen:
            continue
        seen.add(key)
        if ptype not in PROCESS_TABLES:
            unhandled.append({"process_type": ptype, "process_id": int(pidn)})
            continue
        processes.append(_process(r, ptype, int(pidn), row.get("biological_state_id")))

    refs = [{"pubmed_id": x.get("pubmed_id"), "citation": str(x.get("text") or "").strip()}
            for x in r.rows("SELECT pubmed_id, text FROM `references` "
                            "WHERE element_type='Pathway' AND element_id=%s ORDER BY id", (pid,))]
    subs = [x["sub_category"] for x in r.rows(
        "SELECT sub_category FROM pathway_sub_categories WHERE pathway_id=%s", (pid,))]
    sub_pathways = r.rows("SELECT id, reference_pathway_id, sub_pathway_type, altext "
                          "FROM sub_pathways WHERE pathway_id=%s ORDER BY id", (pid,))

    body = {
        "accession": ref["accession"], "pathway_id": pid, "spv_id": ref["spv_id"],
        "name": pw.get("name"), "subject": pw.get("subject"), "sub_categories": subs,
        "status": pw.get("status"),
        "species": {"name": pw.get("species_name"), "taxonomy_id": pw.get("taxonomy_id"),
                    "common_name": pw.get("common_name")},
        "description": pw.get("description"),
        "curator": {"user_id": pw.get("user_id"), "created_at": str(pw.get("created_at") or ""),
                    "updated_at": str(pw.get("updated_at") or "")},
        "mutability": {"private_pathway": bool(pw.get("private_pathway")),
                       "locked": bool(pw.get("locked")),
                       "status_is_null": pw.get("status") is None,
                       "note": "status NULL or locked=0 means the curator can still change it; "
                               "re-extract and compare content_sha256 before scoring"},
        "counts": {k: sum(1 for p in processes if p["process_type"] == k)
                   for k in sorted({p["process_type"] for p in processes})},
        "references": refs, "sub_pathways": sub_pathways,
        "processes": processes, "unhandled_process_types": unhandled,
    }
    payload = {"schema": SCHEMA_EXTRACT,
               "extracted_at": _dt.date.today().isoformat(),
               "source": "PathBank MySQL (read-only); accession resolved via "
                         "super_pathway_visualizations.pw_id",
               **body}
    payload["content_sha256"] = hashlib.sha256(
        json.dumps(body, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")
    ).hexdigest()
    return payload


# ---------------------------------------------------------------------------
# Draft case: the reactions in gold-supplement shape, quotes deliberately empty
# ---------------------------------------------------------------------------

def _term(part: Mapping[str, Any]) -> Dict[str, Any]:
    aliases = [a for a in ([part.get("short_name")] + list(part.get("synonyms") or []))
               if a and a != part.get("name")]
    term: Dict[str, Any] = {"name": part.get("name") or f"[unresolved {part.get('element_type')}"
                                                        f" {part.get('element_id')}]",
                            "aliases": aliases[:12],
                            "role": "cofactor" if part.get("currency") else "defining"}
    ids = part.get("identifiers") or {}
    if ids:
        term["pathbank_identifiers"] = ids
    if part.get("element_type") != "Compound":
        term["entity_kind"] = part.get("element_type")
    return term


def _enzyme(cats: Sequence[Mapping[str, Any]]) -> Optional[Dict[str, Any]]:
    if not cats:
        return None
    c = cats[0]
    comps = list(c.get("components") or [])
    aliases = [x for x in ([m.get("gene_name") for m in comps]
                           + [m.get("name") for m in comps]) if x and x != c.get("name")]
    ec = next((m.get("ec_numbers") for m in comps if m.get("ec_numbers")), None) \
        or c.get("enzyme_class")
    out: Dict[str, Any] = {"name": c.get("name"), "aliases": list(dict.fromkeys(aliases))[:12]}
    if ec:
        out["identifier"] = f"EC {ec}" if not str(ec).upper().startswith("EC") else str(ec)
    if len(cats) > 1:
        out["note"] = f"{len(cats)} catalysts recorded; others: " + \
                      ", ".join(str(x.get('name')) for x in cats[1:])
    if comps:
        out["components"] = [{"name": m.get("name"), "gene_name": m.get("gene_name"),
                              "uniprot": m.get("uniprot")} for m in comps]
    return out


def draft_case(extract_payload: Mapping[str, Any]) -> Dict[str, Any]:
    """The extract rendered as a gold-supplement DRAFT. Not loadable as gold."""

    reactions: List[Dict[str, Any]] = []
    interactions: List[Dict[str, Any]] = []
    transports: List[Dict[str, Any]] = []
    for p in extract_payload.get("processes") or ():
        ptype = p.get("process_type")
        if ptype in ("Reaction", "ReactionCoupledTransport"):
            left = [x for x in p["participants"] if x.get("side") == "left"]
            right = [x for x in p["participants"] if x.get("side") == "right"]
            reactions.append({
                "id": f"R{len(reactions) + 1}",
                "pathbank_id": p.get("accession") or p.get("process_id"),
                "process_type": ptype,
                "label": " + ".join(str(x.get("name")) for x in left) + " -> " +
                         " + ".join(str(x.get("name")) for x in right),
                "inputs": [_term(x) for x in left], "outputs": [_term(x) for x in right],
                "enzyme": _enzyme(p.get("catalysts") or ()),
                "reversible": str(p.get("direction") or "") in REVERSIBLE_DIRECTIONS,
                "compartment": (p.get("biological_state") or {}).get("location"),
                "quote": "", "supporting_quotes": [],
                "evidence_required": "A PathBank record is not evidence about a paper. A human "
                                     "or the paper-gold drafter must supply a verbatim quote "
                                     "from the pinned full text before this row is gold."})
        elif ptype == "Interaction":
            src = [x for x in p["participants"] if x.get("side") == "left"]
            tgt = [x for x in p["participants"] if x.get("side") == "right"]
            interactions.append({
                "id": f"I{len(interactions) + 1}",
                "pathbank_id": p.get("accession") or p.get("process_id"),
                "effect": str(p.get("interaction_type") or "").lower() or "unknown",
                "source": [_term(x) for x in src], "target": [_term(x) for x in tgt],
                "compartment": (p.get("biological_state") or {}).get("location"),
                "quote": ""})
        elif ptype == "Transport":
            moved = [x for x in p["participants"]]
            transports.append({
                "id": f"T{len(transports) + 1}",
                "pathbank_id": p.get("accession") or p.get("process_id"),
                "entity": [_term(x) for x in moved],
                "from_compartment": next(((x.get("from_state") or {}).get("location")
                                          for x in moved if x.get("from_state")), None),
                "to_compartment": next((((x.get("to_state") or {}).get("location"))
                                        for x in moved if x.get("to_state")), None),
                "transporter": _enzyme(p.get("catalysts") or ()),
                "quote": ""})

    sp = extract_payload.get("species") or {}
    return {
        "schema": SCHEMA_DRAFT,
        "drafter": {"prompt": "pathbank_gold.draft_case", "model": "none (database extract)",
                    "date": _dt.date.today().isoformat(), "mode": "pathbank_structural"},
        "case_id": None,
        "not_gold": "Quotes are empty by construction: these statements come from the lab's "
                    "PathBank record, not from any paper. gold.load_supplement refuses this "
                    "file. Use it only as the scope hint for paper_gold_drafter_v1.",
        "pathbank": {"accession": extract_payload.get("accession"),
                     "pathway_id": extract_payload.get("pathway_id"),
                     "name": extract_payload.get("name"),
                     "subject": extract_payload.get("subject"),
                     "content_sha256": extract_payload.get("content_sha256"),
                     "updated_at": (extract_payload.get("curator") or {}).get("updated_at"),
                     "references": extract_payload.get("references")},
        "case": {
            "supported_reactions_complete": False,
            "species": sp.get("name") or "", "scope": "",
            "source": {}, "audit": "",
            "reactions": reactions, "off_pathway_reactions": [], "summary_statements": [],
            "excluded_statements": [], "interactions": interactions, "transports": transports,
        },
        "pathbank_alignment": [], "paper_only": [], "uncertain": [],
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__ or "", formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("accessions", nargs="+", help="PW accession(s) or numeric pathway id(s)")
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--no-draft", action="store_true", help="write only the structural extract")
    args = ap.parse_args(list(argv) if argv is not None else None)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    conn = connect()
    try:
        for token in args.accessions:
            payload = extract(conn, token)
            stem = str(payload.get("accession") or f"pathway{payload['pathway_id']}")
            p = out_dir / f"{stem}__extract.json"
            p.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n",
                         encoding="utf-8")
            print(f"{stem}: pathway {payload['pathway_id']} '{payload['name']}' "
                  f"{payload['counts']} refs={len(payload['references'])} "
                  f"sha256={payload['content_sha256'][:16]} -> {p}")
            if payload["unhandled_process_types"]:
                print(f"  WARNING unhandled process types: {payload['unhandled_process_types']}")
            if not args.no_draft:
                d = out_dir / f"{stem}__draft_case.json"
                draft = draft_case(payload)
                d.write_text(json.dumps(draft, indent=2, ensure_ascii=False, default=str) + "\n",
                             encoding="utf-8")
                c = draft["case"]
                print(f"  draft: {len(c['reactions'])} reactions, {len(c['interactions'])} "
                      f"interactions, {len(c['transports'])} transports -> {d}")
    finally:
        conn.close()
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
