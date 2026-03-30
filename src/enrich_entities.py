from __future__ import annotations

import argparse
import json
import os
import re
import time
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional
from xml.etree import ElementTree

from map_ids import HttpClient


def _safe_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _safe_list(value: Any) -> List[Any]:
    return value if isinstance(value, list) else []


def _canonical(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip())


def _utc_now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _dedupe_preserve(values: List[str]) -> List[str]:
    out: List[str] = []
    seen: set = set()
    for value in values:
        text = _canonical(value)
        if not text:
            continue
        key = text.casefold()
        if key in seen:
            continue
        seen.add(key)
        out.append(text)
    return out


def _clean_list_of_strings(values: Any) -> List[str]:
    return _dedupe_preserve([_canonical(v) for v in _safe_list(values)])


def _split_text_list(value: Any) -> List[str]:
    if isinstance(value, list):
        return _clean_list_of_strings(value)
    text = _canonical(value)
    if not text:
        return []
    chunks = re.split(r"[;,|]", text)
    return _dedupe_preserve([_canonical(chunk) for chunk in chunks if _canonical(chunk)])


def _to_int(value: Any) -> Optional[int]:
    try:
        if value is None or str(value).strip() == "":
            return None
        return int(str(value).strip())
    except Exception:  # noqa: BLE001
        return None


def _to_float(value: Any) -> Optional[float]:
    try:
        if value is None or str(value).strip() == "":
            return None
        return float(str(value).strip())
    except Exception:  # noqa: BLE001
        return None


def _first_non_empty(*values: Any) -> str:
    for value in values:
        text = _canonical(value)
        if text:
            return text
    return ""


class EnrichmentCache:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.data: Dict[str, Dict[str, Any]] = {
            "proteins": {},
            "compounds": {},
        }
        if path.exists():
            try:
                loaded = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    self.data["proteins"] = _safe_dict(loaded.get("proteins"))
                    self.data["compounds"] = _safe_dict(loaded.get("compounds"))
            except Exception:  # noqa: BLE001
                pass

    def get(self, section: str, key: str) -> Optional[Dict[str, Any]]:
        item = _safe_dict(self.data.get(section)).get(key)
        return item if isinstance(item, dict) else None

    def set(self, section: str, key: str, value: Dict[str, Any]) -> None:
        self.data.setdefault(section, {})
        self.data[section][key] = value

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.data, indent=2, ensure_ascii=False), encoding="utf-8")


def _extract_go_bins(payload: Dict[str, Any]) -> Dict[str, List[str]]:
    go_mf: List[str] = []
    go_bp: List[str] = []
    go_cc: List[str] = []
    for ref in _safe_list(payload.get("uniProtKBCrossReferences")):
        if not isinstance(ref, dict):
            continue
        db = _canonical(ref.get("database")).upper()
        gid = _canonical(ref.get("id"))
        if db != "GO" or not gid:
            continue
        term = ""
        for prop in _safe_list(ref.get("properties")):
            if not isinstance(prop, dict):
                continue
            key = _canonical(prop.get("key")).upper()
            if key in {"GO TERM", "TERM", "GOTERM"}:
                term = _canonical(prop.get("value"))
                break
        if term.startswith("F:"):
            go_mf.append(gid)
        elif term.startswith("P:"):
            go_bp.append(gid)
        elif term.startswith("C:"):
            go_cc.append(gid)
        else:
            go_bp.append(gid)
    return {
        "go_mf_ids": _dedupe_preserve(go_mf),
        "go_bp_ids": _dedupe_preserve(go_bp),
        "go_cellular_component_ids": _dedupe_preserve(go_cc),
    }


def _extract_uniprot_ec_numbers(payload: Dict[str, Any]) -> List[str]:
    ec_ids: List[str] = []

    def _scan(node: Any) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                k = _canonical(key).casefold()
                if k in {"ecnumber", "ec"}:
                    if isinstance(value, dict):
                        text = _canonical(value.get("value"))
                        if text:
                            ec_ids.append(text)
                    else:
                        text = _canonical(value)
                        if text:
                            ec_ids.append(text)
                else:
                    _scan(value)
        elif isinstance(node, list):
            for item in node:
                _scan(item)

    _scan(payload)
    cleaned: List[str] = []
    for item in ec_ids:
        parts = re.findall(r"\d+\.\d+\.\d+\.[\d-]+", item)
        if parts:
            cleaned.extend(parts)
            continue
        text = _canonical(item)
        if text:
            cleaned.append(text)
    return _dedupe_preserve(cleaned)


def _parse_uniprot_record(accession: str, payload: Dict[str, Any], source_url: str) -> Dict[str, Any]:
    protein_desc = _safe_dict(payload.get("proteinDescription"))
    recommended = _safe_dict(protein_desc.get("recommendedName"))
    recommended_name = _canonical(_safe_dict(recommended.get("fullName")).get("value"))

    alt_names: List[str] = []
    for row in _safe_list(protein_desc.get("alternativeNames")):
        if not isinstance(row, dict):
            continue
        alt_names.append(_canonical(_safe_dict(row.get("fullName")).get("value")))
    alt_names = _dedupe_preserve(alt_names)

    gene_names: List[str] = []
    for gene in _safe_list(payload.get("genes")):
        if not isinstance(gene, dict):
            continue
        gene_names.append(_canonical(_safe_dict(gene.get("geneName")).get("value")))
        for syn in _safe_list(gene.get("synonyms")):
            if isinstance(syn, dict):
                gene_names.append(_canonical(syn.get("value")))
    gene_names = _dedupe_preserve(gene_names)

    organism = _safe_dict(payload.get("organism"))
    organism_name = _canonical(organism.get("scientificName"))
    taxonomy_id = _canonical(organism.get("taxonId"))

    function_texts: List[str] = []
    pathways: List[str] = []
    catalytic_activity: List[str] = []
    subcellular_locations: List[str] = []

    for comment in _safe_list(payload.get("comments")):
        if not isinstance(comment, dict):
            continue
        ctype = _canonical(comment.get("commentType")).upper()
        text_values = [
            _canonical(_safe_dict(text_row).get("value"))
            for text_row in _safe_list(comment.get("texts"))
            if isinstance(text_row, dict)
        ]
        text_values = [t for t in text_values if t]

        if ctype == "FUNCTION":
            function_texts.extend(text_values)
        elif ctype == "PATHWAY":
            pathways.extend(text_values)
        elif ctype == "CATALYTIC ACTIVITY":
            reaction = _safe_dict(comment.get("reaction"))
            catalytic_activity.append(_canonical(reaction.get("name")))
            catalytic_activity.extend(text_values)
        elif ctype == "SUBCELLULAR LOCATION":
            for row in _safe_list(comment.get("subcellularLocations")):
                if not isinstance(row, dict):
                    continue
                location = _canonical(_safe_dict(row.get("location")).get("value"))
                if location:
                    subcellular_locations.append(location)

    function_text = ""
    if function_texts:
        joined = " ".join(function_texts)
        function_text = joined[:2000]

    feature_rows = _safe_list(payload.get("features"))
    tm_count = 0
    signal_peptide = {"present": False, "region": ""}
    for feature in feature_rows:
        if not isinstance(feature, dict):
            continue
        ftype = _canonical(feature.get("type")).casefold()
        if "transmembrane" in ftype:
            tm_count += 1
        if ftype == "signal peptide":
            signal_peptide["present"] = True
            location = _safe_dict(feature.get("location"))
            start = _canonical(_safe_dict(location.get("start")).get("value"))
            end = _canonical(_safe_dict(location.get("end")).get("value"))
            if start and end:
                signal_peptide["region"] = f"{start}-{end}"

    go_bins = _extract_go_bins(payload)
    pdb_ids: List[str] = []
    reactome_ids: List[str] = []
    string_id = ""
    keywords: List[str] = []

    for row in _safe_list(payload.get("keywords")):
        if isinstance(row, dict):
            keywords.append(_canonical(row.get("name")))

    for ref in _safe_list(payload.get("uniProtKBCrossReferences")):
        if not isinstance(ref, dict):
            continue
        db = _canonical(ref.get("database")).upper()
        rid = _canonical(ref.get("id"))
        if not rid:
            continue
        if db == "PDB":
            pdb_ids.append(rid)
        elif db == "REACTOME":
            reactome_ids.append(rid)
        elif db == "STRING" and not string_id:
            string_id = rid

    protein_existence = _first_non_empty(
        _safe_dict(payload.get("proteinExistence")).get("value"),
        payload.get("proteinExistence"),
    )
    sequence_length = _to_int(_safe_dict(payload.get("sequence")).get("length"))

    return {
        "status": "ok",
        "uniprot_id": accession,
        "recommended_name": recommended_name,
        "alternative_names": alt_names,
        "gene_names": gene_names,
        "organism": organism_name,
        "taxonomy_id": taxonomy_id,
        "function_text": function_text,
        "catalytic_activity": _dedupe_preserve([v for v in catalytic_activity if v]),
        "enzyme_commission": _extract_uniprot_ec_numbers(payload),
        "pathways": _dedupe_preserve([v for v in pathways if v]),
        "keywords": _dedupe_preserve([v for v in keywords if v]),
        "subcellular_locations": _dedupe_preserve([v for v in subcellular_locations if v]),
        "go_mf_ids": go_bins["go_mf_ids"],
        "go_bp_ids": go_bins["go_bp_ids"],
        "go_cellular_component_ids": go_bins["go_cellular_component_ids"],
        "protein_existence": protein_existence,
        "sequence_length": sequence_length if sequence_length is not None else 0,
        "transmembrane_regions_count": tm_count,
        "signal_peptide": signal_peptide,
        "pdb_ids": _dedupe_preserve(pdb_ids),
        "reactome_ids": _dedupe_preserve(reactome_ids),
        "string_id": string_id,
        "provenance": {
            "source": "UniProt",
            "query": accession,
            "source_url": source_url,
            "retrieved_at": _utc_now_iso(),
        },
    }


def _fetch_uniprot_enrichment(client: HttpClient, uniprot_id: str) -> Dict[str, Any]:
    accession = _canonical(uniprot_id).upper()
    url = f"https://rest.uniprot.org/uniprotkb/{accession}.json"
    try:
        resp = client.get(url)
    except Exception as exc:  # noqa: BLE001
        return {
            "status": "error",
            "error_message": f"request_failed:{exc}",
            "provenance": {
                "source": "UniProt",
                "query": accession,
                "source_url": url,
                "retrieved_at": _utc_now_iso(),
            },
        }
    if resp.status_code != 200:
        return {
            "status": "error",
            "error_message": f"HTTP {resp.status_code}",
            "provenance": {
                "source": "UniProt",
                "query": accession,
                "source_url": url,
                "retrieved_at": _utc_now_iso(),
            },
        }
    try:
        payload = _safe_dict(resp.json())
    except Exception as exc:  # noqa: BLE001
        return {
            "status": "error",
            "error_message": f"json_parse_failed:{exc}",
            "provenance": {
                "source": "UniProt",
                "query": accession,
                "source_url": url,
                "retrieved_at": _utc_now_iso(),
            },
        }
    parsed = _parse_uniprot_record(accession, payload, url)
    return {
        "status": "ok",
        "parsed": parsed,
        "raw": payload,
    }


def _xml_tag(tag: str) -> str:
    return tag.split("}")[-1] if "}" in tag else tag


def _normalize_chebi_id(raw: str) -> str:
    text = _canonical(raw).upper()
    if not text:
        return ""
    if text.startswith("CHEBI:"):
        return text
    if text.startswith("CHEBI"):
        digits = re.sub(r"[^0-9]", "", text)
        return f"CHEBI:{digits}" if digits else text
    if text.isdigit():
        return f"CHEBI:{text}"
    return text


def _fetch_chebi_enrichment(client: HttpClient, chebi_id: str) -> Dict[str, Any]:
    cid = _normalize_chebi_id(chebi_id)
    if not cid:
        return {"status": "error", "error_message": "missing_chebi_id"}

    url = "https://www.ebi.ac.uk/webservices/chebi/2.0/test/getCompleteEntity"
    try:
        resp = client.get(url, params={"chebiId": cid})
    except Exception as exc:  # noqa: BLE001
        return {
            "status": "error",
            "error_message": f"request_failed:{exc}",
            "provenance": {"source": "ChEBI", "id": cid, "source_url": url, "retrieved_at": _utc_now_iso()},
        }
    if resp.status_code != 200:
        status = "not_found" if resp.status_code in {400, 404} else "error"
        return {
            "status": status,
            "error_message": f"HTTP {resp.status_code}",
            "provenance": {"source": "ChEBI", "id": cid, "source_url": url, "retrieved_at": _utc_now_iso()},
        }
    try:
        root = ElementTree.fromstring(resp.text)
    except ElementTree.ParseError as exc:
        return {
            "status": "error",
            "error_message": f"xml_parse_error:{exc}",
            "provenance": {"source": "ChEBI", "id": cid, "source_url": url, "retrieved_at": _utc_now_iso()},
        }

    tag_values: Dict[str, List[str]] = {}
    for node in root.iter():
        text = _canonical(node.text)
        if not text:
            continue
        tag_values.setdefault(_xml_tag(node.tag).casefold(), []).append(text)

    primary_name = _first_non_empty(
        (_safe_list(tag_values.get("chebiasciiname")) or [""])[0],
        (_safe_list(tag_values.get("chebiname")) or [""])[0],
    )
    synonyms = _dedupe_preserve(_safe_list(tag_values.get("synonyms")) + _safe_list(tag_values.get("synonym")))
    inchi = _first_non_empty((_safe_list(tag_values.get("inchi")) or [""])[0])
    inchikey = _first_non_empty((_safe_list(tag_values.get("inchikey")) or [""])[0])
    smiles = _first_non_empty((_safe_list(tag_values.get("smiles")) or [""])[0])
    formula = _first_non_empty(
        (_safe_list(tag_values.get("formula")) or [""])[0],
        (_safe_list(tag_values.get("chemicalformula")) or [""])[0],
    )
    charge = _to_int((_safe_list(tag_values.get("charge")) or [""])[0])
    mass_mono = _to_float((_safe_list(tag_values.get("monoisotopicmass")) or [""])[0])
    mass_avg = _to_float((_safe_list(tag_values.get("mass")) or [""])[0])

    roles: List[str] = []
    ontology_parents: List[str] = []
    for node in root.iter():
        if _xml_tag(node.tag).casefold() != "ontologyparent":
            continue
        row = {_xml_tag(ch.tag).casefold(): _canonical(ch.text) for ch in list(node)}
        pid = _canonical(row.get("chebiid"))
        pname = _canonical(row.get("chebiname"))
        ptype = _canonical(row.get("type")).casefold()
        if pid or pname:
            ontology_parents.append(_first_non_empty(f"{pid}:{pname}" if pid and pname else "", pid, pname))
        if "role" in ptype and pname:
            roles.append(pname)

    cross_refs: Dict[str, str] = {"chebi_id": cid}
    for node in root.iter():
        if _xml_tag(node.tag).casefold() != "databaselink":
            continue
        db_name = ""
        accession = ""
        for child in list(node):
            ctag = _xml_tag(child.tag).casefold()
            ctext = _canonical(child.text)
            if ctag in {"type", "dbname"}:
                db_name = ctext.casefold()
            elif ctag in {"data", "accessionnumber"}:
                accession = ctext
        if not db_name or not accession:
            continue
        if "kegg" in db_name:
            cross_refs.setdefault("kegg_id", accession)
        elif "hmdb" in db_name:
            cross_refs.setdefault("hmdb_id", accession)
        elif "drugbank" in db_name:
            cross_refs.setdefault("drugbank_id", accession)
        elif "pubchem" in db_name:
            cross_refs.setdefault("pubchem_id", accession)

    return {
        "status": "ok",
        "source": "ChEBI",
        "id": cid,
        "primary_name": primary_name,
        "synonyms": synonyms,
        "inchi": inchi,
        "inchikey": inchikey,
        "smiles": smiles,
        "formula": formula,
        "charge": charge if charge is not None else 0,
        "mass_monoisotopic": mass_mono if mass_mono is not None else 0.0,
        "mass_average": mass_avg if mass_avg is not None else 0.0,
        "chebi_roles": _dedupe_preserve(roles),
        "ontology_parents": _dedupe_preserve(ontology_parents),
        "cross_references": cross_refs,
        "provenance": {
            "source": "ChEBI",
            "id": cid,
            "source_url": url,
            "retrieved_at": _utc_now_iso(),
            "status": "ok",
        },
    }


def _normalize_kegg_id(raw: str) -> str:
    text = _canonical(raw).upper()
    if not text:
        return ""
    return text.replace("CPD:", "")


def _parse_kegg_entry_text(text: str) -> Dict[str, Any]:
    rows: Dict[str, List[str]] = {}
    current_key = ""
    for line in text.splitlines():
        key = line[:12].strip()
        value = _canonical(line[12:])
        if key:
            current_key = key
            rows.setdefault(key, []).append(value)
        elif current_key:
            rows.setdefault(current_key, []).append(value)

    names: List[str] = []
    for row in rows.get("NAME", []):
        names.extend([_canonical(part) for part in row.split(";")])
    names = [n for n in names if n]

    pathways: List[str] = []
    for row in rows.get("PATHWAY", []):
        parts = row.split(" ", 1)
        pathways.append(parts[1].strip() if len(parts) > 1 else row)

    kegg_class = _dedupe_preserve(rows.get("CLASS", []) + rows.get("BRITE", []))
    exact_mass = _to_float((rows.get("EXACT_MASS", [""])[0] if rows.get("EXACT_MASS") else ""))
    avg_mass = _to_float((rows.get("MOL_WEIGHT", [""])[0] if rows.get("MOL_WEIGHT") else ""))
    formula = _canonical(rows.get("FORMULA", [""])[0] if rows.get("FORMULA") else "")

    cross_refs: Dict[str, str] = {}
    for row in rows.get("DBLINKS", []):
        if ":" not in row:
            continue
        key, value = row.split(":", 1)
        db = _canonical(key).casefold()
        vid = _canonical(value).split(" ")[0]
        if "drugbank" in db:
            cross_refs.setdefault("drugbank_id", vid)
        elif "pubchem" in db:
            cross_refs.setdefault("pubchem_id", vid)
        elif "chebi" in db:
            cross_refs.setdefault("chebi_id", vid)
        elif "hmdb" in db:
            cross_refs.setdefault("hmdb_id", vid)

    return {
        "primary_name": names[0] if names else "",
        "synonyms": _dedupe_preserve(names),
        "formula": formula,
        "mass_monoisotopic": exact_mass if exact_mass is not None else 0.0,
        "mass_average": avg_mass if avg_mass is not None else 0.0,
        "pathways": _dedupe_preserve([p for p in pathways if p]),
        "kegg_class": kegg_class,
        "cross_references": cross_refs,
    }


def _fetch_kegg_enrichment(client: HttpClient, kegg_id: str) -> Dict[str, Any]:
    kid = _normalize_kegg_id(kegg_id)
    if not kid:
        return {"status": "error", "error_message": "missing_kegg_id"}
    url = f"https://rest.kegg.jp/get/cpd:{kid}"
    try:
        resp = client.get(url)
    except Exception as exc:  # noqa: BLE001
        return {
            "status": "error",
            "error_message": f"request_failed:{exc}",
            "provenance": {"source": "KEGG", "id": kid, "source_url": url, "retrieved_at": _utc_now_iso()},
        }
    if resp.status_code != 200 or not _canonical(resp.text):
        status = "not_found" if resp.status_code in {400, 404} else "error"
        return {
            "status": status,
            "error_message": f"HTTP {resp.status_code}",
            "provenance": {"source": "KEGG", "id": kid, "source_url": url, "retrieved_at": _utc_now_iso()},
        }
    parsed = _parse_kegg_entry_text(resp.text)
    return {
        "status": "ok",
        "source": "KEGG",
        "id": kid,
        **parsed,
        "cross_references": {"kegg_id": kid, **_safe_dict(parsed.get("cross_references"))},
        "provenance": {"source": "KEGG", "id": kid, "source_url": url, "retrieved_at": _utc_now_iso(), "status": "ok"},
    }


def _normalize_hmdb_id(raw: str) -> str:
    text = _canonical(raw).upper()
    if not text:
        return ""
    if text.startswith("HMDB"):
        return text
    digits = re.sub(r"[^0-9]", "", text)
    return f"HMDB{digits}" if digits else text


def _fetch_hmdb_json_if_configured(client: HttpClient, hmdb_id: str) -> Optional[Dict[str, Any]]:
    api_url = _canonical(os.getenv("HMDB_API_URL", ""))
    if not api_url:
        return None
    api_key = _canonical(os.getenv("HMDB_API_KEY", ""))
    headers: Dict[str, str] = {"Accept": "application/json"}
    if api_key:
        auth_header = _canonical(os.getenv("HMDB_API_AUTH_HEADER", "X-API-Key")) or "X-API-Key"
        headers[auth_header] = api_key
    try:
        resp = client.get(
            api_url,
            params={
                "id": hmdb_id,
                "hmdb_id": hmdb_id,
                "accession": hmdb_id,
                "query": hmdb_id,
                "q": hmdb_id,
            },
            headers=headers,
        )
        if resp.status_code != 200:
            return None
        payload = resp.json()
    except Exception:  # noqa: BLE001
        return None

    rows: List[Dict[str, Any]] = []
    if isinstance(payload, dict):
        for key in ["results", "data", "items", "metabolites"]:
            value = payload.get(key)
            if isinstance(value, list):
                rows = [item for item in value if isinstance(item, dict)]
                break
        if not rows:
            rows = [payload]
    elif isinstance(payload, list):
        rows = [item for item in payload if isinstance(item, dict)]

    if not rows:
        return None

    exact = None
    for row in rows:
        rid = _normalize_hmdb_id(_first_non_empty(row.get("hmdb_id"), row.get("accession"), row.get("id")))
        if rid == hmdb_id:
            exact = row
            break
    return exact if isinstance(exact, dict) else rows[0]


def _fetch_hmdb_html(client: HttpClient, hmdb_id: str) -> Optional[str]:
    url = f"https://hmdb.ca/metabolites/{hmdb_id}"
    try:
        resp = client.get(url)
    except Exception:  # noqa: BLE001
        return None
    if resp.status_code != 200:
        return None
    return resp.text


def _extract_html_field(html: str, header_text: str) -> str:
    pattern = (
        r"<th[^>]*>\s*" + re.escape(header_text) + r"\s*</th>\s*"
        r"<td[^>]*>\s*(.*?)\s*</td>"
    )
    match = re.search(pattern, html, flags=re.IGNORECASE | re.DOTALL)
    if not match:
        return ""
    value = re.sub(r"<[^>]+>", " ", match.group(1))
    return _canonical(value)


def _fetch_hmdb_enrichment(client: HttpClient, hmdb_id: str) -> Dict[str, Any]:
    hid = _normalize_hmdb_id(hmdb_id)
    if not hid:
        return {"status": "error", "error_message": "missing_hmdb_id"}

    row = _fetch_hmdb_json_if_configured(client, hid)
    if isinstance(row, dict):
        name = _first_non_empty(row.get("name"), row.get("metabolite_name"))
        synonyms = _split_text_list(row.get("synonyms"))
        pathways: List[str] = []
        for pathway in _safe_list(row.get("pathways")):
            if isinstance(pathway, dict):
                pathways.append(_first_non_empty(pathway.get("name"), pathway.get("pathway_name")))
            else:
                pathways.append(_canonical(pathway))
        pathways = _dedupe_preserve([p for p in pathways if p])
        tissue_locations = _split_text_list(row.get("tissue_locations"))
        biofluid_locations = _split_text_list(row.get("biofluid_locations"))
        cellular_locations = _split_text_list(
            _first_non_empty(row.get("cellular_locations"), row.get("cellular_location"), row.get("subcellular_locations"))
        )
        description = _first_non_empty(row.get("description"), row.get("summary"), row.get("desc"))
        taxonomy_raw = row.get("taxonomy")
        taxonomy: Dict[str, Any] = {}
        if isinstance(taxonomy_raw, dict):
            taxonomy = {k: v for k, v in taxonomy_raw.items() if _canonical(v)}
        else:
            taxonomy = {
                "kingdom": _first_non_empty(row.get("kingdom")),
                "super_class": _first_non_empty(row.get("super_class")),
                "class": _first_non_empty(row.get("class")),
                "sub_class": _first_non_empty(row.get("sub_class")),
                "direct_parent": _first_non_empty(row.get("direct_parent")),
            }
            taxonomy = {k: v for k, v in taxonomy.items() if _canonical(v)}
        cross_refs = {
            "hmdb_id": hid,
            "kegg_id": _first_non_empty(row.get("kegg_id")),
            "chebi_id": _first_non_empty(row.get("chebi_id")),
            "pubchem_id": _first_non_empty(row.get("pubchem_id")),
            "drugbank_id": _first_non_empty(row.get("drugbank_id")),
        }
        for key in ["xrefs", "cross_references", "external_links"]:
            links = row.get(key)
            if isinstance(links, dict):
                for lk, lv in links.items():
                    normalized_key = _canonical(lk).casefold()
                    text = _canonical(lv)
                    if not text:
                        continue
                    if "kegg" in normalized_key:
                        cross_refs.setdefault("kegg_id", text)
                    elif "chebi" in normalized_key:
                        cross_refs.setdefault("chebi_id", text)
                    elif "pubchem" in normalized_key:
                        cross_refs.setdefault("pubchem_id", text)
                    elif "drugbank" in normalized_key:
                        cross_refs.setdefault("drugbank_id", text)
            elif isinstance(links, list):
                for item in links:
                    if not isinstance(item, dict):
                        continue
                    db_name = _canonical(item.get("database")).casefold()
                    db_value = _first_non_empty(item.get("id"), item.get("accession"), item.get("value"))
                    if not db_name or not db_value:
                        continue
                    if "kegg" in db_name:
                        cross_refs.setdefault("kegg_id", db_value)
                    elif "chebi" in db_name:
                        cross_refs.setdefault("chebi_id", db_value)
                    elif "pubchem" in db_name:
                        cross_refs.setdefault("pubchem_id", db_value)
                    elif "drugbank" in db_name:
                        cross_refs.setdefault("drugbank_id", db_value)
        cross_refs = {k: v for k, v in cross_refs.items() if v}
        return {
            "status": "ok",
            "source": "HMDB",
            "id": hid,
            "accession": hid,
            "primary_name": name,
            "synonyms": synonyms,
            "description": description,
            "taxonomy": taxonomy,
            "ontology_classification": taxonomy,
            "inchi": _first_non_empty(row.get("inchi")),
            "inchikey": _first_non_empty(row.get("inchikey")),
            "smiles": _first_non_empty(row.get("smiles")),
            "formula": _first_non_empty(row.get("formula")),
            "charge": _to_int(row.get("charge")) or 0,
            "mass_monoisotopic": _to_float(
                _first_non_empty(row.get("monoisotopic_molecular_weight"), row.get("monoisotopic_mass"))
            )
            or 0.0,
            "mass_average": _to_float(_first_non_empty(row.get("average_molecular_weight"), row.get("molecular_weight")))
            or 0.0,
            "pathways": pathways,
            "cellular_locations": cellular_locations,
            "tissue_locations": tissue_locations,
            "biofluid_locations": biofluid_locations,
            "cross_references": cross_refs,
            "external_links": _safe_dict(row.get("external_links")),
            "provenance": {
                "source": "HMDB",
                "id": hid,
                "source_url": _canonical(os.getenv("HMDB_API_URL", "")) or "configured_hmdb_api",
                "retrieved_at": _utc_now_iso(),
                "status": "ok",
            },
        }

    url = f"https://hmdb.ca/metabolites/{hid}"
    try:
        resp = client.get(url)
    except Exception as exc:  # noqa: BLE001
        return {
            "status": "error",
            "error_message": f"hmdb_request_failed:{exc}",
            "provenance": {
                "source": "HMDB",
                "id": hid,
                "source_url": url,
                "retrieved_at": _utc_now_iso(),
            },
        }
    if resp.status_code == 404:
        return {
            "status": "not_found",
            "error_message": "hmdb_not_found",
            "provenance": {
                "source": "HMDB",
                "id": hid,
                "source_url": url,
                "retrieved_at": _utc_now_iso(),
            },
        }
    if resp.status_code != 200:
        status = "not_found" if 400 <= int(resp.status_code) < 500 else "error"
        return {
            "status": status,
            "error_message": f"HTTP {resp.status_code}",
            "provenance": {
                "source": "HMDB",
                "id": hid,
                "source_url": url,
                "retrieved_at": _utc_now_iso(),
            },
        }

    html = resp.text
    title_match = re.search(r"<title>(.*?)</title>", html, flags=re.IGNORECASE | re.DOTALL)
    title = _canonical(re.sub(r"\s*-\s*HMDB.*$", "", title_match.group(1))) if title_match else ""
    return {
        "status": "ok",
        "source": "HMDB",
        "id": hid,
        "accession": hid,
        "primary_name": title,
        "synonyms": _split_text_list(_extract_html_field(html, "Synonyms")),
        "description": _extract_html_field(html, "Description"),
        "taxonomy": {},
        "ontology_classification": {},
        "inchi": _extract_html_field(html, "InChI Identifier"),
        "inchikey": _extract_html_field(html, "InChI Key"),
        "smiles": _extract_html_field(html, "SMILES"),
        "formula": _extract_html_field(html, "Chemical Formula"),
        "charge": _to_int(_extract_html_field(html, "Formal Charge")) or 0,
        "mass_monoisotopic": _to_float(_extract_html_field(html, "Monoisotopic Molecular Weight")) or 0.0,
        "mass_average": _to_float(_extract_html_field(html, "Average Molecular Weight")) or 0.0,
        "pathways": _split_text_list(_extract_html_field(html, "Pathways")),
        "cellular_locations": _split_text_list(_extract_html_field(html, "Cellular Locations")),
        "tissue_locations": _split_text_list(_extract_html_field(html, "Tissue Locations")),
        "biofluid_locations": _split_text_list(_extract_html_field(html, "Biofluid Locations")),
        "cross_references": {"hmdb_id": hid},
        "external_links": {},
        "provenance": {
            "source": "HMDB",
            "id": hid,
            "source_url": url,
            "retrieved_at": _utc_now_iso(),
            "status": "ok",
        },
    }


def _init_compound_enrichment(mapped_ids: Dict[str, Any]) -> Dict[str, Any]:
    cross_refs = {}
    for key, value in _safe_dict(mapped_ids).items():
        text = _canonical(value)
        if not text:
            continue
        normalized = key.strip().lower()
        if normalized in {"chebi", "hmdb", "kegg", "pubchem", "drugbank"}:
            cross_refs[f"{normalized}_id"] = text
        else:
            cross_refs[normalized] = text
    return {
        "enrichment_status": "ok",
        "primary_name": "",
        "synonyms": [],
        "description": "",
        "taxonomy": {},
        "ontology_classification": {},
        "inchi": "",
        "inchikey": "",
        "smiles": "",
        "formula": "",
        "charge": 0,
        "mass_monoisotopic": 0.0,
        "mass_average": 0.0,
        "chebi_roles": [],
        "ontology_parents": [],
        "kegg_class": [],
        "pathways": [],
        "cellular_locations": [],
        "tissue_locations": [],
        "biofluid_locations": [],
        "cross_references": cross_refs,
        "kegg": {},
        "chebi": {},
        "hmdb": {},
        "is_generic_class": False,
        "sources": [],
    }


def _is_generic_class_compound(name: str) -> bool:
    norm = _canonical(name).casefold()
    generic_markers = [
        "triacylglycerol",
        "diacylglycerol",
        "monoglyceride",
    ]
    return any(marker in norm for marker in generic_markers)


def _merge_compound_source(target: Dict[str, Any], source: Dict[str, Any]) -> None:
    status = _canonical(source.get("status")).lower() or "error"
    if status not in {"ok", "not_found", "error"}:
        status = "error"
    provenance = _safe_dict(source.get("provenance"))
    source_name = _canonical(source.get("source")) or _canonical(provenance.get("source")) or "unknown"
    source_id = _canonical(source.get("id")) or _canonical(provenance.get("id"))
    source_key = source_name.strip().lower()

    source_payload = deepcopy(source)
    source_payload["status"] = status
    target[source_key] = source_payload

    source_entry = {
        "source": source_name,
        "id": source_id,
        "retrieved_at": _canonical(provenance.get("retrieved_at")) or _utc_now_iso(),
        "status": status,
        "error_if_any": _canonical(source.get("error_message")),
    }
    target["sources"].append(source_entry)

    cross_refs = _safe_dict(target.get("cross_references"))
    for key, value in _safe_dict(source.get("cross_references")).items():
        text = _canonical(value)
        if text and key not in cross_refs:
            cross_refs[key] = text
    target["cross_references"] = cross_refs


def _source_block(target: Dict[str, Any], source_name: str) -> Dict[str, Any]:
    block = _safe_dict(target.get(source_name))
    if _canonical(block.get("status")).lower() != "ok":
        return {}
    return block


def _pick_scalar_from_sources(target: Dict[str, Any], key: str, *, numeric: bool = False) -> Any:
    order = ["chebi", "pubchem", "hmdb", "kegg"]
    for src in order:
        block = _source_block(target, src)
        if not block:
            continue
        value = block.get(key)
        if numeric:
            if isinstance(value, (int, float)) and float(value) != 0.0:
                return value
        else:
            text = _canonical(value)
            if text:
                return text
    return 0.0 if numeric else ""


def _recompute_compound_merged(target: Dict[str, Any]) -> None:
    target["primary_name"] = _pick_scalar_from_sources(target, "primary_name")
    target["description"] = _pick_scalar_from_sources(target, "description")
    target["inchi"] = _pick_scalar_from_sources(target, "inchi")
    target["inchikey"] = _pick_scalar_from_sources(target, "inchikey")
    target["smiles"] = _pick_scalar_from_sources(target, "smiles")
    target["formula"] = _pick_scalar_from_sources(target, "formula")
    target["charge"] = int(_pick_scalar_from_sources(target, "charge", numeric=True) or 0)
    target["mass_monoisotopic"] = float(_pick_scalar_from_sources(target, "mass_monoisotopic", numeric=True) or 0.0)
    target["mass_average"] = float(_pick_scalar_from_sources(target, "mass_average", numeric=True) or 0.0)

    all_synonyms: List[str] = []
    for src in ["chebi", "hmdb", "kegg"]:
        all_synonyms.extend(_clean_list_of_strings(_safe_dict(target.get(src)).get("synonyms")))
    target["synonyms"] = _dedupe_preserve(all_synonyms)

    chebi = _source_block(target, "chebi")
    hmdb = _source_block(target, "hmdb")
    kegg = _source_block(target, "kegg")

    target["chebi_roles"] = _clean_list_of_strings(chebi.get("chebi_roles"))
    target["ontology_parents"] = _clean_list_of_strings(chebi.get("ontology_parents"))
    target["kegg_class"] = _clean_list_of_strings(kegg.get("kegg_class"))
    target["pathways"] = _dedupe_preserve(
        _clean_list_of_strings(kegg.get("pathways")) + _clean_list_of_strings(hmdb.get("pathways"))
    )
    target["cellular_locations"] = _clean_list_of_strings(hmdb.get("cellular_locations"))
    target["tissue_locations"] = _clean_list_of_strings(hmdb.get("tissue_locations"))
    target["biofluid_locations"] = _clean_list_of_strings(hmdb.get("biofluid_locations"))
    target["taxonomy"] = _safe_dict(hmdb.get("taxonomy"))
    target["ontology_classification"] = _safe_dict(hmdb.get("ontology_classification"))

    statuses = [_canonical(row.get("status")).lower() for row in _safe_list(target.get("sources")) if isinstance(row, dict)]
    ok_count = sum(1 for status in statuses if status == "ok")
    non_ok_count = sum(1 for status in statuses if status in {"not_found", "error"})

    if ok_count == 0:
        target["enrichment_status"] = "error"
    elif non_ok_count > 0:
        target["enrichment_status"] = "ok_with_warnings"
    else:
        target["enrichment_status"] = "ok"


def _best_id_for_compound_dump(enrichment: Dict[str, Any]) -> str:
    refs = _safe_dict(enrichment.get("cross_references"))
    for key in ["hmdb_id", "chebi_id", "kegg_id", "pubchem_id", "drugbank_id"]:
        value = _canonical(refs.get(key))
        if value:
            return value
    return "unmapped"


def _build_enrichment_dump(payload: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    entities = _safe_dict(payload.get("entities"))

    for row in _safe_list(entities.get("compounds")):
        if not isinstance(row, dict):
            continue
        name = _canonical(row.get("name"))
        enrichment = _safe_dict(_safe_dict(row.get("enrichment")).get("compound"))
        if not name or not enrichment:
            continue
        best_id = _best_id_for_compound_dump(enrichment)
        key = f"compound:{name}|{best_id}"
        out[key] = enrichment

    for row in _safe_list(entities.get("proteins")):
        if not isinstance(row, dict):
            continue
        name = _canonical(row.get("name"))
        uniprot = _safe_dict(_safe_dict(_safe_dict(row.get("enrichment")).get("protein")).get("uniprot"))
        if not name or not uniprot:
            continue
        uid = _canonical(uniprot.get("uniprot_id")) or "unmapped"
        key = f"protein:{name}|{uid}"
        out[key] = uniprot

    return out


def enrich_payload(
    payload: Dict[str, Any],
    *,
    cache_path: Path,
) -> Dict[str, Any]:
    working = deepcopy(payload)
    cache = EnrichmentCache(cache_path)
    client = HttpClient()

    report: Dict[str, Any] = {
        "summary": {
            "proteins_total": 0,
            "proteins_with_uniprot": 0,
            "proteins_enriched_ok": 0,
            "proteins_enriched_error": 0,
            "compounds_total": 0,
            "compounds_with_ids": 0,
            "compounds_enriched_ok": 0,
            "compounds_enriched_ok_with_warnings": 0,
            "compounds_enriched_error": 0,
            "cache_hits": 0,
            "api_calls": 0,
        },
        "calls": {"uniprot": 0, "chebi": 0, "hmdb": 0, "kegg": 0},
        "entities": [],
    }

    entities = _safe_dict(working.get("entities"))
    proteins = _safe_list(entities.get("proteins"))
    compounds = _safe_list(entities.get("compounds"))

    report["summary"]["proteins_total"] = len(
        [row for row in proteins if isinstance(row, dict) and _canonical(row.get("name"))]
    )
    report["summary"]["compounds_total"] = len(
        [row for row in compounds if isinstance(row, dict) and _canonical(row.get("name"))]
    )

    for idx, protein in enumerate(proteins):
        if not isinstance(protein, dict):
            continue
        name = _canonical(protein.get("name"))
        if not name:
            continue
        mapped_ids = _safe_dict(protein.get("mapped_ids"))
        uniprot_id = _canonical(mapped_ids.get("uniprot")).upper()
        if not uniprot_id:
            continue
        report["summary"]["proteins_with_uniprot"] += 1
        cache_key = f"uniprot:{uniprot_id}"
        cached = cache.get("proteins", cache_key)
        status = "error"
        if cached is not None:
            report["summary"]["cache_hits"] += 1
            enrichment_obj = _safe_dict(cached.get("parsed"))
            status = _canonical(cached.get("status")).lower() or _canonical(enrichment_obj.get("status")).lower()
        else:
            report["summary"]["api_calls"] += 1
            report["calls"]["uniprot"] += 1
            fetched = _fetch_uniprot_enrichment(client, uniprot_id)
            cache.set("proteins", cache_key, fetched)
            enrichment_obj = _safe_dict(fetched.get("parsed")) if _canonical(fetched.get("status")).lower() == "ok" else fetched
            status = _canonical(fetched.get("status")).lower()

        protein.setdefault("enrichment", {})
        protein["enrichment"].setdefault("protein", {})
        if status == "ok" and enrichment_obj:
            protein["enrichment"]["protein"]["uniprot"] = enrichment_obj
            report["summary"]["proteins_enriched_ok"] += 1
        else:
            protein["enrichment"]["protein"]["uniprot"] = {
                "status": "error",
                "uniprot_id": uniprot_id,
                "error_message": _canonical(_safe_dict(cached).get("error_message"))
                if cached is not None
                else _canonical(_safe_dict(enrichment_obj).get("error_message")),
                "provenance": {
                    "source": "UniProt",
                    "query": uniprot_id,
                    "retrieved_at": _utc_now_iso(),
                },
            }
            report["summary"]["proteins_enriched_error"] += 1

        report["entities"].append(
            {
                "entity_type": "protein",
                "name": name,
                "json_pointer": f"/entities/proteins/{idx}",
                "status": status or "error",
                "cache_key": cache_key,
            }
        )

    for idx, compound in enumerate(compounds):
        if not isinstance(compound, dict):
            continue
        name = _canonical(compound.get("name"))
        if not name:
            continue
        mapped_ids = _safe_dict(compound.get("mapped_ids"))
        candidate_ids = {
            "chebi": _normalize_chebi_id(_canonical(mapped_ids.get("chebi"))),
            "hmdb": _normalize_hmdb_id(_canonical(mapped_ids.get("hmdb"))),
            "kegg": _normalize_kegg_id(_canonical(mapped_ids.get("kegg"))),
        }
        if not any(candidate_ids.values()):
            continue
        report["summary"]["compounds_with_ids"] += 1
        enrichment = _init_compound_enrichment(mapped_ids)
        enrichment["is_generic_class"] = _is_generic_class_compound(name)

        for source_name in ["chebi", "hmdb", "kegg"]:
            source_id = candidate_ids.get(source_name, "")
            if not source_id:
                continue
            cache_key = f"{source_name}:{source_id}"
            cached = cache.get("compounds", cache_key)
            if cached is not None:
                report["summary"]["cache_hits"] += 1
                source_obj = cached
            else:
                report["summary"]["api_calls"] += 1
                report["calls"][source_name] += 1
                if source_name == "chebi":
                    source_obj = _fetch_chebi_enrichment(client, source_id)
                elif source_name == "hmdb":
                    source_obj = _fetch_hmdb_enrichment(client, source_id)
                else:
                    source_obj = _fetch_kegg_enrichment(client, source_id)
                cache.set("compounds", cache_key, source_obj)
            _merge_compound_source(enrichment, source_obj)

        if not enrichment.get("sources"):
            enrichment["sources"] = [
                {
                    "source": "none",
                    "id": "",
                    "retrieved_at": _utc_now_iso(),
                    "status": "error",
                    "error_if_any": "no_supported_compound_ids",
                }
            ]
            enrichment["enrichment_status"] = "error"
        else:
            _recompute_compound_merged(enrichment)

        compound.setdefault("enrichment", {})
        compound["enrichment"]["compound"] = enrichment

        if enrichment["enrichment_status"] == "error":
            report["summary"]["compounds_enriched_error"] += 1
        else:
            report["summary"]["compounds_enriched_ok"] += 1
            if enrichment["enrichment_status"] == "ok_with_warnings":
                report["summary"]["compounds_enriched_ok_with_warnings"] += 1

        report["entities"].append(
            {
                "entity_type": "compound",
                "name": name,
                "json_pointer": f"/entities/compounds/{idx}",
                "status": enrichment["enrichment_status"],
                "cross_references": _safe_dict(enrichment.get("cross_references")),
            }
        )

    cache.save()
    return {"payload": working, "report": report}


def run_enrichment(
    input_path: Path,
    output_path: Path,
    report_path: Path,
    *,
    cache_path: Path,
    dump_path: Optional[Path] = None,
    qa_report: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Mapped input JSON must be an object.")

    result = enrich_payload(payload, cache_path=cache_path)
    enriched_payload = _safe_dict(result.get("payload"))
    report = _safe_dict(result.get("report"))
    output_path.write_text(json.dumps(enriched_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    resolved_dump_path = dump_path or Path("out") / "enrichment_dump.json"
    resolved_dump_path.parent.mkdir(parents=True, exist_ok=True)
    enrichment_dump = _build_enrichment_dump(enriched_payload)
    resolved_dump_path.write_text(json.dumps(enrichment_dump, indent=2, ensure_ascii=False), encoding="utf-8")
    report["artifacts"] = {
        **_safe_dict(report.get("artifacts")),
        "enrichment_dump_path": str(resolved_dump_path),
    }
    if qa_report is not None:
        report["qa_report_summary"] = _safe_dict(qa_report.get("summary"))
        report["qa_flags_count"] = {
            flag_name: len(flag_list)
            for flag_name, flag_list in _safe_dict(qa_report.get("flags")).items()
        }
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Best-effort enrichment for mapped proteins and compounds.")
    parser.add_argument("--in", dest="input_path", required=True, help="Input mapped JSON path")
    parser.add_argument("--out", dest="output_path", default="final.enriched.json", help="Output enriched JSON path")
    parser.add_argument(
        "--report",
        dest="report_path",
        default="enrichment_report.json",
        help="Enrichment report JSON path",
    )
    parser.add_argument(
        "--cache",
        dest="cache_path",
        default="enrichment_cache.json",
        help="Enrichment cache JSON path",
    )
    parser.add_argument(
        "--dump",
        dest="dump_path",
        default="out/enrichment_dump.json",
        help="Consolidated enrichment dump JSON path",
    )
    args = parser.parse_args()
    run_enrichment(
        Path(args.input_path),
        Path(args.output_path),
        Path(args.report_path),
        cache_path=Path(args.cache_path),
        dump_path=Path(args.dump_path),
    )


if __name__ == "__main__":
    main()
