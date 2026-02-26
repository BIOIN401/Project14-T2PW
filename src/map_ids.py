from __future__ import annotations

import argparse
import json
import re
import time
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import quote_plus
from xml.etree import ElementTree

import requests


def _safe_list(value: Any) -> List[Any]:
    return value if isinstance(value, list) else []


def _safe_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _normalize_name(value: str) -> str:
    lowered = re.sub(r"\s+", " ", value.strip().casefold())
    return re.sub(r"[^a-z0-9 ]+", "", lowered)


def _token_set(value: str) -> set:
    return {tok for tok in _normalize_name(value).split(" ") if tok}


def _jaccard(a: str, b: str) -> float:
    sa = _token_set(a)
    sb = _token_set(b)
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


class HttpClient:
    def __init__(self, timeout: int = 15, max_retries: int = 3, backoff_seconds: float = 0.6) -> None:
        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff_seconds = backoff_seconds
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "Project14-T2PW-IDMapper/1.0"})

    def get(self, url: str, *, params: Optional[Dict[str, Any]] = None) -> requests.Response:
        last_exc: Optional[Exception] = None
        for attempt in range(1, self.max_retries + 1):
            try:
                resp = self.session.get(url, params=params, timeout=self.timeout)
                if resp.status_code >= 500:
                    raise requests.HTTPError(f"Server error {resp.status_code}")
                return resp
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                if attempt < self.max_retries:
                    time.sleep(self.backoff_seconds * attempt)
        raise RuntimeError(f"HTTP request failed after retries: {url}; last error: {last_exc}")


class MappingCache:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.data: Dict[str, Dict[str, Any]] = {"proteins": {}, "compounds": {}}
        if path.exists():
            raw = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                self.data["proteins"] = _safe_dict(raw.get("proteins"))
                self.data["compounds"] = _safe_dict(raw.get("compounds"))

    def get(self, section: str, key: str) -> Optional[Dict[str, Any]]:
        value = _safe_dict(self.data.get(section)).get(key)
        return value if isinstance(value, dict) else None

    def set(self, section: str, key: str, value: Dict[str, Any]) -> None:
        self.data.setdefault(section, {})
        self.data[section][key] = value

    def save(self) -> None:
        self.path.write_text(json.dumps(self.data, indent=2, ensure_ascii=False), encoding="utf-8")


def _extract_global_organism(payload: Dict[str, Any]) -> str:
    entities = _safe_dict(payload.get("entities"))
    species_names = [
        (item.get("name") or "").strip()
        for item in _safe_list(entities.get("species"))
        if isinstance(item, dict) and isinstance(item.get("name"), str) and item.get("name").strip()
    ]
    if len(species_names) == 1:
        return species_names[0]
    biological_states = _safe_list(payload.get("biological_states"))
    state_species = {
        (state.get("species") or "").strip()
        for state in biological_states
        if isinstance(state, dict) and isinstance(state.get("species"), str) and state.get("species").strip()
    }
    if len(state_species) == 1:
        return sorted(state_species)[0]
    return ""


def _entity_locations(payload: Dict[str, Any], location_key: str, name_key: str) -> Dict[str, List[str]]:
    states = {
        (item.get("name") or "").strip(): (item.get("subcellular_location") or "").strip()
        for item in _safe_list(payload.get("biological_states"))
        if isinstance(item, dict)
        and isinstance(item.get("name"), str)
        and item.get("name").strip()
        and isinstance(item.get("subcellular_location"), str)
    }
    out: Dict[str, List[str]] = {}
    rows = _safe_list(_safe_dict(payload.get("element_locations")).get(location_key))
    for row in rows:
        if not isinstance(row, dict):
            continue
        name = (row.get(name_key) or "").strip() if isinstance(row.get(name_key), str) else ""
        if not name:
            continue
        state = (row.get("biological_state") or "").strip() if isinstance(row.get("biological_state"), str) else ""
        loc = states.get(state, "")
        out.setdefault(name, [])
        if loc and loc not in out[name]:
            out[name].append(loc)
    return out


def _extract_uniprot_candidates(payload: Dict[str, Any], query_name: str, organism: str) -> List[Dict[str, Any]]:
    results = _safe_list(payload.get("results"))
    out: List[Dict[str, Any]] = []
    for item in results:
        if not isinstance(item, dict):
            continue
        accession = item.get("primaryAccession")
        if not isinstance(accession, str) or not accession:
            continue
        protein_desc = _safe_dict(item.get("proteinDescription"))
        recommended = _safe_dict(protein_desc.get("recommendedName"))
        fullname = _safe_dict(recommended.get("fullName")).get("value")
        alt_names = _safe_list(protein_desc.get("alternativeNames"))
        alt_values: List[str] = []
        for alt in alt_names:
            if not isinstance(alt, dict):
                continue
            alt_full = _safe_dict(alt.get("fullName")).get("value")
            if isinstance(alt_full, str) and alt_full.strip():
                alt_values.append(alt_full.strip())
        gene_names: List[str] = []
        for gene_obj in _safe_list(item.get("genes")):
            if not isinstance(gene_obj, dict):
                continue
            primary = _safe_dict(gene_obj.get("geneName")).get("value")
            if isinstance(primary, str) and primary.strip():
                gene_names.append(primary.strip())
            for synonym in _safe_list(gene_obj.get("synonyms")):
                syn = _safe_dict(synonym).get("value")
                if isinstance(syn, str) and syn.strip():
                    gene_names.append(syn.strip())
        organism_name = _safe_dict(item.get("organism")).get("scientificName", "")
        reviewed = str(item.get("entryType", "")).lower().find("reviewed") != -1

        candidate_names = [v for v in [fullname] if isinstance(v, str)] + [v for v in alt_values if isinstance(v, str)] + gene_names
        best_name_score = max((_jaccard(query_name, c) for c in candidate_names), default=0.0)
        exact_name_match = any(_normalize_name(query_name) == _normalize_name(c) for c in candidate_names)
        organism_score = 0.0
        if organism and isinstance(organism_name, str):
            if _normalize_name(organism) == _normalize_name(organism_name):
                organism_score = 0.25
            elif _normalize_name(organism) in _normalize_name(organism_name):
                organism_score = 0.15
        reviewed_score = 0.05 if reviewed else 0.0
        score = min(1.0, (0.55 if exact_name_match else 0.35 * best_name_score) + organism_score + reviewed_score)

        out.append(
            {
                "accession": accession,
                "protein_name": fullname if isinstance(fullname, str) else "",
                "gene_names": sorted(set(gene_names))[:8],
                "organism": organism_name if isinstance(organism_name, str) else "",
                "reviewed": reviewed,
                "score": round(score, 4),
            }
        )
    out.sort(key=lambda item: item.get("score", 0.0), reverse=True)
    return out


def map_protein_uniprot(client: HttpClient, name: str, organism: str) -> Dict[str, Any]:
    query_parts = [f'(protein_name:"{name}" OR gene:"{name}")']
    if organism:
        query_parts.append(f'organism_name:"{organism}"')
    query = " AND ".join(query_parts)
    params = {
        "query": query,
        "format": "json",
        "size": 10,
        "fields": "accession,protein_name,gene_names,organism_name,reviewed",
    }
    try:
        resp = client.get("https://rest.uniprot.org/uniprotkb/search", params=params)
    except Exception as exc:  # noqa: BLE001
        return {
            "status": "unmapped",
            "reason": f"network_error:{exc}",
            "query": query,
            "candidates": [],
        }
    if resp.status_code != 200:
        return {
            "status": "unmapped",
            "reason": f"UniProt request failed with status {resp.status_code}",
            "query": query,
            "candidates": [],
        }
    payload = resp.json()
    candidates = _extract_uniprot_candidates(payload, query_name=name, organism=organism)
    if not candidates:
        return {"status": "unmapped", "reason": "no_match", "query": query, "candidates": []}
    best = candidates[0]
    second_score = candidates[1]["score"] if len(candidates) > 1 else 0.0
    if best["score"] >= 0.78 and best["score"] >= second_score + 0.1:
        return {
            "status": "mapped",
            "query": query,
            "mapped_ids": {"uniprot": best["accession"]},
            "confidence": best["score"],
            "chosen_rule": "top_unique_candidate",
            "candidates": candidates[:8],
            "reviewed": bool(best.get("reviewed")),
        }
    return {
        "status": "unmapped",
        "reason": "ambiguous",
        "query": query,
        "confidence": best["score"],
        "candidates": candidates[:8],
    }


def _score_compound_candidate(query: str, candidate_name: str) -> float:
    norm_q = _normalize_name(query)
    norm_c = _normalize_name(candidate_name)
    if norm_q == norm_c:
        return 0.95
    jac = _jaccard(query, candidate_name)
    return round(0.35 + 0.6 * jac, 4)


def _query_chebi(client: HttpClient, name: str) -> List[Dict[str, Any]]:
    url = "https://www.ebi.ac.uk/webservices/chebi/2.0/test/getLiteEntity"
    params = {"search": name, "searchCategory": "ALL NAMES", "maximumResults": 10, "stars": "ALL"}
    try:
        resp = client.get(url, params=params)
    except Exception:  # noqa: BLE001
        return []
    if resp.status_code != 200:
        return []
    try:
        root = ElementTree.fromstring(resp.text)
    except ElementTree.ParseError:
        return []

    # Namespace agnostic parsing
    results: List[Dict[str, Any]] = []
    for node in root.iter():
        if node.tag.lower().endswith("liteentity"):
            chebi_id = ""
            chebi_name = ""
            for child in node:
                tag = child.tag.split("}")[-1]
                text = (child.text or "").strip()
                if tag == "chebiId":
                    chebi_id = text
                elif tag == "chebiAsciiName":
                    chebi_name = text
            if chebi_id:
                results.append(
                    {
                        "database": "chebi",
                        "id": chebi_id,
                        "name": chebi_name,
                        "score": _score_compound_candidate(name, chebi_name or chebi_id),
                    }
                )
    results.sort(key=lambda item: item["score"], reverse=True)
    return results[:10]


def _query_kegg(client: HttpClient, name: str) -> List[Dict[str, Any]]:
    encoded = quote_plus(name)
    url = f"https://rest.kegg.jp/find/compound/{encoded}"
    try:
        resp = client.get(url)
    except Exception:  # noqa: BLE001
        return []
    if resp.status_code != 200 or not resp.text.strip():
        return []
    out: List[Dict[str, Any]] = []
    for line in resp.text.splitlines():
        if "\t" not in line:
            continue
        left, right = line.split("\t", 1)
        kid = left.replace("cpd:", "").strip()
        names = [n.strip() for n in right.split(";") if n.strip()]
        best_name = names[0] if names else right.strip()
        score = max((_score_compound_candidate(name, n) for n in names), default=_score_compound_candidate(name, best_name))
        out.append({"database": "kegg", "id": kid, "name": best_name, "score": score})
    out.sort(key=lambda item: item["score"], reverse=True)
    return out[:10]


def _query_hmdb(client: HttpClient, name: str) -> List[Dict[str, Any]]:
    # HMDB has no stable public JSON search API; this is best-effort HTML extraction.
    url = "https://hmdb.ca/unearth/q"
    params = {"query": name, "searcher": "metabolites"}
    try:
        resp = client.get(url, params=params)
    except Exception:  # noqa: BLE001
        return []
    if resp.status_code != 200:
        return []
    text = resp.text
    ids = re.findall(r"/metabolites/(HMDB\d{5,})", text, flags=re.IGNORECASE)
    seen = set()
    out: List[Dict[str, Any]] = []
    for hid in ids:
        hid_norm = hid.upper()
        if hid_norm in seen:
            continue
        seen.add(hid_norm)
        out.append({"database": "hmdb", "id": hid_norm, "name": "", "score": 0.6})
        if len(out) >= 10:
            break
    return out


def map_compound_all(client: HttpClient, name: str) -> Dict[str, Any]:
    chebi_candidates = _query_chebi(client, name)
    kegg_candidates = _query_kegg(client, name)
    hmdb_candidates = _query_hmdb(client, name)
    all_candidates = chebi_candidates + kegg_candidates + hmdb_candidates
    if not all_candidates:
        return {"status": "unmapped", "reason": "no_match", "candidates": []}

    all_candidates.sort(key=lambda item: item["score"], reverse=True)
    best = all_candidates[0]
    second = all_candidates[1]["score"] if len(all_candidates) > 1 else 0.0
    if best["score"] >= 0.82 and best["score"] >= second + 0.1:
        mapped_ids = {best["database"]: best["id"]}
        # Keep additional high-confidence IDs from other databases.
        for cand in all_candidates[1:]:
            if cand["database"] in mapped_ids:
                continue
            if cand["score"] >= 0.9:
                mapped_ids[cand["database"]] = cand["id"]
        return {
            "status": "mapped",
            "mapped_ids": mapped_ids,
            "confidence": best["score"],
            "chosen_rule": "top_unique_candidate",
            "candidates": all_candidates[:12],
        }

    return {
        "status": "unmapped",
        "reason": "ambiguous",
        "confidence": best["score"],
        "candidates": all_candidates[:12],
    }


def run_mapping(
    input_path: Path,
    output_path: Path,
    report_path: Path,
    *,
    cache_path: Path,
) -> Dict[str, Any]:
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Input JSON must be an object.")
    mapped = deepcopy(payload)

    client = HttpClient()
    cache = MappingCache(cache_path)

    entities = _safe_dict(mapped.get("entities"))
    proteins = _safe_list(entities.get("proteins"))
    compounds = _safe_list(entities.get("compounds"))

    global_organism = _extract_global_organism(mapped)
    protein_locations = _entity_locations(mapped, "protein_locations", "protein")
    compound_locations = _entity_locations(mapped, "compound_locations", "compound")

    logs: List[Dict[str, Any]] = []
    proteins_mapped = 0
    compounds_mapped = 0
    protein_ambiguous = 0
    compound_ambiguous = 0

    for idx, protein in enumerate(proteins):
        if not isinstance(protein, dict):
            continue
        name = (protein.get("name") or "").strip() if isinstance(protein.get("name"), str) else ""
        if not name:
            logs.append(
                {
                    "entity_type": "protein",
                    "name": "",
                    "json_pointer": f"/entities/proteins/{idx}",
                    "status": "unmapped",
                    "reason": "missing_name",
                    "location": "",
                }
            )
            continue
        organism = (protein.get("organism") or "").strip() if isinstance(protein.get("organism"), str) else ""
        if not organism and global_organism:
            organism = global_organism
            protein["organism"] = global_organism

        cache_key = f"{_normalize_name(name)}::{_normalize_name(organism)}"
        result = cache.get("proteins", cache_key)
        if result is None:
            result = map_protein_uniprot(client, name, organism)
            cache.set("proteins", cache_key, result)

        protein.setdefault("mapping_meta", {})
        protein["mapping_meta"]["query"] = {"name": name, "organism": organism}
        protein["mapping_meta"]["provider"] = "UniProt"
        protein["mapping_meta"]["candidates"] = result.get("candidates", [])
        protein["mapping_meta"]["chosen_rule"] = result.get("chosen_rule", "")
        protein["mapping_meta"]["confidence"] = float(result.get("confidence", 0.0))
        protein["mapping_meta"]["reviewed"] = bool(result.get("reviewed", False))

        if result.get("status") == "mapped":
            proteins_mapped += 1
            protein["mapped_ids"] = {**_safe_dict(protein.get("mapped_ids")), **_safe_dict(result.get("mapped_ids"))}
            status = "mapped"
            reason = ""
        else:
            status = "unmapped"
            reason = str(result.get("reason", "unknown"))
            if reason == "ambiguous":
                protein_ambiguous += 1

        logs.append(
            {
                "entity_type": "protein",
                "name": name,
                "json_pointer": f"/entities/proteins/{idx}",
                "status": status,
                "reason": reason,
                "location": ", ".join(protein_locations.get(name, [])),
                "organism": organism,
                "candidate_count": len(_safe_list(result.get("candidates"))),
            }
        )

    for idx, compound in enumerate(compounds):
        if not isinstance(compound, dict):
            continue
        name = (compound.get("name") or "").strip() if isinstance(compound.get("name"), str) else ""
        if not name:
            logs.append(
                {
                    "entity_type": "compound",
                    "name": "",
                    "json_pointer": f"/entities/compounds/{idx}",
                    "status": "unmapped",
                    "reason": "missing_name",
                    "location": "",
                }
            )
            continue

        cache_key = _normalize_name(name)
        result = cache.get("compounds", cache_key)
        if result is None:
            result = map_compound_all(client, name)
            cache.set("compounds", cache_key, result)

        compound.setdefault("mapping_meta", {})
        compound["mapping_meta"]["query"] = {"name": name}
        compound["mapping_meta"]["providers"] = ["ChEBI", "KEGG", "HMDB"]
        compound["mapping_meta"]["candidates"] = result.get("candidates", [])
        compound["mapping_meta"]["chosen_rule"] = result.get("chosen_rule", "")
        compound["mapping_meta"]["confidence"] = float(result.get("confidence", 0.0))

        if result.get("status") == "mapped":
            compounds_mapped += 1
            compound["mapped_ids"] = {**_safe_dict(compound.get("mapped_ids")), **_safe_dict(result.get("mapped_ids"))}
            status = "mapped"
            reason = ""
        else:
            status = "unmapped"
            reason = str(result.get("reason", "unknown"))
            if reason == "ambiguous":
                compound_ambiguous += 1

        logs.append(
            {
                "entity_type": "compound",
                "name": name,
                "json_pointer": f"/entities/compounds/{idx}",
                "status": status,
                "reason": reason,
                "location": ", ".join(compound_locations.get(name, [])),
                "candidate_count": len(_safe_list(result.get("candidates"))),
            }
        )

    cache.save()
    output_path.write_text(json.dumps(mapped, indent=2, ensure_ascii=False), encoding="utf-8")

    proteins_total = len([p for p in proteins if isinstance(p, dict) and isinstance(p.get("name"), str) and p.get("name").strip()])
    compounds_total = len([c for c in compounds if isinstance(c, dict) and isinstance(c.get("name"), str) and c.get("name").strip()])
    summary = {
        "proteins_total": proteins_total,
        "proteins_mapped": proteins_mapped,
        "proteins_mapped_pct": round((100.0 * proteins_mapped / proteins_total), 2) if proteins_total else 0.0,
        "proteins_ambiguous": protein_ambiguous,
        "compounds_total": compounds_total,
        "compounds_mapped": compounds_mapped,
        "compounds_mapped_pct": round((100.0 * compounds_mapped / compounds_total), 2) if compounds_total else 0.0,
        "compounds_ambiguous": compound_ambiguous,
    }

    report = {"summary": summary, "entities": logs}
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Deterministic ID mapping for proteins and compounds.")
    parser.add_argument("--in", dest="input_path", required=True, help="Input audited JSON path")
    parser.add_argument("--out", dest="output_path", default="final.mapped.json", help="Output mapped JSON path")
    parser.add_argument(
        "--report",
        dest="report_path",
        default="mapping_report.json",
        help="Output mapping report JSON path",
    )
    parser.add_argument(
        "--cache",
        dest="cache_path",
        default="id_mapping_cache.json",
        help="Cache file path for deterministic mapping reuse",
    )
    args = parser.parse_args()

    report = run_mapping(
        Path(args.input_path),
        Path(args.output_path),
        Path(args.report_path),
        cache_path=Path(args.cache_path),
    )
    print(f"Wrote mapped JSON: {args.output_path}")
    print(
        "Protein mapped: "
        f"{report['summary']['proteins_mapped']}/{report['summary']['proteins_total']} | "
        "Compound mapped: "
        f"{report['summary']['compounds_mapped']}/{report['summary']['compounds_total']}"
    )


if __name__ == "__main__":
    main()
