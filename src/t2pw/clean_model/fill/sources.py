"""Thin source clients: each takes a Fetcher and returns plain dicts / lists (None = not found).

No KEGG calls (P-17): KEGG ids only ever arrive as cross-references carried by UniProt or ChEBI.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

UNIPROT = "https://rest.uniprot.org/uniprotkb"
CHEBI = "https://www.ebi.ac.uk/chebi/backend/api/public/compound/"
OLS4 = "https://www.ebi.ac.uk/ols4/api/search"  # the current ChEBI API has no exact label/synonym search; OLS4 does
QUICKGO = "https://www.ebi.ac.uk/QuickGO/services/ontology/go/"
EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
RHEA = "https://www.rhea-db.org/rhea"

_SAFE = re.compile(r"^[A-Za-z0-9_.\-]+$")
_EC = re.compile(r"^\d+\.\d+\.\d+\.\d+$")
_CHARGED = re.compile(r"\(\d*[+\-−]\)$")


def _cf(s: Any) -> str:
    return re.sub(r"\s+", " ", str(s or "").replace("−", "-")).strip().casefold()


# ---- UniProt ------------------------------------------------------------------------------------------------------

def _names(node: Any) -> List[str]:
    return [node["fullName"]["value"]] if isinstance(node, dict) and isinstance(node.get("fullName"), dict) else []


def uniprot_entry(f: Any, accession: str) -> Optional[Dict[str, Any]]:
    d = f.get_json(f"{UNIPROT}/{accession}.json")
    if not isinstance(d, dict) or "primaryAccession" not in d:
        return None
    pd = d.get("proteinDescription") or {}
    rec, alts = pd.get("recommendedName") or {}, pd.get("alternativeNames") or []
    ecs = [e["value"] for n in [rec] + alts for e in n.get("ecNumbers") or []]
    syn = [x for a in alts for x in _names(a)] + [s["value"] for a in [rec] + alts for s in a.get("shortNames") or []]
    genes, gsyn = [], []
    for g in d.get("genes") or []:
        if isinstance(g.get("geneName"), dict):
            genes.append(g["geneName"]["value"])
        gsyn += [s["value"] for s in g.get("synonyms") or []]
    # The Ensembl xref id is a transcript; the protein id is its ProteinId property. `ensembl` holds protein ids only,
    # `ensembl_gene` the transcript and gene ids (written as isEncodedBy). KEGG ids are gene ids too.
    xr: Dict[str, List[str]] = {"kegg_gene": [], "ensembl": [], "ensembl_gene": [], "go": []}
    for x in d.get("uniProtKBCrossReferences") or []:
        db = x.get("database", "")
        if db == "KEGG":
            xr["kegg_gene"].append(x["id"])
        elif db.startswith("Ensembl"):
            props = {p.get("key"): p.get("value") for p in x.get("properties") or []}
            if props.get("ProteinId"):
                xr["ensembl"].append(props["ProteinId"])
            xr["ensembl_gene"] += [x["id"]] + ([props["GeneId"]] if props.get("GeneId") else [])
        elif db == "GO":
            xr["go"].append(x["id"])
    xr = {k: list(dict.fromkeys(v)) for k, v in xr.items()}
    return {"accession": d["primaryAccession"], "protein_name": (_names(rec) or [None])[0], "gene_names": genes,
            "synonyms": list(dict.fromkeys(syn + gsyn)), "ec_numbers": list(dict.fromkeys(ecs)),
            "taxon_id": str((d.get("organism") or {}).get("taxonId", "")) or None, "xrefs": xr,
            "reviewed": str(d.get("entryType", "")).startswith("UniProtKB reviewed")}


def uniprot_by_gene(f: Any, symbol: str, taxon_id: Any) -> List[Dict[str, Any]]:
    """Exact gene symbol within a taxon (descendants included) -> candidate accessions."""
    if not _SAFE.match(str(symbol)) or not re.match(r"^\d+$", str(taxon_id)):
        return []
    d = f.get_json(f"{UNIPROT}/search", {"query": f"gene_exact:{symbol} AND taxonomy_id:{taxon_id}",
                                          "fields": "accession,gene_names,protein_name,organism_id,reviewed,ec",
                                          "format": "json", "size": 10})
    out = []
    for r in (d or {}).get("results") or []:
        gn = [g["geneName"]["value"] for g in r.get("genes") or [] if isinstance(g.get("geneName"), dict)]
        out.append({"accession": r["primaryAccession"], "gene_names": gn,
                    "protein_name": (_names((r.get("proteinDescription") or {}).get("recommendedName")) or [None])[0],
                    "taxon_id": str((r.get("organism") or {}).get("taxonId", "")) or None,
                    "reviewed": str(r.get("entryType", "")).startswith("UniProtKB reviewed")})
    return out


# ---- ChEBI --------------------------------------------------------------------------------------------------------

def chebi_compound(f: Any, chebi_id: Any) -> Optional[Dict[str, Any]]:
    n = re.sub(r"\D", "", str(chebi_id))
    d = f.get_json(f"{CHEBI}{n}/") if n else None
    if not isinstance(d, dict) or "chebi_accession" not in d:
        return None
    cd, st = d.get("chemical_data") or {}, d.get("default_structure") or {}
    xr: Dict[str, List[str]] = {"kegg_compound": [], "hmdb": [], "cas": [], "pubchem": [], "drugbank": []}
    for typ, rows in (d.get("database_accessions") or {}).items():
        for r in rows:
            src, acc = r.get("source_name", ""), str(r.get("accession_number", ""))
            if typ == "CAS":
                xr["cas"].append(acc)
            elif src == "KEGG COMPOUND":
                xr["kegg_compound"].append(acc)
            elif src == "HMDB":
                xr["hmdb"].append(acc)
            elif src == "PubChem":
                xr["pubchem"].append(acc)
            elif src == "DrugBank":
                xr["drugbank"].append(acc)
    xr = {k: list(dict.fromkeys(v)) for k, v in xr.items()}
    syn = [x["name"] for k, rows in (d.get("names") or {}).items() if k != "IUPAC NAME" for x in rows]
    return {"id": str(d["id"]), "name": d.get("name"), "ascii_name": d.get("ascii_name"), "formula": cd.get("formula"),
            "charge": cd.get("charge"), "inchikey": st.get("standard_inchi_key"), "smiles": st.get("smiles"),
            "synonyms": list(dict.fromkeys(syn)), "xrefs": xr}


def chebi_by_name(f: Any, name: str) -> List[Dict[str, Any]]:
    """Exact label or exact synonym matches only (verified here, not trusted from the index). OLS4 does the lookup."""
    q = _cf(name)
    if not q:
        return []
    d = f.get_json(OLS4, {"q": name, "ontology": "chebi", "exact": "true", "queryFields": "label,synonym",
                          "fieldList": "obo_id,label,synonym", "rows": 50})
    out = {}
    for r in ((d or {}).get("response") or {}).get("docs") or []:
        oid, label = r.get("obo_id", ""), r.get("label") or ""
        if not oid.startswith("CHEBI:"):
            continue
        if _cf(label) == q:
            mt = "exact_label"
        elif q in [_cf(s) for s in r.get("synonym") or []]:
            mt = "exact_synonym"
        else:
            continue
        out[oid[6:]] = {"id": oid[6:], "label": label, "match_type": mt, "charged": bool(_CHARGED.search(label))}
    return list(out.values())


# ---- Gene Ontology (QuickGO) --------------------------------------------------------------------------------------

def go_cc_by_name(f: Any, name: str) -> List[Dict[str, Any]]:
    """Cellular-component terms whose label or exact synonym equals ``name``."""
    q = _cf(name)
    if not q:
        return []
    s = f.get_json(QUICKGO + "search", {"query": name, "limit": 25, "page": 1})
    ids = [r["id"] for r in (s or {}).get("results") or []
           if r.get("aspect") == "cellular_component" and not r.get("isObsolete")]
    if not ids:
        return []
    t = f.get_json(QUICKGO + "terms/" + ",".join(ids))
    out = []
    for r in (t or {}).get("results") or []:
        if r.get("isObsolete") or r.get("aspect") != "cellular_component":
            continue
        if _cf(r.get("name")) == q:
            mt = "exact_label"
        elif q in [_cf(x.get("name")) for x in r.get("synonyms") or [] if x.get("type") == "exact"]:
            mt = "exact_synonym"
        else:
            continue
        out.append({"id": r["id"], "name": r["name"], "match_type": mt})
    return out


def go_ancestors(f: Any, ids: List[str]) -> Dict[str, List[str]]:
    """GO id -> ancestor ids over part_of only (self excluded): the SBML writes the parent as bqbiol:isPartOf, so is_a
    ancestors (a membrane is_a membrane-bounded thing) must not mix in. Verified live 2026-11-10: cytosol -> cytoplasm."""
    if not ids:
        return {}
    d = f.get_json(QUICKGO + "terms/" + ",".join(ids) + "/ancestors", {"relations": "part_of"})
    return {r["id"]: [a for a in r.get("ancestors") or [] if a != r["id"]] for r in (d or {}).get("results") or []}


# ---- NCBI ---------------------------------------------------------------------------------------------------------

def ncbi_taxids(f: Any, organism: str) -> List[str]:
    d = f.get_json(EUTILS + "esearch.fcgi", {"db": "taxonomy", "term": f"{organism}[Scientific Name]", "retmode": "json"})
    return [str(x) for x in ((d or {}).get("esearchresult") or {}).get("idlist") or []]


def ncbi_article(f: Any, pmcid: Any = None, pmid: Any = None) -> Optional[Dict[str, Any]]:
    if pmcid:
        db, uid = "pmc", re.sub(r"\D", "", str(pmcid))
    elif pmid:
        db, uid = "pubmed", re.sub(r"\D", "", str(pmid))
    else:
        return None
    if not uid:
        return None
    d = f.get_json(EUTILS + "esummary.fcgi", {"db": db, "id": uid, "retmode": "json"})
    r = ((d or {}).get("result") or {}).get(uid)
    if not isinstance(r, dict) or r.get("error") or not r.get("title"):
        return None
    ids = {a.get("idtype"): a.get("value") for a in r.get("articleids") or []}
    m = re.match(r"(\d{4})", str(r.get("pubdate") or ""))
    return {"title": r["title"], "authors": [a["name"] for a in r.get("authors") or [] if a.get("name")],
            "year": int(m.group(1)) if m else None, "doi": ids.get("doi"),
            "pmid": ids.get("pmid") or ids.get("pubmed"), "pmcid": ids.get("pmcid")}


# ---- Rhea ---------------------------------------------------------------------------------------------------------

def rhea_by_ec(f: Any, ec: str) -> List[str]:
    if not _EC.match(str(ec)):
        return []
    t = f.get_text(RHEA, {"query": f"ec:{ec}", "columns": "rhea-id,equation,ec", "format": "tsv"})
    rows = [ln.split("\t")[0] for ln in (t or "").splitlines()[1:] if ln.strip()]
    return [r for r in rows if r.startswith("RHEA:")]
