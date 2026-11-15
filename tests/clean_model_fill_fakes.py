"""FakeFetcher and canned responses shaped like the real ones seen in the 2026-11-10 probes. No network."""

from __future__ import annotations

from typing import Any, Dict, List, Optional


class FakeFetcher:
    """routes: list of (url_substring, params_subset_or_None, response). First match wins; no match -> None (404)."""

    def __init__(self, routes: List[tuple]) -> None:
        self.routes, self.calls = routes, []

    def _find(self, url: str, params: Optional[Dict[str, Any]] = None) -> Any:
        self.calls.append((url, dict(params or {})))
        for sub, want, resp in self.routes:
            if sub in url and all((params or {}).get(k) == v for k, v in (want or {}).items()):
                if isinstance(resp, Exception):
                    raise resp
                return resp
        return None

    get_json = get_text = _find


def uniprot(acc="P0AEJ2", gene="entC", taxon=83333, ec="5.4.4.2", name="Isochorismate synthase EntC", reviewed=True):
    return {"entryType": "UniProtKB reviewed (Swiss-Prot)" if reviewed else "UniProtKB unreviewed (TrEMBL)",
            "primaryAccession": acc, "organism": {"taxonId": taxon},
            "proteinDescription": {"recommendedName": {"fullName": {"value": name},
                                                       "ecNumbers": [{"value": ec}] if ec else []},
                                   "alternativeNames": [{"fullName": {"value": "Isochorismate mutase"}}]},
            "genes": [{"geneName": {"value": gene}, "synonyms": [{"value": "entC2"}], "orderedLocusNames": [{"value": "b0593"}]}],
            "uniProtKBCrossReferences": [{"database": "KEGG", "id": "eco:b0593"},
                                         {"database": "Ensembl", "id": "ENST00000269305.9",  # a transcript; the protein id is a property
                                          "properties": [{"key": "ProteinId", "value": "ENSP00000269305.4"},
                                                         {"key": "GeneId", "value": "ENSG00000141510.18"}]},
                                         {"database": "EnsemblBacteria", "id": "AAC73696",
                                          "properties": [{"key": "ProteinId", "value": "AAC73696"}]},
                                         {"database": "GO", "id": "GO:0008909"}]}


def uniprot_search(*entries):
    return {"results": [{"primaryAccession": e["primaryAccession"], "entryType": e["entryType"],
                         "organism": e["organism"], "genes": e["genes"],
                         "proteinDescription": e["proteinDescription"]} for e in entries]}


def chebi(cid=15422, name="ATP", formula="C10H16N5O13P3", ik="ZKHQWZAMYRWXGA-KQYNXXCUSA-N", kegg="C00002",
          hmdb="HMDB0000538", cas="56-65-5"):
    return {"id": cid, "chebi_accession": f"CHEBI:{cid}", "name": name, "ascii_name": name,
            "names": {"IUPAC NAME": [{"name": "iupac"}], "SYNONYM": [{"name": "Adenosine triphosphate"}]},
            "chemical_data": {"formula": formula, "charge": 0}, "default_structure": {"smiles": "C1=NC", "standard_inchi_key": ik},
            "database_accessions": {"CAS": [{"accession_number": cas, "source_name": "KEGG COMPOUND"}],
                                    "MANUAL_X_REF": [{"accession_number": kegg, "source_name": "KEGG COMPOUND"},
                                                     {"accession_number": hmdb, "source_name": "HMDB"}]}}


def ols(*docs):
    return {"response": {"numFound": len(docs), "docs": [{"obo_id": f"CHEBI:{i}", "label": lab, "synonym": syn}
                                                          for i, lab, syn in docs]}}


def qg_search(*terms):
    return {"results": [{"id": i, "name": n, "aspect": a, "isObsolete": False} for i, n, a in terms]}


def qg_terms(*terms):
    return {"results": [{"id": i, "name": n, "aspect": "cellular_component", "isObsolete": False,
                         "synonyms": [{"name": s, "type": "exact"} for s in syn]} for i, n, syn in terms]}


def qg_anc(**kw):
    return {"results": [{"id": k.replace("_", ":"), "ancestors": [k.replace("_", ":")] + [a for a in v]}
                        for k, v in kw.items()]}


PMC = {"result": {"uids": ["11983282"], "11983282": {
    "pubdate": "2025 Apr 10", "title": "Nucleotide-level characterization of arabinose systems.",
    "authors": [{"name": "Kim NM"}, {"name": "Peng D"}],
    "articleids": [{"idtype": "pmid", "value": "40210244"}, {"idtype": "pmcid", "value": "PMC11983282"},
                   {"idtype": "doi", "value": "10.1093/nar/gkaf224"}]}}}

TAX = {"esearchresult": {"count": "1", "idlist": ["562"]}}
RHEA_ONE = "Reaction identifier\tEquation\tEC number\nRHEA:18985\tchorismate = isochorismate\tEC:5.4.4.2\n"
RHEA_MANY = RHEA_ONE + "RHEA:99999\tother\tEC:5.4.4.2\n"
