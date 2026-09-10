"""C-122 round 3 -- what does the api-v7 -> api-v8 key bump actually reach?

Measures, on a synthetic cache holding a LEGACY-SHAPED (v7) OPCL1 entry:
  A. does the ladder collapse a legacy-shaped candidate pair?
  B. after the bump, does a v7 entry get re-parsed, or promoted forward stale?
  C. does a cache with NO prior entry write under api-v8 with the new shape?
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from t2pw.mapping.map_ids import (
    MappingCache,
    _extract_uniprot_candidates,
    _is_same_protein_record_duplicate,
    _map_protein_with_strategy,
    _normalize_name,
    _protein_alias_entries,
    verify_real_protein_identity,
)

PAYLOAD: Dict[str, Any] = {
    "results": [
        {
            "primaryAccession": "Q84P21",
            "entryType": "UniProtKB reviewed (Swiss-Prot)",
            "proteinDescription": {
                "recommendedName": {"fullName": {"value": "Peroxisomal OPC-8:0-CoA ligase 1"}}
            },
            "genes": [{"geneName": {"value": "4CLL5"}, "synonyms": [{"value": "OPCL1"}]}],
            "organism": {"scientificName": "Arabidopsis thaliana", "taxonId": 3702},
        },
        {
            "primaryAccession": "F4HST9",
            "entryType": "UniProtKB unreviewed (TrEMBL)",
            "proteinDescription": {
                "submissionNames": [{"fullName": {"value": "OPC-8:0 CoA ligase1"}}]
            },
            "genes": [
                {"geneName": {"value": "OPCL1"}, "synonyms": [{"value": "OPC-8:0 CoA ligase1"}]}
            ],
            "organism": {"scientificName": "Arabidopsis thaliana", "taxonId": 3702},
        },
    ]
}

NAME = "OPCL1"
ORGANISM = "Arabidopsis thaliana"


class _Resp:
    def __init__(self, payload: Dict[str, Any]) -> None:
        self.status_code = 200
        self._payload = payload

    def json(self) -> Dict[str, Any]:
        return self._payload


class _Client:
    def __init__(self) -> None:
        self.calls = 0

    def get(self, url: str, *, params: Optional[Dict[str, Any]] = None, headers: Any = None) -> _Resp:
        if "rest.uniprot.org" in url:
            self.calls += 1
            return _Resp(copy.deepcopy(PAYLOAD))
        return _Resp({})


def _legacy_shape(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """A v7-era candidate: no primary_gene_names, no taxonomy_id."""
    out = []
    for row in rows:
        legacy = {k: v for k, v in row.items() if k not in ("primary_gene_names", "taxonomy_id")}
        out.append(legacy)
    return out


def _base_key() -> str:
    return f"{_normalize_name(NAME)}::{_normalize_name(ORGANISM)}::::" + json.dumps({}, sort_keys=True)


def main() -> int:
    fresh = _extract_uniprot_candidates(copy.deepcopy(PAYLOAD), query_name=NAME, organism=ORGANISM)
    legacy = _legacy_shape(fresh)

    print("== A. predicate and ladder, fresh vs legacy shape ==")
    print("fresh  collapse:", _is_same_protein_record_duplicate(fresh[0], fresh[1]))
    print("legacy collapse:", _is_same_protein_record_duplicate(legacy[0], legacy[1]))
    for label, rows in (("fresh", fresh), ("legacy", legacy)):
        verdict = verify_real_protein_identity(
            NAME,
            candidates=copy.deepcopy(rows),
            mapped_ids={"uniprot": rows[0]["accession"]},
            organism=ORGANISM,
            source="uniprot",
            resolved_name=str(rows[0]["protein_name"]),
            result={"confidence": rows[0]["score"]},
        )
        print(
            f"{label:6s} ladder: verified={verdict['verified']} reason={verdict['reason']!r} "
            f"margin={verdict['checks'].get('margin')!r} "
            f"collapsed={verdict.get('collapsed_duplicate_accessions')}"
        )

    tmp = Path(sys.argv[1])
    tmp.mkdir(parents=True, exist_ok=True)

    print()
    print("== B. a pre-existing api-v7 entry, read through the legacy chain ==")
    cache_path = tmp / "with_v7.json"
    aliases = _protein_alias_entries(NAME, {})
    alias_key = json.dumps(aliases, sort_keys=True)
    v7_key = f"api-v7::{_base_key()}::{alias_key}"
    cache_path.write_text(
        json.dumps(
            {
                "proteins": {
                    v7_key: {
                        "status": "unmapped",
                        "reason": "ambiguous",
                        "provider": "UniProt",
                        "source": "api",
                        "query": NAME,
                        "candidates": copy.deepcopy(legacy),
                    }
                },
                "compounds": {},
                "complexes": {},
            },
            indent=1,
        ),
        encoding="utf-8",
    )
    cache = MappingCache(cache_path, enabled=True)
    client = _Client()
    _map_protein_with_strategy(
        id_source="api", db=None, client=client, cache=cache, name=NAME, organism=ORGANISM,
        protein_row={"name": NAME},
    )
    keys = sorted(cache.data["proteins"])
    print("http calls issued          :", client.calls)
    print("keys after the run         :", [k.split("::")[0] for k in keys])
    for key in keys:
        if key.startswith("api-v8::"):
            cands = cache.data["proteins"][key].get("candidates") or []
            print("v8 entry candidate count   :", len(cands))
            print("v8 entry has primary_gene_names:",
                  any("primary_gene_names" in c for c in cands if isinstance(c, dict)))

    print()
    print("== C. an EMPTY cache: fresh parse, written under which key? ==")
    cache2 = MappingCache(tmp / "empty.json", enabled=True)
    client2 = _Client()
    _map_protein_with_strategy(
        id_source="api", db=None, client=client2, cache=cache2, name=NAME, organism=ORGANISM,
        protein_row={"name": NAME},
    )
    keys2 = sorted(cache2.data["proteins"])
    print("http calls issued          :", client2.calls)
    print("keys after the run         :", [k.split("::")[0] for k in keys2])
    for key in keys2:
        cands = cache2.data["proteins"][key].get("candidates") or []
        print("candidate count            :", len(cands))
        print("has primary_gene_names     :",
              any("primary_gene_names" in c for c in cands if isinstance(c, dict)))
        print("has taxonomy_id            :",
              any("taxonomy_id" in c for c in cands if isinstance(c, dict)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
