"""ORCH-717 / Q3: is a ``pathbank_compound_id`` a real accession for Priority 1?

The decision packet named Q3 as UNRESOLVED and said Q2's Priority-1 prediction
could not be treated as final until it was settled, because the delta-ALA row
carries a PathBank compound id among its nine identifiers.

This probe settles it from the shipped code and the shipped data, with no change
to either. Three questions, each answered by measurement:

  Q3.1  Does ``_external_ids`` recognise a PathBank namespace at all today?
  Q3.2  Does adding/removing the PathBank id change whether the affected row
        "carries external ids" -- the predicate the Priority-1 branch turns on?
  Q3.3  What SHAPE is a pathbank compound id in this system, and could a
        conditional-acceptance policy resolve one against a real PathBank row?

Usage::  <python> orch717_q3_pathbank.py <repo-root>
Exit code is 0 always; read the printed findings.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(REPO / "src"))

from t2pw.bench.semantic import _external_ids  # noqa: E402

print("=" * 78)
print("Q3.1  the namespaces _external_ids actually recognises")
print("=" * 78)

# The row shape under test. The five RECOGNISED values are the delta-ALA row's
# own, as the decision packet lists them (HMDB0001149 / C00430 / 17549 / 137 /
# DB00855). The four unrecognised ones stand in for the packet's "plus CAS,
# BioCyc, ChemSpider and a PathBank compound id".
#
# HONESTY NOTE, because the record must not overclaim: `pathbank_compound_id 78`
# is NOT known to be the delta-ALA row's own id. It is a REPRESENTATIVE value
# taken from the committed exporter record
# evidence/probe_exporter_identity_mutation_2026-08-06.md, where 78 is Glycine's.
# Nothing below depends on the value: the questions are whether the namespace is
# recognised at all, and whether the row's Priority-1 predicate is sensitive to
# the id's PRESENCE. Both are answered by presence/absence, not by which integer.
NINE = {
    "hmdb": "HMDB0001149",
    "kegg": "C00430",
    "chebi": "17549",
    "pubchem": "137",
    "drugbank": "DB00855",
    "cas": "106-60-5",
    "biocyc": "5-AMINOLEVULINATE",
    "chemspider": "10442",
    "pathbank_compound_id": "78",
}

row_with = {"name": "delta-aminolevulinic acid", "mapped_ids": dict(NINE)}
row_without = {"name": "delta-aminolevulinic acid",
               "mapped_ids": {k: v for k, v in NINE.items()
                              if k != "pathbank_compound_id"}}

ids_with = _external_ids(row_with)
ids_without = _external_ids(row_without)

print(f"  row carries {len(NINE)} identifiers in mapped_ids")
print(f"  _external_ids WITH    pathbank_compound_id -> {ids_with}")
print(f"  _external_ids WITHOUT pathbank_compound_id -> {ids_without}")
print()
print(f"  recognised namespaces      : {sorted(ids_with)}")
print(f"  pathbank recognised?       : "
      f"{any('pathbank' in k for k in ids_with)}")
print(f"  identical with vs without? : {ids_with == ids_without}")

print()
print("=" * 78)
print("Q3.2  does Q3 change the Priority-1 predicate for the affected row?")
print("=" * 78)
# The Priority-1 false_real branch is guarded by `if ids:` -- it needs the row to
# carry ANY recognised accession, not a particular one.
print(f"  bool(ids) WITH    pathbank : {bool(ids_with)}")
print(f"  bool(ids) WITHOUT pathbank : {bool(ids_without)}")
print(f"  -> the predicate is {'UNCHANGED' if bool(ids_with) == bool(ids_without) else 'CHANGED'} "
      f"by the PathBank question")
print(f"  -> the row carries {len(ids_without)} recognised accessions even with")
print(f"     the PathBank id removed entirely, so Q3 CANNOT move Q2's arithmetic")

print()
print("=" * 78)
print("Q3.3  could a conditional-acceptance policy resolve one?")
print("=" * 78)
db = json.loads((REPO / "data" / "pathwhiz_id_db.json").read_text(encoding="utf-8"))
by_id = db["compounds"]["by_id"]
keys = list(by_id)
print(f"  data/pathwhiz_id_db.json compounds.by_id rows : {len(by_id)}")
print(f"  id space sample                               : {keys[:8]}")
allnum = all(k.isdigit() for k in keys)
print(f"  every id is a bare integer                    : {allnum}")
maxlen = max(len(k) for k in keys)
print(f"  longest id                                    : {maxlen} chars")
probe_id = "78"
print(f"  resolve representative id {probe_id!r} (Glycine's) : "
      f"{by_id.get(probe_id, 'NOT PRESENT')}")
print()
print("  The guardrail says 'arbitrary numeric IDs are never accepted'. Every id")
print("  in this space IS a bare small integer, and the committed exporter record")
print("  shows these are produced with")
print("      db_status   = matched_offline_name_index")
print("      chosen_rule = legacy_pathwhiz_id_unverified")
print("  -- an offline NAME-index match carrying an unverified legacy id, which is")
print("  exactly the 'merely sharing a name fragment' case the guardrail excludes.")
