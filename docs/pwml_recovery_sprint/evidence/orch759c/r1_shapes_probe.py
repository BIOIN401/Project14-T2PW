"""ORCH-759c correction round 1: behaviour of the N1/N2/N4 shapes under the code on PYTHONPATH.

Prints, per shape, ok / refusal reasons / overlaps / admission states / removed
entities and locations. Run once on base src, once at the round-0 tip, once at the
round-1 tip. For the chlorophyll leg it also prints the removed-location digest and
every surviving location row that names no entity of its own kind. Prints only.
"""
import gzip
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "tests"))

from test_strict_quarantine_contract_alignment import _base  # noqa: E402
from test_strict_quarantine_real_artifact_replay import _required_contract_codes  # noqa: E402
from t2pw.pipeline.process_normalizer import _entity_name_norms, _normalize  # noqa: E402
from t2pw.pipeline.strict_quarantine import quarantine_and_close  # noqa: E402

GS = "glutathione synthetase"
LOCATES = {
    "compound_locations": ("compound", ("compounds",)),
    "protein_locations": ("protein", ("proteins", "protein_complexes")),
    "nucleic_acid_locations": ("nucleic_acid", ("nucleic_acids",)),
    "element_collection_locations": ("element_collection", ("element_collections",)),
}


def wrapper(p):
    return next(c for c in p["entities"]["protein_complexes"] if c["name"] == f"{GS} complex")


def c3ok():
    p = _base()
    p["entities"]["compounds"].append({"name": GS})
    return p


def c7():
    p = _base()
    p["entities"]["element_collections"] = [{"name": GS}]
    return p


def n2a():
    p = c3ok()
    next(r for r in p["entities"]["proteins"] if r["name"] == GS).pop("mapped_ids")
    return p


def n2b():
    p = _base()
    wrapper(p)["components"] = [{"name": "GS holoenzyme", "stoichiometry": 1}]
    p["entities"]["protein_complexes"].append(
        {"name": "GS holoenzyme", "class": "protein_complex", "species": "Homo sapiens",
         "components": [{"name": GS, "stoichiometry": 1}]}
    )
    return p


def n2c():
    p = _base()
    wrapper(p)["components"] = [{"name": "GSS", "stoichiometry": 1}]
    p["entities"]["nucleic_acids"] = [{"name": "GSS"}]
    return p


def show(tag, payload):
    r = quarantine_and_close(payload, strict_db=True)
    q = r.quarantine_report
    print(f"== {tag}: ok={r.ok} refusal={r.refusal_reasons} overlaps={q['strict_invariants']['entity_type_overlaps']}")
    for row in r.admissions:
        print(f"   {row.get('name')!r}: {row.get('state')} {row.get('reason') or ''}")
    print("   removed entities:", sorted((e["bucket"], e["name"], e["reason"]) for e in r.removed_entity_report["removed_entities"]))
    print("   exporter contract codes:", _required_contract_codes(r.payload) if r.ok else "(not reached: quarantine refused)")


for tag, build in (("c3ok", c3ok), ("c7", c7), ("n2a", n2a), ("n2b", n2b), ("n2c", n2c)):
    show(tag, build())

FIX = ROOT / "tests" / "fixtures" / "orch759c"
leg = json.loads(gzip.decompress((FIX / "chlorophyll_PMC5130987_final_mapped.json.gz").read_bytes()))
ctx = json.loads((FIX / "chlorophyll_PMC5130987_requested_context.json").read_text(encoding="utf-8"))
res = quarantine_and_close(leg, strict_db=True, pathway_context=ctx, mode="pathwhiz")
lines = sorted(f"{e['bucket']}|{e['name']}|{e['reason']}|{e.get('biological_state')}" for e in res.removed_entity_report["removed_locations"])
print(f"== leg: ok={res.ok} resulting={res.quarantine_report['resulting_payload_hash']} removed_locations n={len(lines)} sha={hashlib.sha256(chr(10).join(lines).encode()).hexdigest()}")
for line in lines:
    if "nucleic_acid_locations" in line:
        print("   removed:", line)
ents = res.payload["entities"]
for bucket, (field, owners) in LOCATES.items():
    own = {n for b in owners for n in _entity_name_norms(ents.get(b) or [])}
    for row in (res.payload.get("element_locations") or {}).get(bucket) or []:
        name = row.get(field) or row.get("entity")
        if _normalize(name) not in own:
            print(f"   CROSS-KIND survivor: {bucket} {name!r} state={row.get('biological_state')!r}")
