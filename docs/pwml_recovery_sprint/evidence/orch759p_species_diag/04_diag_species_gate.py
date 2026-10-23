"""Offline diagnosis of the ORCH-759p menaquinone species gate failure.

Imports t2pw (run through bounded_run.py). No network: the NCBI client is a stub
that records every query and answers "no match". Writes nothing outside
C:/t/bt-spc759.
"""
import copy
import json
import sys

from t2pw.mapping import map_ids as M
from t2pw.pwml import ir as IR

LEG = ("C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW/runs_papers/ORCH-759p/legs/"
       "menaquinone-strict/2026-10-22_2154/papers/PMC12421875/strict")
ABBR = "L. lactis subsp. lactis KF147"


class _Resp:
    status_code = 200
    text = ""

    def json(self):
        return {"esearchresult": {"idlist": []}}


class StubClient:
    def __init__(self):
        self.queries = []

    def get(self, url, params=None, **_kw):
        self.queries.append((url.rsplit("/", 1)[-1], (params or {}).get("term") or (params or {}).get("id")))
        return _Resp()


def species_errors(payload):
    rep = IR.validate_required_pwml_contract(payload, strict_db=True)
    return [(e["code"], e.get("pointer"), e.get("species_name")) for e in rep["errors"]
            if str(e.get("code", "")).startswith("species_")], [e["code"] for e in rep["errors"]]


final = json.load(open(f"{LEG}/final_mapped.json", encoding="utf-8"))
print("1. gate on recorded final_mapped.json:", species_errors(final))

print("2. eligibility of", repr(ABBR))
print("   _abbreviated_binomial_parts ->", M._abbreviated_binomial_parts(ABBR))
print("   _is_bare_binomial           ->", M._is_bare_binomial(ABBR))
print("   _binomial_from_organism     ->", repr(M._binomial_from_organism(ABBR)))
print("   _deterministic_species_name ->", repr(IR._deterministic_species_name(ABBR)))

# Rows as backfill saw them: the two full-name rows already resolved in the leg.
rows = [
    {"name": "Lactococcus lactis", "taxonomy_id": "1358", "classification": "Prokaryote"},
    {"name": "Lactococcus lactis subsp. lactis KF147", "taxonomy_id": "684738", "classification": "Prokaryote"},
    {"name": ABBR},
]
print("3. donors:", [(d["name"], d["taxonomy_id"], d["rank_qualified"]) for d in M._species_alias_donors(rows)])
print("   _species_alias_expansion(offline) ->",
      M._species_alias_expansion(ABBR, M._species_alias_donors(rows), client=None, enable_ncbi=False))

p = {"entities": {"species": copy.deepcopy(rows)}}
stub = StubClient()
rep = M.backfill_species_taxonomy(p, client=stub, enable_ncbi=True, cache=None)
print("4. backfill (stub NCBI, no-match) report:", json.dumps(rep))
print("   NCBI queries sent:", stub.queries)
print("   row 3 after:", json.dumps(p["entities"]["species"][2]))

# 5. ORCH-759 (1148) shape: no full-name twin; only the species-rank donor.
rows_1148 = [
    {"name": "Lactococcus lactis", "taxonomy_id": "1358", "classification": "Prokaryote"},
    {"name": ABBR},
]
p2 = {"entities": {"species": copy.deepcopy(rows_1148)}}
stub2 = StubClient()
rep2 = M.backfill_species_taxonomy(p2, client=stub2, enable_ncbi=True, cache=None)
print("5. 1148-shape backfill report:", json.dumps(rep2), "queries:", stub2.queries)

# 6. Counterfactual: the row carries the strain id its full-name twin carries.
sim = copy.deepcopy(final)
sim["entities"]["species"][2]["taxonomy_id"] = "684738"
sim["entities"]["species"][2]["classification"] = "Prokaryote"
errs, all_codes = species_errors(sim)
print("6. gate with row 2 = 684738/Prokaryote: species errors", errs, "all error codes", all_codes)

# 7. What the pre-freeze species stage would then do to the names.
try:
    from t2pw.pwml.prefreeze_resolution import resolve_species_prefreeze
    sim2 = copy.deepcopy(sim)
    for r in sim2["entities"]["species"]:
        r.pop("species_canonicalization", None)
    summ = resolve_species_prefreeze(sim2)
    print("7. prefreeze rename_map:", summ.get("rename_map"), "declined:",
          [d.get("source") for d in summ.get("renames_declined", [])] if isinstance(summ.get("renames_declined"), list) else summ.get("renames_declined"))
    print("   names after:", [(r.get("name"), r.get("taxonomy_id")) for r in sim2["entities"]["species"]])
    print("   gate after prefreeze:", species_errors(sim2))
except Exception as exc:  # noqa: BLE001
    print("7. prefreeze replay failed:", type(exc).__name__, exc)
sys.stdout.flush()
