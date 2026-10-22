"""ORCH-764: cut tests/fixtures/orch764/menA_prenyl_diphosphate.json from the ORCH-759 menaquinone leg.

Offline, read-only on the leg. Run ONCE at the base SHA (79080db7): the ``stage6_gate_at_base``
member is the base code's own answer, pinned so the tip can be checked byte for byte against it.

    PYTHONPATH=<worktree>/src python -u docs/pwml_recovery_sprint/evidence/orch764/make_fixture.py

What is recorded and what is reconstructed (also written into the fixture's ``_provenance``):

* ``recorded_db_match`` -- the leg's own ``final_mapped.json`` row ``entities/compounds/4``
  ``db_match``, verbatim: PathWhizCompoundResolver's answer for ``prenyl diphosphate``.
* ``pathbank_compounds[0]`` -- that answer's ``chosen`` row (PathBank compound 874), verbatim, plus
  ONE reconstructed column: ``synonyms = "prenyl diphosphate"``. The leg never recorded the
  ``compounds.synonyms`` text. The recorded rule ``exact_short_name_or_synonym`` (short_name is
  "Ipe") and Stage 6's cached ``chosen_rule: synonym`` for the same query both prove the column holds
  an entry that normalizes to "prenyl diphosphate"; the fixture carries exactly that and nothing more.
* ``stage6_cache_answer`` -- ``data/id_mapping_cache.json`` (committed) ``compounds``
  ``db::prenyl diphosphate::::{}``, verbatim: map_ids' PathBank answer for the same name.
* ``prefreeze_input`` -- the leg's compound rows 3-5 and reaction 1 (rxn_lock_002, MenA) as they
  stood BEFORE pre-freeze resolution: on row 4 every field ``apply_compound_db_resolution`` stamps
  is removed and the name put back to the recorded ``raw_name``; on the unmatched row 3 the five
  provenance fields it stamps are removed; the reaction's references are put back by the module's own
  ``_propagate``. The row's pre-resolution ``confidence`` and ``synonyms`` were not recorded and are
  left out (``_preserve_original_names`` adds the synonym back on a rename).
"""

from __future__ import annotations

import copy
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
WORKTREE = HERE.parents[3]
PRIMARY = Path("C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW")
LEG = PRIMARY / "runs_papers/ORCH-759/legs/menaquinone-strict/2026-10-22_1148/papers/PMC12421875/strict"
CACHE = WORKTREE / "data/id_mapping_cache.json"
OUT = WORKTREE / "tests/fixtures/orch764/menA_prenyl_diphosphate.json"

sys.path.insert(0, str(WORKTREE / "src"))
from t2pw.mapping.map_ids import _apply_name_plausibility_gate  # noqa: E402
from t2pw.pwml.prefreeze_resolution import _PROVENANCE_FIELDS, _propagate  # noqa: E402

#: Every identity field ``db_resolver.apply_compound_db_resolution`` writes after its early return.
APPLIED_IDENTITY_FIELDS = (
    "db_id", "db_row", "pathwhiz_id", "pathbank_compound_id", "pw_compound_id", "hmdb_id", "kegg_id",
    "pubchem_cid", "chebi_id", "pwc_id", "short_name", "description", "cas", "biocyc_id",
    "chemspider_id", "drugbank_id", "mapped_ids",
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pre_prefreeze(final: dict) -> dict:
    """The leg's whole payload with its recorded pre-freeze rename undone (every reference
    ``_propagate`` rewrites, element locations included)."""
    payload = copy.deepcopy(final)
    renames = {}
    for row in payload["entities"]["compounds"]:
        match = row.get("db_match")
        if not isinstance(match, dict):
            continue  # legacy-id rows: pre-freeze consulted no database for them
        if row.get("db_status") == "matched":
            raw = row["raw_name"]
            renames[row["name"]] = raw
            for field in APPLIED_IDENTITY_FIELDS:
                row.pop(field, None)
            row["name"] = raw
            row["synonyms"] = [s for s in row.get("synonyms") or [] if s != raw]
            if not row["synonyms"]:
                row.pop("synonyms")
        for field in _PROVENANCE_FIELDS:
            row.pop(field, None)
    _propagate(payload, renames)
    return payload, renames


def main() -> int:
    final = json.loads((LEG / "final_mapped.json").read_text(encoding="utf-8"))
    rec = final["entities"]["compounds"][4]
    assert rec["raw_name"] == "prenyl diphosphate" and rec["name"] == "Dimethylallylpyrophosphate", rec["name"]
    match = rec["db_match"]
    row874 = dict(match["chosen"])
    row874["synonyms"] = "prenyl diphosphate"

    cache = json.loads(CACHE.read_text(encoding="utf-8"))
    cache_key = "db::prenyl diphosphate::::{}"
    stage6 = cache["compounds"][cache_key]
    gate_at_base = _apply_name_plausibility_gate(copy.deepcopy(stage6), "prenyl diphosphate", kind="compound")

    pre, renames = pre_prefreeze(final)
    assert renames == {"Dimethylallylpyrophosphate": "prenyl diphosphate"}, renames
    compounds = pre["entities"]["compounds"][3:6]
    assert [c["name"] for c in compounds] == [
        "1,4-dihydroxy-2-naphthoic acid", "prenyl diphosphate", "demethylmenaquinone (DMK)"]
    reaction = pre["processes"]["reactions"][1]
    assert reaction["locked_reaction_id"] == "rxn_lock_002", reaction.get("locked_reaction_id")

    fixture = {
        "_provenance": {
            "task": "ORCH-764",
            "leg": str(LEG.relative_to(PRIMARY)).replace("\\", "/"),
            "final_mapped_sha256": sha(LEG / "final_mapped.json"),
            "id_mapping_cache_sha256": sha(CACHE),
            "id_mapping_cache_key": f"compounds/{cache_key}",
            "cut_by": "docs/pwml_recovery_sprint/evidence/orch764/make_fixture.py (see its docstring)",
            "reconstructed": [
                "pathbank_compounds[0].synonyms: the leg did not record compounds.synonyms; the recorded "
                "rule exact_short_name_or_synonym (short_name 'Ipe') and Stage 6's cached chosen_rule "
                "'synonym' prove an entry normalizing to 'prenyl diphosphate' exists",
                "prefreeze_input: pre-freeze stamps removed, recorded raw_name restored, references put back "
                "by prefreeze_resolution._propagate; pre-resolution confidence/synonyms not recorded, left out",
            ],
            "stage6_gate_at_base": "map_ids._apply_name_plausibility_gate(stage6_cache_answer, "
                                   "'prenyl diphosphate', kind='compound') computed at base 79080db7",
        },
        "pathbank_compounds": [row874],
        "recorded_db_match": match,
        "stage6_cache_answer": stage6,
        "stage6_gate_at_base": gate_at_base,
        "prefreeze_input": {"entities": {"compounds": compounds}, "processes": {"reactions": [reaction]}},
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(fixture, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print("leg final_mapped.json sha256", fixture["_provenance"]["final_mapped_sha256"])
    print("id_mapping_cache.json sha256", fixture["_provenance"]["id_mapping_cache_sha256"])
    print("stage6 gate at base:", gate_at_base.get("status"), (gate_at_base.get("resolution") or {}))
    print("wrote", OUT.relative_to(WORKTREE), "sha256", sha(OUT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
