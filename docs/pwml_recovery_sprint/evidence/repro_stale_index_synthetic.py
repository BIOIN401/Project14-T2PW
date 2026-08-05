"""Minimal synthetic reproduction of the stale positional-index defect (C-010).

WHAT IT SHOWS
-------------
A five-reaction E. coli payload. Indices 0 and 1 quarantine on an undeclared
participant; indices 2, 3 and 4 are admitted and carry enzymes EntC, EntD and
EntF. After ``_drop_quarantined_processes`` compacts the list to three rows, the
admission records still hold indices 2, 3, 4:

  * index 2 resolves to the THIRD surviving row (aliasing -- wrong row),
  * indices 3 and 4 fall out of range and are silently skipped.

So only EntF is seen as referenced, and EntC and EntD are falsely reported
degree-zero. Correct output is ``degree_zero_exports == []`` and ``ok is True``.

WHY IT IS COMMITTED
-------------------
It is the smallest artifact-free proof of the defect and the natural basis for
C-010's first unit test. It needs no run directory, so it works in any checkout
and in an isolated worktree.

INVOCATION
----------
    .venv/Scripts/python.exe docs/pwml_recovery_sprint/evidence/repro_stale_index_synthetic.py

No network, no LLM, no database, no run artifacts.

EXPECTED OUTPUT (on ORIGIN_SHA, i.e. BEFORE C-010)::

    degree_zero_exports: [{"bucket": "proteins", "name": "EntC"},
                          {"bucket": "proteins", "name": "EntD"}]
    report ok          : False
    refusal_reasons    : ['degree_zero_export:2']

AFTER C-010 this script must print ``degree_zero_exports: []`` and ``ok: True``.
"""

from __future__ import annotations

import json

from _repo_root import add_src_to_path

add_src_to_path()

from t2pw.pipeline.strict_quarantine import quarantine_and_close  # noqa: E402

COMPOUNDS = [
    "chorismate", "isochorismate", "DHB", "seryl-AMP", "holo-EntB", "enterobactin",
]


def payload() -> dict:
    return {
        "metadata": {
            "pathway_name": "Enterobactin biosynthesis",
            "pathway_subject": "Metabolic",
            "organism": "Escherichia coli",
            "key_compounds": ["enterobactin"],
            "key_proteins": ["EntD", "EntF"],
        },
        "entities": {
            "species": [{"name": "Escherichia coli", "taxonomy_id": "562"}],
            "subcellular_locations": [{"name": "cytosol"}],
            "compounds": [{"name": name} for name in COMPOUNDS],
            "proteins": [
                {"name": "EntC", "species": "Escherichia coli", "mapped_ids": {"uniprot": "P0AEJ2"}},
                {"name": "EntD", "species": "Escherichia coli", "mapped_ids": {"uniprot": "P19925"}},
                {"name": "EntF", "species": "Escherichia coli", "mapped_ids": {"uniprot": "P11454"}},
            ],
            "protein_complexes": [],
            "nucleic_acids": [],
        },
        "biological_states": [
            {"name": "cytosol", "species": "Escherichia coli", "subcellular_location": "cytosol"}
        ],
        "element_locations": {
            "compound_locations": [
                {"compound": name, "biological_state": "cytosol"} for name in COMPOUNDS
            ],
            "protein_locations": [
                {"protein": name, "biological_state": "cytosol"} for name in ("EntC", "EntD", "EntF")
            ],
        },
        "processes": {
            "reactions": [
                # 0, 1 -- quarantined: the output is declared by no entity bucket.
                {"name": "bad A", "inputs": ["chorismate"], "outputs": ["UNDECLARED-X"],
                 "biological_state": "cytosol"},
                {"name": "bad B", "inputs": ["chorismate"], "outputs": ["UNDECLARED-Y"],
                 "biological_state": "cytosol"},
                # 2, 3, 4 -- admitted. Their enzymes are what must survive closure.
                {"name": "isochorismate synthesis", "inputs": ["chorismate"],
                 "outputs": ["isochorismate"], "enzymes": [{"entity": "EntC"}],
                 "biological_state": "cytosol"},
                {"name": "EntB phosphopantetheinylation", "inputs": ["DHB"],
                 "outputs": ["holo-EntB"], "enzymes": [{"entity": "EntD"}],
                 "biological_state": "cytosol"},
                {"name": "enterobactin assembly", "inputs": ["seryl-AMP", "holo-EntB"],
                 "outputs": ["enterobactin"], "enzymes": [{"entity": "EntF"}],
                 "biological_state": "cytosol"},
            ],
            "transports": [],
            "interactions": [],
        },
    }


def main() -> int:
    result = quarantine_and_close(payload(), strict_db=True)
    invariants = result.quarantine_report["strict_invariants"]

    print("admission states   :", json.dumps(result.states(), indent=2))
    print("surviving reactions:", [row["name"] for row in result.payload["processes"]["reactions"]])
    print("proteins retained  :", [row["name"] for row in result.payload["entities"]["proteins"]])
    print("degree_zero_exports:", json.dumps(invariants["degree_zero_exports"]))
    print("invariants ok      :", invariants["ok"])
    print("report ok          :", result.quarantine_report.get("ok"))
    print("refusal_reasons    :", result.refusal_reasons)
    print()
    if invariants["degree_zero_exports"]:
        print("DEFECT PRESENT: every listed protein is an enzyme on a SURVIVING reaction,")
        print("so the correct degree_zero_exports is [] and the correct verdict is ok=True.")
    else:
        print("DEFECT ABSENT: degree_zero_exports is empty, as it should be.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
