"""ORCH-763: two PRE-EXISTING ways a pooled row still spans incompatible states, outside D-109's
location rule and NOT fixed by this lane. Offline; run with base and tip on PYTHONPATH.

H1  claim chaining through an EMPTY species (cell type, tissue alike): ``_absorb`` keeps the first
    defined state's fields, so a row stating no species lets an E. coli row and a B. subtilis
    row join one pooled row.
H2  one state LABEL defined differently by two draws: ``_conflict`` compares states only when the
    labels differ, so two rows named "state A" -- cytosol in draw 1, mitochondrial matrix in
    draw 2 -- merge without any comparison.
"""
import t2pw.pipeline.stage_one_ensemble as E


def row(name, state):
    return {"name": name, "inputs": ["X1"], "outputs": ["X2"], "biological_state": state, "evidence": name}


def draw(states, *rows):
    return {"biological_states": states, "processes": {"reactions": list(rows)}}


S = [{"name": "plain", "subcellular_location": "cytosol"},
     {"name": "coli", "subcellular_location": "cytosol", "species": "Escherichia coli"},
     {"name": "subtilis", "subcellular_location": "cytosol", "species": "Bacillus subtilis"}]
merged, _ = E.merge_draws([draw(S, row("p", "plain")), draw(S, row("c", "coli")), draw(S, row("s", "subtilis"))])
print("H1 empty-species chain: pooled rows", [r["name"] for r in merged["processes"]["reactions"]],
      "(1 row = E. coli and B. subtilis joined through the row with no species)")
merged, _ = E.merge_draws([draw([{"name": "state A", "subcellular_location": "cytosol"}], row("d1", "state A")),
                           draw([{"name": "state A", "compartment_canonical": "mitochondrial_matrix"}], row("d2", "state A"))])
print("H2 one label, two definitions: pooled rows", [r["name"] for r in merged["processes"]["reactions"]],
      "(1 row = cytosol and mitochondrial matrix joined because the label is equal)")

# How often H2 can bite in kept, complete draws: a label two draws of one set define differently.
import json  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

if len(sys.argv) > 1:
    ev = Path(sys.argv[1]) / "docs" / "pwml_recovery_sprint" / "evidence"
    for name, folder in (("menaquinone-751", ev / "orch754a_inputs" / "PMC12421875_menaquinone"),
                         ("enterobactin-751", ev / "orch754a_inputs" / "PMC12096016_enterobactin"),
                         ("chlorophyll-751", ev / "orch752_chlorophyll_payloads")):
        draws = [json.loads((folder / f"control__run{i}__payload.json").read_text(encoding="utf-8")) for i in (1, 2, 3)]
        defs = {}
        for d, p in enumerate(draws, 1):
            for key, srow in E._states_by_key(p).items():
                prof = E._state_profile(key, E._states_by_key(p))
                defs.setdefault(key, {})[d] = tuple(prof.get(f, "") for f in ("location", "cell_type", "tissue", "species"))
        clash = {k: v for k, v in defs.items() if len(set(v.values())) > 1}
        print(f"H2 in {name}: {len(defs)} state labels, {sum(1 for v in defs.values() if len(v) > 1)} defined by 2+ draws, "
              f"{len(clash)} defined differently: {clash}")
