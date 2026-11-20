"""Per-file created counts and message totals for chosen A/B files. usage: ab_rows.py PWAB.json NAME..."""
import json
import sys

d = json.load(open(sys.argv[1]))["files"]
K = ["compounds", "proteins", "protein_complexes", "reactions", "transports", "interactions", "compound_locations",
     "nucleic_acids", "transport_transporters", "biological_states", "label_locations"]
print("| file | " + " | ".join(k.replace("protein_", "p_").replace("transport_", "t_") for k in K) + " | messages |")
print("|---|" + "---:|" * (len(K) + 1))
for n in sys.argv[2:]:
    r = d[n]
    cells = []
    for k in K:
        b, a = r["created"][k]
        cells.append(str(b) if a == b else f"{b} -> {a}")
    eb, ea = r["errors"]
    print(f"| {n} | " + " | ".join(cells) + f" | {eb} -> {ea} |")
