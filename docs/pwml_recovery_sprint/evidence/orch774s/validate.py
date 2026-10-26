"""ORCH-774s validation: "would import" on the reviewer's corpus leg (replay output at the ORCH-772b r1 tip) and the
ORCH-767 legs' PWML files. Read-only."""
import importlib.util
import json
import sys
from pathlib import Path

EV = Path(sys.argv[1])
spec = importlib.util.spec_from_file_location("ms", EV / "orch770_morning_summary.py")
ms = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ms)

corpus = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
replay_dir = Path(sys.argv[3])
row = next(r for r in corpus["legs"] if r["id"] == "runs_verify/2026-09-06_1425/PMC12376012")
cov = row["replay"]["pwml_coverage"]
dropped = ms.no_enzyme_reactions(replay_dir)
head = cov["total"] - cov["by_design_withheld"]
print(f"PMC12376012 replay: drawn {cov['drawn']} headline denominator {head} -> {100 * cov['drawn'] / head:.1f}%; "
      f"no-enzyme non-spontaneous reactions in the PWML {dropped}; would import {cov['drawn'] - dropped}/{head} "
      f"= {100 * (cov['drawn'] - dropped) / head:.1f}%")
for leg in sorted(Path(sys.argv[4]).glob("*/*/papers/*/strict")):
    print(f"{leg.parts[-5]:22s} dropped on import: {ms.no_enzyme_reactions(leg)}")
