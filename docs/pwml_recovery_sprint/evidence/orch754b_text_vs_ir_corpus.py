"""ORCH-754b repair round 1 -- does any text output state a PWML verdict the IR does not reach?

Over every committed ``final_mapped.json`` with interactions (the ORCH-749 corpus), in the two
configurations of ``orch754b_wrapper_base_proof.py`` (``direct``: the payload as committed;
``replay``: the Stage-6 wrapper step re-run offline first), this builds the IR the export
would build (pre-freeze with the DB down, then ``build_pwml_ir``) and compares, row by row:

* the human-facing record -- ``app.reaction_evidence.render_text`` over ``final_mapped.json``
  plus that IR as ``pwml_ir.json`` -- against the IR's own verdict. Round 2: a PWML file is
  placed beside them only when the export would have written one (no blocking IR error and a
  valid IR, the app's own test at ``streamlit_app.py:4969-4971``); otherwise the record must
  say no PWML was written, never a verdict;
* the LLM-bound ``reaction_summary`` -- which must state no PWML verdict at all.

The reviewer's measurement of ``d73e9bc8`` found 2 of 336 rows where the text said "drawn"
and the IR withheld them. Offline, tip only; writes a JSON record beside this script.
"""

from __future__ import annotations

import json
import logging
import re
import subprocess
import sys
from collections import Counter
from copy import deepcopy
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent / "orch754b_text_vs_ir_corpus.json"


class _DownDb:
    last_error = "harvest_db_down"

    def available(self) -> bool:
        return False


def main() -> int:
    logging.disable(logging.CRITICAL)
    import t2pw
    from t2pw.app.reaction_evidence import build_report, render_text
    from t2pw.mapping.map_ids import MappingCache, _extract_global_organism, _rewrite_reaction_protein_enzymes_to_complexes
    from t2pw.pipeline.draft_graph import build_draft_graph
    from t2pw.pipeline.reaction_summary import generate_reaction_summary
    from t2pw.pwml.ir import build_pwml_ir, validate_pwml_ir
    from t2pw.pwml.writer import blocking_pwml_ir_errors
    from t2pw.pwml.prefreeze_resolution import run_prefreeze_resolution

    legs = sorted(p for p in subprocess.run(["git", "ls-files", "runs", "runs_verify"], cwd=REPO, capture_output=True,
                                            text=True, check=True).stdout.split("\n") if p.endswith("/final_mapped.json"))
    tally = {cfg: Counter() for cfg in ("direct", "replay")}
    disagreements = []
    for leg in legs:
        payload = json.loads((REPO / leg).read_text(encoding="utf-8"))
        if not (payload.get("processes") or {}).get("interactions"):
            continue
        for cfg in ("direct", "replay"):
            mapped = deepcopy(payload)
            if cfg == "replay":
                _rewrite_reaction_protein_enzymes_to_complexes(
                    mapped, db=None, cache=MappingCache(Path("unused.json"), enabled=False),
                    global_organism=_extract_global_organism(mapped))
            summary = generate_reaction_summary(build_draft_graph(deepcopy(mapped)), {})
            if "PWML" in summary:
                tally[cfg]["summary_states_a_pwml_verdict"] += 1
            staged = deepcopy(mapped)
            try:
                run_prefreeze_resolution(staged, strict_db=True, db_resolver=_DownDb(), name_index=None)
                ir, ir_report = build_pwml_ir(staged, strict_db=True, name_index=None)
            except Exception as exc:  # noqa: BLE001 -- the export would stop; no verdict exists to compare
                tally[cfg][f"export_stops:{type(exc).__name__}"] += 1
                continue
            written = not blocking_pwml_ir_errors(ir_report) and not validate_pwml_ir(ir).get("errors")
            tally[cfg]["pwml_written" if written else "no_pwml_written"] += 1
            artifacts = {"final_mapped.json": mapped, "pwml_ir.json": ir}
            if written:
                artifacts["pathway.pwml"] = True
            text = render_text(build_report(artifacts))
            blocks = {c.split("  [")[0]: c.split("\n\n")[0] for c in re.split(r"\n(?=INTERACTION \d+  \[)", text)
                      if re.match(r"INTERACTION \d+  \[", c)}
            for number, row in enumerate(ir["processes"]["interactions"], 1):
                tally[cfg]["rows"] += 1
                if not written:
                    want = "PWML: Not recorded (no PWML written"
                elif row["export_status"] == "withheld":
                    want = f"PWML: withheld -- {row['withheld']['reason']}"
                elif not (isinstance(row.get("left"), dict) and isinstance(row.get("right"), dict)):
                    want = "PWML: Not recorded (the IR resolved no entity for an endpoint of this row)"
                else:
                    want = f"PWML: drawn as {row['interaction_type']}"
                if want.startswith("PWML: drawn as"):
                    tally[cfg]["rows_drawn"] += 1
                if want in blocks.get(f"INTERACTION {number}", ""):
                    tally[cfg]["record_states_the_ir"] += 1
                else:
                    disagreements.append({"leg": leg, "config": cfg, "row": number, "ir": want})
    record = {"t2pw_file": str(Path(t2pw.__file__).resolve()), "tally": {k: dict(v) for k, v in tally.items()},
              "disagreements": disagreements}
    OUT.write_text(json.dumps(record, indent=1), encoding="utf-8")
    print(json.dumps(record["tally"], indent=1))
    print(f"disagreements: {len(disagreements)}")
    print("VERDICT: " + ("no text output contradicts the IR" if not disagreements and not any(
        v.get("summary_states_a_pwml_verdict") for v in tally.values()) else "FAILED"))
    return 0 if not disagreements else 1


if __name__ == "__main__":
    sys.exit(main())
