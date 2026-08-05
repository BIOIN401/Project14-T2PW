"""Stale-index defect replayed on real archived legs, with the counterfactual.

WHAT IT SHOWS
-------------
For each named leg, the quarantine verdict as shipped, then the same payload with
degree-zero resolved against an immutable pre-prune snapshot. This is the
artifact-backed twin of ``repro_stale_index_synthetic.py``: the synthetic case
proves the mechanism, this one proves it happens on production payloads.

PMC12452463's shape is the reference case. Its ``processes.interactions`` are::

    0  Fur binds ferrous iron          entity_1=Fur   (Fur has mapped_ids {})
    1  Fur-Fe2+ represses ent operon   entity_1=Fur
    2  EntD modifies EntB
    3  EntD modifies EntF
    4  EntB binds DHB

Rows 0 and 1 quarantine as ``quarantined_unmapped_entity`` because Fur's
accession was stripped upstream (see DECISIONS D-003 and MASTER_PLAN section 1.4),
so the admitted indices are exactly [2, 3, 4]. After the drop the list holds three
rows, indices 3 and 4 fall out of range, and EntD/EntF are falsely degree-zero.

WHY IT IS COMMITTED
-------------------
It is the chain evidence for MASTER_PLAN section 1: the identity strip and the
index defect are the same failure, not two. C-010 fixes the second; C-033 fixes
the first.

INVOCATION
----------
    .venv/Scripts/python.exe docs/pwml_recovery_sprint/evidence/repro_stale_index_real_artifact.py

No network, no LLM, no database. Reads committed run artifacts only.

ARTIFACT DEPENDENCY
-------------------
Two of the four probed legs live under ``runs_verify/2026-08-04_1754/``, which is
NOT committed by the control-plane setup -- INIT-001 commits it. Until then those
two probes print ``[skip]`` and the other two still run. That is a checkout
difference, not a disagreement.

The ``pathway_context`` below is a RECONSTRUCTION of the Stage-0 context for
enterobactin biosynthesis, used so the coverage check runs in its declared-core
regime. ``allowlist_generator.py`` deliberately passes ``pathway_context=None``
instead, matching what the archived legs actually carry; the two therefore
answer slightly different questions and their coverage numbers are not
comparable. Degree-zero results ARE comparable and agree.
"""

from __future__ import annotations

import copy
import json

from _repo_root import add_src_to_path, require

add_src_to_path()

import t2pw.pipeline.strict_quarantine as SQ  # noqa: E402

ENTEROBACTIN_CONTEXT = {
    "pathway": "enterobactin biosynthesis",
    "organism": "Escherichia coli",
    "key_compounds": ["enterobactin"],
    "key_proteins": ["EntC", "EntB", "EntE", "EntD", "EntF"],
}

LEGS = (
    ("committed 1207 PMC12452463/strict",
     "runs_verify/2026-08-04_1207/papers/PMC12452463/strict/final_mapped.json"),
    ("committed 1234 PMC12096016/strict",
     "runs_verify/2026-08-04_1234/papers/PMC12096016/strict/final_mapped.json"),
    ("INIT-001 1754 PMC12452463/strict",
     "runs_verify/2026-08-04_1754/papers/PMC12452463/strict/final_mapped.json"),
    ("INIT-001 1754 PMC12096016/strict",
     "runs_verify/2026-08-04_1754/papers/PMC12096016/strict/final_mapped.json"),
)


def _patched_degree_zero(snapshot: dict):
    def patched(payload, admissions):
        referenced = SQ._referenced_entity_norms(snapshot["pre"], admissions)
        complexes = SQ._safe_list(SQ._safe_dict(payload.get("entities")).get("protein_complexes"))
        surviving = {
            SQ._normalize(SQ._row_name(row))
            for row in complexes
            if isinstance(row, dict) and SQ._normalize(SQ._row_name(row)) in referenced
        }
        exempt = SQ._complex_component_norms(payload, surviving)
        out = []
        for bucket in SQ._DEGREE_ZERO_BUCKETS:
            for row in SQ._safe_list(SQ._safe_dict(payload.get("entities")).get(bucket)):
                if not isinstance(row, dict):
                    continue
                name = SQ._row_name(row)
                norm = SQ._normalize(name)
                if not norm or norm in referenced or norm in exempt:
                    continue
                out.append({"bucket": bucket, "name": name})
        return out

    return patched


def probe(label: str, relative: str) -> None:
    path = require(relative)
    if path is None:
        return
    payload = json.loads(path.read_text(encoding="utf-8"))

    original_drop = SQ._drop_quarantined_processes
    original_dz = SQ._degree_zero_exports
    snapshot: dict = {}

    def snapshotting_drop(working, admissions):
        snapshot["pre"] = copy.deepcopy(working)
        return original_drop(working, admissions)

    try:
        shipped = SQ.quarantine_and_close(
            copy.deepcopy(payload), strict_db=True, pathway_context=ENTEROBACTIN_CONTEXT
        )
        SQ._drop_quarantined_processes = snapshotting_drop
        SQ._degree_zero_exports = _patched_degree_zero(snapshot)
        fixed = SQ.quarantine_and_close(
            copy.deepcopy(payload), strict_db=True, pathway_context=ENTEROBACTIN_CONTEXT
        )
    finally:
        SQ._drop_quarantined_processes = original_drop
        SQ._degree_zero_exports = original_dz

    shipped_inv = shipped.quarantine_report["strict_invariants"]
    fixed_inv = fixed.quarantine_report["strict_invariants"]

    print(f"--- {label} ---")
    print("  as shipped  degree_zero:", [r["name"] for r in shipped_inv["degree_zero_exports"]],
          "| refusals:", shipped.refusal_reasons, "| ok:", shipped.ok)
    print("  pre-prune   degree_zero:", [r["name"] for r in fixed_inv["degree_zero_exports"]],
          "| refusals:", fixed.refusal_reasons, "| ok:", fixed.ok)
    print("  other invariants (fixed): overlaps=%d unexportable=%s locks=%s converged=%s" % (
        len(fixed_inv["entity_type_overlaps"]),
        [r["name"] for r in fixed_inv["unexportable_entities"]],
        fixed_inv["unaccounted_locked_reactions"],
        fixed_inv["closure_converged"],
    ))
    coverage = fixed.coverage
    print("  coverage (fixed): satisfied=%s ratio=%s matched=%s unmatched=%s" % (
        coverage.get("minimum_core_satisfied"), coverage.get("coverage_ratio"),
        coverage.get("matched_terms"), coverage.get("unmatched_terms"),
    ))
    print()


def main() -> int:
    for label, relative in LEGS:
        probe(label, relative)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
