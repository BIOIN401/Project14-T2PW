"""ORCH-759a non-vacuity: each mutation of the fix must turn the named tests red.

Usage (from the worktree root, through bounded_run.py):
    python mutations.py <python exe> <basetemp root>

Each mutation edits ONE source file in place (exactly one occurrence, or the mutation is
reported INVALID), runs pytest on its target tests in a child process, and restores the file
byte for byte in ``finally``. Exit 0 only when every mutation was killed and every file is
restored. Offline; no model, network or database.
"""

import subprocess
import sys
from pathlib import Path

PY, BASETEMP = sys.argv[1], Path(sys.argv[2])
BASETEMP.mkdir(parents=True, exist_ok=True)  # pytest creates --basetemp but not its parent
ROOT = Path.cwd()
MAP_IDS = ROOT / "src/t2pw/mapping/map_ids.py"
CURATOR = ROOT / "src/t2pw/curation/interactive_curator.py"
UNIT = "tests/test_orch759a_duplicate_protein_fold.py"
APP = "tests/test_orch759a_fold_app_path.py::test_new_the_fold_fires_through_the_apps_own_stage6_call[three_draws]"

MUTATIONS = [
    ("M1 walker reads the record key", CURATOR,
     "                if key in PROMPT_OMITTED_PAYLOAD_KEYS:\n                    continue\n", "",
     [f"{UNIT}::test_new_the_record_key_is_skipped_whatever_it_holds"]),
    ("M2 survivor need not be referenced", MAP_IDS,
     "if (other_index == index or not is_referenced[other_index] or not shared",
     "if (other_index == index or not shared",
     [f"{UNIT}::test_new_two_unreferenced_rows_never_fold_into_each_other"]),
    ("M3 species ignored", MAP_IDS,
     "                    or _species_hint_from_row(other) != species):", "                    ):",
     [f"{UNIT}::test_new_a_different_or_unknown_species_never_folds[different_species]"]),
    ("M4 references ignored", MAP_IDS,
     "if is_referenced[index] or not mine", "if not mine",
     [f"{UNIT}::test_new_a_referenced_duplicate_is_never_folded"]),
    ("M5 every alias counts as a reference", MAP_IDS,
     "own += [name for name in _row_name_variants(row) if _exact_key(name) not in others]",
     "own += _row_name_variants(row)",
     [f"{UNIT}::test_new_an_alias_another_row_holds_does_not_count_as_a_reference_to_the_duplicate",
      f"{UNIT}::test_g9_the_stage6_pass_leaves_a_payload_the_final_gate_passes"]),
    ("M6 raw mapped_ids instead of admitted ids", MAP_IDS,
     "admitted = {index: _admitted_ids(row) for index, row in rows}",
     "admitted = {index: identity_admission.external_accessions(row) for index, row in rows}",
     [f"{UNIT}::test_new_a_candidate_withheld_or_provisional_accession_never_folds"]),
    ("M7 gated on rule 3's pooled draws", MAP_IDS,
     "        _fold_unreferenced_duplicate_proteins(mapped, report)\n",
     "        from t2pw.pipeline.stage_one_ensemble import _pooled_draws\n"
     "        if _pooled_draws() > 1:\n            _fold_unreferenced_duplicate_proteins(mapped, report)\n",
     [f"{UNIT}::test_new_the_fold_fires_at_any_draw_count", APP]),
    ("M8 quotes not moved", MAP_IDS,
     "            if moved:\n                refs", "            if False:\n                refs",
     [f"{UNIT}::test_new_the_fold_keeps_the_name_and_every_quote_and_records_them"]),
    ("M9 report pointers not moved", MAP_IDS,
     "    _move_protein_pointers(report, index_map, set(survivor_of))\n", "",
     [f"{UNIT}::test_new_the_report_pointers_follow_the_fold"]),
]

survivors = 0
for number, (label, path, old, new, targets) in enumerate(MUTATIONS, 1):
    original = path.read_bytes()
    text = original.decode("utf-8")
    newline = "\r\n" if "\r\n" in text else "\n"  # a Windows checkout (core.autocrlf) writes CRLF
    old, new = old.replace("\n", newline), new.replace("\n", newline)
    if text.count(old) != 1:
        print(f"{label}: INVALID (pattern occurs {text.count(old)} times)")
        survivors += 1
        continue
    try:
        path.write_bytes(text.replace(old, new).encode("utf-8"))
        run = subprocess.run([PY, "-m", "pytest", "-q", "-p", "no:cacheprovider",
                              f"--basetemp={BASETEMP / f'm{number}'}", *targets],
                             capture_output=True, text=True, encoding="utf-8", errors="replace")
    finally:
        path.write_bytes(original)
    tail = [line for line in run.stdout.splitlines() if " passed" in line or " failed" in line or " error" in line]
    killed = run.returncode == 1 and "failed" in (tail[-1] if tail else "")
    survivors += not killed
    print(f"{label}: {'KILLED' if killed else 'SURVIVED'} (pytest exit {run.returncode}; {tail[-1] if tail else 'no summary'})")
    for line in [line for line in run.stdout.splitlines() if line.startswith(("FAILED", "ERROR"))][:8]:
        print("    " + line[:220])

restored = subprocess.run(["git", "diff", "--quiet", "--", str(MAP_IDS), str(CURATOR)]).returncode == 0
print(f"sources restored to HEAD: {restored}")
print(f"mutations survived: {survivors} of {len(MUTATIONS)}")
sys.exit(0 if survivors == 0 and restored else 1)
