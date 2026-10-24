"""One-shot ORCH-766 pin move (merge rule 4): Chunk E residual species ROWS 27 -> 19.

Edits ``tests/test_strict_quarantine_real_artifact_replay.py`` and ``docs/change_log.md``
together, as that module requires (they are asserted against each other). Refuses
unless each target text appears exactly once. Not a general tool; it exists so the
edit is reviewable as exact substitutions (precedent: ``evidence/c121_changelog_edit.py``).
Run from the worktree root.
"""
from pathlib import Path

TEST = Path("tests/test_strict_quarantine_real_artifact_replay.py")
LOG = Path("docs/change_log.md")


def sub(path, pairs):
    raw = path.read_bytes().decode("utf-8")
    crlf = "\r\n" in raw
    text = raw.replace("\r\n", "\n")
    for old, new in pairs:
        assert text.count(old) == 1, (path, text.count(old), old[:60])
        text = text.replace(old, new, 1)
    path.write_bytes((text.replace("\n", "\r\n") if crlf else text).encode("utf-8"))


sub(TEST, [
    ('''#: The species-metadata class is untouched: 19 legs / 27 rows before and after.
''', '''#: The species-metadata class is untouched: 19 legs / 27 rows before and after.
#:
#: **MOVED by ORCH-766 / D-111 section 1, 2026-10-23: rows 27 -> 19, legs unchanged.**
#: ``quarantine_and_close`` now removes a species row nothing in the pathway names.
#: Eight such rows sat in three cohort legs, each carrying
#: ``species_missing_taxonomy`` and ``species_missing_classification`` (16 error rows):
#: ``runs/2026-07-28_2122/papers/PMC12421875__.../research`` ("L. lactis subsp. lactis
#: KF147"), ``runs/2026-07-28_2122/papers/PMC12657337__.../strict`` ("Bacillus
#: subtilis") and ``runs/2026-08-02_2130/papers/PMC12421875/strict`` (six: Lactiplantibacillus
#: plantarum, Escherichia coli, Staphylococcus aureus, Mycobacterium tuberculosis,
#: Bacillus subtilis, Flavobacterium meningosepticum). Each is unreferenced by every
#: independent check -- no scalar of the quarantined payload names it, no IR state
#: resolves to it, no protein the writer resolves to it, not IR ``species[0]``
#: (``docs/pwml_recovery_sprint/evidence/orch766/chunk_e_compare.out``). All three legs
#: keep a REFERENCED species row without taxonomy, so each still fails the
#: required-field gate: ``RESIDUAL_CODES_BY_LEG`` stays 19 / 19, and
#: ``FULL_STACK_BASELINE`` does not move (``required_contract_pass`` 9, ``exportable``
#: 9). No leg's verdict changed and no gate was relaxed (merge rule 6).
'''),
    ('''RESIDUAL_CODES_BY_ROW: Dict[str, int] = {
    "species_missing_classification": 27,
    "species_missing_taxonomy": 27,
}''', '''RESIDUAL_CODES_BY_ROW: Dict[str, int] = {
    "species_missing_classification": 19,
    "species_missing_taxonomy": 19,
}'''),
])

sub(LOG, [
    ("9/28 required contract; 9/9 IR; 9/28 exportable; by leg 19/19, by row 27/27).",
     "9/28 required contract; 9/9 IR; 9/28 exportable; by leg 19/19, by row 19/19)."),
    ('''`species_missing_classification` (19 legs, 27 rows), `species_missing_taxonomy`
(19 legs, 27 rows). The gate wants a numeric''',
     '''`species_missing_classification` (19 legs, 19 rows), `species_missing_taxonomy`
(19 legs, 19 rows). The gate wants a numeric'''),
    ('''> was not exported before. The species-metadata class is untouched at 19 legs / 27
> rows and remains the next pipeline defect.
''', '''> was not exported before. The species-metadata class is untouched at 19 legs / 27
> rows and remains the next pipeline defect.

> **ORCH-766 / D-111 § 1 moved the row counts 27 -> 19, 2026-10-23; legs stay 19.**
> `quarantine_and_close` now removes a species row that nothing in the pathway names.
> Eight such rows sat in three cohort legs (16 error rows); each of those legs keeps a
> referenced species row without taxonomy and still fails the gate, so `required
> contract` stays 9 / 28 and `fully exportable` stays 9 / 28. The by-row restatement
> above moved with it (`by row 27/27` -> `19/19`). Rows and evidence:
> `docs/pwml_recovery_sprint/ORCH-766-RESULT.md`.
'''),
])
print("PIN MOVE EDIT OK")
