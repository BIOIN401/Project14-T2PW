# REV-F150 — predictions, written BEFORE any measurement

Reviewer: independent (not the author of the edit). Recorded at the pre-edit tip
`4c077f012793260f05d5e0518c6459ca9ad13cc6`, gold blob
`aee8cb4f1da3d417f36206407867585622b741c0`, before running probe A, the gold-reader
arms, or the instrument-sensitivity measurement.

Anything measured that is not on this list is a FINDING, not a footnote.

## P1 — reproduction of REV-F150.md § 1

```
forbidden_match('5-aminolevulinic acid'    ) -> '5-aminolevulinic acid'
forbidden_match('delta-aminolevulinic acid') -> None
forbidden_match('δ-aminolevulinic acid'    ) -> None
forbidden_identifiers[0].aliases : ['ALA', 'porphobilinogen', 'protoporphyrin IX',
                                    'succinyl-CoA', 'coproporphyrinogen III',
                                    'uroporphyrinogen III']
```

## P2 — the two proposed strings are ONE alias after normalization

`goldset._GREEK` already carries `"δ": "delta"` (`goldset.py`, the `_GREEK` table), and
`normalize_name` applies it before comparison. I therefore predict

```
normalize_name('δ-aminolevulinic acid') == normalize_name('delta-aminolevulinic acid')
                                        == 'delta aminolevulinic acid'
```

so the SECOND proposed string is redundant given the first: adding either alone produces
exactly the same `forbidden_match` behaviour as adding both. This is harmless but it is a
fact about the edit the proposal does not state, and I record it in advance.

## P3 — gold-readers, pre-edit arm

22-file selection, one file per process: **456 passed / 8 skipped / exit 0** (the C-103
baseline).

## P4 — gold-readers, post-edit arm

**456 passed / 8 skipped / exit 0**, unchanged — or a delta I can explain term by term.

## P5 — the recorded prediction I was given

Priority 1 rises **5 → 6** on T-107's committed artifacts, and **6 is still `PASS`** under
D-073 (range 0–6).

## P6 — a mover I predict IN ADVANCE so that it is not an unpredicted one

Forbidden aliases are not read by Priority 1 alone. `acceptance.forbidden_coverage_match`
reads the same `forbidden_match`, and D-072 withholds every matched term from BOTH the
accepted coverage numerator and the accepted denominator. So if any T-107 leg's Stage-0
requested-core draw contains an ALA delta spelling, **Priority 4/5 accepted coverage will
also move**, and the `coverage_status` of a leg could in principle flip to
`undefined_every_term_forbidden`. I predict this seam is reachable in principle; whether
it fires on this corpus is what the measurement decides.

## P7 — V1

I expect to confirm the ASCII delta spelling at `acceptable_enzymes[1].aliases`
("erythroid delta-aminolevulinic acid synthase"). I predict the **Unicode `δ` spelling
appears nowhere in this case**, so V1's internal-consistency argument supports the ASCII
string only.

## P8 — V2 is the condition I expect to be in trouble

`forbidden_identifiers[0].reason` opens "HALLUCINATION TEST: zero occurrences in the
entire 67,304-character file, body and references alike." The case's own
`acceptable_enzymes[1].quote` is "inhibiting the translation of erythroid
delta-aminolevulinic acid synthase (ALAS2), a key enzyme in the heme synthesis pathway".
So the string `delta-aminolevulinic acid` **does occur in the file**. I predict the edit
places under that `reason` an alias whose own stated justification is factually false of
it, and that no prose in `relevance_note` / `export_rationale` / `notes` demands the
addition. I record this before reading further so the finding cannot be back-fitted.
