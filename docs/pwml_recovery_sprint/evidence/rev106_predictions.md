# REV-106 predictions, recorded BEFORE any execution

Written by REV-106 (independent reviewer) at TIP ba2bf68 / BASE c7fb5c5, before running
any test, probe or mutation. R5 requires predictions first.

| # | Prediction |
|---|---|
| P1 | `git ls-files '*quarantine_report.json'` at tip returns 72 paths |
| P2 | Independently derived `withheld` total == 97 |
| P3 | Independently derived `with_matched_forbidden` == 26 |
| P4 | focused `tests/test_c102_coverage_denominator.py` at TIP: 14 passed, exit 0 |
| P5 | same file at BASE c7fb5c5: 2 failed / 12 passed |
| P6 | SMOKE 22-file selection at TIP: 503 passed (473 + 14 + 16) |
| P7 | gold-readers 22-file selection at TIP: 456 passed / 8 skipped / exit 0 |
| P8 | All 8 mutations (M1-M7 + R5) go RED when replayed |
| P9 | A7 preservation: with one c102 test broken, harness aborts, 0 mutations applied |
| P10 | A13: reintroducing `write_text(newline="")` turns the byte-exactness test RED |
| P11 | A8 both directions: prose "3 errors" -> errors=0; genuine `1 failed` -> failed=1 |
| P12 | REV-104's bytes=78077 is wrong; 79745-1673 = 78072 is the correct value |
| P13 | No `src/` path in the diff; no `==` -> `>=` relaxation on the four pins |
| P14 | TEST_MATRIX.md edits above line 477 are line-neutral (base line count == tip line count above 477) |
