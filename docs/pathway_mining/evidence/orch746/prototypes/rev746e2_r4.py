"""SCRATCH-ONLY prototype (never committed): a fourth reading appended to species._READINGS.
The 686ae134 reading, plus: with parentheticals blanked, a participant before an opener is cut
off when two or more ", " follow the opener before the form (the clause close is ambiguous:
", with X, its derivative, reduced", ", which, in turn, resulted in ..."). Appending a reading
to the all-must-agree tuple can only remove credit."""
from t2pw.mining import species as S


def _excluded_unambiguous_close(clause, hit, form_start):
    if S._excluded(clause, hit, form_start):
        return True
    text = S._PARENTHETICAL.sub(lambda m: " " * len(m.group(0)), clause)
    gap = text[hit[1]:form_start]
    opened = S._NEW_SUBJECT.search(gap)
    return bool(opened and len(S._CLAUSE_CLOSE.findall(gap[opened.end():])) > 1)


S._READINGS = S._READINGS + (_excluded_unambiguous_close,)
print("PLUGIN ACTIVE rev746e2_r4 readings=%d module=%s" % (len(S._READINGS), S.__file__))


def pytest_configure(config):  # loadable with pytest -p rev746e2_r4
    pass
