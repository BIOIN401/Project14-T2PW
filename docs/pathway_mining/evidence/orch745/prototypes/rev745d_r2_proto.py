"""SCRATCH-ONLY prototype (never committed, never written into a worktree) of BIO-D-R2-1's
expected correction for species._excluded at 686ae134: (1) a comma inside a parenthetical
("(Fig. 3A, B)", "(9S,13S)") closes nothing; (2) only a comma followed by whitespace closes a
clause; (3) the LAST opener in the gap decides whether a clause is still open at the form;
(4) a head is kept only in the shape "X, <opener> ..., <form>" (opener adjacent to X, one
opener); anything else with an opener in the gap is cut off (fail closed)."""
import re
from t2pw.mining import species as S

_PAREN = re.compile(r"\([^()]*\)")
_CLOSE = re.compile(r",\s")


def _excluded(clause, hit, form_start):
    text = _PAREN.sub(lambda m: " " * len(m.group(0)), clause)
    gap = text[hit[1]:form_start]
    openers = list(S._NEW_SUBJECT.finditer(gap))
    if openers:
        if not _CLOSE.search(gap[openers[-1].end():]):
            return True
        if len(openers) > 1 or gap[:openers[0].start()].strip():
            return True
    opener = None
    for opener in S._NEW_SUBJECT.finditer(text[:hit[0]]):
        pass
    return bool(opener and not _CLOSE.search(text[opener.end():hit[0]]) and _CLOSE.search(gap))


S._excluded = _excluded
print("PLUGIN ACTIVE rev745d_r2_proto module=%s" % S.__file__)
