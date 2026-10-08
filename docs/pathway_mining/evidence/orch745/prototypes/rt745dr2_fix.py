"""SCRATCH-ONLY prototype (never committed, never written into a worktree).
_excluded: the head before a comma-led clause that closes before the form is kept ONLY when
the clause opens right after that participant (it modifies the participant itself:
"JA-Ile, which ..., accumulated"); a clause that opens after other words ("JA-Ile
accumulated after wounding, which, in turn, ...", "OPDA accumulated, with OPC-8:0, the
product, undetectable") cuts the head off whatever commas it holds (fd7ea63a behaviour)."""
from t2pw.mining import species as S


def _excluded(clause, hit, form_start):
    gap = clause[hit[1]:form_start]
    opened = S._NEW_SUBJECT.search(gap)
    if opened and ("," not in gap[opened.end():] or gap[:opened.start()].strip()):
        return True
    opener = None
    for opener in S._NEW_SUBJECT.finditer(clause[:hit[0]]):
        pass
    return bool(opener and "," not in clause[opener.end():hit[0]] and "," in gap)


S._excluded = _excluded
print("PLUGIN ACTIVE: rt745dr2_fix, module=%s" % S.__file__)
