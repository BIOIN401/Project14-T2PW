"""SCRATCH-ONLY prototype of BIO-D-R2-2's expected correction: the negated-comparison / delay
window is already inside one split sentence, so it need not stop at a full stop or colon at all
("(Fig. 3A)", "(ca. 40 pmol/g)", "e.g."); a wider veto window only refuses more."""
import re
from t2pw.mining import species as S
S._NO_EFFECT = re.compile("(?:%s)|(?:%s)" % (S._NO_EFFECT.pattern,
    r"\b(?:(?:did|does|do) not|fail(?:ed|s)? to)\b[^;]{0,60}?\b(?:more|less|lesser|lower|higher|"
    r"fewer|greater|further|again|than|until|significantly)\b"))
print("PLUGIN ACTIVE rev745d_r2_proto_win module=%s" % S.__file__)
