"""SCRATCH-ONLY prototype (never committed): subtract-only no-effect vetoes appended to species._NO_EFFECT
(a veto in grade_sentences only), measuring the cost of closing the residuals this review found.
F1 any "<n> <time unit> ... but" (the shipped entry needs "at" before the number);
F2 a later presence after but / ; written with increased / rose / reached / peaked / recovered;
F3 the code prototype's "no statistically" (not subsumed by statistical / significant);
F4 spelled-out accessions as control words (sentence without "than");
F5 the transient-absence adverb "initially" / "at first"."""
import re
from t2pw.mining import species as S
EXTRA = (
    r"\b\d+(?:\.\d+)?\s*(?:min|minutes?|h|hrs?|hours?)\b[^;]*\bbut\b",
    r"\b(?:but|;)[^.;]*?\b(?:increas\w*|rose|risen|rising|reach\w*|peak\w*|recover\w*)\b",
    r"\bno statistically\b",
    r"^(?!.*\bthan\b).*\b(?:columbia|wassilewskija|landsberg)\b",
    r"\b(?:initially|at first)\b",
)
S._NO_EFFECT = re.compile(S._NO_EFFECT.pattern + "|" + "|".join("(?:%s)" % p for p in EXTRA))
print("PLUGIN ACTIVE rev746e_r2_fix (F1-F5) module=%s" % S.__file__)
