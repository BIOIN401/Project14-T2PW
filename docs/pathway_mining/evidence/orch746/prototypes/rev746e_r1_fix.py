"""SCRATCH-ONLY prototype (never committed): subtract-only no-effect vetoes appended to species._NO_EFFECT
(a veto in grade_sentences only) for the common phrasings the CS-E repair-1 word lists miss.
1. non-significance: no statistically ..., ... (statistical) significance;
2. the genotype joined to the wild type with the wild type FIRST, or a noun after the genotype, or as well as;
3. a delay phrased with a time window, or an absence at a time point followed by but."""
import re
from t2pw.mining import species as S

EXTRA = (
    r"\bno statistically\b", r"\bsignificance\b",
    r"\b(?:wild[- ]?type|col-0|wt)\s+(?:and|or)\s+(?:the\s+)?(?:[\w.-]+\s+){0,2}(?:mutants?|knock-?outs?|lines?|plants?)\b",
    r"\b(?:mutants?|lines?|knock-?outs?|plants?)\s+[a-z]+\s+(?:and|or)\s+(?:also\s+)?(?:in\s+)?(?:the\s+)?(?:wild[- ]?type|col-0|wt)\b",
    r"\bas well as (?:in )?(?:the )?(?:wild[- ]?type|col-0|wt)\b",
    r"\b(?:during|within|in) the first\b", r"\bearly time points?\b", r"\btill\b",
    r"\bat \d+(?:\.\d+)?\s*(?:min|minutes?|h|hrs?|hours?)\b[^;]*\bbut\b",
)
S._NO_EFFECT = re.compile(S._NO_EFFECT.pattern + "|" + "|".join("(?:%s)" % p for p in EXTRA))
print("PLUGIN ACTIVE rev746e_r1_fix module=%s" % S.__file__)


def pytest_configure(config):  # loadable with pytest -p rev746e_r1_fix
    pass
