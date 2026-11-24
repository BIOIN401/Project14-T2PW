"""Currency molecules for drawing and drawing metrics: one list, shared by the diagram renderer (local copies) and
checks/layout_metrics (reading order, duplicate names). Compounds nearly every metabolic reaction shares, plus CoA and
SAM/SAH, which curated maps also draw as local copies. Stdlib only, so importing it pulls in nothing else.
(The pipeline's own classification, pipeline/cofactor_policy.py, is a different question and is not used here.)"""
import re

CURRENCY = frozenset({
    "atp", "adp", "amp", "gtp", "gdp", "udp", "nadh", "nad+", "nad", "nadph", "nadp+", "nadp", "nad(p)h", "nad(p)+", "nadp(h)",
    "fad", "fadh2", "h2o", "water", "h+", "hydrogen ion", "proton", "co2", "carbon dioxide", "o2", "oxygen", "nh3", "ammonia",
    "phosphate", "pi", "ppi", "pyrophosphate", "diphosphate", "coa", "coa-sh", "coenzyme a",
    "sam", "s-adenosyl-l-methionine", "s-adenosylmethionine", "sah", "s-adenosylhomocysteine"})


def norm(name) -> str:
    return re.sub(r"\s+", " ", (name or "").strip().lower())


def is_currency(name) -> bool:
    return norm(name) in CURRENCY
