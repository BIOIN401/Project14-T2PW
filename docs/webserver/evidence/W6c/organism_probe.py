"""Before (the base search, copied verbatim) and after (views.search_organisms) for the E2E-3 queries."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "src"))
from t2pw.service import organisms
from t2pw.web.views import search_organisms


def before(organisms, q, limit=30):
    q = q.strip().casefold()
    prefix, word, sub = [], [], []
    for o in organisms:
        low = o.label.casefold()
        if low.startswith(q):
            prefix.append(o)
        elif (" " + q) in low or ("(" + q) in low:
            word.append(o)
        elif q in low:
            sub.append(o)
        if len(prefix) >= limit:
            break
    return (prefix + word + sub)[:limit]


orgs = organisms.load_organisms()
for q in ["mouse", "human", "rat", "yeast", "E. coli", "e coli", "escherichia coli", "arabidopsis"]:
    for name, fn in (("before", before), ("after ", search_organisms)):
        r = fn(orgs, q)
        print(f"{q!r:20} {name} {len(r):2d}", [o.label for o in r[:4]])
